import os
import json
import shutil
from os import makedirs
from os.path import isfile
from pathlib import Path

import ab.nn.api as lemur
import deepspeed
from ab.nn.util.Util import release_memory, create_file
from peft import (PeftModel)
from tqdm import tqdm

import ab.gpt.NNEval as NNEval
from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.Const import *
from ab.gpt.util.LLM import LLM
from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.LoRA import LoRA
from ab.gpt.util.Util import exists
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt

# from datasets import load_from_disk

ds_conf = conf_dir / 'DeepSpeed.json'


def apply_sliding_window(example, max_length, stride, tokenizer):
    input_ids = example['input_ids']
    attention_mask = example['attention_mask']

    chunks = []
    for i in range(0, len(input_ids), stride):
        end = i + max_length
        if end <= len(input_ids):
            chunk_input_ids = input_ids[i:end]
            chunk_attention_mask = attention_mask[i:end]

            pad_len = max_length - len(chunk_input_ids)
            if pad_len > 0:
                chunk_input_ids += [tokenizer.pad_token_id] * pad_len
                chunk_attention_mask += [0] * pad_len

            chunks.append({
                "input_ids": chunk_input_ids,
                "attention_mask": chunk_attention_mask
            })
    return {"chunks": chunks}


def flatten_chunks(data):
    all_chunks = sum(data["chunks"], [])  # flatten batched list
    return {
        "input_ids": [chunk["input_ids"] for chunk in all_chunks],
        "attention_mask": [chunk["attention_mask"] for chunk in all_chunks],
    }


def load_prompt_config(nn_gen_conf):
    """
    Load prompt configuration from JSON file.
    
    Args:
        nn_gen_conf: Name of the prompt config JSON file
        
    Returns:
        Dictionary containing prompt configuration
    """
    with open(conf_test_dir / nn_gen_conf) as prompt_file:
        prompt_dict = json.load(prompt_file)
    assert isinstance(prompt_dict, dict)
    return prompt_dict


def load_llm_and_chatbot(llm_conf, temperature=1.0, top_k=50, top_p=0.9, llm_path=None, use_deepspeed=None, context_length=None, access_token=None):
    """
    Load LLM model, tokenizer, and create ChatBot instance.
    
    Args:
        llm_conf: Name of LLM config JSON file
        temperature: Temperature for ChatBot generation (default: 1.0)
        top_k: Top-k for ChatBot generation (default: 50)
        top_p: Top-p for ChatBot generation (default: 0.9)
        llm_path: Optional path to saved LoRA layers to load
        use_deepspeed: Optional override for use_deepspeed (if None, reads from config)
        context_length: Optional override for context_length (if None, reads from config)
        access_token: Optional access token (if None and token_from_file=True, reads from file)
        
    Returns:
        Tuple of (model, tokenizer, chat_bot, model_loader)
    """
    with open(conf_llm_dir / llm_conf) as f:
        config = json.load(f)
    assert isinstance(config, dict)

    token_from_file = config['token_from_file']
    base_model_name = config['base_model_name']
    use_deepspeed = use_deepspeed if use_deepspeed is not None else config['use_deepspeed']
    config_context_length = config.get('context_length')
    context_length = context_length if context_length is not None else config_context_length

    if access_token is None and token_from_file:
        with open(ab_root_path / 'token') as f:
            access_token = f.readline()

    # Load model and tokenizer
    model_loader = LLM(
        base_model_name,
        quantization_config_4bit,
        access_token=access_token,
        use_deepspeed=use_deepspeed,
        context_length=context_length
    )
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    
    if llm_path:
        print(f'Load saved LoRA layer from path: {llm_path}')
        model = PeftModel.from_pretrained(model, llm_path, is_trainable=True)
        model = model.merge_and_unload()

    # initialize deepspeed before we do infer in ChatBot, since trainer is not initialized now.
    if use_deepspeed:
        deepspeed.initialize(model=model, config_params=ds_conf)

    # Create ChatBot
    chat_bot = ChatBot(model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)
    
    return model, tokenizer, chat_bot, model_loader


def tune(test_nn, nn_train_epochs, skip_epoch, llm_path, llm_tune_conf, nn_gen_conf, conf_keys, llm_conf, training_args, peft_config,
         max_prompts=None, save_llm_output=True, max_new_tokens=16 * 1024, nn_name_prefix=None, temperature=1.0, top_k=50, top_p=0.9, test_metric=None):
    if not isinstance(conf_keys, (list, tuple)):
        conf_keys = (conf_keys,)
    with open(conf_llm_dir / llm_conf) as f:
        config = json.load(f)
    assert isinstance(config, dict)

    token_from_file = config['token_from_file']
    base_model_name = config['base_model_name']
    llm_tune_epochs = int(config['num_epochs'])
    use_deepspeed = config['use_deepspeed']
    only_best_accuracy = config['only_best_accuracy']
    context_length = config.get('context_length')

    access_token = None
    if token_from_file:
        with open(ab_root_path / 'token') as f:
            access_token = f.readline()

    print(f'[DEBUG]Argument Information:\nSkip generation until Epoch: {skip_epoch}\nPath to saved LoRA Layers: {llm_path}')
    train_config_path = conf_train_dir / llm_tune_conf

    # Load test prompts using extracted function
    prompt_dict = load_prompt_config(nn_gen_conf)

    # Load model, tokenizer, and ChatBot using extracted function
    model, tokenizer, chat_bot, model_loader = load_llm_and_chatbot(
        llm_conf, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p,
        llm_path=llm_path,
        use_deepspeed=use_deepspeed,
        context_length=context_length,
        access_token=access_token
    )

    lora_tuner = LoRA(
        model,
        tokenizer,
        training_args=training_args,
        access_token=access_token,
        peft_config=peft_config,
        test_metric=test_metric)

    print('Using Max Length:', model_loader.get_max_length())

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(llm_tune_epochs):
        print(f'[INFO]Start Epoch {epoch}')
        out_path = epoch_dir(epoch)
        if epoch < skip_epoch:
            print(f'Skipped nn generation at epoch {epoch}')
        else:
            nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix)
        # fine tune model for 1 epoch / Using training_args and save copy
        print(f'[DEBUG]Perform finetune at epoch {epoch}.')
        # data_processor = NNGenPrompt(model_loader.get_max_length(), tokenizer, train_config_path)
        data_processor = NNGenPrompt(context_length if context_length else model_loader.get_max_length(), tokenizer, train_config_path)
        dataset = data_processor.get_dataset(only_best_accuracy, max_prompts=max_prompts)
        # dataset = load_from_disk(nngpt_dir / 'dataset')

        # if context_length:
        #     chunked_dataset = dataset.map(
        #         lambda x: apply_sliding_window(x, context_length, 1024, tokenizer),
        #         remove_columns=dataset.column_names,
        #         batch_size=16
        #     )
        #     dataset = chunked_dataset.map(flatten_chunks, batched=True, remove_columns=["chunks"])

        # print('Dataset length:', len(dataset))
        print('Dataset length:', len(dataset))
        model.train()
        model = lora_tuner.train(dataset, tokenizer, out_path / base_model_name)
        del dataset
        release_memory()


def tune_with_agents(
    test_nn, nn_train_epochs, skip_epoch, llm_path, llm_tune_conf, 
    nn_gen_conf, conf_keys, llm_conf, training_args, peft_config,
    max_prompts=None, save_llm_output=True, max_new_tokens=16 * 1024, 
    nn_name_prefix=None, temperature=1.0, top_k=50, top_p=0.9, 
    test_metric=None, use_predictor=False
):
    """
    LangGraph multi-agent workflow.
    Single source of orchestration logic - all workflow setup here.
    
    Args:
        use_predictor: If True, enable predictor agent
        ... (all other parameters same as tune())
    
    Returns:
        Final state dictionary from LangGraph workflow
    """
    try:
        from langgraph.graph import StateGraph, END
        from ab.gpt.agents.state import AgentState
        from ab.gpt.agents.manager import manager_node
        from ab.gpt.agents.generator import generator_node
        
        if use_predictor:
            from ab.gpt.agents.predictor import predictor_node
        
        print("=" * 60)
        print("🚀 Starting LangGraph Multi-Agent Workflow")
        print("=" * 60)
        print(f"  use_predictor: {use_predictor}")
        print("=" * 60)
        
        # ============================================================
        # CREATE STATEGRAPH WORKFLOW
        # ============================================================
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("manager", manager_node)
        workflow.add_node("generator", generator_node)
        if use_predictor:
            workflow.add_node("predictor", predictor_node)
        
        # Set entry point
        workflow.set_entry_point("manager")
        
        # ============================================================
        # CONDITIONAL ROUTING FUNCTION
        # ============================================================
        def should_continue(state: AgentState) -> str:
            """Route based on manager's next_action decision"""
            return state.get('next_action', 'end')
        
        # ============================================================
        # DEFINE CONDITIONAL EDGES
        # ============================================================
        edges = {
            "generate": "generator",
            "end": END,
        }
        if use_predictor:
            edges["predict"] = "predictor"
        
        workflow.add_conditional_edges(
            "manager",
            should_continue,
            edges
        )
        
        # ============================================================
        # AFTER AGENTS, RETURN TO MANAGER
        # ============================================================
        workflow.add_edge("generator", "manager")
        if use_predictor:
            workflow.add_edge("predictor", "manager")
        
        # ============================================================
        # COMPILE WORKFLOW
        # ============================================================
        app = workflow.compile()
        
        # ============================================================
        # INITIALIZE STATE FROM PARAMETERS
        # ============================================================
        # Get base_out_dir from nngpt_dir (same as tune() uses)
        from ab.gpt.util.Const import nngpt_dir
        
        # Handle conf_keys - convert to tuple if needed
        if not isinstance(conf_keys, (list, tuple)):
            conf_keys = (conf_keys,)
        
        initial_state: AgentState = {
            # Experiment metadata
            "experiment_id": nn_name_prefix or 'exp_default',
            "base_out_dir": str(nngpt_dir),
            
            # TuneNNGen parameters
            "num_train_epochs": training_args.num_train_epochs if hasattr(training_args, 'num_train_epochs') else 3,
            "nn_train_epochs": nn_train_epochs,
            "nn_gen_conf_id": conf_keys[0] if conf_keys else 'improve_classification_only',
            "temperature": temperature,
            "top_k": int(top_k) if isinstance(top_k, (int, str)) else top_k,
            "top_p": float(top_p) if isinstance(top_p, (int, str, float)) else top_p,
            "max_new_tokens": max_new_tokens,
            "save_llm_output": save_llm_output,
            
            # Additional parameters needed by generator
            "llm_conf": llm_conf,
            "nn_gen_conf": nn_gen_conf,
            
            # Workflow control
            "use_predictor": use_predictor,
            "gpu_available": True,
            "next_action": "generate",
            "status": "pending",
        }
        
        # ============================================================
        # RUN WORKFLOW
        # ============================================================
        print("\n🔄 Executing LangGraph workflow...\n")
        final_state = app.invoke(initial_state)
        
        # ============================================================
        # RETURN RESULTS
        # ============================================================
        print("\n" + "=" * 60)
        print("✅ LangGraph Workflow Completed!")
        print("=" * 60)
        print(f"  Status: {final_state.get('status', 'unknown')}")
        if final_state.get('model_code'):
            print(f"  Model generated: {final_state.get('nn_name', 'N/A')}")
            print(f"  Accuracy: {final_state.get('accuracy', 'N/A')}")
        if final_state.get('predicted_best_accuracy'):
            print(f"  Predicted accuracy: {final_state.get('predicted_best_accuracy', 'N/A')}")
            print(f"  Predicted epoch: {final_state.get('predicted_best_epoch', 'N/A')}")
        print("=" * 60)
        
        return final_state
        
    except ImportError as e:
        print(f"❌ Error: LangGraph not available. Install with: pip install langgraph")
        print(f"   Details: {e}")
        raise
    except Exception as e:
        print(f"❌ Error in LangGraph workflow: {e}")
        import traceback
        traceback.print_exc()
        raise


def read_eval_info(model_dir_path):
    """
    Read eval_info.json from model directory.
    
    Args:
        model_dir_path: Path to model directory containing eval_info.json
        
    Returns:
        Dictionary containing eval_info data, or empty dict if file doesn't exist
    """
    eval_file = Path(model_dir_path) / 'eval_info.json'
    if not eval_file.exists():
        return {}
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error reading eval_info.json: {e}")
        return {}


def nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix):
    # Move inside the loop to create new prompt with newly created models.
    print('Preparing prompts for generation, this might take a while...')
    prompts = []
    for key in conf_keys:
        prompt = ''
        prompt_dict = prompt_dict[key]
        for pr in prompt_dict['prompt']:
            prompt += pr + '\n'
        # Get nn-dataset codes
        data = lemur.data(only_best_accuracy=True, task=prompt_dict['task']).groupby(by='nn').sample(n=1)[:test_nn]
        # Get addon nn-dataset codes
        addon_data = lemur.data(only_best_accuracy=True, task=prompt_dict['addon_task'])
        for _, row in data.iterrows():
            para_dict = dict()
            for it in prompt_dict['input_list']:
                para_dict[it['para']] = row[it['value']]
            ## Avoid sampling the same nn_code
            addon_row = addon_data.loc[addon_data.nn != row['nn']].sample(n=1).iloc[0]
            if prompt_dict.get('addon_list'):
                for it in prompt_dict['addon_list']:
                    para_dict[it['para']] = addon_row[it['value']]
            prompts.append((prompt.format(**para_dict), row))
    # produce new CV models
    models_dir = synth_dir(out_path)
    # print(f"prompts: {prompts}")
    for idx, prompt in tqdm(enumerate(prompts)):
        model_dir = models_dir / f'B{idx}'
        prompt, origdf = prompt
        code, hp, tr, full_out = chat_bot.chat(prompt, engineer_prompt=False, max_new_tokens=max_new_tokens)
        if save_llm_output: create_file(model_dir, new_out_file, full_out)
        makedirs(model_dir, exist_ok=True)
        try:
            print(f'Generated params: {hp}')
            hp = json.loads(hp.replace("'", '"'))
            with open(model_dir / hp_file, 'w+') as f:
                json.dump(hp, f)
        except Exception as e:
            print(e)
            continue
        try:
            print(f'Generated transformer:\n\n{tr}\n----\n')
            create_file(model_dir, transformer_file, tr)
        except Exception as e:
            print(e)
            continue
        create_file(model_dir, new_nn_file, code)
        create_file(model_dir, new_out_file, full_out)
        df_file = model_dir / 'dataframe.df'
        if origdf is None:
            if isfile(df_file):  # Clean up dataframe.df, if no additional information generated this time.
                os.remove(df_file)
                print(f'[DEBUG]Removed unmatched file: {df_file}')
        else:
            create_file(model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
            # Store DataFrame information, mainly for passing parameters to evaluator.
            origdf.to_pickle(df_file)
    print('[DEBUG] Release memory.')
    release_memory()
    # evaluate produced CV models
    if exists(models_dir):
        NNEval.main(nn_name_prefix, nn_train_epochs, epoch)
        print('[DEBUG] Release_memory.')
        release_memory()
    print('Clear LEMUR query cache.')
    lemur.data.cache_clear()
    print('The cache has been cleared.')
