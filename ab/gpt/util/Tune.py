import json
import os
import shutil
from os import makedirs
from os.path import isfile

import ab.nn.api as lemur
import deepspeed
from ab.nn.util.Util import release_memory, create_file
from datasets import Dataset
from peft import (PeftModel)
from tqdm import tqdm
from transformers import AutoTokenizer

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


# Helper functions for loading JSONL and rendering with chat template
def load_jsonl(path):
    """Load a JSONL file and return list of parsed JSON objects."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def render_rows_with_template(rows, tokenizer, add_generation_prompt=False):
    """Render chat messages with the tokenizer's apply_chat_template."""
    out = []
    for r in rows:
        msgs = r["messages"]  # system/user/assistant
        # IMPORTANT: let HF build the right prompt for your model
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        out.append({"text": text})
    return out


def _load_chatjsonl_as_dataset(tokenizer, data_dir: str) -> Dataset:
    """Load train.jsonl/dev.jsonl/test.jsonl and render with the model's chat template."""
    def _load(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    
    train_p = os.path.join(data_dir, "train.jsonl")
    dev_p   = os.path.join(data_dir, "dev.jsonl")
    test_p  = os.path.join(data_dir, "test.jsonl")
    rows = []
    if os.path.exists(train_p): rows += _load(train_p)
    if os.path.exists(dev_p):   rows += _load(dev_p)
    if os.path.exists(test_p):  rows += _load(test_p)
    
    # Render with the model's chat template
    rendered = []
    for r in rows:
        msgs = r["messages"]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        rendered.append({"text": text})
    return Dataset.from_list(rendered)


def _load_chatjsonl_splits(tokenizer, data_dir: str):
    """
    Load chat_data → render with chat template → return splits.
    Always produces a text field using the model's chat template (HF recommends this for instruct models).
    No pre-tokenization or truncation - SFTTrainer will handle that internally.
    Returns (train_ds, dev_ds) as separate Dataset objects.
    """
    def _load_one(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
        return rows

    # Resolve absolute path if relative
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)
    
    # Standard filenames
    train_p = os.path.join(data_dir, "train.jsonl")
    dev_p = os.path.join(data_dir, "dev.jsonl")

    if not os.path.exists(train_p):
        # Build helpful error message
        error_msg = (
            f"[chat_data] Training file not found: train.jsonl\n"
            f"  Expected path: {train_p}\n"
            f"  Data directory: {data_dir}\n"
            f"  Current working directory: {os.getcwd()}\n"
            f"  \n"
            f"  Please ensure train.jsonl exists in the data directory.\n"
        )
        raise FileNotFoundError(error_msg)

    train_rows = _load_one(train_p)
    dev_rows   = _load_one(dev_p) if os.path.exists(dev_p) else []

    def _render(rows):
        """
        Render chat messages with chat template, keeping raw text only.
        SFTTrainer will handle tokenization and truncation internally.
        """
        rendered = []
        for r in rows:
            msgs = r["messages"]
            # Apply chat template but don't tokenize - let SFT handle that
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            rendered.append({"text": text})
        return Dataset.from_list(rendered) if rendered else Dataset.from_list([])

    train_ds = _render(train_rows)
    dev_ds   = _render(dev_rows)

    print(f"[chat_data] train={len(train_ds)} dev={len(dev_ds)} (data_dir={data_dir})")
    if len(train_ds) == 0:
        raise RuntimeError("[chat_data] Empty training dataset. Check paths and JSONL contents.")

    return train_ds, dev_ds


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


def tune(test_nn, nn_train_epochs, skip_epoch, llm_path, llm_tune_conf, nn_gen_conf, conf_keys, llm_conf, training_args, peft_config,
         max_prompts=None, save_llm_output=True, max_new_tokens=16 * 1024, nn_name_prefix=None, temperature=1.0, top_k=50, top_p=0.9, data_dir=None):
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

    # Load test prompts
    with open(conf_test_dir / nn_gen_conf) as prompt_file:
        prompt_dict = json.load(prompt_file)
    assert isinstance(prompt_dict, dict)

    # Load model and tokenizer
    # Pass training_args to LLM for ZeRO-3 detection
    model_loader = LLM(
        base_model_name,
        quantization_config_4bit,
        access_token=access_token,
        use_deepspeed=use_deepspeed,
        context_length=context_length,
        training_args=training_args
    )
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    # print(model)
    peft_loaded = False
    if llm_path:
        print(f'[INFO] Loading saved LoRA adapters from path: {llm_path}')
        print(f'[INFO] Continual learning: Will continue training existing adapters (not merging)')
        # Load PEFT checkpoint WITHOUT merging - keep adapters for continual learning
        model = PeftModel.from_pretrained(model, llm_path, is_trainable=True)
        peft_loaded = True
        # Get the existing PEFT config from loaded model
        if hasattr(model, 'peft_config'):
            existing_config = list(model.peft_config.values())[0] if model.peft_config else None
            if existing_config and peft_config is None:
                # Use the loaded config if no new config provided
                peft_config = existing_config
                print(f'[INFO] Using PEFT config from loaded checkpoint: r={existing_config.r}, target_modules={existing_config.target_modules}')

    # recommended when using gradient checkpointing
    try:
        if getattr(model.config, "use_cache", True):
            model.config.use_cache = False
    except Exception:
        pass

    # === If data_dir is provided, fine-tune directly on chat_data and SKIP NN generation ===
    if data_dir:
        print(f"[INFO] Using chat_data from: {data_dir}")
        # Load dataset with raw text only - no pre-tokenization
        train_ds, dev_ds = _load_chatjsonl_splits(tokenizer, data_dir)

        # Configure tokenizer for SFT: truncate from left (keep assistant response), pad on right
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = 4096  # DeepSeek-Coder-7B-Instruct-v1.5 has ~4K context
        
        # Suppress sequence length warnings - SFTTrainer will handle truncation correctly
        import warnings
        warnings.filterwarnings("ignore", message=".*sequence length.*longer than.*maximum.*")
        warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

        # Prepare model for QLoRA training and wrap with PEFT
        from peft import prepare_model_for_kbit_training, get_peft_model
        from ab.gpt.util.LoRA import find_all_linear_names, create_peft_config, print_trainable_parameters
        
        # Only prepare for kbit training if not already a PEFT model
        if not peft_loaded:
            model = prepare_model_for_kbit_training(model)
            model.gradient_checkpointing_enable()
        else:
            # Model already has PEFT adapters, just enable gradient checkpointing
            model.gradient_checkpointing_enable()
        
        # Only wrap with new PEFT adapters if we didn't load existing ones
        if not peft_loaded:
            if peft_config is None:
                modules = find_all_linear_names(model)
                peft_config = create_peft_config(modules)
            
            model = get_peft_model(model, peft_config)
        else:
            # Model already has adapters from previous cycle - use them for continual learning
            print(f'[INFO] Continuing training with existing LoRA adapters from checkpoint')
            if peft_config is None:
                # Extract config from loaded model
                peft_config = list(model.peft_config.values())[0] if model.peft_config else None
        
        # Log trainable parameters
        print(f"[LoRA] Adapters attached. Effective target_modules: {peft_config.target_modules}")
        print("[LoRA] Trainable parameter summary:")
        print_trainable_parameters(model)

        # Use TRL's SFTTrainer on the "text" field (official path)
        from trl import SFTTrainer, SFTConfig

        # When packing=False, SFTTrainer tokenizes text and removes it
        # We MUST allow column removal for proper tokenization
        training_args.remove_unused_columns = True  # Let SFTTrainer remove "text" after tokenization
        
        # Set packing, max_seq_length, and dataset_text_field in SFTConfig
        # Convert to SFTConfig if not already, or set attributes directly
        # CRITICAL: packing=False to prevent concatenating different code examples
        # This ensures each architecture remains distinct and prevents mixing
        if isinstance(training_args, SFTConfig):
            training_args.packing = False  # DISABLED: prevent mixing code examples
            training_args.max_seq_length = 4096  # DeepSeek-Coder-7B-Instruct-v1.5 has ~4K context (4k/4.1k)
            training_args.dataset_text_field = "text"  # tells SFT which column is the raw text
            training_args.remove_unused_columns = True  # Remove "text" after tokenization
        else:
            # If training_args is TrainingArguments, create SFTConfig with all attributes
            sft_config = SFTConfig(**training_args.to_dict())
            sft_config.remove_unused_columns = True  # Remove "text" after tokenization
            sft_config.packing = False  # DISABLED: prevent mixing code examples
            sft_config.max_seq_length = 4096  # DeepSeek-Coder-7B-Instruct-v1.5 has ~4K context (4k/4.1k)
            sft_config.dataset_text_field = "text"  # tells SFT which column is the raw text
            training_args = sft_config
        
        print(f"[TRAINING CONFIG] packing={training_args.packing} (False prevents mixing architectures)")
        print(f"[TRAINING CONFIG] max_seq_length={training_args.max_seq_length}")

        # When packing=False with raw text, SFTTrainer handles tokenization internally
        # We don't need (and shouldn't provide) a custom data collator
        # SFTTrainer will:
        #   1. Tokenize the "text" field automatically
        #   2. Apply truncation based on max_seq_length
        #   3. Create batches with proper padding
        # This trains on the full text (system + user + assistant)
        print("[TRAINING MODE] Full-text training (system + user + assistant)")
        print("[TRAINING MODE] SFTTrainer will handle tokenization internally")

        # SFTTrainer will handle tokenization and truncation internally
        # Since chat_template already added special tokens, tokenizer will use add_special_tokens=False internally
        # SFTTrainer will handle tokenization and truncation internally
        # Since chat_template already added special tokens, tokenizer will use add_special_tokens=False internally
        
        # PATCH: Handle trl version compatibility for SFTTrainer
        # Newer versions might use 'processing_class' instead of 'tokenizer'
        # or might not accept 'tokenizer' if it's inferred from model/args
        try:
            print("[INFO] Attempting SFTTrainer init with 'tokenizer' arg...")
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                eval_dataset=dev_ds if len(dev_ds) > 0 else None,
                args=training_args
            )
        except TypeError as e:
            if "unexpected keyword argument 'tokenizer'" in str(e):
                print("[WARN] SFTTrainer rejected 'tokenizer' arg. Trying with 'processing_class'...")
                try:
                    trainer = SFTTrainer(
                        model=model,
                        processing_class=tokenizer,
                        train_dataset=train_ds,
                        eval_dataset=dev_ds if len(dev_ds) > 0 else None,
                        args=training_args
                    )
                except TypeError as e2:
                     if "unexpected keyword argument 'processing_class'" in str(e2):
                        print("[WARN] SFTTrainer rejected 'processing_class' arg. Trying without tokenizer arg...")
                        trainer = SFTTrainer(
                            model=model,
                            train_dataset=train_ds,
                            eval_dataset=dev_ds if len(dev_ds) > 0 else None,
                            args=training_args
                        )
                     else:
                        raise e2
            else:
                raise e

        trainer.train()
        trainer.save_model("out/qlora-sft/final")
        print("[INFO] SFT on chat_data complete.")
        return

    # NOTE: DeepSpeed initialization is deferred until training starts
    # For inference (ChatBot), we use the model directly without DeepSpeed wrapper
    # DeepSpeed will be initialized by the Trainer during training
    # This avoids distributed initialization issues during inference

    lora_tuner = LoRA(
        model,
        tokenizer,
        training_args=training_args,
        access_token=access_token,
        peft_config=peft_config)

    print('Using Max Length:', model_loader.get_max_length())

    # loop train and eval cycles
    # For inference, use model directly (without DeepSpeed wrapper)
    chat_bot = ChatBot(model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)  # Only initialize ONCE

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
        
        # Original code path using NNGenPrompt (only executes when data_dir is not provided)
        # data_processor = NNGenPrompt(model_loader.get_max_length(), tokenizer, train_config_path)
        data_processor = NNGenPrompt(min(2048, context_length if context_length else model_loader.get_max_length()), tokenizer, train_config_path)
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
        
        # Use completion-only training when using pre-rendered JSONL data
        # This masks loss on system+user tokens, focusing training on assistant responses
        use_completion_only = data_dir is not None  # JSONL data is pre-rendered with chat template
        
        model = lora_tuner.train(
            dataset, 
            tokenizer, 
            out_path / base_model_name,
            train_on_completions_only=use_completion_only,
            response_template=None  # Auto-detect from sample
        )
        del dataset
        release_memory()


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
