import os
import random
import shutil
import json
from os import makedirs
from os.path import isfile
import glob
from pathlib import Path
import numpy as np
import torch
import ab.nn.api as lemur
import deepspeed
from ab.nn.util.Util import release_memory, create_file
from peft import (PeftModel)
from tqdm import tqdm

import ab.gpt.NNEval as NNEval
from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.Const import *

from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.LoRA import LoRA
from ab.gpt.util.Util import exists, extract_delta, extract_code, extract_hyperparam, extract_transform
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from ab.gpt.util.DeltaUtil import apply_delta, validate_delta, repair_code
from ab.gpt.util.Const import nngpt_upload
from ab.gpt.brute.trans.TransformEval import run_eval
from ab.gpt.util.prompt.TransformGenPrompt import TransformGenPrompt, load_data_from_folders

# from datasets import load_from_disk


ds_conf = conf_dir / 'DeepSpeed.json'

# Transform dir paths
TRANSFORM_OUT_DIR = trans_dir / 'dataset_epoch1'
TRANSFORM_RES_DIR = trans_dir / 'result_epoch1'

# Delta mode constants
_MAX_DELTA_RETRIES = 2



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
         max_prompts=None, save_llm_output=True, max_new_tokens=16 * 1024, nn_name_prefix=None, temperature=1.0, top_k=50, top_p=0.9, test_metric=None,
         onnx_run=False, trans_mode=False, prompt_batch=1, use_backbone=False):
    if not isinstance(conf_keys, (list, tuple)):
        conf_keys = (conf_keys,)
    with open(conf_llm_dir / llm_conf) as f:
        config = json.load(f)
    assert isinstance(config, dict)

    token_from_file = config['token_from_file']
    base_model_name = config['base_model_name']
    merged_candidate = nngpt_upload / Path(base_model_name).name

    if merged_candidate.exists():
        print(f"[EVOLUTION] Using merged model: {merged_candidate}")
        base_model_name = str(merged_candidate)
    else:
        print(f"[EVOLUTION] Using base model from config: {base_model_name}")


    llm_tune_epochs = int(config['num_epochs'])
    use_deepspeed = config['use_deepspeed']
    only_best_accuracy = config['only_best_accuracy']
    context_length = config.get('context_length')
    unsloth_max_input_length = config.get('max_input_length', None)
    use_unsloth = config.get('use_unsloth', False)
    unsloth_load_in_4bit = config.get('load_in_4bit', True)
    max_new_tokens = config.get('max_new_tokens', max_new_tokens)
    use_backbone = config.get('backbone', use_backbone)

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
   
    from ab.gpt.util.LLM import LLM

    # Load model and tokenizer
    model_loader = LLM(
        base_model_name,
        quantization_config_4bit,
        access_token=access_token,
        use_deepspeed=use_deepspeed,
        context_length=context_length,
        training_args=training_args,
        use_unsloth=use_unsloth,
        load_in_4bit=unsloth_load_in_4bit
    )
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    # print(model)
    if llm_path:
        print(f'Load saved LoRA layer from path: {llm_path}')
        model = PeftModel.from_pretrained(model, llm_path, is_trainable=True)
        model = model.merge_and_unload()

    # initialize deepspeed before we do infer in ChatBot
    if use_deepspeed:
        deepspeed.initialize(model=model, config_params=ds_conf)

    lora_tuner = LoRA(
        model,
        tokenizer,
        training_args=training_args,
        access_token=access_token,
        peft_config=peft_config,
        use_unsloth=use_unsloth)

    print('Using Max Length:', model_loader.get_max_length())

    # loop train and eval cycles
    chat_bot = ChatBot(model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p) # Only initialize ONCE

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(llm_tune_epochs):
        print(f'[INFO]Start Epoch {epoch}')
        out_path = epoch_dir(epoch)
        if epoch < skip_epoch:
            print(f'Skipped generation at epoch {epoch}')
        else:
            if trans_mode:
                trans_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix)
            else:
                nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix, unsloth_max_input_length, prompt_batch, use_backbone=use_backbone)
                

        # fine tune model for 1 epoch / Using training_args and save copy
        print(f'[DEBUG]Perform finetune at epoch {epoch}.')

        # Select data processor based on mode
        if trans_mode:
            data_processor = TransformGenPrompt(
                context_length if context_length else model_loader.get_max_length(), 
                tokenizer, 
                train_config_path,
                TRANSFORM_OUT_DIR,
                TRANSFORM_RES_DIR
            )
        elif use_backbone:
             from ab.gpt.util.prompt.SFTGenPrompt import SFTGenPrompt
             data_processor = SFTGenPrompt(
                context_length if context_length else model_loader.get_max_length(),
                tokenizer
             )
        else:
            if not use_unsloth:
                data_processor = NNGenPrompt(context_length if context_length else model_loader.get_max_length(), tokenizer, train_config_path)
            else:
                data_processor = NNGenPrompt(unsloth_max_input_length if unsloth_max_input_length else model_loader.get_max_length(), tokenizer, train_config_path)
        dataset = data_processor.get_dataset(only_best_accuracy, max_prompts=max_prompts, max_new_tokens=max_new_tokens)

        print('Dataset length:', len(dataset))
        model.train()
        model = lora_tuner.train(dataset, tokenizer, out_path / base_model_name)
        del dataset
        release_memory()


def nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix, unsloth_max_input_length, prompt_batch, use_backbone=False):
    print('Preparing prompts for generation, this might take a while...')

    # Detect delta mode from nn_name_prefix or config key
    use_delta = nn_name_prefix == 'delta'
    if not use_delta and isinstance(prompt_dict, dict) and conf_keys:
        first_key = conf_keys[0] if isinstance(conf_keys, (list, tuple)) else conf_keys
        key_config = prompt_dict.get(first_key, {})
        if isinstance(key_config, dict):
            use_delta = key_config.get('use_delta', False) or 'delta' in str(first_key).lower()

    prompts = []
    for key in conf_keys:
        prompt = ''
        key_config = prompt_dict[key]
        prompt_dict_key = key_config
        for pr in prompt_dict_key['prompt']:
            prompt += pr + '\n'
        data = lemur.data(only_best_accuracy=True, task=prompt_dict_key['task']).groupby(by='nn').sample(n=1)[:test_nn]
        addon_task = prompt_dict_key.get('addon_task')
        addon_data = lemur.data(only_best_accuracy=True, task=addon_task) if addon_task else None
        from ab.gpt.util.Util import extract_str
        for _, row in data.iterrows():
            para_dict = dict()
            for it in prompt_dict_key['input_list']:
                para_dict[it['para']] = row[it['value']]
            if addon_data is not None and not addon_data.empty:
                available_addon = addon_data.loc[addon_data.nn != row['nn']]
                if not available_addon.empty:
                    addon_row = available_addon.sample(n=1).iloc[0]
                    if prompt_dict_key.get('addon_list'):
                        for it in prompt_dict_key['addon_list']:
                            para_dict[it['para']] = addon_row[it['value']]
            prompts.append((prompt.format(**para_dict), row))

    models_dir = synth_dir(out_path)

    # Delta mode: per-sample processing with retry-and-feedback
    if use_delta:
        for idx, prompt_data in tqdm(enumerate(prompts)):
            model_dir = models_dir / f'B{idx}'
            prompt_text, origdf = prompt_data

            # Per-sample seed for reproducibility and diversity across epochs
            seed = epoch * 10000 + idx
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            if unsloth_max_input_length:
                in_text = [{"role": "user", "content": prompt_text}]
                token_len = len(chat_bot.tokenizer.apply_chat_template(in_text, add_generation_prompt=True))
                print(f'Sample prompt length: {token_len}, max_input_length: {unsloth_max_input_length}')
                if token_len > unsloth_max_input_length:
                    print(f'Prompt is too long, skipping...')
                    continue

            baseline_code = origdf.get('nn_code', '') if origdf is not None else ''

            # Initial generation
            _, hp, tr, full_out = chat_bot.chat(prompt_text, engineer_prompt=False, max_new_tokens=max_new_tokens)

            if use_backbone:
                from ab.gpt.util.SFTUtil import skeleton_code
                import textwrap

                # Extract full blocks (including signatures)
                block_code = extract_str(full_out, '<block>', '</block>')
                init_code = extract_str(full_out, '<init>', '</init>')
                forward_code = extract_str(full_out, '<forward>', '</forward>')

                if block_code and init_code and forward_code:
                    code = skeleton_code

                    # Replace skeleton signatures with LLM-provided blocks (including signatures)
                    # Ensure correct indentation for internal methods
                    sig_block = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
                    code = code.replace(sig_block, textwrap.dedent(block_code))

                    sig_init = "    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
                    code = code.replace(sig_init, textwrap.indent(textwrap.dedent(init_code), "    "))

                    sig_forward = "    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"
                    code = code.replace(sig_forward, textwrap.indent(textwrap.dedent(forward_code), "    "))
                else:
                    from ab.gpt.util.Util import extract_code
                    code = extract_code(full_out)

            if code is None:
                print(f'[ERROR] No code generated for model B{idx}')
                continue # Skip if no code is generated at all

            makedirs(model_dir, exist_ok=True)
            if save_llm_output:
                create_file(model_dir, new_out_file, full_out)

            # Delta extraction + application with retry-and-feedback
            code = None
            current_out = full_out
            current_prompt = prompt_text
            for attempt in range(_MAX_DELTA_RETRIES + 1):
                if attempt > 0:
                    _, _, _, current_out = chat_bot.chat(current_prompt, engineer_prompt=False, max_new_tokens=max_new_tokens)

                delta = extract_delta(current_out)
                if not delta:
                    error_msg = 'No <delta>...</delta> block found in output.'
                elif not validate_delta(delta):
                    error_msg = 'Delta format is invalid (must be unified diff with --- / +++ headers and @@ hunks).'
                else:
                    applied = apply_delta(baseline_code, delta) if baseline_code else None
                    if applied:
                        code = applied
                        print(f'[INFO] Applied delta for B{idx} (attempt {attempt + 1})')
                        break
                    else:
                        error_msg = 'Delta patch failed to apply to the baseline code.'

                if attempt < _MAX_DELTA_RETRIES:
                    print(f'[WARNING] Delta attempt {attempt + 1} failed for B{idx}: {error_msg} Retrying with feedback...')
                    current_prompt = (
                        prompt_text
                        + f'\n\n[SYSTEM FEEDBACK - Attempt {attempt + 1} failed]: {error_msg}'
                        + '\nPlease correct the delta and output it again.'
                    )

            # Syntax-repair fallback when all delta attempts fail
            if code is None:
                print(f'[WARNING] All delta attempts failed for B{idx}. Trying syntax repair on extracted code.')
                raw_code = extract_code(full_out)
                if raw_code:
                    repaired = repair_code(raw_code)
                    if repaired:
                        code = repaired
                        print(f'[INFO] Used syntax-repaired code fallback for B{idx}')

            # Re-parse hp/tr from saved output
            hp_str = extract_hyperparam(full_out)
            tr_str = extract_transform(full_out)

            try:
                print(f'Generated params: {hp_str}')
                if hp_str is not None and hp_str.strip():
                    hp_obj = json.loads(hp_str.replace("'", '"'))
                    with open(model_dir / hp_file, 'w+') as f:
                        json.dump(hp_obj, f)
                else:
                    print('[WARNING] No hyperparameters generated, skipping hp file')
            except Exception as e:
                print(f'[WARNING] Error processing hyperparameters: {e}')

            try:
                print(f'Generated transformer:\n\n{tr_str}\n----\n')
                if tr_str is not None and tr_str.strip():
                    create_file(model_dir, transformer_file, tr_str)
                else:
                    print('[WARNING] No transformer code generated')
            except Exception as e:
                print(f'[WARNING] Error saving transformer: {e}')

            if code is not None and code.strip():
                create_file(model_dir, new_nn_file, code)
                print(f'[INFO] Saved code to {model_dir / new_nn_file}')
            else:
                print(f'[ERROR] No code generated for model B{idx}')
                continue

            create_file(model_dir, new_out_file, full_out)
            df_file = model_dir / 'dataframe.df'
            if origdf is None:
                if isfile(df_file):
                    os.remove(df_file)
                    print(f'[DEBUG]Removed unmatched file: {df_file}')
            else:
                create_file(model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
                origdf.to_pickle(df_file)

    # Standard mode: batch processing
    else:
        pending = []
        for idx, prompt_data in tqdm(enumerate(prompts)):
            prompt, origdf = prompt_data

            if unsloth_max_input_length:
                in_text = [{"role": "user", "content": prompt}]
                output = chat_bot.tokenizer.apply_chat_template(in_text, add_generation_prompt=True)
                print(f'Sample prompt length: {len(output)}, max_input_length: {unsloth_max_input_length}')
                if len(output) > unsloth_max_input_length:
                    print(f'Prompt is too long, skipping...')
                    continue

            pending.append((idx, prompt, origdf))

        if prompt_batch < 1:
            prompt_batch = 1
        if prompt_batch > 1:
            print(f'[INFO] Batch generation enabled: prompt_batch={prompt_batch}')

        for start in range(0, len(pending), prompt_batch):
            batch = pending[start:start + prompt_batch]
            batch_prompts = [item[1] for item in batch]

            if prompt_batch > 1 and hasattr(chat_bot, 'chat_batch'):
                batch_outputs = chat_bot.chat_batch(batch_prompts, engineer_prompt=False, max_new_tokens=max_new_tokens)
            else:
                batch_outputs = [chat_bot.chat(p, engineer_prompt=False, max_new_tokens=max_new_tokens) for p in batch_prompts]

            for (idx, prompt, origdf), output in zip(batch, batch_outputs):
                model_dir = models_dir / f'B{idx}'
                code, hp, tr, full_out = output
                if save_llm_output:
                    create_file(model_dir, new_out_file, full_out)
                makedirs(model_dir, exist_ok=True)

                try:
                    print(f'Generated params: {hp}')
                    if hp is not None and hp.strip():
                        hp = json.loads(hp.replace("'", '"'))
                        with open(model_dir / hp_file, 'w+') as f:
                            json.dump(hp, f)
                    else:
                        print('[WARNING] No hyperparameters generated, skipping hp file')
                except Exception as e:
                    print(f'[WARNING] Error processing hyperparameters: {e}')

                try:
                    print(f'Generated transformer:\n\n{tr}\n----\n')
                    if tr is not None and tr.strip():
                        create_file(model_dir, transformer_file, tr)
                    else:
                        print('[WARNING] No transformer code generated')
                except Exception as e:
                    print(f'[WARNING] Error saving transformer: {e}')

                if code is not None and code.strip():
                    create_file(model_dir, new_nn_file, code)
                    print(f'[INFO] Saved code to {model_dir / new_nn_file}')
                else:
                    print(f'[ERROR] No code generated for model B{idx}')
                    continue
                create_file(model_dir, new_out_file, full_out)
                df_file = model_dir / 'dataframe.df'
                if origdf is None:
                    if isfile(df_file):
                        os.remove(df_file)
                        print(f'[DEBUG]Removed unmatched file: {df_file}')
                else:
                    create_file(model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
                    origdf.to_pickle(df_file)

    print('[DEBUG] Release memory.')
    release_memory()
    if exists(models_dir):
        NNEval.main(nn_name_prefix, nn_train_epochs, epoch)
        print('[DEBUG] Release_memory.')
        release_memory()
    print('Clear LEMUR query cache.')
    lemur.data.cache_clear()
    print('The cache has been cleared.')


def trans_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict_global, test_nn, max_new_tokens, save_llm_output, nn_name_prefix):
    """
    Transform Script Generation
    """
    print('Running Transform Generation...')
    
    out_gen_dir = str(TRANSFORM_OUT_DIR)
    result_gen_dir = str(TRANSFORM_RES_DIR)
  
    prompts = []

    # Load all data from folders to be used for seed prompts
    all_data = load_data_from_folders(out_gen_dir, result_gen_dir, only_best_accuracy=True)
    if len(all_data) == 0:
        print("Warning: No data loaded from folders for generation. Skipping.", flush=True)
        return
        
    for key in conf_keys:
        prompt_config = prompt_dict_global[key]
        prompt = ''
        for pr in prompt_config['prompt']:
            prompt += pr + '\n'

        # Get seed data    
        if len(all_data) < test_nn:
            print(f"Warning: Requested {test_nn} samples, but only {len(all_data)} available. Using all.", flush=True)
            data_sample = all_data.sample(n=len(all_data))
        else:
            data_sample = all_data.sample(n=test_nn)

        addon_data = all_data
        
        for _, row in data_sample.iterrows():
            para_dict = dict()
            row_dict = row.to_dict()
            for it in prompt_config['input_list']:
                para_dict[it['para']] = row_dict.get(it['value'])
            
            # Avoid sampling the same transform
            filtered_addon_data = addon_data.loc[addon_data.id_name != row['id_name']]
            if len(filtered_addon_data) > 0:
                addon_row = filtered_addon_data.sample(n=1).iloc[0].to_dict()
                if prompt_config.get('addon_list'):
                    for it in prompt_config['addon_list']:
                        para_dict[it['para']] = addon_row.get(it['value'])
                prompts.append((prompt.format(**para_dict), row))
            else:
                print(f"Warning: Could not find addon data for {row['id_name']}. Skipping prompt.", flush=True)
                
    models_dir = synth_dir(out_path)
    
    for idx, prompt_data in tqdm(enumerate(prompts)):
        model_dir = models_dir / f'B{idx}'
        prompt, origdf = prompt_data
        
        code, hp, tr, full_out = chat_bot.chat(prompt, engineer_prompt=False, max_new_tokens=max_new_tokens)
        
        if save_llm_output: create_file(model_dir, new_out_file, full_out)
        makedirs(model_dir, exist_ok=True)

        if tr is not None and tr.strip():
            print(f'Generated transformer:\n\n{tr}\n----\n')
            create_file(model_dir, transformer_file, tr)
        else:
            print(f'[ERROR] No code generated for model B{idx}')
            continue  

        df_file = model_dir / 'dataframe.df'
        if origdf is None:
            if isfile(df_file):
                os.remove(df_file)
        else:
            create_file(model_dir, f"original_{origdf['id_name']}.py", origdf['transform_code'])
            origdf.to_pickle(df_file)
            
    print('[DEBUG] Release memory.')
    release_memory()

    # Evaluate produced CV models
    if exists(models_dir):
        try:
            run_eval(epoch_num=epoch, FT_MODE=True)
        except Exception as e:
            print(f"Error running evaluation main(): {e}", flush=True)
            
        print('[DEBUG] Release_memory.')
        release_memory()
        
    print('Folder data reload will occur next epoch.')
