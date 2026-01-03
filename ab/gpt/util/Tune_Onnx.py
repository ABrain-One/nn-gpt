# ab/gpt/util/Tune.py

import os
import shutil
import json
from os import makedirs
from os.path import isfile
from pathlib import Path
import torch
import gc
import ab.nn.api as lemur
import deepspeed
from ab.nn.util.Util import release_memory, create_file
from peft import PeftModel
from tqdm import tqdm
import ab.gpt.NNEval as NNEval
from ab.gpt.util.Chatbot_Onnx import ChatBot
from ab.gpt.util.Const_Onnx import (
    ab_root_path,
    conf_dir,
    conf_llm_dir,
    conf_train_dir,
    conf_test_dir,
    epoch_dir,
    synth_dir,
    new_out_file,
    new_nn_file,
    hp_file,
    transformer_file,
    huggingface_cache,
    huggingface_tokenizer_cache
)

from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.LoRA import LoRA
from ab.gpt.util.Util_Onnx import exists
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt

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
    all_chunks = sum(data["chunks"], [])
    return {
        "input_ids": [chunk["input_ids"] for chunk in all_chunks],
        "attention_mask": [chunk["attention_mask"] for chunk in all_chunks],
    }

def tune(test_nn, nn_train_epochs, skip_epoch, llm_path, llm_tune_conf, nn_gen_conf, conf_keys, llm_conf,
         training_args, peft_config, max_prompts=None, save_llm_output=True, max_new_tokens=16 * 1024,
         nn_name_prefix=None, temperature=1.0, top_k=50, top_p=0.9, test_metric=None, onnx_run=False):

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

    # =========================================================================
    # --- DUAL MODEL SETUP: PyTorch for training, ONNX for generation ---
    

    # 1) Always load PyTorch model for training
    print('Use Pytorch LLM for training...')
    from ab.gpt.util.LLM import LLM as PyTorchLLM

    # Disable quantization when using ONNX workflow (for clean export)
    if onnx_run:
        print('[INFO] ONNX workflow: loading PyTorch model in FP16 (no quantization)')
        # important: Delete cached model with quantization config first
        local_cache = Path(huggingface_cache) / base_model_name
        if local_cache.exists() and (local_cache / 'config.json').exists():
            with open(local_cache / 'config.json', 'r') as f:
                cached_config = json.load(f)
            if 'quantization_config' in cached_config:
                print(f'[INFO] Removing cached quantized model: {local_cache}')
                shutil.rmtree(local_cache, ignore_errors=True)

        pytorch_model_loader = PyTorchLLM(
            base_model_name,
            bnb_config=None,  # No quantization for ONNX export compatibility
            access_token=access_token,
            use_deepspeed=use_deepspeed,
            context_length=context_length,
            training_args=training_args
        )
    else:
        pytorch_model_loader = PyTorchLLM(
            base_model_name,
            quantization_config_4bit,  # Use quantization for pure PyTorch workflow
            access_token=access_token,
            use_deepspeed=use_deepspeed,
            context_length=context_length,
            training_args=training_args
        )

    pytorch_model = pytorch_model_loader.get_model()
    tokenizer = pytorch_model_loader.get_tokenizer()

    # Load LoRA checkpoint if resuming
    if llm_path:
        print(f'Load saved LoRA layer from path: {llm_path}')
        pytorch_model = PeftModel.from_pretrained(pytorch_model, llm_path, is_trainable=True)
        pytorch_model = pytorch_model.merge_and_unload()

    # Initialize DeepSpeed if needed
    if use_deepspeed:
        deepspeed.initialize(model=pytorch_model, config_params=ds_conf)

    # Set up LoRA for training
    lora_tuner = LoRA(
        pytorch_model,
        tokenizer,
        training_args=training_args,
        access_token=access_token,
        peft_config=peft_config
    )

    print('Using Max Length:', pytorch_model_loader.get_max_length())

    # 2) Set up generation model (ONNX or PyTorch)
    if onnx_run:
        print('Use ONNX exported LLM for generation')
        from optimum.onnxruntime import ORTModelForCausalLM
        from ab.gpt.util.OnnxExport import export_llm_to_onnx
        from ab.gpt.util.OnnxWrapper import OnnxCausalLMWrapper

        export_onnx_path = Path(ab_root_path) / 'onnx_output_folder'
        onnx_model_file = export_onnx_path / 'model.onnx'

    
        if not onnx_model_file.exists():
            print('[INFO] <========================================>')
            print('[INFO] No existing ONNX model found.')
            print('[INFO] Exporting base PyTorch model to ONNX...')
            print('[INFO] This usually takes 1-2 minutes.')
            print('[INFO] <========================================>')
            export_onnx_path.mkdir(parents=True, exist_ok=True)

            # Imprtant! Get the BASE model WITHOUT LoRA adapters, this fix is applied because the ONNX was generating gebbiresh
            print('[INFO] Loading clean base model for ONNX export...')
            from ab.gpt.util.LLM import LLM as PyTorchLLM
            
            clean_model_loader = PyTorchLLM(
                base_model_name,
                bnb_config=None,  # No quantization
                access_token=access_token,
                use_deepspeed=False,  # No DeepSpeed for export
                context_length=context_length,
                training_args=training_args
            )
            clean_base_model = clean_model_loader.get_model()
            
            # Export the CLEAN base model
            export_llm_to_onnx(clean_base_model, export_onnx_path, tokenizer)

            # Free memory
            del clean_base_model, clean_model_loader
            torch.cuda.empty_cache()

            print('[INFO] ✓ Base model ONNX export completed ✓ ')
            print(f'[INFO] Location: {onnx_model_file}')
            print('[INFO] ========================================')


        # Load ONNX model for generation
        onnx_model = ORTModelForCausalLM.from_pretrained(
            export_onnx_path,
            provider="CUDAExecutionProvider",
        )
        generation_model = OnnxCausalLMWrapper(onnx_model)
        print(f'[INFO] ONNX model loaded for generation')
        print('[INFO] Moving PyTorch model to CPU to free GPU memory...')
        pytorch_model = pytorch_model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        print('[INFO] ✓ GPU memory freed for ONNX generation ✓')
    else:
        # Use PyTorch for generation
        print('Use PyTorch LLM for generation...')
        generation_model = pytorch_model

    # 3) Create ChatBot with the generation model
    chat_bot = ChatBot(generation_model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)

    # FIX: Check if epoch_dir is a function or Path, to handle accordingly
    epoch_base_dir = epoch_dir() if callable(epoch_dir) else epoch_dir
    shutil.rmtree(epoch_base_dir, ignore_errors=True)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    for epoch in range(llm_tune_epochs):
        print(f'[INFO]Start Epoch {epoch}')
        
        # FIX: Handle both function and Path for epoch_dir
        if callable(epoch_dir):
            out_path = epoch_dir(epoch)
        else:
            out_path = Path(epoch_base_dir) / str(epoch)

        if epoch < skip_epoch:
            print(f'Skipped nn generation at epoch {epoch}')
        else:
            nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix)
        # ============================================================
        # FREE ONNX MODEL FROM GPU BEFORE FINE-TUNING
        # ============================================================
        if onnx_run:
            print('[INFO] ========================================')
            print('[INFO] Freeing ONNX model from GPU before fine-tuning...')
            print('[INFO] ========================================')
            
            # Delete ONNX model and wrapper
            if 'generation_model' in locals():
                del generation_model
            if 'onnx_model' in locals():
                del onnx_model
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Show memory freed
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f'[INFO] GPU 0 after cleanup - Allocated: {gpu_mem_allocated:.2f} GB')
            print(f'[INFO] GPU 0 after cleanup - Reserved: {gpu_mem_reserved:.2f} GB')
            print('[INFO] ✓ ONNX memory freed')
            print('[INFO] ========================================')
        # ============================================================

        # Fine tune PyTorch model for 1 epoch
        print(f'[DEBUG]Perform finetune at epoch {epoch}.')
        data_processor = NNGenPrompt(context_length if context_length else pytorch_model_loader.get_max_length(), tokenizer, train_config_path)
        dataset = data_processor.get_dataset(only_best_accuracy, max_prompts=max_prompts)

        print('Dataset length:', len(dataset))
        print('[INFO] Moving PyTorch model back to GPU for training...')
        pytorch_model = pytorch_model.to('cuda')
        pytorch_model.train()
        pytorch_model = lora_tuner.train(dataset, tokenizer, out_path / base_model_name)
        del dataset
        release_memory()

        # ============================================================
        # RELOAD ONNX MODEL AFTER FINE-TUNING (if not last epoch)
        # ============================================================
        if onnx_run and epoch < llm_tune_epochs - 1:
            print('[INFO] ========================================')
            print('[INFO] Reloading ONNX model for next generation cycle...')
            print('[INFO] ========================================')
            
            # Reload ONNX model for next generation
            onnx_model = ORTModelForCausalLM.from_pretrained(
                export_onnx_path,
                provider="CUDAExecutionProvider",
            )
            generation_model = OnnxCausalLMWrapper(onnx_model)
            
            # Update ChatBot
            chat_bot.model = generation_model
            
            print('[INFO] ✓ ONNX model successfully reloaded')
            print('[INFO] ========================================')
      
       
        # Update ONNX model after each epoch if using ONNX
        if onnx_run and epoch < llm_tune_epochs - 1:
            print('[INFO] ========================================')
            print(f'[INFO] Updating ONNX model after epoch {epoch}...')
            print('[INFO] Using CPU export to avoid GPU memory issues')
            print('[INFO] ========================================')
            
            # Save current LoRA checkpoint
            print('[INFO] Saving LoRA checkpoint...')
            lora_checkpoint_path = out_path / base_model_name / "lora_checkpoint"
            pytorch_model.save_pretrained(lora_checkpoint_path)
            print(f'[INFO] ✓ LoRA checkpoint saved to {lora_checkpoint_path}')
            
            # Move training model to CPU to free GPU
            print('[INFO] Moving training model to CPU to free GPU memory...')
            pytorch_model = pytorch_model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            
            gpu_mem = torch.cuda.memory_allocated(0) / 1e9
            print(f'[INFO] GPU memory after moving to CPU: {gpu_mem:.2f} GB')
            
            # Load clean base model ON CPU (no device_map!)
            print('[INFO] Loading clean base model on CPU for ONNX export...')
            from ab.gpt.util.LLM import LLM as PyTorchLLM
            
            # Don't pass training_args to avoid device_map='auto'
            clean_model_loader = PyTorchLLM(
                base_model_name,
                bnb_config=None,
                access_token=access_token,
                use_deepspeed=False,
                context_length=context_length,
                training_args=None  # ← Don't pass training_args!
            )
            clean_base_model = clean_model_loader.get_model()
            clean_base_model = clean_base_model.cpu()  # Explicitly to CPU
            print('[INFO] ✓ Clean base model loaded on CPU')
            
            # Load LoRA weights onto clean base
            print('[INFO] Loading LoRA weights onto clean base model...')
            from peft import PeftModel
            model_with_lora = PeftModel.from_pretrained(
                clean_base_model,
                lora_checkpoint_path,
                is_trainable=False
            )
            print('[INFO] ✓ LoRA weights loaded')
            
            # Merge LoRA weights
            print('[INFO] Merging LoRA weights into base model...')
            merged_model = model_with_lora.merge_and_unload()
            print('[INFO] ✓ LoRA weights merged')
            
            # Export to ONNX on CPU
            print('[INFO] Exporting merged model to ONNX (on CPU, may take 2-3 minutes)...')
            export_llm_to_onnx(merged_model, export_onnx_path, tokenizer)
            print('[INFO] ✓ ONNX export complete')
            
            # Cleanup CPU models
            print('[INFO] Cleaning up temporary models...')
            del clean_base_model, model_with_lora, merged_model, clean_model_loader
            gc.collect()
            
            # Reload ONNX model on GPU for next generation cycle
            print('[INFO] Loading updated ONNX model to GPU for generation...')
            if 'generation_model' in locals():
                del generation_model
            if 'onnx_model' in locals():
                del onnx_model
            
            torch.cuda.empty_cache()
            
            onnx_model = ORTModelForCausalLM.from_pretrained(
                export_onnx_path,
                provider="CUDAExecutionProvider",
            )
            generation_model = OnnxCausalLMWrapper(onnx_model)
            chat_bot.model = generation_model
            print('[INFO] ✓ ONNX model loaded to GPU')
            
            # Move training model back to GPU for next epoch
            print('[INFO] Moving training model back to GPU for next epoch...')
            pytorch_model = pytorch_model.to('cuda')
            torch.cuda.empty_cache()
            
            gpu_mem = torch.cuda.memory_allocated(0) / 1e9
            print(f'[INFO] GPU memory after reload: {gpu_mem:.2f} GB')
            
            print('[INFO] ✓ ONNX model updated successfully')
            print('[INFO] ========================================')

          
def nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix):
  
    print('Preparing prompts for generation, this might take a while...')
    
    use_delta = False
    if isinstance(prompt_dict, dict) and conf_keys:
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

    for idx, prompt in tqdm(enumerate(prompts)):
        model_dir = models_dir / f'B{idx}'
        prompt, origdf = prompt

        # print(f"\n[DEBUG B{idx}] Prompt length: {len(prompt)} chars")
        # print(f"[DEBUG B{idx}] First 200 chars of prompt:\n{prompt[:200]}")

        code, hp, tr, full_out = chat_bot.chat(prompt, engineer_prompt=False, max_new_tokens=max_new_tokens)

        # DEBUG BLOCK - to check the output of the llm
        print(f"[DEBUG B{idx}] Generated output length: {len(full_out) if full_out else 0} chars")
        print(f"[DEBUG B{idx}] First 500 chars of output:\n{full_out[:500] if full_out else '(EMPTY)'}")
        print(f"[DEBUG B{idx}] Extracted code: {'Found' if code else 'None'}")

        if save_llm_output:
            create_file(model_dir, new_out_file, full_out)

        makedirs(model_dir, exist_ok=True)

        if use_delta and origdf is not None:
            try:
                from ab.gpt.util.DeltaUtil import apply_delta, validate_delta
                from ab.gpt.util.Util import extract_delta

                delta = extract_delta(full_out)
                if delta:
                    if not validate_delta(delta):
                        print(f'[WARNING] Invalid delta format for model B{idx}, using extracted code as fallback')
                    else:
                        baseline_code = origdf.get('nn_code', '')
                        if baseline_code:
                            applied_code = apply_delta(baseline_code, delta)
                            if applied_code:
                                code = applied_code
                                print(f'[INFO] Successfully applied delta to baseline code for model B{idx}')
                            else:
                                print(f'[WARNING] Failed to apply delta for model B{idx} (delta application returned None), using extracted code as fallback')
                        else:
                            print(f'[WARNING] No baseline code found in origdf for model B{idx}, using extracted code')
                else:
                    print(f'[WARNING] No delta found in LLM output for model B{idx}, using extracted code as fallback')
            except ImportError as e:
                print(f'[ERROR] Failed to import delta utilities for model B{idx}: {e}. Using extracted code as fallback.')
            except Exception as e:
                print(f'[WARNING] Unexpected error applying delta for model B{idx}: {e}. Using extracted code as fallback.')

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
