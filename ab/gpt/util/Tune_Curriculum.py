import os
import random
import shutil
import json
from os import makedirs
from os.path import isfile
import glob
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import ab.nn.api as lemur
import deepspeed
from ab.nn.util.Util import release_memory, create_file
from peft import PeftModel
from tqdm import tqdm
from ab.gpt.util.Const import nngpt_dir
import ab.gpt.NNEval as NNEval
from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.Const import *

from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.LoRA import LoRA
from ab.gpt.util.Util import (
    exists,
    extract_delta,
    extract_code,
    extract_hyperparam,
    extract_transform,
)

from ab.gpt.util.DeltaUtil import apply_delta, validate_delta, repair_code
from ab.gpt.util.Const import nngpt_upload
from ab.gpt.brute.trans.TransformEval import run_eval
from ab.gpt.util.prompt.TransformGenPrompt import TransformGenPrompt, load_data_from_folders
from ab.gpt.agents.state import AgentState

from ab.gpt.util.prompt.NNGenPromptCurriculum import NNGenPrompt

# from datasets import load_from_disk


ds_conf = conf_dir / 'DeepSpeed.json'

# Transform dir paths
TRANSFORM_OUT_DIR = trans_dir / 'dataset_epoch1'
TRANSFORM_RES_DIR = trans_dir / 'result_epoch1'

import re
REQUIRED_KEYS = {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

# For each key, what counts as valid usage (appears in code besides supported_hyperparameters)
PARAM_USAGE_PATTERNS = {
    'batch':     "prm['batch']",
    'epoch':     "prm['epoch']",
    'lr':        "prm['lr']",
    'momentum':  "prm['momentum']",
    'dropout':   "prm['dropout']",
    'transform': "prm['transform']",
}

# Inject into train_setup if missing
TRAIN_SETUP_INJECTIONS = {
    'batch':     "        self._batch_size = int(prm['batch'])\n",
    'epoch':     "        self._epochs = int(prm['epoch'])\n",
    'lr':        "",  # usually already used in optimizer
    'momentum':  "",  # usually already used in optimizer
    'dropout':   "        self._dropout = float(prm['dropout'])\n",
    'transform': "        self._transform = prm['transform']\n",
}


def fix_param_usage(code: str) -> str:
    """Ensure all required params appear at least twice in the code."""

    REQUIRED_KEYS = {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

    TRAIN_SETUP_INJECTIONS = {
        'batch':     "        self._batch_size = int(prm['batch'])\n",
        'epoch':     "        self._epochs = int(prm['epoch'])\n",
        'dropout':   "        self._dropout = float(prm['dropout'])\n",
        'transform': "        self._transform = prm['transform']\n",
        'lr':        "",
        'momentum':  "",
    }

    LEARN_REPLACEMENT = (
        "    def learn(self, train_data):\n"
        "        self.train()\n"
        "        for inputs, labels in train_data:\n"
        "            inputs = inputs.to(self.device).float()\n"
        "            labels = labels.to(self.device)\n"
        "            self.optimizer.zero_grad(set_to_none=True)\n"
        "            outputs = self(inputs)\n"
        "            loss = self.criterion(outputs, labels)\n"
        "            if torch.isfinite(loss):\n"
        "                loss.backward()\n"
        "                torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)\n"
        "                self.optimizer.step()\n"
        "        import gc; gc.collect()\n"
    )

    # ── step 1: fix supported_hyperparameters() ──────────────────────────────
    match = re.search(
        r'def supported_hyperparameters\(\):\s*\n\s*return\s*(\{[^}]*\})', code
    )
    if not match:
        inject = (
            "\ndef supported_hyperparameters():\n"
            "    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}\n"
        )
        code = inject + code
        print("[FIX] supported_hyperparameters() missing — injected")
    else:
        existing_keys = set(re.findall(r"['\"](\w+)['\"]", match.group(1)))
        missing_keys = REQUIRED_KEYS - existing_keys
        if missing_keys:
            all_keys = existing_keys | REQUIRED_KEYS
            new_set = "{" + ", ".join(f"'{k}'" for k in sorted(all_keys)) + "}"
            code = code[:match.start(1)] + new_set + code[match.end(1):]
            print(f"[FIX] Added to supported_hyperparameters(): {missing_keys}")

    # ── step 2: fix train_setup() — ONE replace with ALL injections ──────────
    if 'def train_setup(self, prm):' in code:

        # collect ALL keys that need injection before touching the code
        missing_injections = []
        for key, snippet in TRAIN_SETUP_INJECTIONS.items():
            count = code.count(f"'{key}'") + code.count(f'"{key}"')
            if count < 2 and snippet:
                missing_injections.append(snippet)
                print(f"[FIX] Will inject prm['{key}'] into train_setup()")

        if missing_injections:
            all_snippets = "".join(missing_injections)

            if 'self.prm = prm' not in code:
                # inject self.prm = prm AND all missing snippets in one replace
                full_block = "        self.prm = prm\n" + all_snippets
                code = code.replace(
                    'def train_setup(self, prm):',
                    f'def train_setup(self, prm):\n{full_block}',
                    1  # replace only first occurrence
                )
            else:
                # self.prm = prm already exists — append all snippets after it
                # in ONE replace so the anchor text stays stable
                code = code.replace(
                    '        self.prm = prm',
                    '        self.prm = prm\n' + all_snippets.rstrip(),
                    1  # replace only first occurrence
                )

    # ── step 3: fix learn() — replace only if it uses DataLoader ────────────
    lines = code.split('\n')
    learn_start = None
    learn_end = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def learn(') and learn_start is None:
            learn_start = i
        elif learn_start is not None and i > learn_start:
            if (stripped.startswith('def ') or stripped.startswith('class ')) \
                    and not line.startswith(' ' * 8):
                learn_end = i
                break

    if learn_end is None:
        learn_end = len(lines)

    if learn_start is not None:
        learn_body = '\n'.join(lines[learn_start:learn_end])
        if 'DataLoader' in learn_body:
            print("[FIX] learn() uses DataLoader — replacing with DataRoll-compatible version")
            lines[learn_start:learn_end] = LEARN_REPLACEMENT.split('\n')
            code = '\n'.join(lines)
    elif learn_start is None:
        print("[FIX] learn() not found — appending")
        code = code + '\n' + LEARN_REPLACEMENT

    # ── step 4: verify all params appear at least twice ──────────────────────
    all_fixed = True
    for key in REQUIRED_KEYS:
        count = code.count(f"'{key}'") + code.count(f'"{key}"')
        if count < 2:
            print(f"[WARN] '{key}' still appears only {count}x after fix")
            all_fixed = False

    if all_fixed:
        print("[FIX] ✓ All required params appear 2x+")

    # step 5 — fix features() receiving non-raw input
    # detect what variable is passed to self.features()
    feat_match = re.search(r'self\.features\((\w+)\)', code)
    if feat_match:
        feat_var = feat_match.group(1)
        if feat_var != 'x':
            # features is NOT receiving raw image
            print(f"[FIX] self.features({feat_var}) → self.features(x)")
            code = re.sub(
                r'self\.features\(' + feat_var + r'\)',
                'self.features(x)',
                code
            )
            # also fix curr_ch since features now takes raw image
            curr_ch_matches = list(re.finditer(r'\bcurr_ch\s*=\s*(\d+)', code))
            for m in curr_ch_matches:
                val = int(m.group(1))
                if val > 3:
                    code = code.replace(m.group(0), 'curr_ch = 3', 1)
                    print(f"[FIX] curr_ch={val} → 3")
    # step 7 — fix infer_dimensions: ensure dummy defined and on device
    if 'def infer_dimensions' in code:
        lines = code.split('\n')
        in_infer = False
        for i, line in enumerate(lines):
            if 'def infer_dimensions' in line:
                in_infer = True
            elif in_infer and line.strip().startswith('def '):
                in_infer = False
            if in_infer:
                # fix self.features(x) → self.features(dummy)
                if 'self.features(x)' in line:
                    lines[i] = line.replace('self.features(x)', 'self.features(dummy)')
                    print("[FIX] infer_dimensions: features(x) → features(dummy)")
                # fix backbone(x) → backbone(dummy)
                if re.search(r'self\.backbone\w*\(x\)', line):
                    lines[i] = re.sub(r'(self\.backbone\w*\()(x\))', r'\1dummy)', line)
                    print("[FIX] infer_dimensions: backbone(x) → backbone(dummy)")
                # fix dummy without .to(self.device)
                if 'torch.zeros' in line and '.to(' not in line:
                    lines[i] = line.rstrip() + '.to(self.device)'
                    print("[FIX] infer_dimensions: added .to(self.device) to dummy")
        code = '\n'.join(lines)

    # fix wrong permute
    if 'x.permute(0, 3, 1, 2)' in code:
        code = code.replace(
            'x = x.permute(0, 3, 1, 2)',
            '# input already [B,C,H,W]'
        )
        print("[FIX] Removed wrong x.permute(0,3,1,2)")

    # fix self.to(self.device) missing before infer_dimensions call
    if 'infer_dimensions' in code:
        if not re.search(r'self\.to\(self\.device\).*infer_dimensions', code, re.DOTALL):
            code = re.sub(
                r'([ \t]*)(self\.infer_dimensions\w*\()',
                lambda m: m.group(1) + 'self.to(self.device)\n' + m.group(1) + m.group(2),
                code, count=1
            )
            print("[FIX] Added self.to(self.device) before infer_dimensions call")
    return code
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


def tune(test_nn, nn_train_epochs, skip_epoch, llm_path, llm_tune_conf, nn_gen_conf, conf_keys, llm_conf, training_args,
         peft_config,
         max_prompts=None, save_llm_output=True, max_new_tokens=16 * 1024, nn_name_prefix=None, temperature=1.0,
         top_k=50, top_p=0.9, test_metric=None,
         onnx_run=False, trans_mode=False, prompt_batch=1, use_agents=False, use_predictor=False, use_backbone=False, enable_merge=False,use_unsloth=False):
    import torch
    import gc
    from pathlib import Path
    from datasets import Dataset as HFDataset

    if not isinstance(conf_keys, (list, tuple)):
        conf_keys = (conf_keys,)
    with open(conf_llm_dir / llm_conf) as f:
        config = json.load(f)
    assert isinstance(config, dict)

    token_from_file          = config['token_from_file']
    base_model_name          = config['base_model_name']
    llm_tune_epochs          = int(config['num_epochs'])
    use_deepspeed            = config['use_deepspeed']
    only_best_accuracy       = config['only_best_accuracy']
    context_length           = config.get('context_length')
    unsloth_max_input_length = config.get('max_input_length', None)
    use_unsloth              = config.get('use_unsloth', False)
    unsloth_load_in_4bit     = config.get('load_in_4bit', True)
    max_new_tokens           = config.get('max_new_tokens', max_new_tokens)

    # compute once — used to restore tokenizer after SFTTrainer overwrites it
    safe_max_length = context_length if context_length else None  # resolved after model load

    access_token = None
    if token_from_file:
        with open(ab_root_path / 'token') as f:
            access_token = f.readline()

    print(f'[DEBUG]Argument Information:\nSkip generation until Epoch: {skip_epoch}\nPath to saved LoRA Layers: {llm_path}')

    train_config_path = conf_train_dir / llm_tune_conf

    with open(conf_test_dir / nn_gen_conf) as prompt_file:
        prompt_dict = json.load(prompt_file)
    assert isinstance(prompt_dict, dict)

    from ab.gpt.util.LLM import LLM

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
    model     = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()

    # resolve safe_max_length now that model is loaded
    safe_max_length = context_length if context_length else model_loader.get_max_length()
    print(f"[INFO] safe_max_length={safe_max_length}  max_new_tokens={max_new_tokens}  "
          f"difference={safe_max_length - max_new_tokens}")

    if llm_path:
        print(f'Load saved LoRA layer from path: {llm_path}')
        model = PeftModel.from_pretrained(model, llm_path, is_trainable=True)
        model = model.merge_and_unload()

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

    chat_bot = ChatBot(model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)
    state = {
        "experiment_id": nn_name_prefix or "exp_default",
        "nn_name_prefix": nn_name_prefix,
        "current_epoch": 0,
        "llm_tune_epochs": llm_tune_epochs,
        "skip_epoch": skip_epoch,
        "next_action": "generate",
        "status": "pending",

        "model": model,
        "tokenizer": tokenizer,
        "model_loader": model_loader,
        "lora_tuner": lora_tuner,
        "chat_bot": chat_bot,

        "prompt_dict": prompt_dict,
        "conf_keys": conf_keys,
        "test_nn": test_nn,
        "nn_train_epochs": nn_train_epochs,
        "max_new_tokens": max_new_tokens,
        "save_llm_output": save_llm_output,
        "prompt_batch": prompt_batch,

        "context_length": context_length,
        "use_unsloth": use_unsloth,
        "unsloth_max_input_length": unsloth_max_input_length,
        "train_config_path": train_config_path,
        "only_best_accuracy": only_best_accuracy,
        "base_model_name": base_model_name,
        "trans_mode": trans_mode,
        "max_prompts": max_prompts,

        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,

        "use_predictor": use_predictor,
        "use_backbone": use_backbone,
        "enable_merge": enable_merge
    }

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    if use_agents:
        from ab.gpt.agents.run_agent import run_agent_controller
        return run_agent_controller(state)
    for epoch in range(llm_tune_epochs):
        print(f'[INFO]Start Epoch {epoch}')
        out_path = epoch_dir(epoch)

        # ── GENERATION ──────────────────────────────────────────────────────
        if epoch < skip_epoch:
            print(f'Skipped generation at epoch {epoch}')
        else:
            if trans_mode:
                trans_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs,
                          prompt_dict, test_nn, max_new_tokens, save_llm_output,
                          nn_name_prefix)
            else:
                nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs,
                       prompt_dict, test_nn, max_new_tokens, save_llm_output,
                       nn_name_prefix, unsloth_max_input_length, prompt_batch)

                # ── check success rate BEFORE fine-tuning ──
                cycle_file = nngpt_dir / "cycle_results.json"
                if cycle_file.exists():
                    try:
                        cycle_data = json.loads(cycle_file.read_text())
                        successful = cycle_data.get("evaluation", {}).get("models_trained", 0)
                        if successful < 1:
                            print(f"[WARN] No successful models — skipping fine-tuning")
                            release_memory()
                            lemur.data.cache_clear()
                            continue
                        elif successful < 2:
                            print(f"[WARN] Only {successful} model(s) — fine-tuning with limited data")
                        else:
                            print(f"[INFO] {successful} successful models — fine-tuning")
                    except Exception as e:
                        print(f"[WARN] Could not check success rate: {e}")

        # ── FINE-TUNING ──────────────────────────────────────────────────────
        # runs for both trans_mode and nn_mode
        print(f'[DEBUG]Perform finetune at epoch {epoch}.')

        if trans_mode:
            data_processor = TransformGenPrompt(
                context_length if context_length else model_loader.get_max_length(),
                tokenizer, train_config_path, TRANSFORM_OUT_DIR, TRANSFORM_RES_DIR
            )
        else:
            max_len = (unsloth_max_input_length if use_unsloth and unsloth_max_input_length
                       else context_length if context_length
            else model_loader.get_max_length())
            data_processor = NNGenPrompt(max_len, tokenizer, train_config_path)


        raw_df = data_processor.get_raw_dataset(
            only_best_accuracy,
            n_training_prompts=max_prompts,
        )

        train_df = raw_df[
            (raw_df["category"] == "train") &
            (raw_df["text"].str.len() > 0)
        ].copy().reset_index(drop=True)

        print(f"Training rows available: {len(train_df)}")

        if len(train_df) == 0:
            print(f"[WARN] empty training dataset at epoch {epoch}. Skipping fine-tuning.")
            release_memory()
            lemur.data.cache_clear()
            continue

        dataset = HFDataset.from_pandas(train_df[["text"]], preserve_index=False)
        print(f"Dataset length: {len(dataset)}  columns: {dataset.column_names}")

        model.train()

        torch.cuda.empty_cache()
        gc.collect()
        print(f"[INFO] Cache cleared. Free: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

        safe_name        = Path(base_model_name).name
        adapter_save_path = out_path / safe_name

        model = lora_tuner.train(dataset, tokenizer, adapter_save_path)

        # ── CRITICAL: restore tokenizer IMMEDIATELY after train() ────────────
        # LoRA.py line 246 hardcodes model_max_length=4096 inside train()
        # This causes OverflowError in subsequent generation:
        #   max_length = 4096 - 16384 = -12288 (negative → OverflowError)
        tokenizer.model_max_length = safe_max_length
        tokenizer.truncation_side = "right"
        tokenizer.padding_side = "right"
        print(f"[INFO] Tokenizer restored: max_length={tokenizer.model_max_length}, "
              f"truncation_side={tokenizer.truncation_side}, "
              f"padding_side={tokenizer.padding_side}")
        # cast model back to bfloat16 after LoRA training
        # SFTTrainer promotes some layers to float32 during training
        model = model.to(torch.bfloat16)
        print("[INFO] Model cast back to bfloat16")

        # ── reinitialize ChatBot so next epoch uses updated model + dtype ────
        model.eval()
        chat_bot = ChatBot(model, tokenizer, temperature=temperature,
                           top_k=top_k, top_p=top_p)
        print("[INFO] ChatBot reinitialized after fine-tuning")

        del dataset
        release_memory()
        torch.cuda.empty_cache()
        print(f"[INFO] Fine-tuning complete. Free: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

        # verify adapter was saved
        adapter_cfg = adapter_save_path / "adapter_config.json"
        if adapter_cfg.exists():
            print(f"[INFO] Adapter saved: {adapter_save_path}")
        else:
            print(f"[ERROR] adapter_config.json missing at {adapter_save_path} — fine-tuning may have failed")

        lemur.data.cache_clear()
#new nn_gen
def nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens,
           save_llm_output, nn_name_prefix, unsloth_max_input_length, prompt_batch):
    print('Preparing prompts for generation, this might take a while...')

    tmp_prompt_path = conf_test_dir / "__tmp_curriculum_runtime.json"
    with open(tmp_prompt_path, "w") as f:
        json.dump({k: prompt_dict[k] for k in conf_keys}, f, indent=2)

    prompt_builder = NNGenPrompt(
        max_len=chat_bot.tokenizer.model_max_length,
        tokenizer=chat_bot.tokenizer,
        prompts_path=str(tmp_prompt_path),
    )

    df = prompt_builder.get_raw_dataset(
        only_best_accuracy=False,
        n_training_prompts=test_nn * 10,
    )

    if "category" in df.columns:
        df = df[df["category"] == "generation"].copy()

    print(f"[INFO] Curriculum prompt rows ready: {len(df)}")

    if df.empty:
        print("[WARN] No curriculum generation rows available.")
        return

    models_dir = synth_dir(out_path)
    makedirs(models_dir, exist_ok=True)

    for idx, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df)):
        model_dir = models_dir / f'B{idx}'
        prompt = row["instruction"]

        if unsloth_max_input_length:
            rendered = chat_bot.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
            if len(rendered) > unsloth_max_input_length:
                print(f"[WARN] Prompt too long ({len(rendered)}), skipping B{idx}")
                continue

        code, hp, tr, full_out = chat_bot.chat(
            prompt,
            engineer_prompt=False,
            max_new_tokens=max_new_tokens,
        )

        makedirs(model_dir, exist_ok=True)

        if save_llm_output:
            create_file(model_dir, new_out_file, full_out)

        if not code or "<nn>" not in full_out:
            print(f"[ERROR] No valid <nn> block generated for model B{idx}")
            release_memory()
            continue
        try:
            if hp and hp.strip():
                hp_obj = json.loads(hp.replace("'", '"'))
                # ensure it's a dict not a list
                if not isinstance(hp_obj, dict):
                    print(f"[WARN] hp parsed as {type(hp_obj).__name__} — using defaults")
                    hp_obj = {}
                # inject defaults for missing keys
                hp_defaults = {
                    'batch': 32, 'dropout': 0.2, 'epoch': 1,
                    'lr': 0.01, 'momentum': 0.9, 'transform': 'norm_256_flip'
                }
                for k, v in hp_defaults.items():
                    if k not in hp_obj:
                        hp_obj[k] = v
                with open(model_dir / hp_file, "w+") as f:
                    json.dump(hp_obj, f)

            else:
                print("[WARNING] No hyperparameters generated")
        except Exception as e:
            print(f"[WARNING] Error processing hyperparameters: {e}")

        try:
            if tr and tr.strip():
                create_file(model_dir, transformer_file, tr)
            else:
                print("[WARNING] No transformer code generated")
        except Exception as e:
            print(f"[WARNING] Error saving transformer: {e}")

        if code and code.strip():
            import re
            # strip HTML comments
            code = re.sub(r'<!--.*?-->', '', code, flags=re.DOTALL).strip()

            if not code:
                print(f"[ERROR] Code empty after stripping HTML comments for B{idx}")
                continue

            code = fix_param_usage(code)

            # final batch count check
            batch_count = code.count("'batch'") + code.count('"batch"')
            if batch_count < 2:
                print(f"[WARN] B{idx}: batch fix failed — will likely fail Eval")
            else:
                print(f"[INFO] B{idx}: 'batch' appears {batch_count}x ✓")

            # save
            create_file(model_dir, new_nn_file, code)
            print(f'[INFO] Saved code to {model_dir / new_nn_file}')
        else:
            print(f"[ERROR] No code generated for model B{idx}")
            release_memory()
            continue

        try:
            row.to_pickle(model_dir / "dataframe.df")
        except Exception:
            pass

        release_memory()

    print('[DEBUG] Release memory.')
    release_memory()

    valid_nn_files = list(models_dir.glob(f'*/{new_nn_file}')) if exists(models_dir) else []

    if valid_nn_files:
        # ── GPU release before eval ──────────────────────────────────────────
        import torch, gc, time
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        print(f"[INFO] GPU free before eval: {free_mem:.1f}GB")

        max_wait = 180  # seconds
        waited = 0
        while free_mem < 10.0 and waited < max_wait:
            time.sleep(5)
            torch.cuda.empty_cache()
            gc.collect()
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            waited += 5
            print(f"[INFO] Waiting for GPU... free={free_mem:.1f}GB waited={waited}s")

        print(f"[INFO] GPU ready: {free_mem:.1f}GB free — starting eval")
        # ── cleanup spurious ab/nn directory if present ──────────────────────
        import shutil
        spurious_nn = Path("ab/nn")
        if spurious_nn.exists() and not (spurious_nn / "api.py").exists():
            shutil.rmtree(spurious_nn)
            print("[FIX] Removed spurious ab/nn directory")

        NNEval.main(nn_name_prefix, nn_train_epochs, epoch, batch=16, epoch_limit_minutes=8)
        print('[DEBUG] Release_memory.')
        release_memory()
    else:
        print('[WARN] NO valid generated nn files found. Skipping evaluation.')
    # === EPOCH TRACKER ===
    from datetime import datetime
    tracker_file = nngpt_dir / "epoch_tracker.json"

    tracker_data = []
    if tracker_file.exists():
        try:
            with open(tracker_file) as f:
                tracker_data = json.load(f)
        except Exception:
            tracker_data = []

    # get accuracy and success metrics from cycle_results
    accuracy = None
    success_rate = 0.0
    successful_models = 0
    failed_models = 0

    cycle_file = nngpt_dir / "cycle_results.json"
    if cycle_file.exists():
        try:
            with open(cycle_file) as f:
                cycle_data = json.load(f)
            ev = cycle_data.get("evaluation", {})
            accuracy = ev.get("best_accuracy")
            success_rate = ev.get("success_rate", 0.0)
            successful_models = ev.get("models_trained", 0)
            total_generated = cycle_data.get("generation", {}).get("total_generated", 0)
            failed_models = total_generated - successful_models
        except Exception:
            pass

    score = round(success_rate * (accuracy or 0.0), 4)

    tracker_data.append({
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "models_generated": len(list(models_dir.glob("B*"))) if exists(models_dir) else 0,
        "accuracy": accuracy,
        "successful_models": successful_models,
        "failed_models": failed_models,
        "success_rate": round(success_rate, 4),
        "score": score,
    })

    tracker_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tracker_file, "w") as f:
        json.dump(tracker_data, f, indent=2)

    print(f"[EPOCH TRACKER] Wrote epoch {epoch} "
          f"(acc={accuracy}, success={successful_models}, "
          f"failed={failed_models}, score={score})")

    print('Clear LEMUR query cache.')
    lemur.data.cache_clear()
    print('The cache has been cleared.')

def trans_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict_global, test_nn, max_new_tokens,
              save_llm_output, nn_name_prefix):
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
        print(f"[DEBUG] full_out length: {len(full_out) if full_out else 0}")
        print(f"[DEBUG] full_out preview:\n{full_out[:1000] if full_out else 'EMPTY'}")

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


# ============================================================
# SINGLE SOURCE OF TRUTH: STEP WRAPPERS
# These are what the AGENTS call (NOT reimplementing anything)
# ============================================================

def _has_generated_nn_code(out_path) -> bool:
    """Returns True if at least one synthesized model directory B*/ contains new_nn.py."""
    models_dir = synth_dir(out_path)
    if not exists(models_dir):
        return False
    for bdir in glob.glob(str(models_dir / "B*")):
        if isfile(os.path.join(bdir, new_nn_file)):
            return True
    return False


def generate_step(state: AgentState) -> dict:
    epoch = state["current_epoch"]
    skip_epoch = state.get("skip_epoch", 0)
    out_path = epoch_dir(epoch)

    # If generation is skipped, there is nothing new to predict on.
    if epoch < skip_epoch:
        print(f"[INFO] Skipped generation at epoch {epoch}")
        return {"next_action": "finetune"}

    print(f"[INFO] Generation at epoch {epoch}")

    if state.get("trans_mode", False):
        trans_gen(
            epoch,
            out_path,
            state["chat_bot"],
            state["conf_keys"],
            state["nn_train_epochs"],
            state["prompt_dict"],
            state["test_nn"],
            state["max_new_tokens"],
            state["save_llm_output"],
            state.get("nn_name_prefix"),
        )
    else:
        nn_gen(
            epoch,
            out_path,
            state["chat_bot"],
            state["conf_keys"],
            state["nn_train_epochs"],
            state["prompt_dict"],
            state["test_nn"],
            state["max_new_tokens"],
            state["save_llm_output"],
            state.get("nn_name_prefix"),
            state.get("unsloth_max_input_length"),
            state.get("prompt_batch", 1),
            use_backbone=state.get("use_backbone",False),
        )

    if not _has_generated_nn_code(out_path):
        print(f"[INFO] No code generated at epoch {epoch}, skipping evaluation")
        return {"next_action": "finetune"}

    return {"next_action": "evaluate"}


def _evaluate_epoch(epoch, out_path, nn_name_prefix, nn_train_epochs, trans_mode):
    """
    Single source of truth for one evaluation epoch.
    Runs NNEval (trains generated NNs for nn_train_epochs and records accuracy).
    Called by both the classic for-loop and the agent evaluator node.
    Returns a dict with accuracy results that the predictor can read.
    """
    models_dir = synth_dir(out_path)
    results = {"epoch": epoch}

    if exists(models_dir):
        if trans_mode:
            try:
                run_eval(epoch_num=epoch, FT_MODE=True)
                print('[DEBUG] Release_memory.')
            except Exception as e:
                print(f"Error running evaluation main(): {e}", flush=True)
            print('Folder data reload will occur next epoch.')
        else:
            NNEval.main(nn_name_prefix, nn_train_epochs, epoch)


            print('[DEBUG] Release_memory.')
            release_memory()

        print('Clear LEMUR query cache.')
        lemur.data.cache_clear()
        print('The cache has been cleared.')


    # Read accuracy from cycle_results.json (written by NNEval after evaluation)
    cycle_file = out_path.parent / "cycle_results.json"
    if cycle_file.is_file():
        try:
            with open(cycle_file) as f:
                cycle_data = json.load(f)
            best_acc = (
                cycle_data.get("evaluation", {}).get("best_accuracy")
                or cycle_data.get("best_accuracy")
                or cycle_data.get("accuracy")
            )
            if best_acc is not None:
                results[f"epoch_{epoch + 1}_accuracy"] = float(best_acc)
        except Exception:
            pass

    # Collect all predictor inputs from the first successful model's files.
    # Classic for-loop ignores these extra keys — agent evaluate_step passes them to state.
    if exists(models_dir):
        for bdir in sorted(glob.glob(str(models_dir / "B*"))):
            eval_info_path = os.path.join(bdir, "eval_info.json")
            df_path        = os.path.join(bdir, "dataframe.df")
            nn_path        = os.path.join(bdir, new_nn_file)
            tr_path        = os.path.join(bdir, transformer_file)

            if not isfile(eval_info_path):
                continue
            try:
                with open(eval_info_path) as f:
                    eval_info = json.load(f)
                cli  = eval_info.get("cli_args", {})
                args = eval_info.get("eval_args", {})
                # use exact DB column names so predictor can use them directly
                results["task"]           = cli.get("task", "")
                results["dataset"]        = cli.get("dataset", "")
                results["metric"]         = cli.get("metric", "")
                results["prm"]            = args if args else {}
                if isfile(nn_path):
                    with open(nn_path) as f:
                        results["nn_code"] = f.read()
                if isfile(tr_path):
                    with open(tr_path) as f:
                        results["transform_code"] = f.read()
                # fallback: read extra fields from dataframe.df
                if isfile(df_path):
                    try:
                        origdf = pd.read_pickle(df_path)
                        if not results.get("transform_code"):
                            results["transform_code"] = origdf.get("transform_code", "")
                        if not results.get("task"):
                            results["task"] = origdf.get("task", "")
                        if not results.get("dataset"):
                            results["dataset"] = origdf.get("dataset", "")
                        if not results.get("metric"):
                            results["metric"] = origdf.get("metric", "")
                        if not results.get("prm"):
                            results["prm"] = origdf.get("prm", {})
                        # nn name (used by predictor to look up DB IDs)
                        results["nn"] = origdf.get("nn", "")
                    except Exception:
                        pass
                break  # first successful model is enough
            except Exception:
                continue

    return results


def evaluate_step(state: AgentState) -> dict:
    """Thin agent wrapper — all logic lives in _evaluate_epoch()."""
    epoch = state["current_epoch"]
    out_path = epoch_dir(epoch)
    print(f"[INFO] Evaluating at epoch {epoch}")

    results = _evaluate_epoch(
        epoch,
        out_path,
        state.get("nn_name_prefix"),
        state["nn_train_epochs"],
        state.get("trans_mode", False),
    )

    updates = {}

    # Count actual evaluations that produced results (not epoch numbers)
    # epoch_1_accuracy = first real evaluation, epoch_2_accuracy = second real evaluation
    # This works correctly with skip_epoch — epoch 0 skips generation so produces no accuracy
    has_epoch1_in_state = state.get("epoch_1_accuracy") is not None
    has_epoch2_in_state = state.get("epoch_2_accuracy") is not None

    acc_key = f"epoch_{epoch + 1}_accuracy"
    best_acc = results.get(acc_key)

    if best_acc is not None:
        if not has_epoch1_in_state:
            updates["epoch_1_accuracy"] = best_acc
        elif not has_epoch2_in_state:
            updates["epoch_2_accuracy"] = best_acc

    # Pass all predictor inputs to state — names match exact DB column names
    for field in ["nn_code", "prm", "task", "dataset", "metric", "transform_code", "nn"]:
        if field in results:
            updates[field] = results[field]

    # Route to predictor only if enabled AND we have at least 2 epochs of results
    use_predictor = state.get("use_predictor", False)
    has_epoch1 = has_epoch1_in_state or "epoch_1_accuracy" in updates
    has_epoch2 = has_epoch2_in_state or "epoch_2_accuracy" in updates

    if use_predictor and has_epoch1 and has_epoch2:
        updates["next_action"] = "predict"
    else:
        updates["next_action"] = "finetune"

    return updates


def _finetune_epoch(
    epoch, out_path, model, tokenizer, model_loader, lora_tuner,
    context_length, use_unsloth, unsloth_max_input_length,
    train_config_path, only_best_accuracy, max_prompts,
    max_new_tokens, base_model_name, trans_mode,
    temperature=1.0, top_k=50, top_p=0.9,
    use_backbone=False,
):
    """
    Single source of truth for one finetune epoch.
    Called by both the classic for-loop and the agent finetuner node.
    Returns (model, chat_bot) with the newly fine-tuned model.
    """
    if trans_mode:
        data_processor = TransformGenPrompt(
            context_length if context_length else model_loader.get_max_length(),
            tokenizer,
            train_config_path,
            TRANSFORM_OUT_DIR,
            TRANSFORM_RES_DIR,
        )
    elif use_backbone:
        from ab.gpt.util.prompt.SFTGenPrompt import SFTGenPrompt
        data_processor = SFTGenPrompt(
            context_length if context_length else model_loader.get_max_length(),
            tokenizer
        )
    else:
        length = (
            unsloth_max_input_length if (use_unsloth and unsloth_max_input_length)
            else context_length if context_length
            else model_loader.get_max_length()
        )
        data_processor = NNGenPrompt(length, tokenizer, train_config_path)

    dataset = data_processor.get_dataset(
        only_best_accuracy,
        max_prompts=max_prompts,
        max_new_tokens=max_new_tokens,
    )

    print("Dataset length:", len(dataset))
    model.train()
    model = lora_tuner.train(dataset, tokenizer, out_path / base_model_name)

    del dataset
    release_memory()

    chat_bot = ChatBot(model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)
    return model, chat_bot


def finetune_step(state: AgentState) -> dict:
    """Thin agent wrapper — all logic lives in _finetune_epoch()."""
    epoch = state["current_epoch"]
    out_path = epoch_dir(epoch)
    print(f"[DEBUG] Perform finetune at epoch {epoch}")

    model, chat_bot = _finetune_epoch(
        epoch, out_path,
        state["model"], state["tokenizer"], state["model_loader"], state["lora_tuner"],
        state.get("context_length"), state.get("use_unsloth", False),
        state.get("unsloth_max_input_length"),
        state["train_config_path"], state["only_best_accuracy"],
        state.get("max_prompts"), state["max_new_tokens"],
        state["base_model_name"], state.get("trans_mode", False),
        state.get("temperature", 1.0), state.get("top_k", 50), state.get("top_p", 0.9),
        state.get("use_backbone", False),
    )

    return {
        "model": model,
        "chat_bot": chat_bot,
        "current_epoch": epoch + 1,
        "next_action": "generate",
    }