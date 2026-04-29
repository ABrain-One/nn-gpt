# ab/gpt/util/Tune.py
"""
ab/gpt/util/Tune.py — Central tuning pipeline for NNGPT.
- tune() is the ONLY entry point.
- All logic lives here: nn_gen, trans_gen, generate_step, finetune_step.
- Agents only call generate_step() and finetune_step() from this file.
"""


import os
import random
import shutil
import json
from os import makedirs
from os.path import isfile
import glob
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
import ab.nn.api as lemur
import deepspeed
from ab.nn.util.Util import release_memory, create_file
from peft import PeftModel
from tqdm import tqdm

import ab.gpt.NNEval as NNEval
from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.Const import *
from ab.gpt.util.Const import nngpt_dir

from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.LoRA import LoRA
from ab.gpt.util.Util import (
    exists,
    extract_delta,
    extract_code,
    extract_hyperparam,
    extract_transform,
)
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from ab.gpt.util.DeltaUtil import apply_delta, validate_delta, repair_code
from ab.gpt.util.Const import nngpt_upload
import ab.gpt.util.SFTUtil as SFTUtil
from ab.gpt.brute.trans.TransformEval import run_eval
from ab.gpt.util.prompt.TransformGenPrompt import TransformGenPrompt, load_data_from_folders
from ab.gpt.agents.state import AgentState
import ab.gpt.util.training_runtime as TrainingRuntime

ds_conf = conf_dir / 'DeepSpeed.json'
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

            chunks.append({"input_ids": chunk_input_ids, "attention_mask": chunk_attention_mask})
    return {"chunks": chunks}


def flatten_chunks(data):
    all_chunks = sum(data["chunks"], [])  # flatten batched list
    return {
        "input_ids": [chunk["input_ids"] for chunk in all_chunks],
        "attention_mask": [chunk["attention_mask"] for chunk in all_chunks],
    }


# ============================================================
# SINGLE SOURCE OF TRUTH: GENERATION (nn_gen / trans_gen)
# ============================================================

def nn_gen(
    epoch,
    out_path,
    chat_bot,
    conf_keys,
    nn_train_epochs,
    prompt_dict,
    test_nn,
    max_new_tokens,
    save_llm_output,
    nn_name_prefix,
    unsloth_max_input_length,
    prompt_batch,
    use_backbone=False,
):
    print("Preparing prompts for generation, this might take a while...")

    use_delta = nn_name_prefix == "delta"
    if not use_delta and isinstance(prompt_dict, dict) and conf_keys:
        first_key = conf_keys[0] if isinstance(conf_keys, (list, tuple)) else conf_keys
        key_config = prompt_dict.get(first_key, {})
        if isinstance(key_config, dict):
            use_delta = key_config.get("use_delta", False) or "delta" in str(first_key).lower()

    prompts = []
    for key in conf_keys:
        prompt = ""
        key_config = prompt_dict[key]
        for pr in key_config["prompt"]:
            prompt += pr + "\n"

        num_joint_nns = key_config.get("num_joint_nns", 1)
        use_join = num_joint_nns >= 2
        if use_join:
            from ab.nn.util.db.Query import JoinConf
            from ab.gpt.util.lemur_enrichment import patch_join_nn_query, enrich_dataframe
            patch_join_nn_query()
            data = lemur.data(
                only_best_accuracy=True,
                task=key_config["task"],
                sql=JoinConf(
                    num_joint_nns=num_joint_nns,
                    same_columns=tuple(key_config.get("keep_same", [])),
                    diff_columns=tuple(key_config.get("no_repeat", [])),
                    enhance_nn=key_config.get("improve", False),
                ),
            )[:test_nn]
            if key_config.get("output_type") == "classification":
                enrich_dataframe(data)
            addon_data = None
        else:
            data = (
                lemur.data(only_best_accuracy=True, task=key_config["task"])
                .groupby(by="nn")
                .sample(n=1)[:test_nn]
            )
            addon_task = key_config.get("addon_task")
            addon_data = lemur.data(only_best_accuracy=True, task=addon_task) if addon_task else None

        output_type = key_config.get("output_type", "code")
        nn_code_max_chars = key_config.get("nn_code_max_chars")

        for _, row in data.iterrows():
            para_dict = {}
            for it in key_config["input_list"]:
                para_dict[it["para"]] = row[it["value"]]
            if use_backbone:
                target_pattern = None
                if "nn_code" in row and isinstance(row["nn_code"], str):
                    target_pattern = SFTUtil.extract_target_pattern_from_code(row["nn_code"])
                para_dict["target_pattern"] = target_pattern or SFTUtil.available_patterns[len(prompts) % len(SFTUtil.available_patterns)]
            if nn_code_max_chars and "nn_code" in para_dict and isinstance(para_dict["nn_code"], str):
                para_dict["nn_code"] = para_dict["nn_code"][:nn_code_max_chars]

            if addon_data is not None and not addon_data.empty:
                available_addon = addon_data.loc[addon_data.nn != row["nn"]]
                if not available_addon.empty:
                    addon_row = available_addon.sample(n=1).iloc[0]
                    if key_config.get("addon_list"):
                        for it in key_config["addon_list"]:
                            para_dict[it["para"]] = addon_row[it["value"]]

            prompts.append((prompt.format(**para_dict), row, output_type))

    models_dir = synth_dir(out_path)

    if use_delta:
        for idx, prompt_data in tqdm(enumerate(prompts)):
            model_dir = models_dir / f"B{idx}"
            prompt_text, origdf, output_type = prompt_data

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

            _, hp, tr, full_out = chat_bot.chat(prompt_text, engineer_prompt=False, max_new_tokens=max_new_tokens)

            if use_backbone:
                from ab.gpt.util.SFTUtil import skeleton_code
                from ab.gpt.util.Util import extract_str
                import textwrap
                block_code = extract_str(full_out, '<block>', '</block>')
                init_code = extract_str(full_out, '<init>', '</init>')
                forward_code = extract_str(full_out, '<forward>', '</forward>')
                if block_code and init_code and forward_code:
                    code = skeleton_code
                    sig_block = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
                    code = code.replace(sig_block, textwrap.dedent(block_code))
                    sig_init = "    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
                    code = code.replace(sig_init, textwrap.indent(textwrap.dedent(init_code), "    "))
                    sig_forward = "    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"
                    code = code.replace(sig_forward, textwrap.indent(textwrap.dedent(forward_code), "    "))
                else:
                    code = extract_code(full_out)


            if code is None:
                print(f'[ERROR] No code generated for model B{idx}')
                continue  # Skip if no code is generated at all


            makedirs(model_dir, exist_ok=True)
            if save_llm_output:
                create_file(model_dir, new_out_file, full_out)

            code = None
            current_out = full_out
            current_prompt = prompt_text

            for attempt in range(_MAX_DELTA_RETRIES + 1):
                if attempt > 0:
                    _, _, _, current_out = chat_bot.chat(
                        current_prompt, engineer_prompt=False, max_new_tokens=max_new_tokens
                    )

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

            if code is None:
                print(f'[WARNING] All delta attempts failed for B{idx}. Trying syntax repair on extracted code.')
                raw_code = extract_code(full_out)
                if raw_code:
                    repaired = repair_code(raw_code)
                    if repaired:
                        code = repaired
                        print(f'[INFO] Used syntax-repaired code fallback for B{idx}')

            hp_str = extract_hyperparam(full_out)
            tr_str = extract_transform(full_out)

            try:
                print(f'Generated params: {hp_str}')
                if hp_str and hp_str.strip():
                    hp_obj = json.loads(hp_str.replace("'", '"'))
                    with open(model_dir / hp_file, 'w+') as f:
                        json.dump(hp_obj, f)
                else:
                    print('[WARNING] No hyperparameters generated, skipping hp file')
            except Exception as e:
                print(f"[WARNING] Error processing hyperparameters: {e}")

            try:
                print(f'Generated transformer:\n\n{tr_str}\n----\n')
                if tr_str and tr_str.strip():
                    create_file(model_dir, transformer_file, tr_str)
                else:
                    print('[WARNING] No transformer code generated')
            except Exception as e:
                print(f'[WARNING] Error saving transformer: {e}')

            if code and code.strip():
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

    else:
        pending = []
        for idx, prompt_data in tqdm(enumerate(prompts)):
            prompt_text, origdf, output_type = prompt_data

            if unsloth_max_input_length:
                in_text = [{"role": "user", "content": prompt_text}]
                output = chat_bot.tokenizer.apply_chat_template(in_text, add_generation_prompt=True)
                print(f'Sample prompt length: {len(output)}, max_input_length: {unsloth_max_input_length}')
                if len(output) > unsloth_max_input_length:
                    print(f'Prompt is too long, skipping...')
                    continue

            pending.append((idx, prompt_text, origdf, output_type))

        if prompt_batch < 1:
            prompt_batch = 1
        if prompt_batch > 1:
            print(f'[INFO] Batch generation enabled: prompt_batch={prompt_batch}')

        for start in range(0, len(pending), prompt_batch):
            batch = pending[start : start + prompt_batch]
            batch_prompts = [item[1] for item in batch]

            if prompt_batch > 1 and hasattr(chat_bot, 'chat_batch'):
                batch_outputs = chat_bot.chat_batch(batch_prompts, engineer_prompt=False, max_new_tokens=max_new_tokens)
            else:
                batch_outputs = [chat_bot.chat(p, engineer_prompt=False, max_new_tokens=max_new_tokens) for p in batch_prompts]

            for (idx, prompt_text, origdf, output_type), output in zip(batch, batch_outputs):
                model_dir = models_dir / f"B{idx}"
                code, hp, tr, full_out = output
                if use_backbone:
                    code = SFTUtil.assemble_backbone_xml_completion(full_out)
                    if code is None:
                        print(f'[ERROR] Missing backbone XML tags for model B{idx}')

                makedirs(model_dir, exist_ok=True)
                if output_type == "classification":
                    create_file(model_dir, new_out_file, full_out)
                    if origdf is not None:
                        origdf.to_pickle(model_dir / "dataframe.df")
                    continue
                if save_llm_output:
                    create_file(model_dir, new_out_file, full_out)

                try:
                    print(f'Generated params: {hp}')
                    if hp and hp.strip():
                        hp = json.loads(hp.replace("'", '"'))
                        with open(model_dir / hp_file, 'w+') as f:
                            json.dump(hp, f)
                    else:
                        print('[WARNING] No hyperparameters generated, skipping hp file')
                except Exception as e:
                    print(f'[WARNING] Error processing hyperparameters: {e}')

                try:
                    print(f'Generated transformer:\n\n{tr}\n----\n')
                    if tr and tr.strip():
                        create_file(model_dir, transformer_file, tr)
                    else:
                        print('[WARNING] No transformer code generated')
                except Exception as e:
                    print(f'[WARNING] Error saving transformer: {e}')

                if code and code.strip():
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

    # Track generation-side progress even before later merge logic or external
    # tooling reads cycle_results.json.
    tracker_file = nngpt_dir / "epoch_tracker.json"
    if tracker_file.exists():
        try:
            with open(tracker_file) as f:
                tracker_data = json.load(f)
        except Exception:
            tracker_data = []
    else:
        tracker_data = []

    accuracy = None
    cycle_file = nngpt_dir / "cycle_results.json"
    if cycle_file.exists():
        try:
            with open(cycle_file) as f:
                cycle_data = json.load(f)
            accuracy = cycle_data.get("evaluation", {}).get("best_accuracy")
        except Exception:
            pass

    tracker_data.append(
        {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "models_generated": len(list(models_dir.glob("B*"))) if exists(models_dir) else 0,
            "accuracy": accuracy,
        }
    )
    tracker_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tracker_file, "w") as f:
        json.dump(tracker_data, f, indent=2)
    print(f"[EPOCH TRACKER] Wrote epoch {epoch} (acc={accuracy})")

    print('[DEBUG] Release memory.')
    release_memory()


def trans_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict_global, test_nn, max_new_tokens, save_llm_output, nn_name_prefix):
    """
    Transform Script Generation
    """
    print('Running Transform Generation...')

    out_gen_dir = str(TRANSFORM_OUT_DIR)
    result_gen_dir = str(TRANSFORM_RES_DIR)

    prompts = []

    all_data = load_data_from_folders(out_gen_dir, result_gen_dir, only_best_accuracy=True)
    if len(all_data) == 0:
        print("Warning: No data loaded from folders for generation. Skipping.", flush=True)
        return

    for key in conf_keys:
        prompt_config = prompt_dict_global[key]
        prompt = ''
        for pr in prompt_config['prompt']:
            prompt += pr + '\n'

        if len(all_data) < test_nn:
            print(f"Warning: Requested {test_nn} samples, but only {len(all_data)} available. Using all.", flush=True)
            data_sample = all_data.sample(n=len(all_data))
        else:
            data_sample = all_data.sample(n=test_nn)

        addon_data = all_data

        for _, row in data_sample.iterrows():
            para_dict = {}
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
        prompt_text, origdf = prompt_data

        code, hp, tr, full_out = chat_bot.chat(prompt_text, engineer_prompt=False, max_new_tokens=max_new_tokens)

        makedirs(model_dir, exist_ok=True)
        if save_llm_output:
            create_file(model_dir, new_out_file, full_out)

        if tr and tr.strip():
            print(f'Generated transformer:\n\n{tr}\n----\n')
            create_file(model_dir, transformer_file, tr)
        else:
            print(f"[ERROR] No code generated for model B{idx}")
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


def _has_generated_output(out_path) -> bool:
    """Returns True if at least one synthesized model directory B*/ contains full_output.txt."""
    models_dir = synth_dir(out_path)
    if not exists(models_dir):
        return False
    for bdir in glob.glob(str(models_dir / "B*")):
        if isfile(os.path.join(bdir, new_out_file)):
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

    # Classification prompts may intentionally emit labels or structured output
    # without generating a runnable new_nn.py file.
    classification_mode = state.get("classification_mode", False)
    has_output = _has_generated_output(out_path) if classification_mode else _has_generated_nn_code(out_path)
    if not has_output:
        print(f"[INFO] No code generated at epoch {epoch}, skipping evaluation")
        return {"next_action": "finetune"}

    return {"next_action": "evaluate"}


def _evaluate_epoch(epoch, out_path, nn_name_prefix, nn_train_epochs, trans_mode, classification_mode=False):
    """
    Single source of truth for one evaluation epoch.
    Runs NNEval (trains generated NNs for nn_train_epochs and records accuracy).
    Called by both the classic for-loop and the agent evaluator node.
    Returns a dict with accuracy results that the predictor can read.
    """
    models_dir = synth_dir(out_path)
    results = {"epoch": epoch}

    if exists(models_dir):
        if classification_mode:
            from ab.gpt.ClassificationEval import evaluate_epoch as cls_eval

            cls_result = cls_eval(models_dir)
            results[f"epoch_{epoch + 1}_accuracy"] = cls_result["accuracy"]
        elif trans_mode:
            try:
                run_eval(epoch_num=epoch, FT_MODE=True)
                print('[DEBUG] Release_memory.')
            except Exception as e:
                print(f"Error running evaluation main(): {e}", flush=True)
            print('Folder data reload will occur next epoch.')
        else:
            NNEval.main(
                nn_name_prefix=nn_name_prefix,
                nn_train_epochs=nn_train_epochs,
                only_epoch=epoch,
            )
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
        state.get("classification_mode", False),
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
    resume_trainer_checkpoint=None,
    use_backbone=False,
    sft_nn_prefixes=None,
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
            tokenizer,
            nn_prefixes=sft_nn_prefixes,
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
    model = lora_tuner.train(
        dataset,
        tokenizer,
        out_path / base_model_name,
        resume_from_checkpoint=resume_trainer_checkpoint,
        checkpoint_label="trainer",
    )

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
        state.get("trainer_resume_checkpoint"),
        state.get("use_backbone", False),
        state.get("sft_nn_prefixes"),
    )

    return {
        "model": model,
        "chat_bot": chat_bot,
        "current_epoch": epoch + 1,
        "next_action": "generate",
        "trainer_resume_checkpoint": None,
    }


# ============================================================
# MAIN: tune()
# ============================================================


def _resolve_tune_resume_trainer_checkpoint(initial_adapter_path) -> Optional[str]:
    # Classic mode and agent mode both consume the same one-shot trainer resume path.
    resume_spec = TrainingRuntime.resolve_resume_spec(
        trainer_env="NNGPT_TRAIN_RESUME_TRAINER_CHECKPOINT",
        initial_adapter_active=bool(initial_adapter_path),
        initial_adapter_label="llm_path/--peft",
    )
    if resume_spec.trainer_checkpoint is None:
        return None
    return str(resume_spec.trainer_checkpoint)

def tune(
    test_nn,
    nn_train_epochs,
    skip_epoch,
    llm_path,
    llm_tune_conf,
    nn_gen_conf,
    conf_keys,
    llm_conf,
    training_args,
    peft_config,
    max_prompts=None,
    save_llm_output=True,
    max_new_tokens=16 * 1024,
    nn_name_prefix=None,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    test_metric=None,
    onnx_run=False,
    trans_mode=False,
    prompt_batch=1,
    use_agents=False,
    use_predictor=False,
    use_unsloth=False,
    enable_merge=False,
    classification_mode=False,
    use_backbone=False,
    sft_nn_prefixes=None,
    num_cycles=None,
):
    if not isinstance(conf_keys, (list, tuple)):
        conf_keys = (conf_keys,)

    with open(conf_llm_dir / llm_conf) as f:
        config = json.load(f)
    assert isinstance(config, dict)

    token_from_file = config["token_from_file"]
    base_model_name = config["base_model_name"]
    merged_candidate = nngpt_upload / Path(base_model_name).name

    if merged_candidate.exists():
        print(f"[EVOLUTION] Using merged model: {merged_candidate}")
        base_model_name = str(merged_candidate)
    else:
        print(f"[EVOLUTION] Using base model from config: {base_model_name}")

    llm_tune_epochs = int(num_cycles) if num_cycles is not None else int(config["num_epochs"])
    use_deepspeed = config["use_deepspeed"]
    only_best_accuracy = config["only_best_accuracy"]
    context_length = config.get("context_length")
    unsloth_max_input_length = config.get("max_input_length", None)
    use_unsloth = config.get("use_unsloth", use_unsloth)
    unsloth_load_in_4bit = config.get("load_in_4bit", True)
    max_new_tokens = config.get("max_new_tokens", max_new_tokens)
    use_backbone = config.get("backbone", use_backbone)

    access_token = None
    if token_from_file:
        with open(ab_root_path / "token") as f:
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
        load_in_4bit=unsloth_load_in_4bit,
    )

    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    trainer_resume_checkpoint = _resolve_tune_resume_trainer_checkpoint(llm_path)

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
        use_unsloth=use_unsloth,
    )

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
        "sft_nn_prefixes": sft_nn_prefixes,
        "trainer_resume_checkpoint": trainer_resume_checkpoint,
        "enable_merge": enable_merge,
        "classification_mode": classification_mode,
    }

    shutil.rmtree(epoch_dir(), ignore_errors=True)

    if use_agents:
        from ab.gpt.agents.run_agent import run_agent_controller
        return run_agent_controller(state)

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

            _evaluate_epoch(epoch, out_path, nn_name_prefix, nn_train_epochs, trans_mode, classification_mode)

        print(f'[DEBUG]Perform finetune at epoch {epoch}.')
        model, chat_bot = _finetune_epoch(
            epoch, out_path, model, tokenizer, model_loader, lora_tuner,
            context_length, use_unsloth, unsloth_max_input_length,
            train_config_path, only_best_accuracy, max_prompts,
            max_new_tokens, base_model_name, trans_mode,
            temperature, top_k, top_p,
            trainer_resume_checkpoint,
            use_backbone=use_backbone,
            sft_nn_prefixes=sft_nn_prefixes,
        )
        trainer_resume_checkpoint = None
