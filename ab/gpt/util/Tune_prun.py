import sys
import os

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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
from ab.gpt.util.Util import exists, extract_delta, extract_code, extract_hyperparam, extract_transform
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from ab.gpt.util.DeltaUtil import apply_delta, validate_delta, repair_code
from ab.gpt.util.Const import nngpt_upload
from ab.gpt.brute.trans.TransformEval import run_eval
from ab.gpt.util.prompt.TransformGenPrompt import TransformGenPrompt, load_data_from_folders

AgentState = dict
ds_conf = conf_dir / 'DeepSpeed.json'
TRANSFORM_OUT_DIR = trans_dir / 'dataset_epoch1'
TRANSFORM_RES_DIR = trans_dir / 'result_epoch1'
_MAX_DELTA_RETRIES = 2


def nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens,
           save_llm_output, nn_name_prefix, unsloth_max_input_length, prompt_batch, use_backbone=False):
    print("Preparing prompts for generation...")
    prompts = []
    for key in conf_keys:
        prompt = ""
        key_config = prompt_dict[key]
        for pr in key_config["prompt"]:
            prompt += pr + "\n"
        data = lemur.data(only_best_accuracy=True, task=key_config["task"]).groupby(by="nn").sample(n=1)[:test_nn]
        addon_task = key_config.get("addon_task")
        addon_data = lemur.data(only_best_accuracy=True, task=addon_task) if addon_task else None
        for _, row in data.iterrows():
            para_dict = {}
            for it in key_config["input_list"]:
                para_dict[it["para"]] = row[it["value"]]
            if addon_data is not None and not addon_data.empty:
                available_addon = addon_data.loc[addon_data.nn != row["nn"]]
                if not available_addon.empty:
                    addon_row = available_addon.sample(n=1).iloc[0]
                    if key_config.get("addon_list"):
                        for it in key_config["addon_list"]:
                            para_dict[it["para"]] = addon_row[it["value"]]
            prompts.append((prompt.format(**para_dict), row))

    models_dir = synth_dir(out_path)
    pending = []
    for idx, prompt_data in tqdm(enumerate(prompts)):
        prompt_text, origdf = prompt_data
        if unsloth_max_input_length:
            in_text = [{"role": "user", "content": prompt_text}]
            output = chat_bot.tokenizer.apply_chat_template(in_text, add_generation_prompt=True)
            if len(output) > unsloth_max_input_length:
                continue
        pending.append((idx, prompt_text, origdf))

    for start in range(0, len(pending), prompt_batch):
        batch = pending[start:start + prompt_batch]
        batch_prompts = [item[1] for item in batch]
        batch_outputs = [chat_bot.chat(p, engineer_prompt=False, max_new_tokens=max_new_tokens) for p in batch_prompts]
        for (idx, prompt_text, origdf), output in zip(batch, batch_outputs):
            model_dir = models_dir / f"B{idx}"
            code, hp, tr, full_out = output
            makedirs(model_dir, exist_ok=True)
            if save_llm_output:
                create_file(model_dir, new_out_file, full_out)
            if hp and hp.strip():
                try:
                    hp = json.loads(hp.replace("'", '"'))
                    with open(model_dir / hp_file, 'w+') as f:
                        json.dump(hp, f)
                except:
                    pass
            if tr and tr.strip():
                create_file(model_dir, transformer_file, tr)
            if code and code.strip():
                create_file(model_dir, new_nn_file, code)
            create_file(model_dir, new_out_file, full_out)
            df_file = model_dir / 'dataframe.df'
            if origdf is None:
                if isfile(df_file):
                    os.remove(df_file)
            else:
                create_file(model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
                origdf.to_pickle(df_file)

    release_memory()


def _finetune_epoch(epoch, out_path, model, tokenizer, model_loader, lora_tuner, context_length, use_unsloth,
                    unsloth_max_input_length, train_config_path, only_best_accuracy, max_prompts, max_new_tokens,
                    base_model_name, trans_mode, temperature=1.0, top_k=50, top_p=0.9, use_backbone=False):
    length = (unsloth_max_input_length if (use_unsloth and unsloth_max_input_length)
              else context_length if context_length else model_loader.get_max_length())
    data_processor = NNGenPrompt(length, tokenizer, train_config_path)
    dataset = data_processor.get_dataset(only_best_accuracy, max_prompts=max_prompts, max_new_tokens=max_new_tokens)
    print("Dataset length:", len(dataset))
    model.train()
    model = lora_tuner.train(dataset, tokenizer, out_path / base_model_name)
    del dataset
    release_memory()
    chat_bot = ChatBot(model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)
    return model, chat_bot


def tune(test_nn, nn_train_epochs, skip_epoch, llm_path, llm_tune_conf, nn_gen_conf, conf_keys, llm_conf,
         training_args, peft_config, max_prompts=None, save_llm_output=True, max_new_tokens=16 * 1024,
         nn_name_prefix=None, temperature=1.0, top_k=50, top_p=0.9, test_metric=None, onnx_run=False,
         trans_mode=False, prompt_batch=1, use_agents=False, use_predictor=False, use_backbone=False, enable_merge=False):

    if not isinstance(conf_keys, (list, tuple)):
        conf_keys = (conf_keys,)

    with open(conf_llm_dir / llm_conf) as f:
        config = json.load(f)

    base_model_name = config["base_model_name"]
    llm_tune_epochs = int(config["num_epochs"])
    use_deepspeed = config["use_deepspeed"]
    only_best_accuracy = config["only_best_accuracy"]
    context_length = config.get("context_length")
    unsloth_max_input_length = config.get("max_input_length", None)
    use_unsloth = config.get("use_unsloth", False)
    unsloth_load_in_4bit = config.get("load_in_4bit", True)

    access_token = None
    if config.get("token_from_file", False):
        with open(ab_root_path / "token") as f:
            access_token = f.readline()

    train_config_path = conf_train_dir / llm_tune_conf
    with open(conf_test_dir / nn_gen_conf) as prompt_file:
        prompt_dict = json.load(prompt_file)

    from ab.gpt.util.LLM import LLM
    model_loader = LLM(base_model_name, quantization_config_4bit, access_token=access_token,
                       use_deepspeed=use_deepspeed, context_length=context_length, training_args=training_args,
                       use_unsloth=use_unsloth, load_in_4bit=unsloth_load_in_4bit)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()

    if llm_path:
        model = PeftModel.from_pretrained(model, llm_path, is_trainable=True)
        model = model.merge_and_unload()

    if use_deepspeed:
        deepspeed.initialize(model=model, config_params=ds_conf)

    lora_tuner = LoRA(model, tokenizer, training_args=training_args, access_token=access_token,
                      peft_config=peft_config, use_unsloth=use_unsloth)
    chat_bot = ChatBot(model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)

    shutil.rmtree(epoch_dir(), ignore_errors=True)

    for epoch in range(llm_tune_epochs):
        print(f'[INFO] Start Epoch {epoch}')
        out_path = epoch_dir(epoch)
       # if epoch >= skip_epoch:
           # print(f'[INFO] Generation at epoch {epoch}')
           # nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn,
                   #max_new_tokens, save_llm_output, nn_name_prefix, unsloth_max_input_length, prompt_batch)
            # Evaluation would go here
        print(f'[DEBUG] Finetune at epoch {epoch}')
        model, chat_bot = _finetune_epoch(epoch, out_path, model, tokenizer, model_loader, lora_tuner,
                                           context_length, use_unsloth, unsloth_max_input_length, train_config_path,
                                           only_best_accuracy, max_prompts, max_new_tokens, base_model_name,
                                           trans_mode, temperature, top_k, top_p)