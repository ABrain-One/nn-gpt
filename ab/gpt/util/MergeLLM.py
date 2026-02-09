from os import makedirs
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from ab.gpt.util.Const import base_llm, nngpt_model, epoch_dir, llm_tokenizer_out, nngpt_upload


def add_tokenizer(llm_path, tokenizer_path, full_llm_path, model_name):
    target_dir = full_llm_path / model_name
    shutil.rmtree(target_dir, ignore_errors=True)
    makedirs(target_dir, exist_ok=True)
    shutil.copytree(llm_path / model_name, target_dir, dirs_exist_ok=True)
    shutil.copytree(tokenizer_path / model_name, target_dir, dirs_exist_ok=True)

def merge(base_model_path, lora_path, output_path):
    # 1. Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # used one in fine-tuning
        device_map="auto")

    # 2. Connect LoRA to the Base Model
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16)

    # 3.  Merge
    merged_model = lora_model.merge_and_unload()

    # 4. Save
    merged_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    print("Model successfully saved to: ", output_path)


def merge_hp_llm():
    merge('/home/bose/thesis/nn-gpt/out/llm/ABrain/NNGPT-DeepSeek-Coder-1.3B-Instruct',
          '/home/bose/thesis/nn-gpt/nngpt/outputs/checkpoint-1188', '/home/bose/thesis/nn-gpt/out/llm/merged/NNGPT-DeepSeek-Coder-1.3B-Instruct')


def merge_nn_llm(tune_epoch):
    add_tokenizer(nngpt_model, llm_tokenizer_out, nngpt_upload, base_llm)
    merge(nngpt_upload / base_llm, epoch_dir(tune_epoch) / base_llm, nngpt_upload / base_llm)


if __name__ == "__main__":
    # merge_hp_llm()  # Uncomment code to merge weights of hyperparameter prediction LLM for Hugging Face publication
    merge_nn_llm(0)  # Uncomment code to merge neural network generation LLM weights for Hugging Face publication
