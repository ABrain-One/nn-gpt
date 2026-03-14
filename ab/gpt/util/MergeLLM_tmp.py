from os import makedirs
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from ab.gpt.util.Const import base_llm, nngpt_model, epoch_dir, llm_tokenizer_out, nngpt_upload, nngpt_dir
import json
from pathlib import Path


LINEAGE_FILE = nngpt_dir / "accepted_adapters.json"


def add_tokenizer(llm_path, tokenizer_path, full_llm_path, model_name):
    target_dir = full_llm_path / model_name
    print("\n[DEBUG add_tokenizer]")
    print("llm_path:", llm_path)
    print("tokenizer_path:", tokenizer_path)
    print("model_name:", model_name)
    print("llm source:", llm_path / model_name)
    print("tokenizer source:", tokenizer_path / model_name)
    print("target_dir:", target_dir)
    print("llm exists?", (llm_path / model_name).exists())
    print("tokenizer exists?", (tokenizer_path / model_name).exists())
    print()

    shutil.rmtree(target_dir, ignore_errors=True)
    makedirs(target_dir, exist_ok=True)

    shutil.copytree(llm_path / model_name, target_dir, dirs_exist_ok=True)
    shutil.copytree(tokenizer_path / model_name, target_dir, dirs_exist_ok=True)
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
    merge('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
          'finetuned_models/path', 'finetuned_models/merged_model_path')

def merge_nn_llm(tune_epoch):
    add_tokenizer(nngpt_model, llm_tokenizer_out, nngpt_upload, base_llm)
    merge(nngpt_upload / base_llm, epoch_dir(tune_epoch) / base_llm, nngpt_upload / base_llm)

def load_lineage():
    if not LINEAGE_FILE.exists():
        raise RuntimeError("accepted_adapters.json not found")

    with open(LINEAGE_FILE) as f:
        data = json.load(f)

    if "base_model" not in data or "adapters" not in data:
        raise RuntimeError("Invalid lineage file")

    return data


def rebuild_from_lineage():
    lineage = load_lineage()

    base_model_name = lineage["base_model"]
    accepted = lineage["adapters"]

    model_name = Path(base_model_name).name
    output_path = nngpt_upload / model_name

    print("REBUILDING MODEL FROM LINEAGE")
    print(f"Base model: {base_model_name}")
    print(f"Accepted adapters: {[a['epoch'] for a in accepted]}")


    # Load base model ONCE
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    for i, adapter_info in enumerate(accepted):

        # ---- Extract epoch safely ----
        if isinstance(adapter_info, dict):
            epoch = adapter_info["epoch"]
        elif isinstance(adapter_info, int):
            epoch = adapter_info
        else:
            raise RuntimeError(f"Invalid adapter format: {adapter_info}")

        # ---- Compute adapter directory deterministically ----
        adapter_path = nngpt_dir / "llm" / "epoch" / f"A{epoch}"

        if not adapter_path.exists():
            raise RuntimeError(f"Adapter directory missing: {adapter_path}")

        # ---- Locate adapter_config.json ----
        adapter_configs = [
            p for p in adapter_path.rglob("adapter_config.json")
            if "synth_nn" not in str(p)
        ]

        if not adapter_configs:
            raise RuntimeError(f"A{epoch} adapter_config.json not found in {adapter_path}")

        adapter_dir = adapter_configs[0].parent

        print(f"[{i + 1}/{len(accepted)}] Merging A{epoch}")

        # ---- Load and merge adapter into current model ----
        model = PeftModel.from_pretrained(
            model,
            str(adapter_dir),
            torch_dtype=torch.float16
        )

        model = model.merge_and_unload()

    print("\nSaving merged model...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    print(f"\nRebuild complete → {output_path}\n")



if __name__ == "__main__":
    # merge_hp_llm()  # Uncomment code to merge weights of hyperparameter prediction LLM for Hugging Face publication
    #merge_nn_llm(0)  # Uncomment code to merge neural network generation LLM weights for Hugging Face publication
    rebuild_from_lineage()


