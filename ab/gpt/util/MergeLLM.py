from os import makedirs
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from ab.gpt.util.Const import base_llm, nngpt_model, epoch_dir, llm_tokenizer_out, nngpt_upload, nngpt_dir
import json
from pathlib import Path


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
    merge('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
          'finetuned_models/path', 'finetuned_models/merged_model_path')

def merge_nn_llm(tune_epoch):
    add_tokenizer(nngpt_model, llm_tokenizer_out, nngpt_upload, base_llm)
    merge(nngpt_upload / base_llm, epoch_dir(tune_epoch) / base_llm, nngpt_upload / base_llm)


def merge_from_adapter(checkpoint: int | None = None):

    if checkpoint is None:
        outputs_dir = nngpt_dir / "outputs"

        checkpoints = sorted(
            [d for d in outputs_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda x: int(x.name.split("-")[1])
        )

        if not checkpoints:
            raise RuntimeError("No checkpoint-* directories found.")

        checkpoint = int(checkpoints[-1].name.split("-")[1])

        print(f"[MERGE] Auto-detected latest checkpoint: {checkpoint}")

    adapter_dir = nngpt_dir / "outputs" / f"checkpoint-{checkpoint}"

    with open(adapter_dir / "adapter_config.json") as f:
        base_model = json.load(f)["base_model_name_or_path"]

    base_model_path = Path(base_model)
    model_name = base_model_path.name
    parent_dir = base_model_path.parent

    output_path = nngpt_upload / model_name

    print(f"[MERGE] Checkpoint: {adapter_dir}")
    print(f"[BASE ] {base_model_path}")
    print(f"[SAVE ] {output_path}")

    add_tokenizer(parent_dir, llm_tokenizer_out / parent_dir.name, nngpt_upload, model_name)

    merge(output_path, adapter_dir, output_path)

    print("✓ Merge complete.")




if __name__ == "__main__":
    # merge_hp_llm()  # Uncomment code to merge weights of hyperparameter prediction LLM for Hugging Face publication
    #merge_nn_llm(0)  # Uncomment code to merge neural network generation LLM weights for Hugging Face publication
    merge_from_adapter()
