import argparse
import inspect
import sys
import torch
from peft import LoraConfig
from trl import SFTConfig
from ab.gpt.util.Tune import tune
from ab.gpt.util.Const import nngpt_dir, conf_train_dir


def _build_sft_config(args):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    kwargs = {
        "output_dir": str(nngpt_dir / 'outputs'),
        "per_device_train_batch_size": args.sft_batch_size,
        "gradient_accumulation_steps": args.sft_gradient_accumulation,
        "learning_rate": 1e-5,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": 5,
        "bf16": use_bf16,
        "fp16": torch.cuda.is_available() and not use_bf16,
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": False,
        "max_length": args.sft_max_length,
        "gradient_checkpointing": True,
    }
    signature_parameters = inspect.signature(SFTConfig.__init__).parameters
    if "packing_strategy" in signature_parameters:
        kwargs["packing_strategy"] = "wrapped"
    if "padding_free" in signature_parameters:
        kwargs["padding_free"] = False
    if "gradient_checkpointing_kwargs" in signature_parameters:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return SFTConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser(description='Run Backbone Tuning.')
    parser.add_argument('--llm_conf', type=str, default='backbone_sft_config.json', help='LLM config file name')
    parser.add_argument('--test_nn', type=int, default=30, help='Number of NNs to generate')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of LLM fine-tuning epochs')
    parser.add_argument('--num_cycles', type=int, default=None, help='Number of generate/eval/SFT cycles; defaults to llm config num_epochs')
    parser.add_argument('--nn_train_epochs', type=int, default=1, help='Number of training epochs for generated NNs')
    parser.add_argument('--sft_max_length', type=int, default=4096, help='Maximum SFT sequence length')
    parser.add_argument('--sft_batch_size', type=int, default=1, help='Per-device SFT batch size')
    parser.add_argument('--sft_gradient_accumulation', type=int, default=8, help='SFT gradient accumulation steps')
    parser.add_argument('--sft_dataset_limit', type=int, default=None, help='Maximum number of SFT training samples; unset uses all available rows')
    parser.add_argument('--epoch_root', type=str, default=None, help='Output root for A* cycle directories')
    parser.add_argument('--sft_dataset', type=str, default='cifar-10', help='Dataset used for SFT seed/query data')
    parser.add_argument(
        '--sft_nn_prefixes',
        type=lambda raw: tuple(item.strip() for item in raw.split(',') if item.strip()),
        default=('rl-bb-test1',),
        help='Comma-separated NN prefixes used as SFT training data',
    )
    parser.add_argument('--gen_nn_prefix', type=str, default='rl-bb-test1', help='NN prefix for generated/evaluated models')
    
    args = parser.parse_args()

    training_args = _build_sft_config(args)

    # LoRA Config
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Call tune
    # We use None for prompt configs as SFTGenPrompt handles its own data fetching
    tune(
        test_nn=args.test_nn,
        nn_train_epochs=args.nn_train_epochs,
        nn_name_prefix=args.gen_nn_prefix,
        skip_epoch=0,   
        llm_path=None,
        llm_tune_conf='backbone_prompt.json',
        nn_gen_conf=str(conf_train_dir / 'backbone_prompt.json'),
        conf_keys=['backbone_fractal'],
        llm_conf=args.llm_conf,
        training_args=training_args,
        peft_config=peft_config,
        max_prompts=args.sft_dataset_limit,
        use_backbone=True,
        only_best_accuracy=True,
        context_length=4096,
        sft_nn_prefixes=args.sft_nn_prefixes,
        sft_dataset=args.sft_dataset,
        num_cycles=args.num_cycles,
        epoch_root=args.epoch_root,
        temperature=0.8
    )

if __name__ == '__main__':
    main()
