import argparse
import sys
from peft import LoraConfig
from transformers import TrainingArguments
from ab.gpt.util.Tune import tune
from ab.gpt.util.Const import nngpt_dir, conf_train_dir

def main():
    parser = argparse.ArgumentParser(description='Run Backbone Tuning.')
    parser.add_argument('--llm_conf', type=str, default='backbone_sft_config.json', help='LLM config file name')
    parser.add_argument('--test_nn', type=int, default=30, help='Number of NNs to generate')
    parser.add_argument('--num_train_epochs', type=int, default=5, help='Number of LLM fine-tuning epochs')
    parser.add_argument('--num_cycles', type=int, default=None, help='Number of generate/eval/SFT cycles; defaults to llm config num_epochs')
    parser.add_argument('--nn_train_epochs', type=int, default=1, help='Number of training epochs for generated NNs')
    parser.add_argument('--sft_nn_prefixes', type=str, default='rl-bb-test1', help='Comma-separated NN prefixes used as SFT training data')
    parser.add_argument('--gen_nn_prefix', type=str, default='rl-bb-test1', help='NN prefix for generated/evaluated models')
    
    args = parser.parse_args()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=str(nngpt_dir / 'outputs'),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=args.num_train_epochs,
        logging_steps=5,
        bf16=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True
    )

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
        # max_prompts=1000,
        use_backbone=True,
        sft_nn_prefixes=args.sft_nn_prefixes,
        num_cycles=args.num_cycles,
        temperature=0.8
    )

if __name__ == '__main__':
    main()
