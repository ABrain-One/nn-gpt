"""
Delta-based generation for Blip2Fast (OPT-2.7B) Image Captioning.

Generates improved variants of Blip2Fast.py using unified diffs.
The LLM receives the full Blip2Fast baseline and produces a <delta> patch
that modifies only safe zones such as decoding parameters, cleanup logic,
or projection-side hyperparameters.

Do NOT confuse this with the old CrossModalBridge skeleton pipeline.
This script uses: NN_gen_blip2fastopt.json

Usage:
    python -m ab.gpt.NNAlter_7B_delta_captioning --epochs 8
    python -m ab.gpt.NNAlter_7B_delta_captioning --epochs 1 --prefix Blip2FastOpt
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import transformers

transformers.logging.set_verbosity_error()

import torch

# Dynamically inject generation parameters
original_generate = transformers.GenerationMixin.generate

def patched_generate(self, *args, **kwargs):
    if 'repetition_penalty' not in kwargs:
        kwargs['repetition_penalty'] = 1.1
    if 'pad_token_id' not in kwargs and hasattr(self.config, 'eos_token_id'):
        kwargs['pad_token_id'] = self.config.eos_token_id
    if 'attention_mask' not in kwargs:
        if args and isinstance(args[0], torch.Tensor):
            kwargs['attention_mask'] = torch.ones_like(args[0])
        elif 'inputs' in kwargs and isinstance(kwargs['inputs'], torch.Tensor):
            kwargs['attention_mask'] = torch.ones_like(kwargs['inputs'])
    return original_generate(self, *args, **kwargs)

transformers.GenerationMixin.generate = patched_generate

from ab.gpt.util.AlterNN_DeltaOpt import alter_delta


def main():
    """
    Main entry point for delta-based neural network generation for Blip2Fast.

    Uses alter_delta() which:
    1. Loads NN_gen_blip2fastopt.json
    2. Fetches Blip2Fast baseline code
    3. Sends it to the LLM
    4. Receives <hp> + <delta>
    5. Applies the delta directly to the baseline

    No captioning skeleton re-assembly is used here.
    """
    parser = argparse.ArgumentParser(
        description="Generate improved Blip2Fast captioning neural networks using delta-based approach."
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=8,
        help="Maximum number of generation epochs.",
    )

    parser.add_argument(
        "-n",
        "--num-supporting-models",
        type=int,
        default=1,
        help="Number of supporting models to fetch from database for more ideas.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="Blip2FastOpt",
        help="Prefix of the baseline models to use.",
    )

    args = parser.parse_args()

    from ab.gpt.util.Const import conf_test_dir, epoch_dir
    import json
    import glob
    from pathlib import Path
    from ab.gpt.util.Tune import nn_gen
    from ab.gpt.util.LLM import LLM
    from ab.gpt.util.Chatbot import ChatBot
    from ab.gpt.util.LLMUtil import quantization_config_4bit

    prompt_conf_name = "NN_gen_blip2fast.json"
    with open(conf_test_dir / prompt_conf_name) as prompt_file:
        prompt_dict = json.load(prompt_file)
    conf_keys = list(prompt_dict.keys())

    print("[NNGPT] Pre-warming LLM Loader framework...")
    model_loader = LLM(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        quantization_config_4bit,
        use_deepspeed=False,
        load_in_4bit=True
    )

    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    chat_bot = ChatBot(model, tokenizer, temperature=0.7, top_k=50, top_p=0.9)

    # Automatically detect the next epoch directory
    base_epoch_dir = epoch_dir()
    base_epoch_dir.mkdir(parents=True, exist_ok=True)
    existing_epochs = glob.glob(str(base_epoch_dir / "Epoch_*"))
    max_epoch = -1
    for d in existing_epochs:
        try:
            num = int(Path(d).name.split("_")[1])
            max_epoch = max(max_epoch, num)
        except ValueError:
            pass
    start_epoch = max_epoch + 1 if max_epoch >= 0 else 1

    for i in range(args.epochs):
        current_cycle = start_epoch + i
        print(f"\n🚀 [CYCLE START] Launching execution loop for cycle index: {current_cycle}")
        out_path = epoch_dir(current_cycle)

        nn_gen(
            epoch=current_cycle,
            out_path=out_path,
            chat_bot=chat_bot,
            conf_keys=conf_keys,
            nn_train_epochs=1,
            prompt_dict=prompt_dict,
            test_nn=1, # Generate 1 model per cycle exactly like before
            max_new_tokens=4096,
            save_llm_output=True,
            nn_name_prefix=args.prefix,
            unsloth_max_input_length=None,
            prompt_batch=1,
            use_backbone=False
        )

    print("\n[SUCCESS] Custom CrossModalBridge generation sequence completely finalized.")


if __name__ == "__main__":
    main()
