import torch
from transformers import (
    BitsAndBytesConfig
)

# Canonical QLoRA recipe: 4-bit NF4 with bf16 compute and double quantization
quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 quantization type
    bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 compute dtype
    bnb_4bit_use_double_quant=True      # Double quantization for better quality
)

quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)


def tokenize(prompt, tokenizer):
    """
    Tokenizes a string
    """
    return tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=False,
        return_tensors=None,
    )
