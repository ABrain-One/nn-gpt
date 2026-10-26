import gc
import os

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

def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def unwrap_tokenizer(tokenizer_or_processor):
    """Return the real tokenizer behind a multimodal processor.

    VL checkpoints (Qwen3.5-9B, Qwen3.8-27B, ...) load as a *processor*
    (Qwen3VLProcessor), not a tokenizer, and a processor's __call__ takes
    IMAGES as its first positional argument. So `processor(texts, ...)` hands
    the prompts to the image loader and dies with
    "RuntimeError: decode_image ... Unsupported image file". A processor also
    lacks the tokenizer-only API this project relies on: .vocab_size, __len__,
    .convert_tokens_to_ids, .padding_side, .pad_token_id.

    Plain tokenizers (Qwen2.5, DeepSeek, Llama, ...) have no .tokenizer
    attribute and are returned unchanged, so this is a no-op on every
    text-only 7-14B model.
    """
    inner = getattr(tokenizer_or_processor, "tokenizer", None)
    if inner is not None and hasattr(inner, "convert_tokens_to_ids"):
        return inner
    return tokenizer_or_processor


#: transformers marks "no real limit" with a sentinel around 1e30. Anything
#: above this is a sentinel, not a context length.
_ABSURD_MODEL_MAX_LENGTH = 10 ** 7


def prepare_text_tokenizer(tokenizer_or_processor, context_length=None,
                           log_prefix="[tokenizer]"):
    """Unwrap a processor and make the result usable for text-only work.

    Three steps, all idempotent and all no-ops for a plain tokenizer:

      * unwrap the processor (see unwrap_tokenizer);
      * copy chat_template down from the processor. VL processors hold the real
        template while their inner tokenizer often has none, and without this
        the pipeline silently falls back to the generic User/Assistant template
        and fine-tunes on a prompt format the model has never seen - which is
        worse than a crash, because nothing reports it;
      * fill pad_token from eos_token, which generate() and the data collator
        both require.

    `context_length`, when given, also replaces a sentinel model_max_length so
    that generate_and_tokenize_prompt() below truncates against a real bound
    instead of 1e30.
    """
    tok = unwrap_tokenizer(tokenizer_or_processor)
    if tok is not tokenizer_or_processor:
        print(f"{log_prefix} {type(tokenizer_or_processor).__name__} detected; "
              f"using its inner {type(tok).__name__} for all text operations.",
              flush=True)
        if not getattr(tok, "chat_template", None):
            inherited = getattr(tokenizer_or_processor, "chat_template", None)
            if inherited:
                tok.chat_template = inherited
                print(f"{log_prefix} chat_template taken from the processor.",
                      flush=True)
    if getattr(tok, "pad_token_id", None) is None:
        eos = getattr(tok, "eos_token", None)
        if eos is not None:
            tok.pad_token = eos
            print(f"{log_prefix} pad_token was unset; using eos_token "
                  f"({eos!r}).", flush=True)
    if context_length:
        current = getattr(tok, "model_max_length", None)
        if not isinstance(current, int) or current > _ABSURD_MODEL_MAX_LENGTH:
            tok.model_max_length = int(context_length)
            print(f"{log_prefix} model_max_length was {current!r}; set to "
                  f"{int(context_length)}.", flush=True)
    return tok


# Quantizing lm_head is OFF by default because it costs some logit precision.
# Turn it on with NNGPT_QUANTIZE_LM_HEAD=1 when VRAM is the binding constraint:
# bitsandbytes skips lm_head by default (modules_to_not_convert), and on a
# large-vocabulary model that single bf16 table is multiple GB. Passing an empty
# skip list converts it like any other Linear.
# NOTE: this is a no-op when config.tie_word_embeddings is True, because lm_head
# then shares storage with embed_tokens and is not a separate Linear to convert.
def quantization_config_4bit_for(quantize_lm_head: bool = None) -> BitsAndBytesConfig:
    if quantize_lm_head is None:
        quantize_lm_head = _env_flag("NNGPT_QUANTIZE_LM_HEAD")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=[] if quantize_lm_head else None,
    )


def drop_vision_tower(model, log_prefix="[vram]"):
    """Free the vision encoder of a VL checkpoint used for text only.

    Qwen3.8-27B loads as Qwen3_5ForConditionalGeneration: the ViT is resident
    even though this pipeline never passes pixel_values. The language path only
    touches it when pixel_values/pixel_values_videos are given, so dropping it is
    safe for text-only generation and fine-tuning. Idempotent and never fatal.
    """
    freed_params = 0
    for owner in (getattr(model, "model", None), model):
        if owner is None:
            continue
        for attr in ("visual", "vision_tower"):
            tower = getattr(owner, attr, None)
            if tower is None or not hasattr(tower, "parameters"):
                continue
            try:
                freed_params += sum(p.numel() for p in tower.parameters())
                setattr(owner, attr, None)
            except Exception as e:
                print(f"{log_prefix} could not drop {attr}: "
                      f"{type(e).__name__}: {e}", flush=True)
    if freed_params:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{log_prefix} dropped vision tower "
              f"({freed_params / 1e9:.2f}B params) - text-only pipeline.",
              flush=True)
    return freed_params


def vram_report(model, top=8, log_prefix="[vram]"):
    """Print the biggest parameter groups, so sizing is measured not guessed."""
    try:
        sizes = {}
        for n, p in model.named_parameters():
            parts = n.split(".")
            key = parts[1] if parts[0] == "model" and len(parts) > 1 else parts[0]
            sizes[key] = sizes.get(key, 0) + p.numel() * p.element_size()
        total = sum(sizes.values())
        print(f"{log_prefix} parameter memory by group "
              f"(total {total / 2**30:.2f} GiB):", flush=True)
        for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:top]:
            print(f"{log_prefix}   {k:24s} {v / 2**30:6.2f} GiB", flush=True)
    except Exception as e:
        print(f"{log_prefix} report failed ({type(e).__name__}: {e})", flush=True)


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