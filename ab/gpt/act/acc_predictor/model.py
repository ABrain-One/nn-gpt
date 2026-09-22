"""Model loading: base Qwen3 checkpoint (with fallbacks) and the cached HF predictor."""
from __future__ import annotations

from unsloth import FastLanguageModel

from . import config
from .config import *  # noqa: F401,F403



_predictor_model = None
_predictor_tokenizer = None


def _get_predictor_model():
    global _predictor_model, _predictor_tokenizer
    if _predictor_model is None or _predictor_tokenizer is None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=PREDICTOR_HF_REPO,
            max_seq_length=PREDICTOR_MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        FastLanguageModel.for_inference(model)
        _predictor_model = model
        _predictor_tokenizer = tokenizer
    return _predictor_model, _predictor_tokenizer


def _load_model(model_name: str, max_seq_len: int):
    candidates = [model_name, *(m for m in config.MODEL_FALLBACKS if m != model_name)]
    last_error: Exception | None = None

    for model_id in candidates:
        try:
            return FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_len,
                dtype=None,
                load_in_4bit=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not load any Qwen3-8B checkpoint. Tried: {candidates}") from last_error
