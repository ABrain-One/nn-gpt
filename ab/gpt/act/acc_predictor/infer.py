"""Single-record inference against the published ABrain/Accuracy-Prediction checkpoint."""
from __future__ import annotations

import json
import re

import torch

from . import config
from .config import *  # noqa: F401,F403
from .common import _apply_chat_template_for_inference
from .features import _build_user_message
from .model import _get_predictor_model
from .dataset import _get_predictor_proxy_norm_stats
from ab.gpt.util.method.zero_cost_proxies import normalize_proxy_value, compute_proxies




def _parse_prediction_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    if "```" in text:
        text = text.split("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def predict_best_accuracy(
    task: str,
    dataset: str,
    metric: str,
    nn_code: str,
    epoch_1_accuracy: float,
    epoch_2_accuracy: float,
    prm: dict | None = None,
layer_stats: str | None = None,
) -> tuple[float, int]:
    """Predict best_accuracy and best_epoch using ABrain/Accuracy-Prediction.
    If proxies are enabled and nn_code is given, they are computed from the code
    and added to the prompt; prm is only used for that computation."""
    record = {
        "task": task,
        "dataset": dataset,
        "metric": metric,
        "total_epochs": PREDICTOR_DEFAULT_MAX_EPOCHS,
        "accuracy_epoch_1": epoch_1_accuracy,
        "accuracy_epoch_2": epoch_2_accuracy,
    }
    if layer_stats:
        record["layer_stats"] = layer_stats
    _attach_computed_proxies(record, nn_code, prm)
    user_content = _build_user_message(record, nn_code)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    model, tokenizer = _get_predictor_model()
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    prompt_text = _apply_chat_template_for_inference(text_tokenizer, messages)
    input_ids = text_tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(input_ids) > PREDICTOR_MAX_SEQ_LEN:
        input_ids = input_ids[:PREDICTOR_MAX_SEQ_LEN]

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids_t)
    stop_token_ids = []
    for tok_str in ("}", "<|im_end|>", "<|endoftext|>"):
        ids = text_tokenizer.encode(tok_str, add_special_tokens=False)
        if ids:
            stop_token_ids.append(ids[-1])
    if text_tokenizer.eos_token_id is not None:
        stop_token_ids.append(text_tokenizer.eos_token_id)
    stop_token_ids = list(set(stop_token_ids))

    with torch.no_grad():
        outputs = model.generate(
            **{"input_ids": input_ids_t, "attention_mask": attention_mask},
            max_new_tokens=PREDICTOR_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=text_tokenizer.pad_token_id or text_tokenizer.eos_token_id,
            eos_token_id=stop_token_ids or None,
            use_cache=False,
        )

    generated = text_tokenizer.decode(
        outputs[0][input_ids_t.shape[1] :],
        skip_special_tokens=True,
    )
    parsed = _parse_prediction_json(generated)

    fallback_acc = max(float(epoch_1_accuracy), float(epoch_2_accuracy))
    fallback_epoch = 2 if float(epoch_2_accuracy) >= float(epoch_1_accuracy) else 1

    if not parsed:
        return fallback_acc, fallback_epoch

    try:
        best_accuracy = float(parsed.get("best_accuracy", fallback_acc))
    except (TypeError, ValueError):
        best_accuracy = fallback_acc

    try:
        best_epoch = int(parsed.get("best_epoch", fallback_epoch))
        if best_epoch < 1:
            best_epoch = fallback_epoch
    except (TypeError, ValueError):
        best_epoch = fallback_epoch

    return best_accuracy, best_epoch


def _attach_computed_proxies(record: dict, nn_code: str, prm: dict | None = None) -> None:
    """Compute the active proxies from nn_code, normalize them, and store them as
    proxy_<name> keys on record. Best-effort: proxies that fail to compute are
    skipped."""
    if not config.USE_ZERO_COST_PROXIES or not (nn_code or "").strip():
        return
    norm_stats = _get_predictor_proxy_norm_stats()
    proxies = compute_proxies(nn_code, record.get("dataset", "") or "", prm=prm)
    for name in config.ACTIVE_PROXY_NAMES:
        raw_value = proxies.get(name)
        if raw_value is not None:
            record[f"proxy_{name}"] = normalize_proxy_value(name, raw_value, norm_stats)
