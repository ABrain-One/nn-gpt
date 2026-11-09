"""Remote accuracy-prediction client (ABrain/Accuracy-Prediction, Qwen3-8B LoRA).

Adapted from nn-gpt/ab/gpt/util/acc_client.py (the professor's reference client) for the
denoise GA. Two deliberate deviations from that reference, both measured, not assumed:

1. INPUT IS FED IN dB, NOT RAW LEMUR ACCURACY. Validated against the 6 seeds that hold
   full 50-epoch curves (truth = their measured best), feeding the same three early epochs
   at three different scales:

       raw 0-1 (0.7391)   Spearman rho = -0.679   MAE 15.17   <- ANTI-correlated
       percent (73.91)    Spearman rho = +0.030   MAE 26.18   <- no signal
       dB      (35.48)    Spearman rho = +0.841   MAE  1.20   <- usable

   PSNR in dB lands in a numeric range the model can reason about; 0.7391 does not.
   Feeding raw accuracy makes the predictor actively worse than a coin flip on ranking.
   n=6, p~0.036 -- marginal. Treat as provisional until the 50-epoch sweep raises this
   to n=23. That is why the funnel is opt-in (--prof-predictor), not default-on.

2. best_epoch IS DISCARDED. It returns 48 or 49 for every model on every input scale
   (2 unique values across 6 models) and correlates NEGATIVELY with the true best epoch.
   It carries no information, so this client does not expose it.

Fails soft by design: any network/parse error returns None and the caller keeps its
measured value. A predictor outage must never corrupt a GA run.
"""
import json
import os
import re
import urllib.request

VLLM_URL = os.environ.get("PROF_PREDICTOR_URL",
                          "http://132.187.14.67:30031/v1/chat/completions")
MODEL = "accuracy"
MAX_EPOCHS = 50
TIMEOUT = float(os.environ.get("PROF_PREDICTOR_TIMEOUT", "120"))

SYSTEM_PROMPT = """You are a strict JSON generator.
You must output exactly ONE JSON object and nothing else.

The JSON must contain exactly the keys:
best_accuracy
best_epoch

Rules:
best_accuracy must be a float in [0,100] rounded to 2 decimals.
best_epoch must be a positive integer representing the absolute epoch where peak validation accuracy occurs.

Do not explain.
Do not add text.
Stop immediately after the closing brace }.
"""


def _user_msg(task, dataset, metric, nn_code, a1, a2, a3, max_epochs):
    return "\n".join([
        "INPUT", f"task: {task}", f"dataset: {dataset}", f"metric: {metric}", "",
        "TRAINING_BUDGET", f"max_epochs: {max_epochs}", "",
        "EARLY_TRAINING_SIGNAL",
        f"epoch_1_accuracy: {round(float(a1), 6)}",
        f"epoch_2_accuracy: {round(float(a2), 6)}",
        f"epoch_3_accuracy: {round(float(a3), 6)}", "",
        "NEURAL_NETWORK_CODE", "```python", (nn_code or "").strip(), "```", "",
        "Analyze the training dynamics, architecture complexity, and optimization "
        "hyperparameters to estimate the final training outcome.", "",
        "OUTPUT (JSON ONLY):",
    ])


def _parse(text):
    text = re.sub(r"<think>.*?</think>", "", (text or "").strip(), flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    if "```" in text:
        text = text.split("```", 1)[0].strip()
    i = text.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    return None
    return None


def predict_db(nn_code, db1, db2, db3, task="img-denoising", dataset="denoise",
               metric="psnr", max_epochs=MAX_EPOCHS):
    """Predict the 50-epoch best PSNR in dB from the first three epochs, also in dB.

    db1/db2/db3 MUST already be dB (LEMUR accuracy * 48) -- see the scale note above.
    Returns a float dB estimate, or None if the call fails (caller keeps its measured
    value). best_epoch is intentionally not returned."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": _user_msg(task, dataset, metric,
                                                           nn_code, db1, db2, db3,
                                                           max_epochs)}],
        "max_tokens": 64, "temperature": 0.0,
        "stop": ["}", "<|im_end|>", "<|endoftext|>"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        req = urllib.request.Request(
            VLLM_URL, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            gen = json.load(r)["choices"][0]["message"]["content"] or ""
    except Exception:
        return None
    if "{" in gen and "}" not in gen:      # vLLM strips the matched stop string
        gen += "}"
    d = _parse(gen)
    if not d or "best_accuracy" not in d:
        return None
    try:
        v = float(d["best_accuracy"])
    except (TypeError, ValueError):
        return None
    # Sanity band: the model is asked for [0,100] and we interpret the answer as dB.
    # Anything outside a plausible PSNR range means it ignored the scale -- discard
    # rather than feed a nonsense number into fitness.
    return v if 10.0 <= v <= 60.0 else None


def available():
    """Cheap reachability probe so a run can report predictor status up front."""
    try:
        base = VLLM_URL.rsplit("/v1/", 1)[0] + "/v1/models"
        with urllib.request.urlopen(base, timeout=10) as r:
            return any(m.get("id") == MODEL for m in json.load(r).get("data", []))
    except Exception:
        return False
