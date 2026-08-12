import json
import re
import httpx

from ab.gpt.util.method.zero_cost_proxies import (
    compute_proxies, normalize_proxy_value, DEFAULT_PROXY_NORM_STATS,
)

# --- Remote vLLM server ---
VLLM_URL = "http://132.187.14.67:30031/v1/chat/completions"
MODEL = "accuracy"  # --lora-modules accuracy=ABrain/Accuracy-Prediction

PREDICTOR_DEFAULT_MAX_EPOCHS = 50
PREDICTOR_MAX_NEW_TOKENS = 64

# Must match the served model's training config (no source code, 2 early epochs,
# these 5 normalized zero-cost proxies). See AccPredictor.ACTIVE_PROXY_NAMES.
ACTIVE_PROXY_NAMES = ("synflow", "nwot", "grad_norm", "log_params", "depth")

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


def _proxy_lines(nn_code: str, dataset: str, prm: dict | None) -> list[str]:
    """Compute the active zero-cost proxies from nn_code, normalize them with the
    training stats, and format them as the ZERO_COST_PROXIES block. Returns [] if
    none compute (block is then omitted, matching training)."""
    if not (nn_code or "").strip():
        return []
    proxies = compute_proxies(nn_code, dataset or "", prm=prm)
    lines = []
    for name in ACTIVE_PROXY_NAMES:
        raw = proxies.get(name)
        if raw is not None:
            value = normalize_proxy_value(name, raw, DEFAULT_PROXY_NORM_STATS)
            if value is not None:
                lines.append(f"{name}: {round(float(value), 4)}")
    if not lines:
        return []
    return ["ZERO_COST_PROXIES (training-free, normalized architecture signals)", *lines, ""]


def _build_user_message(
    task: str,
    dataset: str,
    metric: str,
    nn_code: str,
    epoch_1_accuracy: float,
    epoch_2_accuracy: float,
    prm: dict | None = None,
) -> str:
    lines = [
        "INPUT",
        f"task: {task}",
        f"dataset: {dataset}",
        f"metric: {metric}",
        "",
        "TRAINING_BUDGET",
        f"max_epochs: {PREDICTOR_DEFAULT_MAX_EPOCHS}",
        "",
        "EARLY_TRAINING_SIGNAL",
        f"epoch_1_accuracy: {round(float(epoch_1_accuracy), 6)}",
        f"epoch_2_accuracy: {round(float(epoch_2_accuracy), 6)}",
        "",
        *_proxy_lines(nn_code, dataset, prm),
        "Analyze the training dynamics and optimization hyperparameters to estimate the final training outcome.",
        "",
        "Important signals to consider:",
        "- Early learning progress (epoch accuracies)",
        "- Saturation of improvement across epochs",
        "- Optimization scale (learning rate, batch size, effective_lr)",
        "- Zero-cost proxy scores, if present (training-free estimates of architecture quality that generalize across architecture families better than early accuracy alone)",
        "- Outcomes of similar architectures, if present (real precedents from structurally similar networks)",
        "",
        "Using these signals, estimate the final best validation accuracy and the epoch where it occurs.",
        "",
        "The value best_epoch represents the training epoch where the highest validation accuracy is reached.",
        "",
        "Constraints:",
        "- best_epoch must be an integer between 1 and max_epochs.",
        "",
        "OUTPUT (JSON ONLY):",
    ]
    return "\n".join(lines)


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
) -> tuple[float, int]:
    """Predict best_accuracy and best_epoch via the remote vLLM server. Builds the
    prompt the served model expects (no source code, 2 early epochs, 5 zero-cost
    proxies computed from nn_code). prm is only used to compute the proxies."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_message(
                task, dataset, metric, nn_code,
                epoch_1_accuracy, epoch_2_accuracy, prm,
            ),
        },
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": PREDICTOR_MAX_NEW_TOKENS,
        "temperature": 0.0,                        # do_sample=False (greedy)
        "stop": ["}", "<|im_end|>", "<|endoftext|>"],
        "add_special_tokens": False,               # match training tokenization
        "chat_template_kwargs": {"enable_thinking": False},
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(VLLM_URL, json=payload)
        resp.raise_for_status()
        generated = resp.json()["choices"][0]["message"]["content"] or ""

    # vLLM strips the matched stop string ("}"), so restore it before parsing.
    if "{" in generated and "}" not in generated:
        generated = generated + "}"

    parsed = _parse_prediction_json(generated)
    if not parsed:
        raise ValueError(f"Could not parse prediction JSON from model output:\n{generated!r}")

    if "best_accuracy" not in parsed or "best_epoch" not in parsed:
        raise ValueError(f"Prediction JSON missing required keys: {parsed!r}")

    best_accuracy = float(parsed["best_accuracy"])
    best_epoch = int(parsed["best_epoch"])
    if best_epoch < 1:
        raise ValueError(f"Invalid best_epoch (< 1): {best_epoch}")

    return best_accuracy, best_epoch


if __name__ == "__main__":
    nn_code = """import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
"""

    # early accuracies on the 0-1 scale, matching the model's training data
    best_accuracy, best_epoch = predict_best_accuracy(
        task="img-classification",
        dataset="cifar-10",
        metric="acc",
        nn_code=nn_code,
        epoch_1_accuracy=0.612,
        epoch_2_accuracy=0.724,
    )
    print(f"Predicted best accuracy: {best_accuracy}")
    print(f"Predicted best epoch:    {best_epoch}")
