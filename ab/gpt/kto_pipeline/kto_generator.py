#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Read-only reuse of existing repo helpers (not modified).
from ab.gpt.markov.code_extractor import CodeExtractor, validate_code, check_net_class
from ab.gpt.util.Const import conf_test_dir, new_nn_file
from ab.gpt.util.LLMUtil import quantization_config_4bit


def _salvage_fenceless_code(raw: str) -> Optional[str]:
    """
    Recover code from a generation that omitted the ```python fence.

    The model is trained to fence its output, but occasionally emits a bare
    class.  Rather than throw away a potentially-good architecture, slice from
    the first plausible code anchor to the end and accept it only if it is
    syntactically valid and contains a proper Net class.
    """
    anchors = ["import torch", "def supported_hyperparameters", "class Net"]
    starts = [raw.find(a) for a in anchors if raw.find(a) != -1]
    if not starts:
        return None
    candidate = raw[min(starts):].strip()
    # Drop any trailing prose after the last dedented blank-line block is hard;
    # rely on the compiler — if the tail is prose it will usually still parse as
    # long as it's commented or absent.  Require both syntax + Net structure.
    ok_syntax, _ = validate_code(candidate)
    if not ok_syntax:
        return None
    has_net, _ = check_net_class(candidate)
    if not has_net:
        return None
    return candidate


# ── Prompt construction ──────────────────────────────────────────────────────

# Base persona — verbatim from the ABrain/NNGPT-UniqueArch-Rag model card.
SYSTEM_PERSONA = (
    "You are an expert PyTorch architecture designer specializing in creating "
    "UNIQUE, high-performing neural networks optimized for first-epoch accuracy."
)

# Fallback execution rules (used only if the conf file cannot be read).  Kept
# deliberately short; the real rules live in
# conf/prompt/test/unique_rag_test_rules.json and are loaded when present.
_FALLBACK_RULES = (
    "\nThe code MUST be directly executable by the evaluation tool.\n"
    "1. Start with `import torch` and `import torch.nn as nn`.\n"
    "2. Define `def supported_hyperparameters(): return {'lr', 'momentum'}` "
    "OUTSIDE all classes.\n"
    "3. Define a `Net(nn.Module)` whose `__init__(self, in_shape, out_shape, "
    "prm, device)` derives in_channels from `int(in_shape[0])` and num_classes "
    "from `int(out_shape[0])` — NEVER hardcode them.\n"
    "4. Net MUST implement `forward(self, x)`, `train_setup(self, prm)` and "
    "`learn(self, train_data)`.\n"
    "5. Complete, executable code only — no placeholders, no `...`.\n"
)


CRITICAL_REINFORCEMENT = (
    "\n\nCRITICAL — read this carefully, it OVERRIDES any earlier instruction "
    "about in_shape:\n"
    "This evaluator instantiates your model as `Net(in_shape, out_shape, prm, "
    "device)` where **in_shape is (batch, channels, height, width)** — a 4-tuple "
    "whose FIRST element (in_shape[0]) is the BATCH size (1), NOT the channels. "
    "The channel count is in_shape[1]. Any earlier rule that says to read input "
    "channels from in_shape[0] is WRONG here and builds a 1-channel first conv "
    "that crashes on the 3-channel images.\n"
    "1. FIRST-LAYER INPUT CHANNELS — derive it robustly with a length check:\n"
    "       in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]\n"
    "   then `nn.Conv2d(in_channels, ...)` for the first conv. For this RGB "
    "dataset in_channels is 3. NEVER use in_shape[0] directly as the channel "
    "count and NEVER hardcode `nn.Conv2d(1, ...)`.\n"
    "2. OUTPUT MUST BE 2-D LOGITS of shape (batch, {num_classes}): "
    "num_classes = int(out_shape[0]). Reduce ALL spatial dimensions before the "
    "classifier — end with global pooling then flatten then Linear, e.g. "
    "`x = nn.AdaptiveAvgPool2d((1, 1))(x); x = torch.flatten(x, 1); "
    "x = self.classifier(x)` where `self.classifier = nn.Linear(C, num_classes)`. "
    "NEVER return a 4-D feature map (N, C, H, W) — the loss only accepts "
    "(N, num_classes)."
)

# Rotated through generations to push structural diversity (→ more novel,
# more desirable models, fewer near-duplicate rejections).  Stays inside the
# model's training distribution: these are families it was trained on.
DIVERSITY_HINTS = [
    "a residual / skip-connection family",
    "a dense / feature-reuse family",
    "an inverted-bottleneck / depthwise-separable family",
    "a multi-branch / inception-style family",
    "a lightweight attention-augmented family",
    "a plain VGG-style stacked-convolution family",
    "a grouped-convolution / channel-shuffle family",
    "a pyramid / multi-scale pooling family",
]


def load_execution_rules() -> str:
    """Load the system_prompt_template execution rules the model was SFT'd on."""
    rules_path = conf_test_dir / "unique_rag_test_rules.json"
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rules = data.get("system_prompt_template", "")
        if rules and rules.strip():
            return rules
    except Exception as e:  # noqa: BLE001 — best-effort, fall back gracefully
        print(f"[GEN][WARN] Could not read {rules_path}: {e}. Using fallback rules.")
    return _FALLBACK_RULES


# Teacher-forcing prefix: the model's answer is seeded with these tokens so it
# starts from the exact NNEval-required structure (imports + the
# supported_hyperparameters function) instead of free-forming the preamble.
# Mirrors how ab/gpt/util/nn_sftcodegen_rag.py uses prefix_code.
_FALLBACK_PREFIX = (
    "```python\n"
    "import torch\n"
    "import torch.nn as nn\n\n"
    "def supported_hyperparameters():\n"
    "    return {'lr', 'momentum'}\n\n"
)


def load_prefix_code() -> str:
    """Load the prefix_code the model is teacher-forced to start its answer with."""
    rules_path = conf_test_dir / "unique_rag_test_rules.json"
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pfx = data.get("prefix_code", "")
        if pfx and pfx.strip():
            return pfx
    except Exception as e:  # noqa: BLE001
        print(f"[GEN][WARN] Could not read prefix_code from {rules_path}: {e}")
    return _FALLBACK_PREFIX


def _quick_structural_check(code: str, channels: int = 3, hw: int = 32, num_classes: int = 10):
    """
    Pre-flight a generated model EXACTLY the way nn-dataset's evaluator does:
    instantiate ``Net(in_shape, out_shape, prm, device)`` with the framework's
    4-tuple ``in_shape = (batch=1, channels, H, W)`` and ``out_shape =
    (num_classes,)``, then forward a real 3-channel batch.  Using the same
    4-tuple is essential — a 3-tuple gave false passes because in_shape[0]
    happened to equal the channel count.

    Catches, before any GPU training is wasted:
      * channels read from in_shape[0] (the batch=1) → 1-channel conv → forward_error
      * 4-D output (no global pool/flatten) → bad_output_shape
      * shape mismatches / missing methods → init_/forward_error

    Returns (ok: bool, message: str).
    """
    import torch as _torch

    ns: Dict = {}
    try:
        exec("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
             "from torch import Tensor\n", ns)
        exec(code, ns)
    except Exception as e:  # noqa: BLE001
        return False, f"exec_error: {type(e).__name__}: {e}"

    Net = ns.get("Net")
    if Net is None:
        return False, "no_Net_class"

    device = _torch.device("cpu")
    prm = {"lr": 0.01, "momentum": 0.9, "dropout": 0.2}
    # nn-dataset passes in_shape as (batch, channels, H, W); out_shape as (num_classes,).
    eval_in_shape = (1, channels, hw, hw)
    try:
        model = Net(eval_in_shape, (num_classes,), prm, device)
    except Exception as e:  # noqa: BLE001
        return False, f"init_error: {type(e).__name__}: {e}"

    try:
        model.eval()
        x = _torch.randn(2, channels, hw, hw)  # a real 3-channel batch
        with _torch.no_grad():
            y = model(x)
    except Exception as e:  # noqa: BLE001
        return False, f"forward_error: {type(e).__name__}: {e}"

    shape = tuple(getattr(y, "shape", ()))
    if shape != (2, num_classes):
        return False, f"bad_output_shape: got {shape}, want (2, {num_classes})"
    return True, "ok"


# Used by KTO_PROMPT_MODE=minimal: the ONLY correctness fix (channel index),
# WITHOUT the diversity hints or output-head guidance that "enhanced" adds.
MINIMAL_INSHAPE_FIX = (
    "\n\nIMPORTANT (overrides any earlier in_shape guidance): this evaluator passes "
    "in_shape as (batch, channels, height, width). Derive the first conv's input "
    "channels as `in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]` "
    "(it is 3 for this RGB dataset) and use `nn.Conv2d(in_channels, ...)`. Do NOT use "
    "in_shape[0] as the channel count, and do not hardcode it."
)


def build_prompt_messages(
    dataset: str = "cifar-10",
    params_limit: int = 500_000,
    diversity_hint: Optional[str] = None,
    execution_rules: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build the [system, user] chat messages used both to GENERATE a model and
    (later, in the orchestrator) to form the KTO ``prompt_messages`` paired
    with that model's completion.  Keeping this in one place guarantees the
    training prompt matches the generation prompt exactly.

    The KTO_PROMPT_MODE env var selects how much of OUR prompt engineering is
    applied (so an ablation .sh can A/B test it without code changes):
      * "enhanced" (default): SYSTEM_PERSONA + repo rules + diversity hint +
        full CRITICAL_REINFORCEMENT (in_shape fix AND output-head guidance).
      * "minimal":  repo rules + ONLY the one-line in_shape channel correction
        (no hints, no head guidance) — isolates "do our quality nudges help?".
      * "original": pure repo prompt, none of our additions. WARNING: the repo
        rules say in_channels=in_shape[0], which is WRONG for this evaluator
        (channels are in_shape[1]) → expect Conv2d(1,...) eval failures.
    """
    mode = os.environ.get("KTO_PROMPT_MODE", "enhanced").strip().lower()
    rules = execution_rules if execution_rules is not None else load_execution_rules()
    system = SYSTEM_PERSONA + "\n\n" + rules

    num_classes = {"cifar-10": 10, "cifar-100": 100}.get(dataset, 10)
    dataset_label = {
        "cifar-10": "CIFAR-10 (32x32 RGB, channels-first C x H x W), 10 classes",
        "cifar-100": "CIFAR-100 (32x32 RGB, channels-first C x H x W), 100 classes",
    }.get(dataset, f"{dataset} (channels-first C x H x W)")

    user = (
        "Task: Design a PyTorch CV model for image classification.\n"
        f"Dataset: {dataset_label}.\n"
        f"Resource limits: params <= {params_limit}; latency budget: tight "
        "(edge-friendly).\n"
        "Constraints: use standard layers only; no pretrained weights."
    )
    # Diversity hint is one of OUR additions → enhanced-only.
    if mode == "enhanced" and diversity_hint:
        user += (
            f"\nArchitectural direction for this design: explore {diversity_hint}. "
            "Produce a distinct, novel architecture (do not repeat a generic CNN)."
        )

    # Reinforcement goes LAST so it is the most recent context before generation.
    if mode == "enhanced":
        user += CRITICAL_REINFORCEMENT.format(num_classes=num_classes)
    elif mode == "minimal":
        user += MINIMAL_INSHAPE_FIX
    # mode == "original": add nothing (pure repo prompt; carries the in_shape[0] bug)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── Model loading ────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(base_model: str, adapter: Optional[str], context_length: int):
    """Load the base NNGPT model in 4-bit, optionally with a prior LoRA adapter."""
    from ab.gpt.util.LLM import LLM

    access_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )

    # Reuse the repo's loader (4-bit + bfloat16 + tokenizer caching). Generation
    # always tokenizes with add_special_tokens=False, so LLM's add_eos_token has
    # no effect here, and single-sequence generation is unaffected by padding_side.
    print(f"[GEN] Loading base model in 4-bit via LLM: {base_model}")
    loader = LLM(
        base_model,
        quantization_config_4bit,
        access_token=access_token,
        context_length=context_length,
    )
    model, tokenizer = loader.get_model(), loader.get_tokenizer()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter:
        adapter_path = Path(adapter)
        if adapter_path.exists():
            from peft import PeftModel

            print(f"[GEN] Attaching previous-cycle LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
        else:
            print(f"[GEN][WARN] Adapter path not found, generating from base only: {adapter_path}")

    model.eval()
    return model, tokenizer


# ── Generation ───────────────────────────────────────────────────────────────

def generate_models(
    *,
    base_model: str,
    adapter: Optional[str],
    out_dir: Path,
    records_file: Path,
    num_models: int,
    start_index: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
    dataset: str,
    params_limit: int,
    context_length: int,
    seed: int,
    max_rejections: int = 5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = CodeExtractor()
    execution_rules = load_execution_rules()
    prefix_code = load_prefix_code()
    num_classes = {"cifar-10": 10, "cifar-100": 100}.get(dataset, 10)

    model, tokenizer = load_model_and_tokenizer(base_model, adapter, context_length)

    # One-time prompt dump so the run is self-verifying: the SLURM log and a file
    # on disk show the EXACT system+user prompt the model was conditioned on
    # (proves whether the channel/output reinforcement is actually in effect).
    _probe_messages = build_prompt_messages(
        dataset=dataset, params_limit=params_limit,
        diversity_hint=DIVERSITY_HINTS[start_index % len(DIVERSITY_HINTS)],
        execution_rules=execution_rules,
    )
    _prompt_mode = os.environ.get("KTO_PROMPT_MODE", "enhanced").strip().lower()
    # flush=True so this appears in the SLURM log immediately (stdout is
    # block-buffered to a file; without flushing it hides behind tqdm's stderr).
    print("=" * 80, flush=True)
    print(f"[GEN] PROMPT IN USE (KTO_PROMPT_MODE={_prompt_mode}):", flush=True)
    print("---- SYSTEM ----", flush=True)
    print(_probe_messages[0]["content"], flush=True)
    print("---- USER ----", flush=True)
    print(_probe_messages[1]["content"], flush=True)
    print(f"---- PREFIX (teacher-forced start) ----\n{prefix_code}", flush=True)
    print(f"[GEN] temperature={temperature}, max_rejections={max_rejections} "
          f"(retry until model instantiates + forwards on a 3x32x32 batch)", flush=True)
    print("=" * 80, flush=True)
    try:
        (out_dir / "_prompt_used.txt").write_text(
            "SYSTEM:\n" + _probe_messages[0]["content"]
            + "\n\nUSER:\n" + _probe_messages[1]["content"]
            + "\n\nPREFIX:\n" + prefix_code,
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass

    records: List[Dict] = []
    n_ok = 0
    for i in range(num_models):
        idx = start_index + i
        model_id = f"gen_{idx:04d}"
        model_dir = out_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        hint = DIVERSITY_HINTS[idx % len(DIVERSITY_HINTS)]

        messages = build_prompt_messages(
            dataset=dataset,
            params_limit=params_limit,
            diversity_hint=hint,
            execution_rules=execution_rules,
        )
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        base_inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        # Teacher-force the answer to begin with prefix_code so structure is anchored.
        if prefix_code:
            prefix_ids = tokenizer(
                prefix_code, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(model.device)
            input_ids = torch.cat([base_inputs.input_ids, prefix_ids], dim=1)
        else:
            input_ids = base_inputs.input_ids
        attention_mask = torch.ones_like(input_ids)

        # ── Rejection sampling: keep regenerating until the model instantiates
        #    and forwards correctly (or we exhaust max_rejections attempts). ──
        accepted_code: Optional[str] = None
        last_raw = ""
        last_reason = "no_attempt"
        attempts_used = 0
        for attempt in range(max_rejections + 1):
            attempts_used = attempt + 1
            _seed_everything(seed + idx + attempt * 7919)
            try:
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = generated[0][input_ids.shape[1]:]
                gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                raw_output = (prefix_code + gen_text) if prefix_code else gen_text
            except Exception as e:  # noqa: BLE001
                last_reason = f"generation_crashed: {type(e).__name__}: {e}"
                torch.cuda.empty_cache()
                continue
            last_raw = raw_output

            code, message = extractor.extract(raw_output)
            if code is None:
                salvaged = _salvage_fenceless_code(raw_output)
                if salvaged is not None:
                    code, message = salvaged, "salvaged_fenceless"
            if code is None:
                last_reason = f"extraction_failed: {message}"
                continue  # reject → retry
            if "import torch" not in code:
                code = "import torch\nimport torch.nn as nn\n\n" + code

            ok_struct, check_msg = _quick_structural_check(
                code, channels=3, hw=32, num_classes=num_classes
            )
            if not ok_struct:
                last_reason = f"structural_check_failed: {check_msg}"
                last_raw = raw_output  # keep this (real bad arch) as the negative
                continue  # reject → retry
            accepted_code = code
            break

        raw_file = model_dir / "raw_output.txt"
        raw_file.write_text(last_raw, encoding="utf-8")

        if accepted_code is not None:
            code_file = model_dir / new_nn_file
            code_file.write_text(accepted_code, encoding="utf-8")
            n_ok += 1
            records.append({
                "model_id": model_id, "index": idx, "ok": True,
                "code_file": str(code_file), "raw_file": str(raw_file),
                "error": None, "diversity_hint": hint, "attempts": attempts_used,
            })
            print(f"[GEN] {model_id}: ok ({hint}) [{attempts_used} attempt(s)]", flush=True)
        else:
            # All attempts failed validation → still a usable KTO negative (the
            # last bad generation), but NOT written as new_nn.py so the evaluator
            # doesn't waste a training run on a model we already know is broken.
            records.append({
                "model_id": model_id, "index": idx, "ok": False,
                "code_file": None, "raw_file": str(raw_file),
                "error": last_reason, "diversity_hint": hint, "attempts": attempts_used,
            })
            print(f"[GEN] {model_id}: rejected after {attempts_used} attempt(s) — "
                  f"{last_reason}", flush=True)

        torch.cuda.empty_cache()

    records_file.parent.mkdir(parents=True, exist_ok=True)
    with open(records_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[GEN] Done. {n_ok}/{num_models} passed structural pre-check "
          f"(will be evaluated). Records → {records_file}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="KTO pipeline self-contained model generator")
    parser.add_argument("--base_model", type=str, required=True,
                        help="HF id or local path of the domain-trained NNGPT model")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Optional LoRA adapter dir from the previous KTO cycle")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output dir (will contain gen_XXXX/new_nn.py for NNEval)")
    parser.add_argument("--records_file", type=str, required=True,
                        help="Where to write generation_records.jsonl")
    parser.add_argument("--num_models", type=int, default=30)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_rejections", type=int, default=5,
                        help="Regenerate a model up to this many times if it fails "
                             "the structural pre-check (instantiate + forward)")
    parser.add_argument("--dataset", type=str, default="cifar-10")
    parser.add_argument("--params_limit", type=int, default=500_000)
    parser.add_argument("--context_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=43)

    args = parser.parse_args()

    generate_models(
        base_model=args.base_model,
        adapter=args.adapter,
        out_dir=Path(args.out_dir),
        records_file=Path(args.records_file),
        num_models=args.num_models,
        start_index=args.start_index,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        dataset=args.dataset,
        params_limit=args.params_limit,
        context_length=args.context_length,
        seed=args.seed,
        max_rejections=args.max_rejections,
    )


if __name__ == "__main__":
    main()
