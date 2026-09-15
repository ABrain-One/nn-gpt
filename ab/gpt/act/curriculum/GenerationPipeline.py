"""
GenerationPipeline.py
----------------------------------------------------------------------------------------
Fully automatic progressive curriculum fine-tuning pipeline.

Location: nn-gpt/ab/gpt/act/curriculum/GenerationPipeline.py

Runs the complete L1 -> L2(k=2) -> L2(k=3) -> L3(k=2) -> L3(k=3) -> L3(k=4)
curriculum in the correct order given only a dataset name. Each level automatically:
  1. Checks DB viability (enough anchor groups in the required band)
  2. Adapts the proven CIFAR-10 prompt for the target dataset
  3. Writes LLM conf, entry script, and prompt configs
  4. Runs the curriculum fine-tuning subprocess
  5. Selects the best epoch adapter by composite score
  6. Merges the adapter into the cumulative model
  7. Cleans epoch outputs and advances to the next level

Supported datasets (viability auto-checked against LEMUR DB):
    cifar-10      Full curriculum viable (all bands)
    svhn          very_low_near only (L3)
    celeba-gender very_low_near only (L3)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from ab.nn.util.Const import ab_root_path, out_dir
from ab.gpt.util.Const import gpt_dir, conf_dir, conf_test_dir, conf_train_dir, conf_llm_dir


# ----- Project root ---------------------------------------------------------------------------------
# Resolved by walking up until the ab/gpt/conf marker is found, rather than by a
# fixed number of .parents hops. This keeps the pipeline correct if the file is
# moved deeper or shallower in the package tree.

_THIS_FILE = Path(__file__).resolve()

NNGPT_OUT   = out_dir / "nngpt"

# Generated entry scripts are written next to this file, and the module path used
# to launch them is derived from that location, so the two can never drift apart.
CURRICULUM_DIR    = _THIS_FILE.parent
CURRICULUM_MODULE = ".".join(CURRICULUM_DIR.relative_to(ab_root_path).parts)

# ----- Base model (original, never modified) ----------------------------------------------------------

BASE_MODEL_NAME = "open-r1/OlympicCoder-7B"
BASE_MODEL_PATH = out_dir / "llm" / "open-r1" / "OlympicCoder-7B"
TOKENIZER_CACHE = out_dir / "tokenizer" / "open-r1" / "OlympicCoder-7B"

# Dynamic override — use merged model from run_config.json if available
_run_cfg_path = NNGPT_OUT / "run_config.json"
if _run_cfg_path.exists():
    try:
        _cfg = json.loads(_run_cfg_path.read_text())
        _merged = _cfg.get("base_model_name") or _cfg.get("base_model_name_or_path", "")
        if _merged and Path(_merged).exists() and _merged != BASE_MODEL_NAME:
            BASE_MODEL_NAME = _merged
    except Exception:
        pass


# ----- Logging --------------------------------------------------------------------------------------
def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ----- Inline setup helpers (no external model_setup module needed) ----------------------------------

def ensure_model(model_dir: Optional[Path] = None) -> Path:
    """Download OlympicCoder-7B weights + tokenizer if any files are missing."""
    from transformers import AutoTokenizer
    REQUIRED = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    target = Path(model_dir) if model_dir else BASE_MODEL_PATH
    target.mkdir(parents=True, exist_ok=True)

    has_weights = (
        (target / "model.safetensors").exists()
        or any(target.glob("model-*-of-*.safetensors"))
        or (target / "pytorch_model.bin").exists()
    )
    missing = [f for f in REQUIRED if not (target / f).exists()]

    if has_weights and not missing:
        return target

    if not has_weights:
        import torch
        from transformers import AutoModelForCausalLM
        log("Model weights not found — downloading open-r1/OlympicCoder-7B (~15GB) ...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME if not Path(BASE_MODEL_NAME).exists() else "open-r1/OlympicCoder-7B",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.save_pretrained(str(target))
        log(f"Model weights saved to {target}")

    if missing:
        log(f"Missing tokenizer files {missing} — downloading ...")
        tok = AutoTokenizer.from_pretrained("open-r1/OlympicCoder-7B", trust_remote_code=True)
        tok.save_pretrained(str(target))
        TOKENIZER_CACHE.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(TOKENIZER_CACHE))
        # Extract chat_template.jinja if stored inline in tokenizer_config.json
        for d in [target, TOKENIZER_CACHE]:
            jinja = d / "chat_template.jinja"
            cfg_path = d / "tokenizer_config.json"
            if not jinja.exists() and cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                    tmpl = cfg.get("chat_template")
                    if tmpl and isinstance(tmpl, str):
                        jinja.write_text(tmpl)
                except Exception:
                    pass
        log(f"Tokenizer saved to {target}")

    return target


# ----- Fixed curriculum sequence ----------------------------------------------------------------------
# Order is immutable — each level builds on the merged adapter of all previous.
# The drift is significant from Medium band to very_low_near band — because
# (0.65 - 0.85) has only 1 anchor group for CIFAR-10.
CURRICULUM_SEQUENCE = [
    {"level": "L1", "band": "high",          "k": 2, "epochs": 10,
     "description": "High-similarity references — establishes LEMUR-compatible code pattern"},
    {"level": "L2", "band": "medium",        "k": 2, "epochs": 8,
     "description": "Medium-similarity, k=2 — moderate diversity after L1 foundation"},
    {"level": "L2", "band": "medium",        "k": 3, "epochs": 12,
     "description": "Medium-similarity, k=3 — three references after two-reference stage"},
    {"level": "L3", "band": "very_low_near", "k": 2, "epochs": 10,
     "description": "Very-low-near, k=2 — moderate diversity after L2 foundation"},
    {"level": "L3", "band": "very_low_near", "k": 3, "epochs": 7,
     "description": "Very-low-near, k=3 — diverse references requiring synthesis"},
    {"level": "L3", "band": "very_low_near", "k": 4, "epochs": 12,
     "description": "Very-low-near, k=4 — maximum reference diversity"},
]

# ----- Dataset configurations ---------------------------------------------------------
DATASET_CONFIGS = {
    "cifar-10": {
        "task":             "img-classification",
        "metric":           "acc",
        "transform":        "norm_256_flip",
        "in_channels":      3,
        "dummy_size":       224,
        "viable_bands":     ["high", "medium", "very_low_near"],
        "note":             "Full curriculum viable — 13,023 NNs, 2,634 acc≥0.85",
        # Architecture constraints for prompt generation
        "anchor_family":    "rl-bb-init",
        "use_pretrained":   True,
        "backbone_whitelist": [
            "resnet50", "densenet169", "mobilenet_v3_large",
            "efficientnet_b0", "efficientnet_b3",
        ],
        "custom_cnn":       False,
        "out_classes":      10,
        "batch_hint":       16,
        "input_size":       256,
        "transform_value":  "norm_256_flip",
        "arch_notes": [
            "Use at most two backbone models total — never three.",
            "Only use these backbone models: 'resnet50', 'densenet169', "
            "'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b3'.",
            "Always use weights='DEFAULT' when instantiating TorchVision.",
            "self.features() must receive raw input x — never backbone output.",
        ],
    },
    "svhn": {
        "task":             "img-classification",
        "metric":           "acc",
        "transform":        "norm_64_flip",
        "in_channels":      3,
        "dummy_size":       64,
        "viable_bands":     ["very_low_near"],
        "note":             "L3 only — 4,568 NNs, unq-family anchor, max J=0.82",
        # Architecture constraints for prompt generation
        "anchor_family":    "unq",
        "use_pretrained":   False,
        "backbone_whitelist": [],
        "custom_cnn":       True,
        "out_classes":      10,
        "batch_hint":       512,
        "input_size":       64,
        "transform_value":  "norm_64_flip",
        "arch_notes": [
            "SVHN contains 32x32 RGB street digit images (10 classes).",
            "Do NOT use pretrained TorchVision backbones.",
            "Implement a compact custom CNN from scratch suitable for "
            "small 32x32 or 64x64 inputs.",
            "Use batch size 512 or 1024 — SVHN training requires large batches.",
            "forward() must return a single Tensor of shape (batch, 10).",
        ],
    },
    "celeba-gender": {
        "task":             "img-classification",
        "metric":           "acc",
        "transform":        "norm_32_flip",
        "in_channels":      3,
        "dummy_size":       32,
        "viable_bands":     [],
        "note":             "Not viable — only 1 unique architecture at high J (ga-352 repeated)",
        # Architecture constraints for prompt generation
        "anchor_family":    "InceptionV3",
        "use_pretrained":   False,
        "backbone_whitelist": [],
        "custom_cnn":       True,
        "out_classes":      2,
        "batch_hint":       4,
        "input_size":       32,
        "transform_value":  "norm_32_flip",
        "arch_notes": [
            "CelebA-Gender is a BINARY classification task (2 classes: male/female).",
            "out_shape will be (batch, 2) — use 2 output neurons, not 10.",
            "Do NOT use pretrained TorchVision backbones.",
            "Implement a compact Inception-style custom CNN.",
            "Use batch size 4 or 8 — CelebA images are large.",
            "forward() must return a single Tensor of shape (batch, 2).",
        ],
    },
    "mnist": {
        "task":             "img-classification",
        "metric":           "acc",
        "transform":        "norm_28",
        "in_channels":      1,
        "dummy_size":       28,
        "viable_bands":     [],
        "note":             "Not viable — max pairwise J=0.23",
        "anchor_family":    "unknown",
        "use_pretrained":   False,
        "backbone_whitelist": [],
        "custom_cnn":       True,
        "out_classes":      10,
        "batch_hint":       64,
        "input_size":       28,
        "transform_value":  "norm_28",
        "arch_notes": [
            "MNIST contains 28x28 greyscale digit images (10 classes).",
            "Input has 1 channel — do not assume 3-channel input.",
            "Use a simple CNN — no pretrained backbones.",
        ],
    },
    "cifar-100": {
        "task":             "img-classification",
        "metric":           "acc",
        "transform":        "norm_256_flip",
        "in_channels":      3,
        "dummy_size":       224,
        "viable_bands":     [],
        "note":             "Not viable — 0 models above 72.9% accuracy",
        "anchor_family":    "unknown",
        "use_pretrained":   True,
        "backbone_whitelist": [
            "resnet50", "densenet169", "efficientnet_b3",
        ],
        "custom_cnn":       False,
        "out_classes":      100,
        "batch_hint":       16,
        "input_size":       256,
        "transform_value":  "norm_256_flip",
        "arch_notes": [
            "CIFAR-100 contains 32x32 RGB images across 100 classes.",
            "out_shape will be (batch, 100) — use 100 output neurons.",
        ],
    },
}

# ----- Proven CIFAR-10 prompt files (adapted for other datasets) --------------------------------------
PROVEN_PROMPTS = {
    ("L1", 2): ("Curriculum_L1_high_k2.json",              "Curriculum_L1_high_k2_train.json"),
    ("L2", 2): ("Curriculum_L2_medium_k2.json",            "Curriculum_L2_medium_k2_train.json"),
    ("L2", 3): ("Curriculum_L2_medium_k3.json",            "Curriculum_L2_medium_k3_train.json"),
    ("L3", 2): ("Curriculum_L3_very_low_near_k2.json",     "Curriculum_L3_very_low_near_k2_train.json"),
    ("L3", 3): ("Curriculum_L3_very_low_near_k3.json",     "Curriculum_L3_very_low_near_k3_train.json"),
    ("L3", 4): ("Curriculum_L3_very_low_near_k4.json",     "Curriculum_L3_very_low_near_k4_train.json"),
}


# ----- Naming helpers ---------------------------------------------------------------------------------
def get_step_id(level: str, k: int) -> str:
    """Unique identifier for a curriculum step. Uses hyphens per LEMUR convention."""
    return f"{level.lower()}-k{k}"


def get_conf_id(dataset: str, level: str, k: int) -> str:
    dataset_safe = dataset.replace("-", "_")
    return f"curriculum_{dataset_safe}_{level.lower()}_k{k}"


def get_prompt_name(dataset: str, level: str, k: int, is_train: bool) -> str:
    dataset_safe = dataset.replace("-", "_")
    suffix = "_train" if is_train else ""
    return f"Curriculum_{dataset_safe}_{level}_k{k}{suffix}.json"


def get_nn_prefix(dataset: str, level: str, k: int) -> str:
    """NN name prefix for LEMUR DB — hyphens not underscores."""
    dataset_safe = dataset.replace("-", "_")
    return f"{dataset_safe}_{level.lower()}-k{k}"


# ----- Progress tracking ------------------------------------------------------------------------------
def progress_path(dataset: str) -> Path:
    return out_dir / "curriculum" / dataset / "progress.json"


def load_progress(dataset: str) -> dict:
    p = progress_path(dataset)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"completed_steps": [], "current_merged_model": str(BASE_MODEL_PATH)}


def save_progress(dataset: str, progress: dict) -> None:
    p = progress_path(dataset)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(progress, indent=2))


def step_key(level: str, k: int) -> str:
    return f"{level}_k{k}"


# ----- DB viability check -----------------------------------------------------------------------------
def check_band_viability(dataset: str, band: str, k: int) -> tuple[bool, int]:
    """
    Check if the LEMUR DB has enough anchor groups for this dataset/band/k.
    Returns (is_viable, row_count).
    """
    try:
        from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
        from transformers import AutoTokenizer
        import tempfile

        if not TOKENIZER_CACHE.exists():
            log("Tokenizer cache not found — downloading from HuggingFace ...")
            try:
                ensure_model()
            except Exception as e:
                log(f"Could not initialise tokenizer: {e} — assuming viable", "WARN")
                return True, k

        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_CACHE), local_files_only=True)
        cfg = DATASET_CONFIGS.get(dataset, {})

        tmp = Path(tempfile.mktemp(suffix=".json"))
        tmp.write_text(json.dumps({
            f"check_{dataset}_{band}": {
                "type":            "curriculum_prompt",
                "is_generation":   True,
                "selection_mode":  "tall",
                "task":            cfg.get("task", "img-classification"),
                "dataset":         dataset,
                "metric":          cfg.get("metric", "acc"),
                "nn_prefixes":     [],
                "num_joint_nns":   k,
                "similarity_mode": "anchor_band_db_minhash",
                "similarity_band": band,
                "anchor_strategy": "auto",
                "input_list":      [{"para": f"acc_{i}", "value": f"acc_{i}"} for i in range(1, k + 1)],
                "prompt":          ["test"],
                "output":          []
            }
        }))

        try:
            builder = NNGenPrompt(max_len=8192, tokenizer=tok, prompts_path=str(tmp))
            df = builder.get_raw_dataset(only_best_accuracy=False, n_training_prompts=100)
            row_count = len(df)
            return row_count >= k, row_count
        except Exception as e:
            log(f"Viability check error: {e}", "WARN")
            return False, 0
        finally:
            tmp.unlink(missing_ok=True)

    except ImportError as e:
        log(f"Cannot import LEMUR ({e}) — skipping viability check", "WARN")
        return True, k


# ----- Prompt adaptation ------------------------------------------------------------------------------
# Lines in the proven CIFAR-10 prompts that are backbone/architecture-specific
# and must be REPLACED (not inherited) when adapting to a different dataset.
_CIFAR10_BACKBONE_LINES = {
    # The backbone whitelist constraint
    "Only use these backbone models:",
    # The pretrained weights instruction
    "Always use weights='DEFAULT'",
    # The self.features() instruction (CIFAR-10 specific)
    "self.features() must receive raw input x",
    # The maximum backbone count
    "Use at most two backbone models total",
    # The TorchVision helper reference
    "TorchVision helper class",
    # The transform hardcode
    "norm_256_flip",
    # The batch hint
    "batch size capped at 16",
}


def _is_backbone_line(line: str) -> bool:
    """Return True if this prompt line is CIFAR-10-specific architecture guidance."""
    return any(marker in line for marker in _CIFAR10_BACKBONE_LINES)


def adapt_prompt(source_cfg: dict, dataset: str, level: str, k: int, is_train: bool) -> dict:
    """
    Adapt a proven CIFAR-10 prompt config for a different dataset.

    For CIFAR-10: only dataset name substitution (trivial no-op).
    For other datasets: replace architecture-specific constraints with
    dataset-appropriate ones derived from DATASET_CONFIGS[dataset].

    This ensures the LLM receives correct guidance about:
      - Whether to use pretrained TorchVision backbones
      - Which specific backbone models are allowed
      - Input image size and number of output classes
      - Batch size appropriate for the dataset
      - Transform name
    """
    cfg     = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS["cifar-10"])
    conf_id = get_conf_id(dataset, level, k)

    source_conf_id = list(source_cfg.keys())[0]
    source_conf    = source_cfg[source_conf_id]

    new_conf        = dict(source_conf)
    new_conf["dataset"] = dataset
    new_conf["task"]    = cfg["task"]
    new_conf["metric"]  = cfg["metric"]

    is_cifar10 = (dataset == "cifar-10")
    transform  = cfg.get("transform_value", cfg.get("transform", "norm_256_flip"))
    in_size    = cfg.get("input_size", 256)
    batch_hint = cfg.get("batch_hint", 16)
    out_cls    = cfg.get("out_classes", 10)
    arch_notes = cfg.get("arch_notes", [])

    adapted_prompt = []
    arch_notes_inserted = False

    for line in source_conf.get("prompt", []):
        # Always substitute dataset name and transform
        line = line.replace("'cifar-10'", f"'{dataset}'")
        line = line.replace("dataset='cifar-10'", f"dataset='{dataset}'")
        line = line.replace('"cifar-10"', f'"{dataset}"')
        line = line.replace("norm_256_flip", transform)

        if is_cifar10:
            # No architecture substitution needed for CIFAR-10
            adapted_prompt.append(line)
            continue

        # For non-CIFAR-10 datasets: skip lines that carry CIFAR-10
        # architecture constraints and replace with dataset-specific ones
        if _is_backbone_line(line):
            if not arch_notes_inserted:
                # Insert dataset-specific architecture constraints once
                adapted_prompt.extend(arch_notes)
                # Insert transform constraint
                adapted_prompt.append(
                    f"The 'transform' value in <hp> must be exactly '{transform}'."
                )
                # Insert out_classes constraint if not the default 10
                if out_cls != 10:
                    adapted_prompt.append(
                        f"The task has {out_cls} output classes — "
                        f"use {out_cls} output neurons in the final layer."
                    )
                # Insert batch constraint
                adapted_prompt.append(
                    f"Use batch size {batch_hint} — appropriate for this dataset."
                )
                arch_notes_inserted = True
            # Skip the original CIFAR-10 backbone line
            continue

        # Substitute 256x256 size references with dataset input size
        if "256, 256" in line and in_size != 256:
            line = line.replace("256, 256", f"{in_size}, {in_size}")
        if "256x256" in line and in_size != 256:
            line = line.replace("256x256", f"{in_size}x{in_size}")

        adapted_prompt.append(line)

    # If no backbone lines were found to trigger insertion,
    # append arch_notes at the end before references
    if not is_cifar10 and not arch_notes_inserted and arch_notes:
        # Find insertion point — just before "Reference A"
        insert_idx = next(
            (i for i, l in enumerate(adapted_prompt) if "Reference A" in l),
            len(adapted_prompt)
        )
        for note in arch_notes:
            adapted_prompt.insert(insert_idx, note)
            insert_idx += 1
        adapted_prompt.insert(
            insert_idx,
            f"The 'transform' value in <hp> must be exactly '{transform}'."
        )
        if out_cls != 10:
            adapted_prompt.insert(
                insert_idx + 1,
                f"The task has {out_cls} output classes — use {out_cls} output neurons."
            )

    new_conf["prompt"] = adapted_prompt

    if is_train:
        adapted_output = []
        for line in source_conf.get("output", []):
            line = line.replace("'cifar-10'", f"'{dataset}'")
            line = line.replace("norm_256_flip", transform)
            adapted_output.append(line)
        new_conf["output"]        = adapted_output
        new_conf["is_generation"] = False
    else:
        new_conf["output"]        = []
        new_conf["is_generation"] = True

    return {conf_id: new_conf}


# ----- Config file writers ----------------------------------------------------------------------------
def write_llm_conf(dataset: str, current_model_path: str, dry_run: bool = False) -> str:
    """
    Write LLM configuration JSON for this curriculum step.
    Uses current_model_path — the merged model from the previous step,
    or the original base model for the first step.
    """
    dataset_safe = dataset.replace("-", "_")
    conf_name    = f"ds_coder_7b_olympic_ft_{dataset_safe}.json"
    conf_path    = conf_llm_dir / conf_name

    # Resolve the model path to write into the conf:
    # 1. Use current_model_path if it points to an existing directory
    # 2. Fall back to run_config.json if available
    # 3. Fall back to BASE_MODEL_NAME as last resort
    resolved_path = BASE_MODEL_NAME
    if current_model_path and Path(current_model_path).exists():
        resolved_path = current_model_path
        log(f"  Using merged model: {Path(current_model_path).name}")
    else:
        run_cfg = NNGPT_OUT / "run_config.json"
        if run_cfg.exists():
            try:
                cfg = json.loads(run_cfg.read_text())
                p = cfg.get("base_model_name", "")
                if p and Path(p).exists() and (Path(p) / "config.json").exists():
                    resolved_path = p
                    log(f"  Using model from run_config.json: {Path(p).name}")
            except Exception:
                pass
        if resolved_path == BASE_MODEL_NAME:
            log("  Using base model (no merged model found)")

    config = {
        "base_model_name":    str(resolved_path),
        "num_epochs":         10,
        "num_test_epochs":    2,
        "use_deepspeed":      False,
        "token_from_file":    False,
        "only_best_accuracy": False,
        "context_length":     8192,
        "max_input_length":   8192,
        "max_new_tokens":     16384,
    }

    if not dry_run:
        conf_path.parent.mkdir(parents=True, exist_ok=True)
        conf_path.write_text(json.dumps(config, indent=2))
        log(f"Wrote LLM conf → {conf_path}")
        log(f"  base_model_name: {resolved_path}")
    else:
        log(f"[DRY RUN] Would write LLM conf: {conf_path}")
        log(f"  base_model_name: {resolved_path}")

    return conf_name


def write_prompts(dataset: str, level: str, k: int, dry_run: bool = False) -> tuple[Path, Path]:
    """
    Write generation and training prompt JSON files.
    Adapts from proven CIFAR-10 prompts — never generates from scratch.
    """
    key = (level, k)
    if key not in PROVEN_PROMPTS:
        raise ValueError(
            f"No proven prompt for level={level} k={k}. Available: {list(PROVEN_PROMPTS.keys())}"
        )

    gen_source_name, train_source_name = PROVEN_PROMPTS[key]
    gen_source_path   = conf_test_dir / gen_source_name
    train_source_path = conf_train_dir / train_source_name

    if not gen_source_path.exists():
        raise FileNotFoundError(
            f"Proven generation prompt not found: {gen_source_path}\n"
            f"Run the CIFAR-10 curriculum first to create the proven prompts."
        )
    if not train_source_path.exists():
        raise FileNotFoundError(
            f"Proven training prompt not found: {train_source_path}"
        )

    gen_source   = json.loads(gen_source_path.read_text())
    train_source = json.loads(train_source_path.read_text())

    gen_name   = get_prompt_name(dataset, level, k, is_train=False)
    train_name = get_prompt_name(dataset, level, k, is_train=True)
    gen_path   = conf_test_dir / gen_name
    train_path = conf_train_dir / train_name

    gen_adapted   = adapt_prompt(gen_source,   dataset, level, k, is_train=False)
    train_adapted = adapt_prompt(train_source, dataset, level, k, is_train=True)

    if not dry_run:
        # force-write even if in gitignore
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        gen_path.write_text(json.dumps(gen_adapted,   indent=2))
        train_path.write_text(json.dumps(train_adapted, indent=2))
        log(f"Wrote generation prompt  → {gen_name}")
        log(f"Wrote training prompt    → {train_name}")
    else:
        log(f"[DRY RUN] Would write: {gen_name}")
        log(f"[DRY RUN] Would write: {train_name}")

    return gen_path, train_path


def write_entry_script(dataset: str, level: str, k: int,
                       llm_conf: str, dry_run: bool = False) -> Path:
    """
    Write a self-contained CurriculumGen entry script for this step.

    The script is written into the same directory as this pipeline
    (ab/gpt/act/curriculum/) so that the module path used to launch it,
    CURRICULUM_MODULE, always matches its location on disk.
    """
    dataset_safe = dataset.replace("-", "_")
    script_name  = f"CurriculumGen_{dataset_safe}_{level}_k{k}.py"

    curr_script_path = out_dir / 'curriculum'
    curr_script_path.mkdir(parents=True, exist_ok=True)
    script_path  = curr_script_path / script_name

    conf_id    = get_conf_id(dataset, level, k)
    gen_name   = get_prompt_name(dataset, level, k, is_train=False)
    train_name = get_prompt_name(dataset, level, k, is_train=True)
    nn_prefix  = get_nn_prefix(dataset, level, k)

    lines = [
        '"""',
        'Auto-generated by ab/gpt/act/curriculum/GenerationPipeline.py',
        f'Dataset: {dataset}  Level: {level}  k: {k}',
        f'Generated: {datetime.now().isoformat()}',
        '"""',
        'from trl import SFTConfig',
        'from peft import LoraConfig',
        'import ab.gpt.util.Tune_Curriculum as Tune_Curriculum',
        'from ab.gpt.util.Const import nngpt_dir',
        '',
        f'LLM_TUNE_CONF   = "{train_name}"',
        f'NN_GEN_CONF     = "{gen_name}"',
        f'NN_GEN_CONF_ID  = "{conf_id}"',
        f'LLM_CONF        = "{llm_conf}"',
        f'NN_NAME_PREFIX  = "{nn_prefix}"',
        'TEST_NN         = 20',
        'NN_TRAIN_EPOCHS = 1',
        'SKIP_EPOCHS     = 1',
        'MAX_NEW_TOKENS  = 16384',
        'MAX_PROMPTS     = 4096',
        'R               = 32',
        'LORA_ALPHA      = 32',
        'LORA_DROPOUT    = 0.05',
        'TARGET_MODULES  = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")',
        'TUNE_LAYERS     = range(0, 24)',
        'LEARNING_RATE   = 1e-6',
        'LR_SCHEDULER    = "cosine"',
        'PER_DEVICE_TRAIN_BATCH_SIZE = 1',
        'GRADIENT_ACCUMULATION_STEPS = 4',
        'WARMUP_RATIO    = 0.05',
        'MAX_GRAD_NORM   = 1.0',
        'LOGGING_STEPS   = 96',
        'OPTIMIZER       = "paged_adamw_8bit"',
        '',
        '',
        'def main():',
        '    layer_list = list(TUNE_LAYERS)',
        '    peft_config = LoraConfig(',
        '        r=R,',
        '        lora_alpha=LORA_ALPHA,',
        '        lora_dropout=LORA_DROPOUT,',
        '        target_modules=list(TARGET_MODULES),',
        '        layers_to_transform=layer_list,',
        '        bias="none",',
        '        task_type="CAUSAL_LM",',
        '    )',
        '    training_args = SFTConfig(',
        '        num_train_epochs=1,',
        '        learning_rate=LEARNING_RATE,',
        '        lr_scheduler_type=LR_SCHEDULER,',
        '        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,',
        '        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,',
        '        warmup_ratio=WARMUP_RATIO,',
        '        max_grad_norm=MAX_GRAD_NORM,',
        '        logging_steps=LOGGING_STEPS,',
        '        optim=OPTIMIZER,',
        '        output_dir=str(nngpt_dir / "outputs"),',
        '        save_strategy="no",',
        '        eval_strategy="no",',
        '        report_to="none",',
        '        gradient_checkpointing=True,',
        '        gradient_checkpointing_kwargs={"use_reentrant": False},',
        '        bf16=True,',
        '        padding_free=False,',
        '        packing_strategy="wrapped",  # bfd strategy auto-enables padding-free; wrapped does not',
        '        dataloader_pin_memory=False,',
        '    )',
        '    Tune_Curriculum.tune(',
        '        test_nn=TEST_NN,',
        '        nn_train_epochs=NN_TRAIN_EPOCHS,',
        '        skip_epoch=SKIP_EPOCHS,',
        '        llm_path=None,',
        '        llm_tune_conf=LLM_TUNE_CONF,',
        '        nn_gen_conf=NN_GEN_CONF,',
        '        conf_keys=(NN_GEN_CONF_ID,),',
        '        llm_conf=LLM_CONF,',
        '        training_args=training_args,',
        '        peft_config=peft_config,',
        '        max_prompts=MAX_PROMPTS,',
        '        save_llm_output=True,',
        '        max_new_tokens=MAX_NEW_TOKENS,',
        '        nn_name_prefix=NN_NAME_PREFIX,',
        '        temperature=1.0,',
        '        top_k=50,',
        '        top_p=0.9,',
        '        trans_mode=False,',
        '        prompt_batch=1,',
        '        use_agents=False,',
        '    )',
        '',
        '',
        'if __name__ == "__main__":',
        '    main()',
    ]

    content = "\n".join(lines) + "\n"

    if not dry_run:
        script_path.write_text(content)
        log(f"Wrote entry script → {script_path}")
    else:
        log(f"[DRY RUN] Would write entry script: {script_path}")

    return script_path


# ----- Merge and clean --------------------------------------------------------------------------------
def select_best_epoch(tracker_path: Path) -> Optional[tuple[int, float]]:
    """
    Read epoch_tracker.json and return (best_epoch_index, best_score).
    Returns None if tracker is empty or all scores are 0.
    """
    if not tracker_path.exists():
        return None
    try:
        data = json.loads(tracker_path.read_text())
        valid = [(i, e["score"]) for i, e in enumerate(data) if e.get("score", 0) > 0]
        if not valid:
            return None
        best_idx, best_score = max(valid, key=lambda x: x[1])
        best_epoch = data[best_idx]["epoch"]
        return best_epoch, best_score
    except Exception as e:
        log(f"Could not read epoch tracker: {e}", "WARN")
        return None


def run_merge(dry_run: bool = False) -> bool:
    """Run MergeLLM to merge the best adapter into the cumulative model."""
    if dry_run:
        log("[DRY RUN] Would run: python -m ab.gpt.util.MergeLLM")
        return True

    log("Running MergeLLM to merge best adapter...")
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    result = subprocess.run(
        [sys.executable, "-m", "ab.gpt.util.MergeLLM"],
        cwd=str(ab_root_path),
        env=env,
    )
    if result.returncode != 0:
        log("MergeLLM failed", "ERROR")
        return False
    log("Merge complete ✓")
    return True


def clean_epoch_dirs(tracker_backup_name: str, dry_run: bool = False) -> None:
    """
    Back up epoch_tracker.json and clean per-epoch outputs.
    Keeps the merged model in out/llm_to_upload/.
    """
    tracker_src = NNGPT_OUT / "epoch_tracker.json"
    tracker_dst = NNGPT_OUT / tracker_backup_name

    to_remove = [
        NNGPT_OUT / "llm" / "epoch",
        NNGPT_OUT / "cycle_results.json",
        NNGPT_OUT / "lineage.json",
    ]
    cycle_glob = list(NNGPT_OUT.glob("cycle_results_*.json"))

    if dry_run:
        log(f"[DRY RUN] Would backup epoch_tracker → {tracker_backup_name}")
        log("[DRY RUN] Would remove: epoch/, cycle_results*.json, lineage.json")
        return

    if tracker_src.exists():
        shutil.copy2(tracker_src, tracker_dst)
        log(f"Backed up epoch_tracker → {tracker_backup_name}")

    for path in to_remove:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            log(f"Removed dir: {path.name}/")
        elif path.exists():
            path.unlink()
            log(f"Removed: {path.name}")

    for p in cycle_glob:
        p.unlink()
        log(f"Removed: {p.name}")


# ----- Single step runner -----------------------------------------------------------------------------
def run_step(dataset: str, step: dict, progress: dict,
             resume: bool = False, dry_run: bool = False) -> bool:
    """
    Run one curriculum step end-to-end:
    viability check → write configs → run subprocess → merge → clean → update progress.

    Returns True if step completed successfully, False otherwise.
    """
    level = step["level"]
    k     = step["k"]
    band  = step["band"]
    key   = step_key(level, k)

    log("")
    log("=" * 60)
    log(f"Starting step: {key}  ({step['description']})")
    log("=" * 60)

    # ----- Check if already completed -----
    if key in progress.get("completed_steps", []):
        if resume:
            log(f"Step {key} already completed — skipping (--resume)")
            return True
        log(f"Step {key} already completed. Use --resume to skip or delete progress.json to restart.")
        return False

    # ----- Viability check -----
    cfg = DATASET_CONFIGS.get(dataset, {})
    viable_bands = cfg.get("viable_bands", [])

    if band not in viable_bands:
        log(f"Band '{band}' not viable for dataset '{dataset}' — skipping step {key}", "WARN")
        log(f"  Viable bands: {viable_bands}")
        log(f"  Reason: {cfg.get('note', 'insufficient anchor groups in DB')}")
        return True  # Not a failure — just skip non-viable steps

    log(f"Band '{band}' is viable for '{dataset}' ✓")

    # ----- DB row count check -----
    log(f"Checking DB viability for {dataset}/{band}/k={k}...")
    is_viable, row_count = check_band_viability(dataset, band, k)
    if not is_viable:
        log(f"Insufficient anchor groups ({row_count} rows, need ≥{k}) — skipping {key}", "WARN")
        return True  # Skip, not fail

    log(f"DB check: {row_count} rows available ✓")

    # ----- Write configuration files -----
    llm_conf = write_llm_conf(dataset, progress.get("current_merged_model", ""), dry_run)
    write_prompts(dataset, level, k, dry_run)
    write_entry_script(dataset, level, k, llm_conf, dry_run)

    dataset_safe = dataset.replace("-", "_")
    script_stem  = f"CurriculumGen_{dataset_safe}_{level}_k{k}"
    module_path  = f"{CURRICULUM_MODULE}.{script_stem}"

    # ----- Run curriculum fine-tuning -----
    log(f"Launching: python -m {module_path}")

    if dry_run:
        log(f"[DRY RUN] Would launch: python -m {module_path}")
        log(f"[DRY RUN] Step {key} configuration complete.")
        return True

    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env.pop("NNGPT_DIR_OVERRIDE", None)  # use default nngpt_dir for main curriculum

    t_start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", module_path],
        cwd=str(ab_root_path),
        env=env,
    )
    elapsed = (time.time() - t_start) / 3600

    if result.returncode != 0:
        log(f"Step {key} failed (return code {result.returncode}) after {elapsed:.1f}h", "ERROR")
        return False

    log(f"Step {key} training complete in {elapsed:.1f}h ✓")

    # ----- Select best epoch -----
    tracker_path = NNGPT_OUT / "epoch_tracker.json"
    best = select_best_epoch(tracker_path)
    if best:
        best_epoch, best_score = best
        log(f"Best epoch: A{best_epoch}  score={best_score:.4f}")
    else:
        log("No successful epochs found — step produced no valid models", "WARN")

    # ----- Merge best adapter -----
    if not run_merge(dry_run=False):
        log(f"Merge failed for step {key}", "ERROR")
        return False

    # ----- Back up tracker and clean -----
    tracker_backup = f"epoch_tracker_{key}.json"
    clean_epoch_dirs(tracker_backup, dry_run=False)

    # ----- Update progress -----
    merged_model_path = str(out_dir / "llm_to_upload" / "OlympicCoder-7B")
    progress["completed_steps"].append(key)
    progress["current_merged_model"] = merged_model_path
    progress.setdefault("step_results", {})[key] = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_hours": round(elapsed, 2),
        "best_epoch": best[0] if best else None,
        "best_score": best[1] if best else None,
        "tracker_backup": tracker_backup,
    }
    save_progress(dataset, progress)
    log(f"Progress saved — completed: {progress['completed_steps']}")

    return True


# ----- Full curriculum runner -------------------------------------------------------------------------
def run_curriculum(dataset: str, resume: bool = False, dry_run: bool = False) -> None:
    """
    Run the complete progressive curriculum for a dataset.
    Automatically determines viable steps, runs them in order,
    merges adapters between levels.
    """
    cfg = DATASET_CONFIGS.get(dataset)
    if cfg is None:
        log(f"Unknown dataset: '{dataset}'. Available: {list(DATASET_CONFIGS.keys())}", "ERROR")
        sys.exit(1)

    if not cfg.get("viable_bands"):
        log(f"Dataset '{dataset}' has no viable bands: {cfg.get('note')}", "ERROR")
        sys.exit(1)

    log("")
    log("Progressive Curriculum Fine-Tuning")
    log(f"Dataset:  {dataset}")
    log(f"Note:     {cfg['note']}")
    log(f"Viable bands: {cfg['viable_bands']}")
    log(f"Dry run:  {dry_run}")
    log(f"Resume:   {resume}")
    log(f"Root:     {ab_root_path}")
    log(f"Conf:     {conf_dir}")
    log(f"Out:      {out_dir}")
    log(f"Module:   {CURRICULUM_MODULE}")

    # Check base model exists — download if needed
    if not dry_run:
        ensure_model()

    progress = load_progress(dataset)

    # Determine viable steps for this dataset
    viable_steps = [
        step for step in CURRICULUM_SEQUENCE
        if step["band"] in cfg["viable_bands"]
    ]

    if not viable_steps:
        log(f"No viable curriculum steps for dataset '{dataset}'", "ERROR")
        sys.exit(1)

    log("")
    log(f"Curriculum plan ({len(viable_steps)} steps):")
    for i, step in enumerate(viable_steps, 1):
        key    = step_key(step["level"], step["k"])
        status = "✓ done" if key in progress.get("completed_steps", []) else "pending"
        log(f"  {i}. {key:<10} {step['band']:<16} {status}")

    log("")

    # Run each step in order
    total_start = time.time()
    failed_steps = []

    for step in viable_steps:
        ok = run_step(dataset, step, progress, resume=resume, dry_run=dry_run)
        if not ok:
            failed_steps.append(step_key(step["level"], step["k"]))
            log("Step failed — stopping curriculum.", "ERROR")
            break

    elapsed = (time.time() - total_start) / 3600
    log("")
    log("=" * 60)
    log(f"Curriculum complete in {elapsed:.1f}h")
    log(f"Completed steps: {progress.get('completed_steps', [])}")
    if failed_steps:
        log(f"Failed steps: {failed_steps}", "ERROR")
    log("=" * 60)

    print_results(dataset)


# ----- Single step runner (for cross-dataset experiments) ---------------------------------------------
def run_single_step(dataset: str, level: str, k: int,
                    resume: bool = False, dry_run: bool = False) -> None:
    """Run a single curriculum step — used for cross-dataset ablation experiments."""
    step = next(
        (s for s in CURRICULUM_SEQUENCE if s["level"] == level and s["k"] == k),
        None
    )
    if step is None:
        log(f"No curriculum step defined for level={level} k={k}", "ERROR")
        log(f"Available: {[(s['level'], s['k']) for s in CURRICULUM_SEQUENCE]}")
        sys.exit(1)

    progress = load_progress(dataset)
    ok = run_step(dataset, step, progress, resume=resume, dry_run=dry_run)
    if not ok:
        sys.exit(1)


# ----- Results display --------------------------------------------------------------------------------
def print_results(dataset: str) -> None:
    """Print a formatted summary of all completed curriculum steps."""
    progress = load_progress(dataset)
    results  = progress.get("step_results", {})

    if not results:
        log(f"No results found for dataset '{dataset}'")
        return

    log("")
    log(f"Results for dataset: {dataset}")
    log(f"{'Step':<12} {'Best Score':<12} {'Best Epoch':<12} {'Time (h)':<10} {'Completed'}")
    log("-" * 65)

    for step in CURRICULUM_SEQUENCE:
        key = step_key(step["level"], step["k"])
        if key not in results:
            continue
        r = results[key]
        log(
            f"{key:<12} "
            f"{str(r.get('best_score', 'N/A')):<12} "
            f"{str(r.get('best_epoch', 'N/A')):<12} "
            f"{str(r.get('elapsed_hours', 'N/A')):<10} "
            f"{r.get('completed_at', '')[:19]}"
        )

    # also check for saved trackers
    trackers = sorted(NNGPT_OUT.glob("epoch_tracker_*.json"))
    if trackers:
        log("")
        log("Saved epoch trackers:")
        for t in trackers:
            log(f"  {t.name}")


# ----- CLI --------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatic progressive curriculum fine-tuning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example command lines:
  # Full automatic curriculum — only dataset required
  python -m ab.gpt.act.curriculum.GenerationPipeline --dataset cifar-10

  # Dry run — preview without running
  python -m ab.gpt.act.curriculum.GenerationPipeline --dataset cifar-10 --dry_run

  # Resume after interruption
  python -m ab.gpt.act.curriculum.GenerationPipeline --dataset cifar-10 --resume

  # Single step for cross-dataset ablation
  python -m ab.gpt.act.curriculum.GenerationPipeline --dataset svhn --level L3 --k 2

  # Show results of completed curriculum
  python -m ab.gpt.act.curriculum.GenerationPipeline --dataset cifar-10 --show_results
        """
    )

    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to run curriculum on"
    )
    parser.add_argument(
        "--level", type=str, default=None,
        choices=["L1", "L2", "L3"],
        help="Run only this curriculum level (requires --k). "
             "Omit to run the full automatic curriculum."
    )
    parser.add_argument(
        "--k", type=int, default=None,
        choices=[2, 3, 4],
        help="Number of reference models (requires --level). "
             "Omit to run the full automatic curriculum."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-completed steps and resume from where the curriculum left off."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Preview what would happen without writing files or running training."
    )
    parser.add_argument(
        "--show_results", action="store_true",
        help="Print results of a completed or partial curriculum and exit."
    )
    parser.add_argument(
        "--show_paths", action="store_true",
        help="Print all resolved project paths and exit — useful for diagnosing layout issues."
    )

    args = parser.parse_args()

    if args.show_paths:
        log(f"This file:        {_THIS_FILE}")
        log(f"NNGPT_DIR:        {ab_root_path}")
        log(f"CONF_DIR:         {conf_dir}          exists={conf_dir.is_dir()}")
        log(f"LLM_CONF_DIR:     {conf_llm_dir}      exists={conf_llm_dir.is_dir()}")
        log(f"PROMPT_TEST_DIR:  {conf_test_dir}   exists={conf_test_dir.is_dir()}")
        log(f"PROMPT_TRAIN_DIR: {conf_train_dir}  exists={conf_train_dir.is_dir()}")
        log(f"OUT_DIR:          {out_dir}           exists={out_dir.is_dir()}")
        log(f"BASE_MODEL_PATH:  {BASE_MODEL_PATH}   exists={BASE_MODEL_PATH.is_dir()}")
        log(f"TOKENIZER_CACHE:  {TOKENIZER_CACHE}   exists={TOKENIZER_CACHE.is_dir()}")
        log(f"CURRICULUM_DIR:   {CURRICULUM_DIR}")
        log(f"CURRICULUM_MODULE:{CURRICULUM_MODULE}")
        return

    if args.show_results:
        print_results(args.dataset)
        return

    if args.level is not None and args.k is not None:
        # Single step mode — for cross-dataset ablation experiments
        log(f"Single step mode: dataset={args.dataset} level={args.level} k={args.k}")
        run_single_step(
            dataset=args.dataset,
            level=args.level,
            k=args.k,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    elif args.level is not None or args.k is not None:
        log("--level and --k must be provided together, or both omitted for full curriculum.", "ERROR")
        sys.exit(1)
    else:
        # Full automatic curriculum mode
        run_curriculum(
            dataset=args.dataset,
            resume=args.resume,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()