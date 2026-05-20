"""
CurriculumGenerationPipeline.py
--------------------------------------------------------------------------------
Multi-dataset comparison and generation pipeline for curriculum fine-tuning.

Each dataset gets its own independent fine-tuning branch starting from
the original OlympicCoder-7B base model.


Supported datasets:
    cifar-10      ← default (CIFAR-10 curriculum)
    cifar-100     ← same 32×32, 3-channel, easiest transfer
    svhn          ← same 32×32, 3-channel
    imagenette    ← 224×224, 3-channel, best fit for TorchVision backbones

Usage:
    # Run single dataset
    python CurriculumGenerationPipeline.py --dataset cifar-100 --level L1 --k 2

    # Run full comparison (all datasets, L1 k=2 only)
    python CurriculumGenerationPipeline.py --compare_all --level L1 --k 2

    # Resume interrupted run
    python CurriculumGenerationPipeline.py --dataset cifar-100 --level L1 --k 2 --resume

    # Dry run — print config only
    python CurriculumGenerationPipeline.py --dataset cifar-100 --level L1 --k 2 --dry_run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
# __file__ is at nn-gpt/ab/gpt/CurriculumGenerationPipeline.py
# parents[2] resolves to the nn-gpt project root
NNGPT_DIR    = Path(__file__).parents[2].resolve()
OUT_DIR      = NNGPT_DIR / "out"

# ── Base model (original) ────────────────────────────────────
BASE_MODEL_PATH = OUT_DIR / "llm" / "open-r1" / "OlympicCoder-7B"

# ── Dataset configurations ────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "cifar-10": {
        "task":        "img-classification",
        "metric":      "acc",
        "transform":   "norm_256_flip",
        "in_channels": 3,
        "input_size":  32,
        "dummy_size":  224,
        "viable":      True,
        "viable_bands": ["high", "medium", "very_low_near"],
        "note":        "13023 NNs, 2634 acc>=0.85 — main curriculum",
    },
    "svhn": {
        "task":        "img-classification",
        "metric":      "acc",
        "transform":   "norm_256_flip",
        "in_channels": 3,
        "input_size":  32,
        "dummy_size":  224,
        "viable":      True,
        "viable_bands": ["very_low_near"],
        "note":        "4568 NNs, 1671 acc>=0.85 — L3 only",
    },
    "celeba-gender": {
        "task":        "img-classification",
        "metric":      "acc",
        "transform":   "norm_256_flip",
        "in_channels": 3,
        "input_size":  224,
        "dummy_size":  224,
        "viable":      True,
        "viable_bands": ["very_low_near"],
        "note":        "3719 NNs, 2171 acc>=0.85 — L3 only",
    },
    "mnist": {
        "viable":      False,
        "note":        "only 4 rows in very_low_near — too few",
    },
    "cifar-100": {
        "viable":      False,
        "note":        "0 models with acc>=0.85",
    },
    "imagenette": {
        "viable":      False,
        "note":        "only 25 models with acc>=0.85",
    },
    "coco": {
        "viable":      False,
        "note":        "no MinHash vectors in DB",
    },
}
# ── Curriculum level configurations ──────────────────────────────────────────
LEVEL_CONFIGS = {
    "L1": {
        "band":        "high",
        "jaccard_min": 0.95,
        "jaccard_max": 0.98,
        "description": "High-similarity references — establishes basic LEMUR pattern",
    },
    "L2": {
        "band":        "medium",
        "jaccard_min": 0.85,
        "jaccard_max": 0.95,
        "description": "Medium-similarity references — combines moderately diverse patterns",
    },
    "L3": {
        "band":        "very_low_near",
        "jaccard_min": 0.30,
        "jaccard_max": 0.60,
        "description": "Very-low-near references — synthesises from architecturally diverse models",
    },
}
# ── Utility functions ─────────────────────────────────────────────────────────

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}", flush=True)


def get_branch_dir(dataset: str, level: str, k: int) -> Path:
    """Return the isolated output directory for this branch."""
    branch_name = f"{dataset.replace('-', '_')}_{level}_k{k}"
    return OUT_DIR / "compare" / branch_name


def get_branch_model_dir(dataset: str, level: str, k: int) -> Path:
    """Return the merged model path for this branch."""
    return get_branch_dir(dataset, level, k) / "merged_model"


def get_conf_id(dataset: str, level: str, k: int) -> str:
    dataset_safe = dataset.replace("-", "_")
    return f"curriculum_{dataset_safe}_{level.lower()}_k{k}"


def get_prompt_name(dataset: str, level: str, k: int, is_train: bool = False) -> str:
    dataset_safe = dataset.replace("-", "_")
    suffix = "_train" if is_train else ""
    return f"Curriculum_{dataset_safe}_{level}_k{k}{suffix}.json"


# ── Prompt generation ─────────────────────────────────────────────────────────

def build_generation_prompt(dataset: str, level: str, k: int) -> dict:
    """
    Build generation prompt by adapting the proven CIFAR-10 prompt
    for the target dataset. Not generating  prompts from scratch.
    """
    cfg = DATASET_CONFIGS[dataset]

    # ── find the proven CIFAR-10 prompt for this level/k ─────────────────
    # Map level+k to the known-good prompt file
    PROVEN_PROMPTS = {
        ("L1", 2): "Curriculum_L1_high_k2.json",
        ("L2", 2): "Curriculum_L2_medium_k2.json",
        ("L2", 3): "Curriculum_L2_medium_k3.json",
        ("L3", 2): "Curriculum_L3_very_low_near_k2.json",
        ("L3", 3): "Curriculum_L3_very_low_near_k3.json",
        ("L3", 4): "Curriculum_L3_very_low_near_k4.json",
    }

    key = (level, k)
    if key not in PROVEN_PROMPTS:
        raise ValueError(
            f"No proven prompt for level={level} k={k}. "
            f"Available: {list(PROVEN_PROMPTS.keys())}"
        )

    prompt_test_dir = NNGPT_DIR / "ab" / "gpt" / "conf" / "prompt" / "test"
    source_path = prompt_test_dir / PROVEN_PROMPTS[key]

    if not source_path.exists():
        raise FileNotFoundError(
            f"Proven prompt not found: {source_path}\n"
            f"Run the CIFAR-10 curriculum first before comparison experiments."
        )

    # ── load proven prompt ────────────────────────────────────────────────
    source_cfg = json.loads(source_path.read_text())
    source_conf_id = list(source_cfg.keys())[0]
    source_conf = source_cfg[source_conf_id]

    # ── adapt for new dataset ─────────────────────────────────────────────
    conf_id = get_conf_id(dataset, level, k)
    new_conf = dict(source_conf)  # copy all proven settings

    # override dataset-specific fields
    new_conf["dataset"] = dataset
    new_conf["task"] = cfg["task"]
    new_conf["metric"] = cfg["metric"]

    # adapt prompt lines that reference dataset-specific values
    adapted_prompt = []
    for line in source_conf["prompt"]:
        # fix dataset name in the generation instruction line
        line = line.replace("'cifar-10'", f"'{dataset}'")
        line = line.replace("dataset='cifar-10'", f"dataset='{dataset}'")

        # fix transform if different
        if "norm_256_flip" in line and cfg["transform"] != "norm_256_flip":
            line = line.replace("norm_256_flip", cfg["transform"])

        # fix dummy tensor size if different input channels
        if "torch.zeros(1, C, 224, 224)" in line and cfg["dummy_size"] != 224:
            line = line.replace(
                "torch.zeros(1, C, 224, 224)",
                f"torch.zeros(1, C, {cfg['dummy_size']}, {cfg['dummy_size']})"
            )

        adapted_prompt.append(line)

    new_conf["prompt"] = adapted_prompt
    new_conf["is_generation"] = True
    new_conf["output"] = []

    return {conf_id: new_conf}


def build_training_prompt(dataset: str, level: str, k: int) -> dict:
    PROVEN_TRAIN_PROMPTS = {
        ("L1", 2): "Curriculum_L1_high_k2_train.json",
        ("L2", 2): "Curriculum_L2_medium_k2_train.json",
        ("L2", 3): "Curriculum_L2_medium_k3_train.json",
        ("L3", 2): "Curriculum_L3_very_low_near_k2_train.json",
        ("L3", 3): "Curriculum_L3_very_low_near_k3_train.json",
        ("L3", 4): "Curriculum_L3_very_low_near_k4_train.json",
    }

    key = (level, k)
    if key not in PROVEN_TRAIN_PROMPTS:
        raise ValueError(f"No proven training prompt for level={level} k={k}.")

    prompt_train_dir = NNGPT_DIR / "ab" / "gpt" / "conf" / "prompt" / "train"
    source_path = prompt_train_dir / PROVEN_TRAIN_PROMPTS[key]

    if not source_path.exists():
        raise FileNotFoundError(f"Proven training prompt not found: {source_path}")

    source_cfg = json.loads(source_path.read_text())
    source_conf_id = list(source_cfg.keys())[0]
    source_conf = source_cfg[source_conf_id]

    conf_id = get_conf_id(dataset, level, k)
    cfg = DATASET_CONFIGS[dataset]
    new_conf = dict(source_conf)

    new_conf["dataset"] = dataset
    new_conf["task"] = cfg["task"]
    new_conf["metric"] = cfg["metric"]

    adapted_prompt = []
    for line in source_conf["prompt"]:
        line = line.replace("'cifar-10'", f"'{dataset}'")
        if "norm_256_flip" in line and cfg["transform"] != "norm_256_flip":
            line = line.replace("norm_256_flip", cfg["transform"])
        adapted_prompt.append(line)

    new_conf["prompt"] = adapted_prompt
    new_conf["is_generation"] = False

    return {conf_id: new_conf}


def write_prompts(dataset: str, level: str, k: int, dry_run: bool = False) -> tuple:
    """Write generation and training prompt JSON files. Returns (gen_path, train_path)."""
    prompt_test_dir  = NNGPT_DIR / "ab" / "gpt" / "conf" / "prompt" / "test"
    prompt_train_dir = NNGPT_DIR / "ab" / "gpt" / "conf" / "prompt" / "train"

    gen_name   = get_prompt_name(dataset, level, k, is_train=False)
    train_name = get_prompt_name(dataset, level, k, is_train=True)
    gen_path   = prompt_test_dir  / gen_name
    train_path = prompt_train_dir / train_name

    if not dry_run:
        prompt_test_dir.mkdir(parents=True, exist_ok=True)
        prompt_train_dir.mkdir(parents=True, exist_ok=True)
        gen_config   = build_generation_prompt(dataset, level, k)
        train_config = build_training_prompt(dataset, level, k)
        gen_path.write_text(json.dumps(gen_config, indent=2))
        train_path.write_text(json.dumps(train_config, indent=2))
        log(f"Wrote generation prompt  → {gen_path.name}")
        log(f"Wrote training prompt    → {train_path.name}")
    else:
        log(f"[DRY RUN] Would write: {gen_path.name}")
        log(f"[DRY RUN] Would write: {train_path.name}")

    return gen_path, train_path


def write_curriculum_gen_script(dataset: str, level: str, k: int,
                                branch_dir: Path, dry_run: bool = False) -> Path:
    """Write a CurriculumGen entry-point script for this branch."""
    dataset_safe = dataset.replace("-", "_")
    script_name  = f"CurriculumGen_{dataset_safe}_{level}_k{k}.py"
    script_path  = NNGPT_DIR / "ab" / "gpt" / script_name
    conf_id      = get_conf_id(dataset, level, k)
    gen_name     = get_prompt_name(dataset, level, k, is_train=False)
    train_name   = get_prompt_name(dataset, level, k, is_train=True)
    prefix       = f"{dataset_safe}_{level.lower()}_k{k}"
    llm_conf     = f"ds_coder_7b_olympic_ft_{dataset_safe}.json"

    # Generate a CurriculumGen module (same pattern as CurriculumGen_L1_k2 etc.)
    module_name  = f"CurriculumGen_{dataset_safe}_{level}_k{k}_module"
    module_file  = script_name  # reuse same file — contains both module + entry

    script_lines = [
        f'"""',
        f'Auto-generated by CurriculumGenerationPipeline.py',
        f'Dataset: {dataset}  Level: {level}  k: {k}',
        f'Generated: {datetime.now().isoformat()}',
        f'"""',
        f'import copy',
        f'from trl import SFTConfig',
        f'from peft import LoraConfig',
        f'import ab.gpt.util.Tune_Curriculum as Tune_Curriculum',
        f'from ab.gpt.util.Const import nngpt_dir',
        f'',
        f'LLM_TUNE_CONF  = "{train_name}"',
        f'NN_GEN_CONF    = "{gen_name}"',
        f'NN_GEN_CONF_ID = "{conf_id}"',
        f'LLM_CONF       = "{llm_conf}"',
        f'NN_NAME_PREFIX = "{prefix}"',
        f'TEST_NN        = 10',
        f'NN_TRAIN_EPOCHS = 1',
        f'SKIP_EPOCHS    = 1',
        f'MAX_NEW_TOKENS = 16384',
        f'MAX_PROMPTS    = 4096',
        f'R              = 32',
        f'LORA_ALPHA     = 32',
        f'LORA_DROPOUT   = 0.05',
        f'TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")',
        f'TUNE_LAYERS    = range(0, 24)',
        f'LEARNING_RATE  = 1e-6',
        f'LR_SCHEDULER   = "cosine"',
        f'PER_DEVICE_TRAIN_BATCH_SIZE = 1',
        f'GRADIENT_ACCUMULATION_STEPS = 4',
        f'WARMUP_RATIO   = 0.05',
        f'MAX_GRAD_NORM  = 1.0',
        f'LOGGING_STEPS  = 96',
        f'OPTIMIZER      = "paged_adamw_8bit"',
        f'',
        f'',
        f'def main():',
        f'    layer_list = list(TUNE_LAYERS)',
        f'    peft_config = LoraConfig(',
        f'        r=R,',
        f'        lora_alpha=LORA_ALPHA,',
        f'        lora_dropout=LORA_DROPOUT,',
        f'        target_modules=list(TARGET_MODULES),',
        f'        layers_to_transform=layer_list,',
        f'        bias="none",',
        f'        task_type="CAUSAL_LM",',
        f'    )',
        f'    training_args = SFTConfig(',
        f'        num_train_epochs=1,',
        f'        learning_rate=LEARNING_RATE,',
        f'        lr_scheduler_type=LR_SCHEDULER,',
        f'        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,',
        f'        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,',
        f'        warmup_ratio=WARMUP_RATIO,',
        f'        max_grad_norm=MAX_GRAD_NORM,',
        f'        logging_steps=LOGGING_STEPS,',
        f'        optim=OPTIMIZER,',
        f'        output_dir=str(nngpt_dir / "outputs"),',
        f'        save_strategy="no",',
        f'        eval_strategy="no",',
        f'        report_to="none",',
        f'        gradient_checkpointing=True,',
        f'        gradient_checkpointing_kwargs={{"use_reentrant": False}},',
        f'        bf16=True,',
        f'        dataloader_pin_memory=False,',
        f'    )',
        f'    Tune_Curriculum.tune(',
        f'        test_nn=TEST_NN,',
        f'        nn_train_epochs=NN_TRAIN_EPOCHS,',
        f'        skip_epoch=SKIP_EPOCHS,',
        f'        llm_path=None,',
        f'        llm_tune_conf=LLM_TUNE_CONF,',
        f'        nn_gen_conf=NN_GEN_CONF,',
        f'        conf_keys=(NN_GEN_CONF_ID,),',
        f'        llm_conf=LLM_CONF,',
        f'        training_args=training_args,',
        f'        peft_config=peft_config,',
        f'        max_prompts=MAX_PROMPTS,',
        f'        save_llm_output=True,',
        f'        max_new_tokens=MAX_NEW_TOKENS,',
        f'        nn_name_prefix=NN_NAME_PREFIX,',
        f'        temperature=1.0,',
        f'        top_k=50,',
        f'        top_p=0.9,',
        f'        trans_mode=False,',
        f'        prompt_batch=1,',
        f'        use_agents=False,',
        f'    )',
        f'',
        f'',
        f'if __name__ == "__main__":',
        f'    main()',
    ]
    content = "\n".join(script_lines) + "\n"

    if not dry_run:
        script_path.write_text(content)
        log(f"Wrote entry script → {script_path.name}")
    else:
        log(f"[DRY RUN] Would write entry script: {script_path.name}")
    return script_path


def write_llm_conf(dataset: str, base_model_path: Path, dry_run: bool = False) -> Path:
    """Write a dataset-specific LLM conf pointing to the original base model."""
    dataset_safe = dataset.replace("-", "_")
    conf_name    = f"ds_coder_7b_olympic_ft_{dataset_safe}.json"
    conf_path    = NNGPT_DIR / "ab" / "gpt" / "conf" / "llm" / conf_name

    # Use short model name so LLM.py correctly resolves
    # tokenizer from out/tokenizer/open-r1/OlympicCoder-7B
    # which has chat_template.jinja
    config = {
        "base_model_name":              "open-r1/OlympicCoder-7B",
        "num_epochs":                   100,
        "num_test_epochs":              2,
        "use_deepspeed":                False,
        "token_from_file":              False,
        "only_best_accuracy":           False,
        "context_length":               8192,
        "max_input_length":             8192,
        "max_new_tokens":               16384,
    }

    if not dry_run:
        conf_path.parent.mkdir(parents=True, exist_ok=True)
        conf_path.write_text(json.dumps(config, indent=2))
        log(f"Wrote LLM conf → {conf_name}")
        log(f"  base_model: {base_model_path}")
    else:
        log(f"[DRY RUN] Would write LLM conf: {conf_name}")
        log(f"  base_model: {base_model_path}")

    return conf_path


# ── Branch setup ──────────────────────────────────────────────────────────────

def setup_branch(dataset: str, level: str, k: int,
                 resume: bool = False, dry_run: bool = False) -> Path:
    """
    Set up an isolated branch directory for this dataset/level/k experiment.
    Returns the branch directory path.
    """
    branch_dir = get_branch_dir(dataset, level, k)

    if branch_dir.exists() and not resume:
        log(f"Branch already exists: {branch_dir}", "WARN")
        log("Use --resume to continue an interrupted run.", "WARN")
        sys.exit(1)

    if not dry_run:
        branch_dir.mkdir(parents=True, exist_ok=True)
        (branch_dir / "nngpt").mkdir(exist_ok=True)
        (branch_dir / "nngpt" / "llm").mkdir(exist_ok=True)

        # Write branch metadata
        meta = {
            "dataset":      dataset,
            "level":        level,
            "k":            k,
            "created_at":   datetime.now().isoformat(),
            "base_model":   str(BASE_MODEL_PATH),
            "band":         LEVEL_CONFIGS[level]["band"],
            "description":  LEVEL_CONFIGS[level]["description"],
        }
        (branch_dir / "branch_meta.json").write_text(json.dumps(meta, indent=2))
        log(f"Branch created: {branch_dir}")
    else:
        log(f"[DRY RUN] Would create branch: {branch_dir}")

    return branch_dir


# ── Results collection ────────────────────────────────────────────────────────

def collect_branch_results(branch_dir: Path) -> dict:
    """Collect epoch tracker results from a branch directory."""
    tracker_path = branch_dir / "nngpt" / "epoch_tracker.json"
    if not tracker_path.exists():
        return {"status": "no_results", "epochs": []}

    try:
        data = json.loads(tracker_path.read_text())
        valid = [e for e in data if e.get("score", 0) > 0]
        best = max(valid, key=lambda x: x["score"]) if valid else None
        return {
            "status":       "complete" if data else "empty",
            "epochs":       data,
            "best_epoch":   best,
            "total_epochs": len(data),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "epochs": []}


def print_comparison_table(results: dict) -> None:
    """Print a summary comparison table across all branches."""
    print("\n" + "=" * 80)
    print("CURRICULUM COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Branch':<35} {'Best Acc':>10} {'Success':>10} {'Score':>10} {'Epoch':>8}")
    print("-" * 80)

    for branch_key, res in sorted(results.items()):
        best = res.get("best_epoch")
        if best:
            print(
                f"{branch_key:<35} "
                f"{best.get('accuracy', 0):.4f}     "
                f"{best.get('success_rate', 0)*100:5.1f}%     "
                f"{best.get('score', 0):.4f}     "
                f"A{best.get('epoch', '?')}"
            )
        else:
            print(f"{branch_key:<35} {'--':>10} {'--':>10} {'--':>10} {'--':>8}")

    print("=" * 80 + "\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_branch(dataset: str, level: str, k: int,
               resume: bool = False, dry_run: bool = False) -> None:
    """Full pipeline for one dataset/level/k branch."""
    log(f"Starting branch: dataset={dataset}  level={level}  k={k}")
    log(f"Band: {LEVEL_CONFIGS[level]['band']} — {LEVEL_CONFIGS[level]['description']}")

    # Viability check

    cfg = DATASET_CONFIGS.get(dataset, {})
    if not cfg.get("viable", True):
        log(f"Dataset '{dataset}' is not viable: {cfg.get('note', 'insufficient models')}", "ERROR")
        sys.exit(1)

    # 1. Validate dataset
    if dataset not in DATASET_CONFIGS:
        log(f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_CONFIGS.keys())}", "ERROR")
        sys.exit(1)

    # 2. Validate base model exists
    if not BASE_MODEL_PATH.exists() and not dry_run:
        log(f"Base model not found: {BASE_MODEL_PATH}", "ERROR")
        log("The original OlympicCoder-7B must exist at out/llm/open-r1/OlympicCoder-7B/", "ERROR")
        sys.exit(1)

    # 3. Setup branch directory
    branch_dir = setup_branch(dataset, level, k, resume=resume, dry_run=dry_run)

    # 4. Write LLM conf pointing to original base model
    write_llm_conf(dataset, BASE_MODEL_PATH, dry_run=dry_run)

    # 5. Write prompt files
    write_prompts(dataset, level, k, dry_run=dry_run)

    # 6. Write entry script
    script_path = write_curriculum_gen_script(dataset, level, k, branch_dir, dry_run=dry_run)

    if dry_run:
        log("[DRY RUN] Configuration complete — no files written or training started.")
        script_stem = f"CurriculumGen_{dataset.replace("-", "_")}_{level}_k{k}"
        log(f"[DRY RUN] To run: python -m ab.gpt.{script_stem}")
        return

    # 7. Run the curriculum generation
    script_stem = script_path.stem
    log(f"Launching: python -m ab.gpt.{script_stem}")
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["NNGPT_DIR_OVERRIDE"] = str(branch_dir / "nngpt")  # ← add this line

    result = subprocess.run(
        [sys.executable, "-m", f"ab.gpt.{script_path.stem}"],
        cwd=str(NNGPT_DIR),
        env=env,
    )
    if result.returncode != 0:
        log(f"Branch failed with return code {result.returncode}", "ERROR")
        sys.exit(result.returncode)

    # 8. Collect and print results
    res = collect_branch_results(branch_dir)
    log(f"Branch complete. Status: {res['status']}")
    if res.get("best_epoch"):
        best = res["best_epoch"]
        log(
            f"Best: A{best['epoch']} — "
            f"acc={best.get('accuracy'):.4f}  "
            f"success={best.get('success_rate')*100:.1f}%  "
            f"score={best.get('score'):.4f}"
        )

    # 9. Save branch result summary
    summary_path = branch_dir / "results_summary.json"
    summary_path.write_text(json.dumps(res, indent=2, default=str))
    log(f"Results saved → {summary_path}")


def run_compare_all(level: str, k: int,
                    resume: bool = False, dry_run: bool = False) -> None:
    """Run all datasets sequentially and print comparison table."""
    datasets = ["cifar-100", "svhn", "imagenette"]
    all_results = {}

    log(f"Starting comparison run: level={level}  k={k}")
    log(f"Datasets: {datasets}")
    log("Note: CIFAR-10 results come from the main curriculum run (not re-run here).")

    for dataset in datasets:
        log(f"\n{'='*60}")
        log(f"Running: {dataset} / {level} / k={k}")
        log(f"{'='*60}")

        try:
            run_branch(dataset, level, k, resume=resume, dry_run=dry_run)
            branch_dir = get_branch_dir(dataset, level, k)
            res = collect_branch_results(branch_dir)
            all_results[f"{dataset}_{level}_k{k}"] = res
        except SystemExit as e:
            log(f"Branch {dataset} failed: {e}", "ERROR")
            all_results[f"{dataset}_{level}_k{k}"] = {"status": "failed"}

    # Add CIFAR-10 results from existing tracker if available
    cifar10_tracker = OUT_DIR / "nngpt" / "epoch_tracker_L1.json"
    if cifar10_tracker.exists():
        try:
            data = json.loads(cifar10_tracker.read_text())
            valid = [e for e in data if e.get("score", 0) > 0]
            best = max(valid, key=lambda x: x["score"]) if valid else None
            all_results[f"cifar-10_{level}_k{k} (main run)"] = {
                "status": "loaded",
                "best_epoch": best,
                "epochs": data,
            }
        except Exception:
            pass

    # Print comparison table
    print_comparison_table(all_results)

    # Save full comparison
    compare_path = OUT_DIR / "compare" / f"comparison_{level}_k{k}_{timestamp()}.json"
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    compare_path.write_text(json.dumps(all_results, indent=2, default=str))
    log(f"Full comparison saved → {compare_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-dataset comparison pipeline for curriculum fine-tuning."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="Target dataset for this branch.",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="L1",
        choices=list(LEVEL_CONFIGS.keys()),
        help="Curriculum level (default: L1).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of reference models (default: 2).",
    )
    parser.add_argument(
        "--compare_all",
        action="store_true",
        help="Run all datasets sequentially and print comparison table.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run without overwriting existing branch.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print config and file paths without writing or running anything.",
    )
    parser.add_argument(
        "--show_results",
        action="store_true",
        help="Show results from existing branches without running.",
    )

    args = parser.parse_args()

    # Show existing results
    if args.show_results:
        compare_dir = OUT_DIR / "compare"
        if not compare_dir.exists():
            log("No comparison results found. Run the pipeline first.", "WARN")
            sys.exit(0)

        all_results = {}
        for branch_dir in sorted(compare_dir.iterdir()):
            if not branch_dir.is_dir():
                continue
            res = collect_branch_results(branch_dir)
            all_results[branch_dir.name] = res
        print_comparison_table(all_results)
        sys.exit(0)

    # Compare all datasets
    if args.compare_all:
        run_compare_all(args.level, args.k, resume=args.resume, dry_run=args.dry_run)
        return

    # Single dataset
    if not args.dataset:
        parser.error("--dataset is required unless --compare_all or --show_results is set.")

    run_branch(args.dataset, args.level, args.k, resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    main()