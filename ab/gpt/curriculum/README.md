# CurriculumGenerationPipeline

Fully automatic progressive curriculum fine-tuning pipeline for LLM-based neural architecture synthesis within the LEMUR framework. The pipeline implements MinHash-scheduled similarity-band ordering: each curriculum level presents architecturally closer references first, allowing the LLM to internalise evaluator-compatible coding patterns before encountering structurally distant designs.

---

## Overview

The pipeline runs six fine-tuning steps in a fixed progressive order:

```
L1 (high band, k=2)
  └─ L2 (medium band, k=2)
       └─ L2 (medium band, k=3)
            └─ L3 (very-low-near band, k=2)
                 └─ L3 (very-low-near band, k=3)
                      └─ L3 (very-low-near band, k=4)
```

Each step: checks DB viability → writes configs → runs LoRA fine-tuning subprocess → merges the best adapter into the cumulative backbone → cleans temporary epoch outputs → advances to the next level.

---

## Requirements

- Python 3.12+
- PyTorch 2.9+ with CUDA
- `transformers`, `peft`, `trl`, `bitsandbytes`
- LEMUR `nn-dataset` repository at `../nn-dataset` relative to this project
- OlympicCoder-7B base model at `out/llm/open-r1/OlympicCoder-7B/`
- LEMUR database with MinHash signatures precomputed

---

## Quick Start

```bash
# Full curriculum — CIFAR-10 only (takes ~7 days on RTX 4090)
python -m ab.gpt.curriculum.CurriculumGenerationPipeline \
    --dataset cifar-10

# Preview what would run without executing anything
python -m ab.gpt.curriculum.CurriculumGenerationPipeline \
    --dataset cifar-10 --dry_run

# Resume after interruption
python -m ab.gpt.curriculum.CurriculumGenerationPipeline \
    --dataset cifar-10 --resume

# Single cross-dataset level (e.g. SVHN L3)
python -m ab.gpt.curriculum.CurriculumGenerationPipeline \
    --dataset svhn --level L3 --k 2

# Show results summary of a completed or partial run
python -m ab.gpt.curriculum.CurriculumGenerationPipeline \
    --dataset cifar-10 --show_results
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | required | Dataset to run. See supported datasets below. |
| `--level` | str | None | Run only this level (`L1`, `L2`, `L3`). Must be combined with `--k`. |
| `--k` | int | None | Number of reference models (2, 3, or 4). Must be combined with `--level`. |
| `--resume` | flag | False | Skip already-completed steps and continue from last checkpoint. |
| `--dry_run` | flag | False | Print the execution plan without writing any files or running training. |
| `--show_results` | flag | False | Print results summary of completed steps and exit. |

---

## Supported Datasets

| Dataset | Viable bands | Models | Notes |
|---|---|---|---|
| `cifar-10` | high, medium, very-low-near | 13,023 | Full curriculum — only fully viable dataset |
| `svhn` | very-low-near (L3) only | 4,568 | `unq`-family anchor, max J=0.82 |
| `celeba-gender` | none | 3,719 | Not viable — 166 high-J models all identical (`ga-352`) |
| `mnist` | none | 3,315 | Not viable — max pairwise J=0.23 |
| `cifar-100` | none | 3,170 | Not viable — no models above 72.9% accuracy |

Dataset viability is checked automatically at runtime against the LEMUR database. Non-viable steps are skipped without error.

---

## File Structure

```
nn-gpt/
├── ab/gpt/curriculum/
│   └── CurriculumGenerationPipeline.py   ← this script
├── ab/gpt/conf/
│   ├── llm/
│   │   └── ds_coder_7b_olympic_ft_{dataset}.json   ← auto-written per step
│   └── prompt/
│       ├── test/
│       │   └── Curriculum_{dataset}_{Level}_k{k}.json   ← generation prompts
│       └── train/
│           └── Curriculum_{dataset}_{Level}_k{k}_train.json   ← training prompts
├── ab/gpt/
│   └── CurriculumGen_{dataset}_{Level}_k{k}.py   ← auto-written entry scripts
└── out/
    ├── llm/open-r1/OlympicCoder-7B/   ← base model (never modified)
    ├── llm_to_upload/OlympicCoder-7B/  ← cumulative merged model (updated each step)
    ├── curriculum/cifar-10/
    │   └── progress.json               ← tracks completed steps and current model path
    └── nngpt/
        ├── epoch_tracker.json          ← live tracker for current step
        ├── epoch_tracker_{step}.json   ← backed-up tracker per completed step
        └── llm/epoch/                  ← per-epoch adapter outputs (cleaned after merge)
```

---

## How It Works

### 1. Band viability check

Before running each step, the pipeline queries the LEMUR database to confirm that enough anchor groups exist for the requested dataset, similarity band, and reference count k. If fewer than k rows are available, the step is skipped automatically.

### 2. Config generation

Three files are written automatically per step:

**LLM conf** (`ds_coder_7b_olympic_ft_{dataset}.json`) — points to the current merged model. For the first step this is the base OlympicCoder-7B; for subsequent steps it is the `llm_to_upload` merged model from the previous step.

**Prompt JSONs** — adapted from the proven CIFAR-10 prompt templates. For CIFAR-10 only the dataset name is substituted. For other datasets, architecture-specific constraints (backbone whitelist, input size, output classes, batch size, transform) are replaced with dataset-appropriate values from `DATASET_CONFIGS`.

**Entry script** (`CurriculumGen_{dataset}_{Level}_k{k}.py`) — a self-contained fine-tuning script with the LoRA configuration, SFTConfig, and `Tune_Curriculum.tune()` call.

### 3. Fine-tuning subprocess

The entry script is launched as a subprocess. It runs `SKIP_EPOCHS=1` (skips epoch 0 generation, starts fine-tuning immediately), then alternates between generation and fine-tuning for up to `num_epochs` epochs. Results are written to `out/nngpt/epoch_tracker.json`.

### 4. Best epoch selection and merge

After the subprocess completes, the pipeline reads `epoch_tracker.json` and selects the epoch with the highest composite score (`SR × best_accuracy`). `MergeLLM` is then called to merge that epoch's LoRA adapter into `out/llm_to_upload/OlympicCoder-7B` using a dequantize–merge–requantize pipeline (NF4 → float16 → NF4).

### 5. Progress tracking

`out/curriculum/{dataset}/progress.json` records completed steps and the current merged model path. This file enables `--resume` to skip already-completed steps correctly. The epoch tracker is backed up as `out/nngpt/epoch_tracker_{step}.json` before being cleared for the next step.

---

## Curriculum Sequence Configuration

The six steps and their epoch budgets are defined in `CURRICULUM_SEQUENCE`:

```python
CURRICULUM_SEQUENCE = [
    {"level": "L1", "band": "high",          "k": 2, "epochs": 10, ...},
    {"level": "L2", "band": "medium",        "k": 2, "epochs": 8,  ...},
    {"level": "L2", "band": "medium",        "k": 3, "epochs": 12, ...},
    {"level": "L3", "band": "very_low_near", "k": 2, "epochs": 10, ...},
    {"level": "L3", "band": "very_low_near", "k": 3, "epochs": 7,  ...},
    {"level": "L3", "band": "very_low_near", "k": 4, "epochs": 12, ...},
]
```

The order is immutable — each level builds on the merged adapter of all previous levels. Do not change the order without understanding the curriculum dependency chain.

---

## Dataset Configuration

Each dataset entry in `DATASET_CONFIGS` controls both pipeline behaviour and prompt content:

```python
"cifar-10": {
    "task":             "img-classification",
    "metric":           "acc",
    "transform":        "norm_256_flip",
    "viable_bands":     ["high", "medium", "very_low_near"],
    "anchor_family":    "rl-bb-init",
    "use_pretrained":   True,
    "backbone_whitelist": ["resnet50", "densenet169", ...],
    "out_classes":      10,
    "batch_hint":       16,
    "input_size":       256,
    "arch_notes": [
        "Use at most two backbone models total — never three.",
        ...
    ],
}
```

The `arch_notes` list is injected directly into the generation prompt, replacing CIFAR-10-specific backbone constraints when adapting to a different dataset. This ensures the LLM receives architecturally correct instructions for each dataset's model family.

---

## Prompt Adaptation

`adapt_prompt()` handles two cases:

**CIFAR-10** — trivial substitution of dataset name only. All structural constraints are inherited from the proven CIFAR-10 templates unchanged.

**Other datasets** — lines containing CIFAR-10 backbone markers (`"Only use these backbone models:"`, `"Always use weights='DEFAULT'"`, `"norm_256_flip"`, etc.) are identified by `_is_backbone_line()` and replaced with the dataset-specific `arch_notes`. Transform name, input size, output class count, and batch size are also substituted.

---

## Resuming After Interruption

If the pipeline crashes or is stopped mid-step:

```bash
# Resume from last completed step
python -m ab.gpt.curriculum.CurriculumGenerationPipeline \
    --dataset cifar-10 --resume
```

If `progress.json` was accidentally deleted, recreate it manually:

```python
from ab.gpt.curriculum.CurriculumGenerationPipeline import save_progress
from pathlib import Path

save_progress("cifar-10", {
    "completed_steps": ["L1_k2", "L2_k2", "L2_k3", "L3_k2", "L3_k3"],
    "current_merged_model": str(
        Path("out/llm_to_upload/OlympicCoder-7B").resolve()
    ),
})
```

Then run with `--resume`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | Set before running to reduce CUDA OOM errors on RTX 4090 |

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

---

## Common Issues

**`ValueError: Unrecognized model in llm_to_upload/OlympicCoder-7B`**
The `llm_to_upload` directory is empty or lacks `config.json`. Copy the base model files into it first:
```bash
cp -r out/llm/open-r1/OlympicCoder-7B/* out/llm_to_upload/OlympicCoder-7B/
```

**`ValueError: tokenizer.chat_template is not set`**
The tokenizer at the current model path lacks a chat template. Copy `tokenizer_config.json` from a previously merged model that has one.

**`Step L1_k2 already completed. Use --resume to skip`**
Always add `--resume` when restarting a partially completed run.

**`No results found for dataset 'cifar-10'`**
`progress.json` exists but `step_results` is empty — the pipeline has not completed any steps yet, or `progress.json` was manually recreated without results. This is not an error; it means no steps have fully completed including the merge.

**MergeLLM fails after fine-tuning**
Check that `out/llm_to_upload/OlympicCoder-7B/config.json` exists and is valid JSON. The merge always reads and writes to this directory.

---

## Experimental Results (CIFAR-10, seed 2, N=15, no repair)

| Level | Band | k | Best SR | Mean SR | Score |
|---|---|---|---|---|---|
| L1 | High (0.95–0.98) | 2 | 60% | 40% | 0.578 |
| L2 | Medium (0.85–0.95) | 2 | 33% | 23% | 0.323 |
| L2 | Medium (0.85–0.95) | 3 | 40% | 27% | 0.388 |
| L3 | Low/VL-near (0.30–0.85) | 2 | 53%* | 37%* | 0.516* |
| L3 | Low/VL-near (0.30–0.85) | 3 | 33% | 25% | 0.322 |
| L3 | Low/VL-near (0.30–0.85) | 4 | 33% | 20% | 0.321 |

\* L3 k=2 uses two-step partial interface repair (`fix_param_usage`). All other levels use no repair.

Score = SR × best accuracy. See paper for full per-epoch trajectories.
