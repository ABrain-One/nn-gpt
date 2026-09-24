# NNGPT-CL: Closed-Loop Neural Network Generation — Usage Guide

This guide covers how to run every experiment described in the NNGPT-CL paper using
the scripts in this directory.  All commands are run from the **repo root** (`nn-gpt/`).

---

## Quick-start: Phase 3 closed-loop (the main experiment)

```bash
python -m ab.gpt.TuneNNGen \
    --llm_conf ds_coder_7b_olympic.json \
    --nn_gen_conf_id improve_classification_only \
    --suppress_thinking \
    --num_cycles 3 \
    --nn_name_prefix nngpt \
    --start_layer 20 --end_layer 28
```

- `--suppress_thinking` closes OlympicCoder-7B's auto-opened `<think>` block immediately,
  reducing the thinking-to-code token ratio from 147× to 14.7× (paper Section 3.2).
- `--start_layer 20 --end_layer 28` selects the struct20 LoRA target (layers 20–27,
  23 M trainable params, 0.53 %).  Paper Section 4.2.
- `--num_cycles 3` reproduces Experiment A (CIFAR-10, 3-cycle, Table 3).

---

## Experiment A — CIFAR-10, 3-cycle (Table 3)

```bash
python -m ab.gpt.TuneNNGen \
    --llm_conf ds_coder_7b_olympic_3ep.json \
    --nn_gen_conf_id improve_classification_only \
    --suppress_thinking \
    --num_cycles 3 \
    --nn_name_prefix nngpt_expA \
    --start_layer 20 --end_layer 28 \
    --temperature 0.6 --top_k 50
```

## Experiment B — CIFAR-10, 6-cycle (Table 3)

```bash
python -m ab.gpt.TuneNNGen \
    --llm_conf ds_coder_7b_olympic.json \
    --nn_gen_conf_id improve_classification_only \
    --suppress_thinking \
    --num_cycles 6 \
    --nn_name_prefix nngpt_expB \
    --start_layer 20 --end_layer 28 \
    --temperature 0.6 --top_k 50
```

## Experiment C — CIFAR-100, 6-cycle (Table 3)

```bash
python -m ab.gpt.TuneNNGen \
    --llm_conf ds_coder_7b_olympic.json \
    --nn_gen_conf_id improve_classification_only \
    --suppress_thinking \
    --num_cycles 6 \
    --nn_name_prefix nngpt_expC \
    --start_layer 20 --end_layer 28 \
    --temperature 0.6 --top_k 50
    # pass the dataset-specific conf key for CIFAR-100
```

---

## Phase 1 — Token budget sweep (Table 2)

Phase 1 measures LEMUR format compliance vs. output token budget.
Each run is generation-only (no fine-tuning), one cycle, with a fixed output cap.

```bash
for BUDGET in 512 1024 2048 4096 6144; do
    python -m ab.gpt.TuneNNGen \
        --llm_conf ds_coder_7b_olympic.json \
        --nn_gen_conf_id improve_classification_only \
        --suppress_thinking \
        --max_output_tokens $BUDGET \
        --num_cycles 1 \
        --skip_lm_finetune \
        --nn_name_prefix "phase1_mnist_${BUDGET}"
done
```

Key flags:
- `--max_output_tokens N`  — sets the output token cap and injects a budget-forcing prompt.
- `--skip_lm_finetune`     — skips LoRA fine-tuning; generation-only run.
- `--num_cycles 1`          — single generate→evaluate cycle.

---

## Phase 2 (exploratory) — lm_head-only fine-tuning

This was an early experiment targeting only the output head.
Result: 0 % extraction rate (Finding F1 in project summary).

```bash
python -m ab.gpt.TuneHeadersNNGen \
    --llm_conf ds_coder_7b_olympic.json \
    --nn_gen_conf_id improve_classification_only \
    --num_train_epochs 2 \
    --learning_rate 5e-6 \
    --r 16 --lora_alpha 16 \
    --nn_name_prefix header
```

`TuneHeadersNNGen.py` is a standalone entrypoint with defaults tuned for lm_head-only
LoRA (rank 16, 5×10⁻⁶ LR, 2 epochs).

---

## Phase 2 (exploratory) — prefix / soft-prompt tuning

Trains only virtual-token embeddings; zero weight changes to the base model.
Result: below LEMUR median (prefix20 in project summary).

```bash
python -m ab.gpt.TuneNNGen \
    --llm_conf ds_coder_7b_olympic.json \
    --nn_gen_conf_id improve_classification_only \
    --use_prefix_tuning \
    --num_virtual_tokens 20 \
    --num_cycles 6 \
    --nn_name_prefix prefix20
```

---

## Phase 2 (exploratory) — struct26 (catastrophic forgetting reference)

Targets only layers 26–27 (≈6 M params, 0.14 %).  Produces catastrophic forgetting
by cycle A4 (paper Section 4.2).  Kept here for reproducibility only.

```bash
python -m ab.gpt.TuneNNGen \
    --llm_conf ds_coder_7b_olympic.json \
    --nn_gen_conf_id improve_classification_only \
    --suppress_thinking \
    --num_cycles 6 \
    --nn_name_prefix struct26 \
    --start_layer 26 --end_layer 28
```

---

## LoRA configuration reference (paper Table 4)

| Config   | `--start_layer` | `--end_layer` | Params  | Notes                       |
|----------|-----------------|---------------|---------|------------------------------|
| struct20 | 20              | 28            | ~23 M   | Production config (Phase 3) |
| struct26 | 26              | 28            | ~6 M    | Catastrophic forgetting A4  |
| lm_head  | —               | —             | ~2.5 M  | Use TuneHeadersNNGen.py      |

---

## Configuration files

Both files live in `ab/gpt/conf/llm/` and are passed via `--llm_conf`.

| File                              | Purpose                                      |
|-----------------------------------|----------------------------------------------|
| `ds_coder_7b_olympic.json`        | OlympicCoder-7B base config (no epoch override) |
| `ds_coder_7b_olympic_3ep.json`    | Experiment A config — 3 training epochs, 2 test epochs |

### `ds_coder_7b_olympic_3ep.json` field reference

```json
{
  "base_model_name": "open-r1/OlympicCoder-7B",
  "num_epochs": 3,
  "num_test_epochs": 2,
  "use_deepspeed": false,
  "token_from_file": false,
  "only_best_accuracy": false,
  "context_length": 4096,
  "max_input_length": 4096,
  "max_new_tokens": 6144
}
```

| Field               | Description                                                                                 |
|---------------------|---------------------------------------------------------------------------------------------|
| `base_model_name`   | HuggingFace model id loaded by the ChatBot                                                  |
| `num_epochs`        | LoRA fine-tuning epochs per cycle (overridden at runtime by `--num_cycles`)                 |
| `num_test_epochs`   | Eval epochs used during validation at the end of each cycle                                 |
| `use_deepspeed`     | Enable DeepSpeed ZeRO for multi-GPU training; `false` for single-GPU runs                  |
| `token_from_file`   | Load HuggingFace auth token from a file instead of the environment variable                 |
| `only_best_accuracy`| Retain only the checkpoint with the highest eval accuracy across cycles                     |
| `context_length`    | Maximum sequence length for the model's attention window                                    |
| `max_input_length`  | Truncation limit for the prompt portion of each sequence                                    |
| `max_new_tokens`    | Default output token cap; overridden at runtime by `--max_output_tokens`                    |

`ds_coder_7b_olympic.json` omits `num_epochs` (uses the pipeline default) and sets `default_context_length: 8192` for full-context runs.

---

## New flags at a glance

| Flag                     | Default | Description                                                 |
|--------------------------|---------|-------------------------------------------------------------|
| `--suppress_thinking`    | off     | Closes `<think>` block before generation (OlympicCoder-7B) |
| `--max_output_tokens N`  | None    | Token budget override + budget-forcing prompt injection     |
| `--num_cycles N`         | config  | Number of generate→evaluate→finetune cycles                 |
| `--skip_lm_finetune`     | off     | Generation-only; skips LoRA update each cycle               |
| `--use_prefix_tuning`    | off     | Soft-prompt (prefix) tuning instead of LoRA                 |
| `--num_virtual_tokens N` | 20      | Virtual token count for prefix tuning                       |

All flags are opt-in with safe defaults; omitting any of them reproduces the
original pipeline behaviour.

---

## Static validity checking

`ab/gpt/util/eval/static_validity.py` is an AST-based structural checker that validates
generated `new_nn.py` files without executing any code.  It serves as a fast
pre-filter between generation and the (expensive) GPU evaluation step, decoupling
generation quality from infra failures such as stale tmp paths or LEMUR duplicates.

### Usage

```python
from ab.gpt.util.eval.static_validity import check_static_validity, batch_validity

# Single file
result = check_static_validity("out/llm/epoch/A0/synth_nn/B0/new_nn.py")
print(result.valid, result.score, result.reason)

# Entire batch directory (returns a JSON-safe dict)
info = batch_validity("out/llm/epoch/A0/synth_nn/B0/")
```

### What it checks

The LEMUR contract requires a specific file structure:

| Element                        | Location            | Score contribution |
|-------------------------------|---------------------|--------------------|
| `supported_hyperparameters()` | module-level function | +0.15            |
| `class Net(nn.Module)`        | top-level class     | +0.10              |
| `Net.__init__`                | method of Net       | +0.10              |
| `Net.train_setup`             | method of Net       | +0.20              |
| `Net.learn`                   | method of Net       | +0.20              |
| `Net.forward`                 | method of Net       | +0.15              |
| Non-trivial `learn` body      | ≥1 real statement   | +0.05              |
| Minimum length                | ≥10 lines           | +0.05              |

`result.valid` is `True` only when all eight checks pass.  `result.score` (0.0–1.0)
gives partial credit for analysis; a syntax error returns score 0.1.

### `ValidityResult` fields

```python
result.valid          # bool — all checks passed
result.score          # float 0.0–1.0 partial-credit score
result.reason         # str — first failing check or "structurally_valid"
result.has_net_class  # bool
result.has_supported_hyperparameters  # bool
result.has_forward_or_learn           # bool
result.has_nontrivial_body            # bool
result.parse_error    # str or None — set on SyntaxError
result.nn_chars       # int — source character count
result.nn_lines       # int — source line count
```

---

## NN cost utilities

`ab/gpt/util/eval/nn_cost.py` computes parameter count and FLOPs (MACs) for a generated
`new_nn.py` without running training.  Used by the analysis scripts to report
model complexity alongside accuracy.

**Requires `torch`; optional backends: `fvcore`, `ptflops`, `thop`.**
Returns `(None, None)` on any failure — safe to call in bulk.

### Usage

```python
from ab.gpt.util.eval.nn_cost import get_nn_cost, batch_cost

# Returns (param_count: int|None, flops_macs: int|None)
params, flops = get_nn_cost("out/llm/epoch/A0/synth_nn/B0/new_nn.py", dataset="cifar-10")

# JSON-serialisable dict — convenient for logging
info = batch_cost("out/llm/epoch/A0/synth_nn/B0/new_nn.py", dataset="mnist")
# {"param_count": 123456, "flop_count": 7890000}
```

### Supported datasets

| Dataset key(s)                  | Input shape (C×H×W) | Classes |
|---------------------------------|----------------------|---------|
| `mnist`                         | 1×28×28              | 10      |
| `svhn`                          | 3×32×32              | 10      |
| `cifar-10` / `cifar10`          | 3×32×32              | 10      |
| `cifar-100` / `cifar100`        | 3×32×32              | 100     |
| `celeba-gender` / `celeba_gender` | 3×128×128          | 2       |
| `imagenette`                    | 3×224×224            | 10      |
| `places365`                     | 3×256×256            | 365     |

### FLOPs backend fallback chain

`get_nn_cost` tries three FLOPs backends in order, returning the first that succeeds:

1. **fvcore** (`FlopCountAnalysis`) — preferred; suppresses unsupported-op warnings
2. **ptflops** (`get_model_complexity_info`) — fallback if fvcore is absent
3. **thop** (`profile`) — last resort

Install any one to enable FLOPs reporting:
```bash
pip install fvcore          # recommended
# or: pip install ptflops
# or: pip install thop
```

---

## Output directory layout

```
out_<prefix>/
└── llm/
    └── epoch/
        ├── A0/          ← cycle 0 (cold-start, base model)
        │   ├── synth_nn/
        │   │   ├── B0/  ← probe 0
        │   │   │   ├── new_nn.py
        │   │   │   ├── full_output.txt
        │   │   │   └── 1.json
        │   │   └── ...
        │   └── <adapter>/  ← LoRA weights after cycle 0
        ├── A1/          ← cycle 1 (fine-tuned once)
        └── ...
```

---

## See also

- `ab/gpt/util/ANALYSIS_SCRIPTS.md` — output-size and epoch analysis tools.
- `ab/gpt/TuneNNGen.py` — main entrypoint (all flags documented with `--help`).
- `ab/gpt/TuneHeadersNNGen.py` — lm_head-only fine-tuning entrypoint.
- `ab/gpt/util/eval/static_validity.py` — AST-based structural checker (no execution).
- `ab/gpt/util/eval/nn_cost.py` — parameter count and FLOPs for generated NNs.
- `ab/gpt/conf/llm/ds_coder_7b_olympic_3ep.json` — OlympicCoder-7B 3-epoch config (Experiment A).
