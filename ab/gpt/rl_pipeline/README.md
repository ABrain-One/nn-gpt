# SFT/RL Reproduction Guide

This document gives the commands needed to reproduce the SFT + full-evaluation
RL experiments. It uses public model IDs and user-provided output paths only.

## 1. Environment

```bash
export RUN_ROOT=<RUN_ROOT>

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
```

Example:

```bash
export RUN_ROOT=/tmp/nn-gpt-runs
```

Run the commands from an NNGPT checkout or from an environment where NNGPT is
installed. NN Dataset is an NNGPT dependency and does not need to be downloaded
or added to `PYTHONPATH` separately.

The installed `nn-dataset` database must contain these source architecture
families:

| Experiment | Source prefixes |
| --- | --- |
| 1-pattern | `rl-bb-test1` |
| 4-pattern | `rl-bb-struct1,rl-bb-struct1-v2` |

## 2. Data Split

There is no runtime parameter to choose the split protocol in these experiments.
The code fixes it to:

```text
trainvaltest
```

This means:

| Dataset | Train | Reward/eval validation | Held-out test |
| --- | --- | --- | --- |
| `cifar-10` | 45k from CIFAR-10 train | 5k from CIFAR-10 train | CIFAR-10 test |
| `cifar-100` | 45k from CIFAR-100 train | 5k from CIFAR-100 train | CIFAR-100 test |
| `imagenette` | 7500 from Imagenette train | 1969 from Imagenette train | Imagenette validation/test split |

Implementation location: `ab/gpt/TuneRLSft.py` sets `SFT_EVAL_SPLIT_PROTOCOL =
"trainvaltest"`, and `ab/gpt/util/DatasetSplit.py` implements the
`trainvaltest` split.

## 3. Main SFT Adapters

The main paper-style comparison uses DeepSeek-Coder only.

### 1-Pattern DeepSeek SFT

```bash
python -m ab.gpt.TuneBackbone \
  --llm_conf backbone_sft_config.json \
  --sft_nn_prefixes rl-bb-test1 \
  --gen_nn_prefix rl-bb-test1-dscoder7b-sftcycle \
  --epoch_root "$RUN_ROOT/sft/1pattern/deepseek"
```

Select the best SFT adapter using SFT-cycle statistics only: formal success
count, mean formal accuracy, max/top-k formal accuracy, then structural
diversity. The selected adapter path has this form:

```text
$RUN_ROOT/sft/1pattern/deepseek/A<cycle>/deepseek-ai/deepseek-coder-6.7b-instruct
```

### 4-Pattern DeepSeek SFT

```bash
python -m ab.gpt.TuneBackbone \
  --llm_conf backbone_sft_config.json \
  --sft_nn_prefixes rl-bb-struct1,rl-bb-struct1-v2 \
  --gen_nn_prefix rl-bb-struct1-v2-dscoder7b-sftcycle \
  --epoch_root "$RUN_ROOT/sft/4pattern/deepseek"
```

Select the best 4-pattern SFT adapter with the same SFT-only rule. The selected
adapter path has this form:

```text
$RUN_ROOT/sft/4pattern/deepseek/A<cycle>/deepseek-ai/deepseek-coder-6.7b-instruct
```

## 4. Main 5-Epoch RL Runs

Run six main RL jobs:

| Pattern | Adapter | Dataset | Seeds |
| --- | --- | --- | --- |
| 1-pattern | selected DeepSeek 1-pattern SFT adapter | `cifar-10` | `42`, `123`, `777` |
| 4-pattern | selected DeepSeek 4-pattern SFT adapter | `cifar-10` | `42`, `123`, `777` |

Set the adapter path and pattern first:

```bash
export PATTERN=<1pattern|4pattern>
export ADAPTER_DIR=<SELECTED_ADAPTER_DIR>
export SEED=<42|123|777>
```

Example for 1-pattern:

```bash
export PATTERN=1pattern
export ADAPTER_DIR=$RUN_ROOT/sft/1pattern/deepseek/A9/deepseek-ai/deepseek-coder-6.7b-instruct
export SEED=42
```

Example for 4-pattern:

```bash
export PATTERN=4pattern
export ADAPTER_DIR=$RUN_ROOT/sft/4pattern/deepseek/A18/deepseek-ai/deepseek-coder-6.7b-instruct
export SEED=42
```

Launch RL:

```bash
export RUN_ID="${PATTERN}_deepseek_cifar10_seed${SEED}"

export NNGPT_SFT_BASE_MODEL_ID=deepseek-ai/deepseek-coder-6.7b-instruct
export NNGPT_SFT_INIT_ADAPTER="$ADAPTER_DIR"
export NNGPT_SFT_LOAD_INITIAL_ADAPTER=1
export NNGPT_SFT_INITIAL_ADAPTER_MODE=trainable

export NNGPT_RL_FORMAL_DATASET=cifar-10
export NNGPT_RL_FORMAL_REWARD_EPOCHS=5
export NNGPT_RL_SEED="$SEED"
export NNGPT_SFT_MAX_STEPS=100

if [ "$PATTERN" = "4pattern" ]; then
  export NNGPT_SFT_RL_NN_PREFIXES=rl-bb-struct1,rl-bb-struct1-v2
else
  export NNGPT_SFT_RL_NN_PREFIXES=rl-bb-test1
fi

export NNGPT_SFT_LOG_DIR="$RUN_ROOT/rl/$RUN_ID/rl_output"
export NNGPT_SFT_MODEL_OUT="$RUN_ROOT/rl/$RUN_ID/model"
export NNGPT_SFT_TRAINER_OUT="$RUN_ROOT/rl/$RUN_ID/trainer"
export NNGPT_SFT_EPOCH_ROOT="$RUN_ROOT/rl/$RUN_ID/epoch_sft"

python -m ab.gpt.TuneRLSft
```

This produces an 800-sample target per run: 8 generations per step for 100
steps. The primary output is:

```text
$NNGPT_SFT_LOG_DIR/generation_samples.jsonl
```

## 5. Auxiliary 3x3 Cross-Dataset Diagnostic

The 3x3 diagnostic uses these three model families:

```text
dscoder
qwen
olympic
```

It does not use Mistral.

The nine source adapters are:

| Model family | RL reward dataset |
| --- | --- |
| `dscoder` | `cifar-10`, `cifar-100`, `imagenette` |
| `qwen` | `cifar-10`, `cifar-100`, `imagenette` |
| `olympic` | `cifar-10`, `cifar-100`, `imagenette` |

Use the 1-pattern source prefix for all nine runs:

```bash
export NNGPT_SFT_RL_NN_PREFIXES=rl-bb-test1
```

Base model IDs:

| Model family | Base model ID |
| --- | --- |
| `dscoder` | `deepseek-ai/deepseek-coder-6.7b-instruct` |
| `qwen` | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `olympic` | `open-r1/OlympicCoder-7B` |

For each model family and reward dataset, run the RL command from Section 4
with:

```bash
export PATTERN=1pattern
export NNGPT_SFT_BASE_MODEL_ID=<BASE_MODEL_ID>
export NNGPT_RL_FORMAL_DATASET=<cifar-10|cifar-100|imagenette>
export NNGPT_RL_FORMAL_REWARD_EPOCHS=1
export NNGPT_SFT_MAX_STEPS=100
```

Example row:

```bash
export PATTERN=1pattern
export NNGPT_SFT_BASE_MODEL_ID=deepseek-ai/deepseek-coder-6.7b-instruct
export NNGPT_RL_FORMAL_DATASET=cifar-10
export NNGPT_RL_FORMAL_REWARD_EPOCHS=1
export NNGPT_SFT_MAX_STEPS=100
```

After each of the nine RL runs, generate 30 candidates from that RL adapter and
evaluate the same 30 candidates on:

```text
cifar-10
cifar-100
imagenette
```

The 3x3 diagnostic therefore has 9 candidate sources and 3 held-out evaluation
datasets.

## 6. Dataset Shapes

| Dataset | Output shape |
| --- | --- |
| `cifar-10` | `(10,)` |
| `imagenette` | `(10,)` |
| `cifar-100` | `(100,)` |

## 7. Metrics

Use `generation_samples.jsonl` as the record for each RL or evaluation run.
Report:

| Quantity | JSONL field |
| --- | --- |
| sample count | number of JSONL rows |
| formal success | `api_result.formal_success_candidate` |
| accuracy | `api_result.formal_horizon_test_acc["5"]` for the main 5-epoch runs; `["1"]` for 1-epoch diagnostics |
| positive reward | `reward > 0` |
| build status | `api_result.built_ok` |
| graph diversity | `api_result.signature` |
| backbone diversity | `api_result.backbone_signature` |

Quick summary:

```bash
export RESULT_JSONL=<generation_samples.jsonl>
export ACC_EPOCH=<1|5>

# Example:
# export RESULT_JSONL=$RUN_ROOT/rl/1pattern_deepseek_cifar10_seed42/rl_output/generation_samples.jsonl
# export ACC_EPOCH=5

python - "$RESULT_JSONL" "$ACC_EPOCH" <<'PY'
import json, math, statistics, sys
from collections import Counter

path, epoch = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(path) if line.strip()]
ok = [r for r in rows if r.get("api_result", {}).get("formal_success_candidate")]
acc = [
    float(r["api_result"]["formal_horizon_test_acc"][epoch])
    for r in ok
    if r.get("api_result", {}).get("formal_horizon_test_acc", {}).get(epoch) is not None
]
pos = sum(1 for r in rows if float(r.get("reward") or 0) > 0)
graphs = Counter(r.get("api_result", {}).get("signature") for r in ok)
backs = Counter(r.get("api_result", {}).get("backbone_signature") for r in ok)

def eff(counter):
    n = sum(counter.values())
    return 0.0 if n == 0 else math.exp(-sum((c/n) * math.log(c/n) for c in counter.values()))

print(f"rows={len(rows)}")
print(f"formal_success={len(ok)}/{len(rows)}")
print(f"positive_reward={pos}/{len(rows)}")
print(f"mean_acc={statistics.mean(acc):.4f}" if acc else "mean_acc=NA")
print(f"max_acc={max(acc):.4f}" if acc else "max_acc=NA")
print(f"backbone_unique={len(backs)} backbone_eff={eff(backs):.2f}")
print(f"graph_unique={len(graphs)} graph_eff={eff(graphs):.2f}")
PY
```

Also check stdout/stderr for OOM, timeout, traceback, killed process, and worker
restart messages. A partial JSONL file is not a completed run.
