"""Configuration: constants, hyperparameters, paths, prompts and experiment toggles.

Single source of truth. Runtime-mutable toggles (USE_*, ACTIVE_PROXY_NAMES,
MODEL_FALLBACKS) are read elsewhere as ``config.<NAME>`` so a write here -- e.g. by
the regen_*/run_step* experiment scripts -- is seen across the whole package.
"""
from __future__ import annotations

from ab.nn.util.Const import out_dir



ACC_DIR = out_dir / "acc_predict"

RAW_INPUT_PATH = ACC_DIR / "llm_finetuning_data.jsonl"
RAW_CSV_PATH = ACC_DIR / "llm_finetuning_data.csv"
DEFAULT_TRAIN_PATH = ACC_DIR / "train_llm_dataset.jsonl"
DEFAULT_VAL_PATH = ACC_DIR / "val_llm_dataset.jsonl"
DEFAULT_TEST_PATH = ACC_DIR / "test_llm_dataset.jsonl"
DEFAULT_OUTPUT_DIR = ACC_DIR / "tuned_model"
DEFAULT_TEST_OUTPUT_PATH = ACC_DIR / "test_predictions.csv"
DEFAULT_TEST_METRICS_PATH = ACC_DIR / "test_metrics.log"
TEST_MAX_NEW_TOKENS = 64
TEST_TEMPERATURE = 0.0

# Zero-cost proxies (see out/IMPROVEMENT_PLAN_Zero_Cost_Proxies.md, Step 3).
# Explicit toggle rather than "use it if the files happen to exist": makes
# the A/B ablation (baseline vs +proxies) an intentional, visible switch,
# and fails loudly if the cache is missing while this is True rather than
# silently falling back to the baseline prompt (which would look like a
# proxy run but secretly not be one).
# Toggled True here for the Fisher-added arm (see
# out/IMPROVEMENT_PLAN_Additional_Proxies.md) -- testing Fisher (the
# strongest single proxy found in the transfer-check, r=0.532) added to the
# current best config (2-epoch, no-code, 5 proxies: R^2=0.820/0.056), one
# variable at a time per the discipline established after the 3-epoch +
# proxies negative result.
USE_ZERO_COST_PROXIES = True
PROXY_CACHE_PATH = ACC_DIR / "proxy_cache.jsonl"
PROXY_NORM_STATS_PATH = ACC_DIR / "proxy_norm_stats.json"

# Network-statistics feature (see out/IMPROVEMENT_PLAN_NN_Stat_Feature.md).
# Adds the single most-predictive nn_stat structural statistic that is NOT
# already captured by the proxies. Feature selection (ab/gpt: select_stat
# analysis) ranked features by |Spearman| with dataset-adjusted best_accuracy
# on the train family only; FLOPs was the strongest novel, continuous,
# cross-family-transferable predictor (rho=+0.130, only 0.71-correlated with
# log_params so it carries genuinely new information). Cache holds the
# train-only z-scored log1p(FLOPs) per architecture.
USE_NN_STATS = False
USE_LAYER_STATS = False
NN_STATS_CACHE_PATH = ACC_DIR / "nn_stat_flops_cache.json"

# Which proxies actually appear in the prompt -- a SUBSET of everything in
# the cache (zero_cost_proxies.PROXY_NAMES now has 8: the original 5 plus
# SNIP/Fisher/GraSP, see IMPROVEMENT_PLAN_Additional_Proxies.md). Every prior
# proxy experiment implicitly used all of PROXY_NAMES; now that the cache has
# more fields than any experiment has tested, this explicit subset is
# required to keep experiments a genuine one-variable-at-a-time comparison.
# The SWAP test (depth -> fisher) was tried but underperformed: the fisher
# variant scored 0.758 vs the depth set's 0.820 (Experiment 7 / index line 27),
# confirming the regression was not about signal COUNT. The original 5-proxy
# set with `depth` is the best-accuracy proxy configuration (0.820) and is the
# set the published ABrain/Accuracy-Prediction checkpoint was trained with, so
# inference must inject this same set. See IMPROVEMENT_PLAN_Additional_Proxies.md.
ACTIVE_PROXY_NAMES = ("synflow", "nwot", "grad_norm", "log_params", "depth")

# Reset to False here: the 3-early-epoch experiment (see
# out/IMPROVEMENT_PLAN_Three_Early_Epochs.md) isolates ONLY the epoch-count
# variable against the early-signals-only baseline, so every other extra
# signal stays off.
USE_SEMANTIC_RETRIEVAL = False
SEMANTIC_RETRIEVAL_K = 3

# Source code in the prompt (see out/IMPROVEMENT_PLAN_Early_Signals_Only.md).
# The paper's own third comparison arm ("Qwen3 Early Signals Only" vs.
# "Qwen3 + Source Code", Table 2) was never implemented in this codebase --
# this toggle adds it. False = omit the NEURAL_NETWORK_CODE block entirely,
# leaving only early accuracy + task/dataset/metric/budget metadata.
USE_SOURCE_CODE = False

# Third early-training-signal epoch (see out/IMPROVEMENT_PLAN_Three_Early_Epochs.md).
# The paper's own problem formulation (s = s1, s2, s3) uses THREE early
# epochs; this codebase originally only extracted/used two (a deviation
# flagged in the project's first analysis). Requires Stage 1
# (data_preprocessing) to have been re-run with _DP_EARLY_EPOCHS including 3.
# Reset to False for the Fisher-added test (out/IMPROVEMENT_PLAN_Additional_Proxies.md)
# -- comparing against the 2-epoch + 5-proxy baseline specifically (R^2=0.820/0.056),
# not the 3-epoch one, to keep this a genuine single-variable test.
USE_THIRD_EARLY_EPOCH = False

# Chain-of-thought reasoning output (see out/IMPROVEMENT_PLAN_Chain_Of_Thought.md).
# When True, the assistant target becomes a short input-grounded reasoning
# trace followed by the same JSON answer, and the model is asked to reason
# before answering -- giving it more effective compute per prediction rather
# than more input fields (which the "signal capacity ceiling" has repeatedly
# shown backfire). Tested against the 2-epoch + 5-proxy reference
# (R^2=0.820/0.056) as a single-variable change: everything else identical.
USE_CHAIN_OF_THOUGHT = False

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MODEL_FALLBACKS = ("unsloth/Qwen3-8B-bnb-4bit", "Qwen/Qwen3-8B")
DEFAULT_MAX_SEQ_LEN = 6144

PREDICTOR_HF_REPO = "ABrain/Accuracy-Prediction"
PREDICTOR_MAX_SEQ_LEN = DEFAULT_MAX_SEQ_LEN
PREDICTOR_MAX_NEW_TOKENS = 64
PREDICTOR_DEFAULT_MAX_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.013
WARMUP_RATIO = 0.02
MAX_GRAD_NORM = 0.5
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 20
MAX_STEPS = 0
SEED = 42
EARLY_STOPPING_PATIENCE = 3
SAVE_TOTAL_LIMIT = 2

TRAIN_ARCH_FAMILIES = frozenset({"cnn", "segmentation"})
VAL_ARCH_FAMILIES = frozenset({"detector", "transformer", "rnn"})
TEST_ARCH_FAMILIES = frozenset({"other"})

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

COT_SYSTEM_PROMPT = """You are a neural-network performance predictor.
First reason briefly (a few short lines) about the training signals you are
given, then output your final answer as exactly ONE JSON object.

The JSON must contain exactly the keys:
best_accuracy
best_epoch

Rules:
best_accuracy must be a float representing the predicted final best validation accuracy.
best_epoch must be a positive integer representing the epoch where peak validation accuracy occurs.

Put your reasoning first, then a line that begins with OUTPUT: followed by the JSON.
Stop immediately after the closing brace }.
"""


MIN_N_FOR_DATASET_METRICS = 5  # below this, correlation is too noisy to report meaningfully
