#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../../.." && pwd)
cd "$REPO_ROOT"

RUN_ROOT=${RUN_ROOT:-.}
SEED=${SEED:-42}
RUN_ID="4pattern_deepseek_cifar10_adapter_seed${SEED}"

export NNGPT_SFT_BASE_MODEL_ID=deepseek-ai/deepseek-coder-6.7b-instruct
export NNGPT_SFT_INIT_ADAPTER=out/sft_full/backbone_struct1_20260716/A3/deepseek-ai/deepseek-coder-6.7b-instruct
export NNGPT_SFT_LOAD_INITIAL_ADAPTER=1
export NNGPT_SFT_INITIAL_ADAPTER_MODE=trainable

export NNGPT_RL_FORMAL_DATASET=cifar-10
export NNGPT_RL_FORMAL_REWARD_EPOCHS=5
export NNGPT_RL_SEED="$SEED"
export NNGPT_SFT_MAX_STEPS=100
export NNGPT_SFT_RL_NN_PREFIXES=rl-bb-struct1,rl-bb-struct1-v2

export NNGPT_SFT_LOG_DIR="$RUN_ROOT/out/rl/$RUN_ID/rl_output"
export NNGPT_SFT_MODEL_OUT="$RUN_ROOT/out/rl/$RUN_ID/model"
export NNGPT_SFT_TRAINER_OUT="$RUN_ROOT/out/rl/$RUN_ID/trainer"
export NNGPT_SFT_EPOCH_ROOT="$RUN_ROOT/out/rl/$RUN_ID/epoch_sft"

exec python -m ab.gpt.TuneRLSft
