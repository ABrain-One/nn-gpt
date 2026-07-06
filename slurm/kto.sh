#!/bin/bash



#SBATCH --job-name=nngpt-kto-origprompt
#SBATCH --output=gslurm-kto-origprompt-%j.out
#SBATCH --error=gslurm-kto-origprompt-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --partition=standard

set -euo pipefail

mkdir -p logs

# ── Configurable variables ─────────────────────────────────────────────────────
# 2671923
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_DIR/nngpt.sif}"

HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_DIR/.cache/hf_datasets}"

# Experiment knobs (this ablation's requested values).
LLM_CONF="${LLM_CONF:-nngpt_unique_arch_rag.json}"   # → ABrain/NNGPT-UniqueArch-Rag
CYCLES="${CYCLES:-21}"
MODELS_PER_CYCLE="${MODELS_PER_CYCLE:-100}"
ACCURACY_THRESHOLD="${ACCURACY_THRESHOLD:-0.40}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-kto_selfcontained}"

# Prompt mode — the point of this ablation.  See header for the warning on "original".
KTO_PROMPT_MODE="${KTO_PROMPT_MODE:-minimal}"

# Per-run output isolation (own out_<RUN_TAG>/ folder).
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
RUN_OUTPUT_REL="out_${RUN_TAG}"

# Generation knobs.
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_K="${TOP_K:-50}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
GEN_MAX_REJECTIONS="${GEN_MAX_REJECTIONS:-3}"

# KTO knobs.
KTO_BETA="${KTO_BETA:-0.1}"
KTO_DESIRABLE_WEIGHT="${KTO_DESIRABLE_WEIGHT:-1.0}"
KTO_UNDESIRABLE_WEIGHT="${KTO_UNDESIRABLE_WEIGHT:-1.0}"
UNDESIRABLE_RATIO="${UNDESIRABLE_RATIO:-1.0}"
MAX_UNDESIRABLE_TOTAL="${MAX_UNDESIRABLE_TOTAL:-1000}"

# Evaluation knobs — card/ab.nn protocol (identical to slurm_nngpt_kto_7b.sh).
EVAL_TRANSFORM="${EVAL_TRANSFORM:-norm_256_flip}"
EVAL_LR="${EVAL_LR:-0.01}"
EVAL_MOMENTUM="${EVAL_MOMENTUM:-0.9}"
EVAL_BATCH="${EVAL_BATCH:-10}"

HF_TOKEN="${HF_TOKEN:-}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
RESUME_FROM_CYCLE="${RESUME_FROM_CYCLE:-}"

# Per-run output paths.
HOST_RUN_ROOT="${PROJECT_DIR}/${RUN_OUTPUT_REL}"
CONTAINER_NNGPT_DIR="/project/${RUN_OUTPUT_REL}/nngpt"
HOST_OUTPUT_DIR="${HOST_RUN_ROOT}/nngpt/${OUTPUT_SUBDIR}"

# Container path mapping (PROJECT_DIR → /project)
if [[ "$HF_HOME" == "$PROJECT_DIR"* ]]; then
    CONTAINER_HF_HOME="${HF_HOME/$PROJECT_DIR//project}"
    EXTRA_BIND=""
else
    CONTAINER_HF_HOME="$HF_HOME"
    EXTRA_BIND="--bind ${HF_HOME}:${HF_HOME} --bind ${HF_DATASETS_CACHE}:${HF_DATASETS_CACHE}"
fi

if [[ "$HF_DATASETS_CACHE" == "$PROJECT_DIR"* ]]; then
    CONTAINER_HF_DATASETS="${HF_DATASETS_CACHE/$PROJECT_DIR//project}"
else
    CONTAINER_HF_DATASETS="$HF_DATASETS_CACHE"
fi

# ── Pre-flight checks & directory setup ───────────────────────────────────────

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HOST_OUTPUT_DIR"

echo "=========================================="
echo "Job ID                : ${SLURM_JOB_ID:-manual}"
echo "Node                  : $(hostname)"
echo "GPU(s)                : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
echo "------------------------------------------"
echo "Mode                  : self-contained KTO — ORIGINAL-PROMPT ablation"
echo "Prompt mode           : ${KTO_PROMPT_MODE}"
echo "LLM config            : ${LLM_CONF}"
echo "Cycles                : ${CYCLES}"
echo "Models per cycle      : ${MODELS_PER_CYCLE}"
echo "Accuracy threshold    : ${ACCURACY_THRESHOLD}"
echo "Num train epochs      : ${NUM_TRAIN_EPOCHS}"
echo "Temperature           : ${TEMPERATURE}"
echo "Eval transform        : ${EVAL_TRANSFORM} (batch ${EVAL_BATCH})"
echo "Run output root       : ${RUN_OUTPUT_REL}/ (isolated per job)"
echo "Output dir            : ${RUN_OUTPUT_REL}/nngpt/${OUTPUT_SUBDIR}"
echo "------------------------------------------"
echo "Container             : ${CONTAINER_IMAGE}"
echo "Project dir           : ${PROJECT_DIR}"
if [ -n "${RESUME_FROM_CYCLE}" ]; then
    echo "Resume from cycle     : ${RESUME_FROM_CYCLE}"
fi
echo "=========================================="

if [ "${KTO_PROMPT_MODE}" = "original" ]; then
    echo "WARNING: KTO_PROMPT_MODE=original uses the repo's in_shape[0] rule, which is"
    echo "         WRONG for this evaluator → expect heavy Conv2d(1,...) eval failures."
fi

if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo ""
    echo "ERROR: Container image not found at: $CONTAINER_IMAGE"
    echo "Build it first:  sbatch slurm_build_sif.sh"
    echo ""
    exit 1
fi

# ── Copy HF cache to node-local disk (avoids NFS mmap failures) ───────────────
LOCAL_HF_HOME="${TMPDIR:-/tmp}/hf_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$LOCAL_HF_HOME"
echo "Syncing HF cache to node-local storage: ${LOCAL_HF_HOME}"
rsync -a "${HF_HOME}/" "${LOCAL_HF_HOME}/"
echo "Sync complete."

# ── Run the self-contained iterative KTO pipeline ─────────────────────────────

PIPELINE_CMD=(
    python -m ab.gpt.kto_pipeline.kto_selfcontained_finetune
    --llm_conf "${LLM_CONF}"
    --cycles "${CYCLES}"
    --models_per_cycle "${MODELS_PER_CYCLE}"
    --accuracy_threshold "${ACCURACY_THRESHOLD}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --temperature "${TEMPERATURE}"
    --top_k "${TOP_K}"
    --top_p "${TOP_P}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --gen_max_rejections "${GEN_MAX_REJECTIONS}"
    --kto_beta "${KTO_BETA}"
    --kto_desirable_weight "${KTO_DESIRABLE_WEIGHT}"
    --kto_undesirable_weight "${KTO_UNDESIRABLE_WEIGHT}"
    --undesirable_ratio "${UNDESIRABLE_RATIO}"
    --max_undesirable_total "${MAX_UNDESIRABLE_TOTAL}"
    --eval_transform "${EVAL_TRANSFORM}"
    --eval_lr "${EVAL_LR}"
    --eval_momentum "${EVAL_MOMENTUM}"
    --eval_batch "${EVAL_BATCH}"
    --output_subdir "${OUTPUT_SUBDIR}"
)
if [ -n "${RESUME_FROM_CYCLE}" ]; then
    PIPELINE_CMD+=(--resume_from_cycle "${RESUME_FROM_CYCLE}")
fi

# shellcheck disable=SC2086  # EXTRA_BIND intentionally unquoted (may be empty)
apptainer exec \
    --nv \
    --cleanenv \
    --writable-tmpfs \
    --pwd /project \
    --bind "${PROJECT_DIR}:/project" \
    --bind "${LOCAL_HF_HOME}:${LOCAL_HF_HOME}" \
    ${EXTRA_BIND} \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env HF_HOME="${LOCAL_HF_HOME}" \
    --env HF_DATASETS_CACHE="${CONTAINER_HF_DATASETS}" \
    --env TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
    --env HF_DATASETS_OFFLINE=0 \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    --env BNB_CUDA_VERSION=126 \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPATH="/project:${PYTHONPATH:-}" \
    --env NNGPT_DIR_OVERRIDE="${CONTAINER_NNGPT_DIR}" \
    --env MKL_THREADING_LAYER=GNU \
    --env KTO_PROMPT_MODE="${KTO_PROMPT_MODE}" \
    "$CONTAINER_IMAGE" \
    "${PIPELINE_CMD[@]}"

echo ""
echo "Original-prompt ablation complete.  Results in: ${HOST_OUTPUT_DIR}"
echo "  Compare all_cycles_results.json against the enhanced-prompt run to see"
echo "  whether removing our prompt additions changed generation quality."
