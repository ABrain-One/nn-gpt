#!/bin/bash

#SBATCH --job-name=edge-k4
#SBATCH --output=slurm-edge-k4-%j.out
#SBATCH --error=slurm-edge-k4-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=standard

# Edge k4 generation pipeline: generate/evaluate/finetune loop followed by
# edge efficiency scoring. Adjust --partition (and add e.g. --constraint=h100
# or --gres=gpu:h100:1 if the cluster names GPU types) to match the target node.
#
# Usage:
#   sbatch slurm/edge_k4.sh                          # container mode (nngpt.sif)
#   USE_CONTAINER=0 sbatch slurm/edge_k4.sh          # bare venv mode
#   EDGE_DATASET=cifar-100 sbatch slurm/edge_k4.sh   # scoring dataset override

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_DIR/nngpt.sif}"
USE_CONTAINER="${USE_CONTAINER:-1}"

HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_DIR/.cache/hf_datasets}"
HF_TOKEN="${HF_TOKEN:-}"

EDGE_DATASET="${EDGE_DATASET:-cifar-10}"
EDGE_PARAM_LIMIT="${EDGE_PARAM_LIMIT:-6000000}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" logs

echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID:-manual}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
echo "Pipeline     : edge k4 (generate/evaluate/finetune + EdgeScore)"
echo "Dataset      : ${EDGE_DATASET}"
echo "Project dir  : ${PROJECT_DIR}"
echo "Container    : ${USE_CONTAINER} (${CONTAINER_IMAGE})"
echo "=========================================="

RUN_CMD="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
python -m ab.gpt.curriculum.Curriculum_Gen_edge_k4_Tune_7B && \
python -m ab.gpt.edge.EdgeScore --dataset ${EDGE_DATASET} --param-limit ${EDGE_PARAM_LIMIT}"

if [ "$USE_CONTAINER" = "1" ]; then
    if [ ! -f "$CONTAINER_IMAGE" ]; then
        echo "ERROR: container image not found at $CONTAINER_IMAGE"
        echo "Either build it (see slurm_build_sif.sh) or run with USE_CONTAINER=0"
        exit 1
    fi

    # Copy HF cache to node-local disk (avoids NFS mmap failures)
    LOCAL_HF_HOME="${TMPDIR:-/tmp}/hf_cache_${SLURM_JOB_ID:-$$}"
    mkdir -p "$LOCAL_HF_HOME"
    rsync -a "${HF_HOME}/" "${LOCAL_HF_HOME}/"

    apptainer exec \
        --nv \
        --cleanenv \
        --writable-tmpfs \
        --pwd /project \
        --bind "${PROJECT_DIR}:/project" \
        --bind "${LOCAL_HF_HOME}:${LOCAL_HF_HOME}" \
        --env HF_HOME="${LOCAL_HF_HOME}" \
        --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
        --env HF_TOKEN="${HF_TOKEN}" \
        --env HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --env PYTHONPATH="/project:${PYTHONPATH:-}" \
        --env MKL_THREADING_LAYER=GNU \
        "$CONTAINER_IMAGE" \
        bash -c "$RUN_CMD"
else
    source "${PROJECT_DIR}/.venv/bin/activate"
    export HF_HOME HF_DATASETS_CACHE HF_TOKEN
    bash -c "$RUN_CMD"
fi

echo ""
echo "Edge run complete."
echo "  Efficiency ranking : out/edge/edge_tracker.json"
echo "  TFLite artifacts   : out/edge/tflite/  (sync to the phone benchmarking machine)"
