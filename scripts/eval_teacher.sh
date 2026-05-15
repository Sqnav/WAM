#!/usr/bin/env bash
set -euo pipefail

#
# Minimal online evaluation launcher (single process).
# Change only SCENE_LIST / TRAJECTORY_RANGE by default.
#

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"

DATASET_ROOT="$root_dir/Dataset"
CHECKPOINT="$root_dir/save_teacher_dit/best.pt"
OUTPUT_DIR="$root_dir/online_eval_teacher"
EXECUTOR_SCRIPT="$root_dir/code/src/executor/trajectory_executor.py"

# =========================
# Dataset selection (usually the only things you change)
# =========================
SCENE_LIST="City_1"
TRAJECTORY_RANGE="451-500"

# =========================
# Runtime / model
# =========================
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi

export PYTHONPATH="$root_dir/code/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export DAGGER_MULTI_WORKER=1

# GPU selection (single GPU)
GPU_ID="2"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
GPU_ID_FOR_AIRSIM="${GPU_ID}"
DEVICE="cuda"

# Phase-A checkpoints (no DiT): append --force-direct-action to the python line below.

# =========================
# AirSim / online rollout
# =========================
SIM_SERVER_HOST="127.0.0.1"
SIM_SERVER_PORT=30000
SCENE_INDEX=1

mkdir -p "${OUTPUT_DIR}"
echo "[eval] scenes=${SCENE_LIST} range=${TRAJECTORY_RANGE}"

# Single process: do NOT use torchrun, DDP, DataParallel, or ThreadPoolExecutor here.
"${PYTHON_BIN}" -m eval.online_eval_teacher \
  --dataset-root "${DATASET_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --executor-script "${EXECUTOR_SCRIPT}" \
  --scene-list "${SCENE_LIST}" \
  --trajectory-range "${TRAJECTORY_RANGE}" \
  --eval-split all \
  --sim-server-host "${SIM_SERVER_HOST}" \
  --sim-server-port "${SIM_SERVER_PORT}" \
  --gpu-id "${GPU_ID_FOR_AIRSIM}" \
  --device "${DEVICE}"
