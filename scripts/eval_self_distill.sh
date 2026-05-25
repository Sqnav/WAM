#!/usr/bin/env bash
set -euo pipefail

#
# Minimal online evaluation launcher for privileged self-distill checkpoint.
# Change only SCENE_LIST / TRAJECTORY_RANGE by default.
#

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"

DATASET_ROOT="$root_dir/Dataset"
CHECKPOINT="$root_dir/save_self_distill/best.pt"
OUTPUT_DIR="$root_dir/online_eval_self_distill"
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

# GPU selection (single process)
GPU_ID="0"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
GPU_ID_FOR_AIRSIM="${GPU_ID}"
DEVICE="cuda"
PRIVILEGED_FUSION_MODE="concat"
USE_TARGET_VISUAL_GUIDANCE=false
USE_ATTENTION_HEATMAP=true
VISUAL_GUIDANCE_FOV_DEG="90.0"
ATTENTION_HEATMAP_SIGMA="0.08"

# =========================
# AirSim / online rollout
# =========================
SIM_SERVER_HOST="127.0.0.1"
SIM_SERVER_PORT=30000

mkdir -p "${OUTPUT_DIR}"
echo "[eval] checkpoint=${CHECKPOINT}"
echo "[eval] scenes=${SCENE_LIST} range=${TRAJECTORY_RANGE}"
echo "[eval] privileged_input=disabled"
echo "[eval] privileged_fusion=${PRIVILEGED_FUSION_MODE}"
echo "[eval] visual_guidance=${USE_TARGET_VISUAL_GUIDANCE} heatmap=${USE_ATTENTION_HEATMAP}"

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
  --device "${DEVICE}" \
  --use-target-visual-guidance "${USE_TARGET_VISUAL_GUIDANCE}" \
  --use-attention-heatmap "${USE_ATTENTION_HEATMAP}" \
  --visual-guidance-fov-deg "${VISUAL_GUIDANCE_FOV_DEG}" \
  --attention-heatmap-sigma "${ATTENTION_HEATMAP_SIGMA}" \
  --privileged-fusion-mode "${PRIVILEGED_FUSION_MODE}"
