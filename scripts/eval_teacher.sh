#!/usr/bin/env bash
set -euo pipefail

#
# Minimal online evaluation launcher (single process).
# Change only SCENE_LIST / TRAJECTORY_RANGE by default.
#

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"

DATASET_ROOT="$root_dir/Dataset"
CHECKPOINT="$root_dir/save_teacher_dit_noDIT/best.pt"
OUTPUT_DIR="$root_dir/online_eval_teacher"
EXECUTOR_SCRIPT="$root_dir/code/src/executor/trajectory_executor.py"

# =========================
# Dataset selection (usually the only things you change)
# =========================
SCENE_LIST="City_1"
TRAJECTORY_RANGE="451-500"

# =========================
# 验证方案（与 train_teacher.sh 对齐，直接写 true / false）
# USE_DIFFUSION_ACTOR=true  → 使用 DiT actor 验证
# USE_DIFFUSION_ACTOR=false → 使用 MLP direct action head 验证（Phase-A/B checkpoint）
# =========================
USE_DIFFUSION_ACTOR=false
# Empty means: use the mode saved in checkpoint cfg.
# Set to "attention" or "concat" only when you intentionally want to override it.
PRIVILEGED_FUSION_MODE=""

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
MAX_YAW_RATE="15.0"
MAX_SPEED_NORM="1.0"
DIT_CANDIDATE_SELECTION=true
DIT_CANDIDATE_COUNT="4"
DIT_CANDIDATE_LATERAL_WEIGHT="1.0"
DIT_CANDIDATE_VERTICAL_WEIGHT="1.0"
DIT_CANDIDATE_DISTANCE_WEIGHT="0.05"
DIT_CANDIDATE_SMOOTH_WEIGHT="0.05"
USE_TARGET_VISUAL_GUIDANCE=false
USE_ATTENTION_HEATMAP=true
VISUAL_GUIDANCE_FOV_DEG="90.0"
ATTENTION_HEATMAP_SIGMA="0.08"

# Phase-A/B checkpoints (no DiT): set USE_DIFFUSION_ACTOR=false above.

# =========================
# AirSim / online rollout
# =========================
SIM_SERVER_HOST="127.0.0.1"
SIM_SERVER_PORT=30000
SCENE_INDEX=1

mkdir -p "${OUTPUT_DIR}"
echo "[eval] scenes=${SCENE_LIST} range=${TRAJECTORY_RANGE} diffusion=${USE_DIFFUSION_ACTOR}"
echo "[eval] privileged_input=disabled"
echo "[eval] dit_candidate_selection=${DIT_CANDIDATE_SELECTION} count=${DIT_CANDIDATE_COUNT}"
echo "[eval] visual_guidance=${USE_TARGET_VISUAL_GUIDANCE} heatmap=${USE_ATTENTION_HEATMAP}"
if [[ -n "${PRIVILEGED_FUSION_MODE}" ]]; then
  echo "[eval] privileged_fusion_override=${PRIVILEGED_FUSION_MODE}"
else
  echo "[eval] privileged_fusion=checkpoint"
fi

extra_args=()
if [[ -n "${PRIVILEGED_FUSION_MODE}" ]]; then
  extra_args+=(--privileged-fusion-mode "${PRIVILEGED_FUSION_MODE}")
fi

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
  --scene-index "${SCENE_INDEX}" \
  --gpu-id "${GPU_ID_FOR_AIRSIM}" \
  --device "${DEVICE}" \
  --max-yaw-rate "${MAX_YAW_RATE}" \
  --max-speed-norm "${MAX_SPEED_NORM}" \
  --dit-candidate-selection "${DIT_CANDIDATE_SELECTION}" \
  --dit-candidate-count "${DIT_CANDIDATE_COUNT}" \
  --dit-candidate-lateral-weight "${DIT_CANDIDATE_LATERAL_WEIGHT}" \
  --dit-candidate-vertical-weight "${DIT_CANDIDATE_VERTICAL_WEIGHT}" \
  --dit-candidate-distance-weight "${DIT_CANDIDATE_DISTANCE_WEIGHT}" \
  --dit-candidate-smooth-weight "${DIT_CANDIDATE_SMOOTH_WEIGHT}" \
  --use-target-visual-guidance "${USE_TARGET_VISUAL_GUIDANCE}" \
  --use-attention-heatmap "${USE_ATTENTION_HEATMAP}" \
  --visual-guidance-fov-deg "${VISUAL_GUIDANCE_FOV_DEG}" \
  --attention-heatmap-sigma "${ATTENTION_HEATMAP_SIGMA}" \
  --use-diffusion-actor "${USE_DIFFUSION_ACTOR}" \
  "${extra_args[@]}"
