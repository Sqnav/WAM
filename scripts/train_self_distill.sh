#!/usr/bin/env bash
set -euo pipefail

# =========================
# Paths
# =========================
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
DATASET_ROOT="$root_dir/Dataset"
TEACHER_CKPT="${TEACHER_CKPT:-$root_dir/experiments/visual_guidance_ablation/dit_heatmap/best.pt}"
SAVE_DIR="${SAVE_DIR:-$root_dir/experiments/visual_guidance_ablation/self_distill_dit_heatmap}"

# ==========================================
# Dataset selection
# ==========================================
SCENE_LIST="${SCENE_LIST:-City_1,City_2,City_3}"
TRAJECTORY_RANGE="${TRAJECTORY_RANGE:-1-450}"
VAL_RATIO="${VAL_RATIO:-0.0}"
SPLIT_SEED="${SPLIT_SEED:-42}"

# =========================
# Environment
# =========================
if command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate ysq_qwen
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "ysq_qwen" ]]; then
  echo "[ERROR] This experiment must run inside conda env ysq_qwen, got CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}" >&2
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi

export PYTHONPATH="$root_dir/code/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# =========================
# GPU selection
# =========================
GPU_IDS="${GPU_IDS:-2,3}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

NUM_GPUS="$(${PYTHON_BIN} - <<'PY'
import os
g = os.environ.get("CUDA_VISIBLE_DEVICES", "")
ids = [x for x in g.split(",") if x.strip()]
print(max(len(ids), 1))
PY
)"

# =========================
# Training config
# =========================
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAX_VEL="${MAX_VEL:-1.0}"
MAX_YAW_RATE="${MAX_YAW_RATE:-15.0}"
MAX_SPEED_NORM="${MAX_SPEED_NORM:-1.0}"
ACTION_SEQUENCE_HORIZON="${ACTION_SEQUENCE_HORIZON:-3}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
MASTER_PORT="${MASTER_PORT:-29503}"
TARGET_TOKEN_FUSION_MODE="${TARGET_TOKEN_FUSION_MODE:-concat}"
USE_TARGET_VISUAL_GUIDANCE="${USE_TARGET_VISUAL_GUIDANCE:-true}"
USE_ATTENTION_HEATMAP="${USE_ATTENTION_HEATMAP:-true}"
VISUAL_GUIDANCE_FOV_DEG="${VISUAL_GUIDANCE_FOV_DEG:-90.0}"
ATTENTION_HEATMAP_SIGMA="${ATTENTION_HEATMAP_SIGMA:-0.08}"

# Self-distillation:
# supervised WAM loss + posterior RSSM belief distillation.
# In DiT mode, action_distill compares aligned denoised x0 and dit_noise compares
# aligned predicted noise under the same x_t/t.
SUP_WEIGHT="${SUP_WEIGHT:-1.0}"
FEAT_DISTILL_WEIGHT="${FEAT_DISTILL_WEIGHT:-0.1}"
ACTION_DISTILL_WEIGHT="${ACTION_DISTILL_WEIGHT:-0.5}"
DIT_NOISE_DISTILL_WEIGHT="${DIT_NOISE_DISTILL_WEIGHT:-0.5}"

INIT_STUDENT_FROM_TEACHER="${INIT_STUDENT_FROM_TEACHER:-1}"

mkdir -p "${SAVE_DIR}"

extra_args=()

if [[ "${INIT_STUDENT_FROM_TEACHER}" == "1" || "${INIT_STUDENT_FROM_TEACHER}" == "true" || "${INIT_STUDENT_FROM_TEACHER}" == "True" ]]; then
  extra_args+=(--init-student-from-teacher)
elif [[ "${INIT_STUDENT_FROM_TEACHER}" == "0" || "${INIT_STUDENT_FROM_TEACHER}" == "false" || "${INIT_STUDENT_FROM_TEACHER}" == "False" ]]; then
  extra_args+=(--student-init-random)
fi

echo "============================================================"
echo "Teacher self-distillation"
echo "============================================================"
echo "root_dir             : ${root_dir}"
echo "dataset_root         : ${DATASET_ROOT}"
echo "teacher_ckpt         : ${TEACHER_CKPT}"
echo "save_dir             : ${SAVE_DIR}"
echo "scene_list           : ${SCENE_LIST}"
echo "trajectory_range     : ${TRAJECTORY_RANGE}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS             : ${NUM_GPUS}"
echo "batch_size/GPU       : ${BATCH_SIZE}"
echo "epochs               : ${EPOCHS}"
echo "lr                   : ${LR}"
echo "low_dim_target_input: off"
echo "target_token_fusion    : ${TARGET_TOKEN_FUSION_MODE}"
echo "visual_guidance      : ${USE_TARGET_VISUAL_GUIDANCE}, heatmap=${USE_ATTENTION_HEATMAP}"
echo "loss weights         : sup=${SUP_WEIGHT}, feat=${FEAT_DISTILL_WEIGHT}, action=${ACTION_DISTILL_WEIGHT}, dit_noise=${DIT_NOISE_DISTILL_WEIGHT}"
echo "============================================================"

common_args=(
  -m train.train_self_distill
  --dataset-root "${DATASET_ROOT}"
  --scene-list "${SCENE_LIST}"
  --trajectory-range "${TRAJECTORY_RANGE}"
  --val-ratio "${VAL_RATIO}"
  --split-seed "${SPLIT_SEED}"
  --teacher-ckpt "${TEACHER_CKPT}"
  --save-dir "${SAVE_DIR}"
  --image-size "${IMAGE_SIZE}"
  --seq-len "${SEQ_LEN}"
  --max-vel "${MAX_VEL}"
  --max-yaw-rate "${MAX_YAW_RATE}"
  --max-speed-norm "${MAX_SPEED_NORM}"
  --action-sequence-horizon "${ACTION_SEQUENCE_HORIZON}"
  --target-token-fusion-mode "${TARGET_TOKEN_FUSION_MODE}"
  --use-target-visual-guidance "${USE_TARGET_VISUAL_GUIDANCE}"
  --use-attention-heatmap "${USE_ATTENTION_HEATMAP}"
  --visual-guidance-fov-deg "${VISUAL_GUIDANCE_FOV_DEG}"
  --attention-heatmap-sigma "${ATTENTION_HEATMAP_SIGMA}"
  --diffusion-steps "${DIFFUSION_STEPS}"
  --sampling-steps "${SAMPLING_STEPS}"
  --freeze-dinov2
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --sup-weight "${SUP_WEIGHT}"
  --feat-distill-weight "${FEAT_DISTILL_WEIGHT}"
  --action-distill-weight "${ACTION_DISTILL_WEIGHT}"
  --dit-noise-distill-weight "${DIT_NOISE_DISTILL_WEIGHT}"
  "${extra_args[@]}"
)

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${NUM_GPUS}" \
    --master_port "${MASTER_PORT}" \
    "${common_args[@]}" \
    --multi-gpu
else
  "${PYTHON_BIN}" "${common_args[@]}"
fi
