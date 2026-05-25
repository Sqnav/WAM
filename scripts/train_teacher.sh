#!/usr/bin/env bash
set -euo pipefail

# =========================
# Paths (auto derived)
# =========================
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
dataset_root="$root_dir/Dataset"
save_dir="$root_dir/save_teacher_dit_noDIT"

epochs="100"
batch_size="4"
seq_len="16"
val_ratio="0.0"
max_yaw_rate="15.0"
max_speed_norm="1.0"
action_sequence_horizon="3"
num_workers="8"

# =========================
# Visual target guidance switches
# global image is always kept; this adds an optional target heatmap cue.
# =========================
USE_TARGET_VISUAL_GUIDANCE=false
USE_ATTENTION_HEATMAP=true
VISUAL_GUIDANCE_FOV_DEG="90.0"
ATTENTION_HEATMAP_SIGMA="0.08"

# =========================
# WAM auxiliary switches
# =========================
TRAIN_NEXT_PRIVILEGED=true
TRAIN_ROLLOUT=true
next_privileged_loss_weight="1.0"
prior_privileged_loss_weight="0.2"
direct_action_loss_weight="2.0"
action_yaw_loss_weight="10.0"
rollout_loss_weight="0.2"
rollout_horizon="3"
# ==========================================
# Dataset selection
# ==========================================
scene_list="City_1,City_2,City_3"
trajectory_range="1-450"

# =========================
# 训练方案（下面一行直接写 true / false）
# KL、策略监督、WAM 辅助由上面开关控制；无 DiT 用 MLP，有 DiT 时使用扩散噪声预测训练。
# =========================
# 在脚本里直接改成 true 或 false（小写）
USE_DIFFUSION_ACTOR=false
PRIVILEGED_FUSION_MODE="concat"

# =========================
# Environment
# =========================
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi
export PYTHONPATH="$root_dir/code/src:${PYTHONPATH:-}"

# 避免 torchrun 自动把每个进程的 OMP_NUM_THREADS 设成 1 后出现警告；需要时可在外部覆盖。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# =========================
# GPU selection
# =========================
GPU_IDS="${GPU_IDS:-2,3}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# DDP debug: 需要定位 unused parameter 时可在外部设为 DETAIL
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

mkdir -p "${save_dir}"

MASTER_PORT="${MASTER_PORT:-29501}"
NUM_GPUS="$(${PYTHON_BIN} - <<'PY'
import os
g = os.environ.get("CUDA_VISIBLE_DEVICES", "")
g = [x for x in g.split(",") if x.strip()]
print(max(len(g), 1))
PY
)"

echo "[train_teacher] use_diffusion_actor=${USE_DIFFUSION_ACTOR} privileged_input=disabled privileged_fusion_mode=${PRIVILEGED_FUSION_MODE}"
echo "[train_teacher] action_sequence_horizon=${action_sequence_horizon} rollout_loss_weight=${rollout_loss_weight}"
echo "[train_teacher] visual_guidance=${USE_TARGET_VISUAL_GUIDANCE} heatmap=${USE_ATTENTION_HEATMAP}"

# Launch training with DDP. Do not use DataParallel here.
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  -m train.train_teacher \
  --dataset-root "${dataset_root}" \
  --scene-list "${scene_list}" \
  --trajectory-range "${trajectory_range}" \
  --save-dir "${save_dir}" \
  --epochs "${epochs}" \
  --batch-size "${batch_size}" \
  --seq-len "${seq_len}" \
  --val-ratio "${val_ratio}" \
  --max-yaw-rate "${max_yaw_rate}" \
  --max-speed-norm "${max_speed_norm}" \
  --action-sequence-horizon "${action_sequence_horizon}" \
  --num-workers "${num_workers}" \
  --train-next-privileged "${TRAIN_NEXT_PRIVILEGED}" \
  --train-rollout "${TRAIN_ROLLOUT}" \
  --next-privileged-loss-weight "${next_privileged_loss_weight}" \
  --prior-privileged-loss-weight "${prior_privileged_loss_weight}" \
  --direct-action-loss-weight "${direct_action_loss_weight}" \
  --action-yaw-loss-weight "${action_yaw_loss_weight}" \
  --rollout-loss-weight "${rollout_loss_weight}" \
  --rollout-horizon "${rollout_horizon}" \
  --use-target-visual-guidance "${USE_TARGET_VISUAL_GUIDANCE}" \
  --use-attention-heatmap "${USE_ATTENTION_HEATMAP}" \
  --visual-guidance-fov-deg "${VISUAL_GUIDANCE_FOV_DEG}" \
  --attention-heatmap-sigma "${ATTENTION_HEATMAP_SIGMA}" \
  --multi-gpu \
  --privileged-fusion-mode "${PRIVILEGED_FUSION_MODE}" \
  --use-diffusion-actor "${USE_DIFFUSION_ACTOR}"
