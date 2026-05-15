#!/usr/bin/env bash
set -euo pipefail

# =========================
# Paths (auto derived)
# =========================
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
dataset_root="$root_dir/Dataset"
save_dir="$root_dir/save_teacher_dit"

epochs="50"
# ==========================================
# Dataset selection
# ==========================================
scene_list="City_1,City_2,City_3"
trajectory_range="1-450"

# =========================
# 训练方案（下面两行直接写 true / false）
# KL、特权重建、策略监督 三档均开启；无 DiT 用 MLP，有 DiT 时对采样动作 vs 专家做 BC（总 loss 不含扩散噪声/x0）。
# =========================
# 方案 A: TRAIN_REWARD_AUX=false  USE_DIFFUSION_ACTOR=false
#   → KL + MLP 与专家 + privileged_recon（无 reward 辅助、无 DiT）
# 方案 B: TRAIN_REWARD_AUX=true   USE_DIFFUSION_ACTOR=false
#   → 方案 A + 后验/先验 reward
# 方案 C: USE_DIFFUSION_ACTOR=true（TRAIN_REWARD_AUX 建议 true）
#   → 方案 B + DiT 采样 BC（推理与训练用 DiT；总 loss 不加噪声/x0 项）
# 在脚本里直接改成 true 或 false（小写）
TRAIN_REWARD_AUX=true
USE_DIFFUSION_ACTOR=true

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
GPU_IDS="${GPU_IDS:-0,1,2,3}"
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
  --multi-gpu \
  --train-reward-aux "${TRAIN_REWARD_AUX}" \
  --use-diffusion-actor "${USE_DIFFUSION_ACTOR}"
