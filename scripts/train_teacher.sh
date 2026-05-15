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
  --multi-gpu
