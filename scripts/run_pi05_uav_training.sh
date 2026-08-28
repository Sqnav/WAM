#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
openpi_root="${OPENPI_ROOT:-$root_dir/third_party/openpi}"
exp_root="${EXP_ROOT:-$root_dir/experiments/fastwam_local_feature_ablation_run}"
openpi_venv="${OPENPI_VENV:-$root_dir/.venvs/openpi_uav}"
train_gpu_ids="${PI05_TRAIN_GPU_IDS:-0,1,2,3,4,5}"
train_steps="${PI05_TRAIN_STEPS:-30000}"
exp_name="${PI05_EXP_NAME:-uav_city1_27_lora}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$exp_root/openpi_data/lerobot}"
export HF_HOME="${HF_HOME:-$exp_root/openpi_data/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$exp_root/openpi_data/cache}"
export OPENPI_UAV_ASSETS_DIR="${OPENPI_UAV_ASSETS_DIR:-$exp_root/openpi_data/assets}"
export OPENPI_UAV_CHECKPOINT_DIR="${OPENPI_UAV_CHECKPOINT_DIR:-$exp_root/models/openpi}"
mkdir -p \
  "$HF_LEROBOT_HOME" \
  "$HF_HOME" \
  "$OPENPI_DATA_HOME" \
  "$OPENPI_UAV_ASSETS_DIR" \
  "$OPENPI_UAV_CHECKPOINT_DIR" \
  "$exp_root/logs"

if [[ ! -x "$openpi_venv/bin/python" ]]; then
  bash "$root_dir/code/scripts/setup_openpi_uav.sh"
fi
openpi_python="${OPENPI_PYTHON:-$openpi_venv/bin/python}"

dataset_complete="$HF_LEROBOT_HOME/worldmodel/uav_pursuit_city1_27_train/.worldmodel_complete.json"
if [[ ! -f "$dataset_complete" ]]; then
  convert_args=()
  if [[ -d "${dataset_complete%/.worldmodel_complete.json}" ]]; then
    convert_args+=(--resume)
  fi
  "$openpi_python" "$openpi_root/examples/uav/convert_uav_data_to_lerobot.py" \
    --dataset-root "$root_dir/Dataset" \
    "${convert_args[@]}" \
    2>&1 | tee "$exp_root/logs/pi05_uav_dataset_conversion.log"
fi

norm_stats="$OPENPI_UAV_ASSETS_DIR/pi05_uav_lora/worldmodel/uav_pursuit_city1_27_train/norm_stats.json"
if [[ ! -f "$norm_stats" ]]; then
  "$openpi_python" "$openpi_root/examples/uav/compute_uav_norm_stats.py" \
    --dataset-dir "$HF_LEROBOT_HOME/worldmodel/uav_pursuit_city1_27_train" \
    --output-dir "$OPENPI_UAV_ASSETS_DIR/pi05_uav_lora/worldmodel/uav_pursuit_city1_27_train" \
    --action-horizon 8 \
    --batch-size 48 \
    2>&1 | tee "$exp_root/logs/pi05_uav_norm_stats.log"
fi

checkpoint_root="$OPENPI_UAV_CHECKPOINT_DIR/pi05_uav_lora/$exp_name"
train_mode=(--overwrite)
existing_checkpoint=""
if [[ -d "$checkpoint_root" ]]; then
  existing_checkpoint="$(find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk '/^[0-9]+$/' | sort -n | tail -1)"
fi
if [[ -n "$existing_checkpoint" ]]; then
  train_mode=(--resume)
fi

echo "[pi05-train] GPUs=$train_gpu_ids steps=$train_steps checkpoint=$checkpoint_root"
(
  cd "$openpi_root"
  CUDA_VISIBLE_DEVICES="$train_gpu_ids" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}" \
    "$openpi_python" scripts/train.py pi05_uav_lora \
      --exp-name "$exp_name" \
      --num-train-steps "$train_steps" \
      "${train_mode[@]}"
) 2>&1 | tee -a "$exp_root/logs/pi05_uav_training.log"

latest_checkpoint="$(find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
  | awk '/^[0-9]+$/' | sort -n | tail -1)"
if [[ -z "$latest_checkpoint" ]]; then
  echo "[ERROR] pi05 training finished without a checkpoint in $checkpoint_root" >&2
  exit 1
fi
echo "$checkpoint_root/$latest_checkpoint" > "$exp_root/models/openpi/pi05_uav_latest_checkpoint.txt"
echo "[pi05-train] complete: $checkpoint_root/$latest_checkpoint"
