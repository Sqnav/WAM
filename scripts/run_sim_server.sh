#!/usr/bin/env bash
set -euo pipefail

# Start SimServerTool (msgpackrpc) for AirSim scenes.
# This script activates conda env `ysq_qwen` and runs code/src/envs/sim_server.py.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"

# =========================
# Conda env
# =========================
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ysq_qwen}"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "/opt/anaconda3")"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [[ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/opt/anaconda3/etc/profile.d/conda.sh"
else
  echo "[sim_server] ERROR: conda.sh not found. Please install/initialize conda."
  exit 1
fi
conda activate "$CONDA_ENV_NAME"

# =========================
# Runtime config
# =========================
export PYTHONPATH="$root_dir/code/src:${PYTHONPATH:-}"

if [[ -z "${GPUS:-}" ]]; then
  if [[ -n "${GPU_ID:-}" ]]; then
    GPUS="$GPU_ID"
  elif [[ -n "${EVAL_SIM_GPU_ID:-}" ]]; then
    GPUS="$EVAL_SIM_GPU_ID"
  elif [[ -n "${SIM_GPU_ID:-}" ]]; then
    GPUS="$SIM_GPU_ID"
  elif [[ -n "${EVAL_GPU_ID:-}" ]]; then
    GPUS="$EVAL_GPU_ID"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPUS="${CUDA_VISIBLE_DEVICES%%,*}"
  else
    GPUS="0"
  fi
fi
PORT="${PORT:-30000}"
ROOT_PATH="${ROOT_PATH:-$root_dir}"
UNREAL_GRAPHICS_ADAPTER_MAP="${UNREAL_GRAPHICS_ADAPTER_MAP:-0:1,1:3,2:4,3:0,4:5,5:2}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SIM_SERVER_PY="${SIM_SERVER_PY:-$root_dir/code/src/envs/sim_server.py}"

echo "[sim_server] env=${CONDA_ENV_NAME} python=$(command -v "$PYTHON_BIN")"
echo "[sim_server] gpus=${GPUS} port=${PORT} root_path=${ROOT_PATH}"
echo "[sim_server] graphics_adapter_map=${UNREAL_GRAPHICS_ADAPTER_MAP}"
echo "[sim_server] script=${SIM_SERVER_PY}"

exec "$PYTHON_BIN" "$SIM_SERVER_PY" \
  --gpus "$GPUS" \
  --graphics-adapter-map "$UNREAL_GRAPHICS_ADAPTER_MAP" \
  --port "$PORT" \
  --root_path "$ROOT_PATH"
