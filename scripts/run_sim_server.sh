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

GPUS="${GPUS:-5}"
PORT="${PORT:-30000}"
ROOT_PATH="${ROOT_PATH:-$root_dir}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SIM_SERVER_PY="${SIM_SERVER_PY:-$root_dir/code/src/envs/sim_server.py}"

echo "[sim_server] env=${CONDA_ENV_NAME} python=$(command -v "$PYTHON_BIN")"
echo "[sim_server] gpus=${GPUS} port=${PORT} root_path=${ROOT_PATH}"
echo "[sim_server] script=${SIM_SERVER_PY}"

exec "$PYTHON_BIN" "$SIM_SERVER_PY" \
  --gpus "$GPUS" \
  --port "$PORT" \
  --root_path "$ROOT_PATH"

