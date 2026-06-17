#!/usr/bin/env bash
set -euo pipefail

# FastWAM ablation runner.
#
# Active experiment table:
#   fastwam_global                  : global image + instruction + FastWAM MoT
#   fastwam_target_relative_token   : baseline + body-frame target position token appended to FastWAM text context
#   self_distill_target_relative_token_to_global
#                                   : target-relative-token teacher -> global student

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
dataset_root="$root_dir/Dataset"
executor_script="$root_dir/code/src/executor/trajectory_executor.py"

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
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXP_NAME="${EXP_NAME:-fastwam_ablation}"
exp_root="${EXP_ROOT:-$root_dir/experiments}"
model_root="${MODEL_OUTPUT_ROOT:-$exp_root/models}"
log_dir="$exp_root/logs"
eval_root="${EVAL_OUTPUT_ROOT:-$exp_root/online_eval}"
eval_log_dir="${EVAL_LOG_DIR:-$exp_root/eval_logs}"
mkdir -p "$exp_root" "$model_root" "$log_dir" "$eval_root" "$eval_log_dir"

RUN_TEACHER_ABLATIONS="${RUN_TEACHER_ABLATIONS:-true}"
RUN_SELF_DISTILL="${RUN_SELF_DISTILL:-true}"
RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-true}"
USE_DEEPSPEED="${USE_DEEPSPEED:-true}"
DEEPSPEED_OFFLOAD_OPTIMIZER="${DEEPSPEED_OFFLOAD_OPTIMIZER:-true}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
CHECKPOINT_SAVE_EVERY_EPOCHS="${CHECKPOINT_SAVE_EVERY_EPOCHS:-5}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-true}"
SAVE_OPTIMIZER_STATE="${SAVE_OPTIMIZER_STATE:-false}"

EXPERIMENTS="${EXPERIMENTS-fastwam_global,fastwam_target_relative_token}"
DISTILL_EXPERIMENTS="${DISTILL_EXPERIMENTS-self_distill_target_relative_token_to_global}"
EVAL_EXTRA_EXPERIMENTS="${EVAL_EXTRA_EXPERIMENTS-}"
SKIP_EXISTING_TRAIN="${SKIP_EXISTING_TRAIN:-true}"
SKIP_EXISTING_SELF_DISTILL="${SKIP_EXISTING_SELF_DISTILL:-true}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-true}"

scene_list="${SCENE_LIST:-City_1,City_2,City_3}"
trajectory_range="${TRAJECTORY_RANGE:-1-450}"
val_ratio="${VAL_RATIO:-0.0}"
split_seed="${SPLIT_SEED:-42}"

train_steps="${TRAIN_STEPS:-0}"
model_train_epochs="${MODEL_TRAIN_EPOCHS:-5}"
teacher_batch_size="${TEACHER_BATCH_SIZE:-16}"
seq_len="${SEQ_LEN:-33}"
image_size="${IMAGE_SIZE:-224}"
num_workers="${NUM_WORKERS:-8}"
teacher_lr="${TEACHER_LR:-1e-4}"
teacher_weight_decay="${TEACHER_WEIGHT_DECAY:-1e-4}"
self_distill_train_steps="${SELF_DISTILL_TRAIN_STEPS:-$train_steps}"
self_distill_epochs="${SELF_DISTILL_EPOCHS:-50}"
self_distill_batch_size="${SELF_DISTILL_BATCH_SIZE:-16}"
self_distill_lr="${SELF_DISTILL_LR:-5e-5}"
self_distill_weight_decay="${SELF_DISTILL_WEIGHT_DECAY:-1e-4}"
self_distill_sup_weight="${SELF_DISTILL_SUP_WEIGHT:-1.0}"
self_distill_feat_weight="${SELF_DISTILL_FEAT_WEIGHT:-0.1}"
self_distill_action_weight="${SELF_DISTILL_ACTION_WEIGHT:-0.0}"
self_distill_init_from_teacher="${SELF_DISTILL_INIT_FROM_TEACHER:-true}"

max_vel="${MAX_VEL:-1.0}"
max_yaw_rate="${MAX_YAW_RATE:-15.0}"
max_speed_norm="${MAX_SPEED_NORM:-1.0}"
action_sequence_horizon="${ACTION_SEQUENCE_HORIZON:-32}"
action_video_freq_ratio="${ACTION_VIDEO_FREQ_RATIO:-4}"
diffusion_steps="${DIFFUSION_STEPS:-20}"
sampling_steps="${SAMPLING_STEPS:-8}"

target_token_fusion_mode="${TARGET_TOKEN_FUSION_MODE:-concat}"
train_next_target_relative="${TRAIN_NEXT_TARGET_RELATIVE:-false}"
next_target_relative_loss_weight="${NEXT_TARGET_RELATIVE_LOSS_WEIGHT:-1.0}"
prior_target_relative_loss_weight="${PRIOR_TARGET_RELATIVE_LOSS_WEIGHT:-0.2}"
direct_action_loss_weight="${DIRECT_ACTION_LOSS_WEIGHT:-1.0}"
action_yaw_loss_weight="${ACTION_YAW_LOSS_WEIGHT:-10.0}"
x0_action_loss_weight="${X0_ACTION_LOSS_WEIGHT:-0.0}"

target_relative_context_scale="${TARGET_RELATIVE_CONTEXT_SCALE:-1.0}"
target_relative_token_scale="${TARGET_RELATIVE_TOKEN_SCALE:-1.0}"
target_relative_context_hidden_dim="${TARGET_RELATIVE_CONTEXT_HIDDEN_DIM:-512}"

use_wan22_encoders="${USE_WAN22_ENCODERS:-true}"
wan22_model_base_path="${WAN22_MODEL_BASE_PATH:-$root_dir/model}"
wan22_fastwam_src_path="${WAN22_FASTWAM_SRC_PATH:-$root_dir/model/FastWAM/src}"
wan22_skip_download="${WAN22_SKIP_DOWNLOAD:-false}"
wan22_text_context_length="${WAN22_TEXT_CONTEXT_LENGTH:-512}"
wan22_text_encode_batch_size="${WAN22_TEXT_ENCODE_BATCH_SIZE:-4}"
wan_latent_cache_root="${WAN_LATENT_CACHE_ROOT:-$root_dir/latents}"
fastwam_skip_dit_load_from_pretrain="${FASTWAM_SKIP_DIT_LOAD_FROM_PRETRAIN:-false}"
fastwam_action_dit_pretrained_path="${FASTWAM_ACTION_DIT_PRETRAINED_PATH:-}"
fastwam_mot_checkpoint_mixed_attn="${FASTWAM_MOT_CHECKPOINT_MIXED_ATTN:-true}"

if [[ "$use_wan22_encoders" == "true" && ! -d "$wan22_fastwam_src_path/fastwam" ]]; then
  echo "[ERROR] FastWAM source not found at: $wan22_fastwam_src_path" >&2
  echo "        Set WAN22_FASTWAM_SRC_PATH=/path/to/FastWAM/src or place it under model/FastWAM." >&2
  exit 1
fi
export FASTWAM_REPO="${FASTWAM_REPO:-$wan22_fastwam_src_path}"
export PYTHONPATH="$wan22_fastwam_src_path:${PYTHONPATH:-}"

use_swanlab="${USE_SWANLAB:-true}"
swanlab_project="${SWANLAB_PROJECT:-WAM-FastWAM}"
swanlab_experiment_prefix="${SWANLAB_EXPERIMENT_PREFIX:-${EXP_NAME}_$(date +%Y%m%d-%H%M%S)}"
swanlab_workspace="${SWANLAB_WORKSPACE:-}"
swanlab_api_key="${SWANLAB_API_KEY:-}"
swanlab_log_dir="${SWANLAB_LOG_DIR:-$exp_root/swanlab_logs}"
swanlab_mode="${SWANLAB_MODE:-cloud}"
mkdir -p "$swanlab_log_dir"
if [[ "$use_swanlab" == "true" || "$use_swanlab" == "1" ]]; then
  export SWANLAB_NO_INTERACTIVE=1
  export SWANLAB_LOG_DIR="$swanlab_log_dir"
  export SWANLAB_DIR="$swanlab_log_dir"
  if [[ -n "$swanlab_api_key" ]]; then
    export SWANLAB_API_KEY="$swanlab_api_key"
  fi
fi

TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3}"
EVAL_GPU_ID="${EVAL_GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"

train_num_gpus="${TRAIN_NUM_GPUS:-}"
if [[ -z "$train_num_gpus" ]]; then
  train_num_gpus="$("$PYTHON_BIN" - <<'PY'
import os
ids = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(max(len(ids), 1))
PY
)"
fi

eval_scene_list="${EVAL_SCENE_LIST:-City_1,City_2,City_3}"
eval_trajectory_range="${EVAL_TRAJECTORY_RANGE:-451-500}"
eval_max_trajectories="${EVAL_MAX_TRAJECTORIES:-0}"
eval_max_steps="${EVAL_MAX_STEPS:-0}"
eval_visualize_trajectory_keys="${EVAL_VISUALIZE_TRAJECTORY_KEYS:-City_1/trajectory_0451}"
eval_save_transformer_attention_maps="${EVAL_SAVE_TRANSFORMER_ATTENTION_MAPS:-true}"
predicted_video_latent_frames="${PREDICTED_VIDEO_LATENT_FRAMES:-3}"
sim_server_host="${SIM_SERVER_HOST:-127.0.0.1}"
sim_server_port="${SIM_SERVER_PORT:-30000}"
scene_index="${SCENE_INDEX:-1}"
capture_distance="${CAPTURE_DISTANCE:-10.0}"
require_visibility_for_success="${REQUIRE_VISIBILITY_FOR_SUCCESS:-false}"
stop_on_collision="${STOP_ON_COLLISION:-true}"

csv_to_array() {
  local raw="$1"
  local -n out_ref="$2"
  out_ref=()
  IFS=',' read -ra parts <<< "$raw"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    if [[ -n "$part" ]]; then
      out_ref+=("$part")
    fi
  done
}

experiment_dir() {
  case "$1" in
    fastwam_global|fastwam_target_relative_token|self_distill_target_relative_token_to_global) echo "$model_root/$1" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'. Valid: fastwam_global, fastwam_target_relative_token, self_distill_target_relative_token_to_global." >&2
      exit 1
      ;;
  esac
}

experiment_uses_diffusion() {
  case "$1" in
    fastwam_global|fastwam_target_relative_token|self_distill_target_relative_token_to_global) echo "true" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'." >&2
      exit 1
      ;;
  esac
}

experiment_uses_fastwam() {
  case "$1" in
    fastwam_global|fastwam_target_relative_token|self_distill_target_relative_token_to_global) echo "true" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'." >&2
      exit 1
      ;;
  esac
}

experiment_uses_target_relative_context() {
  case "$1" in
    fastwam_target_relative_token) echo "true" ;;
    fastwam_global|self_distill_target_relative_token_to_global) echo "false" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'." >&2
      exit 1
      ;;
  esac
}

experiment_port() {
  case "$1" in
    fastwam_global) echo 29621 ;;
    fastwam_target_relative_token) echo 29625 ;;
    self_distill_target_relative_token_to_global) echo 29634 ;;
    *) echo 29629 ;;
  esac
}

eval_checkpoint_for_experiment() {
  case "$1" in
    fastwam_global|fastwam_target_relative_token|self_distill_target_relative_token_to_global) echo "$(experiment_dir "$1")/best.pt" ;;
    *)
      echo "[ERROR] Unknown eval experiment '$1'." >&2
      exit 1
      ;;
  esac
}

distill_teacher_checkpoint_for_experiment() {
  case "$1" in
    self_distill_target_relative_token_to_global) echo "$(experiment_dir fastwam_target_relative_token)/best.pt" ;;
    *)
      echo "[ERROR] Unknown self-distill experiment '$1'." >&2
      exit 1
      ;;
  esac
}

distill_teacher_uses_target_relative_context() {
  case "$1" in
    self_distill_target_relative_token_to_global) echo "true" ;;
    *)
      echo "[ERROR] Unknown self-distill experiment '$1'." >&2
      exit 1
      ;;
  esac
}

checkpoint_matches_train_config() {
  local ckpt_path="$1"
  local expected_target_context="$2"
  "$PYTHON_BIN" - "$ckpt_path" \
    "$scene_list" \
    "$trajectory_range" \
    "$expected_target_context" \
    "$target_relative_context_scale" \
    "$target_relative_token_scale" \
    "$target_relative_context_hidden_dim" \
    "$action_video_freq_ratio" \
    "$action_sequence_horizon" \
    "$target_token_fusion_mode" <<'PY'
import math
import sys
from pathlib import Path

import torch

ckpt = Path(sys.argv[1])
expected = {
    "scene_list": sys.argv[2],
    "trajectory_range": sys.argv[3],
    "use_target_relative_context": sys.argv[4].lower() == "true",
    "target_relative_context_scale": float(sys.argv[5]),
    "target_relative_token_scale": float(sys.argv[6]),
    "target_relative_context_hidden_dim": int(sys.argv[7]),
    "fastwam_action_video_freq_ratio": int(sys.argv[8]),
    "action_sequence_horizon": int(sys.argv[9]),
    "target_token_fusion_mode": sys.argv[10],
}
if not ckpt.exists():
    sys.exit(1)
try:
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
except Exception:
    sys.exit(1)
cfg = data.get("cfg") if isinstance(data, dict) else {}
if not isinstance(cfg, dict):
    sys.exit(1)

def same_float(key):
    try:
        actual = float(cfg.get(key, "nan"))
    except (TypeError, ValueError):
        return False
    return math.isclose(actual, float(expected[key]), rel_tol=1e-6, abs_tol=1e-8)

checks = [
    bool(cfg.get("use_target_relative_context", False)) == expected["use_target_relative_context"],
    same_float("target_relative_context_scale"),
    same_float("target_relative_token_scale"),
    int(cfg.get("target_relative_context_hidden_dim", -1)) == expected["target_relative_context_hidden_dim"],
    int(cfg.get("fastwam_action_video_freq_ratio", -1)) == expected["fastwam_action_video_freq_ratio"],
    int(cfg.get("action_sequence_horizon", -1)) == expected["action_sequence_horizon"],
    str(cfg.get("target_token_fusion_mode", "")) == expected["target_token_fusion_mode"],
]
if "run_args" in data and isinstance(data["run_args"], dict):
    args = data["run_args"]
    checks.extend([
        str(args.get("scene_list", "")) == expected["scene_list"],
        str(args.get("trajectory_range", "")) == expected["trajectory_range"],
    ])
sys.exit(0 if all(checks) else 1)
PY
}

self_distill_checkpoint_matches_config() {
  local ckpt_path="$1"
  local expected_teacher_ckpt="$2"
  local expected_teacher_target_context="$3"
  local expected_student_target_context="$4"
  "$PYTHON_BIN" - "$ckpt_path" \
    "$scene_list" \
    "$trajectory_range" \
    "$expected_teacher_ckpt" \
    "$expected_teacher_target_context" \
    "$expected_student_target_context" \
    "$target_relative_context_scale" \
    "$target_relative_token_scale" \
    "$target_relative_context_hidden_dim" \
    "$action_video_freq_ratio" \
    "$action_sequence_horizon" \
    "$target_token_fusion_mode" <<'PY'
import math
import sys
from pathlib import Path

import torch

ckpt = Path(sys.argv[1])
expected = {
    "scene_list": sys.argv[2],
    "trajectory_range": sys.argv[3],
    "teacher_ckpt": str(Path(sys.argv[4]).resolve()),
    "teacher_target_context": sys.argv[5].lower() == "true",
    "student_target_context": sys.argv[6].lower() == "true",
    "target_relative_context_scale": float(sys.argv[7]),
    "target_relative_token_scale": float(sys.argv[8]),
    "target_relative_context_hidden_dim": int(sys.argv[9]),
    "fastwam_action_video_freq_ratio": int(sys.argv[10]),
    "action_sequence_horizon": int(sys.argv[11]),
    "target_token_fusion_mode": sys.argv[12],
}
if not ckpt.exists():
    sys.exit(1)
try:
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
except Exception:
    sys.exit(1)
cfg = data.get("cfg") if isinstance(data, dict) else {}
teacher_cfg = data.get("teacher_cfg") if isinstance(data, dict) else {}
args = data.get("args") if isinstance(data, dict) else {}
if not isinstance(cfg, dict) or not isinstance(teacher_cfg, dict):
    sys.exit(1)

def same_float(obj, key):
    try:
        actual = float(obj.get(key, "nan"))
    except (TypeError, ValueError):
        return False
    return math.isclose(actual, float(expected[key]), rel_tol=1e-6, abs_tol=1e-8)

checks = [
    bool(teacher_cfg.get("use_target_relative_context", False)) == expected["teacher_target_context"],
    bool(cfg.get("use_target_relative_context", False)) == expected["student_target_context"],
    same_float(cfg, "target_relative_context_scale"),
    same_float(cfg, "target_relative_token_scale"),
    int(cfg.get("target_relative_context_hidden_dim", -1)) == expected["target_relative_context_hidden_dim"],
    int(cfg.get("fastwam_action_video_freq_ratio", -1)) == expected["fastwam_action_video_freq_ratio"],
    int(cfg.get("action_sequence_horizon", -1)) == expected["action_sequence_horizon"],
    str(cfg.get("target_token_fusion_mode", "")) == expected["target_token_fusion_mode"],
]
if isinstance(args, dict):
    checks.extend([
        str(args.get("scene_list", "")) == expected["scene_list"],
        str(args.get("trajectory_range", "")) == expected["trajectory_range"],
        str(Path(str(args.get("teacher_ckpt", ""))).resolve()) == expected["teacher_ckpt"],
        bool(args.get("student_use_target_relative_context", True)) == expected["student_target_context"],
    ])
sys.exit(0 if all(checks) else 1)
PY
}

summary_matches_eval_config() {
  local summary_path="$1"
  local expected_scene_list="$2"
  local expected_trajectory_range="$3"
  local expected_target_context="$4"
  local expected_ckpt="$5"
  local expected_visualize_trajectory_keys="${6:-}"
  "$PYTHON_BIN" - "$summary_path" "$expected_scene_list" "$expected_trajectory_range" "$expected_target_context" "$expected_ckpt" "$expected_visualize_trajectory_keys" "$target_relative_context_scale" "$target_relative_token_scale" "$target_relative_context_hidden_dim" "$sampling_steps" "$eval_save_transformer_attention_maps" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

summary = Path(sys.argv[1])
expected_scene = sys.argv[2]
expected_range = sys.argv[3]
expected_target_context = sys.argv[4].lower()
expected_ckpt = str(Path(sys.argv[5]).resolve())
expected_visualize_keys = sys.argv[6]
expected_target_context_scale = float(sys.argv[7])
expected_target_token_scale = float(sys.argv[8])
expected_target_hidden = int(sys.argv[9])
expected_sampling_steps = int(sys.argv[10])
expected_attention_maps = sys.argv[11].lower() in {"1", "true", "yes", "on"}
if not summary.exists():
    sys.exit(1)
try:
    data = json.loads(summary.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)
args = data.get("args") if isinstance(data, dict) else {}
if not isinstance(args, dict):
    sys.exit(1)
if str(args.get("scene_list", "")) != expected_scene:
    sys.exit(1)
if str(args.get("trajectory_range", "")) != expected_range:
    sys.exit(1)
if str(args.get("use_target_relative_context", "")).lower() != expected_target_context:
    sys.exit(1)
if str(Path(str(args.get("checkpoint", ""))).resolve()) != expected_ckpt:
    sys.exit(1)
if str(args.get("visualize_trajectory_keys", "")) != expected_visualize_keys:
    sys.exit(1)
if int(args.get("sampling_steps", -1)) != expected_sampling_steps:
    sys.exit(1)
resolved_cfg = data.get("resolved_cfg") if isinstance(data, dict) else {}
if not isinstance(resolved_cfg, dict):
    resolved_cfg = {}
if not math.isclose(float(resolved_cfg.get("target_relative_context_scale", float("nan"))), expected_target_context_scale, rel_tol=1e-6, abs_tol=1e-8):
    sys.exit(1)
if not math.isclose(float(resolved_cfg.get("target_relative_token_scale", float("nan"))), expected_target_token_scale, rel_tol=1e-6, abs_tol=1e-8):
    sys.exit(1)
if int(resolved_cfg.get("target_relative_context_hidden_dim", -1)) != expected_target_hidden:
    sys.exit(1)
if int(data.get("num_trajectories") or 0) <= 0:
    sys.exit(1)

def tokens(raw):
    out = []
    for item in re.split(r"[,\s]+", str(raw or "")):
        item = item.strip().replace("\\", "/").replace(":", "/").strip("/")
        if item:
            out.append(item)
    return out

visual_tokens = tokens(expected_visualize_keys)
if visual_tokens and not any(t.lower() in {"all", "*", "none", "false", "off", "0"} for t in visual_tokens):
    out_dir = summary.parent
    expected_dirs = ["rgb"] if bool(args.get("save_rgb", True)) else []
    if expected_attention_maps:
        expected_dirs.append("last_transformer_attention_maps")
    for token in visual_tokens:
        for candidate in [token, re.sub(r"/(\d+)$", lambda m: f"/trajectory_{int(m.group(1)):04d}", token)]:
            traj_dir = out_dir / candidate
            if not expected_dirs:
                continue
            if not traj_dir.exists():
                sys.exit(1)
            for dirname in expected_dirs:
                asset_dir = traj_dir / dirname
                if not asset_dir.exists() or not any(asset_dir.rglob("*.png")):
                    sys.exit(1)
sys.exit(0)
PY
}

write_manifest() {
  cat > "$exp_root/manifest.txt" <<EOF
experiment_name=$EXP_NAME
experiment_root=$exp_root
model_root=$model_root
python=$PYTHON_BIN
conda_env=${CONDA_DEFAULT_ENV}

main_experiments=${EXPERIMENTS}
fastwam_global=global image + instruction + FastWAM video/action MoT
fastwam_target_relative_token=baseline + body-frame target position token appended to FastWAM text context
self_distill_target_relative_token_to_global=target-relative-token teacher -> global student
distill_experiments=${DISTILL_EXPERIMENTS}
eval_extra_experiments=${EVAL_EXTRA_EXPERIMENTS}

scene_list=${scene_list}
trajectory_range=${trajectory_range}
val_ratio=${val_ratio}
split_seed=${split_seed}
train_steps=${train_steps}
model_train_epochs=${model_train_epochs}
teacher_batch_size_per_gpu=${teacher_batch_size}
seq_len=${seq_len}
image_size=${image_size}
num_workers=${num_workers}
teacher_lr=${teacher_lr}
teacher_weight_decay=${teacher_weight_decay}
self_distill_train_steps=${self_distill_train_steps}
self_distill_epochs=${self_distill_epochs}
self_distill_batch_size=${self_distill_batch_size}
self_distill_lr=${self_distill_lr}
self_distill_weight_decay=${self_distill_weight_decay}
self_distill_sup_weight=${self_distill_sup_weight}
self_distill_feat_weight=${self_distill_feat_weight}
self_distill_action_weight=${self_distill_action_weight}
self_distill_init_from_teacher=${self_distill_init_from_teacher}
train_cuda_visible_devices=${TRAIN_GPU_IDS}
train_num_gpus=${train_num_gpus}
use_deepspeed=${USE_DEEPSPEED}
deepspeed_offload_optimizer=${DEEPSPEED_OFFLOAD_OPTIMIZER}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
checkpoint_save_every_epochs=${CHECKPOINT_SAVE_EVERY_EPOCHS}
save_best_checkpoint=${SAVE_BEST_CHECKPOINT}
save_optimizer_state=${SAVE_OPTIMIZER_STATE}

low_dim_target_input=off
legacy_target_locator=removed
architecture=fastwam_video_action_mot_no_rssm
target_token_fusion_mode=${target_token_fusion_mode}
train_next_target_relative=${train_next_target_relative}
next_target_relative_loss_weight=${next_target_relative_loss_weight}
prior_target_relative_loss_weight=${prior_target_relative_loss_weight}
direct_action_loss_weight=${direct_action_loss_weight}
action_yaw_loss_weight=${action_yaw_loss_weight}
x0_action_loss_weight=${x0_action_loss_weight}
action_sequence_horizon=${action_sequence_horizon}
action_video_freq_ratio=${action_video_freq_ratio}
diffusion_steps=${diffusion_steps}
sampling_steps=${sampling_steps}
target_relative_context_scale=${target_relative_context_scale}
target_relative_token_scale=${target_relative_token_scale}
target_relative_context_hidden_dim=${target_relative_context_hidden_dim}
use_wan22_encoders=${use_wan22_encoders}
wan22_model_base_path=${wan22_model_base_path}
wan22_fastwam_src_path=${wan22_fastwam_src_path}
wan22_skip_download=${wan22_skip_download}
wan22_text_context_length=${wan22_text_context_length}
wan22_text_encode_batch_size=${wan22_text_encode_batch_size}
swanlab_enabled=${use_swanlab}
swanlab_project=${swanlab_project}
swanlab_experiment_prefix=${swanlab_experiment_prefix}
swanlab_workspace=${swanlab_workspace}
swanlab_log_dir=${swanlab_log_dir}
swanlab_mode=${swanlab_mode}

run_online_eval=${RUN_ONLINE_EVAL}
eval_cuda_visible_devices=${EVAL_GPU_ID}
eval_scene_list=${eval_scene_list}
eval_trajectory_range=${eval_trajectory_range}
eval_visualize_trajectory_keys=${eval_visualize_trajectory_keys}
eval_save_transformer_attention_maps=${eval_save_transformer_attention_maps}
capture_distance=${capture_distance}
require_visibility_for_success=${require_visibility_for_success}
stop_on_collision=${stop_on_collision}
EOF
}

run_teacher() {
  local name="$1"
  local save_dir
  save_dir="$(experiment_dir "$name")"
  local use_diffusion_actor
  use_diffusion_actor="$(experiment_uses_diffusion "$name")"
  local use_fastwam_mot
  use_fastwam_mot="$(experiment_uses_fastwam "$name")"
  local use_target_relative_context
  use_target_relative_context="$(experiment_uses_target_relative_context "$name")"
  local master_port
  master_port="$(experiment_port "$name")"
  local log_file="$log_dir/${name}.log"
  local resume_ckpt="$save_dir/last.pt"
  local resume_args=()

  if [[ "$SKIP_EXISTING_TRAIN" == "true" && -f "$save_dir/done.marker" && -f "$save_dir/best.pt" ]]; then
    if checkpoint_matches_train_config "$save_dir/best.pt" "$use_target_relative_context"; then
      echo "[train-skip] ${name}: existing checkpoint matches requested train config"
      return 0
    fi
    echo "[train-rerun] ${name}: existing checkpoint does not match requested train config"
  fi
  if [[ -f "$resume_ckpt" ]] && { [[ ! -f "$save_dir/done.marker" ]] || [[ ! -f "$save_dir/best.pt" ]]; }; then
    resume_args+=(--resume "$resume_ckpt")
  fi

  mkdir -p "$save_dir"
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"

  echo "============================================================" | tee "$log_file"
  echo "[train] ${name}" | tee -a "$log_file"
  echo "save_dir=${save_dir}" | tee -a "$log_file"
  echo "use_diffusion_actor=${use_diffusion_actor}" | tee -a "$log_file"
  echo "use_fastwam_mot=${use_fastwam_mot}" | tee -a "$log_file"
  echo "use_target_relative_context=${use_target_relative_context}" | tee -a "$log_file"
  echo "target_relative_context_scale=${target_relative_context_scale}" | tee -a "$log_file"
  echo "target_relative_token_scale=${target_relative_token_scale}" | tee -a "$log_file"
  echo "low_dim_target_input=off" | tee -a "$log_file"
  echo "legacy_target_locator=removed" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$log_file"
  echo "train_num_gpus=${train_num_gpus}" | tee -a "$log_file"
  echo "train_steps=${train_steps}" | tee -a "$log_file"
  echo "epochs=${model_train_epochs}" | tee -a "$log_file"
  if [[ ${#resume_args[@]} -gt 0 ]]; then
    echo "resume=${resume_ckpt}" | tee -a "$log_file"
  else
    echo "resume=none" | tee -a "$log_file"
  fi
  echo "use_deepspeed=${USE_DEEPSPEED}" | tee -a "$log_file"
  echo "swanlab=${use_swanlab}, project=${swanlab_project}, run=${swanlab_experiment_prefix}_${name}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  train_launcher=("$PYTHON_BIN" -m train.train_teacher)
  if [[ "$USE_DEEPSPEED" == "true" || "$USE_DEEPSPEED" == "1" ]]; then
    train_launcher=(env -u CUDA_VISIBLE_DEVICES "$PYTHON_BIN" -m deepspeed.launcher.runner --include "localhost:${TRAIN_GPU_IDS}" --master_port "$master_port" --module train.train_teacher)
  fi

  "${train_launcher[@]}" \
    --dataset-root "$dataset_root" \
    --scene-list "$scene_list" \
    --trajectory-range "$trajectory_range" \
    --save-dir "$save_dir" \
    --epochs "$model_train_epochs" \
    --max-train-steps "$train_steps" \
    --batch-size "$teacher_batch_size" \
    --seq-len "$seq_len" \
    --image-size "$image_size" \
    --wan-latent-cache-root "$wan_latent_cache_root" \
    --val-ratio "$val_ratio" \
    --split-seed "$split_seed" \
    --lr "$teacher_lr" \
    --weight-decay "$teacher_weight_decay" \
    --max-vel "$max_vel" \
    --max-yaw-rate "$max_yaw_rate" \
    --max-speed-norm "$max_speed_norm" \
    --action-sequence-horizon "$action_sequence_horizon" \
    --action-video-freq-ratio "$action_video_freq_ratio" \
    --diffusion-steps "$diffusion_steps" \
    --sampling-steps "$sampling_steps" \
    --num-workers "$num_workers" \
    --train-next-target-relative "$train_next_target_relative" \
    --next-target-relative-loss-weight "$next_target_relative_loss_weight" \
    --prior-target-relative-loss-weight "$prior_target_relative_loss_weight" \
    --direct-action-loss-weight "$direct_action_loss_weight" \
    --action-yaw-loss-weight "$action_yaw_loss_weight" \
    --x0-action-loss-weight "$x0_action_loss_weight" \
    --use-target-relative-context "$use_target_relative_context" \
    --target-relative-context-scale "$target_relative_context_scale" \
    --target-relative-token-scale "$target_relative_token_scale" \
    --target-relative-context-hidden-dim "$target_relative_context_hidden_dim" \
    --use-wan22-encoders "$use_wan22_encoders" \
    --wan22-model-base-path "$wan22_model_base_path" \
    --wan22-fastwam-src-path "$wan22_fastwam_src_path" \
    --wan22-skip-download "$wan22_skip_download" \
    --wan22-text-context-length "$wan22_text_context_length" \
    --wan22-text-encode-batch-size "$wan22_text_encode_batch_size" \
    --target-token-fusion-mode "$target_token_fusion_mode" \
    --use-diffusion-actor "$use_diffusion_actor" \
    --use-fastwam-mot "$use_fastwam_mot" \
    --fastwam-lambda-action 1.0 \
    --fastwam-lambda-video 1.0 \
    --fastwam-skip-dit-load-from-pretrain "$fastwam_skip_dit_load_from_pretrain" \
    --fastwam-action-dit-pretrained-path "$fastwam_action_dit_pretrained_path" \
    --fastwam-mot-checkpoint-mixed-attn "$fastwam_mot_checkpoint_mixed_attn" \
    --save-every-epochs "$CHECKPOINT_SAVE_EVERY_EPOCHS" \
    --save-best-checkpoint "$SAVE_BEST_CHECKPOINT" \
    --save-optimizer-state "$SAVE_OPTIMIZER_STATE" \
    --deepspeed-offload-optimizer "$DEEPSPEED_OFFLOAD_OPTIMIZER" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --use-swanlab "$use_swanlab" \
    --swanlab-project "$swanlab_project" \
    --swanlab-experiment-name "${swanlab_experiment_prefix}_${name}" \
    --swanlab-workspace "$swanlab_workspace" \
    --swanlab-log-dir "$swanlab_log_dir" \
    --swanlab-mode "$swanlab_mode" \
    "${resume_args[@]}" \
    $(if [[ "$USE_DEEPSPEED" == "true" || "$USE_DEEPSPEED" == "1" ]]; then printf '%s' '--deepspeed'; fi) \
    --multi-gpu \
    2>&1 | tee -a "$log_file"
}

run_self_distill() {
  local name="$1"
  local save_dir
  save_dir="$(experiment_dir "$name")"
  local teacher_ckpt
  teacher_ckpt="$(distill_teacher_checkpoint_for_experiment "$name")"
  local teacher_target_context
  teacher_target_context="$(distill_teacher_uses_target_relative_context "$name")"
  local student_target_context
  student_target_context="$(experiment_uses_target_relative_context "$name")"
  local master_port
  master_port="$(experiment_port "$name")"
  local log_file="$log_dir/${name}.log"
  local resume_ckpt="$save_dir/last.pt"
  local resume_args=()
  local init_args=()

  if [[ ! -f "$teacher_ckpt" ]]; then
    echo "[ERROR] Missing teacher checkpoint for self-distill ${name}: $teacher_ckpt" >&2
    exit 1
  fi
  if [[ "$SKIP_EXISTING_SELF_DISTILL" == "true" && -f "$save_dir/done.marker" && -f "$save_dir/best.pt" ]]; then
    if self_distill_checkpoint_matches_config "$save_dir/best.pt" "$teacher_ckpt" "$teacher_target_context" "$student_target_context"; then
      echo "[self-distill-skip] ${name}: existing checkpoint matches requested distill config"
      return 0
    fi
    echo "[self-distill-rerun] ${name}: existing checkpoint does not match requested distill config"
  fi
  if [[ -f "$resume_ckpt" ]] && { [[ ! -f "$save_dir/done.marker" ]] || [[ ! -f "$save_dir/best.pt" ]]; }; then
    resume_args+=(--resume "$resume_ckpt")
  fi
  if [[ "$self_distill_init_from_teacher" == "true" || "$self_distill_init_from_teacher" == "1" ]]; then
    init_args+=(--init-student-from-teacher)
  else
    init_args+=(--student-init-random)
  fi

  mkdir -p "$save_dir"
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"

  echo "============================================================" | tee "$log_file"
  echo "[self-distill] ${name}" | tee -a "$log_file"
  echo "save_dir=${save_dir}" | tee -a "$log_file"
  echo "teacher_ckpt=${teacher_ckpt}" | tee -a "$log_file"
  echo "teacher_use_target_relative_context=${teacher_target_context}" | tee -a "$log_file"
  echo "student_use_target_relative_context=${student_target_context}" | tee -a "$log_file"
  echo "target_relative_context_scale=${target_relative_context_scale}" | tee -a "$log_file"
  echo "target_relative_token_scale=${target_relative_token_scale}" | tee -a "$log_file"
  echo "train_steps=${self_distill_train_steps}" | tee -a "$log_file"
  echo "epochs=${self_distill_epochs}" | tee -a "$log_file"
  echo "sup_weight=${self_distill_sup_weight}" | tee -a "$log_file"
  echo "feat_distill_weight=${self_distill_feat_weight}" | tee -a "$log_file"
  echo "action_distill_weight=${self_distill_action_weight}" | tee -a "$log_file"
  echo "init_student_from_teacher=${self_distill_init_from_teacher}" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$log_file"
  echo "train_num_gpus=${train_num_gpus}" | tee -a "$log_file"
  if [[ ${#resume_args[@]} -gt 0 ]]; then
    echo "resume=${resume_ckpt}" | tee -a "$log_file"
  else
    echo "resume=none" | tee -a "$log_file"
  fi
  echo "use_deepspeed=${USE_DEEPSPEED}" | tee -a "$log_file"
  echo "swanlab=${use_swanlab}, project=${swanlab_project}, run=${swanlab_experiment_prefix}_${name}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  distill_launcher=("$PYTHON_BIN" -m train.train_self_distill)
  if [[ "$USE_DEEPSPEED" == "true" || "$USE_DEEPSPEED" == "1" ]]; then
    distill_launcher=(env -u CUDA_VISIBLE_DEVICES "$PYTHON_BIN" -m deepspeed.launcher.runner --include "localhost:${TRAIN_GPU_IDS}" --master_port "$master_port" --module train.train_self_distill)
  fi

  "${distill_launcher[@]}" \
    --dataset-root "$dataset_root" \
    --scene-list "$scene_list" \
    --trajectory-range "$trajectory_range" \
    --save-dir "$save_dir" \
    --teacher-ckpt "$teacher_ckpt" \
    --epochs "$self_distill_epochs" \
    --max-train-steps "$self_distill_train_steps" \
    --batch-size "$self_distill_batch_size" \
    --seq-len "$seq_len" \
    --image-size "$image_size" \
    --wan-latent-cache-root "$wan_latent_cache_root" \
    --val-ratio "$val_ratio" \
    --split-seed "$split_seed" \
    --lr "$self_distill_lr" \
    --weight-decay "$self_distill_weight_decay" \
    --max-vel "$max_vel" \
    --max-yaw-rate "$max_yaw_rate" \
    --max-speed-norm "$max_speed_norm" \
    --action-sequence-horizon "$action_sequence_horizon" \
    --action-video-freq-ratio "$action_video_freq_ratio" \
    --diffusion-steps "$diffusion_steps" \
    --sampling-steps "$sampling_steps" \
    --num-workers "$num_workers" \
    --use-target-relative-context "$teacher_target_context" \
    --student-use-target-relative-context "$student_target_context" \
    --target-relative-context-scale "$target_relative_context_scale" \
    --target-relative-token-scale "$target_relative_token_scale" \
    --target-relative-context-hidden-dim "$target_relative_context_hidden_dim" \
    --use-wan22-encoders "$use_wan22_encoders" \
    --wan22-model-base-path "$wan22_model_base_path" \
    --wan22-fastwam-src-path "$wan22_fastwam_src_path" \
    --wan22-skip-download "$wan22_skip_download" \
    --wan22-text-context-length "$wan22_text_context_length" \
    --wan22-text-encode-batch-size "$wan22_text_encode_batch_size" \
    --target-token-fusion-mode "$target_token_fusion_mode" \
    --fastwam-skip-dit-load-from-pretrain "$fastwam_skip_dit_load_from_pretrain" \
    --fastwam-action-dit-pretrained-path "$fastwam_action_dit_pretrained_path" \
    --fastwam-mot-checkpoint-mixed-attn "$fastwam_mot_checkpoint_mixed_attn" \
    --sup-weight "$self_distill_sup_weight" \
    --feat-distill-weight "$self_distill_feat_weight" \
    --action-distill-weight "$self_distill_action_weight" \
    --save-every-epochs "$CHECKPOINT_SAVE_EVERY_EPOCHS" \
    --save-best-checkpoint "$SAVE_BEST_CHECKPOINT" \
    --save-optimizer-state "$SAVE_OPTIMIZER_STATE" \
    --deepspeed-offload-optimizer "$DEEPSPEED_OFFLOAD_OPTIMIZER" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --use-swanlab "$use_swanlab" \
    --swanlab-project "$swanlab_project" \
    --swanlab-experiment-name "${swanlab_experiment_prefix}_${name}" \
    --swanlab-workspace "$swanlab_workspace" \
    --swanlab-log-dir "$swanlab_log_dir" \
    --swanlab-mode "$swanlab_mode" \
    "${init_args[@]}" \
    "${resume_args[@]}" \
    $(if [[ "$USE_DEEPSPEED" == "true" || "$USE_DEEPSPEED" == "1" ]]; then printf '%s' '--deepspeed'; fi) \
    --multi-gpu \
    2>&1 | tee -a "$log_file"
}

run_online_eval() {
  local name="$1"
  local ckpt="$2"
  local use_diffusion_actor="$3"
  local use_target_relative_context="$4"
  local out_root="${5:-$eval_root}"
  local log_root="${6:-$eval_log_dir}"
  local scene_list_for_eval="${7:-$eval_scene_list}"
  local trajectory_range_for_eval="${8:-$eval_trajectory_range}"
  local out_dir="$out_root/$name"
  local log_file="$log_root/${name}.log"

  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] Missing checkpoint for eval: $ckpt" >&2
    exit 1
  fi
  if [[ "$SKIP_EXISTING_EVAL" == "true" && -f "$out_dir/summary.json" ]]; then
    if summary_matches_eval_config "$out_dir/summary.json" "$scene_list_for_eval" "$trajectory_range_for_eval" "$use_target_relative_context" "$ckpt" "$eval_visualize_trajectory_keys"; then
      echo "[eval-skip] ${name}: $out_dir/summary.json matches requested eval config"
      return 0
    fi
    echo "[eval-resume] ${name}: existing summary.json does not match requested eval config; rerun"
  fi

  mkdir -p "$out_dir" "$log_root"
  export CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID"
  export DAGGER_MULTI_WORKER=1

  extra_eval_args=()
  if [[ "$eval_max_trajectories" != "0" ]]; then
    extra_eval_args+=(--max-trajectories "$eval_max_trajectories")
  fi
  if [[ "$eval_max_steps" != "0" ]]; then
    extra_eval_args+=(--max-steps "$eval_max_steps")
  fi
  if [[ "$require_visibility_for_success" == "true" || "$require_visibility_for_success" == "1" ]]; then
    extra_eval_args+=(--require-visibility-for-success)
  fi
  if [[ "$stop_on_collision" == "false" || "$stop_on_collision" == "0" ]]; then
    extra_eval_args+=(--no-stop-on-collision)
  fi
  if [[ "$eval_save_transformer_attention_maps" == "true" || "$eval_save_transformer_attention_maps" == "1" ]]; then
    extra_eval_args+=(--save-transformer-attention-maps)
  else
    extra_eval_args+=(--no-save-transformer-attention-maps)
  fi

  echo "============================================================" | tee "$log_file"
  echo "[online-eval] ${name}" | tee -a "$log_file"
  echo "checkpoint=${ckpt}" | tee -a "$log_file"
  echo "output=${out_dir}" | tee -a "$log_file"
  echo "scene_list=${scene_list_for_eval}" | tee -a "$log_file"
  echo "trajectory_range=${trajectory_range_for_eval}" | tee -a "$log_file"
  echo "visualize_trajectory_keys=${eval_visualize_trajectory_keys}" | tee -a "$log_file"
  echo "save_transformer_attention_maps=${eval_save_transformer_attention_maps}" | tee -a "$log_file"
  echo "use_diffusion_actor=${use_diffusion_actor}" | tee -a "$log_file"
  echo "use_target_relative_context=${use_target_relative_context}" | tee -a "$log_file"
  echo "sampling_steps=${sampling_steps}" | tee -a "$log_file"
  echo "capture_distance=${capture_distance}" | tee -a "$log_file"
  echo "stop_on_collision=${stop_on_collision}" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  "$PYTHON_BIN" -m eval.online_eval_teacher \
    --dataset-root "$dataset_root" \
    --checkpoint "$ckpt" \
    --output-dir "$out_dir" \
    --executor-script "$executor_script" \
    --scene-list "$scene_list_for_eval" \
    --trajectory-range "$trajectory_range_for_eval" \
    --visualize-trajectory-keys "$eval_visualize_trajectory_keys" \
    --eval-split all \
    --sim-server-host "$sim_server_host" \
    --sim-server-port "$sim_server_port" \
    --scene-index "$scene_index" \
    --gpu-id "$EVAL_GPU_ID" \
    --device cuda \
    --max-vel "$max_vel" \
    --max-yaw-rate "$max_yaw_rate" \
    --max-speed-norm "$max_speed_norm" \
    --sampling-steps "$sampling_steps" \
    --capture-distance "$capture_distance" \
    --use-target-relative-context "$use_target_relative_context" \
    --target-relative-context-scale "$target_relative_context_scale" \
    --target-relative-token-scale "$target_relative_token_scale" \
    --target-relative-context-hidden-dim "$target_relative_context_hidden_dim" \
    --use-wan22-encoders "$use_wan22_encoders" \
    --wan22-model-base-path "$wan22_model_base_path" \
    --wan22-fastwam-src-path "$wan22_fastwam_src_path" \
    --wan22-skip-download "$wan22_skip_download" \
    --wan22-text-context-length "$wan22_text_context_length" \
    --wan22-text-encode-batch-size "$wan22_text_encode_batch_size" \
    --use-diffusion-actor "$use_diffusion_actor" \
    --no-save-predicted-video \
    --predicted-video-latent-frames "$predicted_video_latent_frames" \
    "${extra_eval_args[@]}" \
    2>&1 | tee -a "$log_file"
}

summarize_eval_results() {
  local summary_root="$1"
  shift
  local summary_title="${1:-online eval summary}"
  shift || true
  "$PYTHON_BIN" - "$summary_root" "$summary_title" "$eval_scene_list" "$eval_trajectory_range" "$@" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
title = sys.argv[2]
expected_scene_list = sys.argv[3]
expected_trajectory_range = sys.argv[4]
models = sys.argv[5:]
model_width = max([30] + [len(model) for model in models])

def num(x):
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")

def partial_count(model):
    partial = root / model / "summary_partial.json"
    if not partial.exists():
        return None
    try:
        return len(json.loads(partial.read_text(encoding="utf-8")).get("summaries", []))
    except Exception:
        return 0

def load_record(model):
    path = root / model / "summary.json"
    if not path.exists():
        n = partial_count(model)
        if n is None:
            return f"{model:{model_width}s} {'missing':>8s}"
        return f"{model:{model_width}s} {'partial':>8s} n={n}"
    data = json.loads(path.read_text(encoding="utf-8"))
    args = data.get("args", {})
    if str(args.get("scene_list", "")) != expected_scene_list or str(args.get("trajectory_range", "")) != expected_trajectory_range:
        return f"{model:{model_width}s} {'stale':>8s} summary_scene={args.get('scene_list')} summary_range={args.get('trajectory_range')}"
    sr = num(data.get("SR", data.get("success_rate")))
    atf = num(data.get("ATF", data.get("average_tracked_frames")))
    track = num(data.get("mean_effective_tracking_ratio", data.get("average_tracked_frame_ratio")))
    coll = num(data.get("collision_rate"))
    final_d = num(data.get("mean_final_distance"))
    mean_d = num(data.get("mean_distance"))
    failures = data.get("failure_reason_counts", {})
    failures_s = ",".join(f"{k}:{v}" for k, v in sorted(failures.items()))
    return (
        f"{model:{model_width}s} "
        f"{sr * 100:7.2f}% "
        f"{atf:8.2f} "
        f"{track * 100:7.2f}% "
        f"{coll * 100:7.2f}% "
        f"{final_d:9.2f} "
        f"{mean_d:9.2f} "
        f"{failures_s:>28s}"
    )

lines = [
    f"[ablation] {title}",
    (
        f"{'model':{model_width}s} {'SR':>8s} {'ATF':>8s} {'track%':>8s} "
        f"{'coll%':>8s} {'final_d':>9s} {'mean_d':>9s} {'failures':>28s}"
    ),
]
for model in models:
    lines.append(load_record(model))
print("\n" + "\n".join(lines))
report_dir = root.parent / "reports"
report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / "heldout_online_eval_report.txt"
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[ablation] saved eval report: {report_path}")
PY
}

write_manifest
csv_to_array "$EXPERIMENTS" experiment_names
csv_to_array "$DISTILL_EXPERIMENTS" distill_experiment_names
csv_to_array "$EVAL_EXTRA_EXPERIMENTS" eval_extra_experiment_names

echo "[ablation] experiment root: $exp_root"
echo "[ablation] experiments: ${experiment_names[*]}"
echo "[ablation] self-distill experiments: ${distill_experiment_names[*]}"
echo "[ablation] eval-only experiments: ${eval_extra_experiment_names[*]}"
echo "[ablation] train GPUs: $TRAIN_GPU_IDS (num_gpus=$train_num_gpus, deepspeed=$USE_DEEPSPEED)"
echo "[ablation] eval GPU: $EVAL_GPU_ID"
echo "[ablation] train_steps: $train_steps"
echo "[ablation] model_train_epochs: $model_train_epochs"
echo "[ablation] self_distill_train_steps: $self_distill_train_steps"
echo "[ablation] skip existing train/self-distill/eval: $SKIP_EXISTING_TRAIN / $SKIP_EXISTING_SELF_DISTILL / $SKIP_EXISTING_EVAL"
echo "[ablation] low-dimensional target vector is off; legacy target locator logic is removed"

if [[ "$RUN_TEACHER_ABLATIONS" == "true" ]]; then
  for name in "${experiment_names[@]}"; do
    run_teacher "$name"
  done
else
  echo "[ablation] RUN_TEACHER_ABLATIONS=false, skip teacher training"
fi

if [[ "$RUN_SELF_DISTILL" == "true" ]]; then
  for name in "${distill_experiment_names[@]}"; do
    run_self_distill "$name"
  done
else
  echo "[ablation] RUN_SELF_DISTILL=false, skip self-distillation"
fi

summary_models=()
for name in "${experiment_names[@]}"; do
  summary_models+=("$name")
done
for name in "${distill_experiment_names[@]}"; do
  summary_models+=("$name")
done
for name in "${eval_extra_experiment_names[@]}"; do
  summary_models+=("$name")
done

if [[ "$RUN_ONLINE_EVAL" == "true" ]]; then
  echo "[ablation] starting online eval"
  for name in "${experiment_names[@]}"; do
    ckpt="$(eval_checkpoint_for_experiment "$name")"
    run_online_eval \
      "$name" \
      "$ckpt" \
      "$(experiment_uses_diffusion "$name")" \
      "$(experiment_uses_target_relative_context "$name")"
  done
  for name in "${distill_experiment_names[@]}"; do
    ckpt="$(eval_checkpoint_for_experiment "$name")"
    run_online_eval \
      "$name" \
      "$ckpt" \
      "$(experiment_uses_diffusion "$name")" \
      "$(experiment_uses_target_relative_context "$name")"
  done
  for name in "${eval_extra_experiment_names[@]}"; do
    ckpt="$(eval_checkpoint_for_experiment "$name")"
    run_online_eval \
      "$name" \
      "$ckpt" \
      "$(experiment_uses_diffusion "$name")" \
      "$(experiment_uses_target_relative_context "$name")"
  done
  summarize_eval_results "$eval_root" "held-out online eval summary (${eval_scene_list} ${eval_trajectory_range})" "${summary_models[@]}"
else
  echo "[ablation] RUN_ONLINE_EVAL=false, skip online eval"
fi

echo "[ablation] finished: $exp_root"
