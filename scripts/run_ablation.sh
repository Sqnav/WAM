#!/usr/bin/env bash
set -euo pipefail

# WAM proposed-method validation suite.
#
# Current project assumptions:
#   - no low-dimensional target vector is fed to the policy input;
#   - local crop has been removed;
#   - FastWAM-style video/action MoT replaces RSSM;
#   - video expert and action expert share MoT attention during training;
#   - inference uses current-frame video tokens + action expert only;
#   - heatmap experiments use global image + patch-level target-projection bias
#     and remain available but are not part of the proposed-method defaults;
#   - target-relative auxiliary prediction heads are disabled by default.
#
# Main experiment table:
#   fastwam_global                              : baseline global image + instruction + FastWAM MoT
#   fastwam_target_belief_tracker                    : baseline + reference-guided temporal target belief tracker
#   self_distill_target_belief_tracker_to_global     : belief-tracker teacher -> global student
#   fastwam_latent_mpc                         : eval-only baseline checkpoint + latent receding-horizon scoring
#   fastwam_target_belief_tracker_latent_mpc   : eval-only belief-tracker checkpoint + latent receding-horizon scoring
#   self_distill_target_belief_tracker_to_global_latent_mpc
#                                               : eval-only distilled global checkpoint + latent receding-horizon scoring
#
# Existing done.marker and summary.json are skipped by default. Override with:
#   SKIP_EXISTING_TRAIN=false
#   SKIP_EXISTING_EVAL=false
#   SKIP_EXISTING_SELF_DISTILL=false

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

# =========================
# Experiment root
# =========================
EXP_NAME="${EXP_NAME:-fastwam_ablation}"
exp_root="${EXP_ROOT:-$root_dir/experiments}"
model_root="${MODEL_OUTPUT_ROOT:-$exp_root/models}"
log_dir="$exp_root/logs"
eval_root="${EVAL_OUTPUT_ROOT:-$exp_root/online_eval}"
eval_log_dir="${EVAL_LOG_DIR:-$exp_root/eval_logs}"

mkdir -p "$exp_root" "$model_root" "$log_dir" "$eval_root" "$eval_log_dir"

# =========================
# Stages
# =========================
RUN_TEACHER_ABLATIONS="${RUN_TEACHER_ABLATIONS:-true}"
RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-true}"
RUN_SELF_DISTILL="${RUN_SELF_DISTILL:-true}"
USE_DEEPSPEED="${USE_DEEPSPEED:-true}"
DEEPSPEED_OFFLOAD_OPTIMIZER="${DEEPSPEED_OFFLOAD_OPTIMIZER:-true}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
CHECKPOINT_SAVE_EVERY_EPOCHS="${CHECKPOINT_SAVE_EVERY_EPOCHS:-5}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-true}"
SAVE_OPTIMIZER_STATE="${SAVE_OPTIMIZER_STATE:-false}"

# Teacher experiments. Self-distillation experiments are listed separately
# below, but are still evaluated as first-class rows.
EXPERIMENTS="${EXPERIMENTS-fastwam_global,fastwam_target_belief_tracker}"
EVAL_EXTRA_EXPERIMENTS="${EVAL_EXTRA_EXPERIMENTS-fastwam_latent_mpc,fastwam_target_belief_tracker_latent_mpc,self_distill_target_belief_tracker_to_global_latent_mpc}"
DISTILL_EXPERIMENTS="${DISTILL_EXPERIMENTS-target_belief_tracker_to_global}"

SKIP_EXISTING_TRAIN="${SKIP_EXISTING_TRAIN:-true}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-true}"
SKIP_EXISTING_SELF_DISTILL="${SKIP_EXISTING_SELF_DISTILL:-true}"

# =========================
# Dataset
# =========================
scene_list="${SCENE_LIST:-City_1,City_2,City_3,City_4,City_5,City_6,City_7,City_8,City_9,City_10,City_11,City_12,City_13,City_14,City_15,City_16,City_17,City_18,City_19,City_20,City_21,City_22,City_23,City_24,City_25,City_26,City_27}"
trajectory_range="${TRAJECTORY_RANGE:-1-450}"
val_ratio="${VAL_RATIO:-0.0}"
split_seed="${SPLIT_SEED:-42}"

# =========================
# Teacher training
# =========================
train_steps="${TRAIN_STEPS:-0}"
teacher_epochs="${TEACHER_EPOCHS:-3}"
teacher_batch_size="${TEACHER_BATCH_SIZE:-16}"
seq_len="${SEQ_LEN:-33}"
image_size="${IMAGE_SIZE:-224}"
num_workers="${NUM_WORKERS:-8}"
teacher_lr="${TEACHER_LR:-1e-4}"
teacher_weight_decay="${TEACHER_WEIGHT_DECAY:-1e-4}"

max_vel="${MAX_VEL:-1.0}"
max_yaw_rate="${MAX_YAW_RATE:-15.0}"
max_speed_norm="${MAX_SPEED_NORM:-1.0}"
action_sequence_horizon="${ACTION_SEQUENCE_HORIZON:-32}"
action_video_freq_ratio="${ACTION_VIDEO_FREQ_RATIO:-4}"
diffusion_steps="${DIFFUSION_STEPS:-20}"
sampling_steps="${SAMPLING_STEPS:-20}"

target_token_fusion_mode="${TARGET_TOKEN_FUSION_MODE:-concat}"
train_next_target_relative="${TRAIN_NEXT_TARGET_RELATIVE:-false}"
next_target_relative_loss_weight="${NEXT_TARGET_RELATIVE_LOSS_WEIGHT:-1.0}"
prior_target_relative_loss_weight="${PRIOR_TARGET_RELATIVE_LOSS_WEIGHT:-0.2}"
direct_action_loss_weight="${DIRECT_ACTION_LOSS_WEIGHT:-1.0}"
action_yaw_loss_weight="${ACTION_YAW_LOSS_WEIGHT:-10.0}"
x0_action_loss_weight="${X0_ACTION_LOSS_WEIGHT:-0.0}"

# Heatmap is only active for *_heatmap rows.
use_attention_heatmap="${USE_ATTENTION_HEATMAP:-true}"
use_heatmap_tensor_encoder="${USE_HEATMAP_TENSOR_ENCODER:-true}"
heatmap_token_scale="${HEATMAP_TOKEN_SCALE:-1.0}"
fastwam_heatmap_context_grid="${FASTWAM_HEATMAP_CONTEXT_GRID:-4}"
visual_guidance_fov_deg="${VISUAL_GUIDANCE_FOV_DEG:-90.0}"
attention_heatmap_sigma="${ATTENTION_HEATMAP_SIGMA:-0.08}"
target_belief_token_scale="${TARGET_BELIEF_TOKEN_SCALE:-1.0}"
target_belief_update_rate="${TARGET_BELIEF_UPDATE_RATE:-0.25}"
target_belief_min_confidence="${TARGET_BELIEF_MIN_CONFIDENCE:-0.05}"
target_belief_temperature="${TARGET_BELIEF_TEMPERATURE:-0.07}"
target_belief_loss_weight="${TARGET_BELIEF_LOSS_WEIGHT:-0.1}"
target_belief_motion_weight="${TARGET_BELIEF_MOTION_WEIGHT:-0.25}"
target_belief_update_sharpness="${TARGET_BELIEF_UPDATE_SHARPNESS:-10.0}"
latent_mpc_candidate_count="${LATENT_MPC_CANDIDATE_COUNT:-4}"
latent_mpc_distance_weight="${LATENT_MPC_DISTANCE_WEIGHT:-0.0}"
latent_mpc_smooth_weight="${LATENT_MPC_SMOOTH_WEIGHT:-0.05}"
latent_mpc_action_weight="${LATENT_MPC_ACTION_WEIGHT:-0.02}"
latent_mpc_visual_weight="${LATENT_MPC_VISUAL_WEIGHT:-0.1}"
latent_mpc_latent_frames="${LATENT_MPC_LATENT_FRAMES:-3}"
latent_mpc_video_sampling_steps="${LATENT_MPC_VIDEO_SAMPLING_STEPS:-4}"
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

# =========================
# Self-distillation
# =========================
self_distill_epochs="${SELF_DISTILL_EPOCHS:-3}"
self_distill_batch_size="${SELF_DISTILL_BATCH_SIZE:-16}"
self_distill_lr="${SELF_DISTILL_LR:-5e-5}"
self_distill_weight_decay="${SELF_DISTILL_WEIGHT_DECAY:-1e-4}"
self_distill_sup_weight="${SELF_DISTILL_SUP_WEIGHT:-1.0}"
self_distill_feat_weight="${SELF_DISTILL_FEAT_WEIGHT:-0.1}"
self_distill_action_weight="${SELF_DISTILL_ACTION_WEIGHT:-0.0}"
self_distill_init_from_teacher="${SELF_DISTILL_INIT_FROM_TEACHER:-true}"

# =========================
# SwanLab visualization
# =========================
use_swanlab="${USE_SWANLAB:-true}"
swanlab_project="${SWANLAB_PROJECT:-WAM-FastWAM}"
swanlab_experiment_prefix="${SWANLAB_EXPERIMENT_PREFIX:-${EXP_NAME}_$(date +%Y%m%d-%H%M%S)}"
swanlab_workspace="${SWANLAB_WORKSPACE:-}"
swanlab_api_key="${SWANLAB_API_KEY:-1BWz76qt6gpB13YRrygMZ}"
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

# =========================
# GPU
# =========================
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-2,3,4,5}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
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

# =========================
# Online eval
# =========================
eval_scene_list="${EVAL_SCENE_LIST:-City_1}"
eval_trajectory_range="${EVAL_TRAJECTORY_RANGE:-451-500}"
eval_max_trajectories="${EVAL_MAX_TRAJECTORIES:-0}"
eval_max_steps="${EVAL_MAX_STEPS:-0}"
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
    fastwam_*) echo "$model_root/$1" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'. Valid examples: fastwam_global, fastwam_heatmap." >&2
      exit 1
      ;;
  esac
}

experiment_uses_diffusion() {
  case "$1" in
    fastwam_*|self_distill_*) echo "true" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'." >&2
      exit 1
      ;;
  esac
}

experiment_uses_fastwam() {
  case "$1" in
    fastwam_*) echo "true" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'." >&2
      exit 1
      ;;
  esac
}

experiment_uses_guidance() {
  case "$1" in
    fastwam_*heatmap*) echo "true" ;;
    fastwam_global|fastwam_target_belief_tracker|fastwam_latent_mpc|fastwam_target_belief_tracker_latent_mpc|self_distill_target_belief_tracker_to_global_latent_mpc) echo "false" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'." >&2
      exit 1
      ;;
  esac
}

experiment_uses_target_belief_tracker() {
  case "$1" in
    fastwam_target_belief_tracker|fastwam_target_belief_tracker_latent_mpc) echo "true" ;;
    fastwam_global|fastwam_latent_mpc|fastwam_heatmap|self_distill_target_belief_tracker_to_global_latent_mpc) echo "false" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'." >&2
      exit 1
      ;;
  esac
}

experiment_uses_latent_mpc() {
  case "$1" in
    fastwam_latent_mpc|fastwam_target_belief_tracker_latent_mpc|self_distill_target_belief_tracker_to_global_latent_mpc) echo "true" ;;
    fastwam_global|fastwam_target_belief_tracker|fastwam_heatmap) echo "false" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'." >&2
      exit 1
      ;;
  esac
}

experiment_port() {
  case "$1" in
    fastwam_global) echo 29621 ;;
    fastwam_heatmap) echo 29622 ;;
    fastwam_target_belief_tracker) echo 29625 ;;
    self_distill_heatmap_to_global) echo 29632 ;;
    self_distill_target_belief_tracker_to_global) echo 29634 ;;
    *) echo 29629 ;;
  esac
}

experiment_loss_weights() {
  case "$1" in
    fastwam_heatmap) echo "2.0 1.0" ;;
    *) echo "1.0 1.0" ;;
  esac
}

eval_checkpoint_for_experiment() {
  case "$1" in
    fastwam_latent_mpc) echo "$(experiment_dir fastwam_global)/best.pt" ;;
    fastwam_target_belief_tracker_latent_mpc) echo "$(experiment_dir fastwam_target_belief_tracker)/best.pt" ;;
    self_distill_target_belief_tracker_to_global_latent_mpc) echo "$model_root/self_distill_target_belief_tracker_to_global/best.pt" ;;
    fastwam_*) echo "$(experiment_dir "$1")/best.pt" ;;
    *)
      echo "[ERROR] Unknown eval experiment '$1'." >&2
      exit 1
      ;;
  esac
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
fastwam_target_belief_tracker=baseline + reference-guided temporal target belief tracker
fastwam_latent_mpc=eval-only baseline checkpoint + predicted-latent receding-horizon scoring
fastwam_target_belief_tracker_latent_mpc=eval-only belief-tracker checkpoint + predicted-latent receding-horizon scoring
self_distill_target_belief_tracker_to_global_latent_mpc=eval-only distilled global checkpoint + predicted-latent receding-horizon scoring
fastwam_heatmap=encoded heatmap tensor + FastWAM video/action MoT + action/video loss 2/1
eval_extra_experiments=${EVAL_EXTRA_EXPERIMENTS}
distill_experiments=${DISTILL_EXPERIMENTS}

scene_list=${scene_list}
trajectory_range=${trajectory_range}
val_ratio=${val_ratio}
split_seed=${split_seed}
train_steps=${train_steps}
teacher_epochs=${teacher_epochs}
teacher_batch_size_per_gpu=${teacher_batch_size}
seq_len=${seq_len}
image_size=${image_size}
num_workers=${num_workers}
teacher_lr=${teacher_lr}
teacher_weight_decay=${teacher_weight_decay}
train_cuda_visible_devices=${TRAIN_GPU_IDS}
train_num_gpus=${train_num_gpus}
use_deepspeed=${USE_DEEPSPEED}
deepspeed_offload_optimizer=${DEEPSPEED_OFFLOAD_OPTIMIZER}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
checkpoint_save_every_epochs=${CHECKPOINT_SAVE_EVERY_EPOCHS}
save_best_checkpoint=${SAVE_BEST_CHECKPOINT}
save_optimizer_state=${SAVE_OPTIMIZER_STATE}

low_dim_target_input=off
architecture=fastwam_video_action_mot_no_rssm
fastwam_loss_weights=per_experiment
local_crop=removed
prediction_heads=disabled_by_default
prediction_head_rollout_loss=removed
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
predicted_video_latent_frames=${predicted_video_latent_frames}
use_attention_heatmap=${use_attention_heatmap}
use_heatmap_tensor_encoder=${use_heatmap_tensor_encoder}
heatmap_token_scale=${heatmap_token_scale}
visual_guidance_fov_deg=${visual_guidance_fov_deg}
attention_heatmap_sigma=${attention_heatmap_sigma}
target_belief_token_scale=${target_belief_token_scale}
target_belief_update_rate=${target_belief_update_rate}
target_belief_min_confidence=${target_belief_min_confidence}
target_belief_temperature=${target_belief_temperature}
target_belief_loss_weight=${target_belief_loss_weight}
target_belief_motion_weight=${target_belief_motion_weight}
target_belief_update_sharpness=${target_belief_update_sharpness}
latent_mpc_candidate_count=${latent_mpc_candidate_count}
latent_mpc_distance_weight=${latent_mpc_distance_weight}
latent_mpc_smooth_weight=${latent_mpc_smooth_weight}
latent_mpc_action_weight=${latent_mpc_action_weight}
latent_mpc_visual_weight=${latent_mpc_visual_weight}
latent_mpc_latent_frames=${latent_mpc_latent_frames}
latent_mpc_video_sampling_steps=${latent_mpc_video_sampling_steps}
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
capture_distance=${capture_distance}
require_visibility_for_success=${require_visibility_for_success}
stop_on_collision=${stop_on_collision}
run_self_distill=${RUN_SELF_DISTILL}
self_distill_experiments=${DISTILL_EXPERIMENTS}
self_distill_epochs=${self_distill_epochs}
EOF
}

run_teacher() {
  local name="$1"
  local save_dir
  save_dir="$(experiment_dir "$name")"
  local use_diffusion_actor
  use_diffusion_actor="$(experiment_uses_diffusion "$name")"
  local use_target_visual_guidance
  use_target_visual_guidance="$(experiment_uses_guidance "$name")"
  local use_target_belief_tracker
  use_target_belief_tracker="$(experiment_uses_target_belief_tracker "$name")"
  local use_fastwam_mot
  use_fastwam_mot="$(experiment_uses_fastwam "$name")"
  local fastwam_lambda_action fastwam_lambda_video
  read -r fastwam_lambda_action fastwam_lambda_video <<< "$(experiment_loss_weights "$name")"
  local master_port
  master_port="$(experiment_port "$name")"
  local log_file="$log_dir/${name}.log"
  local resume_ckpt="$save_dir/last.pt"
  local resume_args=()

  if [[ "$SKIP_EXISTING_TRAIN" == "true" && -f "$save_dir/done.marker" ]]; then
    if [[ -f "$save_dir/best.pt" ]]; then
      echo "[train-skip] ${name}: $save_dir/done.marker and best.pt exist"
      return 0
    fi
    echo "[train-resume] ${name}: stale done.marker without best.pt, check for last.pt"
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
  echo "use_target_visual_guidance=${use_target_visual_guidance}" | tee -a "$log_file"
  echo "use_target_belief_tracker=${use_target_belief_tracker}" | tee -a "$log_file"
  echo "heatmap_tensor_encoder=${use_heatmap_tensor_encoder}, heatmap_token_scale=${heatmap_token_scale}" | tee -a "$log_file"
  echo "target_belief_token_scale=${target_belief_token_scale}" | tee -a "$log_file"
  echo "target_belief_loss_weight=${target_belief_loss_weight}, motion_weight=${target_belief_motion_weight}, sharpness=${target_belief_update_sharpness}" | tee -a "$log_file"
  echo "fastwam_lambda_action=${fastwam_lambda_action}, fastwam_lambda_video=${fastwam_lambda_video}" | tee -a "$log_file"
  echo "low_dim_target_input=off" | tee -a "$log_file"
  echo "prediction_head_rollout_loss=off" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$log_file"
  echo "train_num_gpus=${train_num_gpus}" | tee -a "$log_file"
  echo "train_steps=${train_steps}" | tee -a "$log_file"
  if [[ ${#resume_args[@]} -gt 0 ]]; then
    echo "resume=${resume_ckpt}" | tee -a "$log_file"
  else
    echo "resume=none" | tee -a "$log_file"
  fi
  echo "use_deepspeed=${USE_DEEPSPEED}" | tee -a "$log_file"
  echo "deepspeed_offload_optimizer=${DEEPSPEED_OFFLOAD_OPTIMIZER}" | tee -a "$log_file"
  echo "checkpoint_save_every_epochs=${CHECKPOINT_SAVE_EVERY_EPOCHS}, save_best=${SAVE_BEST_CHECKPOINT}, save_optimizer=${SAVE_OPTIMIZER_STATE}" | tee -a "$log_file"
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
    --epochs "$teacher_epochs" \
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
    --use-target-visual-guidance "$use_target_visual_guidance" \
    --use-attention-heatmap "$use_attention_heatmap" \
    --use-heatmap-tensor-encoder "$use_heatmap_tensor_encoder" \
    --heatmap-token-scale "$heatmap_token_scale" \
    --fastwam-heatmap-context-grid "$fastwam_heatmap_context_grid" \
    --visual-guidance-fov-deg "$visual_guidance_fov_deg" \
    --attention-heatmap-sigma "$attention_heatmap_sigma" \
    --use-target-belief-tracker "$use_target_belief_tracker" \
    --target-belief-token-scale "$target_belief_token_scale" \
    --target-belief-update-rate "$target_belief_update_rate" \
    --target-belief-min-confidence "$target_belief_min_confidence" \
    --target-belief-temperature "$target_belief_temperature" \
    --target-belief-loss-weight "$target_belief_loss_weight" \
    --target-belief-motion-weight "$target_belief_motion_weight" \
    --target-belief-update-sharpness "$target_belief_update_sharpness" \
    --use-wan22-encoders "$use_wan22_encoders" \
    --wan22-model-base-path "$wan22_model_base_path" \
    --wan22-fastwam-src-path "$wan22_fastwam_src_path" \
    --wan22-skip-download "$wan22_skip_download" \
    --wan22-text-context-length "$wan22_text_context_length" \
    --wan22-text-encode-batch-size "$wan22_text_encode_batch_size" \
    --target-token-fusion-mode "$target_token_fusion_mode" \
    --use-diffusion-actor "$use_diffusion_actor" \
    --use-fastwam-mot "$use_fastwam_mot" \
    --fastwam-lambda-action "$fastwam_lambda_action" \
    --fastwam-lambda-video "$fastwam_lambda_video" \
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
  local distill_case="$1"
  local teacher_name student_guidance teacher_target_belief_tracker student_target_belief_tracker
  case "$distill_case" in
    target_belief_tracker_to_global)
      teacher_name="fastwam_target_belief_tracker"
      student_guidance="false"
      teacher_target_belief_tracker="true"
      student_target_belief_tracker="false"
      ;;
    heatmap_to_global)
      teacher_name="fastwam_heatmap"
      student_guidance="false"
      teacher_target_belief_tracker="false"
      student_target_belief_tracker="false"
      ;;
    *)
      echo "[ERROR] Unknown distill experiment '$distill_case'. Valid: target_belief_tracker_to_global, heatmap_to_global." >&2
      exit 1
      ;;
  esac
  local distill_name="self_distill_${distill_case}"
  local distill_dir="$model_root/$distill_name"
  local teacher_dir
  teacher_dir="$(experiment_dir "$teacher_name")"
  local teacher_ckpt="$teacher_dir/best.pt"
  local use_target_visual_guidance
  use_target_visual_guidance="$(experiment_uses_guidance "$teacher_name")"
  local log_file="$log_dir/${distill_name}.log"
  local master_port
  master_port="$(experiment_port "$distill_name")"
  local resume_ckpt="$distill_dir/last.pt"
  local resume_args=()

  if [[ ! -f "$teacher_ckpt" ]]; then
    echo "[ERROR] Missing self-distill teacher checkpoint: $teacher_ckpt" >&2
    exit 1
  fi
  if [[ "$SKIP_EXISTING_SELF_DISTILL" == "true" && -f "$distill_dir/done.marker" ]]; then
    if [[ -f "$distill_dir/best.pt" ]]; then
      echo "[self-distill-skip] ${distill_name}: $distill_dir/done.marker and best.pt exist"
      return 0
    fi
    echo "[self-distill-resume] ${distill_name}: stale done.marker without best.pt, check for last.pt"
  fi
  if [[ -f "$resume_ckpt" ]] && { [[ ! -f "$distill_dir/done.marker" ]] || [[ ! -f "$distill_dir/best.pt" ]]; }; then
    resume_args+=(--resume "$resume_ckpt")
  fi

  mkdir -p "$distill_dir"
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"

  init_args=()
  if [[ "$self_distill_init_from_teacher" == "true" || "$self_distill_init_from_teacher" == "1" ]]; then
    init_args+=(--init-student-from-teacher)
  else
    init_args+=(--student-init-random)
  fi

  echo "============================================================" | tee "$log_file"
  echo "[self-distill] ${distill_name}" | tee -a "$log_file"
  echo "teacher=${teacher_name}" | tee -a "$log_file"
  echo "teacher_ckpt=${teacher_ckpt}" | tee -a "$log_file"
  echo "save_dir=${distill_dir}" | tee -a "$log_file"
  echo "teacher_guidance=${use_target_visual_guidance}" | tee -a "$log_file"
  echo "student_guidance=${student_guidance}" | tee -a "$log_file"
  echo "teacher_target_belief_tracker=${teacher_target_belief_tracker}" | tee -a "$log_file"
  echo "student_target_belief_tracker=${student_target_belief_tracker}" | tee -a "$log_file"
  echo "weights=sup:${self_distill_sup_weight},feat:${self_distill_feat_weight},action:${self_distill_action_weight}" | tee -a "$log_file"
  echo "swanlab=${use_swanlab}, project=${swanlab_project}, run=${swanlab_experiment_prefix}_${distill_name}" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$log_file"
  echo "train_steps=${train_steps}" | tee -a "$log_file"
  if [[ ${#resume_args[@]} -gt 0 ]]; then
    echo "resume=${resume_ckpt}" | tee -a "$log_file"
  else
    echo "resume=none" | tee -a "$log_file"
  fi
  echo "use_deepspeed=${USE_DEEPSPEED}" | tee -a "$log_file"
  echo "deepspeed_offload_optimizer=${DEEPSPEED_OFFLOAD_OPTIMIZER}" | tee -a "$log_file"
  echo "checkpoint_save_every_epochs=${CHECKPOINT_SAVE_EVERY_EPOCHS}, save_best=${SAVE_BEST_CHECKPOINT}, save_optimizer=${SAVE_OPTIMIZER_STATE}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  self_distill_launcher=("$PYTHON_BIN" -m train.train_self_distill)
  if [[ "$USE_DEEPSPEED" == "true" || "$USE_DEEPSPEED" == "1" ]]; then
    self_distill_launcher=(env -u CUDA_VISIBLE_DEVICES "$PYTHON_BIN" -m deepspeed.launcher.runner --include "localhost:${TRAIN_GPU_IDS}" --master_port "$master_port" --module train.train_self_distill)
  fi

  "${self_distill_launcher[@]}" \
    --dataset-root "$dataset_root" \
    --scene-list "$scene_list" \
    --trajectory-range "$trajectory_range" \
    --val-ratio "$val_ratio" \
    --split-seed "$split_seed" \
    --teacher-ckpt "$teacher_ckpt" \
    --save-dir "$distill_dir" \
    --image-size "$image_size" \
    --seq-len "$seq_len" \
    --wan-latent-cache-root "$wan_latent_cache_root" \
    --max-vel "$max_vel" \
    --max-yaw-rate "$max_yaw_rate" \
    --max-speed-norm "$max_speed_norm" \
    --action-sequence-horizon "$action_sequence_horizon" \
    --action-video-freq-ratio "$action_video_freq_ratio" \
    --target-token-fusion-mode "$target_token_fusion_mode" \
    --use-target-visual-guidance "$use_target_visual_guidance" \
    --use-attention-heatmap "$use_attention_heatmap" \
    --student-use-target-visual-guidance "$student_guidance" \
    --student-use-attention-heatmap "$use_attention_heatmap" \
    --use-target-belief-tracker "$teacher_target_belief_tracker" \
    --student-use-target-belief-tracker "$student_target_belief_tracker" \
    --target-belief-token-scale "$target_belief_token_scale" \
    --target-belief-update-rate "$target_belief_update_rate" \
    --target-belief-min-confidence "$target_belief_min_confidence" \
    --target-belief-temperature "$target_belief_temperature" \
    --target-belief-loss-weight "$target_belief_loss_weight" \
    --target-belief-motion-weight "$target_belief_motion_weight" \
    --target-belief-update-sharpness "$target_belief_update_sharpness" \
    --visual-guidance-fov-deg "$visual_guidance_fov_deg" \
    --attention-heatmap-sigma "$attention_heatmap_sigma" \
    --fastwam-heatmap-context-grid "$fastwam_heatmap_context_grid" \
    --use-wan22-encoders "$use_wan22_encoders" \
    --wan22-model-base-path "$wan22_model_base_path" \
    --wan22-fastwam-src-path "$wan22_fastwam_src_path" \
    --wan22-skip-download "$wan22_skip_download" \
    --wan22-text-context-length "$wan22_text_context_length" \
    --wan22-text-encode-batch-size "$wan22_text_encode_batch_size" \
    --fastwam-skip-dit-load-from-pretrain "$fastwam_skip_dit_load_from_pretrain" \
    --fastwam-action-dit-pretrained-path "$fastwam_action_dit_pretrained_path" \
    --fastwam-mot-checkpoint-mixed-attn "$fastwam_mot_checkpoint_mixed_attn" \
    --diffusion-steps "$diffusion_steps" \
    --sampling-steps "$sampling_steps" \
    --batch-size "$self_distill_batch_size" \
    --epochs "$self_distill_epochs" \
    --max-train-steps "$train_steps" \
    --lr "$self_distill_lr" \
    --weight-decay "$self_distill_weight_decay" \
    --num-workers "$num_workers" \
    --save-every-epochs "$CHECKPOINT_SAVE_EVERY_EPOCHS" \
    --save-best-checkpoint "$SAVE_BEST_CHECKPOINT" \
    --save-optimizer-state "$SAVE_OPTIMIZER_STATE" \
    --deepspeed-offload-optimizer "$DEEPSPEED_OFFLOAD_OPTIMIZER" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --sup-weight "$self_distill_sup_weight" \
    --feat-distill-weight "$self_distill_feat_weight" \
    --action-distill-weight "$self_distill_action_weight" \
    --use-swanlab "$use_swanlab" \
    --swanlab-project "$swanlab_project" \
    --swanlab-experiment-name "${swanlab_experiment_prefix}_${distill_name}" \
    --swanlab-workspace "$swanlab_workspace" \
    --swanlab-log-dir "$swanlab_log_dir" \
    --swanlab-mode "$swanlab_mode" \
    "${resume_args[@]}" \
    $(if [[ "$USE_DEEPSPEED" == "true" || "$USE_DEEPSPEED" == "1" ]]; then printf '%s' '--deepspeed'; fi) \
    "${init_args[@]}" \
    2>&1 | tee -a "$log_file"
}

run_online_eval() {
  local name="$1"
  local ckpt="$2"
  local use_diffusion_actor="$3"
  local use_target_visual_guidance="$4"
  local use_target_belief_tracker="${5:-false}"
  local use_latent_mpc="${6:-false}"
  local out_root="${7:-$eval_root}"
  local log_root="${8:-$eval_log_dir}"
  local scene_list_for_eval="${9:-$eval_scene_list}"
  local trajectory_range_for_eval="${10:-$eval_trajectory_range}"
  local out_dir="$out_root/$name"
  local log_file="$log_root/${name}.log"

  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] Missing checkpoint for eval: $ckpt" >&2
    exit 1
  fi
  if [[ "$SKIP_EXISTING_EVAL" == "true" && -f "$out_dir/summary.json" ]]; then
    echo "[eval-skip] ${name}: $out_dir/summary.json exists"
    return 0
  fi

  mkdir -p "$out_dir"
  mkdir -p "$log_root"
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

  echo "============================================================" | tee "$log_file"
  echo "[online-eval] ${name}" | tee -a "$log_file"
  echo "checkpoint=${ckpt}" | tee -a "$log_file"
  echo "output=${out_dir}" | tee -a "$log_file"
  echo "scene_list=${scene_list_for_eval}" | tee -a "$log_file"
  echo "trajectory_range=${trajectory_range_for_eval}" | tee -a "$log_file"
  echo "use_diffusion_actor=${use_diffusion_actor}" | tee -a "$log_file"
  echo "use_target_visual_guidance=${use_target_visual_guidance}" | tee -a "$log_file"
  echo "use_target_belief_tracker=${use_target_belief_tracker}" | tee -a "$log_file"
  echo "target_belief_motion_weight=${target_belief_motion_weight}, update_sharpness=${target_belief_update_sharpness}" | tee -a "$log_file"
  echo "use_latent_mpc=${use_latent_mpc}" | tee -a "$log_file"
  if [[ "$use_latent_mpc" == "true" || "$use_latent_mpc" == "1" ]]; then
    echo "latent_mpc_candidate_count=${latent_mpc_candidate_count}" | tee -a "$log_file"
    echo "latent_mpc_weights=distance:${latent_mpc_distance_weight},smooth:${latent_mpc_smooth_weight},action:${latent_mpc_action_weight},visual:${latent_mpc_visual_weight}" | tee -a "$log_file"
    echo "latent_mpc_latent_frames=${latent_mpc_latent_frames}, video_sampling_steps=${latent_mpc_video_sampling_steps}" | tee -a "$log_file"
  fi
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
    --eval-split all \
    --sim-server-host "$sim_server_host" \
    --sim-server-port "$sim_server_port" \
    --scene-index "$scene_index" \
    --gpu-id "$EVAL_GPU_ID" \
    --device cuda \
    --max-vel "$max_vel" \
    --max-yaw-rate "$max_yaw_rate" \
    --max-speed-norm "$max_speed_norm" \
    --capture-distance "$capture_distance" \
    --use-target-visual-guidance "$use_target_visual_guidance" \
    --use-attention-heatmap "$use_attention_heatmap" \
    --visual-guidance-fov-deg "$visual_guidance_fov_deg" \
    --attention-heatmap-sigma "$attention_heatmap_sigma" \
    --use-target-belief-tracker "$use_target_belief_tracker" \
    --target-belief-token-scale "$target_belief_token_scale" \
    --target-belief-update-rate "$target_belief_update_rate" \
    --target-belief-min-confidence "$target_belief_min_confidence" \
    --target-belief-temperature "$target_belief_temperature" \
    --target-belief-loss-weight "$target_belief_loss_weight" \
    --target-belief-motion-weight "$target_belief_motion_weight" \
    --target-belief-update-sharpness "$target_belief_update_sharpness" \
    --use-latent-mpc "$use_latent_mpc" \
    --latent-mpc-candidate-count "$latent_mpc_candidate_count" \
    --latent-mpc-distance-weight "$latent_mpc_distance_weight" \
    --latent-mpc-smooth-weight "$latent_mpc_smooth_weight" \
    --latent-mpc-action-weight "$latent_mpc_action_weight" \
    --latent-mpc-visual-weight "$latent_mpc_visual_weight" \
    --latent-mpc-latent-frames "$latent_mpc_latent_frames" \
    --latent-mpc-video-sampling-steps "$latent_mpc_video_sampling_steps" \
    --use-wan22-encoders "$use_wan22_encoders" \
    --wan22-model-base-path "$wan22_model_base_path" \
    --wan22-fastwam-src-path "$wan22_fastwam_src_path" \
    --wan22-skip-download "$wan22_skip_download" \
    --wan22-text-context-length "$wan22_text_context_length" \
    --wan22-text-encode-batch-size "$wan22_text_encode_batch_size" \
    --use-diffusion-actor "$use_diffusion_actor" \
    --no-save-transformer-attention-maps \
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
  "$PYTHON_BIN" - "$summary_root" "$summary_title" "$@" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
title = sys.argv[2]
models = sys.argv[3:]

def num(x):
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")

print(f"\n[ablation] {title}")
print(
    f"{'model':26s} {'SR':>8s} {'ATF':>8s} {'track%':>8s} "
    f"{'coll%':>8s} {'final_d':>9s} {'mean_d':>9s} {'failures':>28s}"
)
for model in models:
    path = root / model / "summary.json"
    if not path.exists():
        partial = root / model / "summary_partial.json"
        if partial.exists():
            print(f"{model:26s} {'partial':>8s}")
        else:
            print(f"{model:26s} {'missing':>8s}")
        continue
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    sr = num(data.get("SR", data.get("success_rate")))
    atf = num(data.get("ATF", data.get("average_tracked_frames")))
    track = num(data.get("mean_effective_tracking_ratio", data.get("average_tracked_frame_ratio")))
    coll = num(data.get("collision_rate"))
    final_d = num(data.get("mean_final_distance"))
    mean_d = num(data.get("mean_distance"))
    failures = data.get("failure_reason_counts", {})
    failures_s = ",".join(f"{k}:{v}" for k, v in sorted(failures.items()))
    print(
        f"{model:26s} "
        f"{sr * 100:7.2f}% "
        f"{atf:8.2f} "
        f"{track * 100:7.2f}% "
        f"{coll * 100:7.2f}% "
        f"{final_d:9.2f} "
        f"{mean_d:9.2f} "
        f"{failures_s:>28s}"
    )
PY
}

write_manifest
csv_to_array "$EXPERIMENTS" experiment_names
csv_to_array "$EVAL_EXTRA_EXPERIMENTS" eval_extra_experiment_names
csv_to_array "$DISTILL_EXPERIMENTS" distill_experiment_names

echo "[ablation] experiment root: $exp_root"
echo "[ablation] experiments: ${experiment_names[*]}"
echo "[ablation] eval-only experiments: ${eval_extra_experiment_names[*]}"
echo "[ablation] distill experiments: ${distill_experiment_names[*]}"
echo "[ablation] train GPUs: $TRAIN_GPU_IDS (num_gpus=$train_num_gpus, deepspeed=$USE_DEEPSPEED)"
echo "[ablation] eval GPU: $EVAL_GPU_ID"
echo "[ablation] train_steps: $train_steps"
echo "[ablation] skip existing train/eval/self-distill: $SKIP_EXISTING_TRAIN / $SKIP_EXISTING_EVAL / $SKIP_EXISTING_SELF_DISTILL"
echo "[ablation] low-dimensional target vector is off; local crop is removed"

if [[ "$RUN_TEACHER_ABLATIONS" == "true" ]]; then
  for name in "${experiment_names[@]}"; do
    run_teacher "$name"
  done
else
  echo "[ablation] RUN_TEACHER_ABLATIONS=false, skip teacher training"
fi

summary_models=("${experiment_names[@]}")

if [[ "$RUN_SELF_DISTILL" == "true" ]]; then
  for distill_case in "${distill_experiment_names[@]}"; do
    run_self_distill "$distill_case"
    summary_models+=("self_distill_${distill_case}")
  done
else
  echo "[ablation] RUN_SELF_DISTILL=false, skip self-distillation"
fi
for eval_name in "${eval_extra_experiment_names[@]}"; do
  summary_models+=("$eval_name")
done

if [[ "$RUN_ONLINE_EVAL" == "true" ]]; then
  echo "[ablation] starting online eval"
  for name in "${experiment_names[@]}"; do
    ckpt="$(eval_checkpoint_for_experiment "$name")"
    run_online_eval \
      "$name" \
      "$ckpt" \
      "$(experiment_uses_diffusion "$name")" \
      "$(experiment_uses_guidance "$name")" \
      "$(experiment_uses_target_belief_tracker "$name")" \
      "$(experiment_uses_latent_mpc "$name")"
  done
  for distill_case in "${distill_experiment_names[@]}"; do
    distill_name="self_distill_${distill_case}"
    distill_ckpt="$model_root/$distill_name/best.pt"
    if [[ "$RUN_SELF_DISTILL" == "true" || -f "$distill_ckpt" ]]; then
      run_online_eval "$distill_name" "$distill_ckpt" true false false false
      if [[ "$RUN_SELF_DISTILL" != "true" ]]; then
        summary_models+=("$distill_name")
      fi
    fi
  done
  for name in "${eval_extra_experiment_names[@]}"; do
    ckpt="$(eval_checkpoint_for_experiment "$name")"
    if [[ "$name" == self_distill_* && "$RUN_SELF_DISTILL" != "true" && ! -f "$ckpt" ]]; then
      echo "[eval-skip] ${name}: missing self-distill checkpoint $ckpt"
      continue
    fi
    run_online_eval \
      "$name" \
      "$ckpt" \
      "$(experiment_uses_diffusion "$name")" \
      "$(experiment_uses_guidance "$name")" \
      "$(experiment_uses_target_belief_tracker "$name")" \
      "$(experiment_uses_latent_mpc "$name")"
  done
  summarize_eval_results "$eval_root" "held-out online eval summary (${eval_scene_list} ${eval_trajectory_range})" "${summary_models[@]}"
else
  echo "[ablation] RUN_ONLINE_EVAL=false, skip online eval"
fi

echo "[ablation] finished: $exp_root"
