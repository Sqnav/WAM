#!/usr/bin/env bash
set -euo pipefail

# FastWAM/Faster-WAM experiment runner.
#
# Active experiment chain (the default run list is set by EXPERIMENTS below):
#   fastwam
#     Official 30-layer Video/Action DiTs coupled by FastWAM MoT.
#   fastwam_video_current_box_action_aligned
#     FastWAM plus the complete pretrained Tracker's current b0, injected into
#     Action layers 18/23/26/29. The policy is trained independently without
#     loading fastwam/best.pt.
#   fastwam_current_box_historical_target_memory
#     FastWAM with Current Box plus the horizon-aligned 8-state target memory. The
#     complete policy is independently initialized and trained without loading
#     either FastWAM or Current Box policy checkpoints.
#   fastwam_current_box_capture_value_reranking
#     Training-free N-candidate action-prior reranking on the FastWAM history parent.
#   fasterwam
#     FastWAM Video DiT hub with the Faster-WAM DoT one-layer Action Head.
#   fasterwam_video_current_box_action_aligned
#     Faster-WAM parent plus the complete pretrained Tracker's current b0. The
#     encoded box is injected as an ungated residual into Action layer 0.
#   fasterwam_current_box_historical_target_memory
#     Current-box parent plus an 8-state 2D box/confidence/motion Transformer
#     memory aligned to the 8 Action tokens. Only the new history memory and
#     residual adapter train; the inherited policy remains frozen.
#   fasterwam_current_box_capture_value_reranking
#     Training-free N-candidate reranking on the historical-memory policy. The
#     default action_prior mode uses a separately trained frozen CaptureActionPrior.

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

EXP_NAME="${EXP_NAME:-fastwam_local_feature_ablation}"
exp_root="${EXP_ROOT:-$root_dir/experiments/fastwam_local_feature_ablation_run}"
model_root="${MODEL_OUTPUT_ROOT:-$exp_root/models}"
log_dir="$exp_root/logs"
eval_root="${EVAL_OUTPUT_ROOT:-$exp_root/online_eval}"
eval_log_dir="${EVAL_LOG_DIR:-$exp_root/eval_logs}"
mkdir -p "$exp_root" "$model_root" "$log_dir" "$eval_root" "$eval_log_dir"

RUN_TEACHER_ABLATIONS="${RUN_TEACHER_ABLATIONS:-true}"
RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-true}"
DRY_RUN="${DRY_RUN:-false}"
# auto (default): reuse an existing Tracker checkpoint. Set true only when a
# deliberate Tracker retraining run is required.
RUN_TRACKER_TRAINING="${RUN_TRACKER_TRAINING:-auto}"
RUN_TRACKER_CACHE_PRECOMPUTE="${RUN_TRACKER_CACHE_PRECOMPUTE:-true}"
RUN_CAPTURE_ACTION_PRIOR_TRAINING="${RUN_CAPTURE_ACTION_PRIOR_TRAINING:-auto}"
TRACKER_ONLY="${TRACKER_ONLY:-false}"
if [[ "$TRACKER_ONLY" == "true" || "$TRACKER_ONLY" == "1" ]]; then
  RUN_TRACKER_TRAINING=true
  RUN_TRACKER_CACHE_PRECOMPUTE=false
  RUN_TEACHER_ABLATIONS=false
  RUN_ONLINE_EVAL=false
fi
if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "1" ]]; then
  RUN_ONLINE_EVAL=false
fi
USE_DEEPSPEED="${USE_DEEPSPEED:-true}"
DEEPSPEED_OFFLOAD_OPTIMIZER="${DEEPSPEED_OFFLOAD_OPTIMIZER:-true}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
CHECKPOINT_SAVE_EVERY_EPOCHS="${CHECKPOINT_SAVE_EVERY_EPOCHS:-1}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-true}"
SAVE_OPTIMIZER_STATE="${SAVE_OPTIMIZER_STATE:-false}"

EXPERIMENTS="${EXPERIMENTS-fastwam,fastwam_video_current_box_action_aligned,fastwam_current_box_historical_target_memory,fastwam_current_box_capture_value_reranking,fasterwam,fasterwam_video_current_box_action_aligned,fasterwam_current_box_historical_target_memory,fasterwam_current_box_capture_value_reranking}"
EVAL_EXTRA_EXPERIMENTS="${EVAL_EXTRA_EXPERIMENTS-}"
SKIP_EXISTING_TRAIN="${SKIP_EXISTING_TRAIN:-true}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-true}"
# The conditioned comparisons use a strict Current Box -> V1 -> V2 hierarchy.
# Existing checkpoints are reused by default. Set this explicitly to true only
# when starting a deliberately new retraining generation.
FORCE_RETRAIN_IMPROVED_MODELS="${FORCE_RETRAIN_IMPROVED_MODELS:-false}"
RETRAIN_GENERATION="${RETRAIN_GENERATION:-city1_27_history_reranker_v12}"
retrain_backup_stamp="${RETRAIN_BACKUP_STAMP:-$(date +%Y%m%d-%H%M%S)}"

city_csv_range() {
  local first="$1"
  local last="$2"
  local cities=()
  local index
  for ((index=first; index<=last; index++)); do
    cities+=("City_${index}")
  done
  local IFS=,
  echo "${cities[*]}"
}

default_train_scene_list="$(city_csv_range 1 27)"
default_seen_eval_scene_list="$default_train_scene_list"
default_unseen_eval_scene_list="$(city_csv_range 28 30)"

scene_list="${SCENE_LIST:-$default_train_scene_list}"
trajectory_range="${TRAJECTORY_RANGE:-1-450}"
# Held-out online evaluation uses the unseen tail of the training cities plus
# every trajectory from three completely unseen cities.
val_seen_scene_list="${VAL_SEEN_SCENE_LIST:-$default_seen_eval_scene_list}"
val_seen_trajectory_range="${VAL_SEEN_TRAJECTORY_RANGE:-451-500}"
val_unseen_scene_list="${VAL_UNSEEN_SCENE_LIST:-$default_unseen_eval_scene_list}"
val_unseen_trajectory_range="${VAL_UNSEEN_TRAJECTORY_RANGE:-1-500}"
val_scene_list="${VAL_SCENE_LIST:-$val_seen_scene_list,$val_unseen_scene_list}"
val_trajectory_spec="${VAL_TRAJECTORY_SPEC:-City_1-27:451-500;City_28-30:1-500}"
# Legacy single-range override applies to every evaluation city when supplied.
val_trajectory_range_override="${VAL_TRAJECTORY_RANGE:-}"
val_trajectory_range="${val_trajectory_range_override:-$val_trajectory_spec}"
val_ratio="0.0"
val_every_epochs="${VAL_EVERY_EPOCHS:-1}"
split_seed="${SPLIT_SEED:-42}"

train_steps="${TRAIN_STEPS:-0}"
model_train_epochs="${MODEL_TRAIN_EPOCHS:-5}"
conditioned_train_steps="${CONDITIONED_TRAIN_STEPS:-0}"
conditioned_train_epochs="${CONDITIONED_TRAIN_EPOCHS:-5}"
s0_pretrain_epochs="${S0_PRETRAIN_EPOCHS:-10}"
teacher_batch_size="${TEACHER_BATCH_SIZE:-64}"
gt_center_teacher_batch_size="${GT_CENTER_TEACHER_BATCH_SIZE:-16}"
conditioned_teacher_batch_size="${CONDITIONED_TEACHER_BATCH_SIZE:-32}"
conditioned_gradient_accumulation_steps="${CONDITIONED_GRADIENT_ACCUMULATION_STEPS:-1}"
capture_value_candidate_count="${CAPTURE_VALUE_CANDIDATE_COUNT:-4}"
capture_value_score_mode="action_prior"
capture_action_prior_checkpoint="${CAPTURE_ACTION_PRIOR_CHECKPOINT:-$model_root/capture_action_prior/best.pt}"
capture_action_prior_dimension_weights="${CAPTURE_ACTION_PRIOR_DIMENSION_WEIGHTS:-0 1 1 1}"
capture_value_structured_candidates="${CAPTURE_VALUE_STRUCTURED_CANDIDATES:-true}"
capture_value_selection_margin="${CAPTURE_VALUE_SELECTION_MARGIN:-0.0}"
capture_value_min_center_error="${CAPTURE_VALUE_MIN_CENTER_ERROR:-0.30}"
seq_len="${SEQ_LEN:-9}"
image_size="${IMAGE_SIZE:-224}"
num_workers="${NUM_WORKERS:-0}"
teacher_lr="${TEACHER_LR:-1e-4}"
teacher_weight_decay="${TEACHER_WEIGHT_DECAY:-1e-4}"

max_vel="${MAX_VEL:-1.0}"
max_yaw_rate="${MAX_YAW_RATE:-15.0}"
max_speed_norm="${MAX_SPEED_NORM:-1.0}"
action_sequence_horizon="${ACTION_SEQUENCE_HORIZON:-8}"
action_video_freq_ratio="${ACTION_VIDEO_FREQ_RATIO:-1}"
diffusion_steps="${DIFFUSION_STEPS:-20}"
sampling_steps="${SAMPLING_STEPS:-8}"
eval_compile_action_sampling="${EVAL_COMPILE_ACTION_SAMPLING:-true}"
eval_compile_action_sampling_mode="${EVAL_COMPILE_ACTION_SAMPLING_MODE:-reduce-overhead}"

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
tracker_feature_dim="${TRACKER_FEATURE_DIM:-192}"
tracker_feature_grid_size="${TRACKER_FEATURE_GRID_SIZE:-7}"
tracker_template_size="${TRACKER_TEMPLATE_SIZE:-128}"
tracker_search_size="${TRACKER_SEARCH_SIZE:-256}"
tracker_local_position_embedding="${TRACKER_USE_LOCAL_POSITION_EMBEDDING:-false}"
tracker_fusion_gate_init="${TRACKER_FUSION_GATE_INIT:-0.0}"
tracker_state_action_alignment_version="${TRACKER_STATE_ACTION_ALIGNMENT_VERSION:-3}"
tracker_search_center_jitter_std="${TRACKER_SEARCH_CENTER_JITTER_STD:-0.10}"
tracker_search_center_jitter_max="${TRACKER_SEARCH_CENTER_JITTER_MAX:-0.20}"
tracker_search_scale_jitter="${TRACKER_SEARCH_SCALE_JITTER:-0.10}"
default_tracker_fusion_start_layer="${TRACKER_FUSION_START_LAYER:-18}"
tracker_condition_mode="${TRACKER_CONDITION_MODE:-center_features}"

use_wan22_encoders="${USE_WAN22_ENCODERS:-true}"
wan22_model_base_path="${WAN22_MODEL_BASE_PATH:-$root_dir/model}"
wan22_fastwam_src_path="${WAN22_FASTWAM_SRC_PATH:-$root_dir/model/FastWAM/src}"
wan22_skip_download="${WAN22_SKIP_DOWNLOAD:-false}"
wan22_text_context_length="${WAN22_TEXT_CONTEXT_LENGTH:-512}"
wan22_text_encode_batch_size="${WAN22_TEXT_ENCODE_BATCH_SIZE:-4}"
wan_latent_cache_root="${WAN_LATENT_CACHE_ROOT:-$root_dir/latents}"
tracker_output_dir="${TRACKER_OUTPUT_DIR:-$root_dir/experiments/tracker_artifacts/models/uav_tracker_gt_bbox_square}"
tracker_manifest="${TRACKER_MANIFEST:-$tracker_output_dir/tracking_manifest.json}"
tracker_epochs="${TRACKER_EPOCHS:-10}"
tracker_batch_size="${TRACKER_BATCH_SIZE:-32}"
tracker_num_workers="${TRACKER_NUM_WORKERS:-8}"
tracker_samples_per_epoch="${TRACKER_SAMPLES_PER_EPOCH:-60000}"
tracker_val_samples="0"
tracker_max_gap="${TRACKER_MAX_GAP:-40}"
tracker_lr="${TRACKER_LR:-0.0004}"
tracker_resume="${TRACKER_RESUME:-1}"
tracker_rebuild_manifest="${TRACKER_REBUILD_MANIFEST:-1}"
tracker_require_real_annotations="${TRACKER_REQUIRE_REAL_ANNOTATIONS:-true}"
run_tracker_heldout_eval="${RUN_TRACKER_HELDOUT_EVAL:-true}"
# The standalone Tracker evaluator accepts one uniform range. Its default
# report therefore covers the held-out tail of City_1-27; the full mixed split
# is evaluated end-to-end by the online evaluator below.
tracker_eval_scene_list="${TRACKER_EVAL_SCENE_LIST:-$val_seen_scene_list}"
tracker_eval_trajectory_range="${TRACKER_EVAL_TRAJECTORY_RANGE:-$val_seen_trajectory_range}"
tracker_eval_manifest="${TRACKER_EVAL_MANIFEST:-$tracker_output_dir/heldout_manifest.json}"
tracker_eval_output="${TRACKER_EVAL_OUTPUT:-$tracker_output_dir/heldout_eval.json}"
tracker_eval_gpu_id="${TRACKER_EVAL_GPU_ID:-}"
if [[ -n "${TRACKER_CHECKPOINT:-}" ]]; then
  tracker_checkpoint="$TRACKER_CHECKPOINT"
else
  tracker_checkpoint="$tracker_output_dir/best.pt"
fi
target_history_tracker_cache_root="${TARGET_HISTORY_TRACKER_CACHE_ROOT:-$root_dir/experiments/tracker_artifacts/caches/square_tracker_cache_gt_bbox}"
target_history_length="${TARGET_HISTORY_LENGTH:-8}"
target_history_hidden_dim="${TARGET_HISTORY_HIDDEN_DIM:-256}"
target_history_num_layers="${TARGET_HISTORY_NUM_LAYERS:-2}"
target_history_num_heads="${TARGET_HISTORY_NUM_HEADS:-8}"
target_history_partial_probability="${TARGET_HISTORY_PARTIAL_PROBABILITY:-0.5}"
target_history_center_jitter_std="${TARGET_HISTORY_CENTER_JITTER_STD:-0.01}"
target_history_log_size_jitter_std="${TARGET_HISTORY_LOG_SIZE_JITTER_STD:-0.05}"
target_history_confidence_dropout_probability="${TARGET_HISTORY_CONFIDENCE_DROPOUT_PROBABILITY:-0.1}"
tracker_backbone_pretrained_path="${TRACKER_BACKBONE_PRETRAINED_PATH:-$root_dir/model/pretrained/deit_tiny_patch16_224-a1311bcf.pth}"
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

use_swanlab="${USE_SWANLAB:-false}"
swanlab_project="${SWANLAB_PROJECT:-WAM-FastWAM}"
swanlab_experiment_prefix="${SWANLAB_EXPERIMENT_PREFIX:-${EXP_NAME}_$(date +%Y%m%d-%H%M%S)}"
swanlab_workspace="${SWANLAB_WORKSPACE:-}"
swanlab_api_key="${SWANLAB_API_KEY:-}"
swanlab_log_dir="${SWANLAB_LOG_DIR:-$exp_root/swanlab_logs}"
swanlab_mode="${SWANLAB_MODE:-disabled}"
mkdir -p "$swanlab_log_dir"
if [[ "$use_swanlab" == "true" || "$use_swanlab" == "1" ]]; then
  export SWANLAB_NO_INTERACTIVE=1
  export SWANLAB_LOG_DIR="$swanlab_log_dir"
  export SWANLAB_DIR="$swanlab_log_dir"
  if [[ -n "$swanlab_api_key" ]]; then
    export SWANLAB_API_KEY="$swanlab_api_key"
  fi
fi

AUTO_SELECT_TRAIN_GPUS="${AUTO_SELECT_TRAIN_GPUS:-true}"
AUTO_SELECT_EVAL_GPUS="${AUTO_SELECT_EVAL_GPUS:-false}"
TRAIN_GPU_MIN_FREE_MEM_GB="${TRAIN_GPU_MIN_FREE_MEM_GB:-40}"
EVAL_GPU_MIN_FREE_MEM_GB="${EVAL_GPU_MIN_FREE_MEM_GB:-40}"
GPU_EXCLUDE_IDS="${GPU_EXCLUDE_IDS:-4,5}"
EVAL_PARALLEL_JOBS="${EVAL_PARALLEL_JOBS:-4}"

select_gpus_by_free_memory() {
  local min_free_gb="$1"
  local min_free_mb
  min_free_mb="$((min_free_gb * 1024))"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi is required for automatic GPU selection." >&2
    return 1
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' -v threshold="$min_free_mb" -v excluded="$GPU_EXCLUDE_IDS" '
        BEGIN {
          n = split(excluded, ids, ",")
          for (i = 1; i <= n; i++) { gsub(/[[:space:]]/, "", ids[i]); skip[ids[i]] = 1 }
        }
        { gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2) }
        ($2 + 0) >= (threshold + 0) && !($1 in skip) { print $1 "," $2 }
      ' \
    | sort -t',' -k2,2nr \
    | cut -d',' -f1 \
    | paste -sd, -
}

TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3}"
if [[ -z "$TRAIN_GPU_IDS" && "$AUTO_SELECT_TRAIN_GPUS" == "true" ]]; then
  TRAIN_GPU_IDS="$(select_gpus_by_free_memory "$TRAIN_GPU_MIN_FREE_MEM_GB")"
fi
if [[ -z "$TRAIN_GPU_IDS" ]]; then
  echo "[ERROR] No training GPU has at least ${TRAIN_GPU_MIN_FREE_MEM_GB}GB free. Set TRAIN_GPU_IDS explicitly or lower TRAIN_GPU_MIN_FREE_MEM_GB." >&2
  nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader || true
  exit 1
fi

EVAL_GPU_OVERRIDE="${EVAL_GPU_IDS:-${EVAL_GPU_ID:-}}"
EVAL_GPU_IDS="$EVAL_GPU_OVERRIDE"
if [[ -z "$EVAL_GPU_IDS" ]]; then
  if [[ "$AUTO_SELECT_EVAL_GPUS" == "true" ]]; then
    EVAL_GPU_IDS="$(select_gpus_by_free_memory "$EVAL_GPU_MIN_FREE_MEM_GB")"
    if [[ -z "$EVAL_GPU_IDS" ]]; then
      echo "[ERROR] No evaluation GPU has at least ${EVAL_GPU_MIN_FREE_MEM_GB}GB free. Set EVAL_GPU_IDS explicitly or lower EVAL_GPU_MIN_FREE_MEM_GB." >&2
      nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader || true
      exit 1
    fi
  else
    EVAL_GPU_IDS="0,1,2,3"
  fi
fi
IFS=',' read -ra EVAL_GPU_POOL <<< "$EVAL_GPU_IDS"
EVAL_GPU_ID="${EVAL_GPU_POOL[0]}"
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

eval_seen_scene_list="${EVAL_SEEN_SCENE_LIST:-$val_seen_scene_list}"
eval_seen_trajectory_range="${EVAL_SEEN_TRAJECTORY_RANGE:-$val_seen_trajectory_range}"
eval_unseen_scene_list="${EVAL_UNSEEN_SCENE_LIST:-$val_unseen_scene_list}"
eval_unseen_trajectory_range="${EVAL_UNSEEN_TRAJECTORY_RANGE:-$val_unseen_trajectory_range}"
eval_scene_list="${EVAL_SCENE_LIST:-$eval_seen_scene_list,$eval_unseen_scene_list}"
eval_trajectory_range_override="${EVAL_TRAJECTORY_RANGE:-$val_trajectory_range_override}"
eval_trajectory_range="${eval_trajectory_range_override:-${EVAL_TRAJECTORY_SPEC:-$val_trajectory_spec}}"

csv_contains_value() {
  local csv="$1"
  local expected="$2"
  local values=()
  local value
  IFS=',' read -ra values <<< "$csv"
  for value in "${values[@]}"; do
    value="${value//[[:space:]]/}"
    if [[ "$value" == "$expected" ]]; then
      return 0
    fi
  done
  return 1
}

eval_trajectory_range_for_city() {
  local city="$1"
  if [[ -n "$eval_trajectory_range_override" ]]; then
    echo "$eval_trajectory_range_override"
  elif csv_contains_value "$eval_seen_scene_list" "$city"; then
    echo "$eval_seen_trajectory_range"
  elif csv_contains_value "$eval_unseen_scene_list" "$city"; then
    echo "$eval_unseen_trajectory_range"
  else
    echo "[ERROR] No evaluation trajectory range configured for $city" >&2
    return 1
  fi
}
eval_max_trajectories="${EVAL_MAX_TRAJECTORIES:-0}"
eval_max_steps="${EVAL_MAX_STEPS:-0}"
eval_visualize_trajectory_keys="${EVAL_VISUALIZE_TRAJECTORY_KEYS:-all}"
eval_save_transformer_attention_maps="${EVAL_SAVE_TRANSFORMER_ATTENTION_MAPS:-true}"
eval_attention_trajectory_keys="${EVAL_ATTENTION_TRAJECTORY_KEYS:-}"
eval_save_rgb="${EVAL_SAVE_RGB:-true}"
eval_save_predicted_video="${EVAL_SAVE_PREDICTED_VIDEO:-true}"
eval_predicted_video_trajectory_keys="${EVAL_PREDICTED_VIDEO_TRAJECTORY_KEYS:-City_1/trajectory_0451}"
eval_save_target_crop_action_overlays="${EVAL_SAVE_TARGET_CROP_ACTION_OVERLAYS:-true}"
eval_save_trajectory_3d="${EVAL_SAVE_TRAJECTORY_3D:-false}"
eval_postprocess_visuals="${EVAL_POSTPROCESS_VISUALS:-true}"
eval_profile_step_time="${EVAL_PROFILE_STEP_TIME:-false}"
eval_profile_step_time_interval="${EVAL_PROFILE_STEP_TIME_INTERVAL:-10}"
eval_target_crop_action_overlay_output_name="${EVAL_TARGET_CROP_ACTION_OVERLAY_OUTPUT_NAME:-target_crop_action_trajectory_overlays}"
eval_reuse_tracker_action_sequence="${EVAL_REUSE_TRACKER_ACTION_SEQUENCE:-true}"
eval_tracker_detection_confidence_threshold="${EVAL_TRACKER_DETECTION_CONFIDENCE_THRESHOLD:-0.5}"
eval_tracker_fallback_action_mode="${EVAL_TRACKER_FALLBACK_ACTION_MODE:-remaining_sequence}"
eval_validate_camera_freshness="${EVAL_VALIDATE_CAMERA_FRESHNESS:-true}"
eval_camera_max_vehicle_distance="${EVAL_CAMERA_MAX_VEHICLE_DISTANCE:-5.0}"
eval_camera_render_frames="${EVAL_CAMERA_RENDER_FRAMES:-1}"
eval_camera_capture_mode="${EVAL_CAMERA_CAPTURE_MODE:-fresh_frame}"
eval_camera_render_max_fps="${EVAL_CAMERA_RENDER_MAX_FPS:-60}"
eval_camera_pose_tolerance_m="${EVAL_CAMERA_POSE_TOLERANCE_M:-0.05}"
eval_camera_orientation_tolerance_deg="${EVAL_CAMERA_ORIENTATION_TOLERANCE_DEG:-1.0}"
eval_save_depth="${EVAL_SAVE_DEPTH:-false}"
eval_use_external_camera="${EVAL_USE_EXTERNAL_CAMERA:-true}"
eval_camera_only_virtual_uav="${EVAL_CAMERA_ONLY_VIRTUAL_UAV:-true}"
eval_tracker_fallback_experiments="${EVAL_TRACKER_FALLBACK_EXPERIMENTS:-}"
predicted_video_latent_frames="${PREDICTED_VIDEO_LATENT_FRAMES:-3}"
sim_server_host="${SIM_SERVER_HOST:-127.0.0.1}"
sim_server_port="${SIM_SERVER_PORT:-30000}"
scene_index="${SCENE_INDEX:-1}"
capture_distance="${CAPTURE_DISTANCE:-10.0}"
require_visibility_for_success="${REQUIRE_VISIBILITY_FOR_SUCCESS:-false}"
stop_on_collision="${STOP_ON_COLLISION:-true}"

if [[ -z "$eval_attention_trajectory_keys" ]]; then
  IFS=',' read -ra attention_cities <<< "$eval_scene_list"
  attention_keys=()
  for city in "${attention_cities[@]}"; do
    city="${city//[[:space:]]/}"
    [[ -n "$city" ]] && attention_keys+=("${city}/trajectory_0451")
  done
  eval_attention_trajectory_keys="$(IFS=,; echo "${attention_keys[*]}")"
fi

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

SUPPORTED_EXPERIMENTS=(
  fastwam
  fastwam_video_current_box_action_aligned
  fastwam_current_box_historical_target_memory
  fastwam_current_box_capture_value_reranking
  fasterwam
  fasterwam_video_current_box_action_aligned
  fasterwam_current_box_historical_target_memory
  fasterwam_current_box_capture_value_reranking
)

validate_experiment_name() {
  local name="$1"
  local supported
  for supported in "${SUPPORTED_EXPERIMENTS[@]}"; do
    if [[ "$name" == "$supported" ]]; then
      return 0
    fi
  done
  echo "[ERROR] Unsupported experiment '$name'. Supported models: ${SUPPORTED_EXPERIMENTS[*]}" >&2
  return 1
}

experiment_dir() {
  validate_experiment_name "$1"
  echo "$model_root/$1"
}

experiment_uses_diffusion() {
  validate_experiment_name "$1"
  echo "true"
}

experiment_uses_fastwam() {
  validate_experiment_name "$1"
  echo "true"
}

experiment_uses_fasterwam_dot() {
  case "$1" in
    fastwam|fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking) echo "false" ;;
    fasterwam|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "true" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_uses_training_validation() {
  validate_experiment_name "$1"
  echo "false"
}

prepare_forced_retrain() {
  local name="$1"
  case "$name" in
    fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) ;;
    fastwam|fasterwam) return 0 ;;
    *) validate_experiment_name "$name"; return ;;
  esac
  if [[ "$FORCE_RETRAIN_IMPROVED_MODELS" != "true" && "$FORCE_RETRAIN_IMPROVED_MODELS" != "1" ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "1" ]]; then
    echo "[train-force] ${name}: dry run; existing artifacts are unchanged"
    return 0
  fi
  if [[ ! "$RETRAIN_GENERATION" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[ERROR] RETRAIN_GENERATION contains unsupported characters: $RETRAIN_GENERATION" >&2
    return 1
  fi

  local state_dir="$exp_root/retrain_state/$RETRAIN_GENERATION"
  local state_marker="$state_dir/${name}.prepared"
  if [[ -f "$state_marker" ]]; then
    echo "[train-force] ${name}: generation $RETRAIN_GENERATION already prepared; partial training may resume"
    return 0
  fi

  local backup_root="$exp_root/retrain_backups/${RETRAIN_GENERATION}_${retrain_backup_stamp}"
  local model_dir
  model_dir="$(experiment_dir "$name")"
  if [[ -e "$model_dir" ]]; then
    mkdir -p "$backup_root/models"
    mv "$model_dir" "$backup_root/models/$name"
    echo "[train-force] archived model: $backup_root/models/$name"
  fi
  if [[ -e "$eval_root/$name" ]]; then
    mkdir -p "$backup_root/online_eval"
    mv "$eval_root/$name" "$backup_root/online_eval/$name"
    echo "[train-force] archived eval: $backup_root/online_eval/$name"
  fi
  mkdir -p "$state_dir"
  printf '%s\n' "prepared_at=$(date --iso-8601=seconds)" > "$state_marker"
  printf '%s\n' "backup_root=$backup_root" >> "$state_marker"
}

experiment_uses_current_box_action_conditioning() {
  case "$1" in
    fastwam|fasterwam) echo "false" ;;
    fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "true" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_uses_historical_target_memory() {
  case "$1" in
    fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "true" ;;
    fastwam|fastwam_video_current_box_action_aligned|fasterwam|fasterwam_video_current_box_action_aligned) echo "false" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_uses_target_conditioning_adapter_only() {
  case "$1" in
    fasterwam_current_box_historical_target_memory) echo "true" ;;
    fastwam|fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam|fasterwam_video_current_box_action_aligned|fasterwam_current_box_capture_value_reranking) echo "false" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_uses_capture_value_reranking() {
  case "$1" in
    fastwam_current_box_capture_value_reranking|fasterwam_current_box_capture_value_reranking) echo "true" ;;
    fastwam|fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fasterwam|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory) echo "false" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_eval_semantic_signature() {
  case "$1" in
    fastwam_current_box_capture_value_reranking|fasterwam_current_box_capture_value_reranking)
      local prior_fingerprint="missing"
      if [[ -f "$capture_action_prior_checkpoint" ]]; then
        prior_fingerprint="$(sha256sum "$capture_action_prior_checkpoint" | cut -c1-16)"
      fi
      echo "capture_actionprior_v12_history_multiaxis_${capture_value_score_mode}_n${capture_value_candidate_count}_m${capture_value_selection_margin}_e${capture_value_min_center_error}_w${capture_action_prior_dimension_weights// /_}_p${prior_fingerprint}"
      ;;
    fastwam|fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fasterwam|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory) echo "standard" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_is_training_free() {
  case "$1" in
    fastwam_current_box_capture_value_reranking|fasterwam_current_box_capture_value_reranking) echo "true" ;;
    fastwam|fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fasterwam|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory) echo "false" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_teacher_batch_size() {
  case "$1" in
    fastwam|fasterwam) echo "$teacher_batch_size" ;;
    fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "$conditioned_teacher_batch_size" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_gradient_accumulation_steps() {
  case "$1" in
    fastwam|fasterwam) echo "$GRADIENT_ACCUMULATION_STEPS" ;;
    fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "$conditioned_gradient_accumulation_steps" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_tracker_mot_integration() {
  case "$1" in
    fastwam|fasterwam) echo "none" ;;
    fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "mot_tracker_finetune_local_feature" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_tracker_cache_root() {
  case "$1" in
    fastwam|fasterwam) echo "" ;;
    fastwam_video_current_box_action_aligned|fasterwam_video_current_box_action_aligned) echo "" ;;
    fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "$target_history_tracker_cache_root" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_init_checkpoint() {
  case "$1" in
    fastwam|fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fasterwam) echo "" ;;
    fastwam_current_box_capture_value_reranking) echo "$model_root/fastwam_current_box_historical_target_memory/best.pt" ;;
    fasterwam_video_current_box_action_aligned) echo "$model_root/fasterwam/best.pt" ;;
    fasterwam_current_box_historical_target_memory) echo "$model_root/fasterwam_video_current_box_action_aligned/best.pt" ;;
    fasterwam_current_box_capture_value_reranking) echo "$model_root/fasterwam_current_box_historical_target_memory/best.pt" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_tracker_finetune_checkpoint() {
  case "$1" in
    fastwam|fasterwam) echo "" ;;
    fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "$tracker_checkpoint" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_tracker_alignment_version() {
  case "$1" in
    fastwam|fasterwam) echo "$tracker_state_action_alignment_version" ;;
    fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "8" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_uses_model_driven_tracker_search() {
  case "$1" in
    fastwam|fasterwam) echo "false" ;;
    fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "true" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_uses_tracker_search_crop_jitter() {
  experiment_uses_model_driven_tracker_search "$1"
}

experiment_requires_tracker_cache() {
  case "$1" in
    fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "true" ;;
    fastwam|fastwam_video_current_box_action_aligned|fasterwam|fasterwam_video_current_box_action_aligned) echo "false" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_port() {
  case "$1" in
    fastwam) echo 29749 ;;
    fasterwam) echo 29750 ;;
    fastwam_video_current_box_action_aligned) echo 29751 ;;
    fastwam_current_box_historical_target_memory) echo 29752 ;;
    fasterwam_video_current_box_action_aligned) echo 29753 ;;
    fasterwam_current_box_historical_target_memory) echo 29754 ;;
    fastwam_current_box_capture_value_reranking) echo 29755 ;;
    fasterwam_current_box_capture_value_reranking) echo 29756 ;;
    *) validate_experiment_name "$1" ;;
  esac
}

eval_checkpoint_for_experiment() {
  case "$1" in
    fastwam_current_box_capture_value_reranking) echo "$model_root/fastwam_current_box_historical_target_memory/best.pt" ;;
    fasterwam_current_box_capture_value_reranking) echo "$model_root/fasterwam_current_box_historical_target_memory/best.pt" ;;
    fastwam|fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fasterwam|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory) echo "$model_root/$1/best.pt" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_uses_tracker_fallback() {
  local name="$1"
  validate_experiment_name "$name"
  if [[ ",${eval_tracker_fallback_experiments}," == *",${name},"* ]]; then
    echo "$eval_reuse_tracker_action_sequence"
  else
    echo "false"
  fi
}

experiment_current_box_action_layers() {
  case "$1" in
    fastwam|fasterwam|fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking) echo "18 23 26 29" ;;
    fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "0" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_tracker_fusion_start_layer() {
  case "$1" in
    fastwam|fasterwam|fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking) echo "$default_tracker_fusion_start_layer" ;;
    fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "0" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_tracker_condition_mode() {
  case "$1" in
    fastwam|fasterwam) echo "$tracker_condition_mode" ;;
    fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "none" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_tracker_feature_grid_size() {
  case "$1" in
    fastwam|fasterwam) echo "$tracker_feature_grid_size" ;;
    fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "16" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_tracker_local_position_embedding() {
  case "$1" in
    fastwam|fasterwam) echo "$tracker_local_position_embedding" ;;
    fastwam_video_current_box_action_aligned|fastwam_current_box_historical_target_memory|fastwam_current_box_capture_value_reranking|fasterwam_video_current_box_action_aligned|fasterwam_current_box_historical_target_memory|fasterwam_current_box_capture_value_reranking) echo "false" ;;
    *) validate_experiment_name "$1" ;;
  esac
}

experiment_train_next_target_relative() { validate_experiment_name "$1"; echo "false"; }
experiment_uses_target_relative_context() { validate_experiment_name "$1"; echo "false"; }
experiment_tracker_include_box_token() { validate_experiment_name "$1"; echo "true"; }
experiment_tracker_finetune_init() { validate_experiment_name "$1"; echo "uav_tracker"; }
experiment_uses_tracker_spatial_cross_attention() { validate_experiment_name "$1"; echo "false"; }
experiment_freezes_current_box_action_conditioner() { validate_experiment_name "$1"; echo "false"; }
experiment_x0_action_loss_weight() { validate_experiment_name "$1"; echo "0.0"; }

checkpoint_matches_train_config() {
  local ckpt_path="$1"
  local name="$2"
  validate_experiment_name "$name"
  local expected_fasterwam="$(experiment_uses_fasterwam_dot "$name")"
  local expected_current_box="$(experiment_uses_current_box_action_conditioning "$name")"
  local expected_history="$(experiment_uses_historical_target_memory "$name")"
  local expected_adapter_only="$(experiment_uses_target_conditioning_adapter_only "$name")"
  local expected_tracker_integration="$(experiment_tracker_mot_integration "$name")"
  local expected_tracker_checkpoint="$(experiment_tracker_finetune_checkpoint "$name")"
  local expected_tracker_cache="$(experiment_tracker_cache_root "$name")"
  local expected_tracker_alignment="$(experiment_tracker_alignment_version "$name")"
  local expected_model_driven_search="$(experiment_uses_model_driven_tracker_search "$name")"
  local expected_search_jitter="$(experiment_uses_tracker_search_crop_jitter "$name")"
  local expected_box_layers="$(experiment_current_box_action_layers "$name")"
  local expected_batch_size="$(experiment_teacher_batch_size "$name")"
  local expected_init_checkpoint="$(experiment_init_checkpoint "$name")"

  "$PYTHON_BIN" - "$ckpt_path" "$scene_list" "$trajectory_range" \
    "$expected_batch_size" "$expected_fasterwam" "$expected_current_box" \
    "$expected_history" "$expected_adapter_only" "$expected_tracker_integration" \
    "$expected_tracker_checkpoint" "$expected_tracker_cache" \
    "$expected_tracker_alignment" "$expected_model_driven_search" \
    "$expected_search_jitter" "$expected_box_layers" \
    "$action_sequence_horizon" "$action_video_freq_ratio" \
    "$action_yaw_loss_weight" "$target_history_length" \
    "$target_history_hidden_dim" "$target_history_num_layers" \
    "$target_history_num_heads" "$expected_init_checkpoint" <<'PY'
import math
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
expected = {
    "scene_list": sys.argv[2],
    "trajectory_range": sys.argv[3],
    "batch_size": int(sys.argv[4]),
    "fasterwam": sys.argv[5].lower() == "true",
    "current_box": sys.argv[6].lower() == "true",
    "history": sys.argv[7].lower() == "true",
    "adapter_only": sys.argv[8].lower() == "true",
    "tracker_integration": sys.argv[9],
    "tracker_checkpoint": sys.argv[10],
    "tracker_cache": sys.argv[11],
    "tracker_alignment": int(sys.argv[12]),
    "model_driven_search": sys.argv[13].lower() == "true",
    "search_jitter": sys.argv[14].lower() == "true",
    "box_layers": [int(value) for value in sys.argv[15].split()],
    "horizon": int(sys.argv[16]),
    "frequency_ratio": int(sys.argv[17]),
    "yaw_weight": float(sys.argv[18]),
    "history_length": int(sys.argv[19]),
    "history_dim": int(sys.argv[20]),
    "history_layers": int(sys.argv[21]),
    "history_heads": int(sys.argv[22]),
    "init_checkpoint": sys.argv[23],
}
if not path.is_file():
    raise SystemExit(1)
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except Exception:
    raise SystemExit(1)
cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
run_args = payload.get("run_args", {}) if isinstance(payload, dict) else {}
if not isinstance(cfg, dict) or not isinstance(run_args, dict):
    raise SystemExit(1)

def same_float(key, value):
    try:
        return math.isclose(float(cfg.get(key, "nan")), value, rel_tol=1e-6, abs_tol=1e-8)
    except (TypeError, ValueError):
        return False

checks = [
    bool(cfg.get("use_fastwam_mot", False)),
    bool(cfg.get("use_fasterwam_dot", False)) == expected["fasterwam"],
    bool(cfg.get("use_current_box_action_conditioning", False)) == expected["current_box"],
    bool(cfg.get("use_historical_target_memory", False)) == expected["history"],
    bool(cfg.get("target_conditioning_adapter_only", False)) == expected["adapter_only"],
    not bool(cfg.get("use_capture_value_reranking", False)),
    not bool(cfg.get("use_target_relative_context", False)),
    str(cfg.get("tracker_mot_integration", "none")) == expected["tracker_integration"],
    int(cfg.get("action_sequence_horizon", -1)) == expected["horizon"],
    int(cfg.get("fastwam_action_video_freq_ratio", -1)) == expected["frequency_ratio"],
    same_float("action_yaw_loss_weight", expected["yaw_weight"]),
    same_float("x0_action_loss_weight", 0.0),
    str(run_args.get("scene_list", "")) == expected["scene_list"],
    str(run_args.get("trajectory_range", "")) == expected["trajectory_range"],
    int(run_args.get("batch_size", -1)) == expected["batch_size"],
]
actual_init = str(run_args.get("init_checkpoint") or "")
if expected["init_checkpoint"]:
    checks.append(
        bool(actual_init)
        and Path(actual_init).expanduser().resolve()
        == Path(expected["init_checkpoint"]).expanduser().resolve()
    )
else:
    checks.append(not actual_init)
if expected["current_box"]:
    checks.extend([
        not bool(cfg.get("use_tracker_memory", True)),
        bool(cfg.get("tracker_include_box_token", False)),
        list(cfg.get("current_box_action_layers", [])) == expected["box_layers"],
        int(cfg.get("current_box_action_hidden_dim", -1)) == 1024,
        int(cfg.get("tracker_state_action_alignment_version", -1)) == expected["tracker_alignment"],
        bool(cfg.get("tracker_model_driven_search", False)) == expected["model_driven_search"],
        bool(cfg.get("tracker_search_crop_jitter", False)) == expected["search_jitter"],
        str(Path(str(cfg.get("tracker_finetune_checkpoint", ""))).expanduser().resolve())
        == str(Path(expected["tracker_checkpoint"]).expanduser().resolve()),
    ])
if expected["history"]:
    checks.extend([
        int(cfg.get("target_history_length", -1)) == expected["history_length"],
        int(cfg.get("target_history_hidden_dim", -1)) == expected["history_dim"],
        int(cfg.get("target_history_num_layers", -1)) == expected["history_layers"],
        int(cfg.get("target_history_num_heads", -1)) == expected["history_heads"],
        str(Path(str(cfg.get("target_history_tracker_cache_root", ""))).expanduser().resolve())
        == str(Path(expected["tracker_cache"]).expanduser().resolve()),
    ])
raise SystemExit(0 if all(checks) else 1)
PY
}

checkpoint_matches_training_stage() {
  local ckpt_path="$1"
  local expected_stage="$2"
  local expected_epochs="$3"
  local expected_max_steps="$4"
  local expected_s0_checkpoint="$5"
  "$PYTHON_BIN" - "$ckpt_path" "$expected_stage" "$expected_epochs" \
    "$expected_max_steps" "$expected_s0_checkpoint" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
stage = sys.argv[2]
epochs = int(sys.argv[3])
max_steps = int(sys.argv[4])
expected_s0 = sys.argv[5]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except Exception:
    sys.exit(1)
run_args = payload.get("run_args", {}) if isinstance(payload, dict) else {}
cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
if not isinstance(run_args, dict) or not isinstance(cfg, dict):
    sys.exit(1)
checks = [
    str(run_args.get("training_stage", "joint")) == stage,
    int(run_args.get("epochs", -1)) == epochs,
    int(run_args.get("max_train_steps", -1)) == max_steps,
    bool(cfg.get("include_current_localization_loss", True)) == (stage != "main"),
]
if stage == "main":
    actual_s0 = str(run_args.get("s0_localizer_checkpoint", ""))
    expected_s0_path = Path(expected_s0).expanduser().resolve()
    checks.append(
        bool(actual_s0)
        and Path(actual_s0).expanduser().resolve()
        == expected_s0_path
        and expected_s0_path.is_file()
        and path.stat().st_mtime_ns >= expected_s0_path.stat().st_mtime_ns
    )
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
  local expected_tracker_fallback="${7:-false}"
  local expected_tracker_threshold="${8:-0.5}"
  local expected_tracker_checkpoint="${9}"
  local expected_eval_semantic_signature="${10:-standard}"
  "$PYTHON_BIN" - "$summary_path" "$expected_scene_list" "$expected_trajectory_range" "$expected_target_context" "$expected_ckpt" "$expected_visualize_trajectory_keys" "$target_relative_context_scale" "$target_relative_token_scale" "$target_relative_context_hidden_dim" "$sampling_steps" "$eval_save_transformer_attention_maps" "$eval_save_rgb" "false" "$eval_save_target_crop_action_overlays" "$eval_target_crop_action_overlay_output_name" "$expected_tracker_fallback" "$expected_tracker_threshold" "$eval_attention_trajectory_keys" "$eval_save_trajectory_3d" "$eval_tracker_fallback_action_mode" "$eval_validate_camera_freshness" "$eval_camera_max_vehicle_distance" "$eval_camera_render_frames" "$eval_save_depth" "$eval_use_external_camera" "$eval_camera_only_virtual_uav" "$expected_tracker_checkpoint" "$eval_save_predicted_video" "$eval_predicted_video_trajectory_keys" "$predicted_video_latent_frames" "$eval_camera_capture_mode" "$eval_camera_render_max_fps" "$eval_camera_pose_tolerance_m" "$eval_camera_orientation_tolerance_deg" "$expected_eval_semantic_signature" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

summary = Path(sys.argv[1])
expected_scene = sys.argv[2]
expected_range = sys.argv[3]
expected_target_context = sys.argv[4].lower()
expected_ckpt_path = Path(sys.argv[5]).resolve()
expected_ckpt = str(expected_ckpt_path)
expected_visualize_keys = sys.argv[6]
expected_target_context_scale = float(sys.argv[7])
expected_target_token_scale = float(sys.argv[8])
expected_target_hidden = int(sys.argv[9])
expected_sampling_steps = int(sys.argv[10])
expected_attention_maps = sys.argv[11].lower() in {"1", "true", "yes", "on"}
expected_save_rgb = sys.argv[12].lower() in {"1", "true", "yes", "on"}
expected_save_ortrack = sys.argv[13].lower() in {"1", "true", "yes", "on"}
expected_action_overlays = sys.argv[14].lower() in {"1", "true", "yes", "on"}
expected_action_overlay_name = sys.argv[15]
expected_tracker_fallback = sys.argv[16].lower() in {"1", "true", "yes", "on"}
expected_tracker_threshold = float(sys.argv[17])
expected_attention_keys = sys.argv[18]
expected_save_trajectory_3d = sys.argv[19].lower() in {"1", "true", "yes", "on"}
expected_fallback_mode = sys.argv[20]
expected_camera_freshness = sys.argv[21].lower() in {"1", "true", "yes", "on"}
expected_camera_max_distance = float(sys.argv[22])
expected_camera_render_frames = int(sys.argv[23])
expected_save_depth = sys.argv[24].lower() in {"1", "true", "yes", "on"}
expected_external_camera = sys.argv[25].lower() in {"1", "true", "yes", "on"}
expected_camera_only_virtual_uav = sys.argv[26].lower() in {"1", "true", "yes", "on"}
expected_tracker_checkpoint_path = Path(sys.argv[27]).expanduser().resolve()
expected_tracker_checkpoint = str(expected_tracker_checkpoint_path)
expected_save_predicted_video = sys.argv[28].lower() in {"1", "true", "yes", "on"}
expected_predicted_video_keys = sys.argv[29]
expected_predicted_video_latent_frames = int(sys.argv[30])
expected_camera_capture_mode = sys.argv[31]
expected_camera_render_max_fps = float(sys.argv[32])
expected_camera_pose_tolerance_m = float(sys.argv[33])
expected_camera_orientation_tolerance_deg = float(sys.argv[34])
expected_eval_semantic_signature = sys.argv[35]
if not summary.exists():
    sys.exit(1)
if (
    not expected_ckpt_path.exists()
    or not expected_tracker_checkpoint_path.exists()
    or summary.stat().st_mtime < expected_ckpt_path.stat().st_mtime
    or summary.stat().st_mtime < expected_tracker_checkpoint_path.stat().st_mtime
):
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
if str(args.get("eval_semantic_signature", "standard")) != expected_eval_semantic_signature:
    sys.exit(1)
if str(Path(str(args.get("tracker_checkpoint", ""))).expanduser().resolve()) != expected_tracker_checkpoint:
    sys.exit(1)
if str(args.get("visualize_trajectory_keys", "")) != expected_visualize_keys:
    sys.exit(1)
if str(args.get("attention_trajectory_keys", "")) != expected_attention_keys:
    sys.exit(1)
if bool(args.get("save_predicted_video", False)) != expected_save_predicted_video:
    sys.exit(1)
if str(args.get("predicted_video_trajectory_keys", "")) != expected_predicted_video_keys:
    sys.exit(1)
if int(args.get("predicted_video_latent_frames", -1)) != expected_predicted_video_latent_frames:
    sys.exit(1)
if int(args.get("sampling_steps", -1)) != expected_sampling_steps:
    sys.exit(1)
if bool(args.get("save_rgb", True)) != expected_save_rgb:
    sys.exit(1)
if bool(args.get("save_ortrack_maps", False)) != expected_save_ortrack:
    sys.exit(1)
# Overlay generation is a resumable visualization step and does not change
# rollout semantics. Existing rollouts are completed by postprocessing.
if bool(args.get("save_trajectory_3d", True)) != expected_save_trajectory_3d:
    sys.exit(1)
if bool(args.get("reuse_last_confident_action_sequence", False)) != expected_tracker_fallback:
    sys.exit(1)
if expected_tracker_fallback and not math.isclose(
    float(args.get("tracker_detection_confidence_threshold", float("nan"))),
    expected_tracker_threshold,
    rel_tol=1e-6,
    abs_tol=1e-8,
):
    sys.exit(1)
if expected_tracker_fallback and str(args.get("tracker_fallback_action_mode", "remaining_sequence")) != expected_fallback_mode:
    sys.exit(1)
if bool(args.get("camera_only_virtual_uav", False)) != expected_camera_only_virtual_uav:
    sys.exit(1)
if bool(args.get("validate_camera_freshness", False)) != expected_camera_freshness:
    sys.exit(1)
if not math.isclose(
    float(args.get("camera_max_vehicle_distance", float("nan"))),
    expected_camera_max_distance,
    rel_tol=1e-6,
    abs_tol=1e-8,
):
    sys.exit(1)
if int(args.get("camera_render_frames", -1)) != expected_camera_render_frames:
    sys.exit(1)
if str(args.get("camera_capture_mode", "legacy_step")) != expected_camera_capture_mode:
    sys.exit(1)
if not math.isclose(float(args.get("camera_render_max_fps", float("nan"))), expected_camera_render_max_fps):
    sys.exit(1)
if not math.isclose(float(args.get("camera_pose_tolerance_m", float("nan"))), expected_camera_pose_tolerance_m):
    sys.exit(1)
if not math.isclose(float(args.get("camera_orientation_tolerance_deg", float("nan"))), expected_camera_orientation_tolerance_deg):
    sys.exit(1)
if bool(args.get("save_depth", True)) != expected_save_depth:
    sys.exit(1)
if bool(args.get("use_external_camera", False)) != expected_external_camera:
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
attention_tokens = tokens(expected_attention_keys)
predicted_video_tokens = tokens(expected_predicted_video_keys)
if visual_tokens and not any(t.lower() in {"all", "*", "none", "false", "off", "0"} for t in visual_tokens):
    out_dir = summary.parent
    expected_dirs = ["rgb"] if expected_save_rgb else []
    if expected_action_overlays:
        expected_dirs.append(expected_action_overlay_name)
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
if expected_attention_maps:
    out_dir = summary.parent
    for token in attention_tokens:
        traj_dir = out_dir / token
        asset_dir = traj_dir / "last_transformer_attention_maps"
        if not asset_dir.exists() or not any(asset_dir.rglob("*.png")):
            sys.exit(1)
if expected_save_predicted_video:
    out_dir = summary.parent
    for token in predicted_video_tokens:
        normalized = re.sub(r"/(\d+)$", lambda m: f"/trajectory_{int(m.group(1)):04d}", token)
        rollout_path = out_dir / normalized / "online_rollout.json"
        try:
            rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
            steps = rollout["steps"]
        except (OSError, ValueError, KeyError, TypeError):
            sys.exit(1)
        if not steps:
            sys.exit(1)
        for step in steps:
            paths = step.get("predicted_video_frames") if isinstance(step, dict) else None
            if not paths or any(not (rollout_path.parent / str(path)).is_file() for path in paths):
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

supported_experiments=${SUPPORTED_EXPERIMENTS[*]}
main_experiments=${EXPERIMENTS}
eval_extra_experiments=${EVAL_EXTRA_EXPERIMENTS}
fastwam=official 30-layer Video DiT + 30-layer Action DiT with MoT mixed attention
fastwam_video_current_box_action_aligned=independently trained FastWAM plus frozen complete Tracker b0 injected into Action layers 18/23/26/29
fastwam_current_box_historical_target_memory=independently trained FastWAM with Current Box and horizon-aligned historical target memory
fastwam_current_box_capture_value_reranking=training-free FastWAM shared-KV candidates selected by the frozen CaptureActionPrior
fasterwam=30-layer FastWAM Video DiT hub + KV-Fusion/RoPE-aligned single-layer DoT Action Head
fasterwam_video_current_box_action_aligned=Faster-WAM plus frozen complete Tracker b0 injected as an ungated Action-layer-0 residual
fasterwam_current_box_historical_target_memory=Current Box plus seven previous Tracker states encoded into eight horizon-specific Action conditions
fasterwam_current_box_capture_value_reranking=training-free shared-KV candidates selected by the frozen CaptureActionPrior on the historical-memory parent

scene_list=${scene_list}
trajectory_range=${trajectory_range}
training_validation=disabled
heldout_seen_scenes=${val_seen_scene_list}
heldout_seen_trajectory_range=${val_seen_trajectory_range}
heldout_unseen_scenes=${val_unseen_scene_list}
heldout_unseen_trajectory_range=${val_unseen_trajectory_range}
heldout_online_eval_scene_list=${val_scene_list}
heldout_online_eval_trajectory_range=${val_trajectory_range}
split_seed=${split_seed}

train_steps=${train_steps}
model_train_epochs=${model_train_epochs}
conditioned_train_steps=${conditioned_train_steps}
conditioned_train_epochs=${conditioned_train_epochs}
teacher_batch_size_per_gpu=${teacher_batch_size}
conditioned_teacher_batch_size_per_gpu=${conditioned_teacher_batch_size}
conditioned_gradient_accumulation_steps=${conditioned_gradient_accumulation_steps}
seq_len=${seq_len}
image_size=${image_size}
num_workers=${num_workers}
teacher_lr=${teacher_lr}
teacher_weight_decay=${teacher_weight_decay}
action_sequence_horizon=${action_sequence_horizon}
action_video_freq_ratio=${action_video_freq_ratio}
action_yaw_loss_weight=${action_yaw_loss_weight}
x0_action_loss_weight=0.0
diffusion_steps=${diffusion_steps}
sampling_steps=${sampling_steps}
max_vel=${max_vel}
max_yaw_rate=${max_yaw_rate}
max_speed_norm=${max_speed_norm}

use_wan22_encoders=${use_wan22_encoders}
wan22_model_base_path=${wan22_model_base_path}
wan22_fastwam_src_path=${wan22_fastwam_src_path}
wan22_skip_download=${wan22_skip_download}
wan_latent_cache_root=${wan_latent_cache_root}

run_tracker_training=${RUN_TRACKER_TRAINING}
tracker_only=${TRACKER_ONLY}
tracker_output_dir=${tracker_output_dir}
tracker_checkpoint=${tracker_checkpoint}
tracker_epochs=${tracker_epochs}
tracker_batch_size_per_gpu=${tracker_batch_size}
tracker_num_workers=${tracker_num_workers}
tracker_samples_per_epoch=${tracker_samples_per_epoch}
tracker_max_gap=${tracker_max_gap}
tracker_lr=${tracker_lr}
tracker_template_size=${tracker_template_size}
tracker_search_size=${tracker_search_size}
tracker_backbone_pretrained_path=${tracker_backbone_pretrained_path}

target_history_tracker_cache_root=${target_history_tracker_cache_root}
target_history_length=${target_history_length}
target_history_hidden_dim=${target_history_hidden_dim}
target_history_num_layers=${target_history_num_layers}
target_history_num_heads=${target_history_num_heads}
target_history_partial_probability=${target_history_partial_probability}
target_history_center_jitter_std=${target_history_center_jitter_std}
target_history_log_size_jitter_std=${target_history_log_size_jitter_std}
target_history_confidence_dropout_probability=${target_history_confidence_dropout_probability}

capture_value_score_mode=${capture_value_score_mode}
capture_value_candidate_count=${capture_value_candidate_count}
capture_value_structured_candidates=${capture_value_structured_candidates}
capture_value_selection_margin=${capture_value_selection_margin}
capture_value_min_center_error=${capture_value_min_center_error}
capture_action_prior_checkpoint=${capture_action_prior_checkpoint}
capture_action_prior_dimension_weights=${capture_action_prior_dimension_weights}
run_capture_action_prior_training=${RUN_CAPTURE_ACTION_PRIOR_TRAINING}

train_cuda_visible_devices=${TRAIN_GPU_IDS}
train_num_gpus=${train_num_gpus}
use_deepspeed=${USE_DEEPSPEED}
deepspeed_offload_optimizer=${DEEPSPEED_OFFLOAD_OPTIMIZER}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
checkpoint_save_every_epochs=${CHECKPOINT_SAVE_EVERY_EPOCHS}
save_best_checkpoint=${SAVE_BEST_CHECKPOINT}
save_optimizer_state=${SAVE_OPTIMIZER_STATE}

run_online_eval=${RUN_ONLINE_EVAL}
eval_cuda_visible_devices=${EVAL_GPU_IDS}
eval_scene_list=${eval_scene_list}
eval_trajectory_range=${eval_trajectory_range}
eval_parallel_jobs=${EVAL_PARALLEL_JOBS}
capture_distance=${capture_distance}
require_visibility_for_success=${require_visibility_for_success}
stop_on_collision=${stop_on_collision}
EOF
}

_run_teacher_stage() {
  local name="$1"
  local training_stage="${2:-joint}"
  local save_dir
  save_dir="$(experiment_dir "$name")"
  local s0_localizer_checkpoint=""
  if [[ "$(experiment_uses_current_box_action_conditioning "$name")" == "true" ]]; then
    # Every b0-conditioned model reuses the complete pretrained Tracker head.
    # The old two-stage S0 fine-tuning checkpoint remains load-compatible only.
    s0_localizer_checkpoint="$tracker_checkpoint"
    if [[ "$training_stage" == "s0" ]]; then
      save_dir="$model_root/${name}_s0_localizer"
    fi
  fi
  local use_diffusion_actor
  use_diffusion_actor="$(experiment_uses_diffusion "$name")"
  local use_fastwam_mot
  use_fastwam_mot="$(experiment_uses_fastwam "$name")"
  local use_fasterwam_dot
  use_fasterwam_dot="$(experiment_uses_fasterwam_dot "$name")"
  local use_current_box_action_conditioning
  use_current_box_action_conditioning="$(experiment_uses_current_box_action_conditioning "$name")"
  local use_historical_target_memory
  use_historical_target_memory="$(experiment_uses_historical_target_memory "$name")"
  local use_target_conditioning_adapter_only
  use_target_conditioning_adapter_only="$(experiment_uses_target_conditioning_adapter_only "$name")"
  local freeze_current_box_action_conditioner
  freeze_current_box_action_conditioner="$(experiment_freezes_current_box_action_conditioner "$name")"
  local current_x0_action_loss_weight
  current_x0_action_loss_weight="$(experiment_x0_action_loss_weight "$name")"
  local current_train_next_target_relative
  current_train_next_target_relative="$(experiment_train_next_target_relative "$name")"
  local use_target_relative_context
  use_target_relative_context="$(experiment_uses_target_relative_context "$name")"
  local current_teacher_batch_size
  current_teacher_batch_size="$(experiment_teacher_batch_size "$name")"
  local current_gradient_accumulation_steps
  current_gradient_accumulation_steps="$(experiment_gradient_accumulation_steps "$name")"
  local tracker_mot_integration
  tracker_mot_integration="$(experiment_tracker_mot_integration "$name")"
  local tracker_fusion_start_layer
  tracker_fusion_start_layer="$(experiment_tracker_fusion_start_layer "$name")"
  local current_tracker_condition_mode
  current_tracker_condition_mode="$(experiment_tracker_condition_mode "$name")"
  local current_tracker_local_position_embedding
  current_tracker_local_position_embedding="$(experiment_tracker_local_position_embedding "$name")"
  local current_tracker_feature_grid_size
  current_tracker_feature_grid_size="$(experiment_tracker_feature_grid_size "$name")"
  local current_tracker_include_box_token
  current_tracker_include_box_token="$(experiment_tracker_include_box_token "$name")"
  local current_train_steps="$train_steps"
  local current_model_train_epochs="$model_train_epochs"
  local current_val_ratio="0.0"
  local current_val_scene_list=""
  local current_val_trajectory_range=""
  local current_val_every_epochs="$val_every_epochs"
  local current_save_every_epochs="$CHECKPOINT_SAVE_EVERY_EPOCHS"
  if [[ "$use_current_box_action_conditioning" == "true" ]]; then
    current_train_steps="$conditioned_train_steps"
    current_model_train_epochs="$conditioned_train_epochs"
    current_val_every_epochs="$val_every_epochs"
    current_save_every_epochs="$CHECKPOINT_SAVE_EVERY_EPOCHS"
  fi
  local -a current_box_action_layers
  read -r -a current_box_action_layers <<< "$(experiment_current_box_action_layers "$name")"
  local current_tracker_alignment_version
  current_tracker_alignment_version="$(experiment_tracker_alignment_version "$name")"
  local use_tracker_spatial_cross_attention
  use_tracker_spatial_cross_attention="$(experiment_uses_tracker_spatial_cross_attention "$name")"
  local use_model_driven_tracker_search
  use_model_driven_tracker_search="$(experiment_uses_model_driven_tracker_search "$name")"
  local use_tracker_search_crop_jitter
  use_tracker_search_crop_jitter="$(experiment_uses_tracker_search_crop_jitter "$name")"
  local current_tracker_cache_root
  current_tracker_cache_root="$(experiment_tracker_cache_root "$name")"
  local tracker_finetune_checkpoint
  tracker_finetune_checkpoint="$(experiment_tracker_finetune_checkpoint "$name")"
  local tracker_finetune_init
  tracker_finetune_init="$(experiment_tracker_finetune_init "$name")"
  local init_checkpoint
  init_checkpoint="$(experiment_init_checkpoint "$name")"
  local init_args=()
  if [[ -n "$init_checkpoint" ]]; then
    init_args=(--init-checkpoint "$init_checkpoint")
  fi
  local uses_tracker_cache
  uses_tracker_cache="$(experiment_requires_tracker_cache "$name")"
  local master_port
  master_port="$(experiment_port "$name")"
  local log_file="$log_dir/${name}.log"
  if [[ "$training_stage" == "s0" ]]; then
    log_file="$log_dir/${name}_s0.log"
  fi
  if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "1" ]]; then
    log_file="${log_file%.log}.dry_run.log"
  fi
  local resume_ckpt="$save_dir/last.pt"
  local resume_args=()
  local restart_due_to_config=false

  if [[ "$SKIP_EXISTING_TRAIN" == "true" && -f "$save_dir/done.marker" && -f "$save_dir/best.pt" ]]; then
    if checkpoint_matches_train_config "$save_dir/best.pt" "$name" \
      && checkpoint_matches_training_stage "$save_dir/best.pt" "$training_stage" "$current_model_train_epochs" "$current_train_steps" "$s0_localizer_checkpoint"; then
      echo "[train-skip] ${name}: existing checkpoint matches requested train config"
      return 0
    fi
    echo "[train-rerun] ${name}: existing checkpoint does not match requested train config"
    rm -f "$save_dir/done.marker"
    restart_due_to_config=true
  fi
  if [[ "$restart_due_to_config" != "true" && -f "$resume_ckpt" ]] && { [[ ! -f "$save_dir/done.marker" ]] || [[ ! -f "$save_dir/best.pt" ]]; }; then
    if ! checkpoint_matches_train_config "$resume_ckpt" "$name" \
      || ! checkpoint_matches_training_stage "$resume_ckpt" "$training_stage" "$current_model_train_epochs" "$current_train_steps" "$s0_localizer_checkpoint"; then
      echo "[train-restart] ${name}: partial checkpoint does not match requested train config"
      rm -f "$resume_ckpt" "$save_dir/best.pt" "$save_dir/done.marker"
    elif [[ "$uses_tracker_cache" == "true" && "$resume_ckpt" -ot "$tracker_checkpoint" ]]; then
      echo "[train-restart] ${name}: partial checkpoint predates Tracker checkpoint"
    else
      resume_args+=(--resume "$resume_ckpt")
    fi
  fi

  if [[ -n "$init_checkpoint" && ! -f "$init_checkpoint" ]]; then
    echo "[ERROR] Missing initialization checkpoint for ${name}: $init_checkpoint" >&2
    return 1
  fi

  mkdir -p "$save_dir"
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"

  echo "============================================================" | tee "$log_file"
  echo "[train] ${name}" | tee -a "$log_file"
  echo "training_stage=${training_stage}" | tee -a "$log_file"
  echo "s0_localizer_checkpoint=${s0_localizer_checkpoint:-none}" | tee -a "$log_file"
  echo "save_dir=${save_dir}" | tee -a "$log_file"
  echo "use_diffusion_actor=${use_diffusion_actor}" | tee -a "$log_file"
  echo "use_fastwam_mot=${use_fastwam_mot}" | tee -a "$log_file"
  echo "use_fasterwam_dot=${use_fasterwam_dot}" | tee -a "$log_file"
  echo "use_current_box_action_conditioning=${use_current_box_action_conditioning}" | tee -a "$log_file"
  echo "use_historical_target_memory=${use_historical_target_memory}" | tee -a "$log_file"
  echo "target_conditioning_adapter_only=${use_target_conditioning_adapter_only}" | tee -a "$log_file"
  if [[ "$use_historical_target_memory" == "true" ]]; then
    echo "target_history_tracker_cache_root=${target_history_tracker_cache_root}" | tee -a "$log_file"
  fi
  echo "train_next_target_relative=${current_train_next_target_relative}" | tee -a "$log_file"
  echo "use_target_relative_context=${use_target_relative_context}" | tee -a "$log_file"
  echo "tracker_mot_integration=${tracker_mot_integration}" | tee -a "$log_file"
  echo "tracker_fusion_start_layer=${tracker_fusion_start_layer}" | tee -a "$log_file"
  echo "tracker_feature_grid_size=${current_tracker_feature_grid_size}" | tee -a "$log_file"
  echo "tracker_include_box_token=${current_tracker_include_box_token}" | tee -a "$log_file"
  echo "use_tracker_memory=$(if [[ "$use_current_box_action_conditioning" == "true" ]]; then echo false; else echo true; fi)" | tee -a "$log_file"
  echo "tracker_state_action_alignment_version=${current_tracker_alignment_version}" | tee -a "$log_file"
  echo "tracker_model_driven_search=${use_model_driven_tracker_search}" | tee -a "$log_file"
  echo "tracker_search_crop_jitter=${use_tracker_search_crop_jitter}" | tee -a "$log_file"
  echo "tracker_search_center_jitter_std=${tracker_search_center_jitter_std}" | tee -a "$log_file"
  echo "tracker_search_center_jitter_max=${tracker_search_center_jitter_max}" | tee -a "$log_file"
  echo "tracker_search_scale_jitter=${tracker_search_scale_jitter}" | tee -a "$log_file"
  echo "tracker_spatial_cross_attention=${use_tracker_spatial_cross_attention}" | tee -a "$log_file"
  echo "tracker_cache_root=${current_tracker_cache_root}" | tee -a "$log_file"
  echo "tracker_finetune_checkpoint=${tracker_finetune_checkpoint:-none}" | tee -a "$log_file"
  echo "tracker_finetune_init=${tracker_finetune_init}" | tee -a "$log_file"
  echo "tracker_backbone_pretrained_path=${tracker_backbone_pretrained_path}" | tee -a "$log_file"
  echo "init_checkpoint=${init_checkpoint:-none}" | tee -a "$log_file"
  echo "target_relative_context_scale=${target_relative_context_scale}" | tee -a "$log_file"
  echo "target_relative_token_scale=${target_relative_token_scale}" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$log_file"
  echo "train_num_gpus=${train_num_gpus}" | tee -a "$log_file"
  echo "batch_size_per_gpu=${current_teacher_batch_size}" | tee -a "$log_file"
  echo "gradient_accumulation_steps=${current_gradient_accumulation_steps}" | tee -a "$log_file"
  echo "effective_global_batch=$((current_teacher_batch_size * train_num_gpus * current_gradient_accumulation_steps))" | tee -a "$log_file"
  echo "train_steps=${current_train_steps}" | tee -a "$log_file"
  echo "epochs=${current_model_train_epochs}" | tee -a "$log_file"
  echo "val_ratio=${current_val_ratio}" | tee -a "$log_file"
  if [[ "$(experiment_uses_training_validation "$name")" == "true" ]]; then
    echo "training_validation=ratio_${current_val_ratio},every_${current_val_every_epochs}_epochs" | tee -a "$log_file"
  else
    echo "training_validation=disabled" | tee -a "$log_file"
  fi
  echo "heldout_online_eval=${val_scene_list}:${val_trajectory_range}" | tee -a "$log_file"
  echo "save_every_epochs=${current_save_every_epochs}" | tee -a "$log_file"
  echo "x0_action_loss_weight=${current_x0_action_loss_weight}" | tee -a "$log_file"
  if [[ ${#resume_args[@]} -gt 0 ]]; then
    echo "resume=${resume_ckpt}" | tee -a "$log_file"
  else
    echo "resume=none" | tee -a "$log_file"
  fi
  echo "use_deepspeed=${USE_DEEPSPEED}" | tee -a "$log_file"
  echo "swanlab=${use_swanlab}, project=${swanlab_project}, run=${swanlab_experiment_prefix}_${name}_${training_stage}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] ${name}: configuration validated; training launch skipped" | tee -a "$log_file"
    return 0
  fi

  train_launcher=("$PYTHON_BIN" -m train.train_teacher)
  if [[ "$USE_DEEPSPEED" == "true" || "$USE_DEEPSPEED" == "1" ]]; then
    train_launcher=(env -u CUDA_VISIBLE_DEVICES "$PYTHON_BIN" -m deepspeed.launcher.runner --include "localhost:${TRAIN_GPU_IDS}" --master_port "$master_port" --module train.train_teacher)
  fi

  "${train_launcher[@]}" \
    --dataset-root "$dataset_root" \
    --scene-list "$scene_list" \
    --trajectory-range "$trajectory_range" \
    --save-dir "$save_dir" \
    --training-stage "$training_stage" \
    --s0-localizer-checkpoint "$s0_localizer_checkpoint" \
    --epochs "$current_model_train_epochs" \
    --max-train-steps "$current_train_steps" \
    --batch-size "$current_teacher_batch_size" \
    --seq-len "$seq_len" \
    --image-size "$image_size" \
    --wan-latent-cache-root "$wan_latent_cache_root" \
    --ortrack-cache-root "$current_tracker_cache_root" \
    --val-scene-list "$current_val_scene_list" \
    --val-trajectory-range "$current_val_trajectory_range" \
    --val-ratio "$current_val_ratio" \
    --val-every-epochs "$current_val_every_epochs" \
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
    --train-next-target-relative "$current_train_next_target_relative" \
    --next-target-relative-loss-weight "$next_target_relative_loss_weight" \
    --prior-target-relative-loss-weight "$prior_target_relative_loss_weight" \
    --direct-action-loss-weight "$direct_action_loss_weight" \
    --action-yaw-loss-weight "$action_yaw_loss_weight" \
    --x0-action-loss-weight "$current_x0_action_loss_weight" \
    --use-target-relative-context "$use_target_relative_context" \
    --target-relative-context-scale "$target_relative_context_scale" \
    --target-relative-token-scale "$target_relative_token_scale" \
    --target-relative-context-hidden-dim "$target_relative_context_hidden_dim" \
    --tracker-mot-integration "$tracker_mot_integration" \
    --tracker-finetune-checkpoint "$tracker_finetune_checkpoint" \
    --tracker-finetune-init "$tracker_finetune_init" \
    --tracker-backbone-pretrained-path "$tracker_backbone_pretrained_path" \
    --tracker-template-size "$tracker_template_size" \
    --tracker-search-size "$tracker_search_size" \
    --tracker-feature-dim "$tracker_feature_dim" \
    --tracker-feature-grid-size "$current_tracker_feature_grid_size" \
    --tracker-use-local-position-embedding "$current_tracker_local_position_embedding" \
    --tracker-include-box-token "$current_tracker_include_box_token" \
    --tracker-condition-mode "$current_tracker_condition_mode" \
    --tracker-fusion-start-layer "$tracker_fusion_start_layer" \
    --tracker-fusion-gate-init "$tracker_fusion_gate_init" \
    --tracker-model-driven-search "$use_model_driven_tracker_search" \
    --tracker-search-crop-jitter "$use_tracker_search_crop_jitter" \
    --tracker-search-center-jitter-std "$tracker_search_center_jitter_std" \
    --tracker-search-center-jitter-max "$tracker_search_center_jitter_max" \
    --tracker-search-scale-jitter "$tracker_search_scale_jitter" \
    --tracker-state-action-alignment-version "$current_tracker_alignment_version" \
    --tracker-spatial-cross-attention "$use_tracker_spatial_cross_attention" \
    --use-current-box-action-conditioning "$use_current_box_action_conditioning" \
    --current-box-action-layers "${current_box_action_layers[@]}" \
    --current-box-action-hidden-dim 1024 \
    --current-box-action-gate-init 0.0 \
    --freeze-current-box-action-conditioner "$freeze_current_box_action_conditioner" \
    --use-historical-target-memory "$use_historical_target_memory" \
    --target-history-length "$target_history_length" \
    --target-history-hidden-dim "$target_history_hidden_dim" \
    --target-history-num-layers "$target_history_num_layers" \
    --target-history-num-heads "$target_history_num_heads" \
    --target-history-tracker-cache-root "$target_history_tracker_cache_root" \
    --target-history-partial-probability "$target_history_partial_probability" \
    --target-history-center-jitter-std "$target_history_center_jitter_std" \
    --target-history-log-size-jitter-std "$target_history_log_size_jitter_std" \
    --target-history-confidence-dropout-probability "$target_history_confidence_dropout_probability" \
    --target-conditioning-adapter-only "$use_target_conditioning_adapter_only" \
    --use-tracker-memory "$(if [[ "$use_current_box_action_conditioning" == "true" ]]; then echo false; else echo true; fi)" \
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
    --use-fasterwam-dot "$use_fasterwam_dot" \
    --save-every-epochs "$current_save_every_epochs" \
    --save-best-checkpoint "$SAVE_BEST_CHECKPOINT" \
    --save-optimizer-state "$SAVE_OPTIMIZER_STATE" \
    --deepspeed-offload-optimizer "$DEEPSPEED_OFFLOAD_OPTIMIZER" \
    --gradient-accumulation-steps "$current_gradient_accumulation_steps" \
    --use-swanlab "$use_swanlab" \
    --swanlab-project "$swanlab_project" \
    --swanlab-experiment-name "${swanlab_experiment_prefix}_${name}_${training_stage}" \
    --swanlab-workspace "$swanlab_workspace" \
    --swanlab-log-dir "$swanlab_log_dir" \
    --swanlab-mode "$swanlab_mode" \
    "${init_args[@]}" \
    "${resume_args[@]}" \
    $(if [[ "$USE_DEEPSPEED" == "true" || "$USE_DEEPSPEED" == "1" ]]; then printf '%s' '--deepspeed'; fi) \
    --multi-gpu \
    2>&1 | tee -a "$log_file"
}

run_teacher() {
  local name="$1"
  if [[ "$(experiment_is_training_free "$name")" == "true" ]]; then
    local parent_checkpoint
    parent_checkpoint="$(eval_checkpoint_for_experiment "$name")"
    if [[ ! -f "$parent_checkpoint" ]]; then
      echo "[ERROR] Training-free ${name} requires parent checkpoint: $parent_checkpoint" >&2
      return 1
    fi
    echo "[train-skip] ${name}: frozen action-prior reranker has no policy parameters"
    echo "[parent-reuse] ${name}: $parent_checkpoint"
    return 0
  fi
  prepare_forced_retrain "$name"
  if [[ "$(experiment_uses_current_box_action_conditioning "$name")" == "true" ]]; then
    if [[ ! -f "$tracker_checkpoint" ]]; then
      echo "[ERROR] Complete pretrained Tracker checkpoint is missing: $tracker_checkpoint" >&2
      return 1
    fi
    if [[ "$(experiment_uses_historical_target_memory "$name")" == "true" ]]; then
      if [[ ! -d "$target_history_tracker_cache_root" ]]; then
        echo "[ERROR] Historical Target Memory Tracker cache is missing: $target_history_tracker_cache_root" >&2
        return 1
      fi
    fi
    echo "[s0-reuse] ${name}: frozen complete Tracker head from $tracker_checkpoint"
    _run_teacher_stage "$name" main
  else
    _run_teacher_stage "$name" joint
  fi
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
  local eval_gpu_id="${9:-$EVAL_GPU_ID}"
  local eval_port="${10:-$sim_server_port}"
  local summary_shard_id="${11:-}"
  local out_dir="$out_root/$name"
  local log_suffix="${summary_shard_id:+_${summary_shard_id}}"
  local log_file="$log_root/${name}${log_suffix}.log"
  local partial_summary_path="$out_dir/summary_partial.json"
  local complete_summary_path="$out_dir/summary.json"
  if [[ -n "$summary_shard_id" ]]; then
    partial_summary_path="$out_dir/summary_partial_${summary_shard_id}.json"
    complete_summary_path="$out_dir/summary_${summary_shard_id}.json"
  fi
  local use_tracker_fallback
  local eval_semantic_signature
  local sim_gpu_id="$eval_gpu_id"
  use_tracker_fallback="$(experiment_uses_tracker_fallback "$name")"
  eval_semantic_signature="$(experiment_eval_semantic_signature "$name")"

  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] Missing checkpoint for eval: $ckpt" >&2
    exit 1
  fi
  if [[ "$SKIP_EXISTING_EVAL" == "false" || "$SKIP_EXISTING_EVAL" == "0" ]]; then
    rm -f "$partial_summary_path" "$complete_summary_path"
  fi
  # A merged multi-city summary is intentionally not equivalent to a completed
  # per-city shard. Reuse a valid shard even when its visualization settings
  # differ, because those settings do not affect the rollout metrics.
  if [[ -n "$summary_shard_id" && "$SKIP_EXISTING_EVAL" == "true" && -f "$complete_summary_path" ]]; then
    if "$PYTHON_BIN" - "$complete_summary_path" "$scene_list_for_eval" "$trajectory_range_for_eval" "$use_target_relative_context" "$ckpt" "$tracker_checkpoint" "$sampling_steps" "$use_tracker_fallback" "$eval_tracker_detection_confidence_threshold" "$eval_tracker_fallback_action_mode" "$eval_camera_only_virtual_uav" "$eval_validate_camera_freshness" "$eval_camera_max_vehicle_distance" "$eval_camera_render_frames" "$eval_save_depth" "$eval_use_external_camera" "$eval_save_predicted_video" "$eval_predicted_video_trajectory_keys" "$predicted_video_latent_frames" "$eval_camera_capture_mode" "$eval_camera_render_max_fps" "$eval_camera_pose_tolerance_m" "$eval_camera_orientation_tolerance_deg" "$eval_semantic_signature" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
scene, trajectory_range, target_context = sys.argv[2:5]
checkpoint = str(Path(sys.argv[5]).resolve())
tracker_checkpoint = str(Path(sys.argv[6]).resolve())
sampling_steps = int(sys.argv[7])
fallback = sys.argv[8].lower() in {"1", "true", "yes", "on"}
threshold = float(sys.argv[9])
fallback_mode = sys.argv[10]
camera_only = sys.argv[11].lower() in {"1", "true", "yes", "on"}
freshness = sys.argv[12].lower() in {"1", "true", "yes", "on"}
max_distance = float(sys.argv[13])
render_frames = int(sys.argv[14])
save_depth = sys.argv[15].lower() in {"1", "true", "yes", "on"}
external_camera = sys.argv[16].lower() in {"1", "true", "yes", "on"}
save_predicted_video = sys.argv[17].lower() in {"1", "true", "yes", "on"}
predicted_video_keys = sys.argv[18]
predicted_video_latent_frames = int(sys.argv[19])
camera_capture_mode = sys.argv[20]
camera_render_max_fps = float(sys.argv[21])
camera_pose_tolerance_m = float(sys.argv[22])
camera_orientation_tolerance_deg = float(sys.argv[23])
eval_semantic_signature = sys.argv[24]
range_match = re.fullmatch(r"(\d+)-(\d+)", trajectory_range)
if range_match is None:
    raise SystemExit(1)
range_start, range_end = (int(value) for value in range_match.groups())
expected_trajectory_count = range_end - range_start + 1
if expected_trajectory_count <= 0:
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = payload["args"]
    summaries = payload["summaries"]
except (OSError, ValueError, KeyError, TypeError):
    raise SystemExit(1)
ok = (
    len(summaries) == expected_trajectory_count
    and str(args.get("scene_list")) == scene
    and str(args.get("trajectory_range")) == trajectory_range
    and str(args.get("use_target_relative_context")).lower() == target_context.lower()
    and str(Path(str(args.get("checkpoint", ""))).resolve()) == checkpoint
    and Path(checkpoint).is_file()
    and path.stat().st_mtime_ns >= Path(checkpoint).stat().st_mtime_ns
    and str(args.get("eval_semantic_signature", "standard")) == eval_semantic_signature
    and str(Path(str(args.get("tracker_checkpoint", ""))).resolve()) == tracker_checkpoint
    and Path(tracker_checkpoint).is_file()
    and path.stat().st_mtime_ns >= Path(tracker_checkpoint).stat().st_mtime_ns
    and int(args.get("sampling_steps", -1)) == sampling_steps
    and bool(args.get("reuse_last_confident_action_sequence", False)) == fallback
    and bool(args.get("camera_only_virtual_uav", False)) == camera_only
    and bool(args.get("validate_camera_freshness", False)) == freshness
    and math.isclose(float(args.get("camera_max_vehicle_distance", float("nan"))), max_distance)
    and int(args.get("camera_render_frames", -1)) == render_frames
    and str(args.get("camera_capture_mode", "legacy_step")) == camera_capture_mode
    and math.isclose(float(args.get("camera_render_max_fps", float("nan"))), camera_render_max_fps)
    and math.isclose(float(args.get("camera_pose_tolerance_m", float("nan"))), camera_pose_tolerance_m)
    and math.isclose(float(args.get("camera_orientation_tolerance_deg", float("nan"))), camera_orientation_tolerance_deg)
    and bool(args.get("save_depth", True)) == save_depth
    and bool(args.get("use_external_camera", False)) == external_camera
    and all(str(item.get("failure_reason") or "").lower() != "runtime_error" for item in summaries if isinstance(item, dict))
)
if fallback:
    ok = ok and math.isclose(float(args.get("tracker_detection_confidence_threshold", float("nan"))), threshold) and str(args.get("tracker_fallback_action_mode")) == fallback_mode

def selected(raw, scene_id, trajectory_name):
    tokens = []
    for item in re.split(r"[,\s]+", str(raw or "")):
        token = item.strip().replace("\\", "/").replace(":", "/").strip("/").lower()
        if token:
            tokens.append(token)
    if not tokens or any(token in {"all", "*"} for token in tokens):
        return True
    scene_key = str(scene_id).lower()
    trajectory_key = str(trajectory_name).lower()
    candidates = {trajectory_key, f"{scene_key}/{trajectory_key}"}
    match = re.search(r"(\d+)$", trajectory_key)
    if match:
        idx = int(match.group(1))
        candidates.update({str(idx), f"{scene_key}/{idx}", f"{scene_key}/trajectory_{idx:04d}"})
    return any(token in candidates for token in tokens)

selected_summaries = [
    item for item in summaries
    if isinstance(item, dict)
    and selected(predicted_video_keys, item.get("scene_id", ""), item.get("trajectory_name", ""))
]
if save_predicted_video and selected_summaries:
    ok = (
        ok
        and bool(args.get("save_predicted_video", False))
        and str(args.get("predicted_video_trajectory_keys", "")) == predicted_video_keys
        and int(args.get("predicted_video_latent_frames", -1)) == predicted_video_latent_frames
    )
    for item in selected_summaries:
        traj_dir = path.parent / str(item["scene_id"]) / str(item["trajectory_name"])
        try:
            rollout = json.loads((traj_dir / "online_rollout.json").read_text(encoding="utf-8"))
            steps = rollout["steps"]
        except (OSError, ValueError, KeyError, TypeError):
            ok = False
            break
        if not steps or any(
            not isinstance(step, dict)
            or not step.get("predicted_video_frames")
            or any(not (traj_dir / str(frame)).is_file() for frame in step["predicted_video_frames"])
            for step in steps
        ):
            ok = False
            break
raise SystemExit(0 if ok else 1)
PY
    then
      echo "[eval-shard-skip] ${name}/${summary_shard_id}: valid completed city summary"
      return 0
    fi
    echo "[eval-shard-rerun] ${name}/${summary_shard_id}: semantic config or checkpoint changed"
    rm -f "$partial_summary_path" "$complete_summary_path"
  fi
  if [[ "$SKIP_EXISTING_EVAL" == "true" && -f "$out_dir/summary.json" ]]; then
    if summary_matches_eval_config "$out_dir/summary.json" "$scene_list_for_eval" "$trajectory_range_for_eval" "$use_target_relative_context" "$ckpt" "$eval_visualize_trajectory_keys" "$use_tracker_fallback" "$eval_tracker_detection_confidence_threshold" "$tracker_checkpoint" "$eval_semantic_signature"; then
      if "$PYTHON_BIN" - "$out_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summaries = payload.get("summaries", []) if isinstance(payload, dict) else []
has_runtime_error = any(
    isinstance(item, dict)
    and str(item.get("failure_reason") or "").lower() == "runtime_error"
    for item in summaries
)
raise SystemExit(0 if has_runtime_error else 1)
PY
      then
        echo "[eval-resume] ${name}: rerun runtime-error trajectories and keep completed rollouts"
        rm -f "$out_dir/summary.json"
      else
        echo "[eval-skip] ${name}: $out_dir/summary.json matches requested eval config"
        return 0
      fi
    else
      echo "[eval-resume] ${name}: existing summary.json does not match requested eval config; rerun"
      # Preserve valid per-city shards while discarding only the stale merged
      # result. They are reused and merged after the remaining cities finish.
      rm -f "$out_dir/summary.json" "$out_dir/summary_partial.json"
    fi
  fi

  mkdir -p "$out_dir" "$log_root"
  if [[ -f "$partial_summary_path" ]]; then
    if ! "$PYTHON_BIN" - "$partial_summary_path" "$eval_save_target_crop_action_overlays" "$eval_target_crop_action_overlay_output_name" "$use_tracker_fallback" "$eval_tracker_detection_confidence_threshold" "$eval_tracker_fallback_action_mode" "$eval_validate_camera_freshness" "$eval_camera_max_vehicle_distance" "$eval_camera_render_frames" "$eval_save_depth" "$eval_use_external_camera" "$eval_camera_only_virtual_uav" "$eval_camera_capture_mode" "$eval_camera_render_max_fps" "$eval_camera_pose_tolerance_m" "$eval_camera_orientation_tolerance_deg" "$eval_semantic_signature" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_enabled = sys.argv[2].lower() in {"1", "true", "yes", "on"}
expected_name = sys.argv[3]
expected_tracker_fallback = sys.argv[4].lower() in {"1", "true", "yes", "on"}
expected_tracker_threshold = float(sys.argv[5])
expected_fallback_mode = sys.argv[6]
expected_camera_freshness = sys.argv[7].lower() in {"1", "true", "yes", "on"}
expected_camera_max_distance = float(sys.argv[8])
expected_camera_render_frames = int(sys.argv[9])
expected_save_depth = sys.argv[10].lower() in {"1", "true", "yes", "on"}
expected_external_camera = sys.argv[11].lower() in {"1", "true", "yes", "on"}
expected_camera_only_virtual_uav = sys.argv[12].lower() in {"1", "true", "yes", "on"}
expected_camera_capture_mode = sys.argv[13]
expected_camera_render_max_fps = float(sys.argv[14])
expected_camera_pose_tolerance_m = float(sys.argv[15])
expected_camera_orientation_tolerance_deg = float(sys.argv[16])
expected_eval_semantic_signature = sys.argv[17]
try:
    args = json.loads(path.read_text(encoding="utf-8")).get("args", {})
except Exception:
    raise SystemExit(1)
ok = (
    bool(args.get("reuse_last_confident_action_sequence", False)) == expected_tracker_fallback
    and bool(args.get("camera_only_virtual_uav", False)) == expected_camera_only_virtual_uav
    and bool(args.get("validate_camera_freshness", False)) == expected_camera_freshness
    and abs(float(args.get("camera_max_vehicle_distance", float("nan"))) - expected_camera_max_distance) <= 1.0e-8
    and int(args.get("camera_render_frames", -1)) == expected_camera_render_frames
    and str(args.get("camera_capture_mode", "legacy_step")) == expected_camera_capture_mode
    and abs(float(args.get("camera_render_max_fps", float("nan"))) - expected_camera_render_max_fps) <= 1.0e-8
    and abs(float(args.get("camera_pose_tolerance_m", float("nan"))) - expected_camera_pose_tolerance_m) <= 1.0e-8
    and abs(float(args.get("camera_orientation_tolerance_deg", float("nan"))) - expected_camera_orientation_tolerance_deg) <= 1.0e-8
    and bool(args.get("save_depth", True)) == expected_save_depth
    and bool(args.get("use_external_camera", False)) == expected_external_camera
    and str(args.get("eval_semantic_signature", "standard")) == expected_eval_semantic_signature
    and (
        not expected_tracker_fallback
        or str(args.get("tracker_fallback_action_mode", "remaining_sequence")) == expected_fallback_mode
    )
    and (
        not expected_tracker_fallback
        or abs(float(args.get("tracker_detection_confidence_threshold", float("nan"))) - expected_tracker_threshold) <= 1.0e-8
    )
)
raise SystemExit(0 if ok else 1)
PY
    then
      echo "[eval-restart] ${name}: partial result does not match requested semantic/eval config"
      rm -f "$partial_summary_path" "$complete_summary_path"
    fi
  fi
  export DAGGER_MULTI_WORKER=1

  extra_eval_args=()
  extra_eval_args+=(--sim-gpu-id "$sim_gpu_id")
  extra_eval_args+=(--eval-semantic-signature "$eval_semantic_signature")
  if [[ "$(experiment_uses_capture_value_reranking "$name")" == "true" ]]; then
    extra_eval_args+=(
      --use-capture-value-reranking true
      --capture-value-score-mode "$capture_value_score_mode"
      --capture-value-candidate-count "$capture_value_candidate_count"
      --capture-value-selection-margin "$capture_value_selection_margin"
      --capture-value-min-center-error "$capture_value_min_center_error"
      --capture-action-prior-checkpoint "$capture_action_prior_checkpoint"
      --capture-action-prior-dimension-weights $capture_action_prior_dimension_weights
      --capture-value-structured-candidates "$capture_value_structured_candidates"
    )
  fi
  if [[ -n "$summary_shard_id" ]]; then
    extra_eval_args+=(--summary-shard-id "$summary_shard_id")
  fi
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
  extra_eval_args+=(--no-save-attention-tracker-comparisons)
  extra_eval_args+=(--attention-trajectory-keys "$eval_attention_trajectory_keys")
  if [[ "$eval_save_rgb" == "true" || "$eval_save_rgb" == "1" ]]; then
    extra_eval_args+=(--save-rgb)
  else
    extra_eval_args+=(--no-save-rgb)
  fi
  if [[ "$eval_save_predicted_video" == "true" || "$eval_save_predicted_video" == "1" ]]; then
    extra_eval_args+=(--save-predicted-video)
  else
    extra_eval_args+=(--no-save-predicted-video)
  fi
  extra_eval_args+=(--predicted-video-trajectory-keys "$eval_predicted_video_trajectory_keys")
  if [[ "$eval_save_target_crop_action_overlays" == "true" || "$eval_save_target_crop_action_overlays" == "1" ]]; then
    extra_eval_args+=(--save-target-crop-action-overlays)
  else
    extra_eval_args+=(--no-save-target-crop-action-overlays)
  fi
  extra_eval_args+=(--target-crop-action-overlay-output-name "$eval_target_crop_action_overlay_output_name")
  if [[ "$eval_save_trajectory_3d" == "true" || "$eval_save_trajectory_3d" == "1" ]]; then
    extra_eval_args+=(--save-trajectory-3d)
  else
    extra_eval_args+=(--no-save-trajectory-3d)
  fi
  if [[ "$eval_profile_step_time" == "true" || "$eval_profile_step_time" == "1" ]]; then
    extra_eval_args+=(--profile-step-time --profile-step-time-interval "$eval_profile_step_time_interval")
  fi
  if [[ "$eval_compile_action_sampling" == "true" || "$eval_compile_action_sampling" == "1" ]]; then
    extra_eval_args+=(--compile-action-sampling --compile-action-sampling-mode "$eval_compile_action_sampling_mode")
  else
    extra_eval_args+=(--no-compile-action-sampling)
  fi
  if [[ "$use_tracker_fallback" == "true" || "$use_tracker_fallback" == "1" ]]; then
    extra_eval_args+=(--reuse-last-confident-action-sequence)
    extra_eval_args+=(--tracker-detection-confidence-threshold "$eval_tracker_detection_confidence_threshold")
    extra_eval_args+=(--tracker-fallback-action-mode "$eval_tracker_fallback_action_mode")
  fi
  if [[ "$eval_camera_only_virtual_uav" == "true" || "$eval_camera_only_virtual_uav" == "1" ]]; then
    extra_eval_args+=(--camera-only-virtual-uav)
  else
    extra_eval_args+=(--no-camera-only-virtual-uav)
  fi
  if [[ "$eval_validate_camera_freshness" == "true" || "$eval_validate_camera_freshness" == "1" ]]; then
    extra_eval_args+=(--validate-camera-freshness)
  else
    extra_eval_args+=(--no-validate-camera-freshness)
  fi
  extra_eval_args+=(--camera-max-vehicle-distance "$eval_camera_max_vehicle_distance")
  extra_eval_args+=(--camera-render-frames "$eval_camera_render_frames")
  extra_eval_args+=(--camera-capture-mode "$eval_camera_capture_mode")
  extra_eval_args+=(--camera-render-max-fps "$eval_camera_render_max_fps")
  extra_eval_args+=(--camera-pose-tolerance-m "$eval_camera_pose_tolerance_m")
  extra_eval_args+=(--camera-orientation-tolerance-deg "$eval_camera_orientation_tolerance_deg")
  if [[ "$eval_save_depth" == "true" || "$eval_save_depth" == "1" ]]; then
    extra_eval_args+=(--save-depth)
  else
    extra_eval_args+=(--no-save-depth)
  fi
  if [[ "$eval_use_external_camera" == "true" || "$eval_use_external_camera" == "1" ]]; then
    extra_eval_args+=(--use-external-camera)
  else
    extra_eval_args+=(--no-use-external-camera)
  fi

  echo "============================================================" | tee "$log_file"
  echo "[online-eval] ${name}" | tee -a "$log_file"
  echo "checkpoint=${ckpt}" | tee -a "$log_file"
  echo "output=${out_dir}" | tee -a "$log_file"
  echo "scene_list=${scene_list_for_eval}" | tee -a "$log_file"
  echo "trajectory_range=${trajectory_range_for_eval}" | tee -a "$log_file"
  echo "visualize_trajectory_keys=${eval_visualize_trajectory_keys}" | tee -a "$log_file"
  echo "attention_trajectory_keys=${eval_attention_trajectory_keys}" | tee -a "$log_file"
  echo "save_transformer_attention_maps=${eval_save_transformer_attention_maps}" | tee -a "$log_file"
  echo "save_rgb=${eval_save_rgb}" | tee -a "$log_file"
  echo "save_predicted_video=${eval_save_predicted_video}" | tee -a "$log_file"
  echo "predicted_video_trajectory_keys=${eval_predicted_video_trajectory_keys}" | tee -a "$log_file"
  echo "predicted_video_latent_frames=${predicted_video_latent_frames}" | tee -a "$log_file"
  echo "save_target_crop_action_overlays=${eval_save_target_crop_action_overlays}" | tee -a "$log_file"
  echo "save_trajectory_3d=${eval_save_trajectory_3d}" | tee -a "$log_file"
  echo "profile_step_time=${eval_profile_step_time}" | tee -a "$log_file"
  echo "compile_action_sampling=${eval_compile_action_sampling}" | tee -a "$log_file"
  echo "compile_action_sampling_mode=${eval_compile_action_sampling_mode}" | tee -a "$log_file"
  echo "target_crop_action_overlay_output_name=${eval_target_crop_action_overlay_output_name}" | tee -a "$log_file"
  echo "reuse_last_confident_action_sequence=${use_tracker_fallback}" | tee -a "$log_file"
  echo "tracker_detection_confidence_threshold=${eval_tracker_detection_confidence_threshold}" | tee -a "$log_file"
  echo "tracker_fallback_action_mode=${eval_tracker_fallback_action_mode}" | tee -a "$log_file"
  echo "camera=$(if [[ "$eval_use_external_camera" == "true" || "$eval_use_external_camera" == "1" ]]; then echo external; else echo onboard; fi)" | tee -a "$log_file"
  echo "validate_camera_freshness=${eval_validate_camera_freshness}" | tee -a "$log_file"
  echo "camera_max_vehicle_distance=${eval_camera_max_vehicle_distance}" | tee -a "$log_file"
  echo "camera_render_frames=${eval_camera_render_frames}" | tee -a "$log_file"
  echo "camera_capture_mode=${eval_camera_capture_mode}" | tee -a "$log_file"
  echo "camera_render_max_fps=${eval_camera_render_max_fps}" | tee -a "$log_file"
  echo "camera_pose_tolerance_m=${eval_camera_pose_tolerance_m}" | tee -a "$log_file"
  echo "camera_orientation_tolerance_deg=${eval_camera_orientation_tolerance_deg}" | tee -a "$log_file"
  echo "save_depth=${eval_save_depth}" | tee -a "$log_file"
  echo "use_external_camera=${eval_use_external_camera}" | tee -a "$log_file"
  echo "camera_only_virtual_uav=${eval_camera_only_virtual_uav}" | tee -a "$log_file"
  echo "use_diffusion_actor=${use_diffusion_actor}" | tee -a "$log_file"
  echo "use_target_relative_context=${use_target_relative_context}" | tee -a "$log_file"
  echo "sampling_steps=${sampling_steps}" | tee -a "$log_file"
  echo "eval_semantic_signature=${eval_semantic_signature}" | tee -a "$log_file"
  echo "capture_distance=${capture_distance}" | tee -a "$log_file"
  echo "stop_on_collision=${stop_on_collision}" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=${eval_gpu_id}" | tee -a "$log_file"
  echo "sim_gpu_id=${sim_gpu_id}" | tee -a "$log_file"
  echo "sim_server_port=${eval_port}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  CUDA_VISIBLE_DEVICES="$eval_gpu_id" "$PYTHON_BIN" -m eval.online_eval_teacher \
    --dataset-root "$dataset_root" \
    --checkpoint "$ckpt" \
    --output-dir "$out_dir" \
    --executor-script "$executor_script" \
    --start-sim-server \
    --sim-server-script "$root_dir/code/src/envs/sim_server.py" \
    --sim-server-root-path "$root_dir" \
    --sim-server-log "$log_root/${name}${log_suffix}_sim_server.log" \
    --sim-server-wait-seconds 60 \
    --stop-sim-server-on-exit \
    --scene-list "$scene_list_for_eval" \
    --trajectory-range "$trajectory_range_for_eval" \
    --visualize-trajectory-keys "$eval_visualize_trajectory_keys" \
    --eval-split all \
    --sim-server-host "$sim_server_host" \
    --sim-server-port "$eval_port" \
    --scene-index "$scene_index" \
    --gpu-id "$eval_gpu_id" \
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
    --tracker-checkpoint "$tracker_checkpoint" \
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

capture_action_prior_matches_training_split() {
  local checkpoint="$1"
  "$PYTHON_BIN" - "$checkpoint" "$scene_list" "$trajectory_range" "$target_history_length" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
expected_scenes = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
try:
    start_text, end_text = sys.argv[3].split("-", 1)
    expected_range = [int(start_text), int(end_text)]
    expected_history_length = int(sys.argv[4])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
except (OSError, ValueError, KeyError, TypeError):
    raise SystemExit(1)
ok = (
    metadata.get("train_scenes") == expected_scenes
    and metadata.get("train_range") == expected_range
    and int(metadata.get("history_length", -1)) == expected_history_length
    and bool(metadata.get("validation_disabled", False))
)
raise SystemExit(0 if ok else 1)
PY
}

run_capture_action_prior_training() {
  local prior_script="$root_dir/code/scripts/run_capture_action_prior_training.sh"
  local range_start range_end
  IFS='-' read -r range_start range_end <<< "$trajectory_range"
  if [[ ! -f "$prior_script" ]]; then
    echo "[ERROR] CaptureActionPrior training script not found: $prior_script" >&2
    return 1
  fi
  if [[ ! -d "$target_history_tracker_cache_root" ]]; then
    echo "[ERROR] CaptureActionPrior Tracker cache is missing: $target_history_tracker_cache_root" >&2
    return 1
  fi
  echo "[capture-prior-train] scenes=$scene_list range=$trajectory_range validation=disabled"
  DATASET_ROOT="$dataset_root" \
  TARGET_HISTORY_TRACKER_CACHE_ROOT="$target_history_tracker_cache_root" \
  CAPTURE_ACTION_PRIOR_CHECKPOINT="$capture_action_prior_checkpoint" \
  SCENE_LIST="$scene_list" \
  TRAJECTORY_START="$range_start" \
  TRAJECTORY_END="$range_end" \
  PYTHON_BIN="$PYTHON_BIN" \
    bash "$prior_script"
}

ensure_capture_action_prior() {
  local required=false
  local name
  for name in "${experiment_names[@]}" "${eval_extra_experiment_names[@]}"; do
    if [[ "$(experiment_uses_capture_value_reranking "$name")" == "true" ]]; then
      required=true
      break
    fi
  done
  if [[ "$required" != "true" ]]; then
    return 0
  fi
  case "$RUN_CAPTURE_ACTION_PRIOR_TRAINING" in
    true|1)
      if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "1" ]]; then
        echo "[capture-prior-dry-run] forced training: $scene_list:$trajectory_range"
      else
        run_capture_action_prior_training
      fi
      ;;
    auto|"")
      if capture_action_prior_matches_training_split "$capture_action_prior_checkpoint"; then
        echo "[capture-prior-reuse] $capture_action_prior_checkpoint"
      elif [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "1" ]]; then
        echo "[capture-prior-dry-run] missing/stale checkpoint; would train $scene_list:$trajectory_range"
      else
        run_capture_action_prior_training
      fi
      ;;
    false|0)
      if ! capture_action_prior_matches_training_split "$capture_action_prior_checkpoint"; then
        echo "[ERROR] RUN_CAPTURE_ACTION_PRIOR_TRAINING=false but checkpoint is missing or stale: $capture_action_prior_checkpoint" >&2
        return 1
      fi
      echo "[capture-prior-reuse] $capture_action_prior_checkpoint"
      ;;
    *)
      echo "[ERROR] RUN_CAPTURE_ACTION_PRIOR_TRAINING must be auto, true, or false; got: $RUN_CAPTURE_ACTION_PRIOR_TRAINING" >&2
      return 1
      ;;
  esac
}

run_tracker_training() {
  local range_start range_end tracker_scenes tracker_script
  tracker_script="$root_dir/code/scripts/run_uav_tracker_training.sh"
  IFS='-' read -r range_start range_end <<< "$trajectory_range"
  if [[ -z "${range_start:-}" || -z "${range_end:-}" ]]; then
    echo "[ERROR] TRAJECTORY_RANGE must be START-END, got: $trajectory_range" >&2
    return 1
  fi
  if [[ ! -f "$tracker_script" ]]; then
    echo "[ERROR] Tracker training script not found: $tracker_script" >&2
    return 1
  fi

  tracker_scenes="${scene_list//,/ }"
  echo "[tracker-train] output=$tracker_output_dir"
  echo "[tracker-train] manifest=$tracker_manifest train=$scene_list:$trajectory_range training_validation=disabled"
  echo "[tracker-train] GPUs=$TRAIN_GPU_IDS epochs=$tracker_epochs batch_size=$tracker_batch_size"
  ROOT="$root_dir" \
  CODE_ROOT="$root_dir/code" \
  DATASET_ROOT="$dataset_root" \
  OUTPUT_DIR="$tracker_output_dir" \
  MANIFEST="$tracker_manifest" \
  GPUS="$TRAIN_GPU_IDS" \
  SCENES="$tracker_scenes" \
  TRAJECTORY_START="$range_start" \
  TRAJECTORY_END="$range_end" \
  EPOCHS="$tracker_epochs" \
  BATCH_SIZE="$tracker_batch_size" \
  NUM_WORKERS="$tracker_num_workers" \
  SAMPLES_PER_EPOCH="$tracker_samples_per_epoch" \
  VAL_SAMPLES="$tracker_val_samples" \
  MAX_GAP="$tracker_max_gap" \
  LR="$tracker_lr" \
  RESUME="$tracker_resume" \
  REBUILD_MANIFEST="$tracker_rebuild_manifest" \
  REQUIRE_REAL_ANNOTATIONS="$tracker_require_real_annotations" \
  RUN_TRACKER_CACHE_AFTER_TRAINING=false \
    bash "$tracker_script"

  if [[ ! -s "$tracker_checkpoint" ]]; then
    echo "[ERROR] Tracker training completed without checkpoint: $tracker_checkpoint" >&2
    return 1
  fi
  echo "[tracker-train] checkpoint ready: $tracker_checkpoint"

  if [[ "$run_tracker_heldout_eval" == "true" || "$run_tracker_heldout_eval" == "1" ]]; then
    local eval_start eval_end eval_gpu
    local -a eval_scenes
    IFS='-' read -r eval_start eval_end <<< "$tracker_eval_trajectory_range"
    IFS=',' read -ra eval_scenes <<< "$tracker_eval_scene_list"
    if [[ -z "${eval_start:-}" || -z "${eval_end:-}" || ${#eval_scenes[@]} -eq 0 ]]; then
      echo "[ERROR] Invalid Tracker held-out split: scenes=$tracker_eval_scene_list range=$tracker_eval_trajectory_range" >&2
      return 1
    fi
    eval_gpu="$tracker_eval_gpu_id"
    if [[ -z "$eval_gpu" ]]; then
      eval_gpu="${TRAIN_GPU_IDS%%,*}"
    fi
    if "$PYTHON_BIN" - "$tracker_eval_output" "$tracker_checkpoint" "$tracker_eval_manifest" \
      "$tracker_eval_scene_list" "$eval_start" "$eval_end" <<'PY'
import json
import re
import sys
from pathlib import Path

output = Path(sys.argv[1]).expanduser().resolve()
checkpoint = Path(sys.argv[2]).expanduser().resolve()
manifest = Path(sys.argv[3]).expanduser().resolve()
expected_scenes = {value.strip() for value in sys.argv[4].split(",") if value.strip()}
range_start, range_end = int(sys.argv[5]), int(sys.argv[6])
expected = len(expected_scenes) * (range_end - range_start + 1)
if not output.is_file() or not checkpoint.is_file() or not manifest.is_file():
    raise SystemExit(1)
if output.stat().st_mtime_ns < checkpoint.stat().st_mtime_ns:
    raise SystemExit(1)
try:
    payload = json.loads(output.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
sequences = payload.get("sequences", []) if isinstance(payload, dict) else []
try:
    saved_checkpoint = Path(str(summary.get("checkpoint", ""))).expanduser().resolve()
    saved_manifest = Path(str(summary.get("manifest", ""))).expanduser().resolve()
    trajectories = int(summary.get("trajectories", -1))
except (TypeError, ValueError):
    raise SystemExit(1)
observed = set()
for item in sequences if isinstance(sequences, list) else []:
    trajectory = Path(str(item.get("trajectory", ""))) if isinstance(item, dict) else Path()
    match = re.fullmatch(r"trajectory_(\d+)", trajectory.name)
    if match is None:
        raise SystemExit(1)
    observed.add((trajectory.parent.name, int(match.group(1))))
expected_keys = {
    (scene, trajectory_id)
    for scene in expected_scenes
    for trajectory_id in range(range_start, range_end + 1)
}
complete = (
    saved_checkpoint == checkpoint
    and saved_manifest == manifest
    and trajectories == expected
    and isinstance(sequences, list)
    and len(sequences) == expected
    and observed == expected_keys
)
raise SystemExit(0 if complete else 1)
PY
    then
      echo "[tracker-eval-skip] completed result matches checkpoint and held-out split: $tracker_eval_output"
      return 0
    fi
    "$PYTHON_BIN" -m tracking.build_manifest \
      --dataset-root "$dataset_root" \
      --output "$tracker_eval_manifest" \
      --scenes "${eval_scenes[@]}" \
      --trajectory-start "$eval_start" \
      --trajectory-end "$eval_end" \
      --box-shape square
    "$PYTHON_BIN" - "$tracker_eval_manifest" "${#eval_scenes[@]}" "$eval_start" "$eval_end" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
expected = int(sys.argv[2]) * (int(sys.argv[4]) - int(sys.argv[3]) + 1)
payload = json.loads(manifest.read_text(encoding="utf-8"))
counts = payload.get("counts", {})
total = int(counts.get("trajectories", -1))
real = int(counts.get("real", -1))
projected = int(counts.get("projected_weak", -1))
records = payload.get("train", []) + payload.get("val", []) + payload.get("test", [])

def has_complete_real_annotations(record):
    frames = record.get("frames")
    boxes = record.get("boxes_xywh")
    if (
        record.get("annotation_source") != "real"
        or not isinstance(frames, list)
        or not isinstance(boxes, list)
        or len(frames) != len(boxes)
    ):
        return False
    valid_boxes = [box for box in boxes if box is not None]
    return len(valid_boxes) >= 2 and all(
        isinstance(box, list)
        and len(box) == 4
        and float(box[2]) > 1.0
        and float(box[3]) > 1.0
        for box in valid_boxes
    )

fully_annotated = sum(
    1
    for record in records
    if has_complete_real_annotations(record)
)
if total != expected or real != expected or projected != 0 or fully_annotated != expected:
    raise SystemExit(
        "Tracker held-out manifest is incomplete or contains projected boxes: "
        f"expected={expected}, trajectories={total}, real={real}, projected_weak={projected}, "
        f"fully_annotated={fully_annotated}"
    )
print(f"[tracker-eval] verified complete real held-out boxes: {fully_annotated}/{expected}")
PY
    echo "[tracker-eval] GPU=$eval_gpu scenes=$tracker_eval_scene_list range=$tracker_eval_trajectory_range"
    CUDA_VISIBLE_DEVICES="$eval_gpu" "$PYTHON_BIN" -m tracking.evaluate \
      --checkpoint "$tracker_checkpoint" \
      --manifest "$tracker_eval_manifest" \
      --output "$tracker_eval_output" \
      --device cuda:0
    echo "[tracker-eval] result: $tracker_eval_output"
  fi
}

precompute_square_tracker_cache() {
  local cache_root="${1:-$target_history_tracker_cache_root}"
  local cache_feature_grid_size="${2:-$tracker_feature_grid_size}"
  local cache_gpu_ids="${TRACKER_CACHE_GPU_IDS:-$TRAIN_GPU_IDS}"
  local workers_per_gpu="${TRACKER_CACHE_WORKERS_PER_GPU:-4}"
  local cache_log_dir="${TRACKER_CACHE_LOG_DIR:-$root_dir/experiments/tracker_artifacts/logs/square_tracker_cache_logs}"
  local range_start range_end total_ranges total_workers chunk_size status scene_count expected_count actual_count
  local -a cache_gpus cache_pids

  IFS='-' read -r range_start range_end <<< "$trajectory_range"
  if [[ -z "${range_start:-}" || -z "${range_end:-}" ]]; then
    echo "[ERROR] TRAJECTORY_RANGE must be START-END, got: $trajectory_range" >&2
    return 1
  fi
  IFS=',' read -ra cache_gpus <<< "$cache_gpu_ids"
  if (( ${#cache_gpus[@]} == 0 )); then
    echo "[ERROR] No GPU configured for Square Tracker cache generation." >&2
    return 1
  fi
  mkdir -p "$cache_root" "$cache_log_dir"
  total_ranges=$((10#$range_end - 10#$range_start + 1))
  scene_count="$(awk -F',' '{print NF}' <<< "$scene_list")"
  expected_count=$((total_ranges * scene_count))
  actual_count="$("$PYTHON_BIN" - "$cache_root" "$scene_list" "$range_start" "$range_end" "$tracker_checkpoint" "$dataset_root" "$cache_feature_grid_size" "$tracker_feature_dim" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
scenes = [value.strip() for value in sys.argv[2].split(",") if value.strip()]
start, end = int(sys.argv[3]), int(sys.argv[4])
expected_checkpoint = str(Path(sys.argv[5]).expanduser().resolve())
dataset_root = Path(sys.argv[6]).expanduser().resolve()
tracker_feature_grid_size = int(sys.argv[7])
tracker_feature_dim = int(sys.argv[8])
checkpoint_stat = Path(expected_checkpoint).stat()
expected_checkpoint_size = int(checkpoint_stat.st_size)
expected_checkpoint_mtime_ns = int(checkpoint_stat.st_mtime_ns)
valid = 0
for scene in scenes:
    for index in range(start, end + 1):
        summary_path = root / scene / f"trajectory_{index:04d}" / "summary.json"
        trajectory_path = dataset_root / scene / f"trajectory_{index:04d}"
        expected_frames = len(list((trajectory_path / "rgb").glob("frame_*.png")))
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            frames = summary.get("frames")
            initialization = summary.get("initialization")
            delayed_initialization = (
                isinstance(initialization, dict)
                and int(initialization.get("frame", 0)) > 0
            )
            valid_frame0_source = summary.get("frame0_heatmap_source") == "tracker_response" or (
                delayed_initialization
                and summary.get("frame0_heatmap_source") == "target_absent_before_initialization"
            )
            complete = (
                summary.get("tracker_backend") == "square"
                and summary.get("checkpoint") == expected_checkpoint
                and int(summary.get("checkpoint_size", -1)) == expected_checkpoint_size
                and int(summary.get("checkpoint_mtime_ns", -1)) == expected_checkpoint_mtime_ns
                and isinstance(frames, list)
                and expected_frames > 0
                and len(frames) == expected_frames
                and int(summary.get("frame_count", -1)) == expected_frames
                and isinstance(initialization, dict)
                and str(initialization.get("backend", "")) == "gt_segmentation_bbox_square"
                and valid_frame0_source
                and int(summary.get("tracker_feature_cache_version", 0)) == 1
                and summary.get("tracker_feature_grid_size")
                == [tracker_feature_grid_size, tracker_feature_grid_size]
                and int(summary.get("tracker_feature_dim", -1)) == tracker_feature_dim
                and all(
                    isinstance(frame, dict)
                    and isinstance(frame.get("heatmap"), str)
                    and (summary_path.parent / frame["heatmap"]).is_file()
                    and isinstance(frame.get("tracker_features"), str)
                    and (summary_path.parent / frame["tracker_features"]).is_file()
                    for frame in frames
                )
            )
        except (AttributeError, OSError, ValueError, TypeError):
            complete = False
        valid += int(complete)
print(valid)
PY
)"
  if (( actual_count >= expected_count )); then
    echo "[tracker-cache] complete cache already exists: $cache_root ($actual_count/$expected_count)"
    return 0
  fi
  echo "[tracker-cache] filling incomplete cache: $cache_root ($actual_count/$expected_count)"
  total_workers=$(( ${#cache_gpus[@]} * workers_per_gpu ))
  chunk_size=$(( (total_ranges + total_workers - 1) / total_workers ))
  cache_pids=()

  for ((worker=0; worker<total_workers; worker++)); do
    local start end range gpu log_file
    start=$((10#$range_start + worker * chunk_size))
    (( start > 10#$range_end )) && break
    end=$((start + chunk_size - 1))
    (( end > 10#$range_end )) && end=$((10#$range_end))
    gpu="${cache_gpus[$((worker % ${#cache_gpus[@]}))]}"
    gpu="${gpu//[[:space:]]/}"
    range="$start-$end"
    log_file="$cache_log_dir/gpu${gpu}_${range}.log"
    echo "[tracker-cache] gpu=$gpu range=$range scenes=$scene_list"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      "$PYTHON_BIN" -m eval.square_tracker_cache_batch \
      --dataset-root "$dataset_root" \
      --cache-root "$cache_root" \
      --checkpoint "$tracker_checkpoint" \
      --scenes "$scene_list" \
      --trajectory-range "$range" \
      --require-real-init-bbox \
      --square-init-bbox \
      --native-first-frame-response \
      --require-tracker-features \
      --tracker-feature-grid-size "$cache_feature_grid_size" \
      --tracker-feature-dim "$tracker_feature_dim" \
      --device cuda >"$log_file" 2>&1 &
    cache_pids+=("$!")
  done

  status=0
  for pid in "${cache_pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "[ERROR] Square Tracker cache generation failed; inspect $cache_log_dir" >&2
    return "$status"
  fi
  echo "[tracker-cache] complete: $cache_root"
}

write_manifest
csv_to_array "$EXPERIMENTS" experiment_names
csv_to_array "$EVAL_EXTRA_EXPERIMENTS" eval_extra_experiment_names

for name in "${experiment_names[@]}" "${eval_extra_experiment_names[@]}"; do
  validate_experiment_name "$name" || exit 1
done

echo "[ablation] experiment root: $exp_root"
echo "[ablation] experiments: ${experiment_names[*]}"
echo "[ablation] eval-only experiments: ${eval_extra_experiment_names[*]}"
echo "[ablation] train GPUs: $TRAIN_GPU_IDS (num_gpus=$train_num_gpus, deepspeed=$USE_DEEPSPEED)"
echo "[ablation] eval GPU pool: ${EVAL_GPU_POOL[*]}"
echo "[ablation] train_steps: $train_steps"
echo "[ablation] model_train_epochs: $model_train_epochs"
echo "[ablation] skip existing train/eval: $SKIP_EXISTING_TRAIN / $SKIP_EXISTING_EVAL"
echo "[ablation] force improved retrain: $FORCE_RETRAIN_IMPROVED_MODELS (generation=$RETRAIN_GENERATION)"
echo "[ablation] supported model chain: ${SUPPORTED_EXPERIMENTS[*]}"

needs_pretrained_tracker=false
for name in "${experiment_names[@]}" "${eval_extra_experiment_names[@]}"; do
  if [[ "$(experiment_requires_tracker_cache "$name")" == "true" \
    || -n "$(experiment_tracker_finetune_checkpoint "$name")" ]]; then
    needs_pretrained_tracker=true
    break
  fi
done

if [[ "$needs_pretrained_tracker" != "true" ]]; then
  echo "[ablation] selected experiments do not require a pretrained Tracker checkpoint"
else
case "$RUN_TRACKER_TRAINING" in
  true|1)
    run_tracker_training
    ;;
  auto|"")
    if [[ -s "$tracker_checkpoint" ]]; then
      echo "[ablation] RUN_TRACKER_TRAINING=auto; reusing existing Tracker checkpoint: $tracker_checkpoint"
    else
      echo "[ablation] RUN_TRACKER_TRAINING=auto; Tracker checkpoint missing, starting Tracker training"
      run_tracker_training
    fi
    ;;
  false|0)
    if [[ ! -s "$tracker_checkpoint" ]]; then
      echo "[ERROR] RUN_TRACKER_TRAINING=false and Tracker checkpoint is missing: $tracker_checkpoint" >&2
      exit 1
    fi
    echo "[ablation] RUN_TRACKER_TRAINING=false; using Tracker checkpoint: $tracker_checkpoint"
    ;;
  *)
    echo "[ERROR] RUN_TRACKER_TRAINING must be auto, true, or false; got: $RUN_TRACKER_TRAINING" >&2
    exit 1
    ;;
esac
fi

needs_tracker_cache=false
if [[ "$RUN_TEACHER_ABLATIONS" == "true" ]]; then
  for name in "${experiment_names[@]}"; do
    if [[ "$(experiment_requires_tracker_cache "$name")" == "true" ]]; then
      needs_tracker_cache=true
      break
    fi
  done
fi
if [[ "$RUN_TRACKER_CACHE_PRECOMPUTE" == "true" && "$needs_tracker_cache" == "true" ]]; then
  echo "[ablation] precomputing/verifying historical-target Tracker cache: $target_history_tracker_cache_root"
  precompute_square_tracker_cache "$target_history_tracker_cache_root" "$tracker_feature_grid_size"
elif [[ "$needs_tracker_cache" == "true" ]]; then
  echo "[ablation] RUN_TRACKER_CACHE_PRECOMPUTE=false; training requires the configured Tracker caches"
fi

ensure_capture_action_prior

summary_models=()
for name in "${experiment_names[@]}"; do
  summary_models+=("$name")
done
for name in "${eval_extra_experiment_names[@]}"; do
  summary_models+=("$name")
done

merge_city_eval_shards() {
  local out_dir="$1"
  shift
  local trajectory_spec="$1"
  shift
  "$PYTHON_BIN" - "$out_dir" "$trajectory_spec" "$@" <<'PY'
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
trajectory_spec = sys.argv[2]
shards = sys.argv[3:]
summaries = []
template = None
trajectory_ranges = {}
for shard in shards:
    path = out / f"summary_{shard}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    shard_args = payload.get("args", {})
    shard_range = str(shard_args.get("trajectory_range", ""))
    match = re.fullmatch(r"(\d+)-(\d+)", shard_range)
    if match is None:
        raise RuntimeError(f"Invalid trajectory range in {path}: {shard_range!r}")
    start, end = (int(value) for value in match.groups())
    expected_count = end - start + 1
    shard_summaries = payload.get("summaries", [])
    if str(shard_args.get("scene_list", "")) != shard:
        raise RuntimeError(f"Scene mismatch in {path}")
    if expected_count <= 0 or len(shard_summaries) != expected_count:
        raise RuntimeError(
            f"Incomplete shard {path}: expected {expected_count}, got {len(shard_summaries)}"
        )
    trajectory_ranges[shard] = shard_range
    template = payload if template is None else template
    summaries.extend(shard_summaries)
unique = {}
for item in summaries:
    unique[(str(item.get("scene_id")), str(item.get("trajectory_name")))] = item
summaries = [unique[key] for key in sorted(unique)]
def mean(key, boolean=False):
    values = []
    for item in summaries:
        value = item.get(key)
        if value is not None:
            values.append(1.0 if boolean and bool(value) else (0.0 if boolean else float(value)))
    return float(np.mean(values)) if values else None
failure = Counter(str(item.get("failure_reason") or "unknown") for item in summaries)
agg = dict(template or {})
agg_args = dict(agg.get("args", {}))
agg_args["scene_list"] = ",".join(shards)
agg_args["trajectory_range"] = trajectory_spec
agg_args["trajectory_ranges_by_scene"] = trajectory_ranges
agg.update({
    "num_trajectories": len(summaries),
    "SR": mean("success", True), "success_rate": mean("success", True),
    "ATF": mean("effective_tracked_frames"), "average_tracked_frames": mean("effective_tracked_frames"),
    "average_tracked_frame_ratio": mean("effective_tracking_ratio"),
    "CTF": mean("consecutive_tracked_frames_before_failure"),
    "consecutive_tracked_frames": mean("consecutive_tracked_frames_before_failure"),
    "average_consecutive_tracked_frame_ratio_before_failure": mean("consecutive_tracked_frame_ratio_before_failure"),
    "average_effective_tracked_frames": mean("effective_tracked_frames"),
    "mean_effective_tracking_ratio": mean("effective_tracking_ratio"),
    "mean_close_frame_ratio": mean("close_frame_ratio"),
    "mean_visible_frame_ratio": mean("visible_frame_ratio"),
    "mean_collision_frame_ratio": mean("collision_frame_ratio"),
    "final_close_rate": mean("final_close_enough", True),
    "final_visible_rate": mean("final_visible_by_geometry", True),
    "collision_rate": mean("collision", True),
    "failure_reason_counts": dict(failure),
    "mean_final_distance": mean("final_distance"), "mean_distance": mean("mean_distance"),
    "args": agg_args,
    "summaries": summaries,
})
tmp = out / "summary.json.tmp"
tmp.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
tmp.replace(out / "summary.json")
partial = out / "summary_partial.json.tmp"
partial.write_text(json.dumps({"summaries": summaries, "args": agg.get("args", {})}, indent=2, ensure_ascii=False), encoding="utf-8")
partial.replace(out / "summary_partial.json")
print(f"[eval-merge] {out}: {len(summaries)} trajectories from {shards}")
PY
}

prepare_online_eval() {
  if [[ -z "$EVAL_GPU_OVERRIDE" && "$AUTO_SELECT_EVAL_GPUS" == "true" ]]; then
    EVAL_GPU_IDS="$(select_gpus_by_free_memory "$EVAL_GPU_MIN_FREE_MEM_GB")"
    if [[ -z "$EVAL_GPU_IDS" ]]; then
      echo "[ERROR] No evaluation GPU has at least ${EVAL_GPU_MIN_FREE_MEM_GB}GB free after training." >&2
      nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader || true
      exit 1
    fi
    IFS=',' read -ra EVAL_GPU_POOL <<< "$EVAL_GPU_IDS"
    EVAL_GPU_ID="${EVAL_GPU_POOL[0]}"
  fi
  eval_gpu_count="${#EVAL_GPU_POOL[@]}"
  if (( EVAL_PARALLEL_JOBS < 1 )); then
    echo "[ERROR] EVAL_PARALLEL_JOBS must be at least 1" >&2
    exit 1
  fi
  if (( eval_gpu_count > EVAL_PARALLEL_JOBS )); then
    eval_gpu_count="$EVAL_PARALLEL_JOBS"
  fi
  IFS=',' read -ra eval_city_merge_order <<< "$eval_scene_list"
  # Put the longest city shards in the same parallel batch. This keeps the
  # evaluation set and merged-summary ordering unchanged while preventing the
  # three 500-trajectory unseen cities from occupying two sequential batches.
  mapfile -t eval_city_pool < <(
    for city in "${eval_city_merge_order[@]}"; do
      city_range="$(eval_trajectory_range_for_city "$city")"
      IFS='-' read -r city_start city_end <<< "$city_range"
      printf '%09d\t%s\n' "$((10#$city_end - 10#$city_start + 1))" "$city"
    done | sort -k1,1nr -k2,2V | cut -f2
  )
  if (( eval_gpu_count > ${#eval_city_pool[@]} )); then
    eval_gpu_count="${#eval_city_pool[@]}"
  fi
  echo "[ablation] starting online eval on GPU pool: ${EVAL_GPU_POOL[*]}"
}

eval_summary_is_successful() {
  local summary_path="$1"
  "$PYTHON_BIN" - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
summaries = payload.get("summaries", []) if isinstance(payload, dict) else []
ok = bool(summaries) and all(
    not (
        isinstance(item, dict)
        and str(item.get("failure_reason") or "").strip().lower() == "runtime_error"
    )
    for item in summaries
)
raise SystemExit(0 if ok else 1)
PY
}

evaluate_model_online() {
  local name="$1"
  local ckpt
  ckpt="$(eval_checkpoint_for_experiment "$name")"
  if [[ "$SKIP_EXISTING_EVAL" == "true" && -f "$eval_root/$name/summary.json" ]] && \
     summary_matches_eval_config "$eval_root/$name/summary.json" "$eval_scene_list" "$eval_trajectory_range" "$(experiment_uses_target_relative_context "$name")" "$ckpt" "$eval_visualize_trajectory_keys" "$(experiment_uses_tracker_fallback "$name")" "$eval_tracker_detection_confidence_threshold" "$tracker_checkpoint" "$(experiment_eval_semantic_signature "$name")" && \
     eval_summary_is_successful "$eval_root/$name/summary.json"; then
    echo "[eval-skip] ${name}: complete merged summary exists"
    return 0
  fi
  for ((batch_start=0; batch_start<${#eval_city_pool[@]}; batch_start+=eval_gpu_count)); do
    eval_pids=()
    eval_names=()
    for ((slot=0; slot<eval_gpu_count && batch_start+slot<${#eval_city_pool[@]}; slot++)); do
      city="${eval_city_pool[batch_start+slot]}"
      gpu_id="${EVAL_GPU_POOL[slot]}"
      city_trajectory_range="$(eval_trajectory_range_for_city "$city")"
      # AirSim scenes occupy 30001, 30002, ... based on the City number.
      # Keep manager ports in separate 1000-port blocks to avoid colliding
      # with those scene processes during parallel evaluation.
      eval_port="$((sim_server_port + slot * 1000))"
      echo "[eval-dispatch] model=${name} city=${city} range=${city_trajectory_range} gpu=${gpu_id} manager_port=${eval_port}"
      (
        run_online_eval \
          "$name" \
          "$ckpt" \
          "$(experiment_uses_diffusion "$name")" \
          "$(experiment_uses_target_relative_context "$name")" \
          "$eval_root" \
          "$eval_log_dir" \
          "$city" \
          "$city_trajectory_range" \
          "$gpu_id" \
          "$eval_port" \
          "$city"
      ) &
      eval_pids+=("$!")
      eval_names+=("${name}/${city}:${city_trajectory_range}")
    done
    eval_failed=0
    for ((job=0; job<${#eval_pids[@]}; job++)); do
      if ! wait "${eval_pids[job]}"; then
        echo "[ERROR] Online eval failed: ${eval_names[job]}" >&2
        eval_failed=1
      fi
    done
    if [[ "$eval_failed" == "1" ]]; then
      return 1
    fi
  done
  merge_city_eval_shards "$eval_root/$name" "$eval_trajectory_range" "${eval_city_merge_order[@]}"
  if ! "$PYTHON_BIN" - "$eval_root/$name/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
summaries = payload.get("summaries", []) if isinstance(payload, dict) else []
runtime_errors = [
    item for item in summaries
    if isinstance(item, dict)
    and str(item.get("failure_reason") or "").strip().lower() == "runtime_error"
]
if runtime_errors:
    print(
        f"[eval-invalid] {path}: {len(runtime_errors)}/{len(summaries)} trajectories "
        "ended with runtime_error",
        file=sys.stderr,
    )
    raise SystemExit(1)
if not summaries:
    print(f"[eval-invalid] {path}: no trajectory summaries", file=sys.stderr)
    raise SystemExit(1)
print(f"[eval-valid] {path}: {len(summaries)} trajectories, no runtime_error")
PY
  then
    echo "[ERROR] Online eval produced invalid runtime results: ${name}" >&2
    rm -f "$eval_root/$name/summary.json"
    return 1
  fi
  if [[ "$eval_postprocess_visuals" == "true" || "$eval_postprocess_visuals" == "1" ]]; then
    echo "[postprocess] generating deferred overlays and 3D plots for ${name}"
    "$PYTHON_BIN" -m eval.postprocess_online_eval_visuals \
      --root "$eval_root/$name" \
      --output-name "$eval_target_crop_action_overlay_output_name" \
      $(if [[ "$eval_save_trajectory_3d" == "true" || "$eval_save_trajectory_3d" == "1" ]]; then echo --trajectory-3d; else echo --no-trajectory-3d; fi)
  fi
}

if [[ "$RUN_ONLINE_EVAL" == "true" ]]; then
  prepare_online_eval
fi

if [[ "$RUN_TEACHER_ABLATIONS" == "true" ]]; then
  for name in "${experiment_names[@]}"; do
    run_teacher "$name"
    if [[ "$RUN_ONLINE_EVAL" == "true" ]]; then
      evaluate_model_online "$name"
    fi
  done
else
  echo "[ablation] RUN_TEACHER_ABLATIONS=false, skip teacher training"
fi

if [[ "$RUN_ONLINE_EVAL" == "true" ]]; then
  for name in "${eval_extra_experiment_names[@]}"; do
    evaluate_model_online "$name"
  done
  summarize_eval_results "$eval_root" "held-out online eval summary (${eval_scene_list} ${eval_trajectory_range})" "${summary_models[@]}"
else
  echo "[ablation] RUN_ONLINE_EVAL=false, skip online eval"
fi

echo "[ablation] finished: $exp_root"
