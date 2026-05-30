#!/usr/bin/env bash
set -euo pipefail

# WAM visual-guidance ablation suite.
#
# Current project assumptions:
#   - no low-dimensional target vector is fed to the policy input;
#   - local crop has been removed;
#   - heatmap experiments use global image + patch-level target-projection bias;
#   - the prediction head only supervises next_target_relative;
#   - prediction-head rollout loss is disabled;
#   - RSSM imagination is only used at inference if DiT candidate selection is enabled.
#
# Main experiment table:
#   mlp_global   : global image + instruction, MLP actor
#   mlp_heatmap  : global image + instruction + attention heatmap, MLP actor
# Disabled by default while focusing on MLP visual-guidance comparison:
#   dit_global   : global image + instruction, DiT actor
#   dit_heatmap  : global image + instruction + attention heatmap, DiT actor
#
# Optional:
#   self_distill_dit_heatmap: student initialized/distilled from one teacher.
#
# Existing best.pt and summary.json are skipped by default. Override with:
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
EXP_NAME="${EXP_NAME:-visual_guidance_ablation}"
exp_root="${EXP_ROOT:-$root_dir/experiments/$EXP_NAME}"
log_dir="$exp_root/logs"
eval_root="${EVAL_OUTPUT_ROOT:-$exp_root/online_eval}"
eval_log_dir="${EVAL_LOG_DIR:-$exp_root/eval_logs}"
train_eval_root="${TRAIN_EVAL_OUTPUT_ROOT:-$exp_root/online_eval_train_1_10}"
train_eval_log_dir="${TRAIN_EVAL_LOG_DIR:-$exp_root/eval_logs_train_1_10}"

mkdir -p "$exp_root" "$log_dir" "$eval_root" "$eval_log_dir" "$train_eval_root" "$train_eval_log_dir"

# =========================
# Stages
# =========================
RUN_TEACHER_ABLATIONS="${RUN_TEACHER_ABLATIONS:-true}"
RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-true}"
RUN_TRAIN_ONLINE_EVAL="${RUN_TRAIN_ONLINE_EVAL:-true}"
RUN_SELF_DISTILL="${RUN_SELF_DISTILL:-false}"
RUN_DIT_CANDIDATE_BASELINE="${RUN_DIT_CANDIDATE_BASELINE:-false}"

EXPERIMENTS="${EXPERIMENTS:-mlp_global,mlp_heatmap}"
SELF_DISTILL_TEACHER="${SELF_DISTILL_TEACHER:-dit_heatmap}"

SKIP_EXISTING_TRAIN="${SKIP_EXISTING_TRAIN:-true}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-true}"
SKIP_EXISTING_SELF_DISTILL="${SKIP_EXISTING_SELF_DISTILL:-true}"

# =========================
# Dataset
# =========================
scene_list="${SCENE_LIST:-City_1,City_2,City_3}"
trajectory_range="${TRAJECTORY_RANGE:-1-450}"
val_ratio="${VAL_RATIO:-0.0}"
split_seed="${SPLIT_SEED:-42}"

# =========================
# Teacher training
# =========================
teacher_epochs="${TEACHER_EPOCHS:-50}"
teacher_batch_size="${TEACHER_BATCH_SIZE:-128}"
seq_len="${SEQ_LEN:-16}"
image_size="${IMAGE_SIZE:-224}"
num_workers="${NUM_WORKERS:-8}"
teacher_lr="${TEACHER_LR:-1e-4}"
teacher_weight_decay="${TEACHER_WEIGHT_DECAY:-1e-4}"

max_vel="${MAX_VEL:-1.0}"
max_yaw_rate="${MAX_YAW_RATE:-15.0}"
max_speed_norm="${MAX_SPEED_NORM:-1.0}"
action_sequence_horizon="${ACTION_SEQUENCE_HORIZON:-3}"
diffusion_steps="${DIFFUSION_STEPS:-20}"
sampling_steps="${SAMPLING_STEPS:-20}"

target_token_fusion_mode="${TARGET_TOKEN_FUSION_MODE:-concat}"
train_next_target_relative="${TRAIN_NEXT_TARGET_RELATIVE:-true}"
next_target_relative_loss_weight="${NEXT_TARGET_RELATIVE_LOSS_WEIGHT:-1.0}"
prior_target_relative_loss_weight="${PRIOR_TARGET_RELATIVE_LOSS_WEIGHT:-0.2}"
direct_action_loss_weight="${DIRECT_ACTION_LOSS_WEIGHT:-2.0}"
action_yaw_loss_weight="${ACTION_YAW_LOSS_WEIGHT:-10.0}"
x0_action_loss_weight="${X0_ACTION_LOSS_WEIGHT:-1.0}"

# Heatmap is only active for *_heatmap rows.
use_attention_heatmap="${USE_ATTENTION_HEATMAP:-true}"
visual_guidance_fov_deg="${VISUAL_GUIDANCE_FOV_DEG:-90.0}"
attention_heatmap_sigma="${ATTENTION_HEATMAP_SIGMA:-0.08}"

# =========================
# Self-distillation
# =========================
self_distill_epochs="${SELF_DISTILL_EPOCHS:-30}"
self_distill_batch_size="${SELF_DISTILL_BATCH_SIZE:-1}"
self_distill_lr="${SELF_DISTILL_LR:-5e-5}"
self_distill_weight_decay="${SELF_DISTILL_WEIGHT_DECAY:-1e-4}"
self_distill_sup_weight="${SELF_DISTILL_SUP_WEIGHT:-1.0}"
self_distill_feat_weight="${SELF_DISTILL_FEAT_WEIGHT:-0.1}"
self_distill_action_weight="${SELF_DISTILL_ACTION_WEIGHT:-0.5}"
self_distill_dit_noise_weight="${SELF_DISTILL_DIT_NOISE_WEIGHT:-0.5}"
self_distill_init_from_teacher="${SELF_DISTILL_INIT_FROM_TEACHER:-true}"

# =========================
# GPU
# =========================
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-2}"
EVAL_GPU_ID="${EVAL_GPU_ID:-3}"
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
eval_trajectory_range="${EVAL_TRAJECTORY_RANGE:-451-461}"
eval_max_trajectories="${EVAL_MAX_TRAJECTORIES:-0}"
eval_max_steps="${EVAL_MAX_STEPS:-0}"
train_eval_scene_list="${TRAIN_EVAL_SCENE_LIST:-City_1}"
train_eval_trajectory_range="${TRAIN_EVAL_TRAJECTORY_RANGE:-1-10}"
sim_server_host="${SIM_SERVER_HOST:-127.0.0.1}"
sim_server_port="${SIM_SERVER_PORT:-30000}"
scene_index="${SCENE_INDEX:-1}"
capture_distance="${CAPTURE_DISTANCE:-10.0}"
require_visibility_for_success="${REQUIRE_VISIBILITY_FOR_SUCCESS:-false}"
stop_on_collision="${STOP_ON_COLLISION:-true}"

# Default DiT eval measures the sampled DiT actor directly. Candidate selection
# can be enabled as a separate inference-time enhancement.
dit_candidate_selection="${DIT_CANDIDATE_SELECTION:-false}"
dit_candidate_count="${DIT_CANDIDATE_COUNT:-4}"
dit_candidate_lateral_weight="${DIT_CANDIDATE_LATERAL_WEIGHT:-1.0}"
dit_candidate_vertical_weight="${DIT_CANDIDATE_VERTICAL_WEIGHT:-1.0}"
dit_candidate_distance_weight="${DIT_CANDIDATE_DISTANCE_WEIGHT:-0.05}"
dit_candidate_smooth_weight="${DIT_CANDIDATE_SMOOTH_WEIGHT:-0.05}"
dit_candidate_yaw_angle_weight="${DIT_CANDIDATE_YAW_ANGLE_WEIGHT:-1.0}"
dit_candidate_pitch_angle_weight="${DIT_CANDIDATE_PITCH_ANGLE_WEIGHT:-0.7}"
dit_candidate_final_distance_weight="${DIT_CANDIDATE_FINAL_DISTANCE_WEIGHT:-0.25}"
dit_candidate_progress_weight="${DIT_CANDIDATE_PROGRESS_WEIGHT:-1.0}"
dit_candidate_front_weight="${DIT_CANDIDATE_FRONT_WEIGHT:-0.5}"
dit_candidate_action_weight="${DIT_CANDIDATE_ACTION_WEIGHT:-0.02}"
dit_candidate_temporal_smooth_weight="${DIT_CANDIDATE_TEMPORAL_SMOOTH_WEIGHT:-0.05}"

mlp_global_dir="$exp_root/mlp_global"
mlp_heatmap_dir="$exp_root/mlp_heatmap"
dit_global_dir="$exp_root/dit_global"
dit_heatmap_dir="$exp_root/dit_heatmap"
self_distill_name="self_distill_${SELF_DISTILL_TEACHER}"
self_distill_dir="$exp_root/$self_distill_name"

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
    mlp_global) echo "$mlp_global_dir" ;;
    mlp_heatmap) echo "$mlp_heatmap_dir" ;;
    dit_global) echo "$dit_global_dir" ;;
    dit_heatmap) echo "$dit_heatmap_dir" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'. Valid: mlp_global, mlp_heatmap, dit_global, dit_heatmap." >&2
      exit 1
      ;;
  esac
}

experiment_uses_diffusion() {
  case "$1" in
    dit_global|dit_heatmap) echo "true" ;;
    mlp_global|mlp_heatmap) echo "false" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'." >&2
      exit 1
      ;;
  esac
}

experiment_uses_guidance() {
  case "$1" in
    mlp_heatmap|dit_heatmap) echo "true" ;;
    mlp_global|dit_global) echo "false" ;;
    *)
      echo "[ERROR] Unknown experiment '$1'." >&2
      exit 1
      ;;
  esac
}

experiment_port() {
  case "$1" in
    mlp_global) echo 29621 ;;
    mlp_heatmap) echo 29622 ;;
    dit_global) echo 29623 ;;
    dit_heatmap) echo 29624 ;;
    *) echo 29629 ;;
  esac
}

write_manifest() {
  cat > "$exp_root/manifest.txt" <<EOF
experiment_name=$EXP_NAME
experiment_root=$exp_root
python=$PYTHON_BIN
conda_env=${CONDA_DEFAULT_ENV}

main_experiments=${EXPERIMENTS}
mlp_global=global image + instruction + MLP actor
mlp_heatmap=global image + instruction + patch attention heatmap + MLP actor
dit_global=global image + instruction + DiT actor
dit_heatmap=global image + instruction + patch attention heatmap + DiT actor

scene_list=${scene_list}
trajectory_range=${trajectory_range}
val_ratio=${val_ratio}
split_seed=${split_seed}
teacher_epochs=${teacher_epochs}
teacher_batch_size_per_gpu=${teacher_batch_size}
seq_len=${seq_len}
image_size=${image_size}
num_workers=${num_workers}
teacher_lr=${teacher_lr}
teacher_weight_decay=${teacher_weight_decay}
train_cuda_visible_devices=${TRAIN_GPU_IDS}
train_num_gpus=${train_num_gpus}

low_dim_target_input=off
local_crop=removed
prediction_heads=next_target_relative
prediction_head_rollout_loss=off
target_token_fusion_mode=${target_token_fusion_mode}
train_next_target_relative=${train_next_target_relative}
next_target_relative_loss_weight=${next_target_relative_loss_weight}
prior_target_relative_loss_weight=${prior_target_relative_loss_weight}
direct_action_loss_weight=${direct_action_loss_weight}
action_yaw_loss_weight=${action_yaw_loss_weight}
x0_action_loss_weight=${x0_action_loss_weight}
action_sequence_horizon=${action_sequence_horizon}
diffusion_steps=${diffusion_steps}
sampling_steps=${sampling_steps}
use_attention_heatmap=${use_attention_heatmap}
visual_guidance_fov_deg=${visual_guidance_fov_deg}
attention_heatmap_sigma=${attention_heatmap_sigma}

run_online_eval=${RUN_ONLINE_EVAL}
run_train_online_eval=${RUN_TRAIN_ONLINE_EVAL}
run_dit_candidate_baseline=${RUN_DIT_CANDIDATE_BASELINE}
eval_cuda_visible_devices=${EVAL_GPU_ID}
eval_scene_list=${eval_scene_list}
eval_trajectory_range=${eval_trajectory_range}
train_eval_scene_list=${train_eval_scene_list}
train_eval_trajectory_range=${train_eval_trajectory_range}
train_eval_output_root=${train_eval_root}
capture_distance=${capture_distance}
require_visibility_for_success=${require_visibility_for_success}
stop_on_collision=${stop_on_collision}
dit_candidate_selection=${dit_candidate_selection}
dit_candidate_count=${dit_candidate_count}
dit_candidate_score=tracking
dit_candidate_yaw_angle_weight=${dit_candidate_yaw_angle_weight}
dit_candidate_pitch_angle_weight=${dit_candidate_pitch_angle_weight}
dit_candidate_final_distance_weight=${dit_candidate_final_distance_weight}
dit_candidate_progress_weight=${dit_candidate_progress_weight}
dit_candidate_front_weight=${dit_candidate_front_weight}
dit_candidate_smooth_weight=${dit_candidate_smooth_weight}
dit_candidate_temporal_smooth_weight=${dit_candidate_temporal_smooth_weight}
dit_candidate_action_weight=${dit_candidate_action_weight}

run_self_distill=${RUN_SELF_DISTILL}
self_distill_teacher=${SELF_DISTILL_TEACHER}
self_distill_dir=${self_distill_dir}
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
  local master_port
  master_port="$(experiment_port "$name")"
  local log_file="$log_dir/${name}.log"

  if [[ "$SKIP_EXISTING_TRAIN" == "true" && -f "$save_dir/best.pt" ]]; then
    echo "[train-skip] ${name}: $save_dir/best.pt exists"
    return 0
  fi

  mkdir -p "$save_dir"
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"

  echo "============================================================" | tee "$log_file"
  echo "[train] ${name}" | tee -a "$log_file"
  echo "save_dir=${save_dir}" | tee -a "$log_file"
  echo "use_diffusion_actor=${use_diffusion_actor}" | tee -a "$log_file"
  echo "use_target_visual_guidance=${use_target_visual_guidance}" | tee -a "$log_file"
  echo "low_dim_target_input=off" | tee -a "$log_file"
  echo "prediction_head_rollout_loss=off" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$log_file"
  echo "nproc_per_node=${train_num_gpus}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$train_num_gpus" \
    --master_port "$master_port" \
    -m train.train_teacher \
    --dataset-root "$dataset_root" \
    --scene-list "$scene_list" \
    --trajectory-range "$trajectory_range" \
    --save-dir "$save_dir" \
    --epochs "$teacher_epochs" \
    --batch-size "$teacher_batch_size" \
    --seq-len "$seq_len" \
    --image-size "$image_size" \
    --val-ratio "$val_ratio" \
    --split-seed "$split_seed" \
    --lr "$teacher_lr" \
    --weight-decay "$teacher_weight_decay" \
    --max-vel "$max_vel" \
    --max-yaw-rate "$max_yaw_rate" \
    --max-speed-norm "$max_speed_norm" \
    --action-sequence-horizon "$action_sequence_horizon" \
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
    --visual-guidance-fov-deg "$visual_guidance_fov_deg" \
    --attention-heatmap-sigma "$attention_heatmap_sigma" \
    --target-token-fusion-mode "$target_token_fusion_mode" \
    --use-diffusion-actor "$use_diffusion_actor" \
    --multi-gpu \
    2>&1 | tee -a "$log_file"
}

run_self_distill() {
  local teacher_name="$1"
  local teacher_dir
  teacher_dir="$(experiment_dir "$teacher_name")"
  local teacher_ckpt="$teacher_dir/best.pt"
  local use_target_visual_guidance
  use_target_visual_guidance="$(experiment_uses_guidance "$teacher_name")"
  local log_file="$log_dir/${self_distill_name}.log"

  if [[ ! -f "$teacher_ckpt" ]]; then
    echo "[ERROR] Missing self-distill teacher checkpoint: $teacher_ckpt" >&2
    exit 1
  fi
  if [[ "$SKIP_EXISTING_SELF_DISTILL" == "true" && -f "$self_distill_dir/best.pt" ]]; then
    echo "[self-distill-skip] ${self_distill_name}: $self_distill_dir/best.pt exists"
    return 0
  fi

  mkdir -p "$self_distill_dir"
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"

  init_args=()
  if [[ "$self_distill_init_from_teacher" == "true" || "$self_distill_init_from_teacher" == "1" ]]; then
    init_args+=(--init-student-from-teacher)
  else
    init_args+=(--student-init-random)
  fi

  echo "============================================================" | tee "$log_file"
  echo "[self-distill] ${self_distill_name}" | tee -a "$log_file"
  echo "teacher=${teacher_name}" | tee -a "$log_file"
  echo "teacher_ckpt=${teacher_ckpt}" | tee -a "$log_file"
  echo "save_dir=${self_distill_dir}" | tee -a "$log_file"
  echo "use_target_visual_guidance=${use_target_visual_guidance}" | tee -a "$log_file"
  echo "weights=sup:${self_distill_sup_weight},feat:${self_distill_feat_weight},action:${self_distill_action_weight},dit_noise:${self_distill_dit_noise_weight}" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$train_num_gpus" \
    --master_port 29631 \
    -m train.train_self_distill \
    --dataset-root "$dataset_root" \
    --scene-list "$scene_list" \
    --trajectory-range "$trajectory_range" \
    --val-ratio "$val_ratio" \
    --split-seed "$split_seed" \
    --teacher-ckpt "$teacher_ckpt" \
    --save-dir "$self_distill_dir" \
    --image-size "$image_size" \
    --seq-len "$seq_len" \
    --max-vel "$max_vel" \
    --max-yaw-rate "$max_yaw_rate" \
    --max-speed-norm "$max_speed_norm" \
    --action-sequence-horizon "$action_sequence_horizon" \
    --target-token-fusion-mode "$target_token_fusion_mode" \
    --use-target-visual-guidance "$use_target_visual_guidance" \
    --use-attention-heatmap "$use_attention_heatmap" \
    --visual-guidance-fov-deg "$visual_guidance_fov_deg" \
    --attention-heatmap-sigma "$attention_heatmap_sigma" \
    --diffusion-steps "$diffusion_steps" \
    --sampling-steps "$sampling_steps" \
    --freeze-dinov2 \
    --batch-size "$self_distill_batch_size" \
    --epochs "$self_distill_epochs" \
    --lr "$self_distill_lr" \
    --weight-decay "$self_distill_weight_decay" \
    --num-workers "$num_workers" \
    --sup-weight "$self_distill_sup_weight" \
    --feat-distill-weight "$self_distill_feat_weight" \
    --action-distill-weight "$self_distill_action_weight" \
    --dit-noise-distill-weight "$self_distill_dit_noise_weight" \
    --multi-gpu \
    "${init_args[@]}" \
    2>&1 | tee -a "$log_file"
}

run_online_eval() {
  local name="$1"
  local ckpt="$2"
  local use_diffusion_actor="$3"
  local use_target_visual_guidance="$4"
  local use_candidate_selection="${5:-$dit_candidate_selection}"
  local out_root="${6:-$eval_root}"
  local log_root="${7:-$eval_log_dir}"
  local scene_list_for_eval="${8:-$eval_scene_list}"
  local trajectory_range_for_eval="${9:-$eval_trajectory_range}"
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
  echo "dit_candidate_selection=${use_candidate_selection}" | tee -a "$log_file"
  echo "dit_candidate_score=tracking" | tee -a "$log_file"
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
    --max-yaw-rate "$max_yaw_rate" \
    --max-speed-norm "$max_speed_norm" \
    --capture-distance "$capture_distance" \
    --dit-candidate-selection "$use_candidate_selection" \
    --dit-candidate-count "$dit_candidate_count" \
    --dit-candidate-lateral-weight "$dit_candidate_lateral_weight" \
    --dit-candidate-vertical-weight "$dit_candidate_vertical_weight" \
    --dit-candidate-distance-weight "$dit_candidate_distance_weight" \
    --dit-candidate-smooth-weight "$dit_candidate_smooth_weight" \
    --dit-candidate-yaw-angle-weight "$dit_candidate_yaw_angle_weight" \
    --dit-candidate-pitch-angle-weight "$dit_candidate_pitch_angle_weight" \
    --dit-candidate-final-distance-weight "$dit_candidate_final_distance_weight" \
    --dit-candidate-progress-weight "$dit_candidate_progress_weight" \
    --dit-candidate-front-weight "$dit_candidate_front_weight" \
    --dit-candidate-action-weight "$dit_candidate_action_weight" \
    --dit-candidate-temporal-smooth-weight "$dit_candidate_temporal_smooth_weight" \
    --use-target-visual-guidance "$use_target_visual_guidance" \
    --use-attention-heatmap "$use_attention_heatmap" \
    --visual-guidance-fov-deg "$visual_guidance_fov_deg" \
    --attention-heatmap-sigma "$attention_heatmap_sigma" \
    --use-diffusion-actor "$use_diffusion_actor" \
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

echo "[ablation] experiment root: $exp_root"
echo "[ablation] experiments: ${experiment_names[*]}"
echo "[ablation] train GPUs: $TRAIN_GPU_IDS (nproc_per_node=$train_num_gpus)"
echo "[ablation] eval GPU: $EVAL_GPU_ID"
echo "[ablation] teacher_epochs: $teacher_epochs"
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
  run_self_distill "$SELF_DISTILL_TEACHER"
  summary_models+=("$self_distill_name")
else
  echo "[ablation] RUN_SELF_DISTILL=false, skip self-distillation"
fi

if [[ "$RUN_ONLINE_EVAL" == "true" ]]; then
  echo "[ablation] starting online eval"
  for name in "${experiment_names[@]}"; do
    ckpt="$(experiment_dir "$name")/best.pt"
    run_online_eval "$name" "$ckpt" "$(experiment_uses_diffusion "$name")" "$(experiment_uses_guidance "$name")"
  done
  if [[ "$RUN_DIT_CANDIDATE_BASELINE" == "true" ]]; then
    if [[ "$dit_candidate_selection" == "true" || "$dit_candidate_selection" == "1" ]]; then
      echo "[ablation] DIT_CANDIDATE_SELECTION is already enabled for main DiT eval; skip duplicate *_candidate baseline"
    else
      for name in "${experiment_names[@]}"; do
        if [[ "$(experiment_uses_diffusion "$name")" == "true" ]]; then
          ckpt="$(experiment_dir "$name")/best.pt"
          candidate_name="${name}_candidate"
          run_online_eval "$candidate_name" "$ckpt" true "$(experiment_uses_guidance "$name")" true
          summary_models+=("$candidate_name")
        fi
      done
    fi
  fi
  if [[ "$RUN_SELF_DISTILL" == "true" || -f "$self_distill_dir/best.pt" ]]; then
    run_online_eval \
      "$self_distill_name" \
      "$self_distill_dir/best.pt" \
      "$(experiment_uses_diffusion "$SELF_DISTILL_TEACHER")" \
      "$(experiment_uses_guidance "$SELF_DISTILL_TEACHER")"
    if [[ "$RUN_SELF_DISTILL" != "true" ]]; then
      summary_models+=("$self_distill_name")
    fi
  fi
  summarize_eval_results "$eval_root" "held-out online eval summary (${eval_scene_list} ${eval_trajectory_range})" "${summary_models[@]}"

  if [[ "$RUN_TRAIN_ONLINE_EVAL" == "true" ]]; then
    echo "[ablation] starting train-trajectory online eval"
    train_summary_models=("${experiment_names[@]}")
    for name in "${experiment_names[@]}"; do
      ckpt="$(experiment_dir "$name")/best.pt"
      run_online_eval \
        "$name" \
        "$ckpt" \
        "$(experiment_uses_diffusion "$name")" \
        "$(experiment_uses_guidance "$name")" \
        "$dit_candidate_selection" \
        "$train_eval_root" \
        "$train_eval_log_dir" \
        "$train_eval_scene_list" \
        "$train_eval_trajectory_range"
    done
    if [[ "$RUN_DIT_CANDIDATE_BASELINE" == "true" ]]; then
      if [[ "$dit_candidate_selection" == "true" || "$dit_candidate_selection" == "1" ]]; then
        echo "[ablation] DIT_CANDIDATE_SELECTION is already enabled for train DiT eval; skip duplicate *_candidate baseline"
      else
        for name in "${experiment_names[@]}"; do
          if [[ "$(experiment_uses_diffusion "$name")" == "true" ]]; then
            ckpt="$(experiment_dir "$name")/best.pt"
            candidate_name="${name}_candidate"
            run_online_eval \
              "$candidate_name" \
              "$ckpt" \
              true \
              "$(experiment_uses_guidance "$name")" \
              true \
              "$train_eval_root" \
              "$train_eval_log_dir" \
              "$train_eval_scene_list" \
              "$train_eval_trajectory_range"
            train_summary_models+=("$candidate_name")
          fi
        done
      fi
    fi
    if [[ "$RUN_SELF_DISTILL" == "true" || -f "$self_distill_dir/best.pt" ]]; then
      run_online_eval \
        "$self_distill_name" \
        "$self_distill_dir/best.pt" \
        "$(experiment_uses_diffusion "$SELF_DISTILL_TEACHER")" \
        "$(experiment_uses_guidance "$SELF_DISTILL_TEACHER")" \
        "$dit_candidate_selection" \
        "$train_eval_root" \
        "$train_eval_log_dir" \
        "$train_eval_scene_list" \
        "$train_eval_trajectory_range"
      train_summary_models+=("$self_distill_name")
    fi
    summarize_eval_results "$train_eval_root" "train online eval summary (${train_eval_scene_list} ${train_eval_trajectory_range})" "${train_summary_models[@]}"
  else
    echo "[ablation] RUN_TRAIN_ONLINE_EVAL=false, skip train-trajectory online eval"
  fi
else
  echo "[ablation] RUN_ONLINE_EVAL=false, skip online eval"
fi

echo "[ablation] finished: $exp_root"
