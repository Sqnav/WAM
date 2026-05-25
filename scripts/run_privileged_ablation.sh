#!/usr/bin/env bash
set -euo pipefail

# No-privileged-input ablation after removing local crop.
#
# This script keeps the existing no-visual global baselines and adds heatmap-only
# visual guidance runs:
#   1) no_priv_mlp_global    : no privileged input, global image only, MLP actor
#   2) no_priv_mlp_heatmap   : no privileged input, global image + heatmap token, MLP actor
#   3) no_priv_dit_global    : no privileged input, global image only, DiT actor
#   4) no_priv_dit_heatmap   : no privileged input, global image + heatmap token, DiT actor
#
# Existing checkpoints/eval summaries are skipped by default. Set
# SKIP_EXISTING_TRAIN=false or SKIP_EXISTING_EVAL=false to overwrite/re-run.

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
# Named experiment directory
# =========================
EXP_NAME="${EXP_NAME:-no_priv_visual_guidance_ablation}"
exp_root="${EXP_ROOT:-$root_dir/experiments/$EXP_NAME}"
log_dir="$exp_root/logs"
eval_root="$exp_root/online_eval"
eval_log_dir="$exp_root/eval_logs"

SKIP_EXISTING_TRAIN="${SKIP_EXISTING_TRAIN:-true}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-true}"
SKIP_EXISTING_DIT_TRAIN="${SKIP_EXISTING_DIT_TRAIN:-false}"
SKIP_EXISTING_DIT_EVAL="${SKIP_EXISTING_DIT_EVAL:-false}"

mkdir -p "$exp_root" "$log_dir" "$eval_root" "$eval_log_dir"

# =========================
# Dataset / training config
# =========================
scene_list="${SCENE_LIST:-City_1,City_2,City_3}"
trajectory_range="${TRAJECTORY_RANGE:-1-450}"
val_ratio="${VAL_RATIO:-0.0}"
split_seed="${SPLIT_SEED:-42}"

teacher_epochs="${TEACHER_EPOCHS:-100}"
teacher_batch_size="${TEACHER_BATCH_SIZE:-4}"
seq_len="${SEQ_LEN:-16}"
num_workers="${NUM_WORKERS:-8}"
train_num_gpus="${TRAIN_NUM_GPUS:-}"

max_vel="${MAX_VEL:-1.0}"
max_yaw_rate="${MAX_YAW_RATE:-15.0}"
max_speed_norm="${MAX_SPEED_NORM:-1.0}"
action_sequence_horizon="${ACTION_SEQUENCE_HORIZON:-3}"
privileged_fusion_mode="${PRIVILEGED_FUSION_MODE:-concat}"

next_privileged_loss_weight="${NEXT_PRIVILEGED_LOSS_WEIGHT:-1.0}"
prior_privileged_loss_weight="${PRIOR_PRIVILEGED_LOSS_WEIGHT:-0.2}"
direct_action_loss_weight="${DIRECT_ACTION_LOSS_WEIGHT:-2.0}"
action_yaw_loss_weight="${ACTION_YAW_LOSS_WEIGHT:-10.0}"
rollout_loss_weight="${ROLLOUT_LOSS_WEIGHT:-0.2}"
rollout_horizon="${ROLLOUT_HORIZON:-3}"

# Heatmap-only visual guidance internals used by the *_heatmap experiments.
use_attention_heatmap="${USE_ATTENTION_HEATMAP:-true}"
visual_guidance_fov_deg="${VISUAL_GUIDANCE_FOV_DEG:-90.0}"
attention_heatmap_sigma="${ATTENTION_HEATMAP_SIGMA:-0.08}"

# =========================
# GPU / online eval config
# =========================
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-2,3}"
EVAL_GPU_ID="${EVAL_GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"
if [[ -z "$train_num_gpus" ]]; then
  train_num_gpus="$("$PYTHON_BIN" - <<'PY'
import os
ids = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(max(len(ids), 1))
PY
)"
fi

RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-true}"
eval_scene_list="${EVAL_SCENE_LIST:-City_1}"
eval_trajectory_range="${EVAL_TRAJECTORY_RANGE:-451-500}"
eval_max_trajectories="${EVAL_MAX_TRAJECTORIES:-0}"
eval_max_steps="${EVAL_MAX_STEPS:-0}"
sim_server_host="${SIM_SERVER_HOST:-127.0.0.1}"
sim_server_port="${SIM_SERVER_PORT:-30000}"
scene_index="${SCENE_INDEX:-1}"

# Candidate selection was harmful for the previous DiT eval. Keep deterministic
# DiT sampling by default; override explicitly if you want candidate selection.
dit_candidate_selection="${DIT_CANDIDATE_SELECTION:-false}"
dit_candidate_count="${DIT_CANDIDATE_COUNT:-4}"
dit_candidate_lateral_weight="${DIT_CANDIDATE_LATERAL_WEIGHT:-1.0}"
dit_candidate_vertical_weight="${DIT_CANDIDATE_VERTICAL_WEIGHT:-1.0}"
dit_candidate_distance_weight="${DIT_CANDIDATE_DISTANCE_WEIGHT:-0.05}"
dit_candidate_smooth_weight="${DIT_CANDIDATE_SMOOTH_WEIGHT:-0.05}"

no_priv_mlp_global_dir="$exp_root/no_priv_mlp_global"
no_priv_mlp_heatmap_dir="$exp_root/no_priv_mlp_heatmap"
no_priv_dit_global_dir="$exp_root/no_priv_dit_global"
no_priv_dit_heatmap_dir="$exp_root/no_priv_dit_heatmap"

write_manifest() {
  cat > "$exp_root/manifest.txt" <<EOF
experiment_name=$EXP_NAME
experiment_root=$exp_root
conda_env=${CONDA_DEFAULT_ENV}
python=$PYTHON_BIN
train_cuda_visible_devices=${TRAIN_GPU_IDS}
eval_cuda_visible_devices=${EVAL_GPU_ID}
scene_list=${scene_list}
trajectory_range=${trajectory_range}
val_ratio=${val_ratio}
teacher_epochs=${teacher_epochs}
teacher_batch_size=${teacher_batch_size}
seq_len=${seq_len}
max_yaw_rate=${max_yaw_rate}
max_speed_norm=${max_speed_norm}
action_sequence_horizon=${action_sequence_horizon}
privileged_input=disabled
privileged_fusion_mode=${privileged_fusion_mode}
use_attention_heatmap=${use_attention_heatmap}
visual_guidance_fov_deg=${visual_guidance_fov_deg}
attention_heatmap_sigma=${attention_heatmap_sigma}
run_online_eval=${RUN_ONLINE_EVAL}
eval_scene_list=${eval_scene_list}
eval_trajectory_range=${eval_trajectory_range}
dit_candidate_selection=${dit_candidate_selection}
skip_existing_train=${SKIP_EXISTING_TRAIN}
skip_existing_eval=${SKIP_EXISTING_EVAL}
skip_existing_dit_train=${SKIP_EXISTING_DIT_TRAIN}
skip_existing_dit_eval=${SKIP_EXISTING_DIT_EVAL}
no_priv_mlp_global_dir=${no_priv_mlp_global_dir}
no_priv_mlp_heatmap_dir=${no_priv_mlp_heatmap_dir}
no_priv_dit_global_dir=${no_priv_dit_global_dir}
no_priv_dit_heatmap_dir=${no_priv_dit_heatmap_dir}
note=Local crop has been removed. Heatmap experiments use global image plus a projected [clamped_x, clamped_y, visible] heatmap token.
EOF
}

run_teacher() {
  local name="$1"
  local use_diffusion_actor="$2"
  local use_target_visual_guidance="$3"
  local save_dir="$4"
  local master_port="$5"
  local log_file="$log_dir/${name}.log"
  local skip_existing="$SKIP_EXISTING_TRAIN"
  if [[ "$use_diffusion_actor" == "true" ]]; then
    skip_existing="$SKIP_EXISTING_DIT_TRAIN"
  fi

  if [[ "$skip_existing" == "true" && -f "$save_dir/best.pt" ]]; then
    echo "[train-skip] ${name}: existing checkpoint $save_dir/best.pt"
    return 0
  fi

  mkdir -p "$save_dir"
  echo "============================================================" | tee "$log_file"
  echo "[train] ${name}" | tee -a "$log_file"
  echo "save_dir=${save_dir}" | tee -a "$log_file"
  echo "privileged_input=disabled" | tee -a "$log_file"
  echo "use_diffusion_actor=${use_diffusion_actor}" | tee -a "$log_file"
  echo "use_target_visual_guidance=${use_target_visual_guidance}" | tee -a "$log_file"
  echo "use_attention_heatmap=${use_attention_heatmap}" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$log_file"
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
    --val-ratio "$val_ratio" \
    --split-seed "$split_seed" \
    --max-vel "$max_vel" \
    --max-yaw-rate "$max_yaw_rate" \
    --max-speed-norm "$max_speed_norm" \
    --action-sequence-horizon "$action_sequence_horizon" \
    --num-workers "$num_workers" \
    --train-next-privileged true \
    --train-rollout true \
    --next-privileged-loss-weight "$next_privileged_loss_weight" \
    --prior-privileged-loss-weight "$prior_privileged_loss_weight" \
    --direct-action-loss-weight "$direct_action_loss_weight" \
    --action-yaw-loss-weight "$action_yaw_loss_weight" \
    --rollout-loss-weight "$rollout_loss_weight" \
    --rollout-horizon "$rollout_horizon" \
    --use-target-visual-guidance "$use_target_visual_guidance" \
    --use-attention-heatmap "$use_attention_heatmap" \
    --visual-guidance-fov-deg "$visual_guidance_fov_deg" \
    --attention-heatmap-sigma "$attention_heatmap_sigma" \
    --multi-gpu \
    --privileged-fusion-mode "$privileged_fusion_mode" \
    --use-diffusion-actor "$use_diffusion_actor" \
    2>&1 | tee -a "$log_file"
}

run_online_eval() {
  local name="$1"
  local ckpt="$2"
  local use_diffusion_actor="$3"
  local use_target_visual_guidance="$4"
  local out_dir="$eval_root/$name"
  local log_file="$eval_log_dir/${name}.log"
  local skip_existing="$SKIP_EXISTING_EVAL"
  if [[ "$use_diffusion_actor" == "true" ]]; then
    skip_existing="$SKIP_EXISTING_DIT_EVAL"
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] Missing checkpoint for eval: $ckpt" >&2
    exit 1
  fi
  if [[ "$skip_existing" == "true" && -f "$out_dir/summary.json" ]]; then
    echo "[eval-skip] ${name}: existing summary $out_dir/summary.json"
    return 0
  fi

  mkdir -p "$out_dir"
  export CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID"
  export DAGGER_MULTI_WORKER=1

  extra_eval_args=()
  if [[ "$eval_max_trajectories" != "0" ]]; then
    extra_eval_args+=(--max-trajectories "$eval_max_trajectories")
  fi
  if [[ "$eval_max_steps" != "0" ]]; then
    extra_eval_args+=(--max-steps "$eval_max_steps")
  fi

  echo "============================================================" | tee "$log_file"
  echo "[online-eval] ${name}" | tee -a "$log_file"
  echo "checkpoint=${ckpt}" | tee -a "$log_file"
  echo "output=${out_dir}" | tee -a "$log_file"
  echo "privileged_input=disabled" | tee -a "$log_file"
  echo "use_diffusion_actor=${use_diffusion_actor}" | tee -a "$log_file"
  echo "use_target_visual_guidance=${use_target_visual_guidance}" | tee -a "$log_file"
  echo "dit_candidate_selection=${dit_candidate_selection}" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  "$PYTHON_BIN" -m eval.online_eval_teacher \
    --dataset-root "$dataset_root" \
    --checkpoint "$ckpt" \
    --output-dir "$out_dir" \
    --executor-script "$executor_script" \
    --scene-list "$eval_scene_list" \
    --trajectory-range "$eval_trajectory_range" \
    --eval-split all \
    --sim-server-host "$sim_server_host" \
    --sim-server-port "$sim_server_port" \
    --scene-index "$scene_index" \
    --gpu-id "$EVAL_GPU_ID" \
    --device cuda \
    --max-yaw-rate "$max_yaw_rate" \
    --max-speed-norm "$max_speed_norm" \
    --dit-candidate-selection "$dit_candidate_selection" \
    --dit-candidate-count "$dit_candidate_count" \
    --dit-candidate-lateral-weight "$dit_candidate_lateral_weight" \
    --dit-candidate-vertical-weight "$dit_candidate_vertical_weight" \
    --dit-candidate-distance-weight "$dit_candidate_distance_weight" \
    --dit-candidate-smooth-weight "$dit_candidate_smooth_weight" \
    --use-target-visual-guidance "$use_target_visual_guidance" \
    --use-attention-heatmap "$use_attention_heatmap" \
    --visual-guidance-fov-deg "$visual_guidance_fov_deg" \
    --attention-heatmap-sigma "$attention_heatmap_sigma" \
    --use-diffusion-actor "$use_diffusion_actor" \
    "${extra_eval_args[@]}" \
    2>&1 | tee -a "$log_file"
}

summarize_eval_results() {
  "$PYTHON_BIN" - "$eval_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = [
    "no_priv_mlp_global",
    "no_priv_mlp_heatmap",
    "no_priv_dit_global",
    "no_priv_dit_heatmap",
]

print("\n[ablation] online eval summary")
print(
    f"{'model':24s} {'SR':>7s} {'ATF':>8s} {'track%':>8s} "
    f"{'coll%':>8s} {'final_d':>9s} {'min_d':>8s} {'failures':>24s}"
)
for model in models:
    path = root / model / "summary.json"
    if not path.exists():
        print(f"{model:24s} {'missing':>7s}")
        continue
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    sr = data.get("SR", data.get("success_rate"))
    atf = data.get("ATF", data.get("average_tracked_frames"))
    track = data.get("mean_effective_tracking_ratio")
    coll = data.get("collision_rate")
    final_d = data.get("mean_final_distance")
    min_d = data.get("mean_min_distance")
    failures = data.get("failure_reason_counts", {})
    failures_s = ",".join(f"{k}:{v}" for k, v in sorted(failures.items()))
    print(
        f"{model:24s} "
        f"{(sr * 100 if sr is not None else float('nan')):6.2f}% "
        f"{(atf if atf is not None else float('nan')):8.2f} "
        f"{(track * 100 if track is not None else float('nan')):7.2f}% "
        f"{(coll * 100 if coll is not None else float('nan')):7.2f}% "
        f"{(final_d if final_d is not None else float('nan')):9.2f} "
        f"{(min_d if min_d is not None else float('nan')):8.2f} "
        f"{failures_s:>24s}"
    )
PY
}

write_manifest
echo "[ablation] experiment name: $EXP_NAME"
echo "[ablation] experiment root: $exp_root"
echo "[ablation] logs: $log_dir"
echo "[ablation] skip existing train: $SKIP_EXISTING_TRAIN"
echo "[ablation] skip existing eval : $SKIP_EXISTING_EVAL"
echo "[ablation] skip existing DiT train: $SKIP_EXISTING_DIT_TRAIN"
echo "[ablation] skip existing DiT eval : $SKIP_EXISTING_DIT_EVAL"
echo "[ablation] all training uses no privileged vector input"

run_teacher "no_priv_mlp_global" false false "$no_priv_mlp_global_dir" 29621
run_teacher "no_priv_mlp_heatmap" false true "$no_priv_mlp_heatmap_dir" 29622
run_teacher "no_priv_dit_global" true false "$no_priv_dit_global_dir" 29623
run_teacher "no_priv_dit_heatmap" true true "$no_priv_dit_heatmap_dir" 29624

if [[ "$RUN_ONLINE_EVAL" == "true" ]]; then
  echo "[ablation] training finished; starting online eval"
  run_online_eval "no_priv_mlp_global" "$no_priv_mlp_global_dir/best.pt" false false
  run_online_eval "no_priv_mlp_heatmap" "$no_priv_mlp_heatmap_dir/best.pt" false true
  run_online_eval "no_priv_dit_global" "$no_priv_dit_global_dir/best.pt" true false
  run_online_eval "no_priv_dit_heatmap" "$no_priv_dit_heatmap_dir/best.pt" true true
  summarize_eval_results
else
  echo "[ablation] RUN_ONLINE_EVAL=false, skip online eval"
fi

echo "[ablation] finished: $exp_root"
