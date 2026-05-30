#!/usr/bin/env bash
set -euo pipefail

# Online-evaluate all visual_guidance_ablation teacher variants.
# Defaults are intentionally small for quick sanity checks:
#   EVAL_SCENE_LIST=City_1
#   EVAL_TRAJECTORY_RANGE=1-10
#
# Results are written to online_eval_1_10 by default so previous eval results
# under online_eval are not overwritten.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
dataset_root="$root_dir/Dataset"
executor_script="$root_dir/code/src/executor/trajectory_executor.py"

if command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-ysq_qwen}"
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi

export PYTHONPATH="$root_dir/code/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export DAGGER_MULTI_WORKER=1

EXP_NAME="${EXP_NAME:-visual_guidance_ablation}"
exp_root="${EXP_ROOT:-$root_dir/experiments/$EXP_NAME}"
eval_root="${EVAL_OUTPUT_ROOT:-$exp_root/online_eval_1_10}"
eval_log_dir="${EVAL_LOG_DIR:-$exp_root/eval_logs_1_10}"

eval_scene_list="${EVAL_SCENE_LIST:-City_1}"
eval_trajectory_range="${EVAL_TRAJECTORY_RANGE:-1-10}"
eval_max_trajectories="${EVAL_MAX_TRAJECTORIES:-0}"
eval_max_steps="${EVAL_MAX_STEPS:-0}"

EVAL_GPU_ID="${EVAL_GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID"

sim_server_host="${SIM_SERVER_HOST:-127.0.0.1}"
sim_server_port="${SIM_SERVER_PORT:-30000}"
scene_index="${SCENE_INDEX:-1}"

image_size="${IMAGE_SIZE:-224}"
max_yaw_rate="${MAX_YAW_RATE:-15.0}"
max_speed_norm="${MAX_SPEED_NORM:-1.0}"
sampling_steps="${SAMPLING_STEPS:-20}"

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

use_attention_heatmap="${USE_ATTENTION_HEATMAP:-true}"
visual_guidance_fov_deg="${VISUAL_GUIDANCE_FOV_DEG:-90.0}"
attention_heatmap_sigma="${ATTENTION_HEATMAP_SIGMA:-0.08}"

mkdir -p "$eval_root" "$eval_log_dir"

extra_eval_args=()
if [[ "$eval_max_trajectories" != "0" ]]; then
  extra_eval_args+=(--max-trajectories "$eval_max_trajectories")
fi
if [[ "$eval_max_steps" != "0" ]]; then
  extra_eval_args+=(--max-steps "$eval_max_steps")
fi

run_one() {
  local name="$1"
  local use_diffusion_actor="$2"
  local use_target_visual_guidance="$3"
  local ckpt_name="${4:-$name}"
  local candidate_selection="${5:-$dit_candidate_selection}"
  local ckpt="$exp_root/$ckpt_name/best.pt"
  local out_dir="$eval_root/$name"
  local log_file="$eval_log_dir/${name}.log"

  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] Missing checkpoint: $ckpt" >&2
    exit 1
  fi

  mkdir -p "$out_dir"
  echo "============================================================" | tee "$log_file"
  echo "[online-eval] $name" | tee -a "$log_file"
  echo "checkpoint=$ckpt" | tee -a "$log_file"
  echo "output=$out_dir" | tee -a "$log_file"
  echo "scene_list=$eval_scene_list trajectory_range=$eval_trajectory_range" | tee -a "$log_file"
  echo "use_diffusion_actor=$use_diffusion_actor" | tee -a "$log_file"
  echo "use_target_visual_guidance=$use_target_visual_guidance" | tee -a "$log_file"
  echo "dit_candidate_selection=$candidate_selection" | tee -a "$log_file"
  echo "dit_candidate_score=tracking" | tee -a "$log_file"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$log_file"
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
    --image-size "$image_size" \
    --sampling-steps "$sampling_steps" \
    --max-yaw-rate "$max_yaw_rate" \
    --max-speed-norm "$max_speed_norm" \
    --dit-candidate-selection "$candidate_selection" \
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

summarize() {
  "$PYTHON_BIN" - "$eval_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = [
    "mlp_global",
    "mlp_heatmap",
    "dit_global",
    "dit_heatmap",
    "dit_global_candidate",
    "dit_heatmap_candidate",
]
print("\n[eval_1_10] online eval summary")
print(
    f"{'model':14s} {'SR':>7s} {'ATF':>8s} {'track%':>8s} "
    f"{'close%':>8s} {'coll%':>8s} {'final_d':>9s} {'mean_d':>9s} {'failures':>24s}"
)
for model in models:
    path = root / model / "summary.json"
    if not path.exists():
        print(f"{model:14s} {'missing':>7s}")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    sr = data.get("SR", data.get("success_rate"))
    atf = data.get("ATF", data.get("average_tracked_frames"))
    track = data.get("mean_effective_tracking_ratio")
    close = data.get("mean_close_frame_ratio")
    coll = data.get("collision_rate")
    final_d = data.get("mean_final_distance")
    mean_d = data.get("mean_distance")
    failures = data.get("failure_reason_counts", {})
    failures_s = ",".join(f"{k}:{v}" for k, v in sorted(failures.items()))
    print(
        f"{model:14s} "
        f"{(sr * 100 if sr is not None else float('nan')):6.2f}% "
        f"{(atf if atf is not None else float('nan')):8.2f} "
        f"{(track * 100 if track is not None else float('nan')):7.2f}% "
        f"{(close * 100 if close is not None else float('nan')):7.2f}% "
        f"{(coll * 100 if coll is not None else float('nan')):7.2f}% "
        f"{(final_d if final_d is not None else float('nan')):9.2f} "
        f"{(mean_d if mean_d is not None else float('nan')):9.2f} "
        f"{failures_s:>24s}"
    )
PY
}

echo "[eval_1_10] exp_root=$exp_root"
echo "[eval_1_10] eval_root=$eval_root"
echo "[eval_1_10] scenes=$eval_scene_list range=$eval_trajectory_range"

run_one mlp_global false false
run_one mlp_heatmap false true
run_one dit_global true false
run_one dit_heatmap true true
run_one dit_global_candidate true false dit_global true
run_one dit_heatmap_candidate true true dit_heatmap true
summarize

echo "[eval_1_10] finished: $eval_root"
