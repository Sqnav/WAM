#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${DATASET_ROOT:-/data7/ysq/Worldmodel/Dataset}"
tracker_cache_root="${TARGET_HISTORY_TRACKER_CACHE_ROOT:-/data7/ysq/Worldmodel/experiments/tracker_artifacts/caches/square_tracker_cache_gt_bbox}"
output="${CAPTURE_ACTION_PRIOR_CHECKPOINT:-/data7/ysq/Worldmodel/experiments/fastwam_local_feature_ablation_run/models/capture_action_prior/best.pt}"
device="${CAPTURE_ACTION_PRIOR_DEVICE:-cuda}"
python_bin="${PYTHON_BIN:-/home/ysq/.conda/envs/ysq_qwen/bin/python}"
scene_list="${SCENE_LIST:-$(printf 'City_%s,' {1..27})}"
scene_list="${scene_list%,}"
trajectory_start="${TRAJECTORY_START:-1}"
trajectory_end="${TRAJECTORY_END:-450}"

cd "$repo_root"
PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m train.train_capture_action_prior \
  --dataset-root "$dataset_root" \
  --tracker-cache-root "$tracker_cache_root" \
  --output "$output" \
  --device "$device" \
  --scene-list "$scene_list" \
  --train-start "$trajectory_start" \
  --train-end "$trajectory_end" \
  "$@"
