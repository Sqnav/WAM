#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exp_root="${EXP_ROOT:-$root_dir/experiments/fastwam_local_feature_ablation_run}"
log_file="$exp_root/logs/public_baselines_pipeline.log"
mkdir -p "$exp_root/logs"
exec > >(tee -a "$log_file") 2>&1

echo "[public-pipeline] waiting for current run_ablation lock: $(date --iso-8601=seconds)"
lock_path="$exp_root/.run_ablation.lock"
exec {pipeline_lock_fd}>"$lock_path"
flock "$pipeline_lock_fd"
echo "[public-pipeline] current ablation finished: $(date --iso-8601=seconds)"

bash "$root_dir/code/scripts/setup_openpi_uav.sh"
bash "$root_dir/code/scripts/run_pi05_uav_training.sh"
bash "$root_dir/code/scripts/run_public_policy_baselines.sh"

echo "[public-pipeline] all training and evaluation complete: $(date --iso-8601=seconds)"
