#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exp_root="${EXP_ROOT:-$root_dir/experiments/fastwam_local_feature_ablation_run}"
eval_root="${EVAL_OUTPUT_ROOT:-$exp_root/online_eval}"
log_root="${EVAL_LOG_DIR:-$exp_root/eval_logs}"
python_bin="${PYTHON_BIN:-/home/ysq/.conda/envs/ysq_qwen/bin/python}"
openpi_root="${OPENPI_ROOT:-$root_dir/third_party/openpi}"
openpi_venv="${OPENPI_VENV:-$root_dir/.venvs/openpi_uav}"
gpu_ids_csv="${EVAL_GPU_IDS:-0,1,2,3,4,5}"
parallel_jobs="${EVAL_PARALLEL_JOBS:-6}"
baselines_csv="${PUBLIC_BASELINES:-random,pi05}"
sim_base_port="${SIM_SERVER_PORT:-30000}"
pi05_base_port="${PI05_POLICY_BASE_PORT:-18000}"
retry_count="${EVAL_SHARD_RETRIES:-3}"
visualize_keys="${EVAL_VISUALIZE_TRAJECTORY_KEYS:-City_1/trajectory_0451}"
mkdir -p "$eval_root" "$log_root" "$exp_root/logs"

export PYTHONPATH="$root_dir/code/src:${PYTHONPATH:-}"
IFS=',' read -ra gpu_pool <<< "$gpu_ids_csv"
IFS=',' read -ra baseline_names <<< "$baselines_csv"
if (( parallel_jobs < ${#gpu_pool[@]} )); then
  gpu_pool=("${gpu_pool[@]:0:parallel_jobs}")
fi
if (( ${#gpu_pool[@]} == 0 )); then
  echo "[ERROR] No evaluation GPU configured" >&2
  exit 1
fi

cities=(City_28 City_29 City_30)
for index in $(seq 1 27); do cities+=("City_${index}"); done
merge_cities=()
for index in $(seq 1 30); do merge_cities+=("City_${index}"); done
merge_city_csv="$(IFS=,; echo "${merge_cities[*]}")"
trajectory_spec="City_1-27:451-500;City_28-30:1-500"

city_range() {
  local city_index="${1#City_}"
  if (( 10#$city_index <= 27 )); then echo "451-500"; else echo "1-500"; fi
}

expected_city_count() {
  local range
  range="$(city_range "$1")"
  local start="${range%-*}" end="${range#*-}"
  echo "$((10#$end - 10#$start + 1))"
}

summary_is_valid() {
  local path="$1" city="$2" backend="$3" signature="$4" expected
  expected="$(expected_city_count "$city")"
  "$python_bin" - "$path" "$city" "$backend" "$signature" "$expected" <<'PY'
import json, sys
from pathlib import Path
try:
    data=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows=data["summaries"]
    args=data["args"]
    ok=(
        len(rows)==int(sys.argv[5])
        and str(args.get("scene_list"))==sys.argv[2]
        and str(args.get("policy_backend"))==sys.argv[3]
        and str(args.get("eval_semantic_signature"))==sys.argv[4]
        and all(str(row.get("failure_reason") or "").lower()!="runtime_error" for row in rows)
    )
except Exception:
    ok=False
raise SystemExit(0 if ok else 1)
PY
}

partial_is_compatible() {
  local path="$1" city="$2" backend="$3" signature="$4"
  "$python_bin" - "$path" "$city" "$backend" "$signature" <<'PY'
import json, sys
from pathlib import Path
try:
    data=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    args=data["args"]
    ok=(
        isinstance(data.get("summaries"), list)
        and str(args.get("scene_list"))==sys.argv[2]
        and str(args.get("policy_backend"))==sys.argv[3]
        and str(args.get("eval_semantic_signature"))==sys.argv[4]
    )
except Exception:
    ok=False
raise SystemExit(0 if ok else 1)
PY
}

server_pids=()
stop_pi05_servers() {
  local pid
  for pid in "${server_pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${server_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  server_pids=()
}
trap stop_pi05_servers EXIT INT TERM

openpi_python=""
pi05_checkpoint=""
start_pi05_servers() {
  openpi_python="${OPENPI_PYTHON:-$openpi_venv/bin/python}"
  if [[ ! -x "$openpi_python" ]]; then
    echo "[ERROR] Missing OpenPI Python environment: $openpi_python" >&2
    return 1
  fi
  local checkpoint_file="$exp_root/models/openpi/pi05_uav_latest_checkpoint.txt"
  if [[ ! -f "$checkpoint_file" ]]; then
    echo "[ERROR] Missing pi05 checkpoint pointer: $checkpoint_file" >&2
    return 1
  fi
  pi05_checkpoint="$(<"$checkpoint_file")"
  if [[ ! -d "$pi05_checkpoint" ]]; then
    echo "[ERROR] Missing pi05 checkpoint: $pi05_checkpoint" >&2
    return 1
  fi

  local slot gpu port log_file pid
  for slot in "${!gpu_pool[@]}"; do
    gpu="${gpu_pool[slot]//[[:space:]]/}"
    port="$((pi05_base_port + slot))"
    log_file="$log_root/pi05_policy_server_gpu${gpu}.log"
    (
      cd "$openpi_root"
      exec env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        XLA_PYTHON_CLIENT_MEM_FRACTION="${PI05_SERVER_MEM_FRACTION:-0.55}" \
        OPENPI_UAV_ASSETS_DIR="${OPENPI_UAV_ASSETS_DIR:-$exp_root/openpi_data/assets}" \
        OPENPI_UAV_CHECKPOINT_DIR="${OPENPI_UAV_CHECKPOINT_DIR:-$exp_root/models/openpi}" \
        "$openpi_python" scripts/serve_policy.py --port "$port" policy:checkpoint \
          --policy.config pi05_uav_lora \
          --policy.dir "$pi05_checkpoint"
    ) >"$log_file" 2>&1 &
    pid="$!"
    server_pids+=("$pid")
    echo "[pi05-server] gpu=$gpu port=$port pid=$pid log=$log_file"
  done

  for slot in "${!gpu_pool[@]}"; do
    port="$((pi05_base_port + slot))"
    "$python_bin" - "$port" <<'PY'
import socket, sys, time
port=int(sys.argv[1])
deadline=time.time()+1800
while time.time()<deadline:
    try:
        with socket.create_connection(("127.0.0.1",port),timeout=2):
            break
    except OSError:
        time.sleep(2)
else:
    raise SystemExit(f"pi05 policy server did not start on port {port}")
PY
  done
}

run_city() {
  local backend="$1" signature="$2" slot="$3" city="$4"
  local gpu="${gpu_pool[slot]//[[:space:]]/}"
  local range sim_port policy_port output_dir log_file summary_file partial_file
  range="$(city_range "$city")"
  sim_port="$((sim_base_port + slot * 1000))"
  policy_port="$((pi05_base_port + slot))"
  output_dir="$eval_root/$backend"
  log_file="$log_root/${backend}_${city}.log"
  summary_file="$output_dir/summary_${city}.json"
  partial_file="$output_dir/summary_partial_${city}.json"
  mkdir -p "$output_dir"
  if summary_is_valid "$summary_file" "$city" "$backend" "$signature"; then
    echo "[public-eval-skip] backend=$backend city=$city"
    return 0
  fi
  if [[ -f "$summary_file" ]]; then mv "$summary_file" "$summary_file.stale.$(date +%s)"; fi
  # Keep same-policy partial results across retries. online_eval_teacher drops
  # runtime_error rows when resuming, so only the failed trajectories rerun.
  if [[ -f "$partial_file" ]] && ! partial_is_compatible "$partial_file" "$city" "$backend" "$signature"; then
    mv "$partial_file" "$partial_file.stale.$(date +%s)"
  fi
  echo "[public-eval-dispatch] backend=$backend city=$city range=$range gpu=$gpu port=$sim_port"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m eval.online_eval_teacher \
    --dataset-root "$root_dir/Dataset" \
    --output-dir "$output_dir" \
    --executor-script "$root_dir/code/src/executor/trajectory_executor.py" \
    --policy-backend "$backend" \
    --public-action-horizon 8 \
    --pi05-policy-host 127.0.0.1 \
    --pi05-policy-port "$policy_port" \
    --scene-list "$city" \
    --trajectory-range "$range" \
    --eval-split all \
    --summary-shard-id "$city" \
    --eval-semantic-signature "$signature" \
    --start-sim-server \
    --sim-server-script "$root_dir/code/src/envs/sim_server.py" \
    --sim-server-root-path "$root_dir" \
    --sim-server-log "$log_root/${backend}_${city}_sim_server.log" \
    --sim-server-port "$sim_port" \
    --sim-gpu-id "$gpu" \
    --gpu-id "$gpu" \
    --scene-index 1 \
    --stop-sim-server-on-exit \
    --device cpu \
    --max-vel 1.0 \
    --max-yaw-rate 15.0 \
    --max-speed-norm 1.0 \
    --capture-distance 10.0 \
    --seed 42 \
    --no-compile-action-sampling \
    --camera-only-virtual-uav \
    --use-external-camera \
    --validate-camera-freshness \
    --camera-max-vehicle-distance 5.0 \
    --camera-render-frames 1 \
    --camera-capture-mode fresh_frame \
    --camera-render-max-fps 60 \
    --camera-pose-tolerance-m 0.05 \
    --camera-orientation-tolerance-deg 1.0 \
    --no-save-depth \
    --save-rgb \
    --visualize-trajectory-keys "$visualize_keys" \
    --no-save-transformer-attention-maps \
    --no-save-predicted-video \
    --save-target-crop-action-overlays \
    --no-save-trajectory-3d \
    2>&1 | tee -a "$log_file"
  if ! summary_is_valid "$summary_file" "$city" "$backend" "$signature"; then
    echo "[public-eval-invalid] backend=$backend city=$city summary=$summary_file" >&2
    return 1
  fi
}

evaluate_backend() {
  local backend="$1" signature="$2" next=0 failed=0 slot pid status
  local -a active_pids=()
  local -A pid_slots=() pid_labels=()

  launch_city() {
    local launch_slot="$1" launch_city="$2"
    (
      local attempt
      for ((attempt=1; attempt<=retry_count; attempt++)); do
        if run_city "$backend" "$signature" "$launch_slot" "$launch_city"; then exit 0; fi
        if (( attempt < retry_count )); then
          echo "[public-eval-retry] backend=$backend city=$launch_city next_attempt=$((attempt+1))/$retry_count"
          sleep 10
        fi
      done
      exit 1
    ) &
    pid="$!"
    active_pids+=("$pid")
    pid_slots["$pid"]="$launch_slot"
    pid_labels["$pid"]="$launch_city"
  }

  while (( next < ${#cities[@]} && next < ${#gpu_pool[@]} )); do
    launch_city "$next" "${cities[next]}"
    next=$((next+1))
  done
  while (( ${#active_pids[@]} > 0 )); do
    local done_pid=""
    if wait -n -p done_pid "${active_pids[@]}"; then status=0; else status=$?; fi
    slot="${pid_slots[$done_pid]}"
    if (( status != 0 )); then
      echo "[ERROR] public eval failed: $backend/${pid_labels[$done_pid]}" >&2
      failed=1
    fi
    local -a remaining=()
    for pid in "${active_pids[@]}"; do [[ "$pid" == "$done_pid" ]] || remaining+=("$pid"); done
    active_pids=("${remaining[@]}")
    unset 'pid_slots[$done_pid]' 'pid_labels[$done_pid]'
    if (( next < ${#cities[@]} )); then
      launch_city "$slot" "${cities[next]}"
      next=$((next+1))
    fi
  done
  unset -f launch_city
  (( failed == 0 )) || return 1
  "$python_bin" -m eval.merge_public_baseline_summaries \
    --model-dir "$eval_root/$backend" \
    --scene-list "$merge_city_csv" \
    --trajectory-spec "$trajectory_spec" \
    --policy-backend "$backend"
}

for backend in "${baseline_names[@]}"; do
  backend="${backend//[[:space:]]/}"
  case "$backend" in
    random)
      signature="random_v1_seed42_isotropic_velocity_yaw"
      ;;
    pi05)
      start_pi05_servers
      signature="pi05_uav_lora_$(basename "$pi05_checkpoint")_$(stat -c %Y "$pi05_checkpoint")"
      ;;
    *)
      echo "[ERROR] Unsupported public baseline: $backend" >&2
      exit 1
      ;;
  esac
  evaluate_backend "$backend" "$signature"
  if [[ "$backend" == "pi05" ]]; then stop_pi05_servers; fi
done

echo "[public-baselines] complete: $eval_root"
