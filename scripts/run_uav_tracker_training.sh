#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT="${ROOT:-${DEFAULT_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${ROOT}/code}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/uav_tracker_imagenet_square}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/tracking_manifest.json}"
CONDA_ENV="${CONDA_ENV:-ysq_qwen}"
MAX_GPUS="${MAX_GPUS:-4}"
EXCLUDE_GPUS="${EXCLUDE_GPUS:-1,2,3}"
MAX_USED_MEMORY_MB="${MAX_USED_MEMORY_MB:-2000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
RUN_TRACKER_CACHE_AFTER_TRAINING="${RUN_TRACKER_CACHE_AFTER_TRAINING:-true}"
REQUIRE_REAL_ANNOTATIONS="${REQUIRE_REAL_ANNOTATIONS:-false}"
TRACKER_CACHE_ROOT="${TRACKER_CACHE_ROOT:-${ROOT}/experiments/square_tracker_cache}"
TRACKER_CACHE_SCENE_LIST="${TRACKER_CACHE_SCENE_LIST:-City_1,City_2,City_3}"
TRACKER_CACHE_TRAJECTORY_RANGE="${TRACKER_CACHE_TRAJECTORY_RANGE:-1-450}"
TRACKER_CACHE_WORKERS_PER_GPU="${TRACKER_CACHE_WORKERS_PER_GPU:-4}"
TRACKER_CACHE_LOG_DIR="${TRACKER_CACHE_LOG_DIR:-${ROOT}/experiments/tracker_artifacts/logs/square_tracker_cache_logs}"

mkdir -p "$OUTPUT_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${CODE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [[ ! -s "$MANIFEST" || "${REBUILD_MANIFEST:-1}" == "1" ]]; then
  python -m tracking.build_manifest \
    --dataset-root "$DATASET_ROOT" \
    --output "$MANIFEST" \
    --scenes ${SCENES:-City_1 City_2 City_3} \
    --trajectory-start "${TRAJECTORY_START:-1}" \
    --trajectory-end "${TRAJECTORY_END:-450}" \
    --train-only \
    --fov-deg "${FOV_DEG:-90}" \
    --camera-offset ${CAMERA_OFFSET:-0.46 0.0 0.0} \
    --target-width-m "${TARGET_WIDTH_M:-0.8}" \
    --target-height-m "${TARGET_HEIGHT_M:-0.35}" \
    --box-shape square
fi

if [[ "$REQUIRE_REAL_ANNOTATIONS" == "true" || "$REQUIRE_REAL_ANNOTATIONS" == "1" ]]; then
  "$PYTHON_BIN" - "$MANIFEST" "${SCENES:-City_1 City_2 City_3}" \
    "${TRAJECTORY_START:-1}" "${TRAJECTORY_END:-450}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
scenes = [value for value in sys.argv[2].split() if value]
start, end = int(sys.argv[3]), int(sys.argv[4])
expected_train = len(scenes) * (end - start + 1)
expected = expected_train
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
train_count = len(payload.get("train", []))
val_count = len(payload.get("val", []))
if (
    total != expected
    or train_count != expected_train
    or val_count != 0
    or real != expected
    or projected != 0
    or fully_annotated != expected
):
    raise SystemExit(
        "Tracker manifest is not fully backed by real target_boxes.json annotations: "
        f"expected_train={expected_train}, actual_train={train_count}, "
        f"expected_val=0, actual_val={val_count}, trajectories={total}, "
        f"real={real}, projected_weak={projected}, "
        f"fully_annotated={fully_annotated}. "
        "Finish/recollect the requested dataset range before training."
    )
print(f"[manifest] verified complete real target boxes: {fully_annotated}/{expected}")
PY
fi

if [[ -n "${GPUS:-}" ]]; then
  selected="$GPUS"
else
  selected=""
  while IFS=',' read -r index memory util; do
    index="${index//[[:space:]]/}"
    memory="${memory//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    if [[ ",$EXCLUDE_GPUS," == *",$index,"* ]]; then
      continue
    fi
    if (( memory <= MAX_USED_MEMORY_MB && util <= MAX_GPU_UTIL )); then
      selected="${selected:+${selected},}${index}"
      if (( $(awk -F',' '{print NF}' <<< "$selected") >= MAX_GPUS )); then
        break
      fi
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
fi

if [[ -z "$selected" ]]; then
  echo "No GPU satisfies memory<=${MAX_USED_MEMORY_MB}MiB and utilization<=${MAX_GPU_UTIL}%." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$selected"
NPROC=$(awk -F',' '{print NF}' <<< "$selected")
echo "[$(date '+%F %T')] GPUs=$selected workers=$NPROC output=$OUTPUT_DIR"

pretrained_args=()
PRETRAINED_PATH="${PRETRAINED_PATH:-${ROOT}/model/pretrained/deit_tiny_patch16_224-a1311bcf.pth}"
if [[ -s "$PRETRAINED_PATH" ]]; then
  pretrained_args=(--pretrained-path "$PRETRAINED_PATH")
else
  pretrained_args=(--no-pretrained)
  echo "WARNING: ImageNet backbone weights not found at $PRETRAINED_PATH; training from random initialization." >&2
fi

manifest_sha256="$(sha256sum "$MANIFEST" | awk '{print $1}')"
tracker_source_sha256="$(sha256sum \
  "$CODE_ROOT/src/tracking/data.py" \
  "$CODE_ROOT/src/tracking/losses.py" \
  "$CODE_ROOT/src/tracking/model.py" \
  "$CODE_ROOT/src/tracking/train.py" | sha256sum | awk '{print $1}')"
if [[ -s "$PRETRAINED_PATH" ]]; then
  pretrained_fingerprint="$(stat -c '%s:%Y' "$PRETRAINED_PATH")"
else
  pretrained_fingerprint="missing"
fi
tracker_signature="$(printf '%s\n' \
  "manifest=$manifest_sha256" \
  "source=$tracker_source_sha256" \
  "epochs=${EPOCHS:-10}" \
  "batch_size=${BATCH_SIZE:-32}" \
  "samples_per_epoch=${SAMPLES_PER_EPOCH:-60000}" \
  "val_samples=${VAL_SAMPLES:-0}" \
  "max_gap=${MAX_GAP:-40}" \
  "lr=${LR:-0.0004}" \
  "pretrained=$PRETRAINED_PATH:$pretrained_fingerprint" \
  "square_boxes=true" | sha256sum | awk '{print $1}')"
signature_path="$OUTPUT_DIR/training_signature.sha256"
saved_signature=""
if [[ -s "$OUTPUT_DIR/latest.pt" ]]; then
  saved_signature="$("$PYTHON_BIN" - "$OUTPUT_DIR/latest.pt" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
print(str(args.get("training_signature", "")))
PY
)"
fi
signature_matches=false
if [[ -n "$saved_signature" && "$saved_signature" == "$tracker_signature" ]]; then
  signature_matches=true
fi

resume_args=()
if [[ -s "$OUTPUT_DIR/latest.pt" && "${RESUME:-1}" == "1" && "$signature_matches" == "true" ]]; then
  resume_args=(--resume "$OUTPUT_DIR/latest.pt")
elif [[ -s "$OUTPUT_DIR/latest.pt" && "${RESUME:-1}" == "1" ]]; then
  echo "Tracker training inputs changed; restart instead of resuming stale latest.pt." >&2
fi

tracker_complete=false
if [[ -s "$OUTPUT_DIR/latest.pt" && "$signature_matches" == "true" && "${RESUME:-1}" == "1" ]]; then
  if "$PYTHON_BIN" - "$OUTPUT_DIR/latest.pt" "${EPOCHS:-10}" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
completed_epoch = int(checkpoint.get("epoch", -1))
required_epoch = int(sys.argv[2]) - 1
raise SystemExit(0 if completed_epoch >= required_epoch else 1)
PY
  then
    tracker_complete=true
  fi
fi

if [[ "$tracker_complete" == "true" ]]; then
  echo "[$(date '+%F %T')] Square Tracker already completed; skip training: $OUTPUT_DIR/latest.pt"
else
  torchrun --standalone --nproc_per_node="$NPROC" -m tracking.train \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "${EPOCHS:-10}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --num-workers "${NUM_WORKERS:-8}" \
    --samples-per-epoch "${SAMPLES_PER_EPOCH:-60000}" \
    --val-samples "${VAL_SAMPLES:-0}" \
    --max-gap "${MAX_GAP:-40}" \
    --lr "${LR:-0.0004}" \
    --print-interval "${PRINT_INTERVAL:-20}" \
    --training-signature "$tracker_signature" \
    --square-boxes \
    "${pretrained_args[@]}" \
    "${resume_args[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
fi

"$PYTHON_BIN" - "$OUTPUT_DIR/best.pt" "$tracker_signature" <<'PY'
import sys
from pathlib import Path

import torch

checkpoint_path = Path(sys.argv[1])
expected_signature = sys.argv[2]
if not checkpoint_path.is_file():
    raise SystemExit(f"Tracker training did not produce {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
actual_signature = str(args.get("training_signature", ""))
if actual_signature != expected_signature:
    raise SystemExit(
        f"Tracker best.pt has a stale training signature: expected={expected_signature}, "
        f"actual={actual_signature or '<missing>'}"
    )
print(f"[tracker] verified best checkpoint signature: {expected_signature}")
PY

printf '%s\n' "$tracker_signature" > "$signature_path.tmp"
mv -f "$signature_path.tmp" "$signature_path"

precompute_tracker_cache() {
  local range_start range_end total_ranges total_workers chunk_size status
  local -a cache_gpus pids cache_init_args
  IFS='-' read -r range_start range_end <<< "$TRACKER_CACHE_TRAJECTORY_RANGE"
  if [[ -z "${range_start:-}" || -z "${range_end:-}" ]]; then
    echo "Invalid TRACKER_CACHE_TRAJECTORY_RANGE: $TRACKER_CACHE_TRAJECTORY_RANGE" >&2
    return 1
  fi
  IFS=',' read -ra cache_gpus <<< "$selected"
  if (( ${#cache_gpus[@]} == 0 )); then
    echo "No GPU configured for tracker cache generation." >&2
    return 1
  fi
  mkdir -p "$TRACKER_CACHE_ROOT" "$TRACKER_CACHE_LOG_DIR"
  cache_init_args=(--square-init-bbox --native-first-frame-response)
  if [[ "$REQUIRE_REAL_ANNOTATIONS" == "true" || "$REQUIRE_REAL_ANNOTATIONS" == "1" ]]; then
    cache_init_args+=(--require-real-init-bbox)
  fi
  total_ranges=$((10#$range_end - 10#$range_start + 1))
  total_workers=$(( ${#cache_gpus[@]} * TRACKER_CACHE_WORKERS_PER_GPU ))
  chunk_size=$(( (total_ranges + total_workers - 1) / total_workers ))
  pids=()
  for ((worker=0; worker<total_workers; worker++)); do
    local start end range gpu
    start=$((10#$range_start + worker * chunk_size))
    (( start > 10#$range_end )) && break
    end=$((start + chunk_size - 1))
    (( end > 10#$range_end )) && end=$((10#$range_end))
    gpu="${cache_gpus[$((worker % ${#cache_gpus[@]}))]}"
    range="$start-$end"
    echo "[$(date '+%F %T')] tracker cache gpu=$gpu range=$range"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      "$PYTHON_BIN" -m eval.square_tracker_cache_batch \
      --dataset-root "$DATASET_ROOT" \
      --cache-root "$TRACKER_CACHE_ROOT" \
      --checkpoint "$OUTPUT_DIR/best.pt" \
      --scenes "$TRACKER_CACHE_SCENE_LIST" \
      --trajectory-range "$range" \
      "${cache_init_args[@]}" \
      --device cuda >"$TRACKER_CACHE_LOG_DIR/gpu${gpu}_${range}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  (( status == 0 )) || return "$status"
  echo "[$(date '+%F %T')] Square Tracker cache complete: $TRACKER_CACHE_ROOT"
}

if [[ "$RUN_TRACKER_CACHE_AFTER_TRAINING" == "true" || "$RUN_TRACKER_CACHE_AFTER_TRAINING" == "1" ]]; then
  precompute_tracker_cache
fi
