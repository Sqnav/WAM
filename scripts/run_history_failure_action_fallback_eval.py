#!/usr/bin/env python3
"""Re-evaluate failed history-policy trajectories with cached-action fallback.

The loss signal can be the deployable confidence produced by the model's
fine-tuned Tracker head, or privileged simulator geometry for an oracle study.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "experiments/fastwam_local_feature_ablation_run"
)
DEFAULT_BASELINE = (
    DEFAULT_EXPERIMENT_ROOT
    / "online_eval/fastwam_current_box_historical_target_memory/summary.json"
)
DEFAULT_OUTPUT = (
    DEFAULT_EXPERIMENT_ROOT
    / "online_eval_diagnostics"
    / "fastwam_history_failed_last_visible_sequence_oracle"
)
DEFAULT_MODEL_CONFIDENCE_OUTPUT = (
    DEFAULT_EXPERIMENT_ROOT
    / "online_eval_diagnostics"
    / "fastwam_history_failed_model_confidence_sequence"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("/home/ysq/.conda/envs/ysq_qwen/bin/python"),
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5")
    parser.add_argument("--base-port", type=int, default=42000)
    parser.add_argument("--min-free-memory-gb", type=float, default=40.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=60.0)
    parser.add_argument("--no-wait-for-gpus", action="store_true")
    parser.add_argument(
        "--fallback-loss-signal",
        choices=["model_confidence", "geometry_visibility"],
        default="geometry_visibility",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    return parser.parse_args()


def trajectory_number(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    if match is None:
        raise ValueError(f"Cannot parse trajectory number from {name!r}")
    return int(match.group(1))


def load_failed_by_scene(path: Path) -> tuple[dict[str, Any], dict[str, list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: dict[str, list[int]] = defaultdict(list)
    for item in payload.get("summaries", []):
        if not bool(item.get("success", False)):
            failures[str(item["scene_id"])].append(
                trajectory_number(str(item["trajectory_name"]))
            )
    if not failures:
        raise RuntimeError(f"No failed trajectories found in {path}")
    return payload, {scene: sorted(values) for scene, values in failures.items()}


def assign_scene_groups(
    failed_by_scene: dict[str, list[int]], worker_count: int
) -> list[list[str]]:
    groups: list[list[str]] = [[] for _ in range(worker_count)]
    loads = [0 for _ in range(worker_count)]
    ordered = sorted(
        failed_by_scene,
        key=lambda scene: (-len(failed_by_scene[scene]), trajectory_number(scene)),
    )
    for scene in ordered:
        slot = min(range(worker_count), key=lambda index: (loads[index], index))
        groups[slot].append(scene)
        loads[slot] += len(failed_by_scene[scene])
    for group in groups:
        group.sort(key=trajectory_number)
    print(f"[fallback-eval] worker loads: {loads}", flush=True)
    return groups


def free_gpu_memory_mb() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values: dict[int, int] = {}
    for line in result.stdout.splitlines():
        index, free = (part.strip() for part in line.split(",", 1))
        values[int(index)] = int(free)
    return values


def wait_for_gpus(gpu_ids: list[int], minimum_gb: float, poll_seconds: float) -> None:
    required_mb = int(float(minimum_gb) * 1024)
    while True:
        free = free_gpu_memory_mb()
        missing = {
            gpu: free.get(gpu, 0)
            for gpu in gpu_ids
            if free.get(gpu, 0) < required_mb
        }
        if not missing:
            print(
                f"[fallback-eval] GPUs ready: "
                + ", ".join(f"{gpu}={free[gpu] / 1024:.1f}GB" for gpu in gpu_ids),
                flush=True,
            )
            return
        print(
            "[fallback-eval] waiting for GPU memory: "
            + ", ".join(f"{gpu}={value / 1024:.1f}GB" for gpu, value in missing.items()),
            flush=True,
        )
        time.sleep(max(float(poll_seconds), 5.0))


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def worker_command(
    *,
    python_bin: Path,
    baseline_args: dict[str, Any],
    output_dir: Path,
    failed_by_scene: dict[str, list[int]],
    scenes: list[str],
    gpu_id: int,
    port: int,
    shard_id: str,
    fallback_loss_signal: str,
    confidence_threshold: float,
) -> list[str]:
    trajectory_spec = ",".join(
        f"{scene}:{number}"
        for scene in scenes
        for number in failed_by_scene[scene]
    )
    scene_list = ",".join(scenes)
    command = [
        str(python_bin),
        "-m",
        "eval.online_eval_teacher",
        "--dataset-root",
        str(baseline_args["dataset_root"]),
        "--checkpoint",
        str(baseline_args["checkpoint"]),
        "--output-dir",
        str(output_dir),
        "--executor-script",
        str(baseline_args["executor_script"]),
        "--start-sim-server",
        "--sim-server-script",
        str(baseline_args["sim_server_script"]),
        "--sim-server-root-path",
        str(baseline_args["sim_server_root_path"]),
        "--sim-server-log",
        str(output_dir / "logs" / f"{shard_id}_sim_server.log"),
        "--sim-server-wait-seconds",
        str(baseline_args.get("sim_server_wait_seconds", 60)),
        "--stop-sim-server-on-exit",
        "--scene-list",
        scene_list,
        "--trajectory-range",
        trajectory_spec,
        "--eval-split",
        "all",
        "--summary-shard-id",
        shard_id,
        "--eval-semantic-signature",
        f"history_failed_{fallback_loss_signal}_last_sequence_v1",
        "--sim-server-host",
        str(baseline_args.get("sim_server_host", "127.0.0.1")),
        "--sim-server-port",
        str(port),
        "--scene-index",
        str(baseline_args.get("scene_index", 1)),
        "--gpu-id",
        str(gpu_id),
        "--sim-gpu-id",
        str(gpu_id),
        "--device",
        "cuda",
        "--max-vel",
        str(baseline_args.get("max_vel", 1.0)),
        "--max-yaw-rate",
        str(baseline_args.get("max_yaw_rate", 15.0)),
        "--max-speed-norm",
        str(baseline_args.get("max_speed_norm", 1.0)),
        "--sampling-steps",
        str(baseline_args.get("sampling_steps", 8)),
        "--capture-distance",
        str(baseline_args.get("capture_distance", 10.0)),
        "--use-target-relative-context",
        bool_text(baseline_args.get("use_target_relative_context", False)),
        "--target-relative-context-scale",
        str(baseline_args.get("target_relative_context_scale", 1.0)),
        "--target-relative-token-scale",
        str(baseline_args.get("target_relative_token_scale", 1.0)),
        "--target-relative-context-hidden-dim",
        str(baseline_args.get("target_relative_context_hidden_dim", 512)),
        "--use-wan22-encoders",
        bool_text(baseline_args.get("use_wan22_encoders", True)),
        "--wan22-model-base-path",
        str(baseline_args["wan22_model_base_path"]),
        "--wan22-fastwam-src-path",
        str(baseline_args["wan22_fastwam_src_path"]),
        "--wan22-skip-download",
        bool_text(baseline_args.get("wan22_skip_download", False)),
        "--wan22-text-context-length",
        str(baseline_args.get("wan22_text_context_length", 512)),
        "--wan22-text-encode-batch-size",
        str(baseline_args.get("wan22_text_encode_batch_size", 4)),
        "--use-diffusion-actor",
        bool_text(baseline_args.get("use_diffusion_actor", True)),
        "--tracker-checkpoint",
        str(baseline_args["tracker_checkpoint"]),
        "--reuse-last-confident-action-sequence",
        "--tracker-fallback-action-mode",
        "remaining_sequence",
        "--tracker-fallback-loss-signal",
        str(fallback_loss_signal),
        "--tracker-detection-confidence-threshold",
        str(confidence_threshold),
        "--compile-action-sampling",
        "--compile-action-sampling-mode",
        str(baseline_args.get("compile_action_sampling_mode", "reduce-overhead")),
        "--camera-only-virtual-uav",
        "--validate-camera-freshness",
        "--camera-max-vehicle-distance",
        str(baseline_args.get("camera_max_vehicle_distance", 5.0)),
        "--camera-render-frames",
        str(baseline_args.get("camera_render_frames", 1)),
        "--camera-capture-mode",
        str(baseline_args.get("camera_capture_mode", "fresh_frame")),
        "--camera-render-max-fps",
        str(baseline_args.get("camera_render_max_fps", 60.0)),
        "--camera-pose-tolerance-m",
        str(baseline_args.get("camera_pose_tolerance_m", 0.05)),
        "--camera-orientation-tolerance-deg",
        str(baseline_args.get("camera_orientation_tolerance_deg", 1.0)),
        "--use-external-camera",
        "--no-save-depth",
        "--no-save-rgb",
        "--no-save-transformer-attention-maps",
        "--no-save-attention-tracker-comparisons",
        "--no-save-predicted-video",
        "--no-save-target-crop-action-overlays",
        "--no-save-trajectory-3d",
    ]
    if bool(baseline_args.get("require_visibility_for_success", False)):
        command.append("--require-visibility-for-success")
    return command


def mean(rows: list[dict[str, Any]], key: str, boolean: bool = False) -> float:
    if boolean:
        values = [1.0 if row.get(key) else 0.0 for row in rows]
    else:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / max(len(values), 1)


def merge_results(
    output_dir: Path,
    shard_ids: list[str],
    expected_count: int,
    baseline: dict[str, Any],
    fallback_loss_signal: str,
    confidence_threshold: float,
) -> Path:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for shard_id in shard_ids:
        path = output_dir / f"summary_{shard_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("summaries", []):
            key = (str(item["scene_id"]), str(item["trajectory_name"]))
            if key in unique:
                raise RuntimeError(f"Duplicate diagnostic trajectory: {key}")
            unique[key] = item
    rows = list(unique.values())
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} results, found {len(rows)}")

    recovered = sum(bool(row.get("success")) for row in rows)
    baseline_total = int(baseline["num_trajectories"])
    baseline_successes = sum(bool(row.get("success")) for row in baseline["summaries"])
    failure_reasons = Counter(str(row.get("failure_reason") or "none") for row in rows)
    deployable = fallback_loss_signal == "model_confidence"
    merged = {
        "diagnostic": f"{fallback_loss_signal} cached-action fallback on baseline failures",
        "deployable": deployable,
        "tracker_fallback_loss_signal": fallback_loss_signal,
        "tracker_detection_confidence_threshold": float(confidence_threshold),
        "num_trajectories": len(rows),
        "recovered_successes": int(recovered),
        "failure_recovery_rate": float(recovered / len(rows)),
        "projected_full_success_rate_if_existing_successes_unchanged": float(
            (baseline_successes + recovered) / baseline_total
        ),
        "average_effective_tracked_frames": mean(rows, "effective_tracked_frames"),
        "average_effective_tracking_ratio": mean(rows, "effective_tracking_ratio"),
        "collision_rate": mean(rows, "collision", boolean=True),
        "mean_final_distance": mean(rows, "final_distance"),
        "mean_distance": mean(rows, "mean_distance"),
        "total_tracker_missed_steps": int(
            sum(int(row.get("tracker_missed_steps", 0)) for row in rows)
        ),
        "total_reused_action_sequence_steps": int(
            sum(int(row.get("reused_action_sequence_steps", 0)) for row in rows)
        ),
        "failure_reason_counts": dict(failure_reasons),
        "summaries": rows,
    }
    path = output_dir / "comparison.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path


def main() -> None:
    args = parse_args()
    gpu_ids = [int(value.strip()) for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU")
    baseline_path = args.baseline_summary.expanduser().resolve()
    default_output = (
        DEFAULT_MODEL_CONFIDENCE_OUTPUT
        if args.fallback_loss_signal == "model_confidence"
        else DEFAULT_OUTPUT
    )
    output_dir = (args.output_dir or default_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    baseline, failed_by_scene = load_failed_by_scene(baseline_path)
    failed_count = sum(len(values) for values in failed_by_scene.values())
    print(
        f"[fallback-eval] selected {failed_count} failures from {baseline_path}",
        flush=True,
    )
    groups = assign_scene_groups(failed_by_scene, len(gpu_ids))
    if not args.no_wait_for_gpus:
        wait_for_gpus(gpu_ids, args.min_free_memory_gb, args.gpu_poll_seconds)

    baseline_args = dict(baseline["args"])
    environment = os.environ.copy()
    python_paths = [
        str(CODE_ROOT / "src"),
        str(baseline_args["wan22_fastwam_src_path"]),
        environment.get("PYTHONPATH", ""),
    ]
    environment["PYTHONPATH"] = os.pathsep.join(value for value in python_paths if value)
    processes: list[tuple[str, subprocess.Popen[Any], Any]] = []
    shard_ids: list[str] = []
    for slot, (gpu_id, scenes) in enumerate(zip(gpu_ids, groups)):
        if not scenes:
            continue
        shard_id = f"oracle_gpu{gpu_id}"
        shard_ids.append(shard_id)
        command = worker_command(
            python_bin=args.python_bin.expanduser().resolve(),
            baseline_args=baseline_args,
            output_dir=output_dir,
            failed_by_scene=failed_by_scene,
            scenes=scenes,
            gpu_id=gpu_id,
            port=int(args.base_port) + slot * 1000,
            shard_id=shard_id,
            fallback_loss_signal=str(args.fallback_loss_signal),
            confidence_threshold=float(args.confidence_threshold),
        )
        log_file = open(output_dir / "logs" / f"{shard_id}.log", "a", encoding="utf-8")
        worker_env = environment.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(
            f"[fallback-eval] launch {shard_id}: scenes={','.join(scenes)} "
            f"trajectories={sum(len(failed_by_scene[scene]) for scene in scenes)}",
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=CODE_ROOT,
            env=worker_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_id, process, log_file))

    failed_workers: list[tuple[str, int]] = []
    for shard_id, process, log_file in processes:
        return_code = process.wait()
        log_file.close()
        if return_code != 0:
            failed_workers.append((shard_id, return_code))
    if failed_workers:
        raise RuntimeError(f"Fallback evaluation workers failed: {failed_workers}")
    comparison = merge_results(
        output_dir,
        shard_ids,
        failed_count,
        baseline,
        fallback_loss_signal=str(args.fallback_loss_signal),
        confidence_threshold=float(args.confidence_threshold),
    )
    print(f"[fallback-eval] completed: {comparison}", flush=True)


if __name__ == "__main__":
    main()
