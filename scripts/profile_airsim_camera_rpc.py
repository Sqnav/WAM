#!/usr/bin/env python3
"""Profile AirSim RGB RPC latency at several Unreal render-rate caps."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--executor-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-server-port", type=int, default=31000)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--fps", type=str, default="10,30,60,10")
    parser.add_argument("--scene-id", type=str, default="City_1")
    parser.add_argument("--camera-name", type=str, default="0")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--texture-pool-mb", type=int, default=0)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("camera_probe_executor", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import executor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def xyz(value: dict[str, Any], *, airsim_z: bool) -> np.ndarray:
    z = float(value["z"])
    return np.asarray(
        [float(value["x"]), float(value["y"]), -z if airsim_z else z],
        dtype=np.float32,
    )


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
        "over_500_ms": sum(value > 500.0 for value in values),
        "over_800_ms": sum(value > 800.0 for value in values),
    }


def image_comparison(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict[str, Any]:
    exact = 0
    maes: list[float] = []
    changed_ratios: list[float] = []
    max_errors: list[int] = []
    for ref, image in zip(reference, candidate):
        if np.array_equal(ref, image):
            exact += 1
        delta = np.abs(ref.astype(np.int16) - image.astype(np.int16))
        maes.append(float(delta.mean()))
        changed_ratios.append(float(np.count_nonzero(delta) / delta.size))
        max_errors.append(int(delta.max()))
    return {
        "frames": len(maes),
        "exact_frames": exact,
        "exact_frame_ratio": exact / len(maes),
        "mean_pixel_abs_error": float(statistics.fmean(maes)),
        "mean_changed_channel_ratio": float(statistics.fmean(changed_ratios)),
        "max_channel_error": max(max_errors),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads((args.dataset_dir / "uav_trajectory.json").read_text())
    frames = payload["trajectory"]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError("trajectory has no frames")

    target_asset = str(payload.get("target_asset_name") or "UAV1")
    target_positions = [xyz(frame["target_position"], airsim_z=True) for frame in frames]
    camera_states = []
    for frame in frames:
        quat = frame["uav_orientation_quaternion"]
        camera_states.append(
            {
                "position": xyz(frame["uav_position"], airsim_z=True),
                "orientation": np.asarray(
                    [float(quat["w"]), float(quat["x"]), float(quat["y"]), float(quat["z"])],
                    dtype=np.float64,
                ),
            }
        )

    os.environ["DAGGER_MULTI_WORKER"] = "1"
    module = load_module(args.executor_script.resolve())
    executor = module.TrajectoryExecutor(
        scene_id=args.scene_id,
        sim_server_host="127.0.0.1",
        sim_server_port=args.sim_server_port,
        gpu_id=args.gpu_id,
        scene_index=1,
        target_object_name="UAV1",
        target_asset_name=target_asset,
        camera_name=args.camera_name,
        auto_start_scene=True,
        deterministic_step_mode=True,
        jammer_enabled=False,
    )
    executor.save_depth = False
    executor.require_depth = False
    executor.use_external_camera = True
    executor.validate_camera_freshness = True
    executor.camera_pose_tolerance_m = 0.05
    executor.camera_orientation_tolerance_deg = 1.0
    executor.disable_physics_pose_refresh = True

    all_runs: list[dict[str, Any]] = []
    run_images: list[list[np.ndarray]] = []
    try:
        executor._prepare_target_object()
        executor.initialize_scene_objects_only(np.asarray(target_positions, dtype=np.float32))
        if args.texture_pool_mb > 0:
            commands = [
                f"r.Streaming.PoolSize {args.texture_pool_mb}",
                "r.Streaming.UseFixedPoolSize 1",
            ]
            for command in commands:
                if not executor.client.simRunConsoleCommand(command):
                    raise RuntimeError(f"Unreal rejected console command: {command}")

        for run_index, fps_text in enumerate(args.fps.split(",")):
            fps = float(fps_text.strip())
            executor.configure_camera_rendering(fps)
            executor._last_camera_timestamp = None
            executor.camera_stale_rejections = 0
            executor.camera_pose_rejections = 0
            rpc_values: list[float] = []
            total_values: list[float] = []
            timestamps: list[int] = []
            images: list[np.ndarray] = []
            print(f"[probe] run={run_index} fps={fps:g} frames={len(frames)}", flush=True)

            for frame_index, (target_position, camera_state) in enumerate(
                zip(target_positions, camera_states)
            ):
                executor.move_target_object(target_position)
                executor.set_external_camera_pose_from_state(camera_state)
                started = time.perf_counter()
                image, _ = executor.get_fresh_camera_images()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if image is None:
                    raise RuntimeError(f"empty image at run={run_index} frame={frame_index}")
                profile = executor._last_camera_profile_ms
                rpc_values.append(float(profile["camera_rpc_ms"]))
                total_values.append(elapsed_ms)
                timestamps.append(int(executor._last_camera_timestamp or 0))
                images.append(np.asarray(image, dtype=np.uint8).copy())
                if frame_index % 10 == 0 or frame_index + 1 == len(frames):
                    print(
                        f"[probe] fps={fps:g} frame={frame_index + 1}/{len(frames)} "
                        f"rpc_ms={rpc_values[-1]:.1f} total_ms={total_values[-1]:.1f}",
                        flush=True,
                    )

            run_name = f"run_{run_index:02d}_fps_{fps:g}"
            cv2.imwrite(
                str(args.output_dir / f"{run_name}_first.png"),
                cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR),
            )
            hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
            run_record = {
                "run_index": run_index,
                "fps": fps,
                "rpc": latency_summary(rpc_values),
                "capture_total": latency_summary(total_values),
                "timestamps_strictly_increasing": all(
                    right > left for left, right in zip(timestamps, timestamps[1:])
                ),
                "stale_rejections": int(executor.camera_stale_rejections),
                "pose_rejections": int(executor.camera_pose_rejections),
                "frame_hashes": hashes,
            }
            if run_images:
                run_record["image_difference_from_run_0"] = image_comparison(
                    run_images[0], images
                )
            all_runs.append(run_record)
            run_images.append(images)
            (args.output_dir / "results_partial.json").write_text(
                json.dumps({"runs": all_runs}, indent=2)
            )

        result = {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "scene_id": args.scene_id,
            "frames": len(frames),
            "gpu_id": args.gpu_id,
            "sim_server_port": args.sim_server_port,
            "texture_pool_mb": args.texture_pool_mb,
            "runs": all_runs,
        }
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        try:
            executor._cleanup_after_execution(skip_hover=True)
        except Exception as exc:
            print(f"[probe-warning] cleanup failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
