from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from PIL import Image

from eval.online_eval_teacher import (
    _save_trajectory_3d_plot,
    save_predicted_video_state_overlays,
    save_target_crop_action_overlay,
)
from eval.visualize_tracker_heatmaps import _initial_tracker_bbox
from tracking.runtime import SquareTracker


def _trajectory_dirs(root: Path) -> Iterable[Path]:
    yield from sorted(path.parent for path in root.rglob("online_rollout.json"))


def _load_rollout(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data.get("steps"), list):
        raise ValueError(f"missing steps list: {path}")
    return data


def _save_step_predicted_video_state_overlays(
    trajectory_dir: Path,
    step: Dict[str, Any],
    *,
    overwrite: bool,
) -> bool:
    offline_replay = step.get("offline_future_state_replay")
    if not isinstance(offline_replay, dict):
        offline_replay = {}
    current_box = offline_replay.get(
        "current_target_box_cxcywh", step.get("current_target_box_cxcywh")
    )
    future_boxes = offline_replay.get(
        "future_target_boxes_cxcywh", step.get("future_target_boxes_cxcywh")
    )
    predicted_frames = step.get("predicted_video_frames")
    if not (
        current_box is not None
        and isinstance(future_boxes, list)
        and isinstance(predicted_frames, list)
        and len(predicted_frames) == len(future_boxes) + 1
    ):
        return False

    predicted_paths = [trajectory_dir / str(value) for value in predicted_frames]
    if not predicted_paths:
        return False
    output_dir = predicted_paths[0].parent / "state_overlays"
    expected_paths = [
        output_dir / f"{path.stem}_s{index}.png"
        for index, path in enumerate(predicted_paths)
    ]
    if not overwrite and all(path.is_file() for path in expected_paths):
        step["predicted_video_state_overlays"] = [
            str(path.relative_to(trajectory_dir)) for path in expected_paths
        ]
        return False

    state_overlay_paths = save_predicted_video_state_overlays(
        output_dir,
        predicted_paths,
        [current_box, *future_boxes],
    )
    step["predicted_video_state_overlays"] = [
        str(path.relative_to(trajectory_dir)) for path in state_overlay_paths
    ]
    return True


def postprocess_trajectory(
    trajectory_dir: Path,
    *,
    output_name: str,
    fov_deg: float,
    camera_offset_body: list[float],
    dt: float,
    overwrite: bool,
    tracker: Optional[SquareTracker] = None,
    generate_trajectory_3d: bool = True,
) -> tuple[int, int]:
    rollout_path = trajectory_dir / "online_rollout.json"
    data = _load_rollout(rollout_path)
    steps = data["steps"]
    plot_path = trajectory_dir / "trajectory_3d.png"
    if generate_trajectory_3d and (overwrite or not plot_path.is_file()):
        _save_trajectory_3d_plot(trajectory_dir, steps)

    overlay_dir = trajectory_dir / output_name
    replayed_tracker: Dict[int, tuple[list[float], float]] = {}
    if tracker is not None and steps:
        first_step = steps[0]
        init_bbox = _initial_tracker_bbox(first_step, rollout_path)
        for position, step in enumerate(steps):
            index = int(step.get("step", position))
            rgb_path = trajectory_dir / "rgb" / f"frame_{index:05d}.png"
            if not rgb_path.is_file():
                raise FileNotFoundError(f"Missing RGB for Tracker replay: {rgb_path}")
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if position == 0:
                tracker.initialize(rgb, init_bbox)
                bbox = [float(value) for value in init_bbox]
                confidence = 1.0
            else:
                result = tracker.track(rgb)
                bbox = [float(value) for value in result["bbox"]]
                confidence = float(result["confidence"])
            replayed_tracker[index] = (bbox, confidence)
            step["ortrack_bbox_xywh"] = bbox
            step["ortrack_confidence"] = confidence
            step["tracker_bbox_xywh"] = bbox
            step["tracker_confidence"] = confidence

    generated = 0
    skipped = 0
    for step in steps:
        index = int(step.get("step", -1))
        state_overlays_generated = _save_step_predicted_video_state_overlays(
            trajectory_dir,
            step,
            overwrite=overwrite,
        )
        offline_replay = step.get("offline_future_state_replay")
        if not isinstance(offline_replay, dict):
            offline_replay = {}
        actions = offline_replay.get(
            "action_sequence_physical", step.get("action_sequence_physical")
        )
        relative_target = step.get("relative_target_body")
        if index < 0 or not actions or relative_target is None:
            if state_overlays_generated:
                generated += 1
            else:
                skipped += 1
            continue
        rgb_path = trajectory_dir / "rgb" / f"frame_{index:05d}.png"
        output_path = overlay_dir / f"frame_{index:05d}_target_crop_action_traj.png"
        if not rgb_path.is_file():
            if state_overlays_generated:
                generated += 1
            else:
                skipped += 1
            continue
        if output_path.is_file() and not overwrite:
            if state_overlays_generated:
                generated += 1
            continue
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        replay = replayed_tracker.get(index)
        tracker_bbox = replay[0] if replay is not None else step.get("ortrack_bbox_xywh")
        tracker_confidence = replay[1] if replay is not None else step.get("ortrack_confidence")
        metadata = save_target_crop_action_overlay(
            output_path,
            rgb,
            np.asarray(relative_target, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            fov_deg,
            camera_offset_body,
            dt,
            ortrack_bbox_xywh=tracker_bbox,
            ortrack_confidence=tracker_confidence,
            model_driven_search_geometry=offline_replay.get(
                "model_driven_search_geometry",
                step.get("model_driven_search_geometry"),
            ),
            current_state_box_cxcywh=offline_replay.get(
                "current_target_box_cxcywh",
                step.get("current_target_box_cxcywh"),
            ),
            future_state_box_cxcywh=step.get("future_target_box_cxcywh"),
            future_state_boxes_cxcywh=offline_replay.get(
                "future_target_boxes_cxcywh",
                step.get("future_target_boxes_cxcywh"),
            ),
        )
        metadata["overlay"] = str(output_path.relative_to(trajectory_dir))
        step["target_crop_action_overlay"] = metadata
        generated += 1
    if tracker is not None or generated > 0:
        temporary = rollout_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(rollout_path)
    return generated, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deferred online-eval visualizations.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-name", default="target_crop_action_trajectory_overlays")
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--camera-offset-body", nargs=3, type=float, default=[0.46, 0.0, 0.0])
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--trajectory-3d",
        dest="generate_trajectory_3d",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-trajectory-3d",
        dest="generate_trajectory_3d",
        action="store_false",
    )
    parser.add_argument("--replay-tracker-boxes", action="store_true")
    parser.add_argument("--tracker-checkpoint", type=Path)
    parser.add_argument("--tracker-device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker = None
    if args.replay_tracker_boxes:
        if args.tracker_checkpoint is None:
            raise ValueError("--tracker-checkpoint is required with --replay-tracker-boxes")
        tracker = SquareTracker(args.tracker_checkpoint, args.tracker_device)
    trajectory_dirs = list(_trajectory_dirs(args.root))
    total_generated = 0
    total_skipped = 0
    for index, trajectory_dir in enumerate(trajectory_dirs, start=1):
        generated, skipped = postprocess_trajectory(
            trajectory_dir,
            output_name=args.output_name,
            fov_deg=args.fov_deg,
            camera_offset_body=args.camera_offset_body,
            dt=args.dt,
            overwrite=args.overwrite,
            tracker=tracker,
            generate_trajectory_3d=args.generate_trajectory_3d,
        )
        total_generated += generated
        total_skipped += skipped
        if generated or skipped or index % 100 == 0 or index == len(trajectory_dirs):
            print(
                f"[postprocess] {index}/{len(trajectory_dirs)} {trajectory_dir}: "
                f"generated={generated} skipped={skipped}",
                flush=True,
            )
    print(
        f"[postprocess] complete trajectories={len(trajectory_dirs)} "
        f"generated={total_generated} skipped={total_skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
