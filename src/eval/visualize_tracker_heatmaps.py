from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tracking.runtime import SquareTracker


def _colorize(heatmap: np.ndarray) -> np.ndarray:
    value = np.asarray(heatmap, dtype=np.float32)
    value = value - float(value.min())
    value = value / max(float(value.max()), 1.0e-8)
    bgr = cv2.applyColorMap(np.uint8(value * 255.0), cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _draw_panel(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    bbox: list[float],
    confidence: float,
) -> Image.Image:
    heatmap_rgb = _colorize(heatmap)
    overlay = np.uint8(0.5 * rgb.astype(np.float32) + 0.5 * heatmap_rgb.astype(np.float32))
    left = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(left)
    x, y, width, height = bbox
    draw.rectangle((x, y, x + width, y + height), outline=(255, 50, 40), width=3)
    label = f"Square Tracker confidence={confidence:.4f}"
    draw.text((8, 8), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    right = Image.fromarray(overlay)
    panel = Image.new("RGB", (left.width + right.width, left.height))
    panel.paste(left, (0, 0))
    panel.paste(right, (left.width, 0))
    return panel


def _initial_tracker_bbox(step: dict, rollout_path: Path) -> list[float]:
    bbox = step.get("ortrack_bbox_xywh")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(value) for value in bbox]

    overlay = step.get("target_crop_action_overlay")
    target_crop = overlay.get("target_crop") if isinstance(overlay, dict) else None
    gt_box = target_crop.get("gt_box_xyxy") if isinstance(target_crop, dict) else None
    if isinstance(gt_box, list) and len(gt_box) == 4:
        x1, y1, x2, y2 = (float(value) for value in gt_box)
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2 - x1, y2 - y1]

    raise ValueError(
        f"Missing initial Tracker bbox and GT target crop in {rollout_path}"
    )


def visualize_trajectory(
    trajectory_dir: Path,
    tracker: SquareTracker,
    output_name: str,
) -> int:
    rollout_path = trajectory_dir / "online_rollout.json"
    rgb_paths = sorted((trajectory_dir / "rgb").glob("frame_*.png"))
    if not rollout_path.is_file() or not rgb_paths:
        return 0
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    steps = rollout.get("steps", [])
    frame_count = min(len(rgb_paths), len(steps))
    if frame_count == 0:
        return 0
    init_bbox = _initial_tracker_bbox(steps[0], rollout_path)

    output_dir = trajectory_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    first_rgb = np.asarray(Image.open(rgb_paths[0]).convert("RGB"), dtype=np.uint8)
    tracker.initialize(first_rgb, init_bbox)
    for index, rgb_path in enumerate(rgb_paths[:frame_count]):
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        if index == 0:
            bbox = [float(value) for value in init_bbox]
            confidence = 1.0
            heatmap = np.zeros(rgb.shape[:2], dtype=np.float32)
            x, y, width, height = bbox
            x1, y1 = max(int(round(x)), 0), max(int(round(y)), 0)
            x2 = min(int(round(x + width)), rgb.shape[1])
            y2 = min(int(round(y + height)), rgb.shape[0])
            heatmap[y1:y2, x1:x2] = 1.0
        else:
            result = tracker.track(rgb)
            bbox = [float(value) for value in result["bbox"]]
            confidence = float(result["confidence"])
            heatmap = np.asarray(result["response"], dtype=np.float32)
        panel = _draw_panel(rgb, heatmap, bbox, confidence)
        panel.save(
            output_dir / f"frame_{index:05d}_tracker_heatmap.png",
            compress_level=1,
        )
    return frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay and visualize Square Tracker response heatmaps.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-name", default="tracker_heatmap_visualizations")
    parser.add_argument("--trajectory-start", type=int, default=0)
    parser.add_argument("--trajectory-end", type=int, default=999999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker = SquareTracker(args.checkpoint, args.device)
    trajectories = [
        path
        for path in sorted(args.eval_dir.glob("City_*/trajectory_*"))
        if args.trajectory_start <= int(path.name.rsplit("_", 1)[-1]) <= args.trajectory_end
    ]
    total_frames = 0
    completed = 0
    for index, trajectory_dir in enumerate(trajectories, start=1):
        count = visualize_trajectory(trajectory_dir, tracker, args.output_name)
        if count:
            completed += 1
            total_frames += count
            print(
                f"[{index}/{len(trajectories)}] {trajectory_dir.parent.name}/{trajectory_dir.name}: "
                f"{count} frames",
                flush=True,
            )
    print(f"completed trajectories={completed}, frames={total_frames}", flush=True)


if __name__ == "__main__":
    main()
