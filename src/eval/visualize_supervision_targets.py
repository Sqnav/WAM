from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from eval.compare_gt_attention import GT_GREEN_BGR, _canonical_gt_target, _gt_box_xywh
from eval.compare_tracker_attention import _draw_map_panel


TRACKER_ORANGE_BGR = (0, 165, 255)


def _draw_grid(image: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    grid_h, grid_w = grid_hw
    for column in range(1, grid_w):
        x = int(round(column * width / grid_w))
        cv2.line(canvas, (x, 0), (x, height - 1), (105, 105, 105), 1, cv2.LINE_AA)
    for row in range(1, grid_h):
        y = int(round(row * height / grid_h))
        cv2.line(canvas, (0, y), (width - 1, y), (105, 105, 105), 1, cv2.LINE_AA)
    return canvas


def _draw_box(
    image: np.ndarray,
    bbox_xywh: Optional[Sequence[float]],
    color: tuple[int, int, int],
    label: str,
) -> None:
    if bbox_xywh is None or len(bbox_xywh) < 4:
        return
    x, y, width, height = (int(round(float(value))) for value in bbox_xywh[:4])
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 3)
    cv2.putText(
        image,
        label,
        (x, max(y - 7, 42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _rgb_panel(
    rgb_bgr: np.ndarray,
    gt_bbox_xywh: Sequence[float],
    tracker_bbox_xywh: Optional[Sequence[float]],
    grid_hw: tuple[int, int],
) -> np.ndarray:
    canvas = _draw_grid(rgb_bgr, grid_hw)
    _draw_box(canvas, gt_bbox_xywh, GT_GREEN_BGR, "GT")
    _draw_box(canvas, tracker_bbox_xywh, TRACKER_ORANGE_BGR, "Tracker")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        "RGB + GT/Tracker boxes",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def run(args: argparse.Namespace) -> None:
    rollout = json.loads((args.trajectory / "online_rollout.json").read_text(encoding="utf-8"))
    steps = rollout.get("steps", [])
    step = next(
        (value for position, value in enumerate(steps) if int(value.get("step", position)) == args.frame),
        None,
    )
    if step is None:
        raise ValueError(f"frame {args.frame} is absent from online_rollout.json")
    gt_bbox = _gt_box_xywh(step)
    if gt_bbox is None:
        raise ValueError(f"frame {args.frame} has no valid GT box")

    comparison_dir = args.trajectory / args.comparison_dir
    summary = json.loads((comparison_dir / "summary.json").read_text(encoding="utf-8"))
    row = next((value for value in summary.get("rows", []) if int(value["frame"]) == args.frame), None)
    if row is None:
        raise ValueError(f"frame {args.frame} is absent from Tracker comparison summary")
    grid_hw = tuple(int(value) for value in row["attention_grid_hw"])
    tracker_bbox = row.get("tracker_bbox_xywh")

    stem = f"frame_{args.frame:05d}.png"
    rgb_bgr = cv2.imread(str(args.trajectory / "rgb" / stem), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(args.trajectory / "rgb" / stem)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    gt_target = _canonical_gt_target(gt_bbox, rgb.shape[:2], grid_hw, args.sigma)
    gt_panel = _draw_map_panel(
        rgb,
        gt_target,
        f"GT supervision used by loss ({grid_hw[0]}x{grid_hw[1]})",
        gt_bbox,
        GT_GREEN_BGR,
    )
    gt_panel = _draw_grid(gt_panel, grid_hw)

    tracker_panel = cv2.imread(str(comparison_dir / "tracker_response" / stem), cv2.IMREAD_COLOR)
    model_panel = cv2.imread(str(comparison_dir / "all_queries" / stem), cv2.IMREAD_COLOR)
    if tracker_panel is None or model_panel is None:
        raise FileNotFoundError("Tracker/model standalone comparison panels are missing")
    tracker_panel = _draw_grid(tracker_panel, grid_hw)
    model_panel = _draw_grid(model_panel, grid_hw)
    rgb_panel = _rgb_panel(rgb_bgr, gt_bbox, tracker_bbox, grid_hw)

    output = np.concatenate(
        [np.concatenate([rgb_panel, gt_panel], axis=1), np.concatenate([tracker_panel, model_panel], axis=1)],
        axis=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), output)

    metadata = {
        "frame": args.frame,
        "grid_hw": list(grid_hw),
        "gt_target_pipeline": f"GT bbox center -> canonical 64x64 sigma={args.sigma:.2f} -> bilinear {grid_hw[0]}x{grid_hw[1]} -> normalize",
        "tracker_target_pipeline": f"Tracker native response -> area 64x64 -> bilinear {grid_hw[0]}x{grid_hw[1]} -> normalize",
        "model_attention": "per-head/query normalization over first-frame tokens, then all-head/all-query mean",
        "gt_bbox_xywh": [float(value) for value in gt_bbox],
        "tracker_bbox_xywh": tracker_bbox,
        "gt_target_peak_xy": [
            (int(np.argmax(gt_target)) % grid_hw[1] + 0.5) * rgb.shape[1] / grid_hw[1],
            (int(np.argmax(gt_target)) // grid_hw[1] + 0.5) * rgb.shape[0] / grid_hw[0],
        ],
        "tracker_target_peak_xy": row["tracker_response_peak_xy"],
        "model_attention_peak_xy": row["all_queries_peak_xy"],
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GT, Tracker, and model attention at the loss grid scale.")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--comparison-dir", default="attention_tracker_comparisons")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
