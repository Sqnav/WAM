from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from eval.compare_gt_attention import _gt_box_xywh
from tracking.runtime import SquareTracker


TRACKER_ORANGE_BGR = (0, 165, 255)


def _draw_grid(image: np.ndarray, grid_hw: tuple[int, int]) -> None:
    height, width = image.shape[:2]
    grid_h, grid_w = grid_hw
    for column in range(1, grid_w):
        x = int(round(column * width / grid_w))
        cv2.line(image, (x, 0), (x, height - 1), (115, 115, 115), 1, cv2.LINE_AA)
    for row in range(1, grid_h):
        y = int(round(row * height / grid_h))
        cv2.line(image, (0, y), (width - 1, y), (115, 115, 115), 1, cv2.LINE_AA)


def _draw_box(image: np.ndarray, bbox_xywh: Sequence[float]) -> None:
    x, y, width, height = (int(round(float(value))) for value in bbox_xywh)
    cv2.rectangle(image, (x, y), (x + width, y + height), TRACKER_ORANGE_BGR, 3)


def _direct_area_target(response: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    response = np.maximum(np.asarray(response, dtype=np.float32), 0.0)
    grid_h, grid_w = grid_hw
    target = cv2.resize(response, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    target = np.maximum(target, 0.0)
    return target / max(float(target.sum()), 1.0e-8)


def _target_overlay(
    rgb_bgr: np.ndarray,
    target: np.ndarray,
    tracker_bbox_xywh: Sequence[float],
    alpha: float = 0.48,
) -> np.ndarray:
    height, width = rgb_bgr.shape[:2]
    display = target - float(target.min())
    display /= max(float(display.max()), 1.0e-8)
    # Nearest-neighbor rendering preserves the exact spatial support of every loss token.
    display = cv2.resize(display, (width, height), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(np.uint8(np.clip(display, 0.0, 1.0) * 255), cv2.COLORMAP_VIRIDIS)
    canvas = cv2.addWeighted(rgb_bgr, 1.0 - alpha, colored, alpha, 0.0)
    _draw_grid(canvas, target.shape)
    _draw_box(canvas, tracker_bbox_xywh)
    peak_row, peak_column = np.unravel_index(int(target.argmax()), target.shape)
    peak_x = int(round((peak_column + 0.5) * width / target.shape[1]))
    peak_y = int(round((peak_row + 0.5) * height / target.shape[0]))
    cv2.drawMarker(canvas, (peak_x, peak_y), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.rectangle(canvas, (0, 0), (width, 36), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        f"direct Tracker supervision {target.shape[0]}x{target.shape[1]}  peak=({peak_x},{peak_y})",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _target_values_panel(target: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    height, width = output_hw
    display = target - float(target.min())
    display /= max(float(display.max()), 1.0e-8)
    display = cv2.resize(display, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.applyColorMap(np.uint8(np.clip(display, 0.0, 1.0) * 255), cv2.COLORMAP_VIRIDIS)
    _draw_grid(canvas, target.shape)
    cell_h = height / target.shape[0]
    cell_w = width / target.shape[1]
    for row in range(target.shape[0]):
        for column in range(target.shape[1]):
            value = float(target[row, column])
            if value < 1.0e-4:
                label = "0"
            elif value < 0.01:
                label = f"{value:.3f}"
            else:
                label = f"{value:.2f}"
            x = int(round((column + 0.5) * cell_w))
            y = int(round((row + 0.5) * cell_h))
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0]
            cv2.putText(
                canvas,
                label,
                (x - size[0] // 2, y + size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.rectangle(canvas, (0, 0), (width, 36), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        "exact normalized probability used by KL",
        (8, 25),
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
    init_bbox = _gt_box_xywh(step)
    if init_bbox is None:
        raise ValueError(f"frame {args.frame} has no valid GT initialization box")

    rgb_path = args.trajectory / "rgb" / f"frame_{args.frame:05d}.png"
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    tracker = SquareTracker(args.tracker_checkpoint, args.device)
    tracker.initialize(rgb, init_bbox)
    result = tracker.track(rgb)
    tracker_bbox = [float(value) for value in result["bbox"]]
    response = np.asarray(result["response"], dtype=np.float32)
    target = _direct_area_target(response, (args.grid_size, args.grid_size))

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = _target_overlay(rgb_bgr, target, tracker_bbox)
    values = _target_values_panel(target, rgb.shape[:2])
    comparison = np.concatenate([overlay, values], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), comparison)
    cv2.imwrite(str(args.output.with_name(args.output.stem + "_overlay.png")), overlay)
    cv2.imwrite(str(args.output.with_name(args.output.stem + "_values.png")), values)
    np.save(args.output.with_suffix(".npy"), target.astype(np.float32))

    peak_row, peak_column = np.unravel_index(int(target.argmax()), target.shape)
    metadata = {
        "frame": args.frame,
        "pipeline": "Tracker native response -> search geometry alignment to full RGB -> direct AREA pool to 7x7 -> sum normalization",
        "visualization_resize": "nearest-neighbor from 7x7 to RGB size; no bilinear smoothing",
        "rgb_hw": list(rgb.shape[:2]),
        "target_hw": list(target.shape),
        "tracker_bbox_xywh": tracker_bbox,
        "target_peak_row_col": [int(peak_row), int(peak_column)],
        "target_peak_xy_rgb": [
            (float(peak_column) + 0.5) * rgb.shape[1] / target.shape[1],
            (float(peak_row) + 0.5) * rgb.shape[0] / target.shape[0],
        ],
        "target_probability": target.tolist(),
        "target_sum": float(target.sum()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize direct area-pooled Tracker supervision.")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--tracker-checkpoint", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
