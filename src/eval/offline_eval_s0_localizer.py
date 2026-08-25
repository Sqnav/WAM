from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from data.teacher_dataset_builder import build_records
from model.config import ModelConfig, migrate_legacy_config
from model.model import S0LocalizationModel
from tracking.data import _image_tensor, crop_target
from tracking.evaluate import box_metrics, crop_search


def _normalized_cxcywh_to_xywh(
    box: torch.Tensor, image_height: int, image_width: int
) -> list[float]:
    cx, cy, width, height = [float(value) for value in box.float().cpu()]
    x1 = (cx - 0.5 * width) * image_width
    y1 = (cy - 0.5 * height) * image_height
    x2 = (cx + 0.5 * width) * image_width
    y2 = (cy + 0.5 * height) * image_height
    x1 = min(max(x1, 0.0), float(image_width - 1))
    y1 = min(max(y1, 0.0), float(image_height - 1))
    x2 = min(max(x2, x1 + 1.0), float(image_width))
    y2 = min(max(y2, y1 + 1.0), float(image_height))
    return [x1, y1, x2 - x1, y2 - y1]


def _giou(prediction: Sequence[float], target: Sequence[float]) -> float:
    px, py, pw, ph = map(float, prediction)
    tx, ty, tw, th = map(float, target)
    inter_left, inter_top = max(px, tx), max(py, ty)
    inter_right, inter_bottom = min(px + pw, tx + tw), min(py + ph, ty + th)
    intersection = max(inter_right - inter_left, 0.0) * max(inter_bottom - inter_top, 0.0)
    union = max(pw * ph + tw * th - intersection, 1.0e-6)
    enclosing_left, enclosing_top = min(px, tx), min(py, ty)
    enclosing_right, enclosing_bottom = max(px + pw, tx + tw), max(py + ph, ty + th)
    enclosing = max(enclosing_right - enclosing_left, 0.0) * max(
        enclosing_bottom - enclosing_top, 0.0
    )
    return intersection / union - (enclosing - union) / max(enclosing, 1.0e-6)


def _draw_overlay(
    image_path: Path,
    output_path: Path,
    target: Sequence[float],
    prediction: Sequence[float],
    search_geometry: Sequence[float],
    iou: float,
    center_error: float,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    sx, sy, side = map(float, search_geometry)
    draw.rectangle((sx, sy, sx + side, sy + side), outline="#38bdf8", width=2)
    for box, color, label in (
        (target, "#22c55e", "GT"),
        (prediction, "#ef4444", "S0"),
    ):
        x, y, width, height = map(float, box)
        draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
        draw.text(
            (x + 3, max(y - 15, 2)),
            label,
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )
    caption = f"IoU {iou:.3f}  center error {center_error:.1f}px"
    draw.rectangle((5, 5, min(385, image.width - 5), 29), fill="black")
    draw.text((10, 9), caption, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _make_video(frames_dir: Path, output_path: Path, fps: int) -> bool:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"[warning] ffmpeg unavailable, using OpenCV: {error}", file=sys.stderr, flush=True)
        try:
            import cv2

            frame_paths = sorted(frames_dir.glob("frame_*.png"))
            if not frame_paths:
                raise ValueError(f"No overlay frames found in {frames_dir}")
            first = cv2.imread(str(frame_paths[0]))
            if first is None:
                raise ValueError(f"Cannot read overlay frame {frame_paths[0]}")
            height, width = first.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open video writer for {output_path}")
            try:
                for frame_path in frame_paths:
                    frame = cv2.imread(str(frame_path))
                    if frame is None or frame.shape[:2] != (height, width):
                        raise ValueError(f"Invalid overlay frame {frame_path}")
                    writer.write(frame)
            finally:
                writer.release()
        except (ImportError, RuntimeError, ValueError) as fallback_error:
            print(
                f"[warning] OpenCV MP4 generation failed: {fallback_error}",
                file=sys.stderr,
                flush=True,
            )
            return False
    return True


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[S0LocalizationModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Invalid S0 checkpoint: {checkpoint_path}")
    cfg = ModelConfig(**migrate_legacy_config(checkpoint.get("cfg", {}) or {}))
    model = S0LocalizationModel(cfg)
    state = checkpoint["model"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = build_records(args.dataset_root, [args.scene], f"{args.trajectory}-{args.trajectory}")
    if len(records) != 1:
        raise ValueError(
            f"Expected one record for {args.scene}/trajectory_{args.trajectory:04d}, got {len(records)}"
        )
    record = records[0]
    rgb_paths = [Path(path) for path in record["rgb_paths"]]
    targets = record["target_bboxes_xywh"]
    valid = [bool(value[0]) for value in record["target_bbox_valid"]]
    first = next((index for index, is_valid in enumerate(valid) if is_valid), None)
    if first is None:
        raise ValueError("Trajectory contains no valid target boxes.")

    device = torch.device(args.device)
    model, checkpoint = _load_model(args.checkpoint, device)
    first_image = _image_tensor(rgb_paths[first])
    template, _ = crop_target(first_image, targets[first], 2.0, 128)
    template = template.unsqueeze(0).to(device)
    initial_side = max(float(targets[first][2]), float(targets[first][3]))
    state = list(map(float, targets[first]))
    output_frames = args.output_dir / "overlays"
    frame_results: list[dict[str, Any]] = []

    # Frame zero is ground-truth initialization, not a model prediction.
    initial_search = [
        state[0] + 0.5 * state[2] - 2.0 * initial_side,
        state[1] + 0.5 * state[3] - 2.0 * initial_side,
        4.0 * initial_side,
    ]
    _draw_overlay(rgb_paths[first], output_frames / f"frame_{first:05d}.png", state, state, initial_search, 1.0, 0.0)

    amp_dtype = torch.bfloat16 if model.cfg.wan22_torch_dtype.lower() in {"bf16", "bfloat16"} else torch.float16
    with torch.inference_mode():
        for index in range(first + 1, len(rgb_paths)):
            if not valid[index]:
                continue
            image = _image_tensor(rgb_paths[index])
            center_x = state[0] + 0.5 * state[2]
            center_y = state[1] + 0.5 * state[3]
            anchor = [
                center_x - 0.5 * initial_side,
                center_y - 0.5 * initial_side,
                initial_side,
                initial_side,
            ]
            search, geometry = crop_search(image, anchor, 4.0, 256)
            image_size = torch.tensor(
                [[float(image.shape[-2]), float(image.shape[-1])]],
                device=device,
                dtype=torch.float32,
            )
            search_geometry = torch.tensor([geometry], device=device, dtype=torch.float32)
            autocast_enabled = device.type == "cuda" and model.cfg.wan22_torch_dtype.lower() != "float32"
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                output = model(template, search.unsqueeze(0).to(device), search_geometry, image_size)
            state = _normalized_cxcywh_to_xywh(
                output["current_box"][0], image.shape[-2], image.shape[-1]
            )
            iou, center_error, normalized_error = box_metrics(state, targets[index])
            giou = _giou(state, targets[index])
            frame_results.append(
                {
                    "frame_index": index,
                    "rgb_path": str(rgb_paths[index]),
                    "target_xywh": list(map(float, targets[index])),
                    "prediction_xywh": state,
                    "search_geometry_xy_size": list(map(float, geometry)),
                    "iou": iou,
                    "giou": giou,
                    "center_error_px": center_error,
                    "normalized_center_error": normalized_error,
                }
            )
            _draw_overlay(
                rgb_paths[index],
                output_frames / f"frame_{index:05d}.png",
                targets[index],
                state,
                geometry,
                iou,
                center_error,
            )
            print(
                f"offline S0 {index - first:03d}/{len(rgb_paths) - first - 1:03d} "
                f"frame={index:05d} IoU={iou:.3f} center={center_error:.1f}px",
                flush=True,
            )

    ious = np.asarray([item["iou"] for item in frame_results], dtype=np.float64)
    gious = np.asarray([item["giou"] for item in frame_results], dtype=np.float64)
    errors = np.asarray([item["center_error_px"] for item in frame_results], dtype=np.float64)
    normalized_errors = np.asarray(
        [item["normalized_center_error"] for item in frame_results], dtype=np.float64
    )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "scene": args.scene,
        "trajectory": f"trajectory_{args.trajectory:04d}",
        "protocol": "closed_loop_previous_predicted_s0",
        "initialization_frame": first,
        "evaluated_frames": len(frame_results),
        "mean_iou": float(ious.mean()) if ious.size else 0.0,
        "mean_giou": float(gious.mean()) if gious.size else 0.0,
        "success_0.5": float((ious >= 0.5).mean()) if ious.size else 0.0,
        "precision_20px": float((errors <= 20.0).mean()) if errors.size else 0.0,
        "normalized_precision_0.2": float((normalized_errors <= 0.2).mean()) if errors.size else 0.0,
        "mean_center_error_px": float(errors.mean()) if errors.size else 0.0,
        "median_center_error_px": float(np.median(errors)) if errors.size else 0.0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "metrics.json"
    result_path.write_text(
        json.dumps({"summary": summary, "frames": frame_results}, indent=2), encoding="utf-8"
    )
    if not args.no_video:
        summary["video_created"] = _make_video(output_frames, args.output_dir / "overlay.mp4", args.fps)
        result_path.write_text(
            json.dumps({"summary": summary, "frames": frame_results}, indent=2), encoding="utf-8"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop offline evaluation for the standalone S0 localizer")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--scene", default="City_1")
    parser.add_argument("--trajectory", type=int, default=451)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
