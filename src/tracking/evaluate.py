from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from tracking.data import MEAN, STD, _image_tensor, crop_target
from tracking.model import UAVTracker

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def crop_search(image: torch.Tensor, box: Sequence[float], factor: float, output_size: int):
    x, y, width, height = map(float, box)
    side = max(int(math.ceil(math.sqrt(max(width * height, 1.0)) * factor)), 2)
    center_x, center_y = x + width / 2, y + height / 2
    x1, y1 = int(math.floor(center_x - side / 2)), int(math.floor(center_y - side / 2))
    image_height, image_width = image.shape[-2:]
    left, top = max(-x1, 0), max(-y1, 0)
    right, bottom = max(x1 + side - image_width, 0), max(y1 + side - image_height, 0)
    padded = F.pad(image, (left, right, top, bottom), value=0.0)
    crop = padded[:, y1 + top : y1 + top + side, x1 + left : x1 + left + side]
    crop = F.interpolate(crop.unsqueeze(0), (output_size, output_size), mode="bilinear", align_corners=False)[0]
    return (crop - MEAN) / STD, (x1, y1, side)


def decode_peak(outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
    score = outputs["center_logits"].sigmoid()[0, 0]
    height, width = score.shape
    index = int(score.argmax())
    row, column = index // width, index % width
    offset = outputs["offset"][0, :, row, column]
    size = outputs["size"][0, :, row, column]
    box = torch.stack([(column + offset[0]) / width, (row + offset[1]) / height, size[0], size[1]])
    return box, float(score[row, column])


def map_box(box: torch.Tensor, geometry, image_hw) -> List[float]:
    x1, y1, side = geometry
    cx, cy, width, height = [float(v) for v in box]
    width, height = width * side, height * side
    x, y = cx * side + x1 - width / 2, cy * side + y1 - height / 2
    image_height, image_width = image_hw
    left, top = min(max(x, 0.0), image_width - 1.0), min(max(y, 0.0), image_height - 1.0)
    right = min(max(x + width, left + 1.0), float(image_width))
    bottom = min(max(y + height, top + 1.0), float(image_height))
    return [left, top, right - left, bottom - top]


def box_metrics(prediction: Sequence[float], target: Sequence[float]) -> Tuple[float, float, float]:
    px, py, pw, ph = prediction
    tx, ty, tw, th = target
    left, top = max(px, tx), max(py, ty)
    right, bottom = min(px + pw, tx + tw), min(py + ph, ty + th)
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    union = max(pw * ph + tw * th - intersection, 1e-6)
    center_error = math.hypot(px + pw / 2 - tx - tw / 2, py + ph / 2 - ty - th / 2)
    normalized_error = center_error / max(math.sqrt(tw * th), 1e-6)
    return intersection / union, center_error, normalized_error


def draw_boxes(
    image_path: Path,
    output_path: Path,
    target: Sequence[float],
    prediction: Sequence[float],
    iou: float,
    confidence: float,
    reference: Sequence[float] | None = None,
    reference_iou: float | None = None,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes = [(target, "#22c55e", "target"), (prediction, "#ef4444", "new model")]
    if reference is not None:
        boxes.append((reference, "#3b82f6", "original model"))
    for box, color, label in boxes:
        x, y, width, height = box
        draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
        draw.text((x + 3, max(y - 15, 2)), label, fill=color, stroke_width=2, stroke_fill="black")
    caption = f"new IoU {iou:.3f}  confidence {confidence:.3f}"
    if reference_iou is not None:
        caption += f"  original IoU {reference_iou:.3f}"
    draw.rectangle((5, 5, min(470, image.width - 5), 28), fill="black")
    draw.text((10, 9), caption, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def evaluate_sequence(
    model,
    record: Dict,
    device: torch.device,
    visualization_root: Path | None = None,
    reference_tracker=None,
) -> Dict:
    root = Path(record["trajectory"])
    frames, targets = record["frames"], record["boxes_xywh"]
    first = next((index for index, box in enumerate(targets) if box is not None), None)
    if first is None:
        return {"trajectory": str(root), "frames": 0}
    image = _image_tensor(root / frames[first])
    template, _ = crop_target(image, targets[first], 2.0, 128)
    template = template.unsqueeze(0).to(device)
    state = list(targets[first])
    reference_state = list(targets[first])
    if reference_tracker is not None:
        first_array = image.mul(255).byte().permute(1, 2, 0).numpy()
        reference_tracker.initialize(first_array, reference_state)
    sequence_vis = None if visualization_root is None else visualization_root / root.parent.name / root.name
    if sequence_vis is not None:
        draw_boxes(
            root / frames[first], sequence_vis / f"frame_{first:05d}.png",
            state, state, 1.0, 1.0, reference_state if reference_tracker is not None else None, 1.0,
        )
    ious, errors, normalized_errors, confidences = [], [], [], []
    with torch.inference_mode():
        for index in range(first + 1, len(frames)):
            if targets[index] is None:
                continue
            image = _image_tensor(root / frames[index])
            search, geometry = crop_search(image, state, 4.0, 256)
            outputs = model(template, search.unsqueeze(0).to(device))
            normalized, confidence = decode_peak(outputs)
            state = map_box(normalized.cpu(), geometry, image.shape[-2:])
            iou, error, normalized_error = box_metrics(state, targets[index])
            reference_iou = None
            if reference_tracker is not None:
                image_array = image.mul(255).byte().permute(1, 2, 0).numpy()
                reference_result = reference_tracker.track(image_array)
                reference_state = [float(value) for value in reference_result["bbox"]]
                reference_iou = box_metrics(reference_state, targets[index])[0]
            if sequence_vis is not None:
                draw_boxes(
                    root / frames[index],
                    sequence_vis / f"frame_{index:05d}.png",
                    targets[index],
                    state,
                    iou,
                    confidence,
                    reference_state if reference_tracker is not None else None,
                    reference_iou,
                )
            ious.append(iou)
            errors.append(error)
            normalized_errors.append(normalized_error)
            confidences.append(confidence)
    return {
        "trajectory": str(root),
        "frames": len(ious),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "success_0.5": float(np.mean(np.asarray(ious) >= 0.5)) if ious else 0.0,
        "precision_20px": float(np.mean(np.asarray(errors) <= 20.0)) if errors else 0.0,
        "normalized_precision_0.2": float(np.mean(np.asarray(normalized_errors) <= 0.2)) if errors else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "ious": ious,
        "center_errors": errors,
        "normalized_errors": normalized_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline closed-loop evaluation for the UAV tracker")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--trajectory", help="Only evaluate one trajectory name, for example trajectory_0451")
    parser.add_argument("--reference-tracker", action="store_true", help="Draw the original tracker in blue")
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "model/ortrack/deit_tiny_patch16_224/ORTrack_ep0300.pth.tar",
    )
    parser.add_argument("--reference-root", type=Path, default=PROJECT_ROOT / "third_party/ortrack")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    model = UAVTracker(
        backbone=saved_args.get("backbone", "deit_tiny_patch16_224"),
        pretrained=False,
        square_boxes=bool(saved_args.get("square_boxes", False)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    reference_tracker = None
    if args.reference_tracker:
        from eval.ortrack_target_heatmap import ORTrack640

        reference_tracker = ORTrack640(
            args.reference_root.resolve(), args.reference_checkpoint.resolve(), "deit_tiny_patch16_224", "cuda"
        )
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = payload.get("train", []) + payload.get("val", []) + payload.get("test", [])
    if args.trajectory:
        records = [record for record in records if Path(record["trajectory"]).name == args.trajectory]
        if not records:
            raise ValueError(f"Trajectory {args.trajectory!r} was not found in the manifest")
    results = []
    for index, record in enumerate(records, 1):
        result = evaluate_sequence(model, record, device, args.visualization_dir, reference_tracker)
        results.append(result)
        print(f"eval {index:03d}/{len(records):03d} {Path(record['trajectory']).name} iou={result.get('mean_iou', 0):.4f}", flush=True)
    all_ious = np.concatenate([r["ious"] for r in results if r.get("ious")])
    all_errors = np.concatenate([r["center_errors"] for r in results if r.get("center_errors")])
    all_normalized = np.concatenate([r["normalized_errors"] for r in results if r.get("normalized_errors")])
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "trajectories": len(results),
        "evaluated_frames": int(all_ious.size),
        "success_auc": float(all_ious.mean()),
        "success_0.5": float((all_ious >= 0.5).mean()),
        "precision_20px": float((all_errors <= 20.0).mean()),
        "normalized_precision_0.2": float((all_normalized <= 0.2).mean()),
    }
    output = {"summary": summary, "sequences": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
