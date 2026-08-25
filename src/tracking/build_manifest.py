from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

def atomic_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_real_boxes(path: Path, frame_count: int) -> Optional[List[Optional[List[float]]]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("boxes_xywh", payload.get("frames")) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"Unsupported box annotation format: {path}")
    boxes: List[Optional[List[float]]] = [None] * frame_count
    for index, value in enumerate(values[:frame_count]):
        if isinstance(value, dict):
            frame_index = int(value.get("frame_idx", index))
            value = value.get("bbox_xywh", value.get("bbox"))
        else:
            frame_index = index
        if value is not None and len(value) == 4 and 0 <= frame_index < frame_count:
            boxes[frame_index] = [float(v) for v in value]
    return boxes


def square_box(box: Sequence[float]) -> List[float]:
    x, y, width, height = (float(value) for value in box)
    side = max(width, height)
    return [x + 0.5 * (width - side), y + 0.5 * (height - side), side, side]


def relative_xyz(step: Dict) -> Optional[Sequence[float]]:
    value = step.get("target_position_in_body_frame", step.get("relative_target_body"))
    if isinstance(value, dict):
        return [value.get("x"), value.get("y"), value.get("z")]
    if isinstance(value, list) and len(value) >= 3:
        return value[:3]
    return None


def projected_box(
    xyz: Sequence[float],
    image_width: int,
    image_height: int,
    fov_deg: float,
    camera_offset: Sequence[float],
    target_width: float,
    target_height: float,
    minimum_pixels: float,
    square_boxes: bool,
) -> Optional[List[float]]:
    forward = float(xyz[0]) - float(camera_offset[0])
    right = float(xyz[1]) - float(camera_offset[1])
    down = float(xyz[2]) - float(camera_offset[2])
    if forward <= 1e-3:
        return None
    focal_x = image_width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    focal_y = focal_x
    center_x = image_width / 2.0 + focal_x * right / forward
    center_y = image_height / 2.0 + focal_y * down / forward
    width = max(focal_x * target_width / forward, minimum_pixels)
    height = max(focal_y * target_height / forward, minimum_pixels)
    if square_boxes:
        width = height = max(width, height)
    x1, y1 = center_x - width / 2.0, center_y - height / 2.0
    x2, y2 = center_x + width / 2.0, center_y + height / 2.0
    clipped = [max(x1, 0.0), max(y1, 0.0), min(x2, image_width), min(y2, image_height)]
    if clipped[2] - clipped[0] < 2 or clipped[3] - clipped[1] < 2:
        return None
    return [clipped[0], clipped[1], clipped[2] - clipped[0], clipped[3] - clipped[1]]


def build_record(trajectory: Path, args: argparse.Namespace) -> Optional[Dict]:
    frames = sorted((trajectory / "rgb").glob("frame_*.png"))
    trajectory_file = trajectory / "uav_trajectory.json"
    if len(frames) < 2 or not trajectory_file.exists():
        return None
    width, height = args.image_width, args.image_height
    real_boxes = read_real_boxes(trajectory / args.annotation_name, len(frames))
    if real_boxes is not None:
        boxes = (
            [None if box is None else square_box(box) for box in real_boxes]
            if args.box_shape == "square"
            else real_boxes
        )
        source = "real"
    else:
        payload = json.loads(trajectory_file.read_text(encoding="utf-8"))
        steps = payload.get("trajectory", [])
        boxes = []
        for index in range(len(frames)):
            xyz = relative_xyz(steps[index]) if index < len(steps) else None
            boxes.append(
                None if xyz is None else projected_box(
                    xyz, width, height, args.fov_deg, args.camera_offset,
                    args.target_width_m, args.target_height_m, args.minimum_pixels,
                    args.box_shape == "square",
                )
            )
        source = "projected_weak"
    if sum(box is not None for box in boxes) < 2:
        return None
    return {
        "trajectory": str(trajectory.resolve()),
        "frames": [str(frame.relative_to(trajectory)) for frame in frames],
        "boxes_xywh": boxes,
        "annotation_source": source,
        "image_size": [height, width],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UAV tracking train/validation/test metadata")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotation-name", default="target_boxes.json")
    parser.add_argument("--scenes", nargs="+", default=["City_1", "City_2", "City_3"])
    parser.add_argument("--trajectory-start", type=int, default=1)
    parser.add_argument("--trajectory-end", type=int, default=450)
    parser.add_argument("--val-scenes", nargs="+", default=None)
    parser.add_argument("--val-trajectory-start", type=int, default=None)
    parser.add_argument("--val-trajectory-end", type=int, default=None)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=640)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--camera-offset", type=float, nargs=3, default=[0.46, 0.0, 0.0])
    parser.add_argument("--target-width-m", type=float, default=0.8)
    parser.add_argument("--target-height-m", type=float, default=0.35)
    parser.add_argument("--minimum-pixels", type=float, default=6.0)
    parser.add_argument("--box-shape", choices=["square", "rectangle"], default="square")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    if args.trajectory_start < 1 or args.trajectory_end < args.trajectory_start:
        raise ValueError("trajectory range must satisfy 1 <= start <= end")
    explicit_val = any(
        value is not None
        for value in (args.val_scenes, args.val_trajectory_start, args.val_trajectory_end)
    )
    if explicit_val and (
        not args.val_scenes
        or args.val_trajectory_start is None
        or args.val_trajectory_end is None
    ):
        raise ValueError(
            "--val-scenes, --val-trajectory-start and --val-trajectory-end must be provided together"
        )
    if args.train_only and explicit_val:
        raise ValueError("--train-only cannot be combined with an explicit validation split")
    if explicit_val and (
        args.val_trajectory_start < 1
        or args.val_trajectory_end < args.val_trajectory_start
    ):
        raise ValueError("validation trajectory range must satisfy 1 <= start <= end")
    train_trajectories = [
        args.dataset_root / scene / f"trajectory_{index:04d}"
        for scene in args.scenes
        for index in range(args.trajectory_start, args.trajectory_end + 1)
        if (args.dataset_root / scene / f"trajectory_{index:04d}").is_dir()
    ]
    val_trajectories = []
    if explicit_val:
        val_trajectories = [
            args.dataset_root / scene / f"trajectory_{index:04d}"
            for scene in args.val_scenes
            for index in range(args.val_trajectory_start, args.val_trajectory_end + 1)
            if (args.dataset_root / scene / f"trajectory_{index:04d}").is_dir()
        ]
        overlap = sorted(set(train_trajectories).intersection(val_trajectories))
        if overlap:
            raise ValueError(f"Training and validation trajectories overlap: {overlap[:8]}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        train_candidates = pool.map(lambda path: build_record(path, args), train_trajectories)
        train_records = [record for record in train_candidates if record is not None]
        val_candidates = pool.map(lambda path: build_record(path, args), val_trajectories)
        val_records = [record for record in val_candidates if record is not None]
    if len(train_records) < 3:
        raise RuntimeError(
            f"Only found {len(train_records)} valid training trajectories under {args.dataset_root}"
        )
    random.Random(args.seed).shuffle(train_records)
    if args.train_only:
        records = train_records
        train_split, val_split, test_split = train_records, [], []
    elif explicit_val:
        if not val_records:
            raise RuntimeError(f"No valid validation trajectories found under {args.dataset_root}")
        records = train_records + val_records
        train_split, val_split, test_split = train_records, val_records, []
    else:
        records = train_records
        train_end = max(int(len(records) * 0.8), 1)
        val_end = max(int(len(records) * 0.9), train_end + 1)
        train_split, val_split, test_split = (
            records[:train_end], records[train_end:val_end], records[val_end:]
        )
    payload = {
        "format_version": 1,
        "box_shape": args.box_shape,
        "note": "projected_weak boxes must be visually audited before final training",
        "train": train_split,
        "val": val_split,
        "test": test_split,
        "counts": {
            "trajectories": len(records),
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_split),
            "real": sum(r["annotation_source"] == "real" for r in records),
            "projected_weak": sum(r["annotation_source"] == "projected_weak" for r in records),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload["counts"], ensure_ascii=False))
    print(f"manifest={args.output.resolve()}")


if __name__ == "__main__":
    main()
