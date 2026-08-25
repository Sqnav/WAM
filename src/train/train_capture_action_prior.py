from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.action_mapping import clamp_physical_action_delta, physical_action_to_norm
from model.capture_action_prior import (
    CAPTURE_ACTION_PRIOR_VERSION,
    CaptureActionPrior,
    capture_action_prior_loss,
    featurize_capture_action_context,
)


def _xyz(value: object) -> list[float]:
    row = value if isinstance(value, dict) else {}
    return [float(row.get(axis, 0.0)) for axis in ("x", "y", "z")]


def _normalized_tracker_row(frame: dict, image_h: int, image_w: int) -> tuple[list[float], bool]:
    box = frame.get("bbox_xywh")
    valid = isinstance(box, list) and len(box) == 4
    if valid:
        x, y, width, height = (float(v) for v in box)
        valid = width > 0.0 and height > 0.0
    else:
        x = y = width = height = 0.0
    return [
        (x + 0.5 * width) / max(float(image_w), 1.0),
        (y + 0.5 * height) / max(float(image_h), 1.0),
        width / max(float(image_w), 1.0),
        height / max(float(image_h), 1.0),
        float(frame.get("confidence", 0.0)),
    ], bool(valid)


def build_split(
    dataset_root: Path,
    tracker_cache_root: Path,
    scenes: Iterable[str],
    trajectory_start: int,
    trajectory_end: int,
    history_length: int,
    max_vel: float,
    max_yaw_rate: float,
    max_speed_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    features, targets = [], []
    skipped = []
    for scene in scenes:
        for trajectory_index in range(trajectory_start, trajectory_end + 1):
            name = f"trajectory_{trajectory_index:04d}"
            trajectory_path = dataset_root / scene / name / "uav_trajectory.json"
            tracker_path = tracker_cache_root / scene / name / "summary.json"
            if not trajectory_path.is_file() or not tracker_path.is_file():
                raise FileNotFoundError(f"Missing prior-training input: {scene}/{name}")
            trajectory_json = json.loads(trajectory_path.read_text(encoding="utf-8"))
            tracker_json = json.loads(tracker_path.read_text(encoding="utf-8"))
            trajectory = trajectory_json.get("trajectory", [])
            tracker_frames = tracker_json.get("frames", [])
            if not isinstance(trajectory, list) or len(trajectory) < 2:
                skipped.append(f"{scene}/{name}:missing_action_trajectory")
                continue
            if not isinstance(tracker_frames, list):
                raise ValueError(f"Invalid Tracker frames for {scene}/{name}")
            usable_steps = min(len(trajectory), len(tracker_frames))
            if usable_steps < 2:
                skipped.append(f"{scene}/{name}:insufficient_common_frames")
                continue
            if len(trajectory) != len(tracker_frames):
                print(
                    f"[prior] truncate {scene}/{name}: trajectory={len(trajectory)} "
                    f"tracker={len(tracker_frames)} usable={usable_steps}",
                    flush=True,
                )
            trajectory = trajectory[:usable_steps]
            tracker_frames = tracker_frames[:usable_steps]
            image_h, image_w = (int(v) for v in tracker_json.get("image_size", [640, 640]))
            rows_valid = [
                _normalized_tracker_row(frame, image_h, image_w) for frame in tracker_frames
            ]
            rows = [item[0] for item in rows_valid]
            valid = [item[1] for item in rows_valid]
            physical = []
            for frame in trajectory:
                physical.append(
                    _xyz(frame.get("body_frame_delta"))
                    + [float(frame.get("body_frame_yaw_delta", 0.0) or 0.0)]
                )
            normalized = physical_action_to_norm(
                clamp_physical_action_delta(physical, max_speed_norm=max_speed_norm),
                max_vel=max_vel,
                max_yaw_rate=max_yaw_rate,
            )
            for step in range(len(trajectory)):
                first = max(0, step - history_length + 1)
                history_rows = rows[first:step]
                history_valid = valid[first:step]
                padding = history_length - 1 - len(history_rows)
                previous_history = torch.tensor(
                    [[0.0] * 5] * padding + history_rows, dtype=torch.float32
                )[None]
                previous_valid = torch.tensor(
                    [False] * padding + history_valid, dtype=torch.bool
                )[None]
                current = torch.tensor(rows[step][:4], dtype=torch.float32)[None]
                confidence = torch.tensor([rows[step][4]], dtype=torch.float32)
                previous_action = torch.tensor(
                    normalized[step - 1] if step > 0 else np.zeros(4), dtype=torch.float32
                )[None]
                feature = featurize_capture_action_context(
                    previous_history,
                    previous_valid,
                    current,
                    confidence,
                    previous_action,
                )
                features.append(feature.squeeze(0))
                targets.append(torch.tensor(normalized[step], dtype=torch.float32))
    if not features:
        raise ValueError("CaptureActionPrior training split has no valid action samples.")
    print(
        f"[prior] valid_samples={len(features)} skipped_trajectories={len(skipped)}",
        flush=True,
    )
    for item in skipped:
        print(f"[prior-skip] {item}", flush=True)
    return torch.stack(features), torch.stack(targets)


@torch.no_grad()
def evaluate(model: CaptureActionPrior, loader: DataLoader, device: torch.device) -> dict:
    predictions, targets = [], []
    for features, target in loader:
        prediction = model(features.to(device))["mean"].cpu()
        predictions.append(prediction)
        targets.append(target)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    error = prediction - target
    mse = error.square().mean(dim=0)
    mae = error.abs().mean(dim=0)
    correlations = []
    for dim in range(4):
        x = prediction[:, dim] - prediction[:, dim].mean()
        y = target[:, dim] - target[:, dim].mean()
        correlations.append(float((x * y).mean() / (x.std() * y.std()).clamp_min(1.0e-8)))
    return {
        "mse": [float(v) for v in mse],
        "mae": [float(v) for v in mae],
        "correlation": correlations,
        "mean_mse": float(mse.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the frozen CaptureActionPrior.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tracker-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene-list",
        default=",".join(f"City_{index}" for index in range(1, 28)),
    )
    parser.add_argument("--train-start", type=int, default=1)
    parser.add_argument("--train-end", type=int, default=450)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--max-yaw-rate", type=float, default=15.0)
    parser.add_argument("--max-speed-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scenes = tuple(
        item.strip() for item in str(args.scene_list).split(",") if item.strip()
    )
    if not scenes:
        raise ValueError("--scene-list must contain at least one city.")
    if args.train_start < 1 or args.train_end < args.train_start:
        raise ValueError("Invalid CaptureActionPrior training range.")
    print("[prior] loading training split", flush=True)
    train_features, train_targets = build_split(
        args.dataset_root,
        args.tracker_cache_root,
        scenes,
        args.train_start,
        args.train_end,
        args.history_length,
        args.max_vel,
        args.max_yaw_rate,
        args.max_speed_norm,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = CaptureActionPrior(
        history_length=args.history_length,
        hidden_dim=args.hidden_dim,
    )
    model.set_feature_statistics(train_features.mean(0), train_features.std(0))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    final_train_loss = math.inf
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = capture_action_prior_loss(model(features), target)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(losses["loss"].detach()) * features.size(0)
            count += features.size(0)
        scheduler.step()
        final_train_loss = total_loss / max(count, 1)
        print(
            f"[prior] epoch={epoch:03d} train_loss={final_train_loss:.6f}",
            flush=True,
        )
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "version": CAPTURE_ACTION_PRIOR_VERSION,
                "history_length": args.history_length,
                "hidden_dim": args.hidden_dim,
                "train_scenes": list(scenes),
                "train_range": [args.train_start, args.train_end],
                "validation_disabled": True,
                "train_samples": int(train_features.shape[0]),
                "seed": args.seed,
            },
            "model": best_state,
            "training": {"final_loss": final_train_loss},
        },
        args.output,
    )
    print(
        json.dumps(
            {"output": str(args.output), "final_train_loss": final_train_loss},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
