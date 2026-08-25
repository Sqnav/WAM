from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model.config import ModelConfig, migrate_legacy_config
from model.model import TeacherWorldModelDiT, migrate_legacy_state_dict_keys
from tracking.runtime import SquareTracker


def _normalize_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    array = np.maximum(array, 0.0)
    return array / max(float(array.sum()), 1.0e-8)


def _resize_distribution(value: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    source = _normalize_map(value)
    array = source
    output_h, output_w = output_hw
    if max(array.shape) > 64:
        array = cv2.resize(array, (64, 64), interpolation=cv2.INTER_AREA)
    if array.shape != (output_h, output_w):
        array = (
            torch.nn.functional.interpolate(
                torch.from_numpy(array)[None, None],
                size=(output_h, output_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            .numpy()
        )
    if float(array.sum()) <= 1.0e-8:
        array = cv2.resize(source, (output_w, output_h), interpolation=cv2.INTER_AREA)
    return _normalize_map(array)


def _entropy(value: np.ndarray) -> float:
    distribution = _normalize_map(value).reshape(-1)
    distribution = np.clip(distribution, 1.0e-12, None)
    return float(-(distribution * np.log(distribution)).sum())


def _map_peak_image_xy(value: np.ndarray, image_hw: tuple[int, int]) -> tuple[float, float]:
    array = np.asarray(value)
    row, column = np.unravel_index(int(array.argmax()), array.shape)
    image_h, image_w = image_hw
    return (
        (float(column) + 0.5) * float(image_w) / max(array.shape[1], 1),
        (float(row) + 0.5) * float(image_h) / max(array.shape[0], 1),
    )


def _draw_map_panel(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    title: str,
    tracker_bbox_xywh: Optional[Sequence[float]],
    box_color_bgr: tuple[int, int, int] = (0, 165, 255),
) -> np.ndarray:
    image_h, image_w = rgb.shape[:2]
    value = _normalize_map(heatmap)
    peak_x, peak_y = _map_peak_image_xy(value, (image_h, image_w))
    visualization = value - float(value.min())
    visualization /= max(float(visualization.max()), 1.0e-8)
    visualization = cv2.resize(
        visualization,
        (image_w, image_h),
        interpolation=cv2.INTER_LINEAR,
    )
    colored = cv2.applyColorMap(np.uint8(np.clip(visualization, 0.0, 1.0) * 255.0), cv2.COLORMAP_VIRIDIS)
    canvas = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.55, colored, 0.45, 0.0)
    if tracker_bbox_xywh is not None and len(tracker_bbox_xywh) >= 4:
        x, y, width, height = [int(round(float(component))) for component in tracker_bbox_xywh[:4]]
        cv2.rectangle(canvas, (x, y), (x + width, y + height), box_color_bgr, 3)
    cv2.drawMarker(
        canvas,
        (int(round(peak_x)), int(round(peak_y))),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        22,
        2,
    )
    label = f"{title}  peak=({peak_x:.0f},{peak_y:.0f})  H={_entropy(value):.2f}"
    cv2.rectangle(canvas, (0, 0), (image_w, 34), (0, 0, 0), -1)
    cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    return canvas


def save_comparison_panel(
    output_path: Path,
    rgb: np.ndarray,
    query0_map: np.ndarray,
    all_query_map: np.ndarray,
    tracker_response: np.ndarray,
    tracker_bbox_xywh: Optional[Sequence[float]],
) -> None:
    tracker_target = _resize_distribution(tracker_response, query0_map.shape)
    grid_label = f"{query0_map.shape[0]}x{query0_map.shape[1]}"
    panels = [
        _draw_map_panel(rgb, query0_map, "query 0 (loss-normalized)", tracker_bbox_xywh),
        _draw_map_panel(rgb, all_query_map, "all queries (loss-normalized)", tracker_bbox_xywh),
        _draw_map_panel(
            rgb,
            tracker_target,
            f"Tracker response ({grid_label} target)",
            tracker_bbox_xywh,
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate(panels, axis=1))


def _load_model(checkpoint: Path, device: torch.device) -> tuple[TeacherWorldModelDiT, ModelConfig]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_values = migrate_legacy_config(payload.get("cfg", {}))
    cfg_values["compile_action_sampling"] = False
    cfg = ModelConfig(**cfg_values)
    model = TeacherWorldModelDiT(cfg).to(device)
    state = payload["model"]
    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(migrate_legacy_state_dict_keys(state), strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trainable = [name for name in missing if name in trainable]
    if missing_trainable or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing_trainable={missing_trainable}, unexpected={list(unexpected)}"
        )
    return model.eval(), cfg


def _initial_bbox(step: dict[str, Any]) -> list[float]:
    bbox = step.get("tracker_bbox_xywh", step.get("ortrack_bbox_xywh"))
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(value) for value in bbox]
    crop = step.get("target_crop_action_overlay", {}).get("target_crop", {})
    box = crop.get("gt_box_xyxy")
    if isinstance(box, list) and len(box) == 4:
        return [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
    raise ValueError("The rollout has no usable initial Tracker bbox.")


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, cfg = _load_model(args.checkpoint, device)
    tracker = SquareTracker(args.tracker_checkpoint, args.device)
    rollout_path = args.trajectory / "online_rollout.json"
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    steps = rollout.get("steps", [])
    if not steps:
        raise ValueError(f"No rollout steps in {rollout_path}")
    image_transform = transforms.Compose(
        [transforms.Resize((cfg.image_size, cfg.image_size)), transforms.ToTensor()]
    )
    previous_action = torch.zeros(1, cfg.action_dim, device=device, dtype=torch.float32)
    previous_done = torch.zeros(1, device=device, dtype=torch.float32)
    rssm_state = None
    initial_bbox = _initial_bbox(steps[0])
    output_dir = args.trajectory / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, step in enumerate(steps):
        frame_index = int(step.get("step", position))
        rgb_path = args.trajectory / "rgb" / f"frame_{frame_index:05d}.png"
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if position == 0:
            tracker.initialize(rgb, initial_bbox)
            tracker_result = tracker.track(rgb)
            tracker.initialize(rgb, initial_bbox)
        else:
            tracker_result = tracker.track(rgb)
        tracker_bbox = [float(value) for value in tracker_result["bbox"]]
        image_tensor = image_transform(Image.fromarray(rgb)).unsqueeze(0).to(device).float()
        target_relative = torch.tensor(
            step["relative_target_body"], dtype=torch.float32, device=device
        ).view(1, -1)
        instruction = str(step.get("instruction") or "Keep tracking and approaching the target UAV.")
        text_tokens = torch.zeros(1, 1, dtype=torch.long, device=device)
        text_mask = torch.ones_like(text_tokens)
        prediction, rssm_state = model.act(
            image=image_tensor,
            text_tokens=text_tokens,
            target_relative=target_relative,
            prev_action=previous_action,
            rssm_state=rssm_state,
            attention_mask=text_mask,
            prev_done=previous_done,
            deterministic=True,
            num_steps=args.sampling_steps,
            instruction=instruction,
            save_transformer_attention=True,
            save_predicted_video=False,
        )
        query0 = prediction["last_transformer_attention_query0_map"][0].float().cpu().numpy()
        all_queries = prediction["last_transformer_attention_all_queries_map"][0].float().cpu().numpy()
        response = np.asarray(tracker_result["response"], dtype=np.float32)
        tracker_target = _resize_distribution(response, query0.shape)
        output_path = output_dir / f"frame_{frame_index:05d}_attention_tracker_comparison.png"
        save_comparison_panel(output_path, rgb, query0, all_queries, response, tracker_bbox)
        standalone = {
            "query0": _draw_map_panel(
                rgb, query0, "query 0 (loss-normalized)", tracker_bbox
            ),
            "all_queries": _draw_map_panel(
                rgb, all_queries, "all queries (loss-normalized)", tracker_bbox
            ),
            "tracker_response": _draw_map_panel(
                rgb,
                tracker_target,
                f"Tracker response ({query0.shape[0]}x{query0.shape[1]} target)",
                tracker_bbox,
            ),
        }
        for name, panel in standalone.items():
            panel_path = output_dir / name / f"frame_{frame_index:05d}.png"
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(panel_path), panel)
        previous_action = prediction["action_norm"].detach().float().view(1, -1)
        rows.append(
            {
                "frame": frame_index,
                "comparison": str(output_path.relative_to(args.trajectory)),
                "query0_peak_xy": list(_map_peak_image_xy(query0, rgb.shape[:2])),
                "all_queries_peak_xy": list(_map_peak_image_xy(all_queries, rgb.shape[:2])),
                "tracker_response_peak_xy": list(_map_peak_image_xy(tracker_target, rgb.shape[:2])),
                "query0_entropy": _entropy(query0),
                "all_queries_entropy": _entropy(all_queries),
                "tracker_response_entropy": _entropy(tracker_target),
                "tracker_bbox_xywh": tracker_bbox,
                "attention_grid_hw": [int(query0.shape[0]), int(query0.shape[1])],
            }
        )
        print(f"[compare] {position + 1}/{len(steps)} frame={frame_index}", flush=True)
    query0_distances = [
        math.hypot(
            row["query0_peak_xy"][0] - row["tracker_response_peak_xy"][0],
            row["query0_peak_xy"][1] - row["tracker_response_peak_xy"][1],
        )
        for row in rows
    ]
    all_query_distances = [
        math.hypot(
            row["all_queries_peak_xy"][0] - row["tracker_response_peak_xy"][0],
            row["all_queries_peak_xy"][1] - row["tracker_response_peak_xy"][1],
        )
        for row in rows
    ]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "tracker_checkpoint": str(args.tracker_checkpoint.resolve()),
        "attention_grid": rows[0]["attention_grid_hw"],
        "image_size": [640, 640],
        "normalization": "per head/query over first-frame video tokens, then mean",
        "aggregate": {
            "mean_query0_entropy": float(np.mean([row["query0_entropy"] for row in rows])),
            "mean_all_queries_entropy": float(
                np.mean([row["all_queries_entropy"] for row in rows])
            ),
            "mean_tracker_response_entropy": float(
                np.mean([row["tracker_response_entropy"] for row in rows])
            ),
            "mean_query0_to_tracker_peak_distance_px": float(np.mean(query0_distances)),
            "median_query0_to_tracker_peak_distance_px": float(np.median(query0_distances)),
            "mean_all_queries_to_tracker_peak_distance_px": float(
                np.mean(all_query_distances)
            ),
            "median_all_queries_to_tracker_peak_distance_px": float(
                np.median(all_query_distances)
            ),
        },
        "rows": rows,
    }
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output_dir / "summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FastWAM attention with Tracker response.")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tracker-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="attention_tracker_comparisons")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
