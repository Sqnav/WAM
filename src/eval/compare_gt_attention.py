from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data.visual_guidance import make_canonical_heatmap
from eval.compare_tracker_attention import (
    _draw_map_panel,
    _entropy,
    _load_model,
    _map_peak_image_xy,
    _resize_distribution,
)


GT_GREEN_BGR = (64, 255, 64)


def _gt_box_xywh(step: dict) -> Optional[list[float]]:
    overlay = step.get("target_crop_action_overlay")
    target_crop = overlay.get("target_crop") if isinstance(overlay, dict) else None
    box = target_crop.get("gt_box_xyxy") if isinstance(target_crop, dict) else None
    if not isinstance(box, list) or len(box) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def _canonical_gt_target(
    bbox_xywh: Sequence[float],
    image_hw: tuple[int, int],
    output_hw: tuple[int, int],
    sigma: float,
) -> np.ndarray:
    image_h, image_w = image_hw
    x, y, width, height = (float(value) for value in bbox_xywh)
    center = torch.tensor(
        [(x + 0.5 * width) / max(image_w, 1), (y + 0.5 * height) / max(image_h, 1)],
        dtype=torch.float32,
    )
    heatmap_64 = make_canonical_heatmap(center, (64, 64), sigma=sigma).squeeze().numpy()
    return _resize_distribution(heatmap_64, output_hw)


def _save_comparison(
    output_path: Path,
    rgb: np.ndarray,
    query0: np.ndarray,
    all_queries: np.ndarray,
    gt_target: np.ndarray,
    gt_bbox_xywh: Sequence[float],
    sigma: float,
) -> None:
    grid_label = f"{query0.shape[0]}x{query0.shape[1]}"
    panels = [
        _draw_map_panel(
            rgb,
            query0,
            "query 0 (loss-normalized)",
            gt_bbox_xywh,
            GT_GREEN_BGR,
        ),
        _draw_map_panel(
            rgb,
            all_queries,
            "all queries (loss-normalized)",
            gt_bbox_xywh,
            GT_GREEN_BGR,
        ),
        _draw_map_panel(
            rgb,
            gt_target,
            f"GT canonical ({grid_label}, sigma={sigma:.2f})",
            gt_bbox_xywh,
            GT_GREEN_BGR,
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate(panels, axis=1))


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, cfg = _load_model(args.checkpoint, device)
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
    output_dir = args.trajectory / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, step in enumerate(steps):
        frame_index = int(step.get("step", position))
        bbox = _gt_box_xywh(step)
        if bbox is None:
            print(f"[compare-gt] skip frame={frame_index}: missing GT box", flush=True)
            continue
        rgb_path = args.trajectory / "rgb" / f"frame_{frame_index:05d}.png"
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
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
        gt_target = _canonical_gt_target(
            bbox,
            rgb.shape[:2],
            query0.shape,
            sigma=float(cfg.fastwam_attention_heatmap_sigma),
        )
        output_path = output_dir / f"frame_{frame_index:05d}_attention_gt_comparison.png"
        _save_comparison(
            output_path,
            rgb,
            query0,
            all_queries,
            gt_target,
            bbox,
            float(cfg.fastwam_attention_heatmap_sigma),
        )
        standalone = {
            "query0": _draw_map_panel(
                rgb, query0, "query 0 (loss-normalized)", bbox, GT_GREEN_BGR
            ),
            "all_queries": _draw_map_panel(
                rgb, all_queries, "all queries (loss-normalized)", bbox, GT_GREEN_BGR
            ),
            "gt_heatmap": _draw_map_panel(
                rgb,
                gt_target,
                f"GT canonical ({query0.shape[0]}x{query0.shape[1]}, sigma={cfg.fastwam_attention_heatmap_sigma:.2f})",
                bbox,
                GT_GREEN_BGR,
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
                "gt_heatmap_peak_xy": list(_map_peak_image_xy(gt_target, rgb.shape[:2])),
                "query0_entropy": _entropy(query0),
                "all_queries_entropy": _entropy(all_queries),
                "gt_heatmap_entropy": _entropy(gt_target),
                "gt_bbox_xywh": bbox,
                "attention_grid_hw": [int(query0.shape[0]), int(query0.shape[1])],
            }
        )
        print(f"[compare-gt] {position + 1}/{len(steps)} frame={frame_index}", flush=True)
    if not rows:
        raise RuntimeError("No frames with valid GT boxes were generated.")
    query0_distances = [
        math.dist(row["query0_peak_xy"], row["gt_heatmap_peak_xy"]) for row in rows
    ]
    all_query_distances = [
        math.dist(row["all_queries_peak_xy"], row["gt_heatmap_peak_xy"]) for row in rows
    ]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "gt_source": "online projected target box from target_crop_action_overlay",
        "attention_grid": rows[0]["attention_grid_hw"],
        "image_size": [640, 640],
        "canonical_sigma": float(cfg.fastwam_attention_heatmap_sigma),
        "normalization": "per head/query over first-frame video tokens, then mean",
        "aggregate": {
            "frames": len(rows),
            "mean_query0_entropy": float(np.mean([row["query0_entropy"] for row in rows])),
            "mean_all_queries_entropy": float(
                np.mean([row["all_queries_entropy"] for row in rows])
            ),
            "mean_gt_heatmap_entropy": float(
                np.mean([row["gt_heatmap_entropy"] for row in rows])
            ),
            "mean_query0_to_gt_peak_distance_px": float(np.mean(query0_distances)),
            "median_query0_to_gt_peak_distance_px": float(np.median(query0_distances)),
            "mean_all_queries_to_gt_peak_distance_px": float(np.mean(all_query_distances)),
            "median_all_queries_to_gt_peak_distance_px": float(np.median(all_query_distances)),
        },
        "rows": rows,
    }
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output_dir / "summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FastWAM attention with canonical GT heatmaps.")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="gt_attention_comparisons")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
