from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from eval.compare_tracker_attention import _draw_map_panel, _load_model, _map_peak_image_xy


def _bbox_from_step(step: dict) -> list[float]:
    bbox = step.get("ortrack_bbox_xywh", step.get("tracker_bbox_xywh"))
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("rollout step has no recorded Tracker bbox")
    return [float(value) for value in bbox]


def _save_panel(path: Path, panel: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def _contact_sheet(panels: list[np.ndarray], columns: int = 4) -> np.ndarray:
    if not panels:
        raise ValueError("contact sheet requires at least one panel")
    blank = np.zeros_like(panels[0])
    rows = []
    for start in range(0, len(panels), columns):
        row = panels[start : start + columns]
        row.extend([blank] * (columns - len(row)))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, cfg = _load_model(args.checkpoint, device)

    rollout = json.loads((args.trajectory / "online_rollout.json").read_text(encoding="utf-8"))
    steps = rollout.get("steps", [])
    matches = [step for position, step in enumerate(steps) if int(step.get("step", position)) == args.frame]
    if not matches:
        raise ValueError(f"frame {args.frame} is absent from online_rollout.json")
    step = matches[0]
    rgb_path = args.trajectory / "rgb" / f"frame_{args.frame:05d}.png"
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    transform = transforms.Compose(
        [transforms.Resize((cfg.image_size, cfg.image_size)), transforms.ToTensor()]
    )
    image_tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device).float()
    target_relative = torch.tensor(
        step["relative_target_body"], dtype=torch.float32, device=device
    ).view(1, -1) / max(float(args.target_relative_scale), 1.0e-6)
    bbox = _bbox_from_step(step)
    x, y, width, height = bbox
    tracker_center = torch.tensor(
        [[(x + width / 2.0) / rgb.shape[1], (y + height / 2.0) / rgb.shape[0]]],
        dtype=torch.float32,
        device=device,
    ).clamp(0.0, 1.0)
    confidence = float(step.get("ortrack_confidence", 1.0))
    instruction = str(step.get("instruction") or "Keep tracking and approaching the target UAV.")
    text_tokens = torch.zeros(1, 1, dtype=torch.long, device=device)
    prediction, _ = model.act(
        image=image_tensor,
        text_tokens=text_tokens,
        target_relative=target_relative,
        prev_action=torch.zeros(1, cfg.action_dim, device=device),
        attention_mask=torch.ones_like(text_tokens),
        prev_done=torch.zeros(1, device=device),
        deterministic=True,
        num_steps=args.sampling_steps,
        instruction=instruction,
        save_transformer_attention=True,
        guidance_confidence=torch.tensor([[confidence]], device=device),
        tracker_center=tracker_center,
    )

    raw = prediction["last_transformer_attention_raw"][0].detach().float().cpu().numpy()
    grid_h, grid_w = prediction["last_transformer_attention_map"].shape[-2:]
    spatial = np.maximum(raw[..., : grid_h * grid_w], 0.0).reshape(raw.shape[0], raw.shape[1], grid_h, grid_w)
    output = args.trajectory / args.output_name / f"frame_{args.frame:05d}"
    output.mkdir(parents=True, exist_ok=True)

    pair_panels = []
    records = []
    for head in range(spatial.shape[0]):
        for query in range(spatial.shape[1]):
            value = spatial[head, query]
            mass = float(value.sum())
            panel = _draw_map_panel(rgb, value, f"head {head} / action query {query}  mass={mass:.4g}", bbox)
            _save_panel(output / "head_query" / f"head_{head:02d}_query_{query:02d}.png", panel)
            pair_panels.append(panel)
            records.append(
                {
                    "head": head,
                    "query": query,
                    "first_frame_attention_mass": mass,
                    "peak_xy": list(_map_peak_image_xy(value, rgb.shape[:2])),
                }
            )
    _save_panel(output / "head_query_contact_sheet.png", _contact_sheet(pair_panels, args.columns))

    # These summaries average only the axis named in the title. The pair maps above use no averaging.
    query_panels = []
    for query in range(spatial.shape[1]):
        value = spatial[:, query].sum(axis=0)
        panel = _draw_map_panel(rgb, value, f"action query {query}: sum over heads (raw)", bbox)
        _save_panel(output / "queries_raw_sum" / f"query_{query:02d}.png", panel)
        query_panels.append(panel)
    _save_panel(output / "queries_raw_sum_contact_sheet.png", _contact_sheet(query_panels, args.columns))

    head_panels = []
    for head in range(spatial.shape[0]):
        value = spatial[head].sum(axis=0)
        panel = _draw_map_panel(rgb, value, f"head {head}: sum over action queries (raw)", bbox)
        _save_panel(output / "heads_raw_sum" / f"head_{head:02d}.png", panel)
        head_panels.append(panel)
    _save_panel(output / "heads_raw_sum_contact_sheet.png", _contact_sheet(head_panels, args.columns))

    unnormalized = spatial.sum(axis=(0, 1))
    _save_panel(
        output / "unnormalized_sum.png",
        _draw_map_panel(rgb, unnormalized, "unnormalized raw sum over head/query", bbox),
    )
    report = {
        "frame": args.frame,
        "raw_attention_shape": list(raw.shape),
        "spatial_grid_hw": [grid_h, grid_w],
        "tracker_bbox_xywh": bbox,
        "tracker_center_normalized": tracker_center[0].cpu().tolist(),
        "tracker_confidence": confidence,
        "action_norm": prediction["action_norm"][0].detach().float().cpu().tolist(),
        "normalization": "head_query maps are raw attention with no cross-head/query averaging; display contrast is per panel",
        "gradient_attribution": "unavailable: FastWAM sampling and attention capture are decorated with torch.no_grad; no gradient claim is made",
        "head_query": records,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {len(pair_panels)} unaveraged head/query maps to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize unaveraged FastWAM action attention.")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--target-relative-scale", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--output-name", default="unaveraged_transformer_attention")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
