from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from data.action_mapping import norm_action_to_physical
from eval.online_eval_teacher import (
    CLIPTokenizerFast,
    ModelDrivenTrackerCropper,
    _bbox_from_target_center,
    _dump_json,
    _tracker_bbox_required,
    _tracker_center_required,
    _tracker_geometry_required,
    load_model,
    make_image_transform,
    rgb_to_model_tensor,
    save_predicted_video_state_overlays,
    save_target_crop_action_overlay,
    seed_everything,
    tokenize_instruction,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _model_args(summary_path: Path, checkpoint: Optional[Path], device: str) -> argparse.Namespace:
    summary = _load_json(summary_path)
    saved_args = summary.get("args")
    if not isinstance(saved_args, dict):
        raise ValueError(f"Missing saved eval args: {summary_path}")
    args = argparse.Namespace(**saved_args)
    if checkpoint is not None:
        args.checkpoint = str(checkpoint.expanduser().resolve())
    args.device = device
    # Offline replay has no simulator to hide compile latency from. Eager
    # sampling avoids a large one-off compile for a single short trajectory.
    args.compile_action_sampling = False
    return args


def _tracker_inputs(
    cfg,
    result: Dict[str, Any],
    image_hw: tuple[int, int],
    device: torch.device,
) -> Dict[str, Optional[torch.Tensor]]:
    image_h, image_w = image_hw
    bbox = [float(value) for value in result["bbox"]]
    x, y, width, height = bbox
    tracker_center = None
    if _tracker_center_required(cfg):
        tracker_center = torch.tensor(
            [[
                (x + 0.5 * width) / max(image_w, 1),
                (y + 0.5 * height) / max(image_h, 1),
            ]],
            dtype=torch.float32,
            device=device,
        ).clamp_(0.0, 1.0)

    tracker_bbox = None
    if _tracker_bbox_required(cfg):
        tracker_bbox = torch.tensor(
            [[
                (x + 0.5 * width) / max(image_w, 1),
                (y + 0.5 * height) / max(image_h, 1),
                width / max(image_w, 1),
                height / max(image_h, 1),
            ]],
            dtype=torch.float32,
            device=device,
        ).clamp_(0.0, 1.0)

    tracker_search_geometry = None
    tracker_image_size = None
    if _tracker_geometry_required(cfg):
        tracker_search_geometry = torch.tensor(
            [[float(value) for value in result["search_crop_xy_size"]]],
            dtype=torch.float32,
            device=device,
        )
        tracker_image_size = torch.tensor(
            [[float(image_h), float(image_w)]],
            dtype=torch.float32,
            device=device,
        )

    return {
        "tracker_center": tracker_center,
        "tracker_bbox": tracker_bbox,
        "tracker_search_geometry": tracker_search_geometry,
        "tracker_image_size": tracker_image_size,
        "tracker_template": result["tracker_template"].to(
            device=device, dtype=torch.float32
        ),
        "tracker_search": result["tracker_search"].to(
            device=device, dtype=torch.float32
        ),
    }


@torch.inference_mode()
def replay_trajectory(
    trajectory_dir: Path,
    summary_path: Path,
    *,
    checkpoint: Optional[Path],
    device_name: str,
    seed: int,
    output_name: str,
    save_interval: int,
) -> int:
    args = _model_args(summary_path, checkpoint, device_name)
    seed_everything(seed)
    device = torch.device(
        device_name
        if torch.cuda.is_available() and device_name.startswith("cuda")
        else "cpu"
    )
    model, cfg = load_model(args, device)
    if not bool(getattr(cfg, "use_future_state_dit", False)):
        raise ValueError("Checkpoint does not enable Future State DiT.")
    if not bool(getattr(cfg, "tracker_model_driven_search", False)):
        raise ValueError("Offline replay currently requires model-driven Tracker search.")

    tokenizer = None
    if not cfg.use_wan22_encoders:
        if CLIPTokenizerFast is None:
            raise ImportError("CLIPTokenizerFast is required by this checkpoint.")
        tokenizer = CLIPTokenizerFast.from_pretrained(
            args.tokenizer_name, local_files_only=True
        )
    transform = make_image_transform(cfg.image_size)

    rollout_path = trajectory_dir / "online_rollout.json"
    payload = _load_json(rollout_path)
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Missing rollout steps: {rollout_path}")

    cropper = ModelDrivenTrackerCropper(
        template_size=int(cfg.tracker_template_size),
        search_size=int(cfg.tracker_search_size),
    )
    output_dir = trajectory_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rssm_state = None
    prev_action = torch.zeros(
        1, int(cfg.action_dim), dtype=torch.float32, device=device
    )
    prev_done = torch.zeros(1, dtype=torch.float32, device=device)
    expected_states = int(getattr(cfg, "future_state_horizon", 8)) + 1

    for position, step in enumerate(steps):
        index = int(step.get("step", position))
        rgb_path = trajectory_dir / "rgb" / f"frame_{index:05d}.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing saved RGB: {rgb_path}")
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        relative_target = np.asarray(
            step["relative_target_body"], dtype=np.float32
        )
        recorded_bbox = step.get("ortrack_bbox_xywh")
        if recorded_bbox is None or len(recorded_bbox) < 4:
            raise ValueError(f"Step {index} has no recorded Tracker anchor.")

        if position == 0:
            init_bbox = _bbox_from_target_center(
                relative_target,
                rgb.shape[:2],
                float(args.fov_deg),
                tuple(float(value) for value in args.ortrack_camera_offset_body),
                float(args.ortrack_init_box_frac),
            )
            cropper.initialize(rgb, init_bbox)
        # Recreate the exact crop recorded by the original online trace. This
        # avoids feeding new offline s0 estimates back into fixed saved RGB.
        cropper.state_bbox = [float(value) for value in recorded_bbox[:4]]
        tracker_result = cropper.current(rgb, device)
        recorded_geometry = step.get("model_driven_search_geometry")
        if recorded_geometry is not None:
            actual_geometry = tracker_result["search_crop_xy_size"]
            if [int(value) for value in recorded_geometry] != [
                int(value) for value in actual_geometry
            ]:
                raise ValueError(
                    f"Step {index} search geometry mismatch: "
                    f"recorded={recorded_geometry}, replay={actual_geometry}"
                )

        tracker_inputs = _tracker_inputs(cfg, tracker_result, rgb.shape[:2], device)
        image_t = rgb_to_model_tensor(rgb, transform, device)
        instruction = str(step.get("instruction") or "")
        if cfg.use_wan22_encoders:
            text_tokens = torch.zeros(1, 1, dtype=torch.long, device=device)
            attention_mask = torch.ones_like(text_tokens)
        else:
            text_tokens, attention_mask = tokenize_instruction(
                tokenizer, instruction, cfg.text_context_length, device
            )
        target_relative_t = torch.from_numpy(
            relative_target / max(float(args.target_relative_scale), 1.0e-6)
        ).view(1, -1).to(device=device, dtype=torch.float32)
        pred, rssm_state = model.act(
            image=image_t,
            text_tokens=text_tokens,
            target_relative=target_relative_t,
            prev_action=prev_action,
            rssm_state=rssm_state,
            attention_mask=attention_mask,
            prev_done=prev_done,
            deterministic=bool(args.deterministic_action),
            num_steps=int(args.sampling_steps),
            instruction=instruction,
            save_transformer_attention=False,
            save_predicted_video=False,
            guidance_heatmap=torch.from_numpy(
                np.asarray(tracker_result["heatmap"], dtype=np.float32)
            ).unsqueeze(0).to(device),
            guidance_confidence=torch.ones(
                1, 1, dtype=torch.float32, device=device
            ),
            tracker_features=None,
            tracker_response=None,
            **tracker_inputs,
        )
        state_boxes_t = pred.get("target_state_boxes")
        if state_boxes_t is None or state_boxes_t.ndim != 3:
            raise RuntimeError(f"Step {index} did not return Target State boxes.")
        if int(state_boxes_t.size(1)) != expected_states:
            raise RuntimeError(
                f"Step {index} returned {state_boxes_t.size(1)} states; "
                f"expected {expected_states}."
            )
        state_boxes = (
            state_boxes_t[0].detach().float().cpu().numpy().astype(np.float32)
        )
        action_sequence_norm = (
            pred["action_sequence_norm"][0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        action_sequence_physical = norm_action_to_physical(
            action_sequence_norm,
            max_vel=cfg.max_vel,
            max_yaw_rate=cfg.max_yaw_rate,
            max_speed_norm=cfg.max_speed_norm,
        )

        current_box = state_boxes[0]
        future_boxes = state_boxes[1:]
        replay_record = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "seed": int(seed),
            "sampling_steps": int(args.sampling_steps),
            "search_source": "recorded_online_anchor",
            "model_driven_search_geometry": [
                int(value) for value in tracker_result["search_crop_xy_size"]
            ],
            "current_target_box_cxcywh": current_box.astype(float).tolist(),
            "future_target_boxes_cxcywh": future_boxes.astype(float).tolist(),
            "action_sequence_norm": action_sequence_norm.astype(float).tolist(),
            "action_sequence_physical": action_sequence_physical.astype(float).tolist(),
        }
        step["offline_future_state_replay"] = replay_record
        step["current_target_center_xy"] = current_box[:2].astype(float).tolist()
        step["current_target_box_cxcywh"] = current_box.astype(float).tolist()
        step["future_target_center_xy"] = future_boxes[0, :2].astype(float).tolist()
        step["future_target_box_cxcywh"] = future_boxes[0].astype(float).tolist()
        step["future_target_boxes_cxcywh"] = future_boxes.astype(float).tolist()

        overlay_path = (
            output_dir / f"frame_{index:05d}_target_crop_action_traj.png"
        )
        overlay_metadata = save_target_crop_action_overlay(
            overlay_path,
            rgb,
            relative_target,
            action_sequence_physical,
            float(args.fov_deg),
            args.ortrack_camera_offset_body,
            float(args.dt),
            ortrack_bbox_xywh=recorded_bbox,
            model_driven_search_geometry=tracker_result["search_crop_xy_size"],
            current_state_box_cxcywh=current_box,
            future_state_boxes_cxcywh=future_boxes,
        )
        overlay_metadata["overlay"] = str(overlay_path.relative_to(trajectory_dir))
        step["target_crop_action_overlay"] = overlay_metadata
        predicted_frames = step.get("predicted_video_frames")
        if isinstance(predicted_frames, list):
            predicted_paths = [
                trajectory_dir / str(value) for value in predicted_frames
            ]
            if len(predicted_paths) != len(state_boxes):
                raise ValueError(
                    f"Step {index} predicted video/state mismatch: "
                    f"frames={len(predicted_paths)} states={len(state_boxes)}"
                )
            state_overlay_paths = save_predicted_video_state_overlays(
                predicted_paths[0].parent / "state_overlays",
                predicted_paths,
                state_boxes,
            )
            step["predicted_video_state_overlays"] = [
                str(value.relative_to(trajectory_dir))
                for value in state_overlay_paths
            ]

        # The saved online action is the transition that produced the next RGB.
        # Keep it as prev_action instead of feeding back this offline sample.
        prev_action = torch.tensor(
            [step["action_norm"]], dtype=torch.float32, device=device
        )
        prev_done = torch.tensor(
            [1.0 if bool(step.get("collision", False)) else 0.0],
            dtype=torch.float32,
            device=device,
        )
        if save_interval > 0 and (position + 1) % save_interval == 0:
            _dump_json(rollout_path, payload)
        print(
            f"[offline-state-replay] {position + 1}/{len(steps)} frame={index} "
            f"states={len(state_boxes)}",
            flush=True,
        )

    payload["offline_future_state_replay"] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": int(seed),
        "sampling_steps": int(args.sampling_steps),
        "num_steps": len(steps),
        "num_states_per_step": expected_states,
        "search_source": "recorded_online_anchor",
        "output_name": output_name,
    }
    _dump_json(rollout_path, payload)
    return len(steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay saved online RGB and recover all Future State DiT boxes."
    )
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-name", default="target_crop_action_trajectory_overlays"
    )
    parser.add_argument("--save-interval", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = replay_trajectory(
        args.trajectory_dir.expanduser().resolve(),
        args.summary.expanduser().resolve(),
        checkpoint=args.checkpoint,
        device_name=args.device,
        seed=args.seed,
        output_name=args.output_name,
        save_interval=max(int(args.save_interval), 0),
    )
    print(f"[offline-state-replay] complete steps={count}", flush=True)


if __name__ == "__main__":
    main()
