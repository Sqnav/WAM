from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from data.visual_guidance import project_body_to_image


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_steps(trajectory: Path) -> List[Dict[str, Any]]:
    path = trajectory / "online_rollout.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps = payload.get("steps") if isinstance(payload, dict) else None
    else:
        path = trajectory / "uav_trajectory.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing online_rollout.json or uav_trajectory.json in {trajectory}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps = payload.get("trajectory") if isinstance(payload, dict) else None
    if not isinstance(steps, list):
        raise ValueError(f"Invalid steps in {path}")
    return steps


def _relative_target(step: Dict[str, Any]) -> np.ndarray:
    value = step.get("relative_target_body")
    if isinstance(value, list) and len(value) >= 3:
        return np.asarray(value[:3], dtype=np.float32)
    value = step.get("target_position_in_body_frame")
    if isinstance(value, dict) and all(axis in value for axis in ("x", "y", "z")):
        return np.asarray([value["x"], value["y"], value["z"]], dtype=np.float32)
    raise ValueError("The first step has no target position in the UAV body frame.")


def _bbox_from_target_center(
    target_relative_xyz: np.ndarray,
    image_hw: Tuple[int, int],
    fov_deg: float,
    camera_offset_body: Tuple[float, float, float],
    box_frac: float,
) -> List[float]:
    height, width = image_hw
    target = torch.as_tensor(target_relative_xyz[:3], dtype=torch.float32).view(1, 3)
    center, visible = project_body_to_image(
        target,
        image_hw,
        fov_deg=float(fov_deg),
        camera_offset_body=camera_offset_body,
    )
    if not bool(visible.item()):
        raise RuntimeError("Cannot initialize ORTrack: GT target center is outside the first frame.")
    center_x = float(center[0, 0].item()) * max(width - 1, 1)
    center_y = float(center[0, 1].item()) * max(height - 1, 1)
    side = max(float(min(height, width)) * float(box_frac), 8.0)
    x = float(np.clip(center_x - 0.5 * side, 0.0, max(width - side, 0.0)))
    y = float(np.clip(center_y - 0.5 * side, 0.0, max(height - side, 0.0)))
    return [x, y, min(side, float(width)), min(side, float(height))]


def _real_bbox_for_frame(trajectory: Path, frame_index: int) -> Optional[List[float]]:
    path = trajectory / "target_boxes.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("boxes_xywh", payload.get("frames")) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"Unsupported target box annotation format: {path}")
    for index, value in enumerate(values):
        current_index = index
        if isinstance(value, dict):
            current_index = int(value.get("frame_idx", index))
            value = value.get("bbox_xywh", value.get("bbox"))
        if current_index != frame_index:
            continue
        if not isinstance(value, list) or len(value) != 4:
            return None
        bbox = [float(component) for component in value]
        if not all(math.isfinite(component) for component in bbox) or bbox[2] <= 1.0 or bbox[3] <= 1.0:
            return None
        return bbox
    return None


def _square_bbox(bbox: List[float]) -> List[float]:
    x, y, width, height = bbox
    side = max(width, height)
    return [x + 0.5 * (width - side), y + 0.5 * (height - side), side, side]


def _bbox_heatmap(bbox: List[float], image_hw: Tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    x, y, box_w, box_h = bbox
    x1, y1 = max(int(round(x)), 0), max(int(round(y)), 0)
    x2, y2 = min(int(round(x + box_w)), width), min(int(round(y + box_h)), height)
    heatmap = np.zeros((height, width), dtype=np.float32)
    heatmap[y1:y2, x1:x2] = 1.0
    return heatmap / max(float(heatmap.sum()), 1.0e-8)


def _crop_geometry(box: List[float], factor: float) -> Tuple[int, int, int]:
    x, y, w, h = box
    crop_size = max(int(math.ceil(math.sqrt(max(w * h, 1.0)) * factor)), 1)
    x1 = int(round(x + 0.5 * w - 0.5 * crop_size))
    y1 = int(round(y + 0.5 * h - 0.5 * crop_size))
    return x1, y1, crop_size


def _map_response_to_image(
    response: torch.Tensor,
    previous_box: List[float],
    search_factor: float,
    image_hw: Tuple[int, int],
) -> np.ndarray:
    height, width = image_hw
    x1, y1, crop_size = _crop_geometry(previous_box, search_factor)
    response_np = response.detach().float().cpu().squeeze().numpy()
    response_crop = cv2.resize(response_np, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
    response_crop = np.maximum(response_crop, 0.0)
    full = np.zeros((height, width), dtype=np.float32)
    dst_x1, dst_y1 = max(x1, 0), max(y1, 0)
    dst_x2, dst_y2 = min(x1 + crop_size, width), min(y1 + crop_size, height)
    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        src_x1, src_y1 = dst_x1 - x1, dst_y1 - y1
        src_x2, src_y2 = src_x1 + dst_x2 - dst_x1, src_y1 + dst_y2 - dst_y1
        full[dst_y1:dst_y2, dst_x1:dst_x2] = response_crop[src_y1:src_y2, src_x1:src_x2]
    return full


def _confidence_heatmap(response: np.ndarray, confidence: float) -> np.ndarray:
    response = np.maximum(np.asarray(response, dtype=np.float32), 0.0)
    response = response / max(float(response.sum()), 1.0e-8)
    uniform = np.full_like(response, 1.0 / max(float(response.size), 1.0))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    heatmap = confidence * response + (1.0 - confidence) * uniform
    return heatmap / max(float(heatmap.sum()), 1.0e-8)


def _heatmap_color(heatmap: np.ndarray) -> np.ndarray:
    value = np.asarray(heatmap, dtype=np.float32)
    value -= float(value.min())
    value /= max(float(value.max()), 1.0e-8)
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(value * 255.0), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


class ORTrack640:
    def __init__(self, root: Path, checkpoint: Path, config_name: str, device: str) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"ORTrack checkpoint not found: {checkpoint}. Download the official "
                "deit_tiny_patch16_224 checkpoint linked in third_party/ortrack/README.md."
            )
        if str(device) != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The official ORTrack inference utilities currently require CUDA.")
        sys.path.insert(0, str(root))
        from lib.config.ortrack.config import cfg, update_config_from_file
        from lib.models.ortrack import build_ortrack
        from lib.test.tracker.data_utils import Preprocessor
        from lib.test.utils.hann import hann2d
        from lib.utils.box_ops import clip_box

        processing_path = root / "lib" / "train" / "data" / "processing_utils.py"
        spec = importlib.util.spec_from_file_location("ortrack_processing_utils", processing_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load ORTrack processing utilities from {processing_path}")
        processing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(processing)
        sample_target = processing.sample_target

        update_config_from_file(str(root / "experiments" / "ortrack" / f"{config_name}.yaml"))
        network = build_ortrack(cfg, training=False)
        payload = torch.load(checkpoint, map_location="cpu")
        network.load_state_dict(payload.get("net", payload), strict=True)
        self.network = network.cuda().eval()
        self.cfg = cfg
        self.preprocessor = Preprocessor()
        self.sample_target = sample_target
        self.clip_box = clip_box
        self.template_factor = float(cfg.TEST.TEMPLATE_FACTOR)
        self.template_size = int(cfg.TEST.TEMPLATE_SIZE)
        self.search_factor = float(cfg.TEST.SEARCH_FACTOR)
        self.search_size = int(cfg.TEST.SEARCH_SIZE)
        self.feature_size = self.search_size // int(cfg.MODEL.BACKBONE.STRIDE)
        self.output_window = hann2d(torch.tensor([self.feature_size, self.feature_size]).long(), centered=True).cuda()
        self.is_distill = bool(cfg.MODEL.IS_DISTILL)
        self.state: Optional[List[float]] = None
        self.template_rgb: Optional[np.ndarray] = None

    def initialize(self, image: np.ndarray, bbox: List[float]) -> Dict[str, Any]:
        template, resize_factor, attention_mask = self.sample_target(
            image, bbox, self.template_factor, output_sz=self.template_size
        )
        self.template_rgb = template
        self.template = self.preprocessor.process(template, attention_mask)
        self.state = list(bbox)
        return {"template": template, "resize_factor": float(resize_factor)}

    def track(self, image: np.ndarray) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("ORTrack must be initialized before track().")
        height, width = image.shape[:2]
        previous_box = list(self.state)
        search, resize_factor, attention_mask = self.sample_target(
            image, previous_box, self.search_factor, output_sz=self.search_size
        )
        search_tensor = self.preprocessor.process(search, attention_mask)
        with torch.no_grad():
            output = self.network.forward(
                template=self.template.tensors,
                search=search_tensor.tensors,
                is_distill=self.is_distill,
            )
        raw_score = output["score_map"]
        response = self.output_window * raw_score
        boxes = self.network.box_head.cal_bbox(response, output["size_map"], output["offset_map"]).view(-1, 4)
        box_crop = (boxes.mean(dim=0) * self.search_size / resize_factor).tolist()
        cx_prev = previous_box[0] + 0.5 * previous_box[2]
        cy_prev = previous_box[1] + 0.5 * previous_box[3]
        half_side = 0.5 * self.search_size / resize_factor
        cx, cy, box_w, box_h = box_crop
        mapped = [cx + cx_prev - half_side - 0.5 * box_w, cy + cy_prev - half_side - 0.5 * box_h, box_w, box_h]
        self.state = self.clip_box(mapped, height, width, margin=10)
        confidence = float(response.max().detach().cpu().item())
        full_response = _map_response_to_image(response, previous_box, self.search_factor, (height, width))
        heatmap = _confidence_heatmap(full_response, confidence)
        return {
            "bbox": list(self.state),
            "confidence": confidence,
            "response": full_response,
            "heatmap": heatmap,
            "search_region": search,
            "search_crop_xy_size": list(_crop_geometry(previous_box, self.search_factor)),
        }


def run(args: argparse.Namespace, tracker: Optional[Any] = None) -> Dict[str, Any]:
    trajectory = args.trajectory.resolve()
    rgb_paths = sorted((trajectory / "rgb").glob("frame_*.png"))
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB frames in {trajectory / 'rgb'}")
    steps = _load_steps(trajectory)
    # Tracking only requires RGB after a real initialization box is found.
    # Do not truncate a cache because an auxiliary rollout file is shorter.
    frame_count = len(rgb_paths)
    first_rgb = np.asarray(Image.open(rgb_paths[0]).convert("RGB"), dtype=np.uint8)
    if first_rgb.shape[:2] != (640, 640):
        raise ValueError(f"Expected saved 640x640 RGB, got {first_rgb.shape[:2]}")

    cache_root = getattr(args, "cache_root", None)
    if cache_root is not None:
        cache_root = Path(cache_root).expanduser().resolve()
        cache_dir = cache_root / trajectory.parent.name / trajectory.name
        init_dir = cache_dir / "init"
        output_dir = cache_dir / "heatmaps"
        summary_path = cache_dir / "summary.json"
    else:
        cache_dir = trajectory
        init_dir = trajectory / args.init_dir_name
        output_dir = trajectory / args.output_dir_name
        summary_path = trajectory / args.summary_name
    init_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = cache_dir / "features"
    save_tracker_features = bool(getattr(args, "save_tracker_features", False))
    if save_tracker_features:
        feature_dir.mkdir(parents=True, exist_ok=True)
    init_index = 0
    init_bbox = _real_bbox_for_frame(trajectory, init_index)
    if init_bbox is None and bool(getattr(args, "require_real_init_bbox", False)):
        for candidate_index in range(1, frame_count):
            candidate_bbox = _real_bbox_for_frame(trajectory, candidate_index)
            if candidate_bbox is not None:
                init_index = candidate_index
                init_bbox = candidate_bbox
                break
    if init_bbox is None:
        if bool(getattr(args, "require_real_init_bbox", False)):
            raise RuntimeError(f"No valid initialization bbox in {trajectory / 'target_boxes.json'}")
        if not steps:
            raise RuntimeError(f"Cannot project an initialization bbox without trajectory steps: {trajectory}")
        init_bbox = _bbox_from_target_center(
            _relative_target(steps[0]),
            first_rgb.shape[:2],
            float(args.fov_deg),
            tuple(float(v) for v in args.camera_offset_body),
            float(args.init_box_frac),
        )
        initialization_backend = "gt_projected_fixed_bbox"
    else:
        initialization_backend = "gt_segmentation_bbox"
    if bool(getattr(args, "square_init_bbox", False)):
        init_bbox = _square_bbox(init_bbox)
        initialization_backend += "_square"
    init_rgb = np.asarray(Image.open(rgb_paths[init_index]).convert("RGB"), dtype=np.uint8)
    init_heatmap = _bbox_heatmap(init_bbox, init_rgb.shape[:2])
    _dump_json(
        init_dir / f"frame_{init_index:05d}_init_meta.json",
        {
            "backend": initialization_backend,
            "frame": init_index,
            "ortrack_init_bbox_xywh": init_bbox,
        },
    )

    tracker_backend = str(getattr(args, "tracker_backend", "ortrack"))
    if tracker is None:
        if tracker_backend == "square":
            from tracking.runtime import SquareTracker

            tracker = SquareTracker(Path(args.tracker_checkpoint), args.device)
        else:
            tracker = ORTrack640(args.ortrack_root, args.checkpoint, args.config, args.device)
    init_result = tracker.initialize(init_rgb, init_bbox)
    first_response_result = None
    if bool(getattr(args, "native_first_frame_response", False)):
        first_response_result = tracker.track(init_rgb)
        tracker.initialize(init_rgb, init_bbox)
    save_debug = bool(getattr(args, "save_debug", False)) or cache_root is None
    if save_debug:
        Image.fromarray(init_result["template"]).save(init_dir / "frame_00000_ortrack_template.png")

    rows: List[Dict[str, Any]] = []
    direct_area_size = max(int(getattr(args, "direct_area_heatmap_size", 7)), 1)
    for index, rgb_path in enumerate(rgb_paths[:frame_count]):
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        if index < init_index:
            bbox = [0.0, 0.0, 0.0, 0.0]
            confidence = 0.0
            response = np.zeros(rgb.shape[:2], dtype=np.float32)
            heatmap = response
            search_region = rgb
            search_geometry = None
        elif index == init_index:
            bbox = init_bbox
            if first_response_result is None:
                confidence = 1.0
                response = init_heatmap
                heatmap = _confidence_heatmap(response, confidence)
                search_region = init_result["template"]
                search_geometry = None
            else:
                confidence = first_response_result["confidence"]
                response = first_response_result["response"]
                heatmap = first_response_result["heatmap"]
                search_region = first_response_result["search_region"]
                search_geometry = first_response_result["search_crop_xy_size"]
        else:
            result = tracker.track(rgb)
            bbox = result["bbox"]
            confidence = result["confidence"]
            response = result["response"]
            heatmap = result["heatmap"]
            search_region = result["search_region"]
            search_geometry = result["search_crop_xy_size"]

        stem = f"frame_{index:05d}"
        feature_relpath = None
        tracker_result = first_response_result if index == init_index else (
            result if index > init_index else None
        )
        if save_tracker_features:
            if index < init_index and first_response_result is not None:
                feature_tokens = np.zeros_like(
                    np.asarray(first_response_result["feature_tokens"], dtype=np.float32)
                )
            elif tracker_result is None or tracker_result.get("feature_tokens") is None:
                raise RuntimeError(
                    "Tracker feature caching requires a native Tracker response for every frame."
                )
            else:
                feature_tokens = np.asarray(tracker_result["feature_tokens"], dtype=np.float32)
            if feature_tokens.ndim != 2:
                raise ValueError(
                    f"Tracker feature tokens must have shape [N,C], got {feature_tokens.shape}."
                )
            feature_path = feature_dir / f"{stem}_features.npy"
            np.save(feature_path, feature_tokens.astype(np.float16))
            feature_relpath = str(feature_path.relative_to(cache_dir))
        guidance_heatmap = _confidence_heatmap(response, 1.0)
        direct_area_heatmap = cv2.resize(
            guidance_heatmap,
            (direct_area_size, direct_area_size),
            interpolation=cv2.INTER_AREA,
        )
        direct_area_heatmap = direct_area_heatmap / max(
            float(direct_area_heatmap.sum()), 1.0e-8
        )
        if cache_root is not None:
            guidance_heatmap = cv2.resize(guidance_heatmap, (64, 64), interpolation=cv2.INTER_AREA)
            guidance_heatmap = guidance_heatmap / max(float(guidance_heatmap.sum()), 1.0e-8)
            heatmap_payload = guidance_heatmap.astype(np.float16)
        else:
            heatmap_payload = guidance_heatmap.astype(np.float32)
        np.save(output_dir / f"{stem}_heatmap.npy", heatmap_payload)
        np.save(
            output_dir / f"{stem}_heatmap_area{direct_area_size}.npy",
            direct_area_heatmap.astype(np.float32),
        )
        heatmap_rgb = _heatmap_color(heatmap)
        overlay = np.uint8(0.50 * rgb.astype(np.float32) + 0.50 * heatmap_rgb.astype(np.float32))
        panel = Image.new("RGB", (1280, 640), (0, 0, 0))
        left = Image.fromarray(rgb)
        draw = ImageDraw.Draw(left)
        x, y, w, h = bbox
        draw.rectangle((x, y, x + w, y + h), outline=(255, 50, 40), width=3)
        draw.text((8, 8), f"ORTrack confidence={confidence:.4f}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        panel.paste(left, (0, 0))
        panel.paste(Image.fromarray(overlay), (640, 0))
        if save_debug:
            panel.save(output_dir / f"{stem}_panel.png")
            Image.fromarray(search_region).save(output_dir / f"{stem}_search.png")
        rows.append(
            {
                "frame": index,
                "rgb": str(rgb_path.relative_to(trajectory)),
                "bbox_xywh": [float(v) for v in bbox],
                "confidence": float(confidence),
                "search_crop_xy_size": search_geometry,
                "heatmap": str((output_dir / f"{stem}_heatmap.npy").relative_to(cache_dir)),
                "heatmap_direct_area": str(
                    (output_dir / f"{stem}_heatmap_area{direct_area_size}.npy").relative_to(cache_dir)
                ),
                "tracker_features": feature_relpath,
                "panel": (
                    str((output_dir / f"{stem}_panel.png").relative_to(cache_dir)) if save_debug else None
                ),
            }
        )
    checkpoint_path = (
        Path(args.tracker_checkpoint) if tracker_backend == "square" else Path(args.checkpoint)
    ).expanduser().resolve()
    checkpoint_stat = checkpoint_path.stat()
    summary = {
        "trajectory": str(trajectory),
        "tracker_backend": tracker_backend,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "config": args.config if tracker_backend == "ortrack" else "square_box_uav_tracker",
        "image_size": [640, 640],
        "frame_count": frame_count,
        "init_bbox_xywh": init_bbox,
        "initialization": {
            "backend": initialization_backend,
            "frame": init_index,
            "box_frac": float(args.init_box_frac),
            "fov_deg": float(args.fov_deg),
            "camera_offset_body": [float(v) for v in args.camera_offset_body],
        },
        "template_size": tracker.template_size,
        "search_size": tracker.search_size,
        "search_factor": tracker.search_factor,
        "cached_heatmap_size": [64, 64] if cache_root is not None else [640, 640],
        "cached_heatmap_dtype": "float16" if cache_root is not None else "float32",
        "direct_area_heatmap_size": [direct_area_size, direct_area_size],
        "direct_area_heatmap_dtype": "float32",
        "tracker_feature_cache_version": 1 if save_tracker_features else 0,
        "tracker_feature_grid_size": (
            [int(tracker.feature_grid_size), int(tracker.feature_grid_size)]
            if save_tracker_features and hasattr(tracker, "feature_grid_size")
            else None
        ),
        "tracker_feature_dim": (
            int(np.asarray(first_response_result["feature_tokens"]).shape[-1])
            if save_tracker_features and first_response_result is not None
            else None
        ),
        "tracker_feature_dtype": "float16" if save_tracker_features else None,
        "direct_area_heatmap_pipeline": (
            "native_tracker_response->search_geometry_full_image->area_pool->sum_normalize"
        ),
        "frame0_heatmap_source": (
            "target_absent_before_initialization"
            if init_index > 0
            else ("tracker_response" if first_response_result is not None else "gt_initialization_bbox")
        ),
        "frames": rows,
    }
    _dump_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Track a saved 640x640 trajectory with official ORTrack and export target heatmaps.")
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--ortrack-root", type=Path, default=root / "third_party" / "ortrack")
    parser.add_argument("--checkpoint", type=Path, default=root / "model" / "ortrack" / "deit_tiny_patch16_224" / "ORTrack_ep0300.pth.tar")
    parser.add_argument("--config", type=str, default="deit_tiny_patch16_224")
    parser.add_argument("--tracker-backend", choices=["ortrack", "square"], default="ortrack")
    parser.add_argument(
        "--tracker-checkpoint",
        type=Path,
        default=root / "experiments" / "uav_tracker_imagenet_square" / "best.pt",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--init-dir-name", type=str, default="ortrack640_init")
    parser.add_argument("--output-dir-name", type=str, default="ortrack640_heatmaps")
    parser.add_argument("--summary-name", type=str, default="ortrack640_summary.json")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--save-debug", action="store_true", default=False)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--camera-offset-body", nargs=3, type=float, default=(0.46, 0.0, 0.0))
    parser.add_argument("--init-box-frac", type=float, default=0.10)
    parser.add_argument("--require-real-init-bbox", action="store_true", default=False)
    parser.add_argument("--square-init-bbox", action="store_true", default=False)
    parser.add_argument("--native-first-frame-response", action="store_true", default=False)
    parser.add_argument("--direct-area-heatmap-size", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(f"done frames={len(result['frames'])} summary={result.get('summary_path', result['trajectory'])}")
