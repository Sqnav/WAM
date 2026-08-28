from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import random
import os
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from data.teacher_dataset_builder import build_records
from data.visual_guidance import make_canonical_heatmap
from model.config import ModelConfig
from model.losses import summarize_losses, world_model_dit_loss
from model.current_target_localizer import CurrentTargetLocalizer
from model.target_action_conditioning import augment_target_box_history
from model.model import (
    S0_PARAMETER_PREFIXES,
    S0LocalizationModel,
    TeacherWorldModelDiT,
    migrate_legacy_state_dict_keys,
    normalize_s0_checkpoint_state,
)
from tracking.data import MEAN as TRACKER_MEAN, STD as TRACKER_STD, crop_target
from tracking.evaluate import crop_search

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


_DEFAULT_CFG = ModelConfig()

try:
    import deepspeed
except Exception:
    deepspeed = None


def _str2bool(value: str | bool) -> bool:
    """Parse shell / CLI strings into bool (e.g. true false 1 0 yes no)."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _cuda_amp_dtype(cfg: ModelConfig) -> torch.dtype:
    dtype_name = str(getattr(cfg, "wan22_torch_dtype", "bfloat16")).lower()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _autocast_context(device: torch.device, cfg: ModelConfig):
    if device.type != "cuda":
        return nullcontext()
    amp_dtype = _cuda_amp_dtype(cfg)
    if amp_dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)


def _grad_scaler_enabled(device: torch.device, cfg: ModelConfig, use_deepspeed: bool) -> bool:
    return device.type == "cuda" and (not use_deepspeed) and _cuda_amp_dtype(cfg) == torch.float16


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrajectoryDataset(Dataset):
    REQUIRED_KEYS = [
        "target_relative",
        "next_target_relative",
        "prev_actions",
        "expert_action",
    ]
    LEGACY_KEY_ALIASES = {
        "target_relative": "privileged",
        "next_target_relative": "next_privileged",
    }

    def __init__(
        self,
        records: List[Dict[str, Any]],
        image_size: int,
        seq_len: int,
        target_relative_dim: int,
        action_dim: int,
        direction_bins: int = 8,
        distance_bins: int = 6,
        text_context_length: int = 77,
        random_crop: bool = True,
        wan_latent_cache_root: Optional[str] = None,
        action_video_freq_ratio: int = 1,
        ortrack_cache_root: Optional[str] = None,
        require_ortrack_cache: bool = False,
        canonical_heatmap_sigma: float = 0.08,
        tracker_heatmap_target_mode: str = "canonical",
        guidance_heatmap_source: Optional[str] = None,
        require_tracker_features: bool = False,
        require_tracker_bbox: bool = False,
        require_tracker_response: bool = False,
        require_tracker_geometry: bool = False,
        tracker_feature_grid_size: int = 7,
        tracker_feature_dim: int = 192,
        tracker_response_grid_size: int = 7,
        require_tracker_finetune_inputs: bool = False,
        use_model_driven_tracker_crops: bool = False,
        require_target_boxes: bool = False,
        tracker_search_crop_jitter: bool = False,
        tracker_search_center_jitter_std: float = 0.10,
        tracker_search_center_jitter_max: float = 0.20,
        tracker_search_scale_jitter: float = 0.10,
        localization_only: bool = False,
        require_target_box_history: bool = False,
        target_history_tracker_cache_root: Optional[str] = None,
        target_history_length: int = 8,
        target_history_partial_probability: float = 0.0,
        target_history_center_jitter_std: float = 0.0,
        target_history_log_size_jitter_std: float = 0.0,
        target_history_confidence_dropout_probability: float = 0.0,
    ) -> None:
        self.records = records
        self.seq_len = seq_len
        self.action_video_freq_ratio = max(int(action_video_freq_ratio), 1)
        self.image_size = int(image_size)
        self.target_relative_dim = target_relative_dim
        self.action_dim = action_dim
        self.direction_bins = direction_bins
        self.distance_bins = distance_bins
        self.text_context_length = text_context_length
        self.random_crop = random_crop
        self.wan_latent_cache_root = Path(wan_latent_cache_root) if wan_latent_cache_root else None
        self.ortrack_cache_root = Path(ortrack_cache_root) if ortrack_cache_root else None
        self.require_ortrack_cache = bool(require_ortrack_cache)
        self.require_tracker_features = bool(require_tracker_features)
        self.require_tracker_bbox = bool(require_tracker_bbox)
        self.require_tracker_response = bool(require_tracker_response)
        self.require_tracker_geometry = bool(require_tracker_geometry)
        self.tracker_feature_grid_size = max(int(tracker_feature_grid_size), 1)
        self.tracker_feature_dim = int(tracker_feature_dim)
        self.tracker_response_grid_size = max(int(tracker_response_grid_size), 1)
        self.require_tracker_finetune_inputs = bool(require_tracker_finetune_inputs)
        self.use_model_driven_tracker_crops = bool(use_model_driven_tracker_crops)
        self.require_target_boxes = bool(require_target_boxes)
        self.tracker_search_crop_jitter = bool(tracker_search_crop_jitter)
        self.tracker_search_center_jitter_std = max(float(tracker_search_center_jitter_std), 0.0)
        self.tracker_search_center_jitter_max = max(float(tracker_search_center_jitter_max), 0.0)
        self.tracker_search_scale_jitter = min(max(float(tracker_search_scale_jitter), 0.0), 0.95)
        self.localization_only = bool(localization_only)
        self.require_target_box_history = bool(require_target_box_history)
        self.target_history_tracker_cache_root = (
            Path(target_history_tracker_cache_root)
            if target_history_tracker_cache_root
            else None
        )
        self.target_history_length = int(target_history_length)
        self.target_history_partial_probability = min(
            max(float(target_history_partial_probability), 0.0), 1.0
        )
        self.target_history_center_jitter_std = max(
            float(target_history_center_jitter_std), 0.0
        )
        self.target_history_log_size_jitter_std = max(
            float(target_history_log_size_jitter_std), 0.0
        )
        self.target_history_confidence_dropout_probability = min(
            max(float(target_history_confidence_dropout_probability), 0.0), 1.0
        )
        if self.require_target_box_history:
            if self.target_history_tracker_cache_root is None:
                raise ValueError("Historical Target Memory requires a Tracker cache root.")
            if self.target_history_length < 2:
                raise ValueError("Historical Target Memory requires at least two states.")
        if (
            self.require_tracker_features
            or self.require_tracker_bbox
            or self.require_tracker_response
            or self.require_tracker_geometry
        ) and not self.require_ortrack_cache and not self.use_model_driven_tracker_crops:
            raise ValueError("Tracker conditions require the Tracker cache to be enabled.")
        self.guidance_heatmap_source = str(
            guidance_heatmap_source or ("tracker" if self.require_ortrack_cache else "none")
        ).strip().lower()
        if self.guidance_heatmap_source not in {"none", "gt", "tracker"}:
            raise ValueError("guidance_heatmap_source must be 'none', 'gt', or 'tracker'.")
        self.canonical_heatmap_sigma = float(canonical_heatmap_sigma)
        self.tracker_heatmap_target_mode = str(tracker_heatmap_target_mode).strip().lower()
        if self.tracker_heatmap_target_mode not in {"canonical", "raw", "raw_area"}:
            raise ValueError(
                "tracker_heatmap_target_mode must be 'canonical', 'raw', or 'raw_area'."
            )
        self._ortrack_summary_cache: Dict[str, Dict[str, Any]] = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _record_name(self, record: Dict[str, Any], index: Optional[int] = None) -> str:
        for key in ["id", "trajectory_id", "episode_id", "name"]:
            if key in record:
                return f"record {record[key]}"
        if index is not None:
            return f"record index {index}"
        return "record"

    def _sequence_length(self, record: Dict[str, Any], index: int) -> int:
        rgb_paths = record.get("rgb_paths")
        if rgb_paths is not None:
            return len(rgb_paths)
        for key in ["images", *self.REQUIRED_KEYS]:
            value = record.get(key)
            if value is None and key in self.LEGACY_KEY_ALIASES:
                value = record.get(self.LEGACY_KEY_ALIASES[key])
            if value is None:
                continue
            tensor = torch.load(value, map_location="cpu") if isinstance(value, str) and Path(value).exists() else torch.tensor(value)
            if tensor.ndim >= 1:
                return int(tensor.shape[0])
        raise KeyError(f"{self._record_name(record, index)} 无法推断序列长度。")

    def _select_window(self, length: int, minimum_start: int = 0) -> tuple[int, int]:
        if length <= 0:
            raise ValueError("trajectory length must be positive.")
        history_start = self.target_history_length - 1 if self.require_target_box_history else 0
        minimum_start = max(int(minimum_start), history_start)
        if minimum_start >= length:
            raise ValueError(
                f"Trajectory length {length} has no frames at or after required start "
                f"index {minimum_start}."
            )
        if length - minimum_start >= self.seq_len:
            start = (
                random.randint(minimum_start, length - self.seq_len)
                if self.random_crop
                else minimum_start
            )
            return start, start + self.seq_len
        return minimum_start, length

    def _first_valid_tracker_frame(
        self, record: Dict[str, Any], full_len: int, index: int
    ) -> int:
        boxes = self._ensure_2d(
            self._require_tensor_field(
                record, "target_bboxes_xywh", torch.float32, index
            ),
            full_len,
            4,
            "target_bboxes_xywh",
        )
        valid = (boxes[:, 2] > 1.0) & (boxes[:, 3] > 1.0)
        annotated_valid = self._load_tensor_field(
            record, "target_bbox_valid", torch.float32
        )
        if annotated_valid is not None:
            annotated_valid = self._ensure_2d(
                annotated_valid, full_len, 1, "target_bbox_valid"
            )[:, 0]
            valid &= annotated_valid > 0.5
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError(
                f"{self._record_name(record, index)} has no valid Tracker initialization box."
            )
        return int(valid_indices[0].item())

    def _load_target_box_history(
        self, record: Dict[str, Any], start: int, full_len: int, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.target_history_tracker_cache_root is None:
            raise RuntimeError("Missing Historical Target Memory Tracker cache root.")
        previous_length = self.target_history_length - 1
        first = start - previous_length
        if first < 0:
            raise ValueError("Target box history starts before frame zero.")
        scene = str(record.get("scene_id", "unknown_scene"))
        trajectory = str(record.get("trajectory_name", "unknown_trajectory"))
        summary_path = self.target_history_tracker_cache_root / scene / trajectory / "summary.json"
        summary = self._ortrack_summary_cache.get(str(summary_path))
        if summary is None:
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing target-history Tracker cache: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self._ortrack_summary_cache[str(summary_path)] = summary
        frames = summary.get("frames")
        if not isinstance(frames, list) or len(frames) != full_len:
            raise ValueError(
                f"Target-history Tracker cache length mismatch for {scene}/{trajectory}: "
                f"cache={len(frames) if isinstance(frames, list) else 'invalid'} dataset={full_len}."
            )
        image_h, image_w = (int(v) for v in summary.get("image_size", [640, 640]))
        rows, cache_valid = [], []
        for frame in frames[first:start]:
            box = frame.get("bbox_xywh")
            valid_box = isinstance(box, list) and len(box) == 4
            if valid_box:
                x, y, width, height = (float(v) for v in box)
                valid_box = width > 0.0 and height > 0.0
            else:
                x = y = width = height = 0.0
            rows.append([
                (x + 0.5 * width) / max(float(image_w), 1.0),
                (y + 0.5 * height) / max(float(image_h), 1.0),
                width / max(float(image_w), 1.0),
                height / max(float(image_h), 1.0),
                float(frame.get("confidence", 0.0)),
            ])
            cache_valid.append(bool(valid_box))
        if len(rows) != previous_length:
            raise AssertionError("Target history did not produce K-1 previous states.")
        return augment_target_box_history(
            torch.tensor(rows, dtype=torch.float32),
            torch.tensor(cache_valid, dtype=torch.bool),
            partial_history_probability=self.target_history_partial_probability,
            center_jitter_std=self.target_history_center_jitter_std,
            log_size_jitter_std=self.target_history_log_size_jitter_std,
            confidence_dropout_probability=(
                self.target_history_confidence_dropout_probability
            ),
        )

    def _load_rgb_sequence(self, record: Dict[str, Any], start: Optional[int] = None, end: Optional[int] = None) -> torch.Tensor:
        if "images" in record:
            value = record["images"]
            images = torch.load(value, map_location="cpu") if isinstance(value, str) else torch.tensor(value)
            if images.ndim != 4:
                raise ValueError("images must have shape [T, C, H, W].")
            if start is not None or end is not None:
                images = images[slice(start, end)]
            return images.float()
        rgb_paths = record.get("rgb_paths")
        if rgb_paths is None:
            raise KeyError("每条样本必须包含 images 或 rgb_paths。")
        if start is not None or end is not None:
            rgb_paths = rgb_paths[slice(start, end)]
        frames = []
        for p in rgb_paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.transform(img))
        if len(frames) == 0:
            raise ValueError("rgb_paths 不能为空。")
        return torch.stack(frames, dim=0)

    def _load_tracker_finetune_inputs(
        self, record: Dict[str, Any], start: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build fixed online-state Tracker crops at original RGB resolution."""
        if self.ortrack_cache_root is None:
            raise RuntimeError("Tracker fine-tuning requires --ortrack-cache-root.")
        scene = str(record.get("scene_id", "unknown_scene"))
        traj = str(record.get("trajectory_name", record.get("trajectory_id", "unknown_traj")))
        summary_path = self.ortrack_cache_root / scene / traj / "summary.json"
        summary = self._ortrack_summary_cache.get(str(summary_path))
        if summary is None:
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing ORTrack cache summary: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self._ortrack_summary_cache[str(summary_path)] = summary
        init_box = summary.get("init_bbox_xywh")
        frames = summary.get("frames")
        if not isinstance(init_box, list) or len(init_box) != 4 or not isinstance(frames, list):
            raise ValueError(f"Tracker cache is missing init bbox/frames: {summary_path}")
        geometry = frames[start].get("search_crop_xy_size")
        if not isinstance(geometry, list) or len(geometry) != 3:
            raise ValueError(f"Tracker cache is missing search geometry: {summary_path} frame {start}")
        rgb_paths = record.get("rgb_paths")
        if not isinstance(rgb_paths, list) or start >= len(rgb_paths):
            raise ValueError("Tracker fine-tuning requires rgb_paths for original-resolution crops.")
        with Image.open(rgb_paths[0]) as image:
            template_image = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
        with Image.open(rgb_paths[start]) as image:
            search_image = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
        template, _ = crop_target(template_image, init_box, 2.0, 128)
        x1, y1, side = (int(round(float(v))) for v in geometry)
        height, width = search_image.shape[-2:]
        left, top = max(-x1, 0), max(-y1, 0)
        right, bottom = max(x1 + side - width, 0), max(y1 + side - height, 0)
        padded = F.pad(search_image, (left, right, top, bottom), value=0.0)
        crop = padded[:, y1 + top:y1 + top + side, x1 + left:x1 + left + side]
        search = F.interpolate(crop.unsqueeze(0), (256, 256), mode="bilinear", align_corners=False)[0]
        return template, (search - TRACKER_MEAN) / TRACKER_STD

    def _load_model_driven_tracker_inputs(
        self, record: Dict[str, Any], start: int, full_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-force only the previous center; never load an external Tracker."""
        boxes = self._ensure_2d(
            self._require_tensor_field(record, "target_bboxes_xywh", torch.float32, -1),
            full_len,
            4,
            "target_bboxes_xywh",
        )
        init_index = self._first_valid_tracker_frame(record, full_len, -1)
        if start < init_index:
            raise ValueError(
                f"Tracker search start {start} precedes initialization frame {init_index}."
            )
        init_box = boxes[init_index]
        prev_index = max(int(start) - 1, 0)
        prev_box = boxes[prev_index]
        if float(prev_box[2]) <= 1.0 or float(prev_box[3]) <= 1.0:
            prev_box = init_box
        side = float(torch.maximum(init_box[2], init_box[3]).item())
        center_x = float((prev_box[0] + 0.5 * prev_box[2]).item())
        center_y = float((prev_box[1] + 0.5 * prev_box[3]).item())
        state_side = side
        if self.tracker_search_crop_jitter:
            search_side = 4.0 * side
            max_offset = self.tracker_search_center_jitter_max * search_side
            offset_x = max(
                min(random.gauss(0.0, self.tracker_search_center_jitter_std) * search_side, max_offset),
                -max_offset,
            )
            offset_y = max(
                min(random.gauss(0.0, self.tracker_search_center_jitter_std) * search_side, max_offset),
                -max_offset,
            )
            center_x += offset_x
            center_y += offset_y
            scale_delta = random.uniform(
                -self.tracker_search_scale_jitter, self.tracker_search_scale_jitter
            )
            state_side = max(side * (1.0 + scale_delta), 2.0)
        state_box = [
            center_x - 0.5 * state_side,
            center_y - 0.5 * state_side,
            state_side,
            state_side,
        ]
        rgb_paths = record.get("rgb_paths")
        if not isinstance(rgb_paths, list) or start >= len(rgb_paths):
            raise ValueError("Model-driven Tracker crops require rgb_paths for original-resolution images.")
        with Image.open(rgb_paths[init_index]) as image:
            template_image = torch.from_numpy(
                np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            ).permute(2, 0, 1).float().div_(255.0)
        with Image.open(rgb_paths[start]) as image:
            search_image = torch.from_numpy(
                np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            ).permute(2, 0, 1).float().div_(255.0)
        template, _ = crop_target(template_image, init_box.tolist(), 2.0, 128)
        search, geometry = crop_search(search_image, state_box, 4.0, 256)
        image_size = torch.tensor(
            [float(search_image.shape[-2]), float(search_image.shape[-1])], dtype=torch.float32
        )
        return template, search, torch.tensor(geometry, dtype=torch.float32), image_size

    def _normalized_target_boxes(
        self,
        record: Dict[str, Any],
        full_len: int,
        image_size: torch.Tensor,
    ) -> torch.Tensor:
        """Convert full-resolution xywh annotations to normalized cxcywh."""
        boxes = self._ensure_2d(
            self._require_tensor_field(record, "target_bboxes_xywh", torch.float32, -1),
            full_len,
            4,
            "target_bboxes_xywh",
        ).clone()
        image_h, image_w = image_size.float().clamp_min(1.0).unbind(-1)
        x, y, width, height = boxes.unbind(-1)
        valid = (width > 1.0) & (height > 1.0)
        normalized = torch.stack(
            [
                (x + 0.5 * width) / image_w,
                (y + 0.5 * height) / image_h,
                width / image_w,
                height / image_h,
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        return torch.where(valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))

    def _latent_cache_path(self, record: Dict[str, Any], start: int, end: int) -> Optional[Path]:
        if self.wan_latent_cache_root is None:
            return None
        scene = str(record.get("scene_id", "unknown_scene"))
        traj = str(record.get("trajectory_name", record.get("trajectory_id", "unknown_traj")))
        suffix = "" if self.action_video_freq_ratio == 1 else f"_video{self.action_video_freq_ratio}"
        return self.wan_latent_cache_root / scene / traj / f"seq{self.seq_len}{suffix}_start{start:04d}_end{end:04d}.pt"

    def _load_cached_wan_latents(self, record: Dict[str, Any], start: int, end: int) -> Optional[torch.Tensor]:
        path = self._latent_cache_path(record, start, end)
        if path is None or not path.exists():
            return None
        try:
            latents = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            latents = torch.load(path, map_location="cpu")
        if isinstance(latents, dict):
            latents = latents.get("latents")
        if not torch.is_tensor(latents) or latents.ndim != 4:
            raise ValueError(f"Invalid cached Wan latent at {path}: expected [C,T,H,W].")
        return latents.float()

    def _load_ortrack_window(
        self, record: Dict[str, Any], start: int, end: int, full_len: int
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if not self.require_ortrack_cache:
            return None, None, None, None, None, None, None, None
        if self.ortrack_cache_root is None:
            raise RuntimeError("ORTrack-guided training requires --ortrack-cache-root.")
        scene = str(record.get("scene_id", "unknown_scene"))
        traj = str(record.get("trajectory_name", record.get("trajectory_id", "unknown_traj")))
        cache_dir = self.ortrack_cache_root / scene / traj
        summary_path = cache_dir / "summary.json"
        summary = self._ortrack_summary_cache.get(str(summary_path))
        if summary is None:
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing ORTrack cache summary: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self._ortrack_summary_cache[str(summary_path)] = summary
        frames = summary.get("frames")
        if not isinstance(frames, list) or len(frames) != full_len:
            cached_len = len(frames) if isinstance(frames, list) else "invalid"
            raise ValueError(
                f"ORTrack cache length mismatch for {scene}/{traj}: cache={cached_len} dataset={full_len}"
            )
        if self.require_tracker_features:
            expected_grid = [self.tracker_feature_grid_size] * 2
            if summary.get("tracker_feature_grid_size") != expected_grid:
                raise ValueError(
                    f"Tracker feature grid mismatch for {scene}/{traj}: expected "
                    f"{expected_grid}, cache={summary.get('tracker_feature_grid_size')}. "
                    "Regenerate the Tracker feature cache for this model."
                )
            if int(summary.get("tracker_feature_dim", -1)) != self.tracker_feature_dim:
                raise ValueError(
                    f"Tracker feature dim mismatch for {scene}/{traj}: expected "
                    f"{self.tracker_feature_dim}, cache={summary.get('tracker_feature_dim')}."
                )
        heatmaps = []
        confidences = []
        centers = []
        bboxes = []
        responses = []
        features = []
        search_geometries = []
        image_sizes = []
        image_h, image_w = (int(v) for v in summary.get("image_size", [640, 640]))
        for frame in frames[start:end]:
            bbox = frame.get("bbox_xywh")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Missing tracker bbox for {scene}/{traj} frame {frame.get('frame')}")
            x, y, width, height = (float(v) for v in bbox)
            center = torch.tensor(
                [(x + 0.5 * width) / max(image_w, 1), (y + 0.5 * height) / max(image_h, 1)],
                dtype=torch.float32,
            ).clamp(0.0, 1.0)
            if self.tracker_heatmap_target_mode in {"raw", "raw_area"}:
                heatmap_key = (
                    "heatmap_direct_area"
                    if self.tracker_heatmap_target_mode == "raw_area"
                    else "heatmap"
                )
                heatmap_relpath = frame.get(heatmap_key)
                if not isinstance(heatmap_relpath, str) or not heatmap_relpath:
                    raise ValueError(
                        f"Missing {self.tracker_heatmap_target_mode} tracker heatmap for "
                        f"{scene}/{traj} frame {frame.get('frame')}"
                    )
                heatmap_path = cache_dir / heatmap_relpath
                if not heatmap_path.is_file():
                    raise FileNotFoundError(f"Missing raw tracker heatmap: {heatmap_path}")
                heatmap = torch.from_numpy(np.load(heatmap_path).astype(np.float32))
                if heatmap.ndim != 2:
                    raise ValueError(f"Raw tracker heatmap must be 2D: {heatmap_path}")
                if self.tracker_heatmap_target_mode == "raw" and tuple(heatmap.shape) != (64, 64):
                    heatmap = F.interpolate(
                        heatmap[None, None], size=(64, 64), mode="bilinear", align_corners=False
                    )[0, 0]
                heatmap = heatmap.clamp_min(0.0)
                heatmap = heatmap / heatmap.sum().clamp_min(1.0e-8)
            else:
                heatmap = make_canonical_heatmap(
                    center,
                    (64, 64),
                    sigma=self.canonical_heatmap_sigma,
                ).squeeze(0)
            heatmaps.append(heatmap)
            confidences.append(float(frame.get("confidence", 0.0)))
            centers.append(center)
            if self.require_tracker_geometry:
                geometry = frame.get("search_crop_xy_size")
                if not isinstance(geometry, list) or len(geometry) != 3:
                    raise ValueError(
                        f"Missing Tracker search geometry for {scene}/{traj} "
                        f"frame {frame.get('frame')}"
                    )
                search_geometries.append(torch.tensor(geometry, dtype=torch.float32))
                image_sizes.append(
                    torch.tensor([image_h, image_w], dtype=torch.float32)
                )
            if self.require_tracker_bbox:
                bboxes.append(
                    torch.tensor(
                        [
                            (x + 0.5 * width) / max(image_w, 1),
                            (y + 0.5 * height) / max(image_h, 1),
                            width / max(image_w, 1),
                            height / max(image_h, 1),
                        ],
                        dtype=torch.float32,
                    ).clamp(0.0, 1.0)
                )
            if self.require_tracker_response:
                response_relpath = frame.get("heatmap_direct_area")
                if not isinstance(response_relpath, str) or not response_relpath:
                    raise ValueError(
                        f"Missing Tracker direct response for {scene}/{traj} "
                        f"frame {frame.get('frame')}"
                    )
                response_path = cache_dir / response_relpath
                if not response_path.is_file():
                    raise FileNotFoundError(f"Missing Tracker direct response: {response_path}")
                response = torch.from_numpy(np.load(response_path).astype(np.float32))
                if response.ndim != 2:
                    raise ValueError(f"Tracker direct response must be 2D: {response_path}")
                expected = (self.tracker_response_grid_size, self.tracker_response_grid_size)
                if tuple(response.shape) != expected:
                    response = F.interpolate(
                        response[None, None], size=expected, mode="area"
                    )[0, 0]
                response = response.clamp_min(0.0)
                response = response / response.sum().clamp_min(1.0e-8)
                responses.append(response)
            if self.require_tracker_features:
                feature_relpath = frame.get("tracker_features")
                if not isinstance(feature_relpath, str) or not feature_relpath:
                    raise ValueError(
                        f"Missing Tracker features for {scene}/{traj} frame {frame.get('frame')}"
                    )
                feature_path = cache_dir / feature_relpath
                if not feature_path.is_file():
                    raise FileNotFoundError(f"Missing Tracker feature cache: {feature_path}")
                feature = torch.from_numpy(np.load(feature_path).astype(np.float32))
                if feature.ndim != 2:
                    raise ValueError(
                        f"Tracker features must have shape [tokens, channels]: {feature_path}"
                    )
                expected_feature_shape = (
                    self.tracker_feature_grid_size**2,
                    self.tracker_feature_dim,
                )
                if tuple(feature.shape) != expected_feature_shape:
                    raise ValueError(
                        f"Tracker features must have shape {expected_feature_shape}, "
                        f"got {tuple(feature.shape)}: {feature_path}"
                    )
                features.append(feature)
        maps = torch.stack(heatmaps)
        return (
            maps,
            torch.tensor(confidences, dtype=torch.float32),
            torch.stack(centers),
            torch.stack(features) if self.require_tracker_features else None,
            torch.stack(bboxes) if self.require_tracker_bbox else None,
            torch.stack(responses) if self.require_tracker_response else None,
            torch.stack(search_geometries) if self.require_tracker_geometry else None,
            torch.stack(image_sizes) if self.require_tracker_geometry else None,
        )

    def _load_gt_bbox_window(
        self, record: Dict[str, Any], start: int, end: int, full_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centers = self._ensure_2d(
            self._require_tensor_field(record, "target_bbox_centers", torch.float32, -1),
            full_len,
            2,
            "target_bbox_centers",
        )[start:end]
        valid = self._ensure_2d(
            self._require_tensor_field(record, "target_bbox_valid", torch.float32, -1),
            full_len,
            1,
            "target_bbox_valid",
        )[start:end, 0].clamp(0.0, 1.0)
        heatmaps = make_canonical_heatmap(
            centers,
            (64, 64),
            sigma=self.canonical_heatmap_sigma,
            valid=valid,
        ).squeeze(-3)
        return heatmaps, valid, centers

    def _load_tensor_field(self, record: Dict[str, Any], key: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if key not in record:
            return None
        value = record[key]
        if isinstance(value, str) and Path(value).exists():
            tensor = torch.load(value, map_location="cpu")
        else:
            tensor = torch.tensor(value)
        return tensor.to(dtype=dtype)

    def _require_tensor_field(self, record: Dict[str, Any], key: str, dtype: torch.dtype, index: int) -> torch.Tensor:
        tensor = self._load_tensor_field(record, key, dtype)
        if tensor is None and key in self.LEGACY_KEY_ALIASES:
            tensor = self._load_tensor_field(record, self.LEGACY_KEY_ALIASES[key], dtype)
        if tensor is None:
            legacy = self.LEGACY_KEY_ALIASES.get(key)
            suffix = f" 或旧字段 `{legacy}`" if legacy is not None else ""
            raise KeyError(f"{self._record_name(record, index)} 缺少必需字段 `{key}`{suffix}。")
        return tensor

    def _text_tokens_or_placeholder(self, record: Dict[str, Any], seq_len: int, index: int) -> Dict[str, Optional[torch.Tensor]]:
        text_tokens = self._load_tensor_field(record, "text_tokens", torch.long)
        attention_mask = self._load_tensor_field(record, "attention_mask", torch.long)
        if text_tokens is not None:
            return {"text_tokens": text_tokens.long(), "attention_mask": None if attention_mask is None else attention_mask.long()}

        if record.get("instructions") is None:
            raise KeyError(f"{self._record_name(record, index)} 需要提供 text_tokens 或 instructions。")
        # Wan2.2 consumes raw instruction strings. Keep placeholder token tensors
        # only to satisfy the shared model call signature and collate path.
        return {
            "text_tokens": torch.zeros(seq_len, 1, dtype=torch.long),
            "attention_mask": torch.ones(seq_len, 1, dtype=torch.long),
        }

    def _ensure_2d(self, x: torch.Tensor, length: int, dim: int, key: str) -> torch.Tensor:
        if x.ndim == 1:
            if dim == 1 and x.numel() == length:
                x = x.unsqueeze(-1)
            elif x.numel() == dim:
                x = x.unsqueeze(0).expand(length, -1)
            else:
                raise ValueError(f"`{key}` shape {tuple(x.shape)} cannot be aligned to [T={length}, D={dim}].")
        elif x.ndim == 2:
            if x.shape == (length, dim):
                pass
            elif x.shape == (1, dim):
                x = x.expand(length, -1)
            else:
                raise ValueError(f"`{key}` must have shape [T={length}, D={dim}] or [1, D], got {tuple(x.shape)}.")
        else:
            raise ValueError(f"`{key}` must be 1D or 2D, got shape {tuple(x.shape)}.")
        return x.float()

    def _ensure_1d_bins(self, x: torch.Tensor, length: int, key: str, num_bins: int) -> torch.Tensor:
        if x.ndim == 2 and x.size(-1) == 1:
            x = x.squeeze(-1)
        if x.ndim == 0:
            x = x.view(1).expand(length)
        elif x.ndim == 1:
            if x.numel() == length:
                pass
            elif x.numel() == 1:
                x = x.expand(length)
            else:
                raise ValueError(f"`{key}` must have length T={length} or length 1, got {tuple(x.shape)}.")
        else:
            raise ValueError(f"`{key}` must be 1D or [T,1], got shape {tuple(x.shape)}.")
        x = x.long()
        if torch.any((x < 0) | (x >= num_bins)):
            raise ValueError(f"`{key}` contains values outside [0, {num_bins - 1}].")
        return x

    def _slice_time_tensor(self, x: Optional[torch.Tensor], full_length: int, start: int, end: int) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if x.ndim >= 1 and x.shape[0] == full_length:
            return x[start:end]
        return x

    def _crop_or_pad(self, item: Dict[str, Optional[torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
        length = item["images"].shape[0]  # type: ignore[union-attr]
        static_keys = {"target_box_history", "target_box_history_valid"}
        if length >= self.seq_len:
            start = random.randint(0, length - self.seq_len) if self.random_crop else 0
            end = start + self.seq_len
            cropped: Dict[str, Optional[torch.Tensor]] = {}
            for k, v in item.items():
                if not isinstance(v, torch.Tensor):
                    cropped[k] = v
                elif k in static_keys:
                    cropped[k] = v
                elif v.ndim >= 1 and v.shape[0] == length:
                    cropped[k] = v[start:end]
                else:
                    cropped[k] = v
            cropped["valid_mask"] = torch.ones(self.seq_len, dtype=torch.float32)
            return cropped

        pad = self.seq_len - length
        padded: Dict[str, Optional[torch.Tensor]] = {}
        for k, v in item.items():
            if not isinstance(v, torch.Tensor):
                padded[k] = v
                continue
            if k in static_keys:
                padded[k] = v
                continue
            if v.ndim >= 1 and v.shape[0] == length:
                pad_value = v[-1:].expand(pad, *v.shape[1:])
                padded[k] = torch.cat([v, pad_value], dim=0)
            else:
                padded[k] = v
        valid_mask = torch.cat([torch.ones(length), torch.zeros(pad)], dim=0)
        padded["valid_mask"] = valid_mask.float()
        return padded

    def __getitem__(self, index: int) -> Dict[str, Optional[torch.Tensor]]:
        record = self.records[index]
        for key in self.REQUIRED_KEYS:
            legacy = self.LEGACY_KEY_ALIASES.get(key)
            if key not in record and (legacy is None or legacy not in record):
                suffix = f" 或旧字段 `{legacy}`" if legacy is not None else ""
                raise KeyError(f"{self._record_name(record, index)} 缺少必需字段 `{key}`{suffix}。")

        full_len = self._sequence_length(record, index)
        tracker_init_index = (
            self._first_valid_tracker_frame(record, full_len, index)
            if self.use_model_driven_tracker_crops
            else 0
        )
        start, end = self._select_window(full_len, tracker_init_index)
        if self.localization_only:
            if not self.use_model_driven_tracker_crops:
                raise RuntimeError("Standalone S0 training requires model-driven Tracker crops.")
            tracker_template, tracker_search, tracker_search_geometry, tracker_image_size = (
                self._load_model_driven_tracker_inputs(record, start, full_len)
            )
            target_center_valid = self._ensure_2d(
                self._require_tensor_field(record, "target_bbox_valid", torch.float32, index),
                full_len,
                1,
                "target_bbox_valid",
            )
            target_boxes = self._normalized_target_boxes(
                record, full_len, tracker_image_size
            )[start:end]
            return self._crop_or_pad(
                {
                    "images": torch.zeros(end - start, 0, dtype=torch.float32),
                    "target_boxes": target_boxes.float(),
                    "target_center_valid": target_center_valid[start:end].float(),
                    "tracker_search_geometry": tracker_search_geometry,
                    "tracker_image_size": tracker_image_size,
                    "tracker_template": tracker_template,
                    "tracker_search": tracker_search,
                }
            )
        video_latents = None
        if end - start == self.seq_len:
            video_latents = self._load_cached_wan_latents(record, start=start, end=end)
        # Keep real RGB even on cache hits. A distributed batch can contain a
        # mix of cached and uncached windows; collate falls back to RGB/VAE for
        # the whole batch unless every sample provides compatible latents.
        images = self._load_rgb_sequence(record, start=start, end=end)

        text = self._text_tokens_or_placeholder(record, full_len, index)
        raw_instructions = record.get("instructions")
        if isinstance(raw_instructions, list):
            instruction_text = str(raw_instructions[start] if start < len(raw_instructions) else raw_instructions[0])
        elif raw_instructions is None:
            instruction_text = ""
        else:
            instruction_text = str(raw_instructions)
        target_relative = self._ensure_2d(
            self._require_tensor_field(record, "target_relative", torch.float32, index),
            full_len,
            self.target_relative_dim,
            "target_relative",
        )
        target_centers = self._ensure_2d(
            self._require_tensor_field(record, "target_bbox_centers", torch.float32, index),
            full_len,
            2,
            "target_bbox_centers",
        )
        target_center_valid = self._ensure_2d(
            self._require_tensor_field(record, "target_bbox_valid", torch.float32, index),
            full_len,
            1,
            "target_bbox_valid",
        )
        prev_actions = self._ensure_2d(
            self._require_tensor_field(record, "prev_actions", torch.float32, index),
            full_len,
            self.action_dim,
            "prev_actions",
        )
        next_target_relative = self._ensure_2d(
            self._require_tensor_field(record, "next_target_relative", torch.float32, index),
            full_len,
            self.target_relative_dim,
            "next_target_relative",
        )
        expert_action = self._ensure_2d(
            self._require_tensor_field(record, "expert_action", torch.float32, index),
            full_len,
            self.action_dim,
            "expert_action",
        )
        tracker_template = tracker_search = None
        target_boxes = None
        target_box_history = target_box_history_valid = None
        if self.require_target_box_history:
            target_box_history, target_box_history_valid = self._load_target_box_history(
                record, start, full_len, index
            )
        if self.use_model_driven_tracker_crops:
            guidance_heatmap, guidance_confidence, tracker_center = None, None, None
            tracker_features = tracker_bbox = tracker_response = None
            tracker_template, tracker_search, tracker_search_geometry, tracker_image_size = (
                self._load_model_driven_tracker_inputs(record, start, full_len)
            )
        elif self.require_ortrack_cache:
            (
                guidance_heatmap,
                guidance_confidence,
                tracker_center,
                tracker_features,
                tracker_bbox,
                tracker_response,
                tracker_search_geometry,
                tracker_image_size,
            ) = self._load_ortrack_window(
                record, start, end, full_len
            )
            if self.guidance_heatmap_source != "tracker":
                guidance_heatmap, guidance_confidence = None, None
        elif self.guidance_heatmap_source == "gt":
            guidance_heatmap, guidance_confidence, tracker_center = self._load_gt_bbox_window(
                record, start, end, full_len
            )
            tracker_features = None
            tracker_bbox = None
            tracker_response = None
            tracker_search_geometry = None
            tracker_image_size = None
        else:
            guidance_heatmap, guidance_confidence, tracker_center = None, None, None
            tracker_features = None
            tracker_bbox = None
            tracker_response = None
            tracker_search_geometry = None
            tracker_image_size = None
        if self.require_tracker_finetune_inputs and not self.use_model_driven_tracker_crops:
            tracker_template, tracker_search = self._load_tracker_finetune_inputs(record, start)
        if self.require_target_boxes:
            if tracker_image_size is not None:
                box_image_size = tracker_image_size
            else:
                rgb_paths = record.get("rgb_paths")
                if not isinstance(rgb_paths, list) or start >= len(rgb_paths):
                    raise ValueError("Box supervision requires rgb_paths to determine image size.")
                with Image.open(rgb_paths[start]) as image:
                    box_image_size = torch.tensor(
                        [float(image.height), float(image.width)], dtype=torch.float32
                    )
            target_boxes = self._normalized_target_boxes(record, full_len, box_image_size)[start:end]
            target_centers = target_centers.clone()
            target_centers[start:end] = target_boxes[..., :2]
        item: Dict[str, Optional[torch.Tensor]] = {
            "images": images.float(),
            "text_tokens": self._slice_time_tensor(text["text_tokens"].long(), full_len, start, end),  # type: ignore[union-attr]
            "attention_mask": None if text["attention_mask"] is None else self._slice_time_tensor(text["attention_mask"].long(), full_len, start, end),
            "target_relative": target_relative[start:end].float(),
            "target_centers": target_centers[start:end].float(),
            "target_boxes": None if target_boxes is None else target_boxes.float(),
            "target_center_valid": target_center_valid[start:end].float(),
            "next_target_relative": next_target_relative[start:end].float(),
            "prev_actions": prev_actions[start:end].float(),
            "expert_action": expert_action[start:end].float(),
            "instructions": instruction_text,
            "guidance_heatmap": guidance_heatmap,
            "guidance_confidence": guidance_confidence,
            "tracker_center": tracker_center,
            "tracker_features": tracker_features,
            "tracker_bbox": tracker_bbox,
            "tracker_response": tracker_response,
            "tracker_search_geometry": tracker_search_geometry,
            "tracker_image_size": tracker_image_size,
            "tracker_template": tracker_template,
            "tracker_search": tracker_search,
            "target_box_history": target_box_history,
            "target_box_history_valid": target_box_history_valid,
        }
        if video_latents is not None:
            item["video_latents"] = video_latents
        return self._crop_or_pad(item)


def collate_fn(batch: List[Dict[str, Optional[torch.Tensor]]]) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {}
    keys = set().union(*(sample.keys() for sample in batch))
    for key in keys:
        values = [sample.get(key) for sample in batch]
        if key == "video_latents" and not all(torch.is_tensor(value) for value in values):
            out[key] = None
            continue
        if all(isinstance(v, str) for v in values):
            out[key] = values  # type: ignore[assignment]
            continue
        if all(v is None for v in values):
            out[key] = None
        elif any(v is None for v in values):
            raise ValueError(f"Batch 中 `{key}` 有的样本为 None、有的不是 None，请统一数据格式。")
        else:
            out[key] = torch.stack(values, dim=0)  # type: ignore[arg-type]
    return out


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _format_metrics(metrics: Dict[str, float]) -> str:
    metrics = dict(metrics)
    duplicate_aliases = {
        "total": "total_loss",
        "action": "action_flow_loss",
        "video": "video_flow_loss",
        "state_flow": "state_flow_loss",
        "current_box": "current_box_loss",
    }
    for short_name, explicit_name in duplicate_aliases.items():
        if explicit_name in metrics:
            metrics.pop(short_name, None)
    order = [
        "total_loss",
        "total",
        "action_flow_loss",
        "action",
        "video_flow_loss",
        "video",
        "state_flow_loss",
        "current_box_loss",
        "current_center_spatial",
        "current_box_giou",
        "current_attention",
        "localization_total",
        "predicted_future_state_error",
        "predicted_s0_box_error",
        "predicted_s0_center_error_pixels",
        "state_to_action_gate_mean",
        "current_box_action_gate_mean",
        "history_future_center",
        "history_future_center_error",
        "capture_value",
        "capture_value_ranking_accuracy",
        "capture_value_target_capture",
        "state_valid_ratio",
        "current_box_valid_ratio",
        "next_target_relative",
        "prior_next_target_relative",
        "kl",
    ]
    always_show = {
        "total_loss", "total", "action_flow_loss", "action",
        "video_flow_loss", "video", "state_flow_loss", "current_box_loss",
    }
    hidden_components: set[str] = set()
    parts = []
    for key in order:
        if key in metrics and (key in always_show or abs(metrics[key]) >= 1e-12):
            parts.append(f"{key}={metrics[key]:.4f}")
    for key in sorted(metrics.keys()):
        if key in order or key in hidden_components:
            continue
        if abs(metrics[key]) < 1e-12:
            continue
        parts.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(parts)


def _tracker_gate_summary(model: torch.nn.Module) -> Optional[str]:
    world_model = _unwrap_model(model)
    fastwam = getattr(world_model, "fastwam", None)
    fusion = None if fastwam is None else getattr(fastwam, "tracker_fusion", None)
    if fusion is None:
        return None
    gates = torch.tensor(
        [float(layer.gate.detach().float().cpu()) for layer in fusion.layers.values()],
        dtype=torch.float32,
    )
    if gates.numel() == 0:
        return None
    effective = gates.tanh()
    return (
        f"raw_min={gates.min().item():.6g} raw_mean={gates.mean().item():.6g} "
        f"raw_max={gates.max().item():.6g} effective_mean={effective.mean().item():.6g}"
    )


def _tqdm_train_postfix(avg: Dict[str, float]) -> Dict[str, str]:
    """Return only high-signal metrics that fit on one progress-bar line."""
    metric_specs = (
        (("total_loss", "total"), "tot", True),
        (("action_flow_loss", "action"), "act", True),
        (("video_flow_loss", "video"), "vid", True),
        (("state_flow_loss", "state_flow"), "state", False),
        (("current_box_loss", "current_box"), "box", False),
        (("current_center_spatial",), "ctr", False),
        (("current_box_giou",), "giou", False),
        (("current_attention",), "attn", False),
        (("predicted_s0_center_error_pixels",), "s0_px", False),
        (("predicted_future_state_error",), "future_err", False),
        (("predicted_s0_box_error",), "s0_err", False),
        (("state_to_action_gate_mean",), "s2a_gate", False),
        (("current_box_action_gate_mean",), "b0a_gate", False),
        (("history_future_center",), "hctr", False),
        (("history_future_center_error",), "hctr_err", False),
        (("capture_value",), "value", False),
        (("capture_value_ranking_accuracy",), "value_acc", False),
        (("fastwam_attention_heatmap",), "hm", False),
        (("fastwam_ortrack_consistency",), "trk", False),
        (("next_target_relative",), "next", False),
        (("prior_next_target_relative",), "prior", False),
        (("kl",), "kl", False),
    )
    out: Dict[str, str] = {}
    for source_names, display_name, always_show in metric_specs:
        source_name = next((name for name in source_names if name in avg), None)
        if source_name is None:
            continue
        value = avg[source_name]
        if always_show or abs(value) >= 1e-12:
            out[display_name] = f"{value:.4f}"
    return out


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    if isinstance(model, (torch.nn.DataParallel, DDP)):
        return model.module
    return model


def _trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Checkpoint trainable params plus frozen inherited weights needed at inference."""
    unwrapped = _unwrap_model(model)
    trainable_names = {name for name, param in unwrapped.named_parameters() if param.requires_grad}
    checkpoint_names = trainable_names | set(
        getattr(unwrapped, "_inference_checkpoint_parameter_names", ())
    )
    state = unwrapped.state_dict()
    return {name: tensor for name, tensor in state.items() if name in checkpoint_names}


def _freeze_for_target_conditioning_adapter_training(
    model: torch.nn.Module,
    inherited_parameter_names: Iterable[str],
    cfg: ModelConfig,
) -> tuple[int, int]:
    """Freeze the inherited policy and train only historical target memory."""
    root = "fastwam.target_action_conditioning."
    stage_prefixes = (
        f"{root}history_memory.",
        f"{root}history_adapter.",
        f"{root}next_center_delta_head.",
    )
    inherited = set(inherited_parameter_names)
    adapter_names = set()
    for name, parameter in model.named_parameters():
        train_adapter = name.startswith(stage_prefixes)
        parameter.requires_grad_(train_adapter)
        if train_adapter:
            adapter_names.add(name)
    if not adapter_names:
        raise RuntimeError("Target-conditioning adapter mode found no trainable parameters.")
    # Parent checkpoints intentionally omit separately loaded frozen encoders.
    # Retain their policy tensors together with the adapter so best.pt exactly
    # reconstructs the policy that was used during training.
    model._inference_checkpoint_parameter_names = inherited | adapter_names
    trainable_count = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return len(adapter_names), trainable_count


def _freeze_for_capture_value_training(
    model: torch.nn.Module,
    inherited_parameter_names: Iterable[str],
) -> tuple[int, int]:
    """Freeze the parent policy and train only the Capture-Value Head."""
    prefix = "fastwam.capture_value_head."
    inherited = set(inherited_parameter_names)
    value_names = set()
    for name, parameter in model.named_parameters():
        train_value = name.startswith(prefix)
        parameter.requires_grad_(train_value)
        if train_value:
            value_names.add(name)
    if not value_names:
        raise RuntimeError("Capture-value adapter mode found no Value Head parameters.")
    model._inference_checkpoint_parameter_names = inherited | value_names
    trainable_count = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return len(value_names), trainable_count


def _wan_latent_cache_stats(
    records: List[Dict[str, Any]],
    cache_root: str,
    seq_len: int,
    action_video_freq_ratio: int = 1,
    max_sample_records: int = 128,
) -> Optional[Dict[str, int]]:
    if not cache_root:
        return None
    root = Path(cache_root)
    if not root.exists():
        return {
            "records": len(records),
            "sampled_records": 0,
            "windows": 0,
            "hits": 0,
        }
    sample_count = min(len(records), max(int(max_sample_records), 1))
    if sample_count < len(records):
        sample_indices = [
            (index * (len(records) - 1)) // (sample_count - 1)
            for index in range(sample_count)
        ] if sample_count > 1 else [0]
        sampled_records = [records[index] for index in sample_indices]
    else:
        sampled_records = records

    windows = 0
    hits = 0
    for record in sampled_records:
        rgb_paths = record.get("rgb_paths") or []
        length = len(rgb_paths)
        starts = range(0, length - seq_len + 1) if length >= seq_len else range(0, 1)
        for start in starts:
            end = min(start + seq_len, length)
            if end - start != seq_len:
                continue
            windows += 1
            scene = str(record.get("scene_id", "unknown_scene"))
            traj = str(record.get("trajectory_name", record.get("trajectory_id", "unknown_traj")))
            suffix = "" if int(action_video_freq_ratio) <= 1 else f"_video{int(action_video_freq_ratio)}"
            path = root / scene / traj / f"seq{seq_len}{suffix}_start{start:04d}_end{end:04d}.pt"
            if path.exists():
                hits += 1
    return {
        "records": len(records),
        "sampled_records": len(sampled_records),
        "windows": windows,
        "hits": hits,
    }


def _ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main_process() -> bool:
    return _get_rank() == 0


_RECORD_INDEX_CACHE_VERSION = 1


def _record_index_cache_path(
    cache_root: Path,
    dataset_root: Path,
    scene_list: List[str],
    trajectory_range: str,
    max_vel: float,
    max_yaw_rate: float,
    max_speed_norm: float,
) -> Path:
    signature = {
        "version": _RECORD_INDEX_CACHE_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "scene_list": list(scene_list),
        "trajectory_range": str(trajectory_range),
        "max_vel": float(max_vel),
        "max_yaw_rate": float(max_yaw_rate),
        "max_speed_norm": float(max_speed_norm),
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return cache_root / "teacher_record_indexes" / f"records_{digest}.pkl"


def _build_records_once(
    dataset_root: Path,
    scene_list: List[str],
    trajectory_range: str,
    *,
    max_vel: float,
    max_yaw_rate: float,
    max_speed_norm: float,
    cache_root: Path,
    distributed: bool,
) -> List[Dict[str, Any]]:
    """Build the expensive trajectory index once instead of once per rank."""
    if not distributed or not _ddp_is_initialized():
        return build_records(
            dataset_root,
            scene_list,
            trajectory_range,
            max_vel=max_vel,
            max_yaw_rate=max_yaw_rate,
            max_speed_norm=max_speed_norm,
        )

    cache_path = _record_index_cache_path(
        cache_root,
        dataset_root,
        scene_list,
        trajectory_range,
        max_vel,
        max_yaw_rate,
        max_speed_norm,
    )
    records: Optional[List[Dict[str, Any]]] = None
    if _is_main_process():
        try:
            with cache_path.open("rb") as stream:
                records = pickle.load(stream)
            if not isinstance(records, list):
                raise TypeError("record index cache does not contain a list")
            print(f"[dataset-index] cache hit: {cache_path} ({len(records)} records)", flush=True)
        except (FileNotFoundError, EOFError, OSError, pickle.PickleError, TypeError):
            print(
                f"[dataset-index] rank 0 building {len(scene_list)}-scene index; "
                "other ranks are waiting",
                flush=True,
            )
            records = build_records(
                dataset_root,
                scene_list,
                trajectory_range,
                max_vel=max_vel,
                max_yaw_rate=max_yaw_rate,
                max_speed_norm=max_speed_norm,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
            try:
                with temporary_path.open("wb") as stream:
                    pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary_path, cache_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            print(f"[dataset-index] cached: {cache_path} ({len(records)} records)", flush=True)

    dist.barrier()
    if not _is_main_process():
        with cache_path.open("rb") as stream:
            records = pickle.load(stream)
        if not isinstance(records, list):
            raise TypeError(f"Invalid record index cache: {cache_path}")
    assert records is not None
    return records


def _reduce_metrics(metrics: Dict[str, float], device: torch.device, distributed: bool) -> Dict[str, float]:
    if not distributed or not metrics:
        return metrics
    keys = sorted(metrics.keys())
    values = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / float(_get_world_size())
    return {k: float(v.item()) for k, v in zip(keys, values)}


def _make_deepspeed_config(args: argparse.Namespace) -> Dict[str, Any]:
    world_size = max(_get_world_size(), 1)
    grad_accum = max(int(args.gradient_accumulation_steps), 1)
    micro_batch = int(args.batch_size)
    zero_optimization = {
        "stage": 1,
        "offload_param": {"device": "none"},
        "overlap_comm": False,
        "contiguous_gradients": False,
        "reduce_bucket_size": 2e8,
        "allgather_bucket_size": 2e8,
    }
    if bool(getattr(args, "deepspeed_offload_optimizer", False)):
        zero_optimization["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    return {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "train_batch_size": micro_batch * grad_accum * world_size,
        "bf16": {"enabled": True},
        "zero_optimization": zero_optimization,
        "zero_force_ds_cpu_optimizer": False,
        "gradient_clipping": float(args.grad_clip),
        "steps_per_print": 1000000,
    }


def _cosine_epoch_lr(base_lr: float, epoch: int, total_epochs: int, eta_min: float = 0.0) -> float:
    if total_epochs <= 0:
        return float(base_lr)
    progress = min(max(float(epoch) / float(total_epochs), 0.0), 1.0)
    return float(eta_min + 0.5 * (float(base_lr) - float(eta_min)) * (1.0 + math.cos(math.pi * progress)))


def _cosine_step_lr(base_lr: float, step: int, total_steps: int, eta_min: float = 0.0) -> float:
    if total_steps <= 0:
        return float(base_lr)
    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    return float(eta_min + 0.5 * (float(base_lr) - float(eta_min)) * (1.0 + math.cos(math.pi * progress)))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _init_swanlab(args: argparse.Namespace, cfg: ModelConfig, run_name: str):
    if not bool(getattr(args, "use_swanlab", False)) or not _is_main_process():
        return None
    try:
        import swanlab
    except Exception as exc:
        print(f"[swanlab] disabled: import failed ({exc})")
        return None
    try:
        return swanlab.init(
            project=args.swanlab_project,
            workspace=args.swanlab_workspace or None,
            experiment_name=run_name,
            logdir=args.swanlab_log_dir or None,
            mode=args.swanlab_mode,
            config={
                **cfg.__dict__,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "scene_list": args.scene_list,
                "trajectory_range": args.trajectory_range,
                "val_scene_list": args.val_scene_list,
                "val_trajectory_range": args.val_trajectory_range,
            },
        )
    except Exception as exc:
        print(f"[swanlab] disabled: init failed ({exc})")
        return None


def _swanlab_log(run, metrics: Dict[str, float], step: int, prefix: str) -> None:
    if run is None or not _is_main_process():
        return
    hidden_when_zero = {
        "kl",
        "next_target_relative",
        "prior_next_target_relative",
        "video_x0",
        "x0_action",
    }
    active = {
        k: float(v)
        for k, v in metrics.items()
        if not (k in hidden_when_zero and abs(float(v)) < 1e-12)
    }
    if not active:
        return
    try:
        import swanlab
        swanlab.log({f"{prefix}/{k}": v for k, v in active.items()}, step=step)
    except Exception as exc:
        print(f"[swanlab] log skipped: {exc}")


def _swanlab_finish(run) -> None:
    if run is None or not _is_main_process():
        return
    try:
        import swanlab
        swanlab.finish()
    except Exception as exc:
        print(f"[swanlab] finish skipped: {exc}")


def _s0_localization_loss(
    prediction: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    cfg: ModelConfig,
) -> Dict[str, torch.Tensor]:
    target_boxes = batch.get("target_boxes")
    target_valid = batch.get("target_center_valid")
    image_size = batch.get("tracker_image_size")
    if target_boxes is None or target_valid is None or image_size is None:
        raise RuntimeError("S0 localization supervision is incomplete.")
    current_valid = target_valid[:, 0]
    if current_valid.ndim > 1:
        current_valid = current_valid.squeeze(-1)
    valid_mask = batch.get("valid_mask")
    if valid_mask is not None:
        current_valid = current_valid.bool() & valid_mask[:, 0].bool()
    raw = CurrentTargetLocalizer.losses(
        prediction,
        target_boxes[:, 0],
        current_valid,
        image_size,
        float(cfg.current_attention_sigma),
    )
    total = (
        float(cfg.current_box_weight) * raw["box_l1"]
        + float(cfg.current_center_weight) * raw["center"]
        + float(cfg.current_box_giou_weight) * raw["giou"]
        + float(cfg.current_attention_weight) * raw["attention"]
    )
    return {
        "total": total,
        "current_box": raw["box_l1"],
        "current_center_spatial": raw["center"],
        "current_box_giou": raw["giou"],
        "current_attention": raw["attention"],
        "localization_total": total,
        "predicted_s0_box_error": raw["box_l1"],
        "predicted_s0_center_error_pixels": raw["center_error_pixels"],
        "current_box_valid_ratio": raw["valid_ratio"],
    }


def _forward_and_loss(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    cfg: ModelConfig,
    training_stage: str,
    *,
    localization_only: bool = False,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if training_stage == "s0":
        prediction = model(
            tracker_template=batch["tracker_template"],
            tracker_search=batch["tracker_search"],
            tracker_search_geometry=batch["tracker_search_geometry"],
            tracker_image_size=batch["tracker_image_size"],
        )
        return prediction, _s0_localization_loss(prediction, batch, cfg)
    outputs = model(
        images=batch["images"],
        text_tokens=batch["text_tokens"],
        target_relative=batch["target_relative"],
        prev_actions=batch["prev_actions"],
        attention_mask=batch["attention_mask"],
        expert_action=batch["expert_action"],
        valid_mask=batch["valid_mask"],
        done=batch.get("done"),
        instructions=batch.get("instructions"),
        video_latents=batch.get("video_latents"),
        guidance_heatmap=batch.get("guidance_heatmap"),
        guidance_confidence=batch.get("guidance_confidence"),
        tracker_center=batch.get("tracker_center"),
        tracker_features=batch.get("tracker_features"),
        tracker_bbox=batch.get("tracker_bbox"),
        tracker_response=batch.get("tracker_response"),
        tracker_search_geometry=batch.get("tracker_search_geometry"),
        tracker_image_size=batch.get("tracker_image_size"),
        tracker_template=batch.get("tracker_template"),
        tracker_search=batch.get("tracker_search"),
        target_centers=batch.get("target_centers"),
        target_boxes=batch.get("target_boxes"),
        target_center_valid=batch.get("target_center_valid"),
        target_box_history=batch.get("target_box_history"),
        target_box_history_valid=batch.get("target_box_history_valid"),
    )
    losses = world_model_dit_loss(
        outputs,
        batch,
        cfg,
        valid_mask=batch["valid_mask"],
        localization_only=localization_only,
    )
    return outputs, losses


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: ModelConfig,
    device: torch.device,
    training_stage: str = "joint",
) -> Dict[str, float]:
    model.eval()
    acc: Dict[str, float] = {}
    count = 0
    val_iter = loader
    if tqdm is not None:
        val_iter = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)
    for batch in val_iter:
        batch = move_batch_to_device(batch, device)
        _, losses = _forward_and_loss(model, batch, cfg, training_stage)
        summary = summarize_losses(losses)
        for k, v in summary.items():
            acc[k] = acc.get(k, 0.0) + v
        count += 1
    return {k: v / max(count, 1) for k, v in acc.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--scene-list", type=str, required=True)
    parser.add_argument("--trajectory-range", type=str, default="")
    parser.add_argument("--val-scene-list", type=str, default="")
    parser.add_argument("--val-trajectory-range", type=str, default="")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--val-every-epochs",
        type=int,
        default=1,
        help="Run validation every N epochs; the final epoch is always validated.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-vel", type=float, default=_DEFAULT_CFG.max_vel, help="Physical max velocity for action normalization.")
    parser.add_argument("--max-yaw-rate", type=float, default=_DEFAULT_CFG.max_yaw_rate, help="Physical max yaw rate for action normalization.")
    parser.add_argument("--max-speed-norm", type=float, default=_DEFAULT_CFG.max_speed_norm, help="Physical speed-norm cap used both in training targets and online action execution.")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument(
        "--training-stage",
        choices=["joint", "s0", "main"],
        default="joint",
        help="Train legacy joint model, standalone S0 localizer, or main V5 with frozen S0.",
    )
    parser.add_argument(
        "--s0-localizer-checkpoint",
        type=str,
        default="",
        help="Standalone S0 checkpoint required by --training-stage main.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seq-len", type=int, default=9)
    parser.add_argument("--target-relative-dim", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-train-steps", type=int, default=0, help="Stop after this many optimizer update steps; 0 keeps epoch-based training.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wan-latent-cache-root", type=str, default="")
    parser.add_argument("--ortrack-cache-root", type=str, default="")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-sequence-horizon", type=int, default=_DEFAULT_CFG.action_sequence_horizon)
    parser.add_argument("--action-video-freq-ratio", type=int, default=_DEFAULT_CFG.fastwam_action_video_freq_ratio)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Initialize model weights only; do not restore epoch, optimizer, or scheduler state.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every-epochs", type=int, default=1, help="Write last.pt every N epochs; always save on final epoch.")
    parser.add_argument("--save-best-checkpoint", type=_str2bool, default=True, help="Whether to write best.pt on checkpoint epochs.")
    parser.add_argument("--save-optimizer-state", type=_str2bool, default=True, help="Include optimizer/scheduler state in checkpoints when not using DeepSpeed.")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable multi-GPU training. Prefer --deepspeed; legacy DDP/DataParallel remains available when --deepspeed is off.",
    )
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed engine instead of DDP/DataParallel.")
    parser.add_argument("--deepspeed-config", type=str, default=None, help="Path to DeepSpeed JSON config.")
    parser.add_argument("--deepspeed-offload-optimizer", type=_str2bool, default=False)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1, help="Passed by DeepSpeed launcher.")
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=120,
        help="NCCL collective timeout; checkpointing this model can exceed 30 minutes.",
    )
    parser.add_argument("--use-swanlab", type=_str2bool, default=False)
    parser.add_argument("--swanlab-project", type=str, default="WAM-FastWAM")
    parser.add_argument("--swanlab-experiment-name", type=str, default=None)
    parser.add_argument("--swanlab-workspace", type=str, default="")
    parser.add_argument("--swanlab-log-dir", type=str, default=None)
    parser.add_argument("--swanlab-mode", type=str, default="cloud", choices=["cloud", "local", "offline", "disabled"])
    parser.add_argument("--use-wan22-encoders", type=_str2bool, default=_DEFAULT_CFG.use_wan22_encoders)
    parser.add_argument("--wan22-model-base-path", type=str, default=_DEFAULT_CFG.wan22_model_base_path)
    parser.add_argument("--wan22-fastwam-src-path", type=str, default=_DEFAULT_CFG.wan22_fastwam_src_path)
    parser.add_argument("--wan22-skip-download", type=_str2bool, default=_DEFAULT_CFG.wan22_skip_download)
    parser.add_argument("--wan22-text-context-length", type=int, default=_DEFAULT_CFG.wan22_text_context_length)
    parser.add_argument("--wan22-text-encode-batch-size", type=int, default=_DEFAULT_CFG.wan22_text_encode_batch_size)
    parser.add_argument(
        "--use-diffusion-actor",
        type=_str2bool,
        default=True,
        help="true/false: true=DiT diffusion denoising actor；false=MLP direct action head。",
    )
    parser.add_argument(
        "--use-fastwam-mot",
        type=_str2bool,
        default=_DEFAULT_CFG.use_fastwam_mot,
        help="true/false: true=FastWAM video/action MoT；false=legacy MLP/DiT actor path。",
    )
    parser.add_argument(
        "--target-token-fusion-mode",
        type=str,
        default=_DEFAULT_CFG.target_token_fusion_mode,
        choices=["attention", "concat"],
        help="attention=null target token participates in cross-attention；concat=append null target embedding after image/text fusion。",
    )
    parser.add_argument("--train-next-target-relative", type=_str2bool, default=_DEFAULT_CFG.train_next_target_relative)
    parser.add_argument("--train-rollout", type=_str2bool, default=False, help="Deprecated; prediction-head rollout supervision is disabled.")
    parser.add_argument("--next-target-relative-loss-weight", type=float, default=_DEFAULT_CFG.next_target_relative_loss_weight)
    parser.add_argument("--prior-target-relative-loss-weight", type=float, default=_DEFAULT_CFG.prior_target_relative_loss_weight)
    parser.add_argument("--rollout-loss-weight", type=float, default=_DEFAULT_CFG.rollout_loss_weight)
    parser.add_argument("--rollout-horizon", type=int, default=_DEFAULT_CFG.rollout_horizon)
    parser.add_argument("--direct-action-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-yaw-loss-weight", type=float, default=_DEFAULT_CFG.action_yaw_loss_weight)
    parser.add_argument("--x0-action-loss-weight", type=float, default=_DEFAULT_CFG.x0_action_loss_weight)
    parser.add_argument("--use-target-relative-context", type=_str2bool, default=_DEFAULT_CFG.use_target_relative_context)
    parser.add_argument("--target-relative-context-scale", type=float, default=_DEFAULT_CFG.target_relative_context_scale)
    parser.add_argument("--target-relative-token-scale", type=float, default=_DEFAULT_CFG.target_relative_token_scale)
    parser.add_argument("--target-relative-context-hidden-dim", type=int, default=_DEFAULT_CFG.target_relative_context_hidden_dim)
    parser.add_argument("--use-tracker-center-context", type=_str2bool, default=_DEFAULT_CFG.use_tracker_center_context)
    parser.add_argument("--tracker-center-context-hidden-dim", type=int, default=_DEFAULT_CFG.tracker_center_context_hidden_dim)
    parser.add_argument("--tracker-center-token-scale", type=float, default=_DEFAULT_CFG.tracker_center_token_scale)
    parser.add_argument(
        "--tracker-mot-integration",
        choices=[
            "none",
            "frozen_deit_tracker_fusion",
            "frozen_deit_tracker_local_feature",
            "mot_tracker_finetune_local_feature",
        ],
        default=_DEFAULT_CFG.tracker_mot_integration,
    )
    parser.add_argument("--tracker-feature-dim", type=int, default=_DEFAULT_CFG.tracker_feature_dim)
    parser.add_argument("--tracker-finetune-checkpoint", type=str, default="")
    parser.add_argument(
        "--tracker-finetune-init",
        choices=["uav_tracker", "imagenet_deit"],
        default=_DEFAULT_CFG.tracker_finetune_init,
    )
    parser.add_argument("--tracker-backbone-pretrained-path", type=str, default="")
    parser.add_argument("--tracker-template-size", type=int, default=_DEFAULT_CFG.tracker_template_size)
    parser.add_argument("--tracker-search-size", type=int, default=_DEFAULT_CFG.tracker_search_size)
    parser.add_argument("--tracker-feature-grid-size", type=int, default=_DEFAULT_CFG.tracker_feature_grid_size)
    parser.add_argument(
        "--tracker-use-local-position-embedding",
        type=_str2bool,
        default=_DEFAULT_CFG.tracker_use_local_position_embedding,
    )
    parser.add_argument(
        "--tracker-include-box-token",
        type=_str2bool,
        default=_DEFAULT_CFG.tracker_include_box_token,
    )
    parser.add_argument(
        "--tracker-condition-mode",
        choices=[
            "none", "center", "bbox", "features", "response",
            "center_features", "bbox_response", "bbox_response_features",
        ],
        default=_DEFAULT_CFG.tracker_condition_mode,
    )
    parser.add_argument("--tracker-response-grid-size", type=int, default=_DEFAULT_CFG.tracker_response_grid_size)
    parser.add_argument("--tracker-fusion-gate-init", type=float, default=_DEFAULT_CFG.tracker_fusion_gate_init)
    parser.add_argument("--tracker-feature-context-hidden-dim", type=int, default=_DEFAULT_CFG.tracker_feature_context_hidden_dim)
    parser.add_argument("--tracker-feature-token-scale", type=float, default=_DEFAULT_CFG.tracker_feature_token_scale)
    parser.add_argument(
        "--tracker-feature-confidence-gate",
        type=_str2bool,
        default=_DEFAULT_CFG.tracker_feature_confidence_gate,
    )
    parser.add_argument("--tracker-fusion-start-layer", type=int, default=_DEFAULT_CFG.tracker_fusion_start_layer)
    parser.add_argument(
        "--tracker-future-target-alignment",
        type=_str2bool,
        default=_DEFAULT_CFG.tracker_future_target_alignment,
    )
    parser.add_argument(
        "--tracker-future-target-start-layer",
        type=int,
        default=_DEFAULT_CFG.tracker_future_target_start_layer,
    )
    parser.add_argument(
        "--tracker-model-driven-search",
        type=_str2bool,
        default=_DEFAULT_CFG.tracker_model_driven_search,
    )
    parser.add_argument(
        "--tracker-center-flow-supervision",
        type=_str2bool,
        default=_DEFAULT_CFG.tracker_center_flow_supervision,
    )
    parser.add_argument(
        "--tracker-center-flow-loss-weight",
        type=float,
        default=_DEFAULT_CFG.tracker_center_flow_loss_weight,
    )
    parser.add_argument(
        "--tracker-search-crop-jitter",
        type=_str2bool,
        default=_DEFAULT_CFG.tracker_search_crop_jitter,
    )
    parser.add_argument(
        "--tracker-search-center-jitter-std",
        type=float,
        default=_DEFAULT_CFG.tracker_search_center_jitter_std,
    )
    parser.add_argument(
        "--tracker-search-center-jitter-max",
        type=float,
        default=_DEFAULT_CFG.tracker_search_center_jitter_max,
    )
    parser.add_argument(
        "--tracker-search-scale-jitter",
        type=float,
        default=_DEFAULT_CFG.tracker_search_scale_jitter,
    )
    parser.add_argument(
        "--tracker-state-action-alignment-version",
        type=int,
        default=_DEFAULT_CFG.tracker_state_action_alignment_version,
    )
    parser.add_argument(
        "--tracker-current-center-loss-weight",
        type=float,
        default=_DEFAULT_CFG.tracker_current_center_loss_weight,
    )
    parser.add_argument(
        "--tracker-future-center-loss-weight",
        type=float,
        default=_DEFAULT_CFG.tracker_future_center_loss_weight,
    )
    parser.add_argument(
        "--tracker-center-transition-loss-weight",
        type=float,
        default=_DEFAULT_CFG.tracker_center_transition_loss_weight,
    )
    parser.add_argument(
        "--tracker-box-l1-loss-weight",
        type=float,
        default=_DEFAULT_CFG.tracker_box_l1_loss_weight,
    )
    parser.add_argument(
        "--tracker-box-giou-loss-weight",
        type=float,
        default=_DEFAULT_CFG.tracker_box_giou_loss_weight,
    )
    parser.add_argument(
        "--tracker-future-horizon-discount",
        type=float,
        default=_DEFAULT_CFG.tracker_future_horizon_discount,
    )
    parser.add_argument(
        "--tracker-spatial-cross-attention",
        type=_str2bool,
        default=_DEFAULT_CFG.tracker_spatial_cross_attention,
    )
    parser.add_argument("--use-future-state-dit", type=_str2bool, default=_DEFAULT_CFG.use_future_state_dit)
    parser.add_argument("--future-state-dim", type=int, default=_DEFAULT_CFG.future_state_dim)
    parser.add_argument("--future-state-horizon", type=int, default=_DEFAULT_CFG.future_state_horizon)
    parser.add_argument("--future-state-hidden-dim", type=int, default=_DEFAULT_CFG.future_state_hidden_dim)
    parser.add_argument("--future-state-ffn-dim", type=int, default=_DEFAULT_CFG.future_state_ffn_dim)
    parser.add_argument("--future-state-num-layers", type=int, default=_DEFAULT_CFG.future_state_num_layers)
    parser.add_argument("--future-state-flow-weight", type=float, default=_DEFAULT_CFG.future_state_flow_weight)
    parser.add_argument("--current-box-weight", type=float, default=_DEFAULT_CFG.current_box_weight)
    parser.add_argument("--current-center-weight", type=float, default=_DEFAULT_CFG.current_center_weight)
    parser.add_argument(
        "--current-box-giou-weight", type=float, default=_DEFAULT_CFG.current_box_giou_weight
    )
    parser.add_argument(
        "--current-attention-weight", type=float, default=_DEFAULT_CFG.current_attention_weight
    )
    parser.add_argument(
        "--current-attention-sigma", type=float, default=_DEFAULT_CFG.current_attention_sigma
    )
    parser.add_argument(
        "--localization-warmup-steps",
        type=int,
        default=_DEFAULT_CFG.localization_warmup_steps,
        help="For structured future models, optimize only b0 localization for this many steps.",
    )
    parser.add_argument(
        "--use-current-box-action-conditioning",
        type=_str2bool,
        default=_DEFAULT_CFG.use_current_box_action_conditioning,
    )
    parser.add_argument(
        "--freeze-current-box-action-conditioner",
        type=_str2bool,
        default=_DEFAULT_CFG.freeze_current_box_action_conditioner,
    )
    parser.add_argument(
        "--use-historical-target-memory",
        type=_str2bool,
        default=_DEFAULT_CFG.use_historical_target_memory,
    )
    parser.add_argument(
        "--target-history-length",
        type=int,
        default=_DEFAULT_CFG.target_history_length,
    )
    parser.add_argument(
        "--target-history-hidden-dim",
        type=int,
        default=_DEFAULT_CFG.target_history_hidden_dim,
    )
    parser.add_argument(
        "--target-history-num-layers",
        type=int,
        default=_DEFAULT_CFG.target_history_num_layers,
    )
    parser.add_argument(
        "--target-history-num-heads",
        type=int,
        default=_DEFAULT_CFG.target_history_num_heads,
    )
    parser.add_argument(
        "--target-history-action-layers",
        type=int,
        nargs="+",
        default=list(_DEFAULT_CFG.target_history_action_layers),
    )
    parser.add_argument(
        "--target-history-future-center-loss-weight",
        type=float,
        default=_DEFAULT_CFG.target_history_future_center_loss_weight,
    )
    parser.add_argument(
        "--target-history-tracker-cache-root",
        type=str,
        default=_DEFAULT_CFG.target_history_tracker_cache_root,
    )
    parser.add_argument(
        "--target-conditioning-adapter-only",
        type=_str2bool,
        default=_DEFAULT_CFG.target_conditioning_adapter_only,
    )
    parser.add_argument(
        "--use-capture-value-reranking",
        type=_str2bool,
        default=_DEFAULT_CFG.use_capture_value_reranking,
    )
    parser.add_argument(
        "--capture-value-candidate-count",
        type=int,
        default=_DEFAULT_CFG.capture_value_candidate_count,
    )
    parser.add_argument(
        "--capture-value-hidden-dim",
        type=int,
        default=_DEFAULT_CFG.capture_value_hidden_dim,
    )
    parser.add_argument(
        "--capture-value-num-layers",
        type=int,
        default=_DEFAULT_CFG.capture_value_num_layers,
    )
    parser.add_argument(
        "--capture-value-num-heads",
        type=int,
        default=_DEFAULT_CFG.capture_value_num_heads,
    )
    parser.add_argument(
        "--capture-value-loss-weight",
        type=float,
        default=_DEFAULT_CFG.capture_value_loss_weight,
    )
    parser.add_argument(
        "--capture-value-candidate-noise-std",
        type=float,
        default=_DEFAULT_CFG.capture_value_candidate_noise_std,
    )
    parser.add_argument(
        "--capture-value-capture-distance",
        type=float,
        default=_DEFAULT_CFG.capture_value_capture_distance,
    )
    parser.add_argument(
        "--capture-value-distance-score-weight",
        type=float,
        default=_DEFAULT_CFG.capture_value_distance_score_weight,
    )
    parser.add_argument(
        "--capture-value-visibility-score-weight",
        type=float,
        default=_DEFAULT_CFG.capture_value_visibility_score_weight,
    )
    parser.add_argument(
        "--capture-value-adapter-only",
        type=_str2bool,
        default=_DEFAULT_CFG.capture_value_adapter_only,
    )
    parser.add_argument("--target-history-partial-probability", type=float, default=0.5)
    parser.add_argument("--target-history-center-jitter-std", type=float, default=0.01)
    parser.add_argument("--target-history-log-size-jitter-std", type=float, default=0.05)
    parser.add_argument(
        "--target-history-confidence-dropout-probability", type=float, default=0.1
    )
    parser.add_argument(
        "--current-box-action-layers",
        type=int,
        nargs="+",
        default=list(_DEFAULT_CFG.current_box_action_layers),
    )
    parser.add_argument(
        "--current-box-action-hidden-dim",
        type=int,
        default=_DEFAULT_CFG.current_box_action_hidden_dim,
    )
    parser.add_argument(
        "--current-box-action-gate-init",
        type=float,
        default=_DEFAULT_CFG.current_box_action_gate_init,
    )
    parser.add_argument(
        "--use-tracker-memory", type=_str2bool, default=_DEFAULT_CFG.use_tracker_memory
    )
    parser.add_argument("--tracker-expert-hidden-dim", type=int, default=_DEFAULT_CFG.tracker_expert_hidden_dim)
    parser.add_argument("--tracker-expert-ffn-dim", type=int, default=_DEFAULT_CFG.tracker_expert_ffn_dim)
    parser.add_argument(
        "--tracker-heatmap-target-mode",
        choices=["canonical", "raw", "raw_area"],
        default=_DEFAULT_CFG.tracker_heatmap_target_mode,
    )
    parser.add_argument(
        "--tracker-attention-query-mode",
        choices=["query0", "all_queries"],
        default=_DEFAULT_CFG.tracker_attention_query_mode,
    )
    parser.add_argument("--fastwam-lambda-action", type=float, default=_DEFAULT_CFG.fastwam_lambda_action)
    parser.add_argument("--fastwam-lambda-video", type=float, default=_DEFAULT_CFG.fastwam_lambda_video)
    parser.add_argument("--fastwam-skip-dit-load-from-pretrain", type=_str2bool, default=_DEFAULT_CFG.fastwam_skip_dit_load_from_pretrain)
    parser.add_argument("--fastwam-action-dit-pretrained-path", type=str, default=_DEFAULT_CFG.fastwam_action_dit_pretrained_path)
    parser.add_argument("--fastwam-mot-checkpoint-mixed-attn", type=_str2bool, default=_DEFAULT_CFG.fastwam_mot_checkpoint_mixed_attn)
    parser.add_argument("--use-fasterwam-dot", type=_str2bool, default=_DEFAULT_CFG.use_fasterwam_dot)
    parser.add_argument("--use-fastwam-attention-heatmap-loss", type=_str2bool, default=_DEFAULT_CFG.use_fastwam_attention_heatmap_loss)
    parser.add_argument("--use-fastwam-tracker-heatmap-loss", type=_str2bool, default=_DEFAULT_CFG.use_fastwam_tracker_heatmap_loss)
    parser.add_argument("--use-fastwam-attention-bias", type=_str2bool, default=_DEFAULT_CFG.use_fastwam_attention_bias)
    parser.add_argument("--use-gt-center-attention-bias", type=_str2bool, default=_DEFAULT_CFG.use_gt_center_attention_bias)
    parser.add_argument("--gt-center-attention-sigma", type=float, default=_DEFAULT_CFG.gt_center_attention_sigma)
    parser.add_argument("--gt-center-attention-beta", type=float, default=_DEFAULT_CFG.gt_center_attention_beta)
    parser.add_argument("--gt-center-attention-zero-mean", type=_str2bool, default=_DEFAULT_CFG.gt_center_attention_zero_mean)
    parser.add_argument("--gt-center-guided-layers", type=int, default=_DEFAULT_CFG.gt_center_guided_layers)
    parser.add_argument("--gt-center-guided-head-ratio", type=float, default=_DEFAULT_CFG.gt_center_guided_head_ratio)
    parser.add_argument("--fastwam-heatmap-source", choices=["none", "gt", "tracker"], default=_DEFAULT_CFG.fastwam_heatmap_source)
    parser.add_argument("--fastwam-attention-heatmap-loss-weight", type=float, default=_DEFAULT_CFG.fastwam_attention_heatmap_loss_weight)
    parser.add_argument("--fastwam-attention-heatmap-sigma", type=float, default=_DEFAULT_CFG.fastwam_attention_heatmap_sigma)
    parser.add_argument("--fastwam-attention-heatmap-fov-deg", type=float, default=_DEFAULT_CFG.fastwam_attention_heatmap_fov_deg)
    parser.add_argument("--use-fastwam-heatmap-guidance", type=_str2bool, default=_DEFAULT_CFG.use_fastwam_heatmap_guidance)
    parser.add_argument("--fastwam-heatmap-guidance-scale", type=float, default=_DEFAULT_CFG.fastwam_heatmap_guidance_scale)
    parser.add_argument("--fastwam-heatmap-guidance-sigma", type=float, default=_DEFAULT_CFG.fastwam_heatmap_guidance_sigma)
    parser.add_argument("--fastwam-heatmap-guidance-fov-deg", type=float, default=_DEFAULT_CFG.fastwam_heatmap_guidance_fov_deg)
    parser.add_argument("--fastwam-ortrack-consistency-loss-weight", type=float, default=_DEFAULT_CFG.fastwam_ortrack_consistency_loss_weight)
    args = parser.parse_args()

    seed_everything(args.seed + _get_rank())
    save_dir = Path(args.save_dir)
    if _is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    use_deepspeed = bool(args.deepspeed)
    if use_deepspeed and deepspeed is None:
        raise ImportError("DeepSpeed is not installed in this environment.")
    use_distributed = (use_deepspeed or args.multi_gpu) and torch.cuda.is_available() and _get_world_size() > 1
    use_ddp = (not use_deepspeed) and args.multi_gpu and torch.cuda.is_available() and _get_world_size() > 1
    if use_deepspeed:
        torch.cuda.set_device(_get_local_rank())
        if dist.is_available() and not dist.is_initialized() and _get_world_size() > 1:
            deepspeed.init_distributed(
                dist_backend="nccl",
                timeout=timedelta(minutes=max(int(args.distributed_timeout_minutes), 1)),
            )
        device = torch.device("cuda", _get_local_rank())
    elif use_ddp:
        torch.cuda.set_device(_get_local_rank())
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=max(int(args.distributed_timeout_minutes), 1)),
        )
        device = torch.device("cuda", _get_local_rank())
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = ModelConfig(
        image_size=args.image_size,
        text_context_length=args.wan22_text_context_length if args.use_wan22_encoders else _DEFAULT_CFG.text_context_length,
        use_wan22_encoders=args.use_wan22_encoders,
        wan22_model_base_path=args.wan22_model_base_path,
        wan22_fastwam_src_path=args.wan22_fastwam_src_path,
        wan22_skip_download=args.wan22_skip_download,
        wan22_text_context_length=args.wan22_text_context_length,
        wan22_text_encode_batch_size=args.wan22_text_encode_batch_size,
        target_relative_dim=args.target_relative_dim,
        action_dim=args.action_dim,
        action_sequence_horizon=args.action_sequence_horizon,
        fastwam_action_video_freq_ratio=max(int(args.action_video_freq_ratio), 1),
        action_diffusion_steps=args.diffusion_steps,
        action_sampling_steps=args.sampling_steps,
        max_vel=args.max_vel,
        max_yaw_rate=args.max_yaw_rate,
        max_speed_norm=args.max_speed_norm,
        target_token_fusion_mode=args.target_token_fusion_mode,
        use_target_relative_context=args.use_target_relative_context,
        target_relative_context_scale=args.target_relative_context_scale,
        target_relative_token_scale=args.target_relative_token_scale,
        target_relative_context_hidden_dim=args.target_relative_context_hidden_dim,
        use_tracker_center_context=args.use_tracker_center_context,
        tracker_center_context_hidden_dim=args.tracker_center_context_hidden_dim,
        tracker_center_token_scale=args.tracker_center_token_scale,
        tracker_mot_integration=args.tracker_mot_integration,
        tracker_finetune_checkpoint=args.tracker_finetune_checkpoint,
        tracker_finetune_init=args.tracker_finetune_init,
        tracker_backbone_pretrained_path=args.tracker_backbone_pretrained_path,
        tracker_template_size=args.tracker_template_size,
        tracker_search_size=args.tracker_search_size,
        tracker_feature_dim=args.tracker_feature_dim,
        tracker_feature_grid_size=args.tracker_feature_grid_size,
        tracker_use_local_position_embedding=args.tracker_use_local_position_embedding,
        tracker_include_box_token=args.tracker_include_box_token,
        tracker_condition_mode=args.tracker_condition_mode,
        tracker_response_grid_size=args.tracker_response_grid_size,
        tracker_fusion_gate_init=args.tracker_fusion_gate_init,
        tracker_feature_context_hidden_dim=args.tracker_feature_context_hidden_dim,
        tracker_feature_token_scale=args.tracker_feature_token_scale,
        tracker_feature_confidence_gate=args.tracker_feature_confidence_gate,
        tracker_fusion_start_layer=args.tracker_fusion_start_layer,
        tracker_future_target_alignment=args.tracker_future_target_alignment,
        tracker_future_target_start_layer=args.tracker_future_target_start_layer,
        tracker_model_driven_search=args.tracker_model_driven_search,
        tracker_center_flow_supervision=args.tracker_center_flow_supervision,
        tracker_center_flow_loss_weight=args.tracker_center_flow_loss_weight,
        tracker_search_crop_jitter=args.tracker_search_crop_jitter,
        tracker_search_center_jitter_std=args.tracker_search_center_jitter_std,
        tracker_search_center_jitter_max=args.tracker_search_center_jitter_max,
        tracker_search_scale_jitter=args.tracker_search_scale_jitter,
        tracker_state_action_alignment_version=args.tracker_state_action_alignment_version,
        tracker_current_center_loss_weight=args.tracker_current_center_loss_weight,
        tracker_future_center_loss_weight=args.tracker_future_center_loss_weight,
        tracker_center_transition_loss_weight=args.tracker_center_transition_loss_weight,
        tracker_box_l1_loss_weight=args.tracker_box_l1_loss_weight,
        tracker_box_giou_loss_weight=args.tracker_box_giou_loss_weight,
        tracker_future_horizon_discount=args.tracker_future_horizon_discount,
        tracker_spatial_cross_attention=args.tracker_spatial_cross_attention,
        use_future_state_dit=args.use_future_state_dit,
        future_state_dim=args.future_state_dim,
        future_state_horizon=args.future_state_horizon,
        future_state_hidden_dim=args.future_state_hidden_dim,
        future_state_ffn_dim=args.future_state_ffn_dim,
        future_state_num_layers=args.future_state_num_layers,
        future_state_flow_weight=args.future_state_flow_weight,
        current_box_weight=args.current_box_weight,
        current_center_weight=args.current_center_weight,
        current_box_giou_weight=args.current_box_giou_weight,
        current_attention_weight=args.current_attention_weight,
        current_attention_sigma=args.current_attention_sigma,
        localization_warmup_steps=max(int(args.localization_warmup_steps), 0),
        include_current_localization_loss=(args.training_stage != "main"),
        use_current_box_action_conditioning=args.use_current_box_action_conditioning,
        current_box_action_layers=tuple(int(value) for value in args.current_box_action_layers),
        current_box_action_hidden_dim=args.current_box_action_hidden_dim,
        current_box_action_gate_init=args.current_box_action_gate_init,
        freeze_current_box_action_conditioner=args.freeze_current_box_action_conditioner,
        use_historical_target_memory=args.use_historical_target_memory,
        target_history_length=args.target_history_length,
        target_history_hidden_dim=args.target_history_hidden_dim,
        target_history_num_layers=args.target_history_num_layers,
        target_history_num_heads=args.target_history_num_heads,
        target_history_action_layers=tuple(
            int(value) for value in args.target_history_action_layers
        ),
        target_history_future_center_loss_weight=(
            args.target_history_future_center_loss_weight
        ),
        target_history_tracker_cache_root=args.target_history_tracker_cache_root,
        target_conditioning_adapter_only=args.target_conditioning_adapter_only,
        use_capture_value_reranking=args.use_capture_value_reranking,
        capture_value_candidate_count=args.capture_value_candidate_count,
        capture_value_hidden_dim=args.capture_value_hidden_dim,
        capture_value_num_layers=args.capture_value_num_layers,
        capture_value_num_heads=args.capture_value_num_heads,
        capture_value_loss_weight=args.capture_value_loss_weight,
        capture_value_candidate_noise_std=args.capture_value_candidate_noise_std,
        capture_value_capture_distance=args.capture_value_capture_distance,
        capture_value_distance_score_weight=args.capture_value_distance_score_weight,
        capture_value_visibility_score_weight=args.capture_value_visibility_score_weight,
        capture_value_adapter_only=args.capture_value_adapter_only,
        use_tracker_memory=args.use_tracker_memory,
        tracker_expert_hidden_dim=args.tracker_expert_hidden_dim,
        tracker_expert_ffn_dim=args.tracker_expert_ffn_dim,
        tracker_heatmap_target_mode=args.tracker_heatmap_target_mode,
        tracker_attention_query_mode=args.tracker_attention_query_mode,
        use_diffusion_actor=args.use_diffusion_actor,
        use_fastwam_mot=args.use_fastwam_mot,
        use_rssm=False,
        train_kl=False,
        train_direct_action=True,
        train_next_target_relative=args.train_next_target_relative,
        train_rollout=False,
        next_target_relative_loss_weight=args.next_target_relative_loss_weight,
        prior_target_relative_loss_weight=args.prior_target_relative_loss_weight,
        rollout_loss_weight=args.rollout_loss_weight,
        rollout_horizon=args.rollout_horizon,
        direct_action_loss_weight=args.direct_action_loss_weight,
        action_yaw_loss_weight=args.action_yaw_loss_weight,
        x0_action_loss_weight=args.x0_action_loss_weight,
        fastwam_lambda_action=args.fastwam_lambda_action,
        fastwam_lambda_video=args.fastwam_lambda_video,
        fastwam_skip_dit_load_from_pretrain=args.fastwam_skip_dit_load_from_pretrain,
        fastwam_action_dit_pretrained_path=args.fastwam_action_dit_pretrained_path,
        fastwam_mot_checkpoint_mixed_attn=args.fastwam_mot_checkpoint_mixed_attn,
        use_fasterwam_dot=args.use_fasterwam_dot,
        use_fastwam_attention_heatmap_loss=args.use_fastwam_attention_heatmap_loss,
        use_fastwam_tracker_heatmap_loss=args.use_fastwam_tracker_heatmap_loss,
        use_fastwam_attention_bias=args.use_fastwam_attention_bias,
        use_gt_center_attention_bias=args.use_gt_center_attention_bias,
        gt_center_attention_sigma=args.gt_center_attention_sigma,
        gt_center_attention_beta=args.gt_center_attention_beta,
        gt_center_attention_zero_mean=args.gt_center_attention_zero_mean,
        gt_center_guided_layers=args.gt_center_guided_layers,
        gt_center_guided_head_ratio=args.gt_center_guided_head_ratio,
        fastwam_heatmap_source=args.fastwam_heatmap_source,
        fastwam_attention_heatmap_loss_weight=args.fastwam_attention_heatmap_loss_weight,
        fastwam_attention_heatmap_sigma=args.fastwam_attention_heatmap_sigma,
        fastwam_attention_heatmap_fov_deg=args.fastwam_attention_heatmap_fov_deg,
        use_fastwam_heatmap_guidance=args.use_fastwam_heatmap_guidance,
        fastwam_heatmap_guidance_scale=args.fastwam_heatmap_guidance_scale,
        fastwam_heatmap_guidance_sigma=args.fastwam_heatmap_guidance_sigma,
        fastwam_heatmap_guidance_fov_deg=args.fastwam_heatmap_guidance_fov_deg,
        fastwam_ortrack_consistency_loss_weight=args.fastwam_ortrack_consistency_loss_weight,
    )
    action_video_freq_ratio = max(int(args.action_video_freq_ratio), 1)
    if args.training_stage == "main" and not cfg.use_current_box_action_conditioning:
        raise ValueError(
            "--training-stage main requires current-box Action conditioning."
        )
    if args.training_stage == "main" and cfg.localization_warmup_steps != 0:
        raise ValueError(
            "Current-state conditioned main stage requires --localization-warmup-steps 0."
        )
    if (args.seq_len - 1) % action_video_freq_ratio != 0:
        raise ValueError(
            "--seq-len must satisfy (seq_len - 1) % action_video_freq_ratio == 0; "
            f"got seq_len={args.seq_len}, action_video_freq_ratio={action_video_freq_ratio}."
        )
    sampled_video_len = (args.seq_len - 1) // action_video_freq_ratio + 1
    if sampled_video_len % 4 != 1:
        raise ValueError(
            "Sampled video frame count must satisfy T % 4 == 1 for Wan VAE; "
            f"got sampled_video_len={sampled_video_len}."
        )

    scene_list = [s.strip() for s in args.scene_list.split(",") if s.strip()]
    if not scene_list:
        raise ValueError("--scene-list is empty.")
    record_cache_root = Path(args.wan_latent_cache_root) if args.wan_latent_cache_root else save_dir.parent
    train_records = _build_records_once(
        Path(args.dataset_root),
        scene_list,
        args.trajectory_range.strip(),
        max_vel=args.max_vel,
        max_yaw_rate=args.max_yaw_rate,
        max_speed_norm=args.max_speed_norm,
        cache_root=record_cache_root,
        distributed=use_distributed,
    )
    if not train_records:
        raise RuntimeError("No trajectory selected. Check --scene-list / --trajectory-range.")
    explicit_val = bool(args.val_scene_list.strip() or args.val_trajectory_range.strip())
    if explicit_val and not (args.val_scene_list.strip() and args.val_trajectory_range.strip()):
        raise ValueError("--val-scene-list and --val-trajectory-range must be provided together.")
    if explicit_val:
        val_scenes = [s.strip() for s in args.val_scene_list.split(",") if s.strip()]
        val_records = _build_records_once(
            Path(args.dataset_root),
            val_scenes,
            args.val_trajectory_range.strip(),
            max_vel=args.max_vel,
            max_yaw_rate=args.max_yaw_rate,
            max_speed_norm=args.max_speed_norm,
            cache_root=record_cache_root,
            distributed=use_distributed,
        )
        if not val_records:
            raise RuntimeError(
                "No validation trajectory selected. Check --val-scene-list / --val-trajectory-range."
            )
        train_keys = {(record["scene_id"], record["trajectory_name"]) for record in train_records}
        val_keys = {(record["scene_id"], record["trajectory_name"]) for record in val_records}
        overlap = sorted(train_keys.intersection(val_keys))
        if overlap:
            raise ValueError(f"Training and validation trajectories overlap: {overlap[:8]}")
    else:
        rng = random.Random(args.split_seed)
        rng.shuffle(train_records)
        val_n = int(len(train_records) * args.val_ratio)
        if args.val_ratio > 0.0 and len(train_records) > 1:
            val_n = max(1, val_n)
        val_n = min(val_n, max(len(train_records) - 1, 0))
        val_records = train_records[:val_n]
        train_records = train_records[val_n:] if val_n > 0 else train_records
    random.Random(args.split_seed).shuffle(train_records)
    if _is_main_process():
        if val_records:
            print(
                f"[dataset] train={len(train_records)} "
                f"({args.scene_list} {args.trajectory_range}), val={len(val_records)} "
                f"({args.val_scene_list or 'ratio split'} {args.val_trajectory_range or args.val_ratio})"
            )
        else:
            print(f"[dataset] train={len(train_records)} ({args.scene_list} {args.trajectory_range})")
        cache_stats = _wan_latent_cache_stats(train_records, args.wan_latent_cache_root, args.seq_len, action_video_freq_ratio)
        if cache_stats is not None:
            hits = cache_stats["hits"]
            windows = cache_stats["windows"]
            ratio = (hits / windows) if windows else 0.0
            print(
                f"[wan-latents] cache_root={args.wan_latent_cache_root} "
                f"seq_len={args.seq_len} video_ratio={action_video_freq_ratio} "
                f"sampled_records={cache_stats['sampled_records']}/{cache_stats['records']} "
                f"hits={hits}/{windows} ({ratio:.1%})"
            )
            if windows > 0 and hits == 0:
                print("[wan-latents] WARNING: no matching cached latents; training will encode RGB videos online.")
        print(
            "[cfg curriculum] "
            f"diffusion={cfg.use_diffusion_actor} | "
            f"architecture=fastwam_video_action_mot_no_rssm fusion={cfg.target_token_fusion_mode} | "
            f"fastwam_mot={cfg.use_fastwam_mot} | "
            f"action_video_freq_ratio={cfg.fastwam_action_video_freq_ratio} | "
            f"kl={cfg.train_kl} direct_action={cfg.train_direct_action} | "
            f"action_w={cfg.direct_action_loss_weight} yaw_w={cfg.action_yaw_loss_weight} | "
            f"WAM auxiliary: next_target_relative={cfg.train_next_target_relative} rollout_head=false | "
            f"target_relative_context={cfg.use_target_relative_context} "
            f"target_relative_token_scale={cfg.target_relative_token_scale} | "
            f"tracker_center_context={cfg.use_tracker_center_context} "
            f"tracker_heatmap_target={cfg.tracker_heatmap_target_mode} | "
            f"tracker_mot_integration={cfg.tracker_mot_integration} "
            f"tracker_finetune_init={cfg.tracker_finetune_init}"
        )
    local_feature_mode = cfg.tracker_mot_integration in {
        "frozen_deit_tracker_local_feature", "mot_tracker_finetune_local_feature"
    }
    tracker_finetune_mode = cfg.tracker_mot_integration == "mot_tracker_finetune_local_feature"
    train_dataset = TrajectoryDataset(
        records=train_records,
        image_size=args.image_size,
        seq_len=args.seq_len,
        target_relative_dim=args.target_relative_dim,
        action_dim=args.action_dim,
        direction_bins=cfg.direction_bins,
        distance_bins=cfg.distance_bins,
        text_context_length=cfg.text_context_length,
        random_crop=True,
        wan_latent_cache_root=args.wan_latent_cache_root if args.wan_latent_cache_root else None,
        action_video_freq_ratio=cfg.fastwam_action_video_freq_ratio,
        ortrack_cache_root=args.ortrack_cache_root,
        require_ortrack_cache=((
            cfg.tracker_mot_integration != "none"
            and (cfg.tracker_condition_mode != "none" or local_feature_mode)
            and not cfg.tracker_model_driven_search
        ) or (
            cfg.fastwam_heatmap_source == "tracker" and (
                cfg.use_fastwam_tracker_heatmap_loss
                or cfg.use_fastwam_attention_bias
                or cfg.use_tracker_center_context
            )
        )),
        require_tracker_features=(
            (local_feature_mode and not tracker_finetune_mode)
            or (
                cfg.tracker_mot_integration != "none"
                and "features" in cfg.tracker_condition_mode
            )
        ),
        require_tracker_bbox=(
            (local_feature_mode and not tracker_finetune_mode)
            or (
                cfg.tracker_mot_integration != "none"
                and cfg.tracker_condition_mode
                in {"bbox", "bbox_response", "bbox_response_features"}
            )
        ),
        require_tracker_response=(
            cfg.tracker_mot_integration != "none"
            and "response" in cfg.tracker_condition_mode
        ),
        require_tracker_geometry=local_feature_mode,
        require_tracker_finetune_inputs=tracker_finetune_mode,
        use_model_driven_tracker_crops=cfg.tracker_model_driven_search,
        require_target_boxes=(
            (cfg.tracker_future_target_alignment and cfg.tracker_center_flow_supervision)
            or cfg.use_future_state_dit
            or (
                cfg.use_historical_target_memory
                and cfg.target_history_future_center_loss_weight > 0.0
            )
        ),
        tracker_search_crop_jitter=cfg.tracker_search_crop_jitter,
        tracker_search_center_jitter_std=cfg.tracker_search_center_jitter_std,
        tracker_search_center_jitter_max=cfg.tracker_search_center_jitter_max,
        tracker_search_scale_jitter=cfg.tracker_search_scale_jitter,
        tracker_feature_grid_size=cfg.tracker_feature_grid_size,
        tracker_feature_dim=cfg.tracker_feature_dim,
        tracker_response_grid_size=cfg.tracker_response_grid_size,
        canonical_heatmap_sigma=cfg.fastwam_attention_heatmap_sigma,
        tracker_heatmap_target_mode=cfg.tracker_heatmap_target_mode,
        guidance_heatmap_source=cfg.fastwam_heatmap_source,
        localization_only=(args.training_stage == "s0"),
        require_target_box_history=cfg.use_historical_target_memory,
        target_history_tracker_cache_root=cfg.target_history_tracker_cache_root,
        target_history_length=cfg.target_history_length,
        target_history_partial_probability=(
            args.target_history_partial_probability
            if cfg.use_historical_target_memory
            else 0.0
        ),
        target_history_center_jitter_std=(
            args.target_history_center_jitter_std
            if cfg.use_historical_target_memory
            else 0.0
        ),
        target_history_log_size_jitter_std=(
            args.target_history_log_size_jitter_std
            if cfg.use_historical_target_memory
            else 0.0
        ),
        target_history_confidence_dropout_probability=(
            args.target_history_confidence_dropout_probability
            if cfg.use_historical_target_memory
            else 0.0
        ),
    )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=_get_world_size(),
            rank=_get_rank(),
            shuffle=True,
            drop_last=False,
        )
        if use_distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    # For simplicity: run validation only on rank0 under DDP.
    val_loader = None
    if val_records and ((not use_distributed) or _is_main_process()):
        val_dataset = TrajectoryDataset(
            records=val_records,
            image_size=args.image_size,
            seq_len=args.seq_len,
            target_relative_dim=args.target_relative_dim,
            action_dim=args.action_dim,
            direction_bins=cfg.direction_bins,
            distance_bins=cfg.distance_bins,
            text_context_length=cfg.text_context_length,
            random_crop=False,
            wan_latent_cache_root=args.wan_latent_cache_root if args.wan_latent_cache_root else None,
            action_video_freq_ratio=cfg.fastwam_action_video_freq_ratio,
            ortrack_cache_root=args.ortrack_cache_root,
            require_ortrack_cache=((
                cfg.tracker_mot_integration != "none"
                and (cfg.tracker_condition_mode != "none" or local_feature_mode)
                and not cfg.tracker_model_driven_search
            ) or (
                cfg.fastwam_heatmap_source == "tracker" and (
                    cfg.use_fastwam_tracker_heatmap_loss
                    or cfg.use_fastwam_attention_bias
                    or cfg.use_tracker_center_context
                )
            )),
            require_tracker_features=(
                (local_feature_mode and not tracker_finetune_mode)
                or (
                    cfg.tracker_mot_integration != "none"
                    and "features" in cfg.tracker_condition_mode
                )
            ),
            require_tracker_bbox=(
                (local_feature_mode and not tracker_finetune_mode)
                or (
                    cfg.tracker_mot_integration != "none"
                    and cfg.tracker_condition_mode
                    in {"bbox", "bbox_response", "bbox_response_features"}
                )
            ),
            require_tracker_response=(
                cfg.tracker_mot_integration != "none"
                and "response" in cfg.tracker_condition_mode
            ),
            require_tracker_geometry=local_feature_mode,
            require_tracker_finetune_inputs=tracker_finetune_mode,
            use_model_driven_tracker_crops=cfg.tracker_model_driven_search,
            require_target_boxes=(
                (cfg.tracker_future_target_alignment and cfg.tracker_center_flow_supervision)
                or cfg.use_future_state_dit
                or (
                    cfg.use_historical_target_memory
                    and cfg.target_history_future_center_loss_weight > 0.0
                )
            ),
            tracker_search_crop_jitter=False,
            tracker_search_center_jitter_std=cfg.tracker_search_center_jitter_std,
            tracker_search_center_jitter_max=cfg.tracker_search_center_jitter_max,
            tracker_search_scale_jitter=cfg.tracker_search_scale_jitter,
            tracker_feature_grid_size=cfg.tracker_feature_grid_size,
            tracker_feature_dim=cfg.tracker_feature_dim,
            tracker_response_grid_size=cfg.tracker_response_grid_size,
            canonical_heatmap_sigma=cfg.fastwam_attention_heatmap_sigma,
            tracker_heatmap_target_mode=cfg.tracker_heatmap_target_mode,
            guidance_heatmap_source=cfg.fastwam_heatmap_source,
            localization_only=(args.training_stage == "s0"),
            require_target_box_history=cfg.use_historical_target_memory,
            target_history_tracker_cache_root=cfg.target_history_tracker_cache_root,
            target_history_length=cfg.target_history_length,
            target_history_partial_probability=0.0,
            target_history_center_jitter_std=0.0,
            target_history_log_size_jitter_std=0.0,
            target_history_confidence_dropout_probability=0.0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    model: torch.nn.Module
    if args.training_stage == "s0":
        model = S0LocalizationModel(cfg).to(device)
    else:
        model = TeacherWorldModelDiT(cfg).to(device)
    inherited_init_parameter_names: set[str] = set()
    if cfg.use_historical_target_memory and cfg.target_conditioning_adapter_only:
        if not args.init_checkpoint:
            raise ValueError(
                "Adapter-only Historical Target Memory training requires its "
                "Current Box parent checkpoint via --init-checkpoint."
            )
    if cfg.use_capture_value_reranking and cfg.capture_value_adapter_only:
        if not args.init_checkpoint:
            raise ValueError(
                "Capture-value adapter training requires its Current Box parent "
                "checkpoint via --init-checkpoint."
            )
    if args.init_checkpoint:
        if args.training_stage == "s0":
            raise ValueError("Standalone S0 training does not accept --init-checkpoint.")
        if cfg.use_future_state_dit:
            raise ValueError("V4 Future State DiT must start without --init-checkpoint.")
        init_path = Path(args.init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError(f"--init-checkpoint does not exist: {init_path}")
        init_ckpt = torch.load(init_path, map_location="cpu")
        if cfg.use_historical_target_memory:
            parent_cfg = init_ckpt.get("cfg", {}) if isinstance(init_ckpt, dict) else {}
            valid_parent = isinstance(parent_cfg, dict) and bool(
                parent_cfg.get("use_current_box_action_conditioning", False)
            )
            valid_parent = valid_parent and not bool(
                parent_cfg.get("use_historical_target_memory", False)
            )
            valid_parent = valid_parent and (
                bool(parent_cfg.get("use_fasterwam_dot", False))
                == bool(cfg.use_fasterwam_dot)
            )
            if not valid_parent:
                raise ValueError(
                    "Historical Target Memory must initialize from the matching "
                    "FastWAM/FasterWAM Current Box baseline."
                )
        if cfg.use_capture_value_reranking and cfg.capture_value_adapter_only:
            parent_cfg = init_ckpt.get("cfg", {}) if isinstance(init_ckpt, dict) else {}
            valid_parent = (
                isinstance(parent_cfg, dict)
                and bool(parent_cfg.get("use_fasterwam_dot", False))
                and bool(parent_cfg.get("use_current_box_action_conditioning", False))
                and not bool(parent_cfg.get("use_historical_target_memory", False))
                and not bool(parent_cfg.get("use_capture_value_reranking", False))
            )
            if not valid_parent:
                raise ValueError(
                    "Capture-value reranking must initialize from the FasterWAM "
                    "Current Box parent checkpoint."
                )
        init_state = init_ckpt.get("model", init_ckpt) if isinstance(init_ckpt, dict) else init_ckpt
        init_state = migrate_legacy_state_dict_keys(init_state)
        skipped_tracker_gates = []
        if cfg.tracker_mot_integration == "frozen_deit_tracker_local_feature":
            skipped_tracker_gates = [
                key
                for key in init_state
                if "fastwam.tracker_fusion.layers." in key and key.endswith(".gate")
            ]
            init_state = {
                key: value
                for key, value in init_state.items()
                if key not in skipped_tracker_gates
            }
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        inherited_init_parameter_names = {
            name for name, _ in model.named_parameters() if name in init_state
        }
        if _is_main_process():
            print(
                f"[init-checkpoint] {init_path} strict=False "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"zero_init_tracker_gates={len(skipped_tracker_gates)}"
            )
    if args.training_stage == "main":
        s0_path = Path(args.s0_localizer_checkpoint)
        if not s0_path.is_file():
            raise FileNotFoundError(
                "Current-state conditioned main stage requires an existing "
                f"--s0-localizer-checkpoint: {s0_path}"
            )
        s0_payload = torch.load(s0_path, map_location="cpu", weights_only=False)
        s0_state = normalize_s0_checkpoint_state(s0_payload)
        expected_s0 = {
            name
            for name, _ in model.named_parameters()
            if name.startswith(S0_PARAMETER_PREFIXES)
        }
        missing_s0 = sorted(expected_s0.difference(s0_state))
        if missing_s0:
            raise ValueError(
                f"S0 checkpoint is incomplete; missing parameters: {missing_s0[:8]}"
            )
        _, unexpected_s0 = model.load_state_dict(s0_state, strict=False)
        if unexpected_s0:
            raise ValueError(f"Unexpected S0 checkpoint parameters: {list(unexpected_s0)[:8]}")
        for name, parameter in model.named_parameters():
            if name.startswith(S0_PARAMETER_PREFIXES):
                parameter.requires_grad_(False)
        if _is_main_process():
            print(
                f"[s0-load] loaded and froze {len(expected_s0)} parameters from {s0_path}"
            )
    if cfg.use_historical_target_memory and cfg.target_conditioning_adapter_only:
        adapter_tensors, adapter_parameters = _freeze_for_target_conditioning_adapter_training(
            model, inherited_init_parameter_names, cfg
        )
        if _is_main_process():
            print(
                "[target-conditioning-adapter] froze inherited parent policy; "
                f"trainable_tensors={adapter_tensors} "
                f"trainable_parameters={adapter_parameters:,}"
            )
    if cfg.use_capture_value_reranking and cfg.capture_value_adapter_only:
        value_tensors, value_parameters = _freeze_for_capture_value_training(
            model, inherited_init_parameter_names
        )
        if _is_main_process():
            print(
                "[capture-value-adapter] froze inherited parent policy; "
                f"trainable_tensors={value_tensors} "
                f"trainable_parameters={value_parameters:,}"
            )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found; check training mode and freeze settings.")
    if _is_main_process():
        trainable_count = sum(int(p.numel()) for p in trainable_params)
        print(f"[train] trainable parameters: {trainable_count:,}")
    optimizer_parameters: Any = trainable_params
    if args.training_stage == "s0" and bool(cfg.tracker_include_box_token):
        optimizer_parameters = model.tracker.parameter_groups(
            args.lr, backbone_multiplier=0.1
        )
        if _is_main_process():
            print(
                f"[s0-optimizer] Tracker head/position lr={args.lr:g}, "
                f"backbone lr={args.lr * 0.1:g}"
            )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if use_deepspeed:
        ds_config = args.deepspeed_config if args.deepspeed_config is not None else _make_deepspeed_config(args)
        model, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=trainable_params,
            optimizer=optimizer,
            config=ds_config,
        )
        if _is_main_process():
            print(f"[train] DeepSpeed enabled on world_size={_get_world_size()} (local_rank={_get_local_rank()})")
    elif use_ddp:
        model = DDP(
            model,
            device_ids=[_get_local_rank()],
            output_device=_get_local_rank(),
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        if _is_main_process():
            print(f"[train] DDP enabled on world_size={_get_world_size()} (local_rank={_get_local_rank()})")
    else:
        use_dp = args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1
        if use_dp:
            model = torch.nn.DataParallel(model)
            print(f"[train] DataParallel enabled on {torch.cuda.device_count()} GPUs")
        else:
            print(f"[train] Device: {device}")
    scaler = torch.amp.GradScaler("cuda", enabled=_grad_scaler_enabled(device, cfg, use_deepspeed))
    if _is_main_process() and device.type == "cuda":
        print(f"[train] AMP dtype: {_cuda_amp_dtype(cfg)}, grad_scaler={scaler.is_enabled()}")

    start_epoch = 0
    global_step = 0
    best_val = math.inf
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        resume_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        resume_args = ckpt.get("run_args", {}) if isinstance(ckpt, dict) else {}
        if str(resume_args.get("training_stage", "joint")) != args.training_stage:
            raise ValueError(
                "Resume checkpoint training stage does not match this run: "
                f"checkpoint={resume_args.get('training_stage', 'joint')} requested={args.training_stage}."
            )
        if cfg.use_future_state_dit and (
            not bool(resume_cfg.get("use_future_state_dit", False))
            or int(resume_cfg.get("tracker_state_action_alignment_version", -1)) != 4
        ):
            raise ValueError("Incompatible legacy checkpoint: V4 StateDiT resume requires version 4.")
        missing, unexpected = _unwrap_model(model).load_state_dict(
            migrate_legacy_state_dict_keys(ckpt["model"]),
            strict=False,
        )
        if cfg.use_future_state_dit:
            trainable = {
                name for name, parameter in _unwrap_model(model).named_parameters()
                if parameter.requires_grad
            }
            missing_trainable = sorted(name for name in missing if name in trainable)
            if missing_trainable or unexpected:
                raise ValueError(
                    "Incompatible V4 checkpoint parameters: "
                    f"missing_trainable={missing_trainable}, unexpected={list(unexpected)}"
                )
        if _is_main_process() and (missing or unexpected):
            print(f"[resume] load strict=False: missing={len(missing)} unexpected={len(unexpected)}")
        if not use_deepspeed:
            if ckpt.get("optimizer") and ckpt.get("scheduler"):
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
            elif _is_main_process():
                print("[resume] optimizer/scheduler state missing; restarting optimizer state.")
        start_epoch = ckpt["epoch"] + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val = ckpt.get("best_val", best_val)

    total_pbar = None
    run_name = args.swanlab_experiment_name or save_dir.name
    swanlab_run = _init_swanlab(args, cfg, run_name)
    if _is_main_process():
        print(
            "[running-model] "
            f"model={save_dir.name} | run={run_name} | save_dir={save_dir} | "
            f"training_stage={args.training_stage} | "
            f"target_relative_context={cfg.use_target_relative_context} | "
            f"fastwam_mot={cfg.use_fastwam_mot} | "
            f"fasterwam_dot={cfg.use_fasterwam_dot} | "
            f"tracker_condition={cfg.tracker_condition_mode} | "
            f"tracker_gate_init={cfg.tracker_fusion_gate_init}"
        )
        if cfg.use_future_state_dit:
            print(
                "[localization-curriculum] "
                f"warmup_steps={cfg.localization_warmup_steps} | "
                f"box_l1_w={cfg.current_box_weight:g} | "
                f"center_w={cfg.current_center_weight:g} | "
                f"box_giou_w={cfg.current_box_giou_weight:g} | "
                f"attention_w={cfg.current_attention_weight:g} | "
                f"attention_sigma={cfg.current_attention_sigma:g}"
            )
    if tqdm is not None and _is_main_process():
        if int(args.max_train_steps) > 0:
            total_steps = max(int(args.max_train_steps) - int(global_step), 0)
            desc = f"train steps {global_step}->{int(args.max_train_steps)}"
        else:
            total_steps = max(args.epochs - start_epoch, 0) * max(len(train_loader), 1)
            desc = f"train {start_epoch:03d}->{args.epochs - 1:03d}"
        total_pbar = tqdm(
            total=total_steps,
            desc=desc,
            leave=True,
            dynamic_ncols=True,
        )

    try:
        reached_max_steps = False
        for epoch in range(start_epoch, args.epochs):
            if int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps):
                break
            model.train()
            if args.training_stage == "main":
                unwrapped = _unwrap_model(model)
                unwrapped.tracker.eval()
                if unwrapped.fastwam.tracker_fusion is not None:
                    unwrapped.fastwam.tracker_fusion.eval()
                if unwrapped.fastwam.current_target_localizer is not None:
                    unwrapped.fastwam.current_target_localizer.eval()
            if use_deepspeed:
                epoch_lr = (
                    _cosine_step_lr(args.lr, global_step, int(args.max_train_steps))
                    if int(args.max_train_steps) > 0
                    else _cosine_epoch_lr(args.lr, epoch, args.epochs)
                )
                _set_optimizer_lr(optimizer, epoch_lr)
            else:
                epoch_lr = float(optimizer.param_groups[0]["lr"])
            running: Dict[str, float] = {}
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            num_train_batches = 0
            for step, batch in enumerate(train_loader):
                if int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps):
                    reached_max_steps = True
                    break
                if int(args.max_train_steps) > 0:
                    step_lr = _cosine_step_lr(args.lr, global_step, int(args.max_train_steps))
                    _set_optimizer_lr(optimizer, step_lr)
                    epoch_lr = step_lr
                batch = move_batch_to_device(batch, device)
                if not use_deepspeed:
                    optimizer.zero_grad(set_to_none=True)
                amp_ctx = nullcontext() if use_deepspeed else _autocast_context(device, cfg)
                with amp_ctx:
                    localization_only = bool(
                        args.training_stage == "joint"
                        and cfg.use_future_state_dit
                        and global_step < int(cfg.localization_warmup_steps)
                    )
                    _, losses = _forward_and_loss(
                        model,
                        batch,
                        cfg,
                        args.training_stage,
                        localization_only=localization_only,
                    )
                    loss = losses["total"]

                if use_deepspeed:
                    optimizer_step = bool(model.is_gradient_accumulation_boundary())
                    model.backward(loss)
                    model.step()
                else:
                    optimizer_step = True
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                summary = summarize_losses(losses)
                for k, v in summary.items():
                    running[k] = running.get(k, 0.0) + v

                num_train_batches += 1
                avg = {k: v / (step + 1) for k, v in running.items()}
                if optimizer_step:
                    global_step += 1
                if total_pbar is not None:
                    if optimizer_step:
                        postfix = {"ep": f"{epoch:02d}", **_tqdm_train_postfix(avg)}
                        total_pbar.set_postfix(postfix, refresh=False)
                        total_pbar.update(1)
                elif _is_main_process() and (step + 1) % 20 == 0:
                    print(f"[Epoch {epoch:03d} | Step {step + 1:05d}] {_format_metrics(avg)}")

            if not use_deepspeed:
                if int(args.max_train_steps) <= 0:
                    scheduler.step()
            train_avg = {k: v / max(num_train_batches, 1) for k, v in running.items()}
            train_avg = _reduce_metrics(train_avg, device, use_distributed)
            if _is_main_process():
                msg = f">>> Epoch {epoch:03d} train: {_format_metrics(train_avg)} | lr={epoch_lr:.6g} | global_step={global_step}"
                tqdm.write(msg) if tqdm is not None else print(msg)
                gate_summary = _tracker_gate_summary(model)
                if gate_summary is not None:
                    gate_msg = f">>> Epoch {epoch:03d} tracker gates: {gate_summary}"
                    tqdm.write(gate_msg) if tqdm is not None else print(gate_msg)
                _swanlab_log(swanlab_run, {**train_avg, "lr": float(epoch_lr), "global_step": int(global_step)}, step=global_step, prefix="train")

            val_avg = None
            validate_this_epoch = bool(
                val_loader is not None
                and (
                    (epoch + 1) % max(int(args.val_every_epochs), 1) == 0
                    or epoch + 1 == args.epochs
                    or (
                        int(args.max_train_steps) > 0
                        and global_step >= int(args.max_train_steps)
                    )
                )
            )
            if validate_this_epoch:
                val_avg = evaluate(
                    _unwrap_model(model), val_loader, cfg, device, args.training_stage
                )
                if _is_main_process():
                    msg = f">>> Epoch {epoch:03d} val:   {_format_metrics(val_avg)}"
                    tqdm.write(msg) if tqdm is not None else print(msg)
                    _swanlab_log(swanlab_run, val_avg, step=epoch, prefix="val")

            metric = (
                val_avg["total"]
                if val_avg is not None
                else (train_avg["total"] if val_loader is None else None)
            )
            should_save = (
                _is_main_process()
                and int(args.save_every_epochs) > 0
                and (
                    (epoch + 1) % int(args.save_every_epochs) == 0
                    or epoch + 1 == args.epochs
                    or (
                        int(args.max_train_steps) > 0
                        and global_step >= int(args.max_train_steps)
                    )
                )
            )
            if should_save:
                is_best = metric is not None and metric < best_val
                if is_best:
                    best_val = metric
                ckpt = {
                    "epoch": epoch,
                    "global_step": int(global_step),
                    "max_train_steps": int(args.max_train_steps),
                    "model": _trainable_state_dict(model),
                    "model_state_format": "trainable_only",
                    "optimizer": {} if (use_deepspeed or not args.save_optimizer_state) else optimizer.state_dict(),
                    "scheduler": {} if (use_deepspeed or not args.save_optimizer_state) else scheduler.state_dict(),
                    "cfg": cfg.__dict__,
                    "run_args": vars(args),
                    "best_val": best_val,
                }
                torch.save(ckpt, save_dir / "last.pt")
                if bool(args.save_best_checkpoint) and is_best:
                    torch.save(ckpt, save_dir / "best.pt")

            if use_distributed:
                dist.barrier()
            if reached_max_steps or (int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps)):
                break
        if _is_main_process():
            done_marker = {
                "status": "complete",
                "training_stage": args.training_stage,
                "epochs": int(args.epochs),
                "global_step": int(global_step),
                "max_train_steps": int(args.max_train_steps),
                "best_val": float(best_val),
                "localization_warmup_steps": int(cfg.localization_warmup_steps),
                "s0_localizer_checkpoint": args.s0_localizer_checkpoint,
            }
            with open(save_dir / "done.marker", "w", encoding="utf-8") as f:
                json.dump(done_marker, f, indent=2, ensure_ascii=False)
    finally:
        if total_pbar is not None:
            total_pbar.close()
        _swanlab_finish(swanlab_run)
        if use_distributed and _ddp_is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
