from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import random
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torchvision import transforms

from data.action_mapping import clamp_physical_action_speed, norm_action_to_physical
from eval.ortrack_target_heatmap import (
    ORTrack640,
    _bbox_from_target_center,
    _bbox_heatmap,
    _confidence_heatmap,
    _real_bbox_for_frame,
)
from model.config import ModelConfig, migrate_legacy_config
from model.model import (
    S0_PARAMETER_PREFIXES,
    TeacherWorldModelDiT,
    migrate_legacy_state_dict_keys,
    normalize_s0_checkpoint_state,
)
from model.target_action_conditioning import make_online_target_box_history
from tracking.data import crop_target
from tracking.evaluate import crop_search

PROJECT_ROOT = Path(__file__).resolve().parents[3]

try:
    from data.instruction_generator import EPISODE_INSTRUCTION
except Exception:  # pragma: no cover
    EPISODE_INSTRUCTION = (
        "The target is the black UAV initially located near the image center. "
        "Keep tracking the same UAV throughout the episode."
    )

try:
    from transformers import CLIPTokenizerFast
except Exception:  # pragma: no cover
    CLIPTokenizerFast = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from train.train_teacher import LOCAL_CLIP_MODEL_PATH, LOCAL_DINOV2_MODEL_PATH, seed_everything
except Exception:  # pragma: no cover
    LOCAL_CLIP_MODEL_PATH = str(PROJECT_ROOT / "model/clip-vit-base-patch32")
    LOCAL_DINOV2_MODEL_PATH = str(PROJECT_ROOT / "model/dinov2-base")

    def seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

DEFAULT_MODEL_CFG = ModelConfig()


# -----------------------------
# Generic helpers
# -----------------------------


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _trajectory_key(scene_id: str, trajectory_name: str) -> str:
    return f"{scene_id}/{trajectory_name}"


def _trajectory_index_token(trajectory_name: str) -> Optional[str]:
    match = re.search(r"(\d+)$", str(trajectory_name))
    if match is None:
        return None
    return str(int(match.group(1)))


def _trajectory_visualization_tokens(raw: str) -> List[str]:
    tokens = []
    for item in re.split(r"[,\s]+", str(raw or "")):
        token = item.strip().replace("\\", "/").replace(":", "/").strip("/")
        if token:
            tokens.append(token.lower())
    return tokens


def _visualization_enabled_for_key(args: argparse.Namespace, scene_id: str, trajectory_name: str) -> bool:
    raw = str(getattr(args, "visualize_trajectory_keys", "") or "").strip()
    if not raw:
        return True
    tokens = _trajectory_visualization_tokens(raw)
    if any(token in {"all", "*"} for token in tokens):
        return True
    if not tokens or all(token in {"none", "false", "off", "0"} for token in tokens):
        return False
    scene = str(scene_id).lower()
    traj = str(trajectory_name).lower()
    candidates = {traj, f"{scene}/{traj}"}
    idx = _trajectory_index_token(trajectory_name)
    if idx is not None:
        candidates.update({idx, f"{scene}/{idx}", f"{scene}/trajectory_{int(idx):04d}"})
    return any(token in candidates for token in tokens)


def _visualization_enabled_for_trajectory(args: argparse.Namespace, traj: "OnlineTrajectory") -> bool:
    return _visualization_enabled_for_key(args, traj.scene_id, traj.trajectory_name)


def _attention_enabled_for_key(args: argparse.Namespace, scene_id: str, trajectory_name: str) -> bool:
    raw = str(getattr(args, "attention_trajectory_keys", "") or "").strip()
    if not raw:
        return True
    proxy_args = argparse.Namespace(visualize_trajectory_keys=raw)
    return _visualization_enabled_for_key(proxy_args, scene_id, trajectory_name)


def _predicted_video_enabled_for_key(
    args: argparse.Namespace,
    scene_id: str,
    trajectory_name: str,
) -> bool:
    raw = str(getattr(args, "predicted_video_trajectory_keys", "") or "").strip()
    if not raw:
        return True
    proxy_args = argparse.Namespace(visualize_trajectory_keys=raw)
    return _visualization_enabled_for_key(proxy_args, scene_id, trajectory_name)


def _tracker_runtime_required(cfg: ModelConfig, args: argparse.Namespace) -> bool:
    if bool(getattr(cfg, "tracker_model_driven_search", False)):
        # The future-token cropper provides Template/Search inputs itself.
        # Do not even construct the external checkpoint-backed Tracker.
        return False
    tracker_model = str(getattr(cfg, "fastwam_heatmap_source", "none")) == "tracker" and (
        bool(getattr(cfg, "use_fastwam_attention_bias", False))
        or bool(getattr(cfg, "use_fastwam_tracker_heatmap_loss", False))
        or bool(getattr(cfg, "use_tracker_center_context", False))
    )
    integration = str(getattr(cfg, "tracker_mot_integration", "none"))
    tracker_condition = integration != "none" and (
        integration in {"frozen_deit_tracker_local_feature", "mot_tracker_finetune_local_feature"}
        or str(getattr(cfg, "tracker_condition_mode", "center_features")) != "none"
    )
    return bool(
        tracker_model
        or tracker_condition
        or getattr(args, "reuse_last_confident_action_sequence", False)
    )


def _tracker_center_required(cfg: ModelConfig) -> bool:
    condition_mode = str(getattr(cfg, "tracker_condition_mode", "center_features"))
    return bool(
        getattr(cfg, "use_tracker_center_context", False)
        or getattr(cfg, "use_gt_center_attention_bias", False)
        or (
            str(getattr(cfg, "tracker_mot_integration", "none")).strip().lower()
            == "frozen_deit_tracker_fusion"
            and condition_mode in {"center", "center_features"}
        )
    )


def _tracker_bbox_required(cfg: ModelConfig) -> bool:
    integration = str(getattr(cfg, "tracker_mot_integration", "none"))
    return bool(
        integration in {"frozen_deit_tracker_local_feature", "mot_tracker_finetune_local_feature"}
        or (
            integration == "frozen_deit_tracker_fusion"
            and str(getattr(cfg, "tracker_condition_mode", "center_features"))
            in {"bbox", "bbox_response", "bbox_response_features"}
        )
    )


def _tracker_response_required(cfg: ModelConfig) -> bool:
    return bool(
        str(getattr(cfg, "tracker_mot_integration", "none")) == "frozen_deit_tracker_fusion"
        and "response" in str(getattr(cfg, "tracker_condition_mode", "center_features"))
    )


def _tracker_features_required(cfg: ModelConfig) -> bool:
    integration = str(getattr(cfg, "tracker_mot_integration", "none"))
    return bool(
        integration in {"frozen_deit_tracker_local_feature", "mot_tracker_finetune_local_feature"}
        or (
            integration == "frozen_deit_tracker_fusion"
            and "features"
            in str(getattr(cfg, "tracker_condition_mode", "center_features"))
        )
    )


def _tracker_geometry_required(cfg: ModelConfig) -> bool:
    return bool(
        str(getattr(cfg, "tracker_mot_integration", "none"))
        in {"frozen_deit_tracker_local_feature", "mot_tracker_finetune_local_feature"}
    )


def _model_driven_initial_bbox(
    target_relative_body: np.ndarray,
    image_hw: Tuple[int, int],
    fov_deg: float,
    camera_offset_body: Tuple[float, float, float],
    box_frac: float,
) -> List[float]:
    """Initialize Template/Search from the target in the current online image."""
    return _bbox_from_target_center(
        target_relative_body,
        image_hw,
        fov_deg,
        camera_offset_body,
        box_frac,
    )


@dataclass
class ModelDrivenTrackerCropper:
    """Online crop state centered on the previous frame's estimated target."""

    template_size: int
    search_size: int
    template_factor: float = 2.0
    search_factor: float = 4.0
    template: Optional[torch.Tensor] = None
    state_bbox: Optional[List[float]] = None
    target_side: float = 0.0

    @staticmethod
    def _image_tensor(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)

    def initialize(self, image: np.ndarray, init_bbox: Sequence[float]) -> None:
        tensor = self._image_tensor(image)
        x, y, width, height = (float(value) for value in init_bbox)
        self.target_side = max(width, height, 2.0)
        self.state_bbox = [
            x + 0.5 * width - 0.5 * self.target_side,
            y + 0.5 * height - 0.5 * self.target_side,
            self.target_side,
            self.target_side,
        ]
        self.template, _ = crop_target(
            tensor, init_bbox, self.template_factor, self.template_size
        )

    def current(self, image: np.ndarray, device: torch.device) -> Dict[str, Any]:
        if self.template is None or self.state_bbox is None:
            raise RuntimeError("ModelDrivenTrackerCropper must be initialized before use.")
        tensor = self._image_tensor(image)
        search, geometry = crop_search(
            tensor, self.state_bbox, self.search_factor, self.search_size
        )
        height, width = image.shape[:2]
        return {
            "bbox": list(self.state_bbox),
            "confidence": 1.0,
            "heatmap": np.full(
                (height, width), 1.0 / max(float(height * width), 1.0), dtype=np.float32
            ),
            "search_crop_xy_size": [int(value) for value in geometry],
            "tracker_template": self.template.unsqueeze(0).to(device),
            "tracker_search": search.unsqueeze(0).to(device),
        }

    def advance(self, center_xy: Sequence[float], image_hw: Sequence[int]) -> List[float]:
        if self.state_bbox is None:
            raise RuntimeError("ModelDrivenTrackerCropper must be initialized before advance.")
        image_h, image_w = (max(float(value), 1.0) for value in image_hw)
        center_x = float(np.clip(float(center_xy[0]), 0.0, 1.0)) * image_w
        center_y = float(np.clip(float(center_xy[1]), 0.0, 1.0)) * image_h
        self.state_bbox = [
            center_x - 0.5 * self.target_side,
            center_y - 0.5 * self.target_side,
            self.target_side,
            self.target_side,
        ]
        return list(self.state_bbox)


def _current_state_center_for_next_search(state_centers: torch.Tensor) -> np.ndarray:
    """Carry the observed current state s0 forward as the next search anchor."""
    return state_centers[0, 0].detach().float().cpu().numpy().astype(np.float32)


def _normalized_tracker_response(response: np.ndarray, grid_size: int) -> np.ndarray:
    value = np.maximum(np.asarray(response, dtype=np.float32), 0.0)
    value /= max(float(value.sum()), 1.0e-8)
    value = cv2.resize(
        value,
        (max(int(grid_size), 1), max(int(grid_size), 1)),
        interpolation=cv2.INTER_AREA,
    )
    return value / max(float(value.sum()), 1.0e-8)


def _expected_visual_asset_dirs(
    args: argparse.Namespace,
    cfg: ModelConfig,
    scene_id: str,
    trajectory_name: str,
) -> List[str]:
    dirs: List[str] = []
    save_visual_assets = _visualization_enabled_for_key(args, scene_id, trajectory_name)
    if save_visual_assets:
        if bool(getattr(args, "save_rgb", False)):
            dirs.append("rgb")
        if bool(args.save_transformer_attention_maps) and _attention_enabled_for_key(
            args, scene_id, trajectory_name
        ):
            dirs.append("last_transformer_attention_maps")
    if bool(args.save_predicted_video) and _predicted_video_enabled_for_key(
        args, scene_id, trajectory_name
    ):
        dirs.append("predicted_video")
    tracker_runtime_enabled = _tracker_runtime_required(cfg, args)
    if save_visual_assets and bool(getattr(args, "save_ortrack_maps", False)) and tracker_runtime_enabled:
        dirs.append("ortrack")
    return dirs


def _predicted_video_assets_complete(out_dir: Path) -> bool:
    rollout_path = out_dir / "online_rollout.json"
    try:
        payload = _load_json(rollout_path)
    except Exception:
        return False
    steps = payload.get("steps", []) if isinstance(payload, dict) else []
    if not isinstance(steps, list) or not steps:
        return False
    for step in steps:
        paths = step.get("predicted_video_frames") if isinstance(step, dict) else None
        if not isinstance(paths, list) or not paths:
            return False
        if any(not (out_dir / str(path)).is_file() for path in paths):
            return False
    return True


def _visual_assets_exist_for_key(args: argparse.Namespace, cfg: ModelConfig, scene_id: str, trajectory_name: str) -> bool:
    expected_dirs = _expected_visual_asset_dirs(args, cfg, scene_id, trajectory_name)
    if not expected_dirs:
        return True
    out_dir = Path(args.output_dir) / scene_id / trajectory_name
    for dirname in expected_dirs:
        if dirname == "predicted_video":
            if not _predicted_video_assets_complete(out_dir):
                return False
            continue
        path = out_dir / dirname
        if not path.exists() or not any(path.rglob("*.png")):
            return False
    return True


def _load_resumable_partial(
    partial_path: Path,
    expected_keys: set[str],
    *,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    if not partial_path.exists():
        return []
    try:
        payload = _load_json(partial_path)
    except Exception as exc:
        if verbose:
            print(f"[resume-warn] ignore unreadable partial summary {partial_path}: {exc}")
        return []
    summaries = payload.get("summaries", []) if isinstance(payload, dict) else []
    if not isinstance(summaries, list):
        if verbose:
            print(f"[resume-warn] ignore malformed partial summary {partial_path}: summaries is not a list")
        return []

    kept: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        if str(summary.get("failure_reason") or "").lower() == "runtime_error":
            continue
        scene_id = str(summary.get("scene_id") or "")
        trajectory_name = str(summary.get("trajectory_name") or "")
        key = _trajectory_key(scene_id, trajectory_name)
        if key in expected_keys and key not in seen:
            kept.append(summary)
            seen.add(key)
    if verbose and kept:
        print(f"[resume] loaded {len(kept)} completed trajectories from {partial_path}")
    return kept


def _jsonable_cfg(cfg: ModelConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in dataclasses.asdict(cfg).items():
        if isinstance(value, tuple):
            out[key] = list(value)
        elif isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _xyz_from_any(obj: Any) -> Optional[np.ndarray]:
    if obj is None:
        return None
    if isinstance(obj, dict) and all(k in obj for k in ("x", "y", "z")):
        return np.asarray([float(obj["x"]), float(obj["y"]), float(obj["z"])], dtype=np.float32)
    # AirSim Vector3r-style object.
    if all(hasattr(obj, k) for k in ("x_val", "y_val", "z_val")):
        return np.asarray([float(obj.x_val), float(obj.y_val), float(obj.z_val)], dtype=np.float32)
    if isinstance(obj, np.ndarray) and obj.size >= 3:
        flat = np.asarray(obj, dtype=np.float64).reshape(-1)
        return np.asarray([float(flat[0]), float(flat[1]), float(flat[2])], dtype=np.float32)
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        return np.asarray([float(obj[0]), float(obj[1]), float(obj[2])], dtype=np.float32)
    return None


def _dataset_xyz_to_airsim(pos: Any) -> Optional[np.ndarray]:
    """Saved Dataset coordinates use z-up; AirSim uses z-down.

    Important: for the collected Dataset, y is already in the AirSim/world axis used by
    your executor. Therefore only z is flipped. This is different from the planner-side
    JSON, where the executor had to flip both y and z.
    """
    xyz = _xyz_from_any(pos)
    if xyz is None:
        return None
    return np.asarray([xyz[0], xyz[1], -xyz[2]], dtype=np.float32)


def _airsim_xyz_to_dataset(pos: Any) -> Optional[Dict[str, float]]:
    xyz = _xyz_from_any(pos)
    if xyz is None:
        return None
    return {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(-xyz[2])}


def _airsim_xyz_to_dataset_array(pos: Any) -> np.ndarray:
    xyz = _xyz_from_any(pos)
    if xyz is None:
        raise ValueError(f"Cannot parse AirSim position as xyz: {pos!r}")
    return np.asarray([xyz[0], xyz[1], -xyz[2]], dtype=np.float32)


def _natural_key(path: Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def _parse_scene_list(value: str) -> List[str]:
    out: List[str] = []
    for chunk in re.split(r"[,\s]+", value.strip()):
        if chunk:
            out.append(chunk)
    if not out:
        raise ValueError("scene list is empty")
    return out


def _parse_range_spec(spec: str) -> Dict[str, List[Tuple[int, int]]]:
    """Parse: "1-50" or "City_1:1-50,City_2:51-100"."""
    spec = (spec or "").strip()
    result: Dict[str, List[Tuple[int, int]]] = {}
    if not spec:
        return result

    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            scene, rng = item.split(":", 1)
            key = scene.strip()
        else:
            key = "*"
            rng = item
        if "-" in rng:
            a, b = rng.split("-", 1)
            lo, hi = int(a), int(b)
        else:
            lo = hi = int(rng)
        if lo > hi:
            lo, hi = hi, lo
        result.setdefault(key, []).append((lo, hi))
    return result


def _extract_trajectory_number(name: str) -> Optional[int]:
    m = re.search(r"trajectory[_-]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def _in_range(scene_id: str, traj_name: str, ranges: Dict[str, List[Tuple[int, int]]]) -> bool:
    if not ranges:
        return True
    n = _extract_trajectory_number(traj_name)
    if n is None:
        return True
    candidates = ranges.get(scene_id, []) + ranges.get("*", [])
    if not candidates:
        return True
    return any(lo <= n <= hi for lo, hi in candidates)


def _case_insensitive_child(root: Path, name: str) -> Optional[Path]:
    direct = root / name
    if direct.exists():
        return direct
    name_l = name.lower()
    try:
        for p in root.iterdir():
            if p.is_dir() and p.name.lower() == name_l:
                return p
    except Exception:
        return None
    return None


def dynamic_import_module(py_file: Path):
    py_file = py_file.resolve()
    if not py_file.exists():
        raise FileNotFoundError(f"executor script not found: {py_file}")

    module_name = py_file.stem + "_online_eval_loaded"
    spec = importlib.util.spec_from_file_location(module_name, str(py_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import executor script: {py_file}")

    # Same import strategy as your batch launcher: make nearby dirs importable.
    parent_dir = str(py_file.parent)
    project_src = str(py_file.parent.parent)
    for p in [parent_dir, project_src]:
        if p not in sys.path:
            sys.path.insert(0, p)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------
# Dataset trajectory loading
# -----------------------------


@dataclass
class OnlineTrajectory:
    scene_id: str
    trajectory_name: str
    dataset_dir: Path
    uav_start_airsim: np.ndarray
    target_traj_airsim: np.ndarray
    jammer_trajs_airsim: Dict[str, np.ndarray]
    target_asset_name: Optional[str]
    jammer_asset_names: Dict[str, str]
    saved_instructions: Optional[List[str]]
    expert_action_physical: List[Optional[np.ndarray]]
    num_frames: int


def _load_instruction_series(dataset_dir: Path, uav_payload: Dict[str, Any]) -> Optional[List[str]]:
    candidates = [dataset_dir / "instruction.json", dataset_dir / "instructions.json"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = _load_json(path)
        except Exception:
            continue
        value = None
        for key in ["instructions", "instruction", "texts", "text"]:
            if key in data:
                value = data[key]
                break
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    for k in ["instruction", "text", "caption"]:
                        if isinstance(item.get(k), str):
                            out.append(item[k])
                            break
            if out:
                return [out[0]]
        frames = data.get("trajectory") or data.get("frames")
        if isinstance(frames, list):
            out = []
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                text = frame.get("instruction") or frame.get("text")
                if isinstance(text, str):
                    out.append(text)
            if out:
                return [out[0]]

    frames = uav_payload.get("trajectory")
    if isinstance(frames, list):
        out = []
        for frame in frames:
            if isinstance(frame, dict):
                text = frame.get("instruction") or frame.get("text")
                if isinstance(text, str):
                    out.append(text)
        if out:
            return [out[0]]
    return [EPISODE_INSTRUCTION]


def _load_target_trajectory(dataset_dir: Path, uav_frames: List[Dict[str, Any]]) -> np.ndarray:
    target_path = dataset_dir / "target_trajectory.json"
    raw: List[Any] = []
    if target_path.exists():
        data = _load_json(target_path)
        for key in ["target_trajectory_airsim", "target_trajectory", "trajectory"]:
            if isinstance(data.get(key), list):
                raw = data[key]
                break
    if not raw:
        for frame in uav_frames:
            if isinstance(frame, dict) and frame.get("target_position") is not None:
                raw.append(frame["target_position"])
    arr = [_dataset_xyz_to_airsim(p) for p in raw]
    arr = [p for p in arr if p is not None]
    if not arr:
        raise ValueError(f"cannot find target trajectory in {dataset_dir}")
    return np.stack(arr, axis=0).astype(np.float32)


def _load_jammer_trajectories(dataset_dir: Path, uav_frames: List[Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    jammer_path = dataset_dir / "jammer_trajectories.json"
    out: Dict[str, np.ndarray] = {}
    asset_names: Dict[str, str] = {}

    if jammer_path.exists():
        data = _load_json(jammer_path)
        assets = data.get("jammer_asset_names")
        if isinstance(assets, dict):
            asset_names = {str(k): str(v) for k, v in assets.items()}

        raw_map = data.get("jammer_trajectories_airsim") or data.get("jammer_trajectories")
        if isinstance(raw_map, dict):
            for did, series in raw_map.items():
                if not isinstance(series, list):
                    continue
                arr = [_dataset_xyz_to_airsim(p) for p in series]
                arr = [p for p in arr if p is not None]
                if arr:
                    out[str(did)] = np.stack(arr, axis=0).astype(np.float32)
        elif isinstance(raw_map, list):
            arr = [_dataset_xyz_to_airsim(p) for p in raw_map]
            arr = [p for p in arr if p is not None]
            if arr:
                out["1"] = np.stack(arr, axis=0).astype(np.float32)

    # Fallback: read per-frame `jammers` from uav_trajectory.json.
    if not out:
        raw_by_id: Dict[str, List[Any]] = {}
        for frame in uav_frames:
            if not isinstance(frame, dict):
                continue
            jammers = frame.get("jammers")
            if isinstance(jammers, list):
                for item in jammers:
                    if not isinstance(item, dict):
                        continue
                    did = str(item.get("id", "1"))
                    pos = item.get("position")
                    if pos is not None:
                        raw_by_id.setdefault(did, []).append(pos)
            elif frame.get("jammer_position") is not None:
                raw_by_id.setdefault("1", []).append(frame["jammer_position"])
        for did, series in raw_by_id.items():
            arr = [_dataset_xyz_to_airsim(p) for p in series]
            arr = [p for p in arr if p is not None]
            if arr:
                out[did] = np.stack(arr, axis=0).astype(np.float32)

    return out, asset_names


def _load_expert_actions(uav_frames: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
    actions: List[Optional[np.ndarray]] = []
    for frame in uav_frames:
        if not isinstance(frame, dict):
            actions.append(None)
            continue
        vel = frame.get("velocity_in_body_frame")
        yaw = frame.get("yaw_rate")
        vel_xyz = _xyz_from_any(vel)
        if vel_xyz is None or yaw is None:
            actions.append(None)
        else:
            actions.append(np.asarray([vel_xyz[0], vel_xyz[1], vel_xyz[2], float(yaw)], dtype=np.float32))
    return actions


def load_online_trajectory(dataset_dir: Path, scene_id: str) -> OnlineTrajectory:
    uav_path = dataset_dir / "uav_trajectory.json"
    if not uav_path.exists():
        raise FileNotFoundError(f"missing uav_trajectory.json: {uav_path}")
    uav_payload = _load_json(uav_path)
    frames = uav_payload.get("trajectory")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"invalid trajectory field in {uav_path}")

    uav0 = _dataset_xyz_to_airsim(frames[0].get("uav_position"))
    if uav0 is None:
        raise ValueError(f"missing first uav_position in {uav_path}")

    target = _load_target_trajectory(dataset_dir, frames)
    jammers, jammer_assets = _load_jammer_trajectories(dataset_dir, frames)
    instructions = _load_instruction_series(dataset_dir, uav_payload)
    expert_actions = _load_expert_actions(frames)
    lengths = [len(frames), len(target)]
    if jammers:
        lengths.extend([len(v) for v in jammers.values()])
    num_frames = int(min(lengths))

    return OnlineTrajectory(
        scene_id=scene_id,
        trajectory_name=dataset_dir.name,
        dataset_dir=dataset_dir,
        uav_start_airsim=uav0.astype(np.float32),
        target_traj_airsim=target[:num_frames],
        jammer_trajs_airsim={k: v[:num_frames] for k, v in jammers.items()},
        target_asset_name=uav_payload.get("target_asset_name"),
        jammer_asset_names=jammer_assets,
        saved_instructions=instructions,
        expert_action_physical=expert_actions[:num_frames],
        num_frames=num_frames,
    )


def discover_dataset_trajectories(
    dataset_root: Path,
    scene_ids: Sequence[str],
    trajectory_range: str = "",
    split: str = "all",
    val_ratio: float = 0.1,
    split_seed: int = 42,
    max_trajectories: int = 0,
) -> List[Path]:
    ranges = _parse_range_spec(trajectory_range)
    all_dirs: List[Path] = []
    seen_dirs: set[Path] = set()
    for scene_id in scene_ids:
        scene_dir = _case_insensitive_child(dataset_root, scene_id)
        if scene_dir is None:
            print(f"[warn] scene directory not found: {dataset_root / scene_id}")
            continue
        scene_dirs: List[Path] = []
        for uav_json in scene_dir.rglob("uav_trajectory.json"):
            d = uav_json.parent
            if _in_range(scene_id, d.name, ranges):
                resolved = d.resolve()
                if resolved not in seen_dirs:
                    scene_dirs.append(resolved)
                    seen_dirs.add(resolved)
        all_dirs.extend(sorted(scene_dirs, key=_natural_key))

    if split not in {"all", "train", "val"}:
        raise ValueError("--eval-split must be all/train/val")
    if split != "all":
        rng = random.Random(split_seed)
        shuffled = list(all_dirs)
        rng.shuffle(shuffled)
        val_n = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val_dirs = shuffled[:val_n]
        train_dirs = shuffled[val_n:] if val_n > 0 else shuffled
        all_dirs = val_dirs if split == "val" else train_dirs
        scene_rank = {str(scene_id).lower(): idx for idx, scene_id in enumerate(scene_ids)}
        all_dirs = sorted(
            all_dirs,
            key=lambda p: (
                scene_rank.get(p.parent.name.lower(), len(scene_rank)),
                _natural_key(p),
            ),
        )

    if max_trajectories > 0:
        all_dirs = all_dirs[: int(max_trajectories)]
    return all_dirs


# -----------------------------
# Model helpers
# -----------------------------


def _make_cfg_from_checkpoint(ckpt: Dict[str, Any], args: argparse.Namespace) -> ModelConfig:
    field_names = {f.name for f in dataclasses.fields(ModelConfig)}
    raw_cfg = migrate_legacy_config(ckpt.get("cfg", {}) or {})
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k in field_names}
    state = _strip_module_prefix(ckpt.get("model", {}) or {})
    if "use_fastwam_mot" not in cfg_kwargs:
        cfg_kwargs["use_fastwam_mot"] = any(k.startswith("fastwam.") for k in state)
    if "action_sequence_horizon" not in cfg_kwargs:
        token = state.get("actor.action_token_embed")
        if token is not None and getattr(token, "ndim", 0) == 3:
            cfg_kwargs["action_sequence_horizon"] = max(int(token.shape[1]) // max(int(args.action_dim), 1), 1)
    cfg_kwargs.update(
        {
            "image_size": args.image_size,
            "dinov2_model_name": args.dinov2_model_name,
            "dinov2_freeze": args.freeze_dinov2,
            "clip_text_model_name": args.clip_text_model_name,
            "clip_text_freeze": args.freeze_clip_text,
            "wan22_model_base_path": args.wan22_model_base_path,
            "wan22_fastwam_src_path": args.wan22_fastwam_src_path,
            "wan22_skip_download": args.wan22_skip_download,
            "wan22_text_encode_batch_size": args.wan22_text_encode_batch_size,
            "target_relative_dim": args.target_relative_dim,
            "action_dim": args.action_dim,
            "action_sampling_steps": args.sampling_steps,
            "compile_action_sampling": bool(args.compile_action_sampling),
            "compile_action_sampling_mode": str(args.compile_action_sampling_mode),
            "max_vel": args.max_vel,
            "max_yaw_rate": args.max_yaw_rate,
            "max_speed_norm": args.max_speed_norm,
        }
    )
    if getattr(args, "use_diffusion_actor", None) is not None:
        cfg_kwargs["use_diffusion_actor"] = bool(args.use_diffusion_actor)
    if getattr(args, "use_wan22_encoders", None) is not None:
        cfg_kwargs["use_wan22_encoders"] = bool(args.use_wan22_encoders)
    if getattr(args, "wan22_text_context_length", None) is not None:
        cfg_kwargs["wan22_text_context_length"] = int(args.wan22_text_context_length)
        if bool(cfg_kwargs.get("use_wan22_encoders", False)):
            cfg_kwargs["text_context_length"] = int(args.wan22_text_context_length)
    if getattr(args, "use_fastwam_mot", None) is not None:
        cfg_kwargs["use_fastwam_mot"] = bool(args.use_fastwam_mot)
    if getattr(args, "use_capture_value_reranking", None) is not None:
        cfg_kwargs["use_capture_value_reranking"] = bool(
            args.use_capture_value_reranking
        )
    capture_overrides = (
        "capture_value_score_mode",
        "capture_value_candidate_count",
        "capture_value_control_dt",
        "capture_value_horizontal_fov_deg",
        "capture_value_vertical_fov_deg",
        "capture_value_bbox_depth_scale",
        "capture_value_min_depth",
        "capture_value_max_depth",
        "capture_value_target_box_size",
        "capture_value_box_size_sigma",
        "capture_value_discount",
        "capture_value_recenter_sigma",
        "capture_value_pursuit_center_sigma",
        "capture_value_out_of_frame_weight",
        "capture_value_first_action_smooth_weight",
        "capture_value_temporal_smooth_weight",
        "capture_value_recenter_weight",
        "capture_value_pursuit_weight",
        "capture_value_smooth_weight",
        "capture_value_consensus_weight",
        "capture_value_short_horizon",
        "capture_value_selection_margin",
        "capture_value_min_center_error",
        "capture_action_prior_checkpoint",
        "capture_action_prior_dimension_weights",
        "capture_value_structured_candidates",
    )
    for name in capture_overrides:
        value = getattr(args, name, None)
        if value is not None:
            cfg_kwargs[name] = value
    if getattr(args, "target_token_fusion_mode", None) is not None:
        cfg_kwargs["target_token_fusion_mode"] = str(args.target_token_fusion_mode)
    if getattr(args, "dit_candidate_selection", None) is not None:
        cfg_kwargs["dit_candidate_selection"] = bool(args.dit_candidate_selection)
    if getattr(args, "dit_candidate_count", None) is not None:
        cfg_kwargs["dit_candidate_count"] = int(args.dit_candidate_count)
    if getattr(args, "dit_candidate_lateral_weight", None) is not None:
        cfg_kwargs["dit_candidate_lateral_weight"] = float(args.dit_candidate_lateral_weight)
    if getattr(args, "dit_candidate_vertical_weight", None) is not None:
        cfg_kwargs["dit_candidate_vertical_weight"] = float(args.dit_candidate_vertical_weight)
    if getattr(args, "dit_candidate_distance_weight", None) is not None:
        cfg_kwargs["dit_candidate_distance_weight"] = float(args.dit_candidate_distance_weight)
    if getattr(args, "dit_candidate_smooth_weight", None) is not None:
        cfg_kwargs["dit_candidate_smooth_weight"] = float(args.dit_candidate_smooth_weight)
    if getattr(args, "dit_candidate_yaw_angle_weight", None) is not None:
        cfg_kwargs["dit_candidate_yaw_angle_weight"] = float(args.dit_candidate_yaw_angle_weight)
    if getattr(args, "dit_candidate_pitch_angle_weight", None) is not None:
        cfg_kwargs["dit_candidate_pitch_angle_weight"] = float(args.dit_candidate_pitch_angle_weight)
    if getattr(args, "dit_candidate_final_distance_weight", None) is not None:
        cfg_kwargs["dit_candidate_final_distance_weight"] = float(args.dit_candidate_final_distance_weight)
    if getattr(args, "dit_candidate_progress_weight", None) is not None:
        cfg_kwargs["dit_candidate_progress_weight"] = float(args.dit_candidate_progress_weight)
    if getattr(args, "dit_candidate_front_weight", None) is not None:
        cfg_kwargs["dit_candidate_front_weight"] = float(args.dit_candidate_front_weight)
    if getattr(args, "dit_candidate_action_weight", None) is not None:
        cfg_kwargs["dit_candidate_action_weight"] = float(args.dit_candidate_action_weight)
    if getattr(args, "dit_candidate_temporal_smooth_weight", None) is not None:
        cfg_kwargs["dit_candidate_temporal_smooth_weight"] = float(args.dit_candidate_temporal_smooth_weight)
    if getattr(args, "use_target_relative_context", None) is not None:
        cfg_kwargs["use_target_relative_context"] = bool(args.use_target_relative_context)
    if getattr(args, "target_relative_context_scale", None) is not None:
        cfg_kwargs["target_relative_context_scale"] = float(args.target_relative_context_scale)
    if getattr(args, "target_relative_token_scale", None) is not None:
        cfg_kwargs["target_relative_token_scale"] = float(args.target_relative_token_scale)
    if getattr(args, "target_relative_context_hidden_dim", None) is not None:
        cfg_kwargs["target_relative_context_hidden_dim"] = int(args.target_relative_context_hidden_dim)
    if getattr(args, "force_direct_action", False):
        cfg_kwargs["use_diffusion_actor"] = False
    return ModelConfig(**cfg_kwargs)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return migrate_legacy_state_dict_keys(state_dict)


def _load_companion_s0_state(
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], Optional[Path]]:
    """Load the frozen S0 parameters omitted by a V5 main-stage checkpoint."""
    state_format = str(checkpoint.get("model_state_format") or "full")
    run_args = checkpoint.get("run_args", {}) or {}
    if state_format != "trainable_only" or str(run_args.get("training_stage", "")) != "main":
        return {}, None

    raw_path = str(run_args.get("s0_localizer_checkpoint", "")).strip()
    if not raw_path:
        raise ValueError(
            "V5 main trainable-only checkpoint is missing run_args.s0_localizer_checkpoint."
        )
    s0_path = Path(raw_path).expanduser()
    if not s0_path.is_absolute():
        local_path = checkpoint_path.parent / s0_path
        s0_path = local_path if local_path.is_file() else s0_path
    s0_path = s0_path.resolve()
    if not s0_path.is_file():
        raise FileNotFoundError(f"V5 companion S0 checkpoint not found: {s0_path}")

    payload = torch.load(s0_path, map_location="cpu", weights_only=False)
    try:
        state = normalize_s0_checkpoint_state(payload)
    except ValueError as exc:
        raise ValueError(f"Invalid companion S0 checkpoint {s0_path}: {exc}") from exc
    return state, s0_path


def _summarize_checkpoint_load(
    model: torch.nn.Module,
    cfg: ModelConfig,
    missing: Sequence[str],
    unexpected: Sequence[str],
    ckpt: Dict[str, Any],
) -> None:
    state_format = str(ckpt.get("model_state_format") or "full")
    if state_format == "trainable_only":
        trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
        expected_missing = [name for name in missing if name not in trainable_names]
        unexpected_missing = [name for name in missing if name in trainable_names]
    else:
        expected_missing = []
        unexpected_missing = list(missing)

    if expected_missing:
        groups = {
            "image_encoder": 0,
            "text_encoder": 0,
            "inactive_actor": 0,
            "other_frozen": 0,
        }
        for name in expected_missing:
            if name.startswith("image_encoder."):
                groups["image_encoder"] += 1
            elif name.startswith("text_encoder."):
                groups["text_encoder"] += 1
            elif (not cfg.use_diffusion_actor) and name.startswith("actor."):
                groups["inactive_actor"] += 1
            else:
                groups["other_frozen"] += 1
        group_s = ", ".join(f"{k}={v}" for k, v in groups.items() if v)
        print(
            f"[checkpoint] trainable_only checkpoint; skipped expected frozen/inactive keys: "
            f"{len(expected_missing)} ({group_s})"
        )
    if unexpected_missing:
        print(f"[warn] missing trainable keys when loading checkpoint: {unexpected_missing}")
    if unexpected:
        print(f"[warn] unexpected keys when loading checkpoint: {list(unexpected)}")


def load_model(args: argparse.Namespace, device: torch.device) -> Tuple[TeacherWorldModelDiT, ModelConfig]:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _make_cfg_from_checkpoint(ckpt, args)
    if bool(getattr(cfg, "use_future_state_dit", False)) and int(
        getattr(cfg, "tracker_state_action_alignment_version", -1)
    ) != 4:
        raise ValueError("Incompatible checkpoint: Future State DiT online eval requires V4.")
    model = TeacherWorldModelDiT(cfg).to(device)
    main_state = _strip_module_prefix(ckpt["model"])
    s0_state, s0_path = _load_companion_s0_state(ckpt_path, ckpt)
    combined_state = {**s0_state, **main_state}
    if s0_path is not None:
        required_s0 = {
            name
            for name, _ in model.named_parameters()
            if name.startswith(S0_PARAMETER_PREFIXES)
        }
        missing_s0 = sorted(required_s0.difference(combined_state))
        if missing_s0:
            raise ValueError(
                f"Companion S0 checkpoint is incomplete; missing parameters: {missing_s0[:8]}"
            )
    missing, unexpected = model.load_state_dict(combined_state, strict=False)
    if (
        bool(getattr(cfg, "use_future_state_dit", False))
        or bool(getattr(cfg, "use_current_box_action_conditioning", False))
    ):
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        missing_trainable = sorted(name for name in missing if name in trainable)
        if missing_trainable or unexpected:
            raise ValueError(
                "Incompatible structured future checkpoint parameters: "
                f"missing_trainable={missing_trainable}, unexpected={list(unexpected)}"
            )
    _summarize_checkpoint_load(model, cfg, missing, unexpected, ckpt)
    model.eval()
    if s0_path is not None:
        print(f"[model] loaded frozen S0 companion checkpoint: {s0_path}")
    print(f"[model] loaded checkpoint: {ckpt_path}")
    print(
        f"[model] low_dim_target_input=off, "
        f"target_token_fusion_mode={cfg.target_token_fusion_mode}, "
        f"use_diffusion_actor={cfg.use_diffusion_actor}, "
        f"use_fastwam_mot={cfg.use_fastwam_mot}, "
        f"dit_candidate_selection={cfg.dit_candidate_selection}, "
        f"dit_candidate_count={cfg.dit_candidate_count}, "
        f"candidate_score=tracking, "
        f"target_relative_context={cfg.use_target_relative_context}, "
        f"target_relative_token_scale={cfg.target_relative_token_scale}, "
        f"tracker_center_context={cfg.use_tracker_center_context}, "
        f"tracker_heatmap_target={cfg.tracker_heatmap_target_mode}, "
        f"tracker_mot_integration={cfg.tracker_mot_integration}, "
        f"tracker_condition_mode={cfg.tracker_condition_mode}"
    )
    return model, cfg


def make_image_transform(image_size: int):
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])


def rgb_to_model_tensor(rgb: np.ndarray, transform, device: torch.device) -> torch.Tensor:
    if rgb is None:
        raise RuntimeError("AirSim returned empty RGB image")
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return transform(img).unsqueeze(0).to(device).float()


def tokenize_instruction(tokenizer, text: str, max_length: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def warmup_compiled_action_sampler(
    model: TeacherWorldModelDiT,
    cfg: ModelConfig,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Finish the lazy Inductor compile before any AirSim scene is opened."""
    if not bool(args.compile_action_sampling) or model.fastwam is None or not device.type == "cuda":
        return

    print("[torch.compile] warming up action sampler before AirSim startup...", flush=True)
    started = time.perf_counter()
    instruction = "Keep tracking and approaching the target UAV."
    image = torch.zeros(
        1,
        3,
        int(cfg.image_size),
        int(cfg.image_size),
        dtype=torch.float32,
        device=device,
    )
    if cfg.use_wan22_encoders:
        text_tokens = torch.zeros(1, 1, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(text_tokens)
    else:
        text_tokens, attention_mask = tokenize_instruction(
            tokenizer,
            instruction,
            cfg.text_context_length,
            device,
        )

    guidance_heatmap = None
    guidance_confidence = None
    if bool(getattr(cfg, "use_fastwam_attention_bias", False)) or bool(
        args.reuse_last_confident_action_sequence
    ) or bool(getattr(cfg, "use_tracker_center_context", False)):
        guidance_heatmap = torch.full(
            (1, int(cfg.image_size), int(cfg.image_size)),
            1.0 / float(int(cfg.image_size) * int(cfg.image_size)),
            dtype=torch.float32,
            device=device,
        )
        guidance_confidence = torch.ones(1, 1, dtype=torch.float32, device=device)

    tracker_center = None
    if _tracker_center_required(cfg):
        tracker_center = torch.full((1, 2), 0.5, dtype=torch.float32, device=device)
    tracker_features = None
    if _tracker_features_required(cfg):
        tracker_features = torch.zeros(
            1,
            int(cfg.tracker_feature_grid_size) ** 2,
            int(cfg.tracker_feature_dim),
            dtype=torch.float32,
            device=device,
        )
    tracker_bbox = None
    if _tracker_bbox_required(cfg):
        tracker_bbox = torch.tensor([[0.5, 0.5, 0.1, 0.1]], dtype=torch.float32, device=device)
    tracker_response = None
    if _tracker_response_required(cfg):
        response_size = int(getattr(cfg, "tracker_response_grid_size", 7))
        tracker_response = torch.full(
            (1, response_size, response_size),
            1.0 / float(response_size * response_size),
            dtype=torch.float32,
            device=device,
        )
    tracker_search_geometry = None
    tracker_image_size = None
    if _tracker_geometry_required(cfg):
        image_size = float(cfg.image_size)
        tracker_search_geometry = torch.tensor(
            [[0.0, 0.0, image_size]], dtype=torch.float32, device=device
        )
        tracker_image_size = torch.tensor(
            [[image_size, image_size]], dtype=torch.float32, device=device
        )
    tracker_template = tracker_search = None
    if str(getattr(cfg, "tracker_mot_integration", "none")) == "mot_tracker_finetune_local_feature":
        tracker_template = torch.zeros(
            1, 3, int(cfg.tracker_template_size), int(cfg.tracker_template_size),
            dtype=torch.float32, device=device,
        )
        tracker_search = torch.zeros(
            1, 3, int(cfg.tracker_search_size), int(cfg.tracker_search_size),
            dtype=torch.float32, device=device,
        )
    target_box_history = target_box_history_valid = None
    if bool(getattr(cfg, "use_historical_target_memory", False)):
        previous_length = int(getattr(cfg, "target_history_length", 8)) - 1
        target_box_history = torch.zeros(
            1, previous_length, 5, dtype=torch.float32, device=device
        )
        target_box_history_valid = torch.zeros(
            1, previous_length, dtype=torch.bool, device=device
        )

    model.act(
        image=image,
        text_tokens=text_tokens,
        target_relative=torch.zeros(
            1,
            int(cfg.target_relative_dim),
            dtype=torch.float32,
            device=device,
        ),
        prev_action=torch.zeros(1, int(cfg.action_dim), dtype=torch.float32, device=device),
        rssm_state=None,
        attention_mask=attention_mask,
        prev_done=torch.zeros(1, dtype=torch.float32, device=device),
        deterministic=bool(args.deterministic_action),
        num_steps=int(args.sampling_steps),
        instruction=instruction,
        save_transformer_attention=False,
        save_predicted_video=False,
        guidance_heatmap=guidance_heatmap,
        guidance_confidence=guidance_confidence,
        tracker_center=tracker_center,
        tracker_features=tracker_features,
        tracker_bbox=tracker_bbox,
        tracker_response=tracker_response,
        tracker_search_geometry=tracker_search_geometry,
        tracker_image_size=tracker_image_size,
        tracker_template=tracker_template,
        tracker_search=tracker_search,
        target_box_history=target_box_history,
        target_box_history_valid=target_box_history_valid,
    )
    torch.cuda.synchronize(device)
    print(
        f"[torch.compile] action sampler warmup finished in {time.perf_counter() - started:.1f}s; "
        "AirSim may now start.",
        flush=True,
    )


def close_requested_scenes_before_compile(args: argparse.Namespace, scene_ids: Sequence[str]) -> None:
    """Release stale Unreal workers so compile warmup owns the model GPU."""
    try:
        import msgpackrpc

        client = msgpackrpc.Client(
            msgpackrpc.Address(args.sim_server_host, int(args.sim_server_port)),
            timeout=5,
        )
        try:
            client.call("close_scenes", args.sim_server_host, list(scene_ids))
        finally:
            client.close()
        print(f"[precompile] closed existing AirSim scenes: {list(scene_ids)}", flush=True)
    except Exception:
        # This is expected when --start-sim-server is used and the server is not up yet.
        pass


# -----------------------------
# Online control helpers
# -----------------------------


def _quat_to_airsim_quat(quat_xyzw: np.ndarray):
    import airsim

    return airsim.Quaternionr(
        x_val=float(quat_xyzw[0]),
        y_val=float(quat_xyzw[1]),
        z_val=float(quat_xyzw[2]),
        w_val=float(quat_xyzw[3]),
    )


def _wxyz_quat_to_airsim_quat(quat_wxyz: Sequence[float]):
    import airsim

    return airsim.Quaternionr(
        w_val=float(quat_wxyz[0]),
        x_val=float(quat_wxyz[1]),
        y_val=float(quat_wxyz[2]),
        z_val=float(quat_wxyz[3]),
    )


def _yaw_to_airsim_quat(yaw_rad: float):
    quat_xyzw = R.from_euler("xyz", [0.0, 0.0, float(yaw_rad)], degrees=False).as_quat()
    return _quat_to_airsim_quat(quat_xyzw)


def set_vehicle_pose_static(
    executor,
    position_airsim: np.ndarray,
    quat,
    retries: int = 3,
    tol_xy: float = 0.3,
    tol_z: float = 0.3,
) -> Dict[str, Any]:
    """Set UAV pose while paused without advancing physics frames."""
    import airsim

    pos = np.asarray(position_airsim, dtype=np.float32).reshape(3)
    executor._safe_sim_pause(True)
    last_state = None
    for _ in range(int(retries)):
        executor.client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(float(pos[0]), float(pos[1]), float(pos[2])),
                quat,
            ),
            ignore_collision=True,
            vehicle_name=executor.uav_vehicle_name,
        )
        last_state = executor.get_uav_state()
        actual = np.asarray(last_state["position"], dtype=np.float32)
        err_xy = float(np.linalg.norm(actual[:2] - pos[:2]))
        err_z = float(abs(actual[2] - pos[2]))
        if err_xy <= float(tol_xy) and err_z <= float(tol_z):
            return last_state
    if last_state is not None:
        actual = np.asarray(last_state["position"], dtype=np.float32)
        raise RuntimeError(
            f"static pose set failed: target=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}), "
            f"actual=({actual[0]:.2f},{actual[1]:.2f},{actual[2]:.2f})"
        )
    raise RuntimeError("static pose set failed: no UAV state returned")


def _get_yaw_from_state(uav_state: Dict[str, Any]) -> float:
    q = uav_state["orientation"]
    rot = R.from_quat([float(q[1]), float(q[2]), float(q[3]), float(q[0])])
    return float(rot.as_euler("xyz", degrees=False)[2])


def compute_target_relative_body(executor, uav_state: Dict[str, Any], target_pos_airsim: np.ndarray) -> np.ndarray:
    uav_pos_airsim = uav_state["position"]
    q = uav_state["orientation"]
    uav_pos_dataset = np.asarray([uav_pos_airsim[0], uav_pos_airsim[1], -uav_pos_airsim[2]], dtype=np.float32)
    target_pos_dataset = np.asarray([target_pos_airsim[0], target_pos_airsim[1], -target_pos_airsim[2]], dtype=np.float32)
    rel_dataset = target_pos_dataset - uav_pos_dataset
    rel_body = executor._world_to_body_frame(
        rel_dataset,
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    )
    return np.asarray(rel_body, dtype=np.float32)


def _wrap_angle_rad(angle: float) -> float:
    return float(np.arctan2(np.sin(float(angle)), np.cos(float(angle))))


def _target_facing_yaw_airsim(uav_pos_airsim: np.ndarray, target_pos_airsim: np.ndarray, fallback_yaw: float) -> float:
    dx = float(target_pos_airsim[0]) - float(uav_pos_airsim[0])
    dy = float(target_pos_airsim[1]) - float(uav_pos_airsim[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return float(fallback_yaw)
    return _wrap_angle_rad(math.atan2(dy, dx))


def _normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < eps:
        return np.zeros_like(arr)
    return arr / n


def _planner_direct_chase_next_dataset(
    tracker_pos_dataset: np.ndarray,
    target_now_dataset: np.ndarray,
    target_next_dataset: np.ndarray,
    step_length: float,
) -> np.ndarray:
    """One-step online version of planner `_build_tracker_direct_chase`.

    The offline planner samples a complete tracker trajectory and validates
    collision/FOV globally. Online eval only has the current closed-loop state,
    so this mirrors the planner's local direct-chase candidate selection.
    """
    curr = np.asarray(tracker_pos_dataset, dtype=np.float64)
    target_now = np.asarray(target_now_dataset, dtype=np.float64)
    target_next = np.asarray(target_next_dataset, dtype=np.float64)

    rel_now = target_now - curr
    rel_dir_now = _normalize_np(rel_now)
    if float(np.linalg.norm(rel_dir_now)) < 1e-6:
        return curr.copy()

    max_next_z = float(target_next[2])
    capped_now = target_now.copy()
    capped_now[2] = min(float(capped_now[2]), max_next_z)
    capped_next = target_next.copy()
    capped_next[2] = min(float(capped_next[2]), max_next_z)
    aim_points = [
        target_now,
        target_next,
        capped_now,
        capped_next,
        0.5 * target_now + 0.5 * target_next,
    ]

    cos_thresh = math.cos(math.radians(30.0))
    candidates: List[Tuple[float, np.ndarray]] = []
    for aim in aim_points:
        move_dir = _normalize_np(np.asarray(aim, dtype=np.float64) - curr)
        if float(np.linalg.norm(move_dir)) < 1e-6:
            continue
        align = float(np.dot(move_dir, rel_dir_now))
        if align < cos_thresh:
            continue
        nxt = curr + float(step_length) * move_dir
        if float(nxt[2]) > max_next_z + 1e-6:
            continue
        candidates.append((align, nxt))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return np.asarray(candidates[0][1], dtype=np.float32)

    return np.asarray(curr + float(step_length) * rel_dir_now, dtype=np.float32)


def compute_live_expert_action(
    executor,
    uav_state: Dict[str, Any],
    target_now_airsim: np.ndarray,
    target_next_airsim: Optional[np.ndarray],
    max_speed_norm: float,
    max_yaw_rate: float,
) -> np.ndarray:
    """Planner-style oracle action for the current online state.

    It mirrors the direct-chase branch in
    `trajectory_generator_with_5_jammers.py`: choose a one-step tracker move
    from current UAV state toward current/lookahead target candidates, then
    convert that planned tracker step into the same body-frame action format
    used by training labels.
    """
    uav_pos_airsim = np.asarray(uav_state["position"], dtype=np.float32)
    q = uav_state["orientation"]
    target_now = np.asarray(target_now_airsim, dtype=np.float32)
    target_next = np.asarray(
        target_next_airsim if target_next_airsim is not None else target_now_airsim,
        dtype=np.float32,
    )

    tracker_dataset = np.asarray([uav_pos_airsim[0], uav_pos_airsim[1], -uav_pos_airsim[2]], dtype=np.float32)
    target_now_dataset = np.asarray([target_now[0], target_now[1], -target_now[2]], dtype=np.float32)
    target_next_dataset = np.asarray([target_next[0], target_next[1], -target_next[2]], dtype=np.float32)

    step_length = max(float(max_speed_norm), 1e-6)
    next_tracker_dataset = _planner_direct_chase_next_dataset(
        tracker_dataset,
        target_now_dataset,
        target_next_dataset,
        step_length=step_length,
    )
    delta_dataset = np.asarray(next_tracker_dataset - tracker_dataset, dtype=np.float32)
    velocity_body = executor._world_to_body_frame(
        delta_dataset,
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    )

    current_yaw = _get_yaw_from_state(uav_state)
    next_tracker_airsim = np.asarray(
        [next_tracker_dataset[0], next_tracker_dataset[1], -next_tracker_dataset[2]],
        dtype=np.float32,
    )
    desired_yaw = _target_facing_yaw_airsim(next_tracker_airsim, target_next, fallback_yaw=current_yaw)
    yaw_rate_deg = math.degrees(_wrap_angle_rad(desired_yaw - current_yaw))
    yaw_cap = abs(float(max_yaw_rate))
    if yaw_cap > 0.0:
        yaw_rate_deg = float(np.clip(yaw_rate_deg, -yaw_cap, yaw_cap))

    action = np.asarray(
        [float(velocity_body[0]), float(velocity_body[1]), float(velocity_body[2]), float(yaw_rate_deg)],
        dtype=np.float32,
    )
    return np.asarray(
        clamp_physical_action_speed(action, max_speed_norm=max_speed_norm),
        dtype=np.float32,
    )


def _axis_word(value: float, pos_word: str, neg_word: str, threshold: float) -> Optional[str]:
    if abs(value) < threshold:
        return None
    return pos_word if value >= 0 else neg_word


def instruction_from_relative(rel_body: np.ndarray, next_rel_body: Optional[np.ndarray] = None) -> str:
    x, y, z = [float(v) for v in rel_body]
    horizontal = []
    fb = _axis_word(x, "front", "behind", 1.0)
    lr = _axis_word(y, "right", "left", 1.0)
    if fb:
        horizontal.append(fb)
    if lr:
        horizontal.append(lr)
    if not horizontal:
        horizontal.append("near the center")

    vertical = _axis_word(z, "above", "below", 0.75)
    if vertical is None:
        vertical_phrase = "at a similar altitude"
    else:
        vertical_phrase = f"slightly {vertical}" if abs(z) < 5.0 else vertical

    moving_phrase = ""
    if next_rel_body is not None:
        d = next_rel_body - rel_body
        axes = [
            (abs(float(d[0])), "forward" if d[0] >= 0 else "backward"),
            (abs(float(d[1])), "right" if d[1] >= 0 else "left"),
            (abs(float(d[2])), "up" if d[2] >= 0 else "down"),
        ]
        mag, word = max(axes, key=lambda x: x[0])
        if mag > 0.2:
            moving_phrase = f", moving {word}"

    return f"Target is {'-'.join(horizontal)} and {vertical_phrase}{moving_phrase}. Keep approaching while maintaining visual lock."


def physical_action_to_norm(action_physical: np.ndarray, max_vel: float, max_yaw_rate: float) -> np.ndarray:
    out = np.zeros(4, dtype=np.float32)
    out[:3] = np.asarray(action_physical[:3], dtype=np.float32) / max(float(max_vel), 1e-6)
    out[3] = float(action_physical[3]) / max(float(max_yaw_rate), 1e-6)
    return np.clip(out, -1.0, 1.0)


def apply_action_by_pose(
    executor,
    action_physical: np.ndarray,
    dt: float,
    max_step_norm: float,
) -> Dict[str, Any]:
    """Apply predicted action by deterministic pose integration.

    This matches your data-collection executor style better than a fully dynamic AirSim
    velocity command, because dataset generation itself used pose setting and frame stepping.
    """
    uav_state = executor.get_uav_state()
    pos = np.asarray(uav_state["position"], dtype=np.float32)
    q = uav_state["orientation"]
    rot = R.from_quat([float(q[1]), float(q[2]), float(q[3]), float(q[0])])

    action = np.asarray(action_physical, dtype=np.float32).copy()
    # The dataset action z comes from TrajectoryExecutor._world_to_body_frame(),
    # which already uses AirSim's body-frame z sign after its saved-coordinate
    # conversion. Execute it directly here; flipping it again makes vertical
    # tracking diverge.
    body_ned = np.asarray([action[0], action[1], action[2]], dtype=np.float32)
    step_norm = float(np.linalg.norm(body_ned) * dt)
    if max_step_norm > 0 and step_norm > max_step_norm:
        body_ned *= float(max_step_norm / max(step_norm, 1e-6))

    delta_world_airsim = rot.apply(body_ned) * float(dt)
    new_pos = pos + delta_world_airsim.astype(np.float32)

    euler = rot.as_euler("xyz", degrees=False)
    new_yaw = float(euler[2]) + math.radians(float(action[3]) * float(dt))
    new_rot = R.from_euler("xyz", [float(euler[0]), float(euler[1]), new_yaw], degrees=False)
    new_quat_xyzw = new_rot.as_quat()

    quat = _quat_to_airsim_quat(new_quat_xyzw)
    # Set the predicted pose while paused, but do not advance physics here.
    # All rendering is concentrated immediately before the next RGB capture.
    return set_vehicle_pose_static(
        executor,
        new_pos,
        quat,
        retries=3,
        tol_xy=0.8,
        tol_z=0.8,
    )


def apply_action_to_virtual_state(
    uav_state: Dict[str, Any],
    action_physical: np.ndarray,
    dt: float,
    max_step_norm: float,
) -> Dict[str, Any]:
    pos = np.asarray(uav_state["position"], dtype=np.float32)
    q = np.asarray(uav_state["orientation"], dtype=np.float64)
    rot = R.from_quat([float(q[1]), float(q[2]), float(q[3]), float(q[0])])
    action = np.asarray(action_physical, dtype=np.float32)
    body_ned = action[:3].copy()
    step_norm = float(np.linalg.norm(body_ned) * dt)
    if max_step_norm > 0 and step_norm > max_step_norm:
        body_ned *= float(max_step_norm / max(step_norm, 1.0e-6))
    new_pos = pos + (rot.apply(body_ned) * float(dt)).astype(np.float32)
    euler = rot.as_euler("xyz", degrees=False)
    new_yaw = float(euler[2]) + math.radians(float(action[3]) * float(dt))
    new_q_xyzw = R.from_euler(
        "xyz", [float(euler[0]), float(euler[1]), new_yaw], degrees=False
    ).as_quat()
    return {
        "position": new_pos,
        "orientation": np.asarray(
            [new_q_xyzw[3], new_q_xyzw[0], new_q_xyzw[1], new_q_xyzw[2]],
            dtype=np.float64,
        ),
        "has_collided": False,
        "collision_time_stamp": None,
    }


def apply_action_by_velocity(executor, action_physical: np.ndarray, dt: float) -> Dict[str, Any]:
    import airsim

    action = np.asarray(action_physical, dtype=np.float32)
    executor._safe_sim_pause(False)
    executor.client.moveByVelocityBodyFrameAsync(
        float(action[0]),
        float(action[1]),
        float(action[2]),
        float(dt),
        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(action[3])),
        vehicle_name=executor.uav_vehicle_name,
    ).join()
    if getattr(executor, "deterministic_step_mode", False):
        executor._safe_sim_pause(True)
    return executor.get_uav_state()


def is_visible_by_geometry(rel_body: np.ndarray, fov_deg: float = 90.0) -> bool:
    x, y, z = [float(v) for v in rel_body]
    if x <= 0.1:
        return False
    h_ang = abs(math.degrees(math.atan2(y, max(x, 1e-6))))
    v_ang = abs(math.degrees(math.atan2(z, max(math.sqrt(x * x + y * y), 1e-6))))
    return h_ang <= fov_deg / 2.0 and v_ang <= fov_deg / 2.0


def _delta_xyz_to_dict(delta: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if delta is None:
        return None
    arr = np.asarray(delta, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    return {"x": float(arr[0]), "y": float(arr[1]), "z": float(arr[2])}


def _collision_info_to_dict(collision_info: Any) -> Optional[Dict[str, Any]]:
    if collision_info is None:
        return None
    out: Dict[str, Any] = {"has_collided": bool(getattr(collision_info, "has_collided", False))}
    for attr in ["object_name", "object_id", "time_stamp", "penetration_depth"]:
        if hasattr(collision_info, attr):
            try:
                value = getattr(collision_info, attr)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    out[attr] = value
            except Exception:
                pass
    for attr in ["position", "impact_point", "normal"]:
        if hasattr(collision_info, attr):
            try:
                xyz = _xyz_from_any(getattr(collision_info, attr))
                if xyz is not None:
                    # Store positions in the same z-up convention as online_rollout.json.
                    out[attr] = _airsim_xyz_to_dataset(xyz)
            except Exception:
                pass
    return out


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_attention_map_overlay(
    path: Path,
    rgb: np.ndarray,
    attention_map: torch.Tensor,
    alpha: float = 0.45,
) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    attn = attention_map.detach().float().cpu().squeeze().numpy()
    if attn.ndim != 2:
        raise ValueError(f"expected 2D attention map after squeeze, got shape {attn.shape}")
    vis = attn - float(np.min(attn))
    denom = float(np.max(vis))
    if denom > 1e-8:
        vis = vis / denom
    vis_u8 = np.clip(vis * 255.0, 0, 255).astype(np.uint8)
    vis_u8 = cv2.resize(vis_u8, (rgb_u8.shape[1], rgb_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored_bgr = cv2.applyColorMap(vis_u8, cv2.COLORMAP_VIRIDIS)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.addWeighted(rgb_u8, 1.0 - float(alpha), colored_rgb, float(alpha), 0.0)
    cv2.imwrite(str(path.with_suffix(".png")), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))


def save_tracker_attention_overlay(
    path: Path,
    rgb: np.ndarray,
    attention_map: torch.Tensor,
    search_geometry: torch.Tensor,
    label: Optional[str] = None,
    alpha: float = 0.45,
) -> None:
    """Project a 16x16 Tracker-search attention map back into the full RGB image."""
    import cv2

    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    height, width = rgb_u8.shape[:2]
    attn = attention_map.detach().float().cpu().squeeze().numpy()
    if attn.ndim != 2:
        raise ValueError(f"expected 2D Tracker attention map, got shape {attn.shape}")
    x1, y1, side = search_geometry.detach().float().view(-1)[:3].cpu().numpy()
    x1, y1, side = int(round(x1)), int(round(y1)), max(int(round(side)), 1)
    x2, y2 = x1 + side, y1 + side
    clip_x1, clip_y1 = max(x1, 0), max(y1, 0)
    clip_x2, clip_y2 = min(x2, width), min(y2, height)
    full = np.zeros((height, width), dtype=np.float32)
    if clip_x2 > clip_x1 and clip_y2 > clip_y1:
        local = attn - float(attn.min())
        local /= max(float(local.max()), 1.0e-8)
        resized = cv2.resize(local, (side, side), interpolation=cv2.INTER_CUBIC)
        full[clip_y1:clip_y2, clip_x1:clip_x2] = resized[
            clip_y1 - y1:clip_y2 - y1, clip_x1 - x1:clip_x2 - x1
        ]
    colored = cv2.applyColorMap(np.clip(full * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    overlay = cv2.addWeighted(rgb_u8, 1.0 - float(alpha), cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), float(alpha), 0)
    if label:
        canvas = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        overlay = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path.with_suffix(".png")), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_target_crop_action_overlay(
    path: Path,
    rgb: np.ndarray,
    target_relative_body: np.ndarray,
    action_sequence_physical: np.ndarray,
    fov_deg: float,
    camera_offset_body: Sequence[float],
    dt: float,
    ortrack_bbox_xywh: Optional[Sequence[float]] = None,
    ortrack_confidence: Optional[float] = None,
    model_driven_search_geometry: Optional[Sequence[float]] = None,
    current_state_box_cxcywh: Optional[Sequence[float]] = None,
    future_state_box_cxcywh: Optional[Sequence[float]] = None,
    future_state_boxes_cxcywh: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Draw target boxes plus predicted state and body-frame action rollouts."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]
    green = (64, 255, 64)
    gt_box = None
    try:
        bbox = _bbox_from_target_center(
            np.asarray(target_relative_body, dtype=np.float32),
            (height, width),
            float(fov_deg),
            tuple(float(v) for v in camera_offset_body),
            0.10,
        )
        bx, by, bw, bh = [int(round(v)) for v in bbox]
        bx = max(0, min(bx, width - 1))
        by = max(0, min(by, height - 1))
        bw = max(1, min(bw, width - bx))
        bh = max(1, min(bh, height - by))
        gt_box = [bx, by, bx + bw, by + bh]
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), green, 3)
        cv2.putText(canvas, "GT", (bx, max(by - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)
        cv2.drawMarker(canvas, (bx + bw // 2, by + bh // 2), green, cv2.MARKER_CROSS, 18, 2)
    except RuntimeError:
        pass

    search_box = None
    if model_driven_search_geometry is not None and len(model_driven_search_geometry) >= 3:
        sx, sy, side = [int(round(float(v))) for v in model_driven_search_geometry[:3]]
        search_box = [sx, sy, sx + side, sy + side]
        cyan = (255, 220, 0)
        cv2.rectangle(
            canvas,
            (max(sx, 0), max(sy, 0)),
            (min(sx + side, width - 1), min(sy + side, height - 1)),
            cyan,
            2,
        )
        cv2.putText(
            canvas,
            "Search crop",
            (max(sx, 4), max(sy + 18, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            cyan,
            2,
        )

    def draw_normalized_state_box(
        box: Optional[Sequence[float]], label: str, color: tuple[int, int, int]
    ) -> Optional[List[int]]:
        if box is None or len(box) < 4:
            return None
        cx, cy, bw, bh = [float(v) for v in box[:4]]
        x1 = int(round((cx - 0.5 * bw) * width))
        y1 = int(round((cy - 0.5 * bh) * height))
        x2 = int(round((cx + 0.5 * bw) * width))
        y2 = int(round((cy + 0.5 * bh) * height))
        clipped_x1, clipped_y1 = max(x1, 0), max(y1, 0)
        clipped_x2, clipped_y2 = min(x2, width - 1), min(y2, height - 1)
        if clipped_x2 >= clipped_x1 and clipped_y2 >= clipped_y1:
            cv2.rectangle(canvas, (clipped_x1, clipped_y1), (clipped_x2, clipped_y2), color, 3)
            cv2.putText(
                canvas,
                label,
                (clipped_x1, max(clipped_y1 - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.drawMarker(
                canvas,
                (int(round(cx * width)), int(round(cy * height))),
                color,
                cv2.MARKER_CROSS,
                18,
                2,
            )
        return [x1, y1, x2, y2]

    future_boxes = future_state_boxes_cxcywh
    if future_boxes is None and future_state_box_cxcywh is not None:
        future_boxes = [future_state_box_cxcywh]
    future_state_boxes: List[List[int]] = []
    state_points: List[List[int]] = []
    if current_state_box_cxcywh is not None and len(current_state_box_cxcywh) >= 2:
        current_cx, current_cy = [float(v) for v in current_state_box_cxcywh[:2]]
        if math.isfinite(current_cx) and math.isfinite(current_cy):
            state_points.append(
                [int(round(current_cx * width)), int(round(current_cy * height))]
            )
    valid_future_boxes: List[Sequence[float]] = []
    if future_boxes is not None:
        valid_future_boxes = [
            box
            for box in future_boxes
            if box is not None
            and len(box) >= 4
            and all(math.isfinite(float(value)) for value in box[:4])
        ]
    for box in valid_future_boxes:
        cx, cy = [float(v) for v in box[:2]]
        state_points.append([int(round(cx * width)), int(round(cy * height))])

    current_state_box = draw_normalized_state_box(
        current_state_box_cxcywh, "s0 current", (255, 0, 255)
    )
    for box in valid_future_boxes:
        cx, cy, box_width, box_height = [float(value) for value in box[:4]]
        future_state_boxes.append(
            [
                int(round((cx - 0.5 * box_width) * width)),
                int(round((cy - 0.5 * box_height) * height)),
                int(round((cx + 0.5 * box_width) * width)),
                int(round((cy + 0.5 * box_height) * height)),
            ]
        )

    ortrack_box = None
    # In model-driven mode this is only the previous-state search anchor, not a
    # current detection. Keep drawing legacy Tracker output, but hide the anchor.
    if (
        current_state_box_cxcywh is None
        and ortrack_bbox_xywh is not None
        and len(ortrack_bbox_xywh) >= 4
    ):
        ox, oy, ow, oh = [int(round(float(v))) for v in ortrack_bbox_xywh[:4]]
        ox = max(0, min(ox, width - 1))
        oy = max(0, min(oy, height - 1))
        ow = max(1, min(ow, width - ox))
        oh = max(1, min(oh, height - oy))
        ortrack_box = [ox, oy, ox + ow, oy + oh]
        orange = (0, 165, 255)
        cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), orange, 3)
        label = "Tracker"
        if (
            current_state_box_cxcywh is None
            and ortrack_confidence is not None
            and math.isfinite(float(ortrack_confidence))
        ):
            label += f" {float(ortrack_confidence):.2f}"
        label_y = max(oy - 6, 16)
        cv2.putText(canvas, label, (ox, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, orange, 2)
        cv2.drawMarker(
            canvas,
            (ox + ow // 2, oy + oh // 2),
            orange,
            cv2.MARKER_TILTED_CROSS,
            18,
            2,
        )

    origin = np.asarray([width * 0.5, height - 18.0], dtype=np.float32)
    body_position = np.zeros(3, dtype=np.float32)
    yaw = 0.0
    visualization_action_scale = 2.0
    points: List[List[int]] = [[int(origin[0]), int(origin[1])]]
    for action in np.asarray(action_sequence_physical, dtype=np.float32):
        if action.size < 4:
            continue
        if not np.isfinite(action[:4]).all():
            break
        cy, sy = math.cos(yaw), math.sin(yaw)
        forward, lateral, vertical = [float(v) for v in action[:3]]
        body_position[0] += (cy * forward - sy * lateral) * float(dt) * visualization_action_scale
        body_position[1] += (sy * forward + cy * lateral) * float(dt) * visualization_action_scale
        body_position[2] += vertical * float(dt) * visualization_action_scale
        yaw += math.radians(float(action[3]) * float(dt))
        px = origin[0] + body_position[1] * 22.0
        py = origin[1] - body_position[0] * 12.0 + body_position[2] * 22.0
        if not math.isfinite(float(px)) or not math.isfinite(float(py)):
            break
        points.append([int(round(px)), int(round(py))])

    visible_points = [p for p in points if 0 <= p[0] < width and 0 <= p[1] < height]
    for idx in range(1, len(points)):
        p0, p1 = points[idx - 1], points[idx]
        ratio = idx / max(len(points) - 1, 1)
        color = (255, int(220 * (1.0 - ratio)), int(80 + 120 * ratio))
        cv2.line(canvas, tuple(p0), tuple(p1), color, 8, cv2.LINE_AA)
    cv2.circle(canvas, tuple(points[0]), 9, (255, 120, 0), -1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return {
        "overlay": path.name,
        "target_crop": {
            "gt_box_xyxy": gt_box,
            "ortrack_box_xyxy": ortrack_box,
            "ortrack_confidence": ortrack_confidence,
            "search_crop_xyxy": search_box,
            "current_state_box_xyxy": current_state_box,
            "future_state_box_xyxy": (
                future_state_boxes[0] if future_state_boxes else None
            ),
            "future_state_boxes_xyxy": future_state_boxes,
        },
        "state_trajectory": {
            "pixel_points": state_points,
            "num_future_states": len(future_state_boxes),
            "sequence": "s0_to_future_states",
            "drawn_on_current_rgb": False,
        },
        "action_trajectory": {
            "pixel_points": points,
            "num_points": max(len(points) - 1, 0),
            "num_visible_points": max(len(visible_points) - 1, 0),
            "visualization_mode": "body_forward_to_rgb_length",
            "visualization_action_scale": visualization_action_scale,
            "overlay_origin": "bottom_center",
        },
    }


def save_predicted_video_frames(out_dir: Path, frames: torch.Tensor) -> List[str]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    arr = frames.detach().cpu().numpy() if torch.is_tensor(frames) else np.asarray(frames)
    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(f"expected batch size 1 for online predicted video, got shape {arr.shape}")
        arr = arr[0]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected predicted frames [T,H,W,3], got shape {arr.shape}")
    rel_names: List[str] = []
    for idx, frame in enumerate(arr):
        frame_u8 = np.asarray(frame, dtype=np.uint8)
        path = out_dir / f"pred_{idx:03d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))
        rel_names.append(path.name)
    return rel_names


def save_predicted_video_state_overlays(
    out_dir: Path,
    frame_paths: Sequence[Path],
    state_boxes_cxcywh: Sequence[Sequence[float]],
) -> List[Path]:
    """Draw s0...s8 on their corresponding decoded predicted RGB frames."""
    import cv2

    if len(frame_paths) != len(state_boxes_cxcywh):
        raise ValueError(
            "Predicted video/state length mismatch: "
            f"frames={len(frame_paths)} states={len(state_boxes_cxcywh)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    state_count = len(state_boxes_cxcywh)
    for index, (frame_path, box) in enumerate(
        zip(frame_paths, state_boxes_cxcywh)
    ):
        canvas = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if canvas is None:
            raise FileNotFoundError(f"Cannot read predicted RGB frame: {frame_path}")
        if len(box) < 4 or not all(
            math.isfinite(float(value)) for value in box[:4]
        ):
            raise ValueError(f"Invalid s{index} box: {box}")
        height, width = canvas.shape[:2]
        cx, cy, box_width, box_height = [float(value) for value in box[:4]]
        x1 = int(round((cx - 0.5 * box_width) * width))
        y1 = int(round((cy - 0.5 * box_height) * height))
        x2 = int(round((cx + 0.5 * box_width) * width))
        y2 = int(round((cy + 0.5 * box_height) * height))
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width - 1), min(y2, height - 1)
        if index == 0:
            color = (255, 0, 255)
            label = "s0 current"
        else:
            ratio = index / max(state_count - 1, 1)
            color = (
                int(round(220 * ratio)),
                int(round(210 - 120 * ratio)),
                255,
            )
            label = f"s{index}"
        if x2 >= x1 and y2 >= y1:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                canvas,
                label,
                (x1, max(y1 - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.drawMarker(
                canvas,
                (int(round(cx * width)), int(round(cy * height))),
                color,
                cv2.MARKER_CROSS,
                18,
                2,
            )
        output_path = out_dir / f"{Path(frame_path).stem}_s{index}.png"
        cv2.imwrite(str(output_path), canvas)
        outputs.append(output_path)
    return outputs


# -----------------------------
# Executor setup
# -----------------------------


def build_executor(module, args: argparse.Namespace, scene_id: str):
    target_asset_name = None if args.random_target_asset else args.target_asset_name
    jammer_asset_name = None if args.random_jammer_asset else args.jammer_asset_name
    executor = module.TrajectoryExecutor(
        scene_id=scene_id,
        sim_server_host=args.sim_server_host,
        sim_server_port=args.sim_server_port,
        gpu_id=args.gpu_id if args.sim_gpu_id is None else args.sim_gpu_id,
        scene_index=args.scene_index,
        uav_vehicle_name=args.uav_vehicle_name,
        target_object_name=args.target_object_name,
        target_asset_name=target_asset_name,
        target_object_scale=tuple(args.target_scale),
        camera_name=args.camera_name,
        auto_start_scene=True,
        deterministic_step_mode=True,
        jammer_enabled=(not args.disable_jammer),
        jammer_object_name=args.jammer_object_name,
        jammer_asset_name=jammer_asset_name,
        jammer_object_scale=tuple(args.jammer_scale),
    )
    executor.validate_camera_freshness = bool(args.validate_camera_freshness)
    executor.camera_max_vehicle_distance = float(args.camera_max_vehicle_distance)
    executor.camera_pose_tolerance_m = float(args.camera_pose_tolerance_m)
    executor.camera_orientation_tolerance_deg = float(
        args.camera_orientation_tolerance_deg
    )
    executor.save_depth = bool(args.save_depth)
    executor.require_depth = bool(args.save_depth)
    executor.use_external_camera = bool(args.use_external_camera)
    executor.disable_physics_pose_refresh = True
    return executor


def cleanup_executor(executor) -> None:
    try:
        if executor is not None:
            try:
                executor._cleanup_after_execution(skip_hover=True)
            except Exception:
                pass
            sim_tool = getattr(executor, "sim_client_tool", None)
            if sim_tool is not None:
                try:
                    sim_tool._closeConnection()
                except Exception:
                    pass
                try:
                    sim_tool._closeSocketConnection()
                except Exception:
                    pass
            try:
                executor.disconnect()
            except Exception:
                pass
    except Exception:
        pass


def _set_saved_assets_for_trajectory(executor, traj: OnlineTrajectory, args: argparse.Namespace) -> None:
    if args.reuse_saved_assets and traj.target_asset_name:
        executor.target_asset_name = str(traj.target_asset_name)
        executor._target_asset_name_explicitly_set = True
    elif args.random_target_asset:
        executor._target_asset_name_explicitly_set = False
    else:
        executor.target_asset_name = args.target_asset_name
        executor._target_asset_name_explicitly_set = True

    if args.reuse_saved_assets and traj.jammer_asset_names:
        executor._jammer_asset_name_explicitly_set = True
    elif args.random_jammer_asset:
        executor._jammer_asset_name_explicitly_set = False
    else:
        executor.jammer_asset_name = args.jammer_asset_name
        executor._jammer_asset_name_explicitly_set = True


def _prepare_objects(executor, traj: OnlineTrajectory, args: argparse.Namespace) -> None:
    selected_target = executor._prepare_target_object()
    executor._selected_target_asset_name = selected_target

    if not traj.jammer_trajs_airsim:
        executor._all_jammer_trajectories_airsim = None
        executor._primary_jammer_id = None
        return

    executor._all_jammer_trajectories_airsim = traj.jammer_trajs_airsim
    sorted_ids = sorted(traj.jammer_trajs_airsim.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    executor._primary_jammer_id = str(sorted_ids[0])

    # Prefer saved assets if they were written by the dataset collection code.
    if args.reuse_saved_assets and traj.jammer_asset_names:
        executor._jammer_object_names_by_id = {}
        executor._jammer_asset_names_by_id = {}
        unique_suffix = int(time.time() * 1000) % 100000
        for idx, did in enumerate(sorted_ids):
            asset = traj.jammer_asset_names.get(str(did), args.jammer_asset_name)
            object_name = f"{asset}_{unique_suffix}_{random.randint(1000, 9999)}_j{did}"
            executor._jammer_object_names_by_id[str(did)] = object_name
            executor._jammer_asset_names_by_id[str(did)] = asset
            if idx == 0:
                executor.jammer_object_name = object_name
                executor.jammer_asset_name = asset
                executor._selected_jammer_asset_name = asset
    else:
        executor._prepare_all_jammer_objects(traj.jammer_trajs_airsim)


def _save_trajectory_3d_plot(out_dir: Path, steps: List[Dict[str, Any]]) -> None:
    """Save the generic UAV/target/jammer rollout visualization."""
    if not steps:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] skip 3D trajectory plot: matplotlib unavailable ({exc})")
        return

    def collect_xyz(key: str) -> np.ndarray:
        points: List[List[float]] = []
        for step in steps:
            value = step.get(key)
            if isinstance(value, dict) and all(axis in value for axis in ("x", "y", "z")):
                points.append(
                    [float(value["x"]), float(value["y"]), float(value["z"])]
                )
        if not points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)

    uav_xyz = collect_xyz("uav_position_after")
    target_xyz = collect_xyz("target_position")
    jammer_series: Dict[str, List[List[float]]] = {}
    for step in steps:
        jammers = step.get("jammers")
        if not isinstance(jammers, dict):
            continue
        for jammer_id, position in jammers.items():
            if isinstance(position, dict) and all(
                axis in position for axis in ("x", "y", "z")
            ):
                jammer_series.setdefault(str(jammer_id), []).append(
                    [
                        float(position["x"]),
                        float(position["y"]),
                        float(position["z"]),
                    ]
                )

    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(111, projection="3d")
    if len(uav_xyz) > 0:
        axes.plot(
            uav_xyz[:, 0], uav_xyz[:, 1], uav_xyz[:, 2],
            color="tab:blue", linewidth=2.0, label="uav",
        )
        axes.scatter(
            uav_xyz[0, 0], uav_xyz[0, 1], uav_xyz[0, 2],
            color="tab:blue", marker="o", s=25, label="uav_start",
        )
        axes.scatter(
            uav_xyz[-1, 0], uav_xyz[-1, 1], uav_xyz[-1, 2],
            color="tab:blue", marker="x", s=35, label="uav_end",
        )
    if len(target_xyz) > 0:
        axes.plot(
            target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2],
            color="tab:red", linestyle="--", linewidth=2.0, label="target",
        )
    for jammer_id, points in sorted(jammer_series.items()):
        array = np.asarray(points, dtype=np.float32)
        if len(array) > 0:
            axes.plot(
                array[:, 0], array[:, 1], array[:, 2],
                linewidth=1.2, alpha=0.85, label=f"jammer_{jammer_id}",
            )

    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_zlabel("z (m, up)")
    axes.set_title("Online Eval 3D Trajectory")
    handles, _ = axes.get_legend_handles_labels()
    try:
        if not handles:
            raise ValueError("no labeled trajectories")
        axes.legend(loc="best", fontsize=8)
    except (AttributeError, ValueError):
        pass
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / "trajectory_3d.png", dpi=180)
    plt.close(figure)


# -----------------------------
# Main online rollout
# -----------------------------


@torch.no_grad()
def run_online_trajectory(
    model: TeacherWorldModelDiT,
    cfg: ModelConfig,
    tokenizer,
    transform,
    executor,
    traj: OnlineTrajectory,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    out_dir = Path(args.output_dir) / traj.scene_id / traj.trajectory_name
    rgb_out_dir = out_dir / "rgb"
    attention_map_dir = out_dir / "last_transformer_attention_maps"
    tracker_attention_summary_dir = out_dir / "action_tracker_attention_summary"
    attention_comparison_dir = out_dir / "attention_tracker_comparisons"
    predicted_video_dir = out_dir / "predicted_video"
    ortrack_dir = out_dir / "ortrack"
    action_overlay_dir = out_dir / args.target_crop_action_overlay_output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_visual_assets = _visualization_enabled_for_trajectory(args, traj)
    if args.save_rgb and save_visual_assets:
        rgb_out_dir.mkdir(parents=True, exist_ok=True)
    save_transformer_attention_maps_enabled = bool(
        save_visual_assets
        and args.save_transformer_attention_maps
        and _attention_enabled_for_key(args, traj.scene_id, traj.trajectory_name)
    )
    save_predicted_video_enabled = bool(
        args.save_predicted_video
        and _predicted_video_enabled_for_key(args, traj.scene_id, traj.trajectory_name)
    )
    save_action_overlays_enabled = bool(save_visual_assets and args.save_target_crop_action_overlays)
    if save_transformer_attention_maps_enabled:
        attention_map_dir.mkdir(parents=True, exist_ok=True)
        tracker_attention_summary_dir.mkdir(parents=True, exist_ok=True)
    if save_predicted_video_enabled:
        predicted_video_dir.mkdir(parents=True, exist_ok=True)
    if save_action_overlays_enabled:
        action_overlay_dir.mkdir(parents=True, exist_ok=True)
    use_ortrack_guidance = bool(getattr(cfg, "use_fastwam_attention_bias", False)) or bool(
        getattr(cfg, "use_gt_center_attention_bias", False)
    )
    heatmap_source = str(getattr(cfg, "fastwam_heatmap_source", "none"))
    tracker_runtime_enabled = _tracker_runtime_required(cfg, args)
    model_driven_tracker_search = bool(getattr(cfg, "tracker_model_driven_search", False))
    if save_visual_assets and args.save_ortrack_maps and tracker_runtime_enabled:
        ortrack_dir.mkdir(parents=True, exist_ok=True)

    ortrack = None
    if tracker_runtime_enabled:
        from tracking.runtime import SquareTracker

        ortrack = SquareTracker(
            Path(args.tracker_checkpoint),
            args.device,
            feature_grid_size=int(getattr(cfg, "tracker_feature_grid_size", 7)),
        )
    ortrack_initialized = False
    model_driven_cropper = (
        ModelDrivenTrackerCropper(
            template_size=int(cfg.tracker_template_size),
            search_size=int(cfg.tracker_search_size),
        )
        if model_driven_tracker_search
        else None
    )
    last_confident_action_sequence_norm: Optional[np.ndarray] = None
    last_confident_action_sequence_physical: Optional[np.ndarray] = None
    last_confident_action_sequence_index = 0

    _set_saved_assets_for_trajectory(executor, traj, args)
    _prepare_objects(executor, traj, args)

    # The external camera owns the virtual UAV pose, so physical multirotor
    # initialization is both unnecessary and harmful after a previous rollout.
    if args.camera_only_virtual_uav:
        executor.initialize_scene_objects_only(
            traj.target_traj_airsim,
            jammer_trajs_by_id=traj.jammer_trajs_airsim if traj.jammer_trajs_airsim else None,
        )
    else:
        executor._initialize_simulation(
            np.asarray([traj.uav_start_airsim], dtype=np.float32),
            traj.target_traj_airsim,
            jammer_trajs_by_id=traj.jammer_trajs_airsim if traj.jammer_trajs_airsim else None,
        )
    # Scene initialization establishes the AirSim client used by this command.
    executor.configure_camera_rendering(float(args.camera_render_max_fps))
    executor._reset_collision_state()
    executor._last_camera_timestamp = None
    executor.camera_stale_rejections = 0
    executor.camera_distance_rejections = 0
    executor.camera_pose_rejections = 0
    init_yaw = math.atan2(
        float(traj.target_traj_airsim[0][1] - traj.uav_start_airsim[1]),
        float(traj.target_traj_airsim[0][0] - traj.uav_start_airsim[0]),
    )
    init_q_xyzw = R.from_euler("xyz", [0.0, 0.0, init_yaw], degrees=False).as_quat()
    virtual_uav_state = {
        "position": np.asarray(traj.uav_start_airsim, dtype=np.float32).copy(),
        "orientation": np.asarray(
            [init_q_xyzw[3], init_q_xyzw[0], init_q_xyzw[1], init_q_xyzw[2]], dtype=np.float64
        ),
        "has_collided": False,
        "collision_time_stamp": None,
    }
    if args.camera_only_virtual_uav:
        executor.camera_pose_state_override = virtual_uav_state
        executor._safe_sim_pause(True)
        if not getattr(executor, "_physical_uav_parked_for_virtual_camera", False):
            # Park once per Executor so the real Drone_1 cannot appear in the
            # external-camera image. Later trajectories never reposition it.
            park_position = np.asarray(traj.uav_start_airsim, dtype=np.float32).copy()
            park_position[:2] += 10000.0
            try:
                set_vehicle_pose_static(
                    executor,
                    park_position,
                    _yaw_to_airsim_quat(init_yaw),
                    retries=1,
                    tol_xy=1.0,
                    tol_z=1.0,
                )
                executor._physical_uav_parked_for_virtual_camera = True
            except Exception as exc:
                print(f"[warn] failed to park physical UAV outside the scene: {exc}", flush=True)
    else:
        try:
            executor.client.enableApiControl(True, vehicle_name=executor.uav_vehicle_name)
            executor.client.armDisarm(True, vehicle_name=executor.uav_vehicle_name)
        except Exception:
            pass
        executor._ensure_uav_flying_state()
        set_vehicle_pose_static(
            executor,
            traj.uav_start_airsim,
            _yaw_to_airsim_quat(init_yaw),
            retries=3,
            tol_xy=0.8,
            tol_z=0.8,
        )

    num_steps = traj.num_frames if args.max_steps <= 0 else min(traj.num_frames, args.max_steps)
    rssm_state = None
    prev_action = torch.zeros(1, cfg.action_dim, device=device, dtype=torch.float32)
    prev_done = torch.zeros(1, device=device, dtype=torch.float32)

    steps: List[Dict[str, Any]] = []
    distances: List[float] = []
    success = False
    success_step: Optional[int] = None
    collision = False
    visible_count = 0
    action_abs_err: List[float] = []
    action_mse: List[float] = []
    profile_records: List[Dict[str, float]] = []
    prev_uav_after_pos: Optional[np.ndarray] = None
    prev_uav_after_state: Optional[Dict[str, Any]] = None
    target_history_boxes: List[np.ndarray] = []
    target_history_confidences: List[float] = []

    iterator: Iterable[int] = range(num_steps)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"online {traj.scene_id}/{traj.trajectory_name}", unit="step", dynamic_ncols=True)

    for t in iterator:
        profile_enabled = bool(getattr(args, "profile_step_time", False))
        step_profile: Dict[str, float] = {}
        profile_step_start = time.perf_counter()
        target_t = traj.target_traj_airsim[t]
        executor.move_target_object(target_t)
        jammer_positions_now: Dict[str, np.ndarray] = {}
        if traj.jammer_trajs_airsim:
            for did, series in traj.jammer_trajs_airsim.items():
                if t >= len(series):
                    continue
                obj_name = executor._jammer_object_names_by_id.get(str(did), executor.jammer_object_name)
                asset_name = executor._jammer_asset_names_by_id.get(str(did), executor.jammer_asset_name)
                moved = executor.move_named_object(
                    obj_name,
                    asset_name,
                    executor.jammer_object_scale,
                    series[t],
                )
                if moved:
                    # The steady-state mover already checks simSetObjectPose and
                    # periodically verifies the pose. Avoid a duplicate per-frame RPC.
                    jammer_positions_now[str(did)] = np.asarray(
                        series[t], dtype=np.float32
                    ).copy()
                else:
                    # Preserve actual-state reporting if strict recovery failed.
                    try:
                        pos = executor.get_named_object_position(obj_name)
                        if pos is not None:
                            jammer_positions_now[str(did)] = pos
                    except Exception:
                        pass
        if (
            not args.camera_only_virtual_uav
            and args.control_mode == "pose"
            and args.hold_uav_pose_during_scene_step
            and prev_uav_after_state is not None
        ):
            prev_pos = np.asarray(prev_uav_after_state["position"], dtype=np.float32)
            prev_quat = _wxyz_quat_to_airsim_quat(prev_uav_after_state["orientation"])
            set_vehicle_pose_static(
                executor,
                prev_pos,
                prev_quat,
                retries=1,
                tol_xy=0.8,
                tol_z=0.8,
            )

        profile_camera_pose_started = time.perf_counter()
        if args.use_external_camera:
            camera_state = (
                virtual_uav_state
                if args.camera_only_virtual_uav
                else (prev_uav_after_state if prev_uav_after_state is not None else executor.get_uav_state())
            )
            executor.set_external_camera_pose_from_state(camera_state)
        profile_camera_pose_done = time.perf_counter()

        profile_camera_render_started = time.perf_counter()
        if args.camera_capture_mode == "legacy_step" and args.camera_render_frames > 0:
            if args.camera_only_virtual_uav:
                executor._safe_continue_for_frames(args.camera_render_frames)
            else:
                executor.render_frames_holding_vehicle_pose(args.camera_render_frames)
        profile_camera_render_done = time.perf_counter()

        uav_state_before = (
            {
                "position": np.asarray(virtual_uav_state["position"], dtype=np.float32).copy(),
                "orientation": np.asarray(virtual_uav_state["orientation"], dtype=np.float64).copy(),
                "has_collided": False,
                "collision_time_stamp": None,
            }
            if args.camera_only_virtual_uav
            else executor.get_uav_state()
        )
        uav_before_pos = np.asarray(uav_state_before["position"], dtype=np.float32)
        pose_jump_dataset = None
        pose_jump_norm = None
        pose_jump_z_abs = None
        large_pose_jump = False
        if prev_uav_after_pos is not None:
            prev_dataset = np.asarray(
                [prev_uav_after_pos[0], prev_uav_after_pos[1], -prev_uav_after_pos[2]],
                dtype=np.float32,
            )
            curr_dataset = np.asarray(
                [uav_before_pos[0], uav_before_pos[1], -uav_before_pos[2]],
                dtype=np.float32,
            )
            pose_jump_dataset = curr_dataset - prev_dataset
            pose_jump_norm = float(np.linalg.norm(pose_jump_dataset))
            pose_jump_z_abs = float(abs(pose_jump_dataset[2]))
            large_pose_jump = bool(
                pose_jump_norm > float(args.pose_jump_warn_threshold)
                or pose_jump_z_abs > float(args.pose_jump_z_warn_threshold)
            )

        collision_info_before_action = None
        collision_before_action = False
        if not args.camera_only_virtual_uav:
            collision_info_before_action = executor.client.simGetCollisionInfo(
                vehicle_name=executor.uav_vehicle_name
            )
            collision_before_action = (
                bool(collision_info_before_action.has_collided)
                if collision_info_before_action is not None
                else False
            )

        target_now = executor.get_object_position()
        if target_now is None:
            target_now = target_t
        target_now = np.asarray(target_now, dtype=np.float32)

        rel_body = compute_target_relative_body(executor, uav_state_before, target_now)
        next_rel_body = None
        if t + 1 < num_steps:
            next_rel_body = compute_target_relative_body(executor, uav_state_before, traj.target_traj_airsim[t + 1])

        if getattr(args, "force_live_instruction", False):
            instruction = instruction_from_relative(rel_body, next_rel_body)
        elif traj.saved_instructions and t < len(traj.saved_instructions):
            instruction = traj.saved_instructions[t]
        elif traj.saved_instructions and len(traj.saved_instructions) == 1:
            instruction = traj.saved_instructions[0]
        else:
            instruction = instruction_from_relative(rel_body, next_rel_body)

        profile_scene_done = time.perf_counter()
        if args.camera_only_virtual_uav:
            executor.camera_pose_state_override = uav_state_before
        if args.camera_capture_mode == "fresh_frame":
            rgb_img, _ = executor.get_fresh_camera_images()
            if (
                not args.camera_only_virtual_uav
                and args.control_mode == "pose"
                and args.hold_uav_pose_during_scene_step
            ):
                set_vehicle_pose_static(
                    executor,
                    np.asarray(uav_state_before["position"], dtype=np.float32),
                    _wxyz_quat_to_airsim_quat(uav_state_before["orientation"]),
                    retries=1,
                    tol_xy=0.3,
                    tol_z=0.3,
                )
        else:
            rgb_img, _ = executor.get_camera_images()
        if rgb_img is None:
            raise RuntimeError(f"failed to capture RGB at step {t}")
        profile_camera_done = time.perf_counter()
        camera_capture_profile = getattr(executor, "_last_camera_profile_ms", {})
        if args.save_rgb and save_visual_assets:
            save_rgb(rgb_out_dir / f"frame_{t:05d}.png", rgb_img)
        profile_rgb_done = time.perf_counter()

        rgb_array = np.asarray(rgb_img, dtype=np.uint8)
        ortrack_result = None
        if use_ortrack_guidance and heatmap_source == "gt":
            try:
                gt_bbox = _bbox_from_target_center(
                    rel_body,
                    rgb_array.shape[:2],
                    float(args.fov_deg),
                    tuple(float(v) for v in args.ortrack_camera_offset_body),
                    float(args.ortrack_init_box_frac),
                )
                gt_heatmap = _bbox_heatmap(gt_bbox, rgb_array.shape[:2])
                gt_confidence = 1.0
            except RuntimeError:
                gt_bbox = None
                gt_heatmap = np.full(rgb_array.shape[:2], 1.0 / float(np.prod(rgb_array.shape[:2])), dtype=np.float32)
                gt_confidence = 0.0
            ortrack_result = {
                "bbox": gt_bbox,
                "confidence": gt_confidence,
                "heatmap": gt_heatmap,
            }
        elif model_driven_cropper is not None and not ortrack_initialized:
            init_bbox = _model_driven_initial_bbox(
                rel_body,
                rgb_array.shape[:2],
                float(args.fov_deg),
                tuple(float(v) for v in args.ortrack_camera_offset_body),
                float(args.ortrack_init_box_frac),
            )
            model_driven_cropper.initialize(rgb_array, init_bbox)
            ortrack_result = model_driven_cropper.current(rgb_array, torch.device(device))
            ortrack_initialized = True
        elif model_driven_cropper is not None:
            ortrack_result = model_driven_cropper.current(rgb_array, torch.device(device))
        elif tracker_runtime_enabled and not ortrack_initialized:
            init_bbox = _bbox_from_target_center(
                rel_body,
                rgb_array.shape[:2],
                float(args.fov_deg),
                tuple(float(v) for v in args.ortrack_camera_offset_body),
                float(args.ortrack_init_box_frac),
            )
            init_heatmap = _bbox_heatmap(init_bbox, rgb_array.shape[:2])
            assert ortrack is not None
            ortrack.initialize(rgb_array, init_bbox)
            if str(getattr(cfg, "tracker_condition_mode", "center_features")) != "none" or str(
                getattr(cfg, "tracker_mot_integration", "none")
            ) in {"frozen_deit_tracker_local_feature", "mot_tracker_finetune_local_feature"}:
                first_result = ortrack.track(rgb_array)
                ortrack.initialize(rgb_array, init_bbox)
                ortrack_result = dict(first_result)
                ortrack_result["bbox"] = list(init_bbox)
            else:
                ortrack_result = {
                    "bbox": list(ortrack.state or init_bbox),
                    "confidence": 1.0,
                    "heatmap": init_heatmap,
                }
            ortrack_initialized = True
        elif tracker_runtime_enabled:
            assert ortrack is not None
            ortrack_result = ortrack.track(rgb_array)
        if save_visual_assets and args.save_ortrack_maps and ortrack_result is not None:
            heatmap = np.asarray(ortrack_result["heatmap"], dtype=np.float32)
            heatmap = heatmap / max(float(heatmap.max()), 1.0e-8)
            Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0)).save(
                ortrack_dir / f"frame_{t:05d}.png"
            )

        dataset_expert_action = traj.expert_action_physical[t] if t < len(traj.expert_action_physical) else None
        target_next_for_expert = traj.target_traj_airsim[t + 1] if t + 1 < len(traj.target_traj_airsim) else target_now
        expert_action = (
            np.asarray(dataset_expert_action, dtype=np.float32)
            if args.replay_expert_action
            else compute_live_expert_action(
                executor,
                uav_state_before,
                target_now,
                target_next_for_expert,
                max_speed_norm=cfg.max_speed_norm,
                max_yaw_rate=cfg.max_yaw_rate,
            )
        )
        expert_action_source = "dataset" if args.replay_expert_action else "live_planner"
        expert_action_norm = None
        action_source = "model"
        pred = None
        predicted_current_tracker_center = None
        predicted_future_tracker_center = None
        predicted_current_tracker_box = None
        predicted_future_tracker_box = None
        predicted_future_tracker_boxes = None
        attention_map_relpath = None
        attention_comparison_relpath = None
        predicted_video_relpaths = None
        predicted_video_state_overlay_relpaths = None
        action_sequence_norm = None
        action_sequence_physical = None
        capture_value_diagnostics = None
        action_overlay_metadata = None
        ortrack_bbox = (
            None
            if ortrack_result is None or ortrack_result.get("bbox") is None
            else [float(v) for v in ortrack_result["bbox"]]
        )
        ortrack_confidence = None if ortrack_result is None else float(ortrack_result["confidence"])
        tracker_center_t = None
        if _tracker_center_required(cfg):
            bbox = None if ortrack_result is None else ortrack_result.get("bbox")
            if bbox is None:
                center_xy = [float("nan"), float("nan")]
            else:
                x, y, width, height = (float(v) for v in bbox)
                image_h, image_w = rgb_array.shape[:2]
                center_xy = [
                    float(np.clip((x + 0.5 * width) / max(image_w, 1), 0.0, 1.0)),
                    float(np.clip((y + 0.5 * height) / max(image_h, 1), 0.0, 1.0)),
                ]
            tracker_center_t = torch.tensor(
                [center_xy], dtype=torch.float32, device=device
            )
        tracker_features_t = None
        if _tracker_features_required(cfg) and not model_driven_tracker_search:
            if ortrack_result is None or ortrack_result.get("feature_tokens") is None:
                raise RuntimeError("Tracker feature integration requires online feature tokens.")
            tracker_features_t = torch.from_numpy(
                np.asarray(ortrack_result["feature_tokens"], dtype=np.float32)
            ).unsqueeze(0).to(device)
        tracker_bbox_t = None
        if _tracker_bbox_required(cfg):
            if ortrack_bbox is None:
                raise RuntimeError("Tracker bbox condition requires an online bbox.")
            x, y, width, height = ortrack_bbox
            image_h, image_w = rgb_array.shape[:2]
            tracker_bbox_t = torch.tensor(
                [[
                    (x + 0.5 * width) / max(image_w, 1),
                    (y + 0.5 * height) / max(image_h, 1),
                    width / max(image_w, 1),
                    height / max(image_h, 1),
                ]],
                dtype=torch.float32,
                device=device,
            ).clamp_(0.0, 1.0)
        tracker_response_t = None
        if _tracker_response_required(cfg):
            if ortrack_result is None or ortrack_result.get("response") is None:
                raise RuntimeError("Tracker response condition requires an online response map.")
            response = _normalized_tracker_response(
                ortrack_result["response"],
                int(getattr(cfg, "tracker_response_grid_size", 7)),
            )
            tracker_response_t = torch.from_numpy(response).unsqueeze(0).to(device)
        tracker_search_geometry_t = None
        tracker_image_size_t = None
        if _tracker_geometry_required(cfg):
            geometry = (
                None
                if ortrack_result is None
                else ortrack_result.get("search_crop_xy_size")
            )
            if geometry is None or len(geometry) != 3:
                raise RuntimeError(
                    "Tracker local-feature fusion requires online search crop geometry."
                )
            image_h, image_w = rgb_array.shape[:2]
            tracker_search_geometry_t = torch.tensor(
                [[float(value) for value in geometry]],
                dtype=torch.float32,
                device=device,
            )
            tracker_image_size_t = torch.tensor(
                [[float(image_h), float(image_w)]],
                dtype=torch.float32,
                device=device,
            )
        tracker_template_t = tracker_search_t = None
        if str(getattr(cfg, "tracker_mot_integration", "none")) == "mot_tracker_finetune_local_feature":
            if ortrack_result is None:
                raise RuntimeError("Tracker fine-tuning integration requires online Tracker crops.")
            tracker_template_t = ortrack_result.get("tracker_template")
            tracker_search_t = ortrack_result.get("tracker_search")
            if tracker_template_t is None or tracker_search_t is None:
                raise RuntimeError("Online Tracker runtime did not provide template/search crops.")
            tracker_template_t = tracker_template_t.to(device=device, dtype=torch.float32)
            tracker_search_t = tracker_search_t.to(device=device, dtype=torch.float32)
        target_box_history_t = target_box_history_valid_t = None
        if bool(getattr(cfg, "use_historical_target_memory", False)):
            target_box_history_t, target_box_history_valid_t = (
                make_online_target_box_history(
                    [torch.from_numpy(value) for value in target_history_boxes],
                    target_history_confidences,
                    previous_length=int(getattr(cfg, "target_history_length", 8)) - 1,
                    device=torch.device(device),
                )
            )
        tracker_detection_missed = bool(
            args.reuse_last_confident_action_sequence
            and ortrack_confidence is not None
            and ortrack_confidence < float(args.tracker_detection_confidence_threshold)
        )
        reused_action_sequence_index = None
        profile_planner_done = time.perf_counter()

        if profile_enabled and torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        profile_model_start = time.perf_counter()
        if args.replay_expert_action:
            if expert_action is None:
                break
            expert_action_norm = physical_action_to_norm(expert_action, cfg.max_vel, cfg.max_yaw_rate)
            action_norm = expert_action_norm.astype(np.float32)
            action_physical = np.asarray(expert_action, dtype=np.float32)
            action_source = "expert"
        elif tracker_detection_missed and last_confident_action_sequence_norm is not None:
            sequence_len = int(last_confident_action_sequence_norm.shape[0])
            if args.tracker_fallback_action_mode == "first_action":
                reused_action_sequence_index = 0
            else:
                reused_action_sequence_index = min(last_confident_action_sequence_index, sequence_len - 1)
            action_norm = last_confident_action_sequence_norm[reused_action_sequence_index].copy()
            action_physical = last_confident_action_sequence_physical[reused_action_sequence_index].copy()
            action_sequence_norm = last_confident_action_sequence_norm[reused_action_sequence_index:].copy()
            action_sequence_physical = last_confident_action_sequence_physical[reused_action_sequence_index:].copy()
            if args.tracker_fallback_action_mode == "remaining_sequence":
                last_confident_action_sequence_index = min(
                    last_confident_action_sequence_index + 1,
                    sequence_len - 1,
                )
            action_source = "last_confident_action_sequence"
        else:
            image_t = rgb_to_model_tensor(rgb_img, transform, device)
            if cfg.use_wan22_encoders:
                text_tokens = torch.zeros(1, 1, dtype=torch.long, device=device)
                attention_mask = torch.ones_like(text_tokens)
            else:
                text_tokens, attention_mask = tokenize_instruction(tokenizer, instruction, cfg.text_context_length, device)
            target_relative_np = rel_body / max(float(args.target_relative_scale), 1e-6)
            target_relative_t = torch.from_numpy(target_relative_np.astype(np.float32)).view(1, -1).to(device)

            pred, rssm_state = model.act(
                image=image_t,
                text_tokens=text_tokens,
                target_relative=target_relative_t,
                prev_action=prev_action,
                rssm_state=rssm_state,
                attention_mask=attention_mask,
                prev_done=prev_done,
                deterministic=args.deterministic_action,
                num_steps=args.sampling_steps,
                instruction=instruction,
                save_transformer_attention=save_transformer_attention_maps_enabled,
                save_predicted_video=save_predicted_video_enabled,
                predicted_video_latent_frames=args.predicted_video_latent_frames,
                guidance_heatmap=(
                    None
                    if ortrack_result is None
                    else torch.from_numpy(np.asarray(ortrack_result["heatmap"], dtype=np.float32))
                    .unsqueeze(0)
                    .to(device)
                ),
                guidance_confidence=(
                    torch.zeros((1, 1), dtype=torch.float32, device=device)
                    if ortrack_result is None or ortrack_result.get("bbox") is None
                    else torch.tensor([[float(ortrack_result["confidence"])]], device=device)
                ),
                tracker_center=tracker_center_t,
                tracker_features=tracker_features_t,
                tracker_bbox=tracker_bbox_t,
                tracker_response=tracker_response_t,
                tracker_search_geometry=tracker_search_geometry_t,
                tracker_image_size=tracker_image_size_t,
                tracker_template=tracker_template_t,
                tracker_search=tracker_search_t,
                target_box_history=target_box_history_t,
                target_box_history_valid=target_box_history_valid_t,
            )
            state_centers = pred.get("target_state_centers")
            state_boxes = pred.get("target_state_boxes")
            if state_centers is not None:
                predicted_current_tracker_center = _current_state_center_for_next_search(
                    state_centers
                )
            if state_boxes is not None:
                predicted_current_tracker_box = (
                    state_boxes[0, 0].detach().float().cpu().numpy().astype(np.float32)
                )
            if (
                bool(getattr(cfg, "use_future_state_dit", False))
                and state_centers is not None
                and state_centers.size(1) > 1
            ):
                predicted_future_tracker_center = (
                    state_centers[0, 1].detach().float().cpu().numpy().astype(np.float32)
                )
                if state_boxes is not None:
                    predicted_future_tracker_boxes = (
                        state_boxes[0, 1:].detach().float().cpu().numpy().astype(np.float32)
                    )
                    predicted_future_tracker_box = predicted_future_tracker_boxes[0]
            if model_driven_cropper is not None:
                if predicted_current_tracker_center is None:
                    raise RuntimeError(
                        "Model-driven Tracker search requires current Target state predictions."
                    )
                # Frame t+1 is searched around the observed state s0(t). Future
                # states remain predictions and must not overwrite tracker memory.
                model_driven_cropper.advance(
                    predicted_current_tracker_center, rgb_array.shape[:2]
                )
            if bool(getattr(cfg, "use_historical_target_memory", False)):
                if predicted_current_tracker_box is None:
                    raise RuntimeError("Target-history baseline did not return current Tracker b0.")
                target_history_boxes.append(predicted_current_tracker_box.copy())
                target_history_confidences.append(
                    float(pred["tracker_confidence"].detach().view(-1)[0].cpu().item())
                    if pred.get("tracker_confidence") is not None
                    else 0.0
                )
                keep = max(int(getattr(cfg, "target_history_length", 8)) - 1, 0)
                target_history_boxes[:] = target_history_boxes[-keep:]
                target_history_confidences[:] = target_history_confidences[-keep:]
            if save_transformer_attention_maps_enabled and "last_transformer_attention_map" in pred:
                attention_map_path = attention_map_dir / f"frame_{t:05d}"
                save_attention_map_overlay(
                    attention_map_path, rgb_img, pred["last_transformer_attention_query0_map"]
                )
                attention_map_relpath = str(attention_map_path.with_suffix(".png").relative_to(out_dir))
                tracker_attention = pred.get("last_tracker_cross_attention")
                if tracker_attention is not None and tracker_search_geometry_t is not None:
                    # One interpretable map per frame: mean over all Action queries and heads.
                    tracker_mean = tracker_attention[0].float().mean(dim=0)
                    token_count = int(tracker_mean.size(-1))
                    # Tracker memory can be either N spatial tokens or N spatial
                    # tokens plus one box token. Infer the latter from a square
                    # spatial prefix instead of assuming it is always present.
                    candidate_spatial_count = token_count - 1
                    candidate_grid_size = int(round(candidate_spatial_count ** 0.5))
                    has_box_token = (
                        candidate_spatial_count > 0
                        and candidate_grid_size * candidate_grid_size == candidate_spatial_count
                    )
                    spatial_count = candidate_spatial_count if has_box_token else token_count
                    grid_size = int(round(spatial_count ** 0.5))
                    if grid_size * grid_size == spatial_count:
                        spatial_map = tracker_mean[:, :spatial_count].mean(dim=0).reshape(grid_size, grid_size)
                        if has_box_token:
                            bbox_weight = float(tracker_mean[:, -1].mean().item())
                            label = f"Action->Tracker mean | bbox token: {bbox_weight:.4f}"
                        else:
                            label = "Action->Tracker mean | spatial tokens only"
                        save_tracker_attention_overlay(
                            tracker_attention_summary_dir / f"frame_{t:05d}",
                            rgb_img,
                            spatial_map,
                            tracker_search_geometry_t[0],
                            label=label,
                        )
                    else:
                        print(
                            "[attention] skipping Tracker overlay: "
                            f"token count {token_count} has no square spatial layout.",
                            flush=True,
                        )
                query0_map = pred.get("last_transformer_attention_query0_map")
                if query0_map is not None:
                    if (
                        args.save_attention_tracker_comparisons
                        and tracker_runtime_enabled
                        and ortrack_result is not None
                    ):
                        from eval.compare_tracker_attention import save_comparison_panel

                        response = ortrack_result.get("response", ortrack_result.get("heatmap"))
                        if response is not None:
                            comparison_path = (
                                attention_comparison_dir
                                / f"frame_{t:05d}_attention_tracker_comparison.png"
                            )
                            save_comparison_panel(
                                comparison_path,
                                rgb_array,
                                query0_map[0].detach().float().cpu().numpy(),
                                pred["last_transformer_attention_all_queries_map"][0]
                                .detach().float().cpu().numpy(),
                                np.asarray(response, dtype=np.float32),
                                ortrack_bbox,
                            )
                            attention_comparison_relpath = str(comparison_path.relative_to(out_dir))
            if save_predicted_video_enabled and "predicted_video_latents" in pred:
                decoded_video = model.image_encoder.decode_video_latents(pred["predicted_video_latents"])
                pred_frame_dir = predicted_video_dir / f"frame_{t:05d}"
                pred_frame_names = save_predicted_video_frames(pred_frame_dir, decoded_video)
                predicted_video_relpaths = [
                    str((pred_frame_dir / name).relative_to(out_dir)) for name in pred_frame_names
                ]
                if (
                    predicted_current_tracker_box is not None
                    and predicted_future_tracker_boxes is not None
                ):
                    state_boxes = np.concatenate(
                        [
                            predicted_current_tracker_box[None],
                            predicted_future_tracker_boxes,
                        ],
                        axis=0,
                    )
                    predicted_frame_paths = [
                        pred_frame_dir / name for name in pred_frame_names
                    ]
                    state_overlay_paths = save_predicted_video_state_overlays(
                        pred_frame_dir / "state_overlays",
                        predicted_frame_paths,
                        state_boxes,
                    )
                    predicted_video_state_overlay_relpaths = [
                        str(path.relative_to(out_dir)) for path in state_overlay_paths
                    ]
            action_norm = pred["action_norm"].detach().float().view(-1).cpu().numpy().astype(np.float32)
            action_physical = pred["action_physical"].detach().float().view(-1).cpu().numpy().astype(np.float32)
            if "action_sequence_norm" in pred:
                action_sequence_norm = (
                    pred["action_sequence_norm"].detach().float().squeeze(0).cpu().numpy().astype(np.float32)
                )
                action_sequence_physical = norm_action_to_physical(
                    action_sequence_norm,
                    max_vel=cfg.max_vel,
                    max_yaw_rate=cfg.max_yaw_rate,
                    max_speed_norm=cfg.max_speed_norm,
                )
                if (
                    args.reuse_last_confident_action_sequence
                    and ortrack_confidence is not None
                    and ortrack_confidence >= float(args.tracker_detection_confidence_threshold)
                    and len(action_sequence_norm) > 0
                ):
                    last_confident_action_sequence_norm = action_sequence_norm.copy()
                    last_confident_action_sequence_physical = action_sequence_physical.copy()
                    last_confident_action_sequence_index = min(1, len(action_sequence_norm) - 1)
            if "capture_value_selected_index" in pred:
                capture_value_diagnostics = {
                    "selected_index": int(
                        pred["capture_value_selected_index"]
                        .detach()
                        .view(-1)[0]
                        .cpu()
                    ),
                    "candidate_scores": pred["capture_value_scores"]
                    .detach()
                    .float()
                    .squeeze(0)
                    .cpu()
                    .tolist(),
                    "candidate_action_sequences_norm": pred[
                        "capture_value_candidates"
                    ]
                    .detach()
                    .float()
                    .squeeze(0)
                    .cpu()
                    .tolist(),
                }
                scalar_fields = {
                    "raw_selected_index": "capture_value_raw_selected_index",
                    "score_advantage": "capture_value_score_advantage",
                    "used_fallback": "capture_value_used_fallback",
                    "capture_probability": "capture_value_selected_capture_probability",
                    "predicted_final_distance": "capture_value_selected_final_distance",
                    "visibility_probability": "capture_value_selected_visibility",
                    "selected_final_box_size": "capture_value_selected_final_box_size",
                    "observed_center_velocity": "capture_value_observed_center_velocity",
                    "center_error": "capture_value_center_error",
                    "switch_allowed": "capture_value_switch_allowed",
                }
                for output_name, prediction_name in scalar_fields.items():
                    value = pred.get(prediction_name)
                    if value is None:
                        continue
                    scalar = value.detach().view(-1)[0].cpu().item()
                    if output_name in {"raw_selected_index"}:
                        scalar = int(scalar)
                    elif output_name in {"used_fallback", "switch_allowed"}:
                        scalar = bool(scalar)
                    else:
                        scalar = float(scalar)
                    capture_value_diagnostics[output_name] = scalar
                vector_fields = {
                    "recenter_costs": "capture_value_recenter_costs",
                    "pursuit_costs": "capture_value_pursuit_costs",
                    "smooth_costs": "capture_value_smooth_costs",
                    "consensus_costs": "capture_value_consensus_costs",
                    "selected_final_center_error": "capture_value_selected_final_center_error",
                    "action_prior_mean": "capture_action_prior_mean",
                    "action_prior_std": "capture_action_prior_std",
                }
                for output_name, prediction_name in vector_fields.items():
                    value = pred.get(prediction_name)
                    if value is not None:
                        capture_value_diagnostics[output_name] = (
                            value.detach().float().squeeze(0).cpu().tolist()
                        )
        if action_sequence_physical is None or len(action_sequence_physical) == 0:
            action_sequence_norm = np.asarray(action_norm, dtype=np.float32).reshape(1, -1)
            action_sequence_physical = np.asarray(action_physical, dtype=np.float32).reshape(1, -1)
        if save_action_overlays_enabled:
            overlay_path = action_overlay_dir / f"frame_{t:05d}_target_crop_action_traj.png"
            action_overlay_metadata = save_target_crop_action_overlay(
                overlay_path,
                rgb_img,
                rel_body,
                action_sequence_physical,
                args.fov_deg,
                args.ortrack_camera_offset_body,
                args.dt,
                ortrack_bbox_xywh=ortrack_bbox,
                ortrack_confidence=ortrack_confidence,
                model_driven_search_geometry=(
                    None
                    if model_driven_cropper is None or ortrack_result is None
                    else ortrack_result.get("search_crop_xy_size")
                ),
                current_state_box_cxcywh=predicted_current_tracker_box,
                future_state_boxes_cxcywh=predicted_future_tracker_boxes,
            )
            action_overlay_metadata["overlay"] = str(overlay_path.relative_to(out_dir))
        if profile_enabled and torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        profile_model_done = time.perf_counter()

        if args.camera_only_virtual_uav:
            uav_state_after = apply_action_to_virtual_state(
                uav_state_before, action_physical, args.dt, args.max_step_norm
            )
            virtual_uav_state = uav_state_after
        elif args.control_mode == "velocity":
            uav_state_after = apply_action_by_velocity(executor, action_physical, args.dt)
        else:
            uav_state_after = apply_action_by_pose(executor, action_physical, args.dt, args.max_step_norm)

        post_rel_body = compute_target_relative_body(executor, uav_state_after, target_now)
        distance = float(np.linalg.norm(post_rel_body))
        distances.append(distance)
        visible = is_visible_by_geometry(post_rel_body, fov_deg=args.fov_deg)
        if visible:
            visible_count += 1

        collision_info_after_action = None
        collision_after_action = False
        if not args.camera_only_virtual_uav:
            collision_info_after_action = executor.client.simGetCollisionInfo(
                vehicle_name=executor.uav_vehicle_name
            )
            collision_after_action = (
                bool(collision_info_after_action.has_collided)
                if collision_info_after_action is not None
                else False
            )
        collision_now = bool(collision_before_action or collision_after_action)
        collision = bool(collision or collision_now)
        close_enough = bool(distance <= float(args.capture_distance))
        effectively_tracked = bool(close_enough and visible and not collision_now)

        if expert_action is not None:
            if expert_action_norm is None:
                expert_action_norm = physical_action_to_norm(expert_action, cfg.max_vel, cfg.max_yaw_rate)
            diff = action_physical - expert_action
            action_abs_err.append(float(np.mean(np.abs(diff))))
            action_mse.append(float(np.mean(diff ** 2)))

        step_record = {
            "step": t,
            "instruction": instruction,
            "uav_position_before": _airsim_xyz_to_dataset(uav_state_before["position"]),
            "uav_position_after": _airsim_xyz_to_dataset(uav_state_after["position"]),
            "uav_pose_jump_from_prev_after": _delta_xyz_to_dict(pose_jump_dataset),
            "uav_pose_jump_norm": pose_jump_norm,
            "uav_pose_jump_z_abs": pose_jump_z_abs,
            "large_pose_jump": bool(large_pose_jump),
            "target_position": _airsim_xyz_to_dataset(target_now),
            "relative_target_body": rel_body.astype(float).tolist(),
            "relative_target_body_after": post_rel_body.astype(float).tolist(),
            "distance_after": distance,
            "visible_by_geometry": bool(visible),
            "close_enough": bool(close_enough),
            "effectively_tracked": bool(effectively_tracked),
            "action_source": action_source,
            "action_norm": action_norm.astype(float).tolist(),
            "action_physical": action_physical.astype(float).tolist(),
            "expert_action_physical": None if expert_action is None else expert_action.astype(float).tolist(),
            "expert_action_norm": None if expert_action_norm is None else expert_action_norm.astype(float).tolist(),
            "expert_action_source": expert_action_source,
            "collision": bool(collision_now),
            "collision_before_action": bool(collision_before_action),
            "collision_after_action": bool(collision_after_action),
            "collision_info_before_action": _collision_info_to_dict(collision_info_before_action),
            "collision_info_after_action": _collision_info_to_dict(collision_info_after_action),
            "camera_timestamp": executor._last_camera_timestamp,
            "jammers": {
                str(did): _airsim_xyz_to_dataset(pos) for did, pos in jammer_positions_now.items()
            },
        }
        if action_sequence_norm is not None:
            step_record["action_sequence_norm"] = action_sequence_norm.astype(float).tolist()
            step_record["action_sequence_physical"] = action_sequence_physical.astype(float).tolist()
        if capture_value_diagnostics is not None:
            step_record["capture_value_reranking"] = capture_value_diagnostics
        if action_overlay_metadata is not None:
            step_record["target_crop_action_overlay"] = action_overlay_metadata
        if ortrack_bbox is not None:
            step_record["ortrack_bbox_xywh"] = ortrack_bbox
            step_record["ortrack_confidence"] = ortrack_confidence
        if predicted_current_tracker_center is not None:
            step_record["current_target_center_xy"] = (
                predicted_current_tracker_center.astype(float).tolist()
            )
        if predicted_current_tracker_box is not None:
            step_record["current_target_box_cxcywh"] = (
                predicted_current_tracker_box.astype(float).tolist()
            )
        if predicted_future_tracker_center is not None:
            step_record["future_target_center_xy"] = (
                predicted_future_tracker_center.astype(float).tolist()
            )
        if predicted_future_tracker_box is not None:
            step_record["future_target_box_cxcywh"] = (
                predicted_future_tracker_box.astype(float).tolist()
            )
        if predicted_future_tracker_boxes is not None:
            step_record["future_target_boxes_cxcywh"] = (
                predicted_future_tracker_boxes.astype(float).tolist()
            )
        if model_driven_cropper is not None:
            step_record["model_driven_search_bbox_xywh"] = ortrack_bbox
            if ortrack_result is not None and ortrack_result.get("search_crop_xy_size") is not None:
                step_record["model_driven_search_geometry"] = [
                    int(value) for value in ortrack_result["search_crop_xy_size"]
                ]
        if tracker_runtime_enabled and ortrack_confidence is not None:
            step_record["tracker_detection_confidence_threshold"] = float(
                args.tracker_detection_confidence_threshold
            )
            step_record["tracker_target_detected"] = bool(
                ortrack_confidence >= float(args.tracker_detection_confidence_threshold)
            )
        if reused_action_sequence_index is not None:
            step_record["reused_action_sequence_index"] = int(reused_action_sequence_index)
        if attention_map_relpath is not None:
            step_record["last_transformer_attention_map"] = attention_map_relpath
        if attention_comparison_relpath is not None:
            step_record["attention_tracker_comparison"] = attention_comparison_relpath
        if predicted_video_relpaths is not None:
            step_record["predicted_video_frames"] = predicted_video_relpaths
        if predicted_video_state_overlay_relpaths is not None:
            step_record["predicted_video_state_overlays"] = (
                predicted_video_state_overlay_relpaths
            )
        if pred is not None and "candidate_scores" in pred:
            selected_candidate = int(pred["selected_candidate"].detach().view(-1)[0].cpu().item())
            candidate_scores = pred["candidate_scores"].detach().float().view(-1).cpu().numpy()
            step_record["dit_candidate_selection"] = {
                "selected": selected_candidate,
                "scores": candidate_scores.astype(float).tolist(),
                "selected_score": float(candidate_scores[selected_candidate]),
            }
            for key in (
                "candidate_yaw_angle",
                "candidate_pitch_angle",
                "candidate_final_distance_norm",
                "candidate_progress_penalty",
                "candidate_front_penalty",
                "candidate_smooth_prev",
                "candidate_temporal_smooth",
                "candidate_action_effort",
            ):
                if key in pred:
                    values = pred[key].detach().float().view(-1).cpu().numpy()
                    step_record["dit_candidate_selection"][key.replace("candidate_", "")] = values.astype(float).tolist()
        steps.append(step_record)

        prev_action = torch.from_numpy(action_norm).view(1, -1).to(device).float()

        # The requested success metric is an end-of-trajectory metric: only the
        # final frame counts. Therefore, do not mark prev_done or stop early just
        # because an intermediate frame is within the capture radius.
        prev_done = torch.tensor([1.0 if collision_now else 0.0], device=device)
        profile_control_done = time.perf_counter()

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            pred_act = ",".join(f"{x:.2f}" for x in action_physical.reshape(-1))
            expert_act = (
                ",".join(f"{x:.2f}" for x in expert_action.reshape(-1))
                if expert_action is not None
                else "n/a"
            )
            iterator.set_postfix(
                dist=f"{distance:.2f}",
                action=pred_act,
                expert=expert_act,
            )

        prev_uav_after_pos = np.asarray(uav_state_after["position"], dtype=np.float32).copy()
        prev_uav_after_state = {
            "position": np.asarray(uav_state_after["position"], dtype=np.float32).copy(),
            "orientation": np.asarray(uav_state_after["orientation"], dtype=np.float32).copy(),
        }

        if profile_enabled:
            profile_end = time.perf_counter()
            step_profile = {
                "scene_state_ms": (profile_scene_done - profile_step_start) * 1000.0,
                "camera_pose_ms": (profile_camera_pose_done - profile_camera_pose_started) * 1000.0,
                "camera_render_ms": (profile_camera_render_done - profile_camera_render_started) * 1000.0,
                "camera_ms": (profile_camera_done - profile_scene_done) * 1000.0,
                "camera_rpc_ms": float(camera_capture_profile.get("camera_rpc_ms", 0.0)),
                "camera_decode_ms": float(camera_capture_profile.get("camera_decode_ms", 0.0)),
                "rgb_save_ms": (profile_rgb_done - profile_camera_done) * 1000.0,
                "tracker_planner_ms": (profile_planner_done - profile_rgb_done) * 1000.0,
                "model_ms": (profile_model_done - profile_model_start) * 1000.0,
                "control_state_ms": (profile_control_done - profile_model_done) * 1000.0,
                "bookkeeping_ms": (profile_end - profile_control_done) * 1000.0,
                "total_ms": (profile_end - profile_step_start) * 1000.0,
            }
            step_record["time_profile_ms"] = step_profile
            profile_records.append(step_profile)
            interval = max(int(args.profile_step_time_interval), 1)
            if len(profile_records) % interval == 0:
                recent = profile_records[-interval:]
                means = {key: float(np.mean([item[key] for item in recent])) for key in recent[0]}
                detail = " ".join(f"{key}={value:.1f}" for key, value in means.items())
                print(f"[profile-step-time] {traj.scene_id}/{traj.trajectory_name} step={t} {detail}")

        if collision_now and args.stop_on_collision:
            break

    effective_flags = [bool(s.get("effectively_tracked", False)) for s in steps]
    close_flags = [bool(s.get("close_enough", False)) for s in steps]
    visible_flags = [bool(s.get("visible_by_geometry", False)) for s in steps]
    collision_flags = [bool(s.get("collision", False)) for s in steps]

    tracked_frames_before_failure = 0
    for step in steps:
        if bool(step.get("collision", False)):
            break
        if not bool(step.get("close_enough", False)):
            break
        if not bool(step.get("visible_by_geometry", False)):
            break
        tracked_frames_before_failure += 1

    # Success criterion: only the final frame counts, and any collision fails.
    # Visibility is included only when explicitly requested.
    final_distance = float(distances[-1]) if distances else float("inf")
    final_visible = bool(steps[-1].get("visible_by_geometry", False)) if steps else False
    final_close = bool(final_distance <= float(args.capture_distance))
    success = bool((not collision) and final_close and ((not args.require_visibility_for_success) or final_visible))
    success_step = (len(steps) - 1) if (success and steps) else None
    success_criterion = "no collision and final_distance <= capture_distance"
    if args.require_visibility_for_success:
        success_criterion += " and final_visible_by_geometry"

    failure_step = None
    failure_reason = "none" if success else "unknown"
    if not success:
        if collision:
            for idx, step in enumerate(steps):
                if bool(step.get("collision", False)):
                    failure_step = idx
                    break
            failure_reason = "collision"
        elif not final_close:
            failure_step = (len(steps) - 1) if steps else None
            failure_reason = "out_of_capture_distance"
        elif args.require_visibility_for_success and not final_visible:
            failure_step = (len(steps) - 1) if steps else None
            failure_reason = "target_not_visible"

    summary = {
        "scene_id": traj.scene_id,
        "trajectory_name": traj.trajectory_name,
        "dataset_dir": str(traj.dataset_dir),
        "num_steps": len(steps),
        "success": bool(success),
        "success_step": success_step,
        "collision": bool(collision),
        "failure_step": failure_step,
        "failure_reason": failure_reason,
        "tracker_detection_confidence_threshold": (
            float(args.tracker_detection_confidence_threshold)
            if args.reuse_last_confident_action_sequence
            else None
        ),
        "tracker_fallback_action_mode": (
            str(args.tracker_fallback_action_mode)
            if args.reuse_last_confident_action_sequence
            else None
        ),
        "tracker_missed_steps": int(sum(not bool(s.get("tracker_target_detected", True)) for s in steps)),
        "reused_action_sequence_steps": int(
            sum(s.get("action_source") == "last_confident_action_sequence" for s in steps)
        ),
        "tracked_frames_before_failure": int(tracked_frames_before_failure),
        "tracked_frame_ratio_before_failure": float(tracked_frames_before_failure / max(len(steps), 1)),
        "consecutive_tracked_frames_before_failure": int(tracked_frames_before_failure),
        "consecutive_tracked_frame_ratio_before_failure": float(tracked_frames_before_failure / max(len(steps), 1)),
        "effective_tracked_frames": int(sum(1 for x in effective_flags if x)),
        "effective_tracking_ratio": float(np.mean(effective_flags)) if effective_flags else None,
        "close_frame_ratio": float(np.mean(close_flags)) if close_flags else None,
        "visible_frame_ratio": float(np.mean(visible_flags)) if visible_flags else None,
        "collision_frame_ratio": float(np.mean(collision_flags)) if collision_flags else None,
        "final_distance": final_distance if distances else None,
        "final_close_enough": bool(final_close),
        "final_visible_by_geometry": bool(final_visible),
        "success_criterion": success_criterion,
        "mean_distance": float(np.mean(distances)) if distances else None,
        "visible_ratio_geometry": float(visible_count / max(len(steps), 1)),
        "mean_action_abs_error_physical": float(np.mean(action_abs_err)) if action_abs_err else None,
        "rmse_action_physical": float(math.sqrt(np.mean(action_mse))) if action_mse else None,
        "target_asset_name": getattr(executor, "target_asset_name", None),
        "jammer_asset_names": getattr(executor, "_jammer_asset_names_by_id", {}),
        "control_mode": "camera_only_virtual_pose" if args.camera_only_virtual_uav else args.control_mode,
        "camera_only_virtual_uav": bool(args.camera_only_virtual_uav),
        "camera_stale_rejections": int(executor.camera_stale_rejections),
        "camera_distance_rejections": int(executor.camera_distance_rejections),
        "camera_pose_rejections": int(executor.camera_pose_rejections),
        "replay_expert_action": bool(args.replay_expert_action),
        "hold_uav_pose_during_scene_step": bool(args.hold_uav_pose_during_scene_step),
        "capture_distance": args.capture_distance,
        "require_visibility_for_success": args.require_visibility_for_success,
    }
    if profile_records:
        summary["time_profile_mean_ms"] = {
            key: float(np.mean([item[key] for item in profile_records])) for key in profile_records[0]
        }

    _dump_json(out_dir / "online_rollout.json", {"summary": summary, "steps": steps})
    if args.save_trajectory_3d:
        _save_trajectory_3d_plot(out_dir, steps)
    return summary


# -----------------------------
# Delayed sim_server startup helpers
# -----------------------------


def _port_is_listening(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _terminate_process_group(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def start_sim_server_after_model_if_needed(args: argparse.Namespace) -> Optional[subprocess.Popen]:
    """Start msgpack sim_server only after model/checkpoint has been loaded."""
    if not getattr(args, "start_sim_server", False):
        print("[sim_server] delayed auto-start disabled; expecting an existing sim_server.")
        return None

    host = str(args.sim_server_host)
    port = int(args.sim_server_port)
    if _port_is_listening(host, port, timeout=1.0):
        print(f"[sim_server] port {host}:{port} already listening; reuse existing sim_server.")
        return None

    script = Path(args.sim_server_script).expanduser().resolve()
    if not script.exists():
        raise FileNotFoundError(f"sim_server.py not found: {script}")

    log_path = Path(args.sim_server_log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_path = str(Path(args.sim_server_root_path).expanduser().resolve())

    cmd = [
        sys.executable,
        str(script),
        "--gpus",
        str(args.gpu_id),
        "--port",
        str(port),
        "--root_path",
        root_path,
    ]
    print(f"[sim_server] starting after model load: {' '.join(cmd)}")
    print(f"[sim_server] log: {log_path}")
    log_f = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    proc._online_eval_log_file = log_f  # type: ignore[attr-defined]

    wait_seconds = float(args.sim_server_wait_seconds)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"sim_server exited early with code {proc.returncode}. Check log: {log_path}"
            )
        if _port_is_listening(host, port, timeout=1.0):
            print(f"[sim_server] ready on {host}:{port}")
            return proc
        time.sleep(0.5)

    raise TimeoutError(f"sim_server did not listen on {host}:{port} within {wait_seconds:.1f}s. Check log: {log_path}")


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Online AirSim validation for Teacher WorldModel + DiT action head")

    # Paths
    parser.add_argument("--dataset-root", type=str, required=True, help=f"Collected Dataset root, e.g. {PROJECT_ROOT / 'Dataset'}")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--executor-script", type=str, default=str(PROJECT_ROOT / "code/src/executor/trajectory_executor.py"))
    parser.add_argument("--start-sim-server", action="store_true", default=False, help="Start sim_server.py inside this script after the model is loaded.")
    parser.add_argument("--sim-server-script", type=str, default=str(PROJECT_ROOT / "code/src/envs/sim_server.py"))
    parser.add_argument("--sim-server-root-path", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--sim-server-log", type=str, default=str(PROJECT_ROOT / "online_eval_teacher_dit/sim_server_30000.log"))
    parser.add_argument("--sim-server-wait-seconds", type=float, default=60.0)
    parser.add_argument("--stop-sim-server-on-exit", action="store_true", default=False)

    # Dataset selection
    parser.add_argument("--scene-list", type=str, default="City_1")
    parser.add_argument("--trajectory-range", type=str, default="")
    parser.add_argument("--eval-split", type=str, default="val", choices=["all", "train", "val"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--summary-shard-id", type=str, default="")
    parser.add_argument("--eval-semantic-signature", type=str, default="standard")

    # Model config
    parser.add_argument("--tokenizer-name", type=str, default=LOCAL_CLIP_MODEL_PATH)
    parser.add_argument("--clip-text-model-name", type=str, default=LOCAL_CLIP_MODEL_PATH)
    parser.add_argument("--dinov2-model-name", type=str, default=LOCAL_DINOV2_MODEL_PATH)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--target-relative-dim", type=int, default=3)
    parser.add_argument("--target-relative-scale", type=float, default=1.0, help="Use 1.0 if training used raw target_position_in_body_frame; set 100.0 only if your builder normalized by 100.")
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--compile-action-sampling", action="store_true", default=True)
    parser.add_argument("--no-compile-action-sampling", action="store_false", dest="compile_action_sampling")
    parser.add_argument(
        "--compile-action-sampling-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    parser.add_argument("--dit-candidate-selection", type=_str2bool, default=None)
    parser.add_argument("--dit-candidate-count", type=int, default=None)
    parser.add_argument("--dit-candidate-lateral-weight", type=float, default=None)
    parser.add_argument("--dit-candidate-vertical-weight", type=float, default=None)
    parser.add_argument("--dit-candidate-distance-weight", type=float, default=None)
    parser.add_argument("--dit-candidate-smooth-weight", type=float, default=None)
    parser.add_argument("--dit-candidate-yaw-angle-weight", type=float, default=None)
    parser.add_argument("--dit-candidate-pitch-angle-weight", type=float, default=None)
    parser.add_argument("--dit-candidate-final-distance-weight", type=float, default=None)
    parser.add_argument("--dit-candidate-progress-weight", type=float, default=None)
    parser.add_argument("--dit-candidate-front-weight", type=float, default=None)
    parser.add_argument("--dit-candidate-action-weight", type=float, default=None)
    parser.add_argument("--dit-candidate-temporal-smooth-weight", type=float, default=None)
    parser.add_argument("--use-target-relative-context", type=_str2bool, default=None)
    parser.add_argument("--target-relative-context-scale", type=float, default=None)
    parser.add_argument("--target-relative-token-scale", type=float, default=None)
    parser.add_argument("--target-relative-context-hidden-dim", type=int, default=None)
    parser.add_argument("--max-vel", type=float, default=DEFAULT_MODEL_CFG.max_vel)
    parser.add_argument("--max-yaw-rate", type=float, default=DEFAULT_MODEL_CFG.max_yaw_rate)
    parser.add_argument("--max-speed-norm", type=float, default=DEFAULT_MODEL_CFG.max_speed_norm)
    parser.add_argument("--freeze-dinov2", action="store_true", default=True)
    parser.add_argument("--finetune-dinov2", action="store_false", dest="freeze_dinov2")
    parser.add_argument("--freeze-clip-text", action="store_true", default=True)
    parser.add_argument("--finetune-clip-text", action="store_false", dest="freeze_clip_text")
    parser.add_argument("--use-wan22-encoders", type=_str2bool, default=None)
    parser.add_argument("--wan22-model-base-path", type=str, default=DEFAULT_MODEL_CFG.wan22_model_base_path)
    parser.add_argument("--wan22-fastwam-src-path", type=str, default=DEFAULT_MODEL_CFG.wan22_fastwam_src_path)
    parser.add_argument("--wan22-skip-download", type=_str2bool, default=DEFAULT_MODEL_CFG.wan22_skip_download)
    parser.add_argument("--wan22-text-context-length", type=int, default=DEFAULT_MODEL_CFG.wan22_text_context_length)
    parser.add_argument("--wan22-text-encode-batch-size", type=int, default=DEFAULT_MODEL_CFG.wan22_text_encode_batch_size)
    parser.add_argument("--deterministic-action", action="store_true", default=True)
    parser.add_argument("--stochastic-action", action="store_false", dest="deterministic_action")
    parser.add_argument(
        "--use-diffusion-actor",
        type=_str2bool,
        default=None,
        help="true/false: override cfg.use_diffusion_actor from checkpoint. By default, use checkpoint cfg.",
    )
    parser.add_argument(
        "--use-fastwam-mot",
        type=_str2bool,
        default=None,
        help="true/false: override cfg.use_fastwam_mot from checkpoint. By default, infer from checkpoint cfg/keys.",
    )
    parser.add_argument("--use-capture-value-reranking", type=_str2bool, default=None)
    parser.add_argument(
        "--capture-value-score-mode",
        type=str,
        default=None,
        choices=["learned", "geometric", "action_prior"],
    )
    parser.add_argument("--capture-value-candidate-count", type=int, default=None)
    parser.add_argument("--capture-value-control-dt", type=float, default=None)
    parser.add_argument("--capture-value-horizontal-fov-deg", type=float, default=None)
    parser.add_argument("--capture-value-vertical-fov-deg", type=float, default=None)
    parser.add_argument("--capture-value-bbox-depth-scale", type=float, default=None)
    parser.add_argument("--capture-value-min-depth", type=float, default=None)
    parser.add_argument("--capture-value-max-depth", type=float, default=None)
    parser.add_argument("--capture-value-target-box-size", type=float, default=None)
    parser.add_argument("--capture-value-box-size-sigma", type=float, default=None)
    parser.add_argument("--capture-value-discount", type=float, default=None)
    parser.add_argument("--capture-value-recenter-sigma", type=float, default=None)
    parser.add_argument("--capture-value-pursuit-center-sigma", type=float, default=None)
    parser.add_argument("--capture-value-out-of-frame-weight", type=float, default=None)
    parser.add_argument("--capture-value-first-action-smooth-weight", type=float, default=None)
    parser.add_argument("--capture-value-temporal-smooth-weight", type=float, default=None)
    parser.add_argument("--capture-value-recenter-weight", type=float, default=None)
    parser.add_argument("--capture-value-pursuit-weight", type=float, default=None)
    parser.add_argument("--capture-value-smooth-weight", type=float, default=None)
    parser.add_argument("--capture-value-consensus-weight", type=float, default=None)
    parser.add_argument("--capture-value-short-horizon", type=int, default=None)
    parser.add_argument("--capture-value-selection-margin", type=float, default=None)
    parser.add_argument("--capture-value-min-center-error", type=float, default=None)
    parser.add_argument("--capture-action-prior-checkpoint", type=str, default=None)
    parser.add_argument(
        "--capture-action-prior-dimension-weights",
        nargs=4,
        type=float,
        default=None,
    )
    parser.add_argument(
        "--capture-value-structured-candidates", type=_str2bool, default=None
    )
    parser.add_argument(
        "--target-token-fusion-mode",
        type=str,
        default=None,
        choices=["attention", "concat"],
        help="Override cfg.target_token_fusion_mode from checkpoint. By default, use checkpoint cfg.",
    )

    # AirSim / executor config
    parser.add_argument("--sim-server-host", type=str, default="127.0.0.1")
    parser.add_argument("--sim-server-port", type=int, default=30000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--sim-gpu-id",
        type=int,
        default=None,
        help="Physical GPU requested from SimServer for Unreal; defaults to --gpu-id.",
    )
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--uav-vehicle-name", type=str, default="Drone_1")
    parser.add_argument("--target-object-name", type=str, default="UAV1")
    parser.add_argument("--target-asset-name", type=str, default="UAV1")
    parser.add_argument("--jammer-object-name", type=str, default="JammerUAV")
    parser.add_argument("--jammer-asset-name", type=str, default="UAV1")
    parser.add_argument("--camera-name", type=str, default="0")
    parser.add_argument("--validate-camera-freshness", action="store_true", default=False)
    parser.add_argument(
        "--no-validate-camera-freshness",
        action="store_false",
        dest="validate_camera_freshness",
    )
    parser.add_argument("--camera-max-vehicle-distance", type=float, default=5.0)
    parser.add_argument("--camera-pose-tolerance-m", type=float, default=0.05)
    parser.add_argument("--camera-orientation-tolerance-deg", type=float, default=1.0)
    parser.add_argument(
        "--camera-capture-mode",
        type=str,
        default="legacy_step",
        choices=["legacy_step", "fresh_frame"],
        help="legacy_step preserves the old pre-render path; fresh_frame captures the first new frame while briefly unpaused.",
    )
    parser.add_argument(
        "--camera-render-max-fps",
        type=float,
        default=0.0,
        help="Unreal offscreen render cap. Values <= 0 preserve the engine default.",
    )
    parser.add_argument(
        "--camera-render-frames",
        type=int,
        default=1,
        help="Frames rendered after moving scene objects and before RGB capture.",
    )
    parser.add_argument("--save-depth", action="store_true", default=False)
    parser.add_argument("--no-save-depth", action="store_false", dest="save_depth")
    parser.add_argument("--use-external-camera", action="store_true", default=False)
    parser.add_argument("--no-use-external-camera", action="store_false", dest="use_external_camera")
    parser.add_argument("--target-scale", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--jammer-scale", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--disable-jammer", action="store_true", default=False)
    parser.add_argument("--reuse-saved-assets", action="store_true", default=True)
    parser.add_argument("--do-not-reuse-saved-assets", action="store_false", dest="reuse_saved_assets")
    parser.add_argument("--random-target-asset", action="store_true", default=False)
    parser.add_argument("--random-jammer-asset", action="store_true", default=False)

    # Online rollout config
    parser.add_argument("--control-mode", type=str, default="pose", choices=["pose", "velocity"])
    parser.add_argument("--camera-only-virtual-uav", action="store_true", default=False)
    parser.add_argument("--no-camera-only-virtual-uav", action="store_false", dest="camera_only_virtual_uav")
    parser.add_argument("--replay-expert-action", action="store_true", default=False, help="Sanity-check mode: execute dataset expert actions instead of model actions.")
    parser.add_argument("--hold-uav-pose-during-scene-step", action="store_true", default=True, help="In pose control, keep UAV fixed while stepping target/jammer scene updates.")
    parser.add_argument("--no-hold-uav-pose-during-scene-step", action="store_false", dest="hold_uav_pose_during_scene_step")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-step-norm", type=float, default=1.0, help="Safety clamp for one pose-control step. <=0 disables clamp.")
    parser.add_argument("--force-live-instruction", action="store_true", default=False, help="Generate instruction from current online relative target state instead of replaying saved dataset text.")
    parser.add_argument("--capture-distance", type=float, default=10.0)
    parser.add_argument("--require-visibility-for-success", action="store_true", default=False)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--pose-jump-warn-threshold", type=float, default=5.0, help="Warn/log large UAV pose discontinuities between previous after-pose and current before-pose.")
    parser.add_argument("--pose-jump-z-warn-threshold", type=float, default=5.0, help="Warn/log large UAV z discontinuities between previous after-pose and current before-pose.")
    parser.add_argument("--stop-on-success", action="store_true", default=False, help="Kept for compatibility; final-frame success does not stop early.")
    parser.add_argument("--stop-on-collision", action="store_true", default=True)
    parser.add_argument("--no-stop-on-collision", action="store_false", dest="stop_on_collision")
    parser.add_argument("--save-rgb", action="store_true", default=True)
    parser.add_argument("--no-save-rgb", action="store_false", dest="save_rgb")
    parser.add_argument(
        "--visualize-trajectory-keys",
        type=str,
        default="",
        help=(
            "Comma/space separated whitelist for saving visual assets, "
            "for example City_1/trajectory_0451. Empty or 'all' saves all trajectories."
        ),
    )
    parser.add_argument("--save-transformer-attention-maps", action="store_true", default=False)
    parser.add_argument("--no-save-transformer-attention-maps", action="store_false", dest="save_transformer_attention_maps")
    parser.add_argument("--save-attention-tracker-comparisons", action="store_true", default=False)
    parser.add_argument(
        "--no-save-attention-tracker-comparisons",
        action="store_false",
        dest="save_attention_tracker_comparisons",
    )
    parser.add_argument(
        "--attention-trajectory-keys",
        type=str,
        default="",
        help="Trajectory whitelist for transformer attention maps; RGB uses --visualize-trajectory-keys.",
    )
    parser.add_argument("--save-predicted-video", action="store_true", default=False)
    parser.add_argument("--no-save-predicted-video", action="store_false", dest="save_predicted_video")
    parser.add_argument(
        "--predicted-video-trajectory-keys",
        type=str,
        default="",
        help=(
            "Comma/space separated whitelist for decoding predicted videos at every online step. "
            "Empty or 'all' enables every trajectory when --save-predicted-video is set."
        ),
    )
    parser.add_argument("--predicted-video-latent-frames", type=int, default=3)
    parser.add_argument("--save-ortrack-maps", action="store_true", default=False)
    parser.add_argument("--save-target-crop-action-overlays", action="store_true", default=True)
    parser.add_argument(
        "--no-save-target-crop-action-overlays",
        action="store_false",
        dest="save_target_crop_action_overlays",
    )
    parser.add_argument(
        "--target-crop-action-overlay-output-name",
        type=str,
        default="target_crop_action_trajectory_overlays",
    )
    parser.add_argument("--save-trajectory-3d", action="store_true", default=True)
    parser.add_argument(
        "--no-save-trajectory-3d", action="store_false", dest="save_trajectory_3d"
    )
    parser.add_argument("--profile-step-time", action="store_true", default=False)
    parser.add_argument("--profile-step-time-interval", type=int, default=10)
    parser.add_argument("--ortrack-root", type=str, default=str(PROJECT_ROOT / "third_party/ortrack"))
    parser.add_argument(
        "--tracker-checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "experiments/tracker_artifacts/models/uav_tracker_gt_bbox_square/best.pt"),
    )
    parser.add_argument(
        "--ortrack-checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "model/ortrack/deit_tiny_patch16_224/ORTrack_ep0300.pth.tar"),
    )
    parser.add_argument("--ortrack-config", type=str, default="deit_tiny_patch16_224")
    parser.add_argument("--ortrack-camera-offset-body", nargs=3, type=float, default=[0.46, 0.0, 0.0])
    parser.add_argument("--ortrack-init-box-frac", type=float, default=0.10)
    parser.add_argument(
        "--reuse-last-confident-action-sequence",
        action="store_true",
        default=False,
        help="When tracker confidence is low, execute the remaining actions from the last confident sequence.",
    )
    parser.add_argument(
        "--tracker-detection-confidence-threshold",
        type=float,
        default=0.5,
        help="Tracker confidence below this value is treated as target loss.",
    )
    parser.add_argument(
        "--tracker-fallback-action-mode",
        choices=["remaining_sequence", "first_action"],
        default="remaining_sequence",
        help=(
            "Fallback policy after target loss: advance through the last confident sequence, "
            "or repeatedly execute its first action."
        ),
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-direct-action",
        action="store_true",
        default=False,
        help="At inference use the MLP direct action head (phase-A checkpoints); overrides cfg.use_diffusion_actor from checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.camera_only_virtual_uav and not args.use_external_camera:
        raise SystemExit("--camera-only-virtual-uav requires --use-external-camera")
    seed_everything(args.seed)

    # Force a single evaluator process. Multiple scenes, if requested, are
    # evaluated sequentially by reopening one AirSim scene at a time.
    # Keep DAGGER_MULTI_WORKER=1 even in single-GPU mode: in your TrajectoryExecutor,
    # this flag prevents connect() from proactively calling close_scenes before every
    # connection/retry. It does NOT create extra evaluator processes here.
    os.environ["DAGGER_MULTI_WORKER"] = "1"

    scene_ids = _parse_scene_list(args.scene_list)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("[mode] single GPU / single Python evaluator / sequential AirSim scenes")
    print(f"[device] {device}")
    print(f"[DAGGER_MULTI_WORKER] {os.environ.get('DAGGER_MULTI_WORKER')}")
    print(f"[dataset-root] {args.dataset_root}")
    print(f"[checkpoint] {args.checkpoint}")

    model, cfg = load_model(args, device)
    tokenizer = None
    if not cfg.use_wan22_encoders:
        if CLIPTokenizerFast is None:
            raise ImportError("transformers.CLIPTokenizerFast is required for online instruction tokenization")
        tokenizer = CLIPTokenizerFast.from_pretrained(args.tokenizer_name, local_files_only=True)

    # A previous failed evaluator may have left its Unreal worker alive. Close
    # only the scenes owned by this evaluator before the compile warmup.
    close_requested_scenes_before_compile(args, scene_ids)

    # Finish lazy Inductor/CUDA Graph compilation before opening an Unreal scene.
    # Otherwise the first non-attention trajectory compiles while AirSim is live
    # and can starve the camera renderer when both processes share a GPU.
    warmup_compiled_action_sampler(model, cfg, tokenizer, args, device)

    # IMPORTANT: start sim_server/AirSim only after model loading and compile warmup.
    sim_server_proc = start_sim_server_after_model_if_needed(args)

    transform = make_image_transform(cfg.image_size)
    executor_module = dynamic_import_module(Path(args.executor_script))

    traj_dirs = discover_dataset_trajectories(
        Path(args.dataset_root),
        scene_ids,
        trajectory_range=args.trajectory_range,
        split=args.eval_split,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        max_trajectories=args.max_trajectories,
    )
    if not traj_dirs:
        raise RuntimeError("No collected dataset trajectories found. Check --dataset-root/--scene-list/--trajectory-range.")

    print(f"[eval] trajectories={len(traj_dirs)} split={args.eval_split} scenes={scene_ids}")
    print(f"[output] {args.output_dir}")

    expected_keys = {_trajectory_key(p.parent.name, p.name) for p in traj_dirs}
    shard_suffix = f"_{args.summary_shard_id}" if args.summary_shard_id else ""
    partial_path = Path(args.output_dir) / f"summary_partial{shard_suffix}.json"
    all_summaries: List[Dict[str, Any]] = _load_resumable_partial(partial_path, expected_keys)
    completed_keys = {
        _trajectory_key(str(s.get("scene_id") or ""), str(s.get("trajectory_name") or ""))
        for s in all_summaries
    }
    if all_summaries:
        _dump_json(partial_path, {"summaries": all_summaries, "args": vars(args)})
    current_scene = None
    executor = None

    try:
        for idx, dataset_dir in enumerate(traj_dirs, start=1):
            scene_id = dataset_dir.parent.name
            traj_key = _trajectory_key(scene_id, dataset_dir.name)
            refreshed_summaries = _load_resumable_partial(partial_path, expected_keys, verbose=False)
            if len(refreshed_summaries) > len(all_summaries):
                all_summaries = refreshed_summaries
                completed_keys = {
                    _trajectory_key(str(s.get("scene_id") or ""), str(s.get("trajectory_name") or ""))
                    for s in all_summaries
                }
            if traj_key in completed_keys:
                if _visual_assets_exist_for_key(args, cfg, scene_id, dataset_dir.name):
                    print(f"[resume-skip] {traj_key} ({idx}/{len(traj_dirs)})")
                    continue
                print(f"[resume-visualize] {traj_key} ({idx}/{len(traj_dirs)}): rerun to create requested visual assets")
                all_summaries = [
                    summary
                    for summary in all_summaries
                    if _trajectory_key(str(summary.get("scene_id") or ""), str(summary.get("trajectory_name") or "")) != traj_key
                ]
                completed_keys.discard(traj_key)
            traj = load_online_trajectory(dataset_dir, scene_id)
            if current_scene != scene_id or executor is None:
                cleanup_executor(executor)
                executor = build_executor(executor_module, args, scene_id)
                current_scene = scene_id

            print("=" * 100)
            print(f"online eval {scene_id}/{traj.trajectory_name}")
            print(f"dataset trajectory: {dataset_dir}")
            print(f"frames={traj.num_frames}, target_asset={traj.target_asset_name}, jammers={list(traj.jammer_trajs_airsim.keys())}")
            try:
                summary = run_online_trajectory(
                    model, cfg, tokenizer, transform, executor, traj, args, device
                )
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)
                print(
                    f"[trajectory-error] {traj_key}: {error_type}: {error_message}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
                summary = {
                    "scene_id": scene_id,
                    "trajectory_name": traj.trajectory_name,
                    "num_frames": int(traj.num_frames),
                    "success": False,
                    "collision": False,
                    "failure_reason": "runtime_error",
                    "error_type": error_type,
                    "error_message": error_message,
                    "final_distance": None,
                    "mean_distance": None,
                    "effective_tracked_frames": 0,
                    "effective_tracking_ratio": 0.0,
                    "consecutive_tracked_frames_before_failure": 0,
                    "consecutive_tracked_frame_ratio_before_failure": 0.0,
                    "close_frame_ratio": None,
                    "visible_frame_ratio": None,
                    "collision_frame_ratio": None,
                    "final_close_enough": False,
                    "final_visible_by_geometry": False,
                    "control_mode": (
                        "camera_only_virtual_pose" if args.camera_only_virtual_uav else args.control_mode
                    ),
                    "camera_only_virtual_uav": bool(args.camera_only_virtual_uav),
                }
                failure_out_dir = Path(args.output_dir) / scene_id / traj.trajectory_name
                _dump_json(failure_out_dir / "online_rollout.json", {"summary": summary, "steps": []})
                # AirSim failures can leave an RPC client or scene state unusable.
                # Reopen it before the next trajectory while retaining this failure.
                cleanup_executor(executor)
                executor = None
                current_scene = None
            all_summaries.append(summary)
            completed_keys.add(traj_key)
            _dump_json(partial_path, {"summaries": all_summaries, "args": vars(args)})

        success_values = [1.0 if s.get("success") else 0.0 for s in all_summaries]
        collision_values = [1.0 if s.get("collision") else 0.0 for s in all_summaries]
        final_distances = [float(s["final_distance"]) for s in all_summaries if s.get("final_distance") is not None]
        mean_distances = [float(s["mean_distance"]) for s in all_summaries if s.get("mean_distance") is not None]
        effective_frames = [
            float(s["effective_tracked_frames"])
            for s in all_summaries
            if s.get("effective_tracked_frames") is not None
        ]
        effective_tracking_ratios = [
            float(s["effective_tracking_ratio"])
            for s in all_summaries
            if s.get("effective_tracking_ratio") is not None
        ]
        consecutive_frames = [
            float(s["consecutive_tracked_frames_before_failure"])
            for s in all_summaries
            if s.get("consecutive_tracked_frames_before_failure") is not None
        ]
        consecutive_frame_ratios = [
            float(s["consecutive_tracked_frame_ratio_before_failure"])
            for s in all_summaries
            if s.get("consecutive_tracked_frame_ratio_before_failure") is not None
        ]
        close_frame_ratios = [
            float(s["close_frame_ratio"])
            for s in all_summaries
            if s.get("close_frame_ratio") is not None
        ]
        visible_frame_ratios = [
            float(s["visible_frame_ratio"])
            for s in all_summaries
            if s.get("visible_frame_ratio") is not None
        ]
        collision_frame_ratios = [
            float(s["collision_frame_ratio"])
            for s in all_summaries
            if s.get("collision_frame_ratio") is not None
        ]
        final_close_values = [1.0 if s.get("final_close_enough") else 0.0 for s in all_summaries]
        final_visible_values = [1.0 if s.get("final_visible_by_geometry") else 0.0 for s in all_summaries]

        failure_reasons: Dict[str, int] = {}
        for summary in all_summaries:
            reason = str(summary.get("failure_reason") or "unknown")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        agg = {
            "num_trajectories": len(all_summaries),
            "SR": float(np.mean(success_values)) if success_values else None,
            "success_rate": float(np.mean(success_values)) if success_values else None,
            "ATF": float(np.mean(effective_frames)) if effective_frames else None,
            "average_tracked_frames": float(np.mean(effective_frames)) if effective_frames else None,
            "average_tracked_frame_ratio": float(np.mean(effective_tracking_ratios)) if effective_tracking_ratios else None,
            "CTF": float(np.mean(consecutive_frames)) if consecutive_frames else None,
            "consecutive_tracked_frames": float(np.mean(consecutive_frames)) if consecutive_frames else None,
            "average_consecutive_tracked_frame_ratio_before_failure": float(np.mean(consecutive_frame_ratios)) if consecutive_frame_ratios else None,
            "average_effective_tracked_frames": float(np.mean(effective_frames)) if effective_frames else None,
            "mean_effective_tracking_ratio": float(np.mean(effective_tracking_ratios)) if effective_tracking_ratios else None,
            "mean_close_frame_ratio": float(np.mean(close_frame_ratios)) if close_frame_ratios else None,
            "mean_visible_frame_ratio": float(np.mean(visible_frame_ratios)) if visible_frame_ratios else None,
            "mean_collision_frame_ratio": float(np.mean(collision_frame_ratios)) if collision_frame_ratios else None,
            "final_close_rate": float(np.mean(final_close_values)) if final_close_values else None,
            "final_visible_rate": float(np.mean(final_visible_values)) if final_visible_values else None,
            "collision_rate": float(np.mean(collision_values)) if collision_values else None,
            "failure_reason_counts": failure_reasons,
            "mean_final_distance": float(np.mean(final_distances)) if final_distances else None,
            "mean_distance": float(np.mean(mean_distances)) if mean_distances else None,
            "args": vars(args),
            "resolved_cfg": _jsonable_cfg(cfg),
            "summaries": all_summaries,
        }
        _dump_json(Path(args.output_dir) / f"summary{shard_suffix}.json", agg)
        print("=" * 100)
        print(json.dumps({k: v for k, v in agg.items() if k not in {"args", "summaries"}}, indent=2, ensure_ascii=False))
    finally:
        cleanup_executor(executor)
        if args.stop_sim_server_on_exit:
            _terminate_process_group(sim_server_proc)
        try:
            if sim_server_proc is not None and hasattr(sim_server_proc, "_online_eval_log_file"):
                sim_server_proc._online_eval_log_file.close()  # type: ignore[attr-defined]
        except Exception:
            pass


if __name__ == "__main__":
    main()
