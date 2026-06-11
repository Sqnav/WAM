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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torchvision import transforms

from data.action_mapping import clamp_physical_action_speed
from data.visual_guidance import make_attention_heatmap
from model.config import ModelConfig, migrate_legacy_config
from model.model import TeacherWorldModelDiT, migrate_legacy_state_dict_keys

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
    LOCAL_CLIP_MODEL_PATH = "/data1/ysq/Worldmodel/model/clip-vit-base-patch32"
    LOCAL_DINOV2_MODEL_PATH = "/data1/ysq/Worldmodel/model/dinov2-base"

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
    for scene_id in scene_ids:
        scene_dir = _case_insensitive_child(dataset_root, scene_id)
        if scene_dir is None:
            print(f"[warn] scene directory not found: {dataset_root / scene_id}")
            continue
        for uav_json in scene_dir.rglob("uav_trajectory.json"):
            d = uav_json.parent
            if _in_range(scene_id, d.name, ranges):
                all_dirs.append(d.resolve())

    all_dirs = sorted(set(all_dirs), key=_natural_key)
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
        all_dirs = sorted(all_dirs, key=_natural_key)

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
    if getattr(args, "use_target_visual_guidance", None) is not None:
        cfg_kwargs["use_target_visual_guidance"] = bool(args.use_target_visual_guidance)
    if getattr(args, "use_attention_heatmap", None) is not None:
        cfg_kwargs["use_attention_heatmap"] = bool(args.use_attention_heatmap)
    if getattr(args, "visual_guidance_fov_deg", None) is not None:
        cfg_kwargs["visual_guidance_fov_deg"] = float(args.visual_guidance_fov_deg)
    if getattr(args, "attention_heatmap_sigma", None) is not None:
        cfg_kwargs["attention_heatmap_sigma"] = float(args.attention_heatmap_sigma)
    if getattr(args, "use_target_belief_tracker", None) is not None:
        cfg_kwargs["use_target_belief_tracker"] = bool(args.use_target_belief_tracker)
    if getattr(args, "target_belief_token_scale", None) is not None:
        cfg_kwargs["target_belief_token_scale"] = float(args.target_belief_token_scale)
    if getattr(args, "target_belief_update_rate", None) is not None:
        cfg_kwargs["target_belief_update_rate"] = float(args.target_belief_update_rate)
    if getattr(args, "target_belief_min_confidence", None) is not None:
        cfg_kwargs["target_belief_min_confidence"] = float(args.target_belief_min_confidence)
    if getattr(args, "target_belief_temperature", None) is not None:
        cfg_kwargs["target_belief_temperature"] = float(args.target_belief_temperature)
    if getattr(args, "target_belief_loss_weight", None) is not None:
        cfg_kwargs["target_belief_loss_weight"] = float(args.target_belief_loss_weight)
    if getattr(args, "target_belief_motion_weight", None) is not None:
        cfg_kwargs["target_belief_motion_weight"] = float(args.target_belief_motion_weight)
    if getattr(args, "target_belief_update_sharpness", None) is not None:
        cfg_kwargs["target_belief_update_sharpness"] = float(args.target_belief_update_sharpness)
    if getattr(args, "use_latent_mpc", None) is not None:
        cfg_kwargs["use_latent_mpc"] = bool(args.use_latent_mpc)
    if getattr(args, "latent_mpc_candidate_count", None) is not None:
        cfg_kwargs["latent_mpc_candidate_count"] = int(args.latent_mpc_candidate_count)
    if getattr(args, "latent_mpc_distance_weight", None) is not None:
        cfg_kwargs["latent_mpc_distance_weight"] = float(args.latent_mpc_distance_weight)
    if getattr(args, "latent_mpc_smooth_weight", None) is not None:
        cfg_kwargs["latent_mpc_smooth_weight"] = float(args.latent_mpc_smooth_weight)
    if getattr(args, "latent_mpc_action_weight", None) is not None:
        cfg_kwargs["latent_mpc_action_weight"] = float(args.latent_mpc_action_weight)
    if getattr(args, "latent_mpc_visual_weight", None) is not None:
        cfg_kwargs["latent_mpc_visual_weight"] = float(args.latent_mpc_visual_weight)
    if getattr(args, "latent_mpc_latent_frames", None) is not None:
        cfg_kwargs["latent_mpc_latent_frames"] = int(args.latent_mpc_latent_frames)
    if getattr(args, "latent_mpc_video_sampling_steps", None) is not None:
        cfg_kwargs["latent_mpc_video_sampling_steps"] = int(args.latent_mpc_video_sampling_steps)
    if getattr(args, "force_direct_action", False):
        cfg_kwargs["use_diffusion_actor"] = False
    return ModelConfig(**cfg_kwargs)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return migrate_legacy_state_dict_keys(state_dict)


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
            "inactive_visual_guidance": 0,
            "other_frozen": 0,
        }
        for name in expected_missing:
            if name.startswith("image_encoder."):
                groups["image_encoder"] += 1
            elif name.startswith("text_encoder."):
                groups["text_encoder"] += 1
            elif (not cfg.use_diffusion_actor) and name.startswith("actor."):
                groups["inactive_actor"] += 1
            elif (not cfg.use_target_visual_guidance) and name == "fusion.target_bias_embed":
                groups["inactive_visual_guidance"] += 1
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
    model = TeacherWorldModelDiT(cfg).to(device)
    missing, unexpected = model.load_state_dict(_strip_module_prefix(ckpt["model"]), strict=False)
    _summarize_checkpoint_load(model, cfg, missing, unexpected, ckpt)
    model.eval()
    print(f"[model] loaded checkpoint: {ckpt_path}")
    print(
        f"[model] low_dim_target_input=off, "
        f"target_token_fusion_mode={cfg.target_token_fusion_mode}, "
        f"use_diffusion_actor={cfg.use_diffusion_actor}, "
        f"use_fastwam_mot={cfg.use_fastwam_mot}, "
        f"dit_candidate_selection={cfg.dit_candidate_selection}, "
        f"dit_candidate_count={cfg.dit_candidate_count}, "
        f"candidate_score=tracking, "
        f"visual_guidance={cfg.use_target_visual_guidance}, "
        f"target_belief_tracker={cfg.use_target_belief_tracker}, "
        f"latent_mpc={cfg.use_latent_mpc}"
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
    return set_vehicle_pose_static(
        executor,
        new_pos,
        quat,
        retries=3,
        tol_xy=0.8,
        tol_z=0.8,
    )


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


def save_heatmap_overlay(path: Path, rgb: np.ndarray, heatmap: torch.Tensor, alpha: float = 0.45) -> None:
    import cv2

    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    hm = heatmap.detach().float().cpu().squeeze().numpy()
    if hm.ndim != 2:
        raise ValueError(f"expected 2D heatmap after squeeze, got shape {hm.shape}")
    hm = hm - float(np.min(hm))
    denom = float(np.max(hm))
    if denom > 1e-8:
        hm = hm / denom
    hm_u8 = np.clip(hm * 255.0, 0, 255).astype(np.uint8)
    hm_u8 = cv2.resize(hm_u8, (rgb_u8.shape[1], rgb_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored_bgr = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.addWeighted(rgb_u8, 1.0 - float(alpha), colored_rgb, float(alpha), 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))


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


def _save_trajectory_3d_plot(out_dir: Path, steps: List[Dict[str, Any]]) -> None:
    """Save per-trajectory 3D path plot (UAV/target/jammers)."""
    if not steps:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] skip 3D trajectory plot: matplotlib unavailable ({e})")
        return

    def _collect_xyz(key: str) -> np.ndarray:
        pts: List[List[float]] = []
        for s in steps:
            v = s.get(key)
            if isinstance(v, dict) and all(k in v for k in ("x", "y", "z")):
                pts.append([float(v["x"]), float(v["y"]), float(v["z"])])
        if not pts:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32)

    uav_xyz = _collect_xyz("uav_position_after")
    target_xyz = _collect_xyz("target_position")

    jammer_series: Dict[str, List[List[float]]] = {}
    for s in steps:
        jam = s.get("jammers")
        if not isinstance(jam, dict):
            continue
        for did, pos in jam.items():
            if isinstance(pos, dict) and all(k in pos for k in ("x", "y", "z")):
                jammer_series.setdefault(str(did), []).append(
                    [float(pos["x"]), float(pos["y"]), float(pos["z"])]
                )

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if len(uav_xyz) > 0:
        ax.plot(uav_xyz[:, 0], uav_xyz[:, 1], uav_xyz[:, 2], color="tab:blue", linewidth=2.0, label="uav")
        ax.scatter(uav_xyz[0, 0], uav_xyz[0, 1], uav_xyz[0, 2], color="tab:blue", marker="o", s=25, label="uav_start")
        ax.scatter(uav_xyz[-1, 0], uav_xyz[-1, 1], uav_xyz[-1, 2], color="tab:blue", marker="x", s=35, label="uav_end")
    if len(target_xyz) > 0:
        ax.plot(
            target_xyz[:, 0],
            target_xyz[:, 1],
            target_xyz[:, 2],
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
            label="target",
        )
    for did, pts in sorted(jammer_series.items()):
        arr = np.asarray(pts, dtype=np.float32)
        if len(arr) == 0:
            continue
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], linewidth=1.2, alpha=0.85, label=f"jammer_{did}")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m, up)")
    ax.set_title("Online Eval 3D Trajectory")
    try:
        ax.legend(loc="best", fontsize=8)
    except Exception:
        pass
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "trajectory_3d.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------
# Executor setup
# -----------------------------


def build_executor(module, args: argparse.Namespace, scene_id: str):
    target_asset_name = None if args.random_target_asset else args.target_asset_name
    jammer_asset_name = None if args.random_jammer_asset else args.jammer_asset_name
    return module.TrajectoryExecutor(
        scene_id=scene_id,
        sim_server_host=args.sim_server_host,
        sim_server_port=args.sim_server_port,
        gpu_id=args.gpu_id,
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
    heatmap_overlay_dir = out_dir / "heatmap_overlay"
    attention_map_dir = out_dir / "last_transformer_attention_maps"
    predicted_video_dir = out_dir / "predicted_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rgb:
        rgb_out_dir.mkdir(parents=True, exist_ok=True)
    save_heatmap_overlay_enabled = bool(
        args.save_heatmap_overlay and cfg.use_target_visual_guidance and cfg.use_attention_heatmap
    )
    if save_heatmap_overlay_enabled:
        heatmap_overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_transformer_attention_maps:
        attention_map_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predicted_video:
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    _set_saved_assets_for_trajectory(executor, traj, args)
    _prepare_objects(executor, traj, args)

    # Initialize from the already collected Dataset trajectory, not from Plandataset.
    executor._initialize_simulation(
        np.asarray([traj.uav_start_airsim], dtype=np.float32),
        traj.target_traj_airsim,
        jammer_trajs_by_id=traj.jammer_trajs_airsim if traj.jammer_trajs_airsim else None,
    )
    executor._reset_collision_state()
    try:
        executor.client.enableApiControl(True, vehicle_name=executor.uav_vehicle_name)
        executor.client.armDisarm(True, vehicle_name=executor.uav_vehicle_name)
    except Exception:
        pass
    executor._ensure_uav_flying_state()
    init_yaw = math.atan2(
        float(traj.target_traj_airsim[0][1] - traj.uav_start_airsim[1]),
        float(traj.target_traj_airsim[0][0] - traj.uav_start_airsim[0]),
    )
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
    reference_image_t: Optional[torch.Tensor] = None
    reference_visual_tokens: Optional[torch.Tensor] = None
    reference_target_relative_t: Optional[torch.Tensor] = None
    target_belief_t: Optional[torch.Tensor] = None

    steps: List[Dict[str, Any]] = []
    distances: List[float] = []
    success = False
    success_step: Optional[int] = None
    collision = False
    visible_count = 0
    action_abs_err: List[float] = []
    action_mse: List[float] = []
    prev_uav_after_pos: Optional[np.ndarray] = None
    prev_uav_after_state: Optional[Dict[str, Any]] = None

    iterator: Iterable[int] = range(num_steps)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"online {traj.scene_id}/{traj.trajectory_name}", unit="step", dynamic_ncols=True)

    for t in iterator:
        target_t = traj.target_traj_airsim[t]
        executor.move_target_object(target_t)
        jammer_positions_now: Dict[str, np.ndarray] = {}
        if traj.jammer_trajs_airsim:
            for did, series in traj.jammer_trajs_airsim.items():
                if t >= len(series):
                    continue
                obj_name = executor._jammer_object_names_by_id.get(str(did), executor.jammer_object_name)
                asset_name = executor._jammer_asset_names_by_id.get(str(did), executor.jammer_asset_name)
                executor.move_named_object(obj_name, asset_name, executor.jammer_object_scale, series[t])
                try:
                    pos = executor.get_named_object_position(obj_name)
                    if pos is not None:
                        jammer_positions_now[str(did)] = pos
                except Exception:
                    pass
        executor._step_if_needed(1)

        # In pose-control evaluation, target/jammer scene stepping can let the
        # multirotor physics update the UAV before we apply the next action.
        # Keep the UAV exactly at the previous controlled pose so expert replay
        # is a check of action/coordinate consistency, not free-fall dynamics.
        if (
            args.control_mode == "pose"
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

        uav_state_before = executor.get_uav_state()
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

        rgb_img, _ = executor.get_camera_images()
        if rgb_img is None:
            raise RuntimeError(f"failed to capture RGB at step {t}")
        if args.save_rgb:
            save_rgb(rgb_out_dir / f"frame_{t:05d}.png", rgb_img)

        if cfg.use_target_belief_tracker and reference_target_relative_t is None:
            reference_target_np = rel_body / max(float(args.target_relative_scale), 1e-6)
            reference_target_relative_t = torch.from_numpy(reference_target_np.astype(np.float32)).view(1, -1).to(device)
            reference_image_t = rgb_to_model_tensor(rgb_img, transform, device)
            reference_visual_tokens = model.encode_reference_visual_tokens(reference_image_t)

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
        heatmap_overlay_relpath = None
        attention_map_relpath = None
        predicted_video_relpaths = None

        if args.replay_expert_action:
            if expert_action is None:
                break
            expert_action_norm = physical_action_to_norm(expert_action, cfg.max_vel, cfg.max_yaw_rate)
            action_norm = expert_action_norm.astype(np.float32)
            action_physical = np.asarray(expert_action, dtype=np.float32)
            action_source = "expert"
        else:
            image_t = rgb_to_model_tensor(rgb_img, transform, device)
            if cfg.use_wan22_encoders:
                text_tokens = torch.zeros(1, 1, dtype=torch.long, device=device)
                attention_mask = torch.ones_like(text_tokens)
            else:
                text_tokens, attention_mask = tokenize_instruction(tokenizer, instruction, cfg.text_context_length, device)
            target_relative_np = rel_body / max(float(args.target_relative_scale), 1e-6)
            target_relative_t = torch.from_numpy(target_relative_np.astype(np.float32)).view(1, -1).to(device)
            attention_heatmap_t = None
            if cfg.use_target_visual_guidance:
                raw_target_relative_t = torch.from_numpy(rel_body.astype(np.float32)).view(1, -1).to(device)
                if cfg.use_attention_heatmap:
                    attention_heatmap_t = make_attention_heatmap(
                        raw_target_relative_t,
                        image_hw=(image_t.shape[-2], image_t.shape[-1]),
                        fov_deg=cfg.visual_guidance_fov_deg,
                        sigma=cfg.attention_heatmap_sigma,
                    )
                    if save_heatmap_overlay_enabled:
                        heatmap_overlay_path = heatmap_overlay_dir / f"frame_{t:05d}.png"
                        save_heatmap_overlay(heatmap_overlay_path, rgb_img, attention_heatmap_t)
                        heatmap_overlay_relpath = str(heatmap_overlay_path.relative_to(out_dir))

            pred, rssm_state = model.act(
                image=image_t,
                text_tokens=text_tokens,
                target_relative=target_relative_t,
                prev_action=prev_action,
                rssm_state=rssm_state,
                attention_mask=attention_mask,
                attention_heatmap=attention_heatmap_t,
                prev_done=prev_done,
                deterministic=args.deterministic_action,
                num_steps=args.sampling_steps,
                instruction=instruction,
                save_transformer_attention=args.save_transformer_attention_maps,
                save_predicted_video=args.save_predicted_video,
                predicted_video_latent_frames=args.predicted_video_latent_frames,
                latent_mpc=args.use_latent_mpc,
                latent_mpc_candidate_count=args.latent_mpc_candidate_count,
                target_next_relative=None,
                reference_target_relative=reference_target_relative_t,
                reference_image=reference_image_t,
                reference_visual_tokens=reference_visual_tokens,
                target_belief=target_belief_t,
            )
            if "target_belief" in pred:
                target_belief_t = pred["target_belief"].detach()
            if args.save_transformer_attention_maps and "last_transformer_attention_map" in pred:
                attention_map_path = attention_map_dir / f"frame_{t:05d}"
                save_attention_map_overlay(attention_map_path, rgb_img, pred["last_transformer_attention_map"])
                attention_map_relpath = str(attention_map_path.with_suffix(".png").relative_to(out_dir))
            if args.save_predicted_video and "predicted_video_latents" in pred:
                decoded_video = model.image_encoder.decode_video_latents(pred["predicted_video_latents"])
                pred_frame_dir = predicted_video_dir / f"frame_{t:05d}"
                pred_frame_names = save_predicted_video_frames(pred_frame_dir, decoded_video)
                predicted_video_relpaths = [
                    str((pred_frame_dir / name).relative_to(out_dir)) for name in pred_frame_names
                ]
            action_norm = pred["action_norm"].detach().float().view(-1).cpu().numpy().astype(np.float32)
            action_physical = pred["action_physical"].detach().float().view(-1).cpu().numpy().astype(np.float32)

        if args.control_mode == "velocity":
            uav_state_after = apply_action_by_velocity(executor, action_physical, args.dt)
        else:
            uav_state_after = apply_action_by_pose(executor, action_physical, args.dt, args.max_step_norm)

        post_rel_body = compute_target_relative_body(executor, uav_state_after, target_now)
        distance = float(np.linalg.norm(post_rel_body))
        distances.append(distance)
        visible = is_visible_by_geometry(post_rel_body, fov_deg=args.fov_deg)
        if visible:
            visible_count += 1

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
            "jammers": {
                str(did): _airsim_xyz_to_dataset(pos) for did, pos in jammer_positions_now.items()
            },
        }
        if heatmap_overlay_relpath is not None:
            step_record["heatmap_overlay"] = heatmap_overlay_relpath
        if attention_map_relpath is not None:
            step_record["last_transformer_attention_map"] = attention_map_relpath
        if predicted_video_relpaths is not None:
            step_record["predicted_video_frames"] = predicted_video_relpaths
        if pred is not None and "target_belief_confidence" in pred:
            step_record["target_belief_confidence"] = (
                pred["target_belief_confidence"].detach().float().view(-1).cpu().numpy().astype(float).tolist()
            )
        if pred is not None and "target_belief_entropy" in pred:
            step_record["target_belief_entropy"] = (
                pred["target_belief_entropy"].detach().float().view(-1).cpu().numpy().astype(float).tolist()
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
        if pred is not None and "latent_mpc_scores" in pred:
            selected_mpc = int(pred["latent_mpc_selected"].detach().view(-1)[0].cpu().item())
            mpc_scores = pred["latent_mpc_scores"].detach().float().view(-1).cpu().numpy()
            step_record["latent_mpc"] = {
                "selected": selected_mpc,
                "scores": mpc_scores.astype(float).tolist(),
                "selected_score": float(mpc_scores[selected_mpc]),
            }
            for key in (
                "latent_mpc_distance",
                "latent_mpc_smooth",
                "latent_mpc_action_effort",
                "latent_mpc_visual_cost",
                "latent_mpc_visual_change",
            ):
                if key in pred:
                    values = pred[key].detach().float().view(-1).cpu().numpy()
                    step_record["latent_mpc"][key.replace("latent_mpc_", "")] = values.astype(float).tolist()
        steps.append(step_record)

        prev_action = torch.from_numpy(action_norm).view(1, -1).to(device).float()

        # The requested success metric is an end-of-trajectory metric: only the
        # final frame counts. Therefore, do not mark prev_done or stop early just
        # because an intermediate frame is within the capture radius.
        prev_done = torch.tensor([1.0 if collision_now else 0.0], device=device)

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
                src=action_source,
                expert=expert_act,
                expert_src=expert_action_source,
            )

        prev_uav_after_pos = np.asarray(uav_state_after["position"], dtype=np.float32).copy()
        prev_uav_after_state = {
            "position": np.asarray(uav_state_after["position"], dtype=np.float32).copy(),
            "orientation": np.asarray(uav_state_after["orientation"], dtype=np.float32).copy(),
        }

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
        "control_mode": args.control_mode,
        "replay_expert_action": bool(args.replay_expert_action),
        "hold_uav_pose_during_scene_step": bool(args.hold_uav_pose_during_scene_step),
        "capture_distance": args.capture_distance,
        "require_visibility_for_success": args.require_visibility_for_success,
    }

    _dump_json(out_dir / "online_rollout.json", {"summary": summary, "steps": steps})
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
    parser.add_argument("--dataset-root", type=str, required=True, help="Collected Dataset root, e.g. /data1/ysq/Worldmodel/Dataset")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--executor-script", type=str, default="/data1/ysq/Worldmodel/code/src/executor/trajectory_executor.py")
    parser.add_argument("--start-sim-server", action="store_true", default=False, help="Start sim_server.py inside this script after the model is loaded.")
    parser.add_argument("--sim-server-script", type=str, default="/data1/ysq/Worldmodel/code/src/executor/sim_server.py")
    parser.add_argument("--sim-server-root-path", type=str, default="/data1/ysq/Worldmodel")
    parser.add_argument("--sim-server-log", type=str, default="/data1/ysq/Worldmodel/online_eval_teacher_dit/sim_server_30000.log")
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

    # Model config
    parser.add_argument("--tokenizer-name", type=str, default=LOCAL_CLIP_MODEL_PATH)
    parser.add_argument("--clip-text-model-name", type=str, default=LOCAL_CLIP_MODEL_PATH)
    parser.add_argument("--dinov2-model-name", type=str, default=LOCAL_DINOV2_MODEL_PATH)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--target-relative-dim", type=int, default=3)
    parser.add_argument("--target-relative-scale", type=float, default=1.0, help="Use 1.0 if training used raw target_position_in_body_frame; set 100.0 only if your builder normalized by 100.")
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=20)
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
    parser.add_argument("--use-target-visual-guidance", type=_str2bool, default=None)
    parser.add_argument("--use-attention-heatmap", type=_str2bool, default=None)
    parser.add_argument("--visual-guidance-fov-deg", type=float, default=None)
    parser.add_argument("--attention-heatmap-sigma", type=float, default=None)
    parser.add_argument("--use-target-belief-tracker", type=_str2bool, default=None)
    parser.add_argument("--target-belief-token-scale", type=float, default=None)
    parser.add_argument("--target-belief-update-rate", type=float, default=None)
    parser.add_argument("--target-belief-min-confidence", type=float, default=None)
    parser.add_argument("--target-belief-temperature", type=float, default=None)
    parser.add_argument("--target-belief-loss-weight", type=float, default=None)
    parser.add_argument("--target-belief-motion-weight", type=float, default=None)
    parser.add_argument("--target-belief-update-sharpness", type=float, default=None)
    parser.add_argument("--use-latent-mpc", type=_str2bool, default=None)
    parser.add_argument("--latent-mpc-candidate-count", type=int, default=None)
    parser.add_argument("--latent-mpc-distance-weight", type=float, default=None)
    parser.add_argument("--latent-mpc-smooth-weight", type=float, default=None)
    parser.add_argument("--latent-mpc-action-weight", type=float, default=None)
    parser.add_argument("--latent-mpc-visual-weight", type=float, default=None)
    parser.add_argument("--latent-mpc-latent-frames", type=int, default=None)
    parser.add_argument("--latent-mpc-video-sampling-steps", type=int, default=None)
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
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--uav-vehicle-name", type=str, default="Drone_1")
    parser.add_argument("--target-object-name", type=str, default="UAV1")
    parser.add_argument("--target-asset-name", type=str, default="UAV1")
    parser.add_argument("--jammer-object-name", type=str, default="JammerUAV")
    parser.add_argument("--jammer-asset-name", type=str, default="UAV1")
    parser.add_argument("--camera-name", type=str, default="0")
    parser.add_argument("--target-scale", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--jammer-scale", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--disable-jammer", action="store_true", default=False)
    parser.add_argument("--reuse-saved-assets", action="store_true", default=True)
    parser.add_argument("--do-not-reuse-saved-assets", action="store_false", dest="reuse_saved_assets")
    parser.add_argument("--random-target-asset", action="store_true", default=False)
    parser.add_argument("--random-jammer-asset", action="store_true", default=False)

    # Online rollout config
    parser.add_argument("--control-mode", type=str, default="pose", choices=["pose", "velocity"])
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
    parser.add_argument("--save-heatmap-overlay", action="store_true", default=True)
    parser.add_argument("--no-save-heatmap-overlay", action="store_false", dest="save_heatmap_overlay")
    parser.add_argument("--save-transformer-attention-maps", action="store_true", default=False)
    parser.add_argument("--no-save-transformer-attention-maps", action="store_false", dest="save_transformer_attention_maps")
    parser.add_argument("--save-predicted-video", action="store_true", default=False)
    parser.add_argument("--no-save-predicted-video", action="store_false", dest="save_predicted_video")
    parser.add_argument("--predicted-video-latent-frames", type=int, default=3)

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
    seed_everything(args.seed)

    # Force a single evaluator process and a single AirSim scene.
    # Keep DAGGER_MULTI_WORKER=1 even in single-GPU mode: in your TrajectoryExecutor,
    # this flag prevents connect() from proactively calling close_scenes before every
    # connection/retry. It does NOT create extra evaluator processes here.
    os.environ["DAGGER_MULTI_WORKER"] = "1"

    scene_ids = _parse_scene_list(args.scene_list)
    if len(scene_ids) != 1:
        raise ValueError(
            "This online evaluator is single-GPU/single-process/single-scene only. "
            f"Please pass exactly one scene in --scene-list, got: {scene_ids}"
        )

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("[mode] single GPU / single Python evaluator / single AirSim scene")
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

    # IMPORTANT: start sim_server/AirSim only after model loading has finished.
    # This avoids launching UE while DINOv2/CLIP/checkpoint tensors are being loaded.
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

    all_summaries: List[Dict[str, Any]] = []
    current_scene = None
    executor = None

    try:
        for idx, dataset_dir in enumerate(traj_dirs, start=1):
            scene_id = dataset_dir.parent.name
            traj = load_online_trajectory(dataset_dir, scene_id)
            if current_scene != scene_id or executor is None:
                cleanup_executor(executor)
                executor = build_executor(executor_module, args, scene_id)
                current_scene = scene_id

            print("=" * 100)
            print(f"online eval {scene_id}/{traj.trajectory_name}")
            print(f"dataset trajectory: {dataset_dir}")
            print(f"frames={traj.num_frames}, target_asset={traj.target_asset_name}, jammers={list(traj.jammer_trajs_airsim.keys())}")
            summary = run_online_trajectory(model, cfg, tokenizer, transform, executor, traj, args, device)
            all_summaries.append(summary)
            _dump_json(Path(args.output_dir) / "summary_partial.json", {"summaries": all_summaries})

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
        _dump_json(Path(args.output_dir) / "summary.json", agg)
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
