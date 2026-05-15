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

from model.config import ModelConfig
from model.model import PrivilegedTeacherWorldModelDiT

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
                return out
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
                return out

    frames = uav_payload.get("trajectory")
    if isinstance(frames, list):
        out = []
        for frame in frames:
            if isinstance(frame, dict):
                text = frame.get("instruction") or frame.get("text")
                if isinstance(text, str):
                    out.append(text)
        if out:
            return out
    return None


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
    raw_cfg = ckpt.get("cfg", {}) or {}
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k in field_names}
    cfg_kwargs.update(
        {
            "image_size": args.image_size,
            "dinov2_model_name": args.dinov2_model_name,
            "dinov2_freeze": args.freeze_dinov2,
            "clip_text_model_name": args.clip_text_model_name,
            "clip_text_freeze": args.freeze_clip_text,
            "privileged_dim": args.privileged_dim,
            "action_dim": args.action_dim,
            "action_sampling_steps": args.sampling_steps,
            "max_vel": args.max_vel,
            "max_yaw_rate": args.max_yaw_rate,
            "max_speed_norm": args.max_speed_norm,
        }
    )
    return ModelConfig(**cfg_kwargs)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_model(args: argparse.Namespace, device: torch.device) -> Tuple[PrivilegedTeacherWorldModelDiT, ModelConfig]:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _make_cfg_from_checkpoint(ckpt, args)
    model = PrivilegedTeacherWorldModelDiT(cfg).to(device)
    missing, unexpected = model.load_state_dict(_strip_module_prefix(ckpt["model"]), strict=False)
    if missing:
        print(f"[warn] missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys when loading checkpoint: {unexpected}")
    model.eval()
    print(f"[model] loaded checkpoint: {ckpt_path}")
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
    body_ned = np.asarray([action[0], action[1], -action[2]], dtype=np.float32)
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
    ok, verify_pos, pos_error, err_xy, err_z = executor._set_vehicle_pose_paused(
        float(new_pos[0]),
        float(new_pos[1]),
        float(new_pos[2]),
        quat,
        retries=3,
        tol_xy=0.8,
        tol_z=0.8,
    )
    if not ok:
        raise RuntimeError(
            f"pose action failed: target=({new_pos[0]:.2f},{new_pos[1]:.2f},{new_pos[2]:.2f}), "
            f"actual=({verify_pos[0]:.2f},{verify_pos[1]:.2f},{verify_pos[2]:.2f}), "
            f"err={pos_error:.2f}m xy={err_xy:.2f} z={err_z:.2f}"
        )
    executor._step_if_needed(1)
    return executor.get_uav_state()


def apply_action_by_velocity(executor, action_physical: np.ndarray, dt: float) -> Dict[str, Any]:
    import airsim

    action = np.asarray(action_physical, dtype=np.float32)
    # AirSim body-frame z is down, while the model/data action z is up.
    executor._safe_sim_pause(False)
    executor.client.moveByVelocityBodyFrameAsync(
        float(action[0]),
        float(action[1]),
        float(-action[2]),
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
    model: PrivilegedTeacherWorldModelDiT,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rgb:
        rgb_out_dir.mkdir(parents=True, exist_ok=True)

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

    num_steps = traj.num_frames if args.max_steps <= 0 else min(traj.num_frames, args.max_steps)
    rssm_state = None
    prev_action = torch.zeros(1, cfg.action_dim, device=device, dtype=torch.float32)
    prev_done = torch.zeros(1, device=device, dtype=torch.float32)

    steps: List[Dict[str, Any]] = []
    distances: List[float] = []
    min_distance = float("inf")
    success = False
    success_step: Optional[int] = None
    collision = False
    visible_count = 0
    action_abs_err: List[float] = []
    action_mse: List[float] = []
    prev_uav_after_pos: Optional[np.ndarray] = None

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

        if traj.saved_instructions and t < len(traj.saved_instructions):
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

        image_t = rgb_to_model_tensor(rgb_img, transform, device)
        text_tokens, attention_mask = tokenize_instruction(tokenizer, instruction, cfg.text_context_length, device)
        privileged_np = rel_body / max(float(args.privileged_scale), 1e-6)
        privileged_t = torch.from_numpy(privileged_np.astype(np.float32)).view(1, -1).to(device)

        pred, rssm_state = model.act(
            image=image_t,
            text_tokens=text_tokens,
            privileged=privileged_t,
            prev_action=prev_action,
            rssm_state=rssm_state,
            attention_mask=attention_mask,
            prev_done=prev_done,
            deterministic=args.deterministic_action,
            num_steps=args.sampling_steps,
        )
        action_norm = pred["action_norm"].detach().float().view(-1).cpu().numpy().astype(np.float32)
        action_physical = pred["action_physical"].detach().float().view(-1).cpu().numpy().astype(np.float32)
        pred_reward = float(pred["reward"].detach().float().view(-1)[0].cpu()) if "reward" in pred else None

        if args.control_mode == "velocity":
            uav_state_after = apply_action_by_velocity(executor, action_physical, args.dt)
        else:
            uav_state_after = apply_action_by_pose(executor, action_physical, args.dt, args.max_step_norm)

        post_rel_body = compute_target_relative_body(executor, uav_state_after, target_now)
        distance = float(np.linalg.norm(post_rel_body))
        distances.append(distance)
        min_distance = min(min_distance, distance)
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

        expert_action = traj.expert_action_physical[t] if t < len(traj.expert_action_physical) else None
        expert_action_norm = None
        if expert_action is not None:
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
            "action_norm": action_norm.astype(float).tolist(),
            "action_physical": action_physical.astype(float).tolist(),
            "expert_action_physical": None if expert_action is None else expert_action.astype(float).tolist(),
            "expert_action_norm": None if expert_action_norm is None else expert_action_norm.astype(float).tolist(),
            "pred_reward": pred_reward,
            "collision": bool(collision_now),
            "collision_before_action": bool(collision_before_action),
            "collision_after_action": bool(collision_after_action),
            "collision_info_before_action": _collision_info_to_dict(collision_info_before_action),
            "collision_info_after_action": _collision_info_to_dict(collision_info_after_action),
            "jammers": {
                str(did): _airsim_xyz_to_dataset(pos) for did, pos in jammer_positions_now.items()
            },
        }
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
                pred=pred_act,
                expert=expert_act,
            )

        prev_uav_after_pos = np.asarray(uav_state_after["position"], dtype=np.float32).copy()

        if collision_now and args.stop_on_collision:
            break

    # Success criterion: only the final frame counts.
    # Success means: final target is within capture_distance and visible.
    final_distance = float(distances[-1]) if distances else float("inf")
    final_visible = bool(steps[-1].get("visible_by_geometry", False)) if steps else False
    success = bool(final_distance <= float(args.capture_distance) and final_visible)
    success_step = (len(steps) - 1) if (success and steps) else None

    summary = {
        "scene_id": traj.scene_id,
        "trajectory_name": traj.trajectory_name,
        "dataset_dir": str(traj.dataset_dir),
        "num_steps": len(steps),
        "success": bool(success),
        "success_step": success_step,
        "collision": bool(collision),
        "final_distance": final_distance if distances else None,
        "final_visible_by_geometry": bool(final_visible),
        "success_criterion": "final_distance <= capture_distance and final_visible_by_geometry",
        "min_distance": float(min_distance) if distances else None,
        "mean_distance": float(np.mean(distances)) if distances else None,
        "visible_ratio_geometry": float(visible_count / max(len(steps), 1)),
        "mean_action_abs_error_physical": float(np.mean(action_abs_err)) if action_abs_err else None,
        "rmse_action_physical": float(math.sqrt(np.mean(action_mse))) if action_mse else None,
        "target_asset_name": getattr(executor, "target_asset_name", None),
        "jammer_asset_names": getattr(executor, "_jammer_asset_names_by_id", {}),
        "control_mode": args.control_mode,
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
    parser.add_argument("--privileged-dim", type=int, default=3)
    parser.add_argument("--privileged-scale", type=float, default=1.0, help="Use 1.0 if training used raw target_position_in_body_frame; set 100.0 only if your builder normalized by 100.")
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--max-vel", type=float, default=DEFAULT_MODEL_CFG.max_vel)
    parser.add_argument("--max-yaw-rate", type=float, default=DEFAULT_MODEL_CFG.max_yaw_rate)
    parser.add_argument("--max-speed-norm", type=float, default=DEFAULT_MODEL_CFG.max_speed_norm)
    parser.add_argument("--freeze-dinov2", action="store_true", default=True)
    parser.add_argument("--finetune-dinov2", action="store_false", dest="freeze_dinov2")
    parser.add_argument("--freeze-clip-text", action="store_true", default=True)
    parser.add_argument("--finetune-clip-text", action="store_false", dest="freeze_clip_text")
    parser.add_argument("--deterministic-action", action="store_true", default=True)
    parser.add_argument("--stochastic-action", action="store_false", dest="deterministic_action")

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
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-step-norm", type=float, default=1.0, help="Safety clamp for one pose-control step. <=0 disables clamp.")
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

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
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

    if CLIPTokenizerFast is None:
        raise ImportError("transformers.CLIPTokenizerFast is required for online instruction tokenization")
    tokenizer = CLIPTokenizerFast.from_pretrained(args.tokenizer_name, local_files_only=True)

    model, cfg = load_model(args, device)

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
        min_distances = [float(s["min_distance"]) for s in all_summaries if s.get("min_distance") is not None]
        final_distances = [float(s["final_distance"]) for s in all_summaries if s.get("final_distance") is not None]
        agg = {
            "num_trajectories": len(all_summaries),
            "success_rate": float(np.mean(success_values)) if success_values else None,
            "collision_rate": float(np.mean(collision_values)) if collision_values else None,
            "mean_min_distance": float(np.mean(min_distances)) if min_distances else None,
            "mean_final_distance": float(np.mean(final_distances)) if final_distances else None,
            "args": vars(args),
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
