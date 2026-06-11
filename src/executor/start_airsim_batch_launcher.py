#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接基于 TrajectoryExecutor 批量执行 AirSim 轨迹，不再依赖 batch_generate_dataset.py。

默认行为：
1. 直接导入执行器脚本（默认 /mnt/data/airsim_trajectory_executor.py）
2. 按 scene -> gpu 轮询分配
3. 每个 GPU 一个 worker，顺序处理其分配到的 scene
4. 每个 scene 下批量执行所有轨迹文件

可直接运行：
    python /mnt/data/start_airsim_batch_launcher.py
"""

import os
import re
import sys
import argparse
import importlib.util
import json
import math
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_EXECUTOR_SCRIPT = "/data1/ysq/Worldmodel/code/src/executor/trajectory_executor.py"
DEFAULT_TRAJECTORY_DIR = "/data1/ysq/Worldmodel/Plandataset"
DEFAULT_DATASET_BASE_DIR = "/data1/ysq/Worldmodel/Dataset"
DEFAULT_SCENES = [f"City_{i}" for i in range(1, 31)]
DEFAULT_GPUS = [1]
DEFAULT_SIM_SERVER_HOST = "127.0.0.1"
DEFAULT_SIM_SERVER_PORT = 30000
DEFAULT_SCENE_INDEX = 1
DEFAULT_UAV_VEHICLE_NAME = "Drone_1"
DEFAULT_TARGET_OBJECT_NAME = "UAV1"
DEFAULT_TARGET_ASSET_NAME = "UAV1"
DEFAULT_CAMERA_NAME = "0"
DEFAULT_TARGET_SCALE = (1.0, 1.0, 1.0)
DEFAULT_TRAJECTORY_PATTERN = "trajectory_*_uav.json"
DEFAULT_MULTI_WORKER = True
DEFAULT_SAVE_DATASET = True
DEFAULT_SKIP_HOVER = True
DEFAULT_RANDOM_TARGET_ASSET = True
DEFAULT_MAX_RETRIES = 5
DEFAULT_JUMP_THRESHOLD = 1.5
DEFAULT_RECOVER_ABNORMAL_JUMP = True
DEFAULT_JAMMER_ENABLED = True
DEFAULT_JAMMER_OBJECT_NAME = "JammerUAV"
DEFAULT_JAMMER_ASSET_NAME = "UAV1"
DEFAULT_RANDOM_JAMMER_ASSET = True
DEFAULT_JAMMER_SCALE = (1.0, 1.0, 1.0)
DEFAULT_CHECK_DATASET_ANOMALIES = True
DEFAULT_DELETE_BAD_TRAJECTORIES = True
DEFAULT_DELETE_STEP_THRESHOLD = 1.5
DEFAULT_CHECK_HEAD = 0
DEFAULT_REPAIR_UNTIL_CLEAN = True
DEFAULT_MAX_REPAIR_ROUNDS = 5
DEFAULT_EPISODE_INSTRUCTION = (
    "The target is the black UAV initially located near the image center. "
    "Keep tracking the same UAV throughout the episode."
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a UAV visual pursuit agent operating in the Body Frame: "
    "X-forward, Y-right, Z-down. Track and intercept the true target UAV "
    "from FPV images and user instructions. Ignore distractor UAVs even if "
    "they appear closer or cross the view. Keep the target centered, pursue "
    "smoothly, and avoid obstacles or collisions."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direct launcher for AirSim trajectory batch execution"
    )

    parser.add_argument(
        "--executor-script",
        type=str,
        default=DEFAULT_EXECUTOR_SCRIPT,
        help="TrajectoryExecutor 所在脚本路径",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default=DEFAULT_TRAJECTORY_DIR,
        help="轨迹根目录",
    )
    parser.add_argument(
        "--trajectory-pattern",
        type=str,
        default=DEFAULT_TRAJECTORY_PATTERN,
        help="轨迹文件匹配模式，通常是 trajectory_*_uav.json",
    )
    parser.add_argument(
        "--dataset-base-dir",
        type=str,
        default=DEFAULT_DATASET_BASE_DIR,
        help="数据集输出根目录",
    )

    parser.add_argument(
        "--scene-id",
        nargs="+",
        default=DEFAULT_SCENES,
        help="场景列表，例如 City_1 City_2 City_3 City_4；默认 City_1..City_30",
    )
    parser.add_argument(
        "--gpu-id",
        nargs="+",
        default=[str(x) for x in DEFAULT_GPUS],
        help="GPU 列表，例如 0 1",
    )

    parser.add_argument("--sim-server-host", type=str, default=DEFAULT_SIM_SERVER_HOST)
    parser.add_argument("--sim-server-port", type=int, default=DEFAULT_SIM_SERVER_PORT)
    parser.add_argument("--scene-index", type=int, default=DEFAULT_SCENE_INDEX)
    parser.add_argument("--uav-vehicle-name", type=str, default=DEFAULT_UAV_VEHICLE_NAME)
    parser.add_argument("--target-object-name", type=str, default=DEFAULT_TARGET_OBJECT_NAME)
    parser.add_argument("--target-asset-name", type=str, default=DEFAULT_TARGET_ASSET_NAME)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--jammer-object-name", type=str, default=DEFAULT_JAMMER_OBJECT_NAME)
    parser.add_argument("--jammer-asset-name", type=str, default=DEFAULT_JAMMER_ASSET_NAME)
    parser.add_argument(
        "--target-scale",
        nargs=3,
        type=float,
        default=list(DEFAULT_TARGET_SCALE),
        metavar=("SX", "SY", "SZ"),
        help="目标对象缩放",
    )

    parser.add_argument(
        "--jammer-scale",
        nargs=3,
        type=float,
        default=list(DEFAULT_JAMMER_SCALE),
        metavar=("SX", "SY", "SZ"),
        help="干扰机对象缩放",
    )

    parser.add_argument(
        "--enable-jammer",
        dest="jammer_enabled",
        action="store_true",
        default=DEFAULT_JAMMER_ENABLED,
        help="启用干扰机轨迹读取和同步运动",
    )
    parser.add_argument(
        "--disable-jammer",
        dest="jammer_enabled",
        action="store_false",
        help="禁用干扰机",
    )
    parser.add_argument(
        "--random-jammer-asset",
        dest="random_jammer_asset",
        action="store_true",
        default=DEFAULT_RANDOM_JAMMER_ASSET,
        help="每条轨迹随机从 UAV1-UAV20 里挑选干扰机资产",
    )
    parser.add_argument(
        "--fixed-jammer-asset",
        dest="random_jammer_asset",
        action="store_false",
        help="固定使用 --jammer-asset-name 指定的干扰机资产",
    )

    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--jump-threshold", type=float, default=DEFAULT_JUMP_THRESHOLD)
    parser.add_argument(
        "--recover-abnormal-jump",
        dest="recover_abnormal_jump",
        action="store_true",
        default=DEFAULT_RECOVER_ABNORMAL_JUMP,
        help="AirSim 当前位姿异常跑飞时，强制恢复到 planner 本帧位姿继续采集",
    )
    parser.add_argument(
        "--abort-on-abnormal-jump",
        dest="recover_abnormal_jump",
        action="store_false",
        help="保持旧行为：检测到 abnormal jump 时中止当前轨迹并重试",
    )

    parser.add_argument(
        "--multi-worker",
        dest="multi_worker",
        action="store_true",
        default=DEFAULT_MULTI_WORKER,
        help="多个 GPU 并行处理",
    )
    parser.add_argument(
        "--no-multi-worker",
        dest="multi_worker",
        action="store_false",
        help="关闭多 worker，并串行执行",
    )

    parser.add_argument(
        "--save-dataset",
        dest="save_dataset",
        action="store_true",
        default=DEFAULT_SAVE_DATASET,
        help="保存数据集",
    )
    parser.add_argument(
        "--no-save-dataset",
        dest="save_dataset",
        action="store_false",
        help="不保存数据集",
    )

    parser.add_argument(
        "--skip-hover",
        dest="skip_hover",
        action="store_true",
        default=DEFAULT_SKIP_HOVER,
        help="轨迹执行完成后不进入悬停死循环（批处理建议开启）",
    )
    parser.add_argument(
        "--no-skip-hover",
        dest="skip_hover",
        action="store_false",
        help="轨迹执行完成后进入悬停状态",
    )

    parser.add_argument(
        "--random-target-asset",
        dest="random_target_asset",
        action="store_true",
        default=DEFAULT_RANDOM_TARGET_ASSET,
        help="每条轨迹随机从 UAV1-UAV20 里挑选目标资产",
    )
    parser.add_argument(
        "--fixed-target-asset",
        dest="random_target_asset",
        action="store_false",
        help="固定使用 --target-asset-name 指定的目标资产",
    )

    parser.add_argument(
        "--check-dataset-anomalies",
        dest="check_dataset_anomalies",
        action="store_true",
        default=DEFAULT_CHECK_DATASET_ANOMALIES,
        help="采集完成后检查 Dataset 异常并生成报告",
    )
    parser.add_argument(
        "--no-check-dataset-anomalies",
        dest="check_dataset_anomalies",
        action="store_false",
        help="采集完成后不运行 Dataset 异常检查",
    )
    parser.add_argument(
        "--delete-step-threshold",
        type=float,
        default=DEFAULT_DELETE_STEP_THRESHOLD,
        help="异常检查时，UAV 单步距离超过该值则判定为可删除坏轨迹",
    )
    parser.add_argument(
        "--delete-bad-trajectories",
        dest="delete_bad_trajectories",
        action="store_true",
        default=DEFAULT_DELETE_BAD_TRAJECTORIES,
        help="异常检查时自动删除超过阈值的坏轨迹",
    )
    parser.add_argument(
        "--no-delete-bad-trajectories",
        dest="delete_bad_trajectories",
        action="store_false",
        help="异常检查只生成报告，不删除坏轨迹",
    )
    parser.add_argument(
        "--anomaly-output",
        type=str,
        default="",
        help="异常检查报告输出路径；为空则写到 Dataset/dataset_anomaly_report_时间.json",
    )
    parser.add_argument(
        "--anomaly-head",
        type=int,
        default=DEFAULT_CHECK_HEAD,
        help="每个 scene 只检查前 N 条轨迹；0 表示检查全部",
    )
    parser.add_argument(
        "--repair-until-clean",
        dest="repair_until_clean",
        action="store_true",
        default=DEFAULT_REPAIR_UNTIL_CLEAN,
        help="发现缺失/异常/被删除轨迹后，在同一次启动中自动补采直到数据集干净",
    )
    parser.add_argument(
        "--no-repair-until-clean",
        dest="repair_until_clean",
        action="store_false",
        help="只执行一轮采集和检查，不自动补采被删除或缺失的轨迹",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=DEFAULT_MAX_REPAIR_ROUNDS,
        help="自动补采最多执行多少轮；每轮会跳过已完整轨迹，只重试缺失/被删轨迹",
    )

    return parser.parse_args()


def dynamic_import_module(py_file: Path):
    py_file = py_file.resolve()
    if not py_file.exists():
        raise FileNotFoundError(f"执行器脚本不存在: {py_file}")

    module_name = py_file.stem + "_direct_launcher_loaded"
    spec = importlib.util.spec_from_file_location(module_name, str(py_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入执行器脚本: {py_file}")

    module = importlib.util.module_from_spec(spec)

    parent_dir = str(py_file.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    project_root = str(py_file.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    spec.loader.exec_module(module)
    return module


def normalize_gpu_ids(gpu_ids: Sequence[str]):
    parsed = []
    for g in gpu_ids:
        try:
            parsed.append(int(g))
        except ValueError:
            parsed.append(g)
    return parsed


def natural_key(path_obj: Path):
    parts = re.split(r"(\d+)", path_obj.stem)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return key


def discover_trajectory_files(trajectory_root: Path, scene_id: str, pattern: str) -> List[Path]:
    trajectory_root = trajectory_root.resolve()
    collected: List[Path] = []

    # Scene directories in datasets are often inconsistent in casing (e.g. City_1 vs city_1).
    # Prefer an exact match, but fall back to case-insensitive directory lookup.
    search_dirs: List[Path] = []
    direct_scene_dir = trajectory_root / scene_id
    if direct_scene_dir.exists():
        search_dirs.append(direct_scene_dir)
    else:
        scene_id_lower = scene_id.lower()
        try:
            for d in trajectory_root.iterdir():
                if d.is_dir() and d.name.lower() == scene_id_lower:
                    search_dirs.append(d)
                    break
        except Exception:
            pass

    search_dirs.append(trajectory_root)

    candidate_patterns = [pattern]
    if pattern != "*_uav.json":
        candidate_patterns.append("*_uav.json")
    # Common dataset naming: trajectory_0001.json (no *_uav suffix)
    if pattern != "trajectory_*.json":
        candidate_patterns.append("trajectory_*.json")

    for base_dir in search_dirs:
        for pat in candidate_patterns:
            try:
                for p in base_dir.rglob(pat):
                    if not p.is_file():
                        continue
                    if base_dir == trajectory_root:
                        # 根目录递归搜索时，只保留路径里包含 scene_id 的结果，避免串场景
                        scene_id_lower = scene_id.lower()
                        parts_lower = [x.lower() for x in p.parts]
                        if scene_id_lower not in parts_lower and p.parent.name.lower() != scene_id_lower:
                            continue
                    collected.append(p)
            except Exception:
                continue
        if collected:
            break

    # 去重 + 排序
    unique_files = sorted({p.resolve() for p in collected}, key=natural_key)
    return unique_files


def assign_scenes_to_gpus(scene_ids: Sequence[str], gpu_ids: Sequence[int]) -> Dict[int, List[str]]:
    if not gpu_ids:
        raise ValueError("gpu_id 不能为空")

    assignment: Dict[int, List[str]] = {gpu: [] for gpu in gpu_ids}
    for idx, scene in enumerate(scene_ids):
        gpu = gpu_ids[idx % len(gpu_ids)]
        assignment[gpu].append(scene)
    return {gpu: scenes for gpu, scenes in assignment.items() if scenes}


@dataclass
class TrajectoryIssue:
    scene: str
    trajectory: str
    severity: str
    issue_type: str
    detail: str


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _xyz_from_any(obj: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(obj, dict) and all(k in obj for k in ("x", "y", "z")):
        xyz = (obj["x"], obj["y"], obj["z"])
    elif isinstance(obj, (list, tuple)) and len(obj) >= 3:
        xyz = (obj[0], obj[1], obj[2])
    else:
        return None
    if not all(_is_finite_number(v) for v in xyz):
        return None
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def _check_frame_numeric(frame: Dict[str, Any], idx: int, scene: str, traj: str, issues: List[TrajectoryIssue]) -> None:
    vec_keys = [
        "uav_position",
        "target_position",
        "target_position_in_body_frame",
        "velocity_in_body_frame",
        "next_target_position",
        "relative_position",
        "jammer_position",
        "jammer_relative_position",
        "jammer_position_in_body_frame",
    ]
    scalar_keys = ["distance", "yaw_rate", "jammer_distance"]

    for key in vec_keys:
        if key in frame and frame[key] is not None and _xyz_from_any(frame[key]) is None:
            issues.append(
                TrajectoryIssue(
                    scene,
                    traj,
                    "error",
                    "invalid_vector",
                    f"frame[{idx}] key='{key}' is not a valid finite xyz vector",
                )
            )
    for key in scalar_keys:
        if key in frame and frame[key] is not None and not _is_finite_number(frame[key]):
            issues.append(
                TrajectoryIssue(
                    scene,
                    traj,
                    "error",
                    "invalid_scalar",
                    f"frame[{idx}] key='{key}' is not a finite number",
                )
            )


def _parse_frame_index(name: str) -> Optional[int]:
    stem = Path(name).stem
    if not stem.startswith("frame_"):
        return None
    value = stem.replace("frame_", "", 1)
    return int(value) if value.isdigit() else None


def _check_rgb_sequence(rgb_files: List[Path], scene: str, traj: str, issues: List[TrajectoryIssue]) -> None:
    if not rgb_files:
        issues.append(TrajectoryIssue(scene, traj, "error", "missing_rgb", "rgb directory has no frame_*.png files"))
        return
    idxs = sorted(i for i in (_parse_frame_index(p.name) for p in rgb_files) if i is not None)
    if not idxs:
        issues.append(TrajectoryIssue(scene, traj, "error", "invalid_rgb_names", "no valid frame index found in rgb filenames"))
        return
    gaps = []
    for a, b in zip(idxs[:-1], idxs[1:]):
        if b != a + 1:
            gaps.append((a, b))
            if len(gaps) >= 5:
                break
    if gaps:
        issues.append(
            TrajectoryIssue(
                scene,
                traj,
                "warning",
                "rgb_index_gap",
                f"rgb frame indices are non-contiguous, first gaps: {gaps}",
            )
        )


def _check_uav_step_distance(
    frames: List[Any],
    scene: str,
    traj: str,
    expected_step: float,
    step_tol: float,
    hard_delete_threshold: float,
    issues: List[TrajectoryIssue],
    stats: Dict[str, Any],
) -> bool:
    if len(frames) < 2:
        return False
    dists: List[float] = []
    bad_steps: List[Tuple[int, float]] = []
    prev_pos: Optional[Tuple[float, float, float]] = None
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            prev_pos = None
            continue
        pos = _xyz_from_any(frame.get("uav_position"))
        if pos is None:
            prev_pos = None
            continue
        if prev_pos is not None:
            dist = math.dist(prev_pos, pos)
            dists.append(dist)
            is_final_short_step = i == len(frames) - 1 and dist < expected_step
            if (not is_final_short_step) and abs(dist - expected_step) > step_tol:
                bad_steps.append((i, dist))
        prev_pos = pos

    if not dists:
        return False

    stats["uav_step_dist_mean"] = float(sum(dists) / len(dists))
    stats["uav_step_dist_min"] = float(min(dists))
    stats["uav_step_dist_max"] = float(max(dists))
    stats["uav_step_dist_expected"] = float(expected_step)
    stats["uav_step_dist_tol"] = float(step_tol)
    stats["uav_step_dist_bad_count"] = int(len(bad_steps))
    stats["uav_step_dist_delete_threshold"] = float(hard_delete_threshold)

    if bad_steps:
        preview = ", ".join([f"(frame={i}, dist={dist:.4f})" for i, dist in bad_steps[:10]])
        issues.append(
            TrajectoryIssue(
                scene,
                traj,
                "warning",
                "uav_step_distance_not_expected",
                (
                    f"{len(bad_steps)}/{len(dists)} steps differ from expected {expected_step:.4f}m "
                    f"(tol={step_tol:.4f}); first: {preview}"
                ),
            )
        )
    return bool(stats["uav_step_dist_max"] > hard_delete_threshold)


def check_collected_trajectory(
    trajectory_dir: Path,
    scene: str,
    delete_bad_trajectories: bool,
    delete_step_threshold: float,
) -> Tuple[List[TrajectoryIssue], Dict[str, Any]]:
    traj = trajectory_dir.name
    issues: List[TrajectoryIssue] = []
    stats: Dict[str, Any] = {"scene": scene, "trajectory": traj}

    uav_json = trajectory_dir / "uav_trajectory.json"
    rgb_dir = trajectory_dir / "rgb"
    target_json = trajectory_dir / "target_trajectory.json"
    jammer_json = trajectory_dir / "jammer_trajectories.json"

    if not uav_json.exists():
        issues.append(TrajectoryIssue(scene, traj, "error", "missing_uav_json", str(uav_json)))
        return issues, stats
    if not rgb_dir.exists():
        issues.append(TrajectoryIssue(scene, traj, "error", "missing_rgb_dir", str(rgb_dir)))

    try:
        uav = _load_json(uav_json)
    except Exception as exc:
        issues.append(TrajectoryIssue(scene, traj, "error", "uav_json_parse_error", str(exc)))
        return issues, stats

    frames = uav.get("trajectory")
    if not isinstance(frames, list):
        issues.append(TrajectoryIssue(scene, traj, "error", "invalid_uav_trajectory", "uav_trajectory.json missing list field 'trajectory'"))
        return issues, stats
    if len(frames) == 0:
        issues.append(TrajectoryIssue(scene, traj, "error", "empty_uav_trajectory", "trajectory list is empty"))
        return issues, stats

    stats["uav_frames"] = len(frames)
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            issues.append(TrajectoryIssue(scene, traj, "error", "invalid_frame_type", f"frame[{i}] is not a dict"))
            continue
        _check_frame_numeric(frame, i, scene, traj, issues)
        if "frame_idx" in frame and isinstance(frame["frame_idx"], (int, float)) and int(frame["frame_idx"]) != i:
            issues.append(TrajectoryIssue(scene, traj, "warning", "frame_idx_mismatch", f"frame[{i}] has frame_idx={frame['frame_idx']}"))

    should_delete = _check_uav_step_distance(
        frames=frames,
        scene=scene,
        traj=traj,
        expected_step=1.0,
        step_tol=max(0.2, float(delete_step_threshold) - 1.0),
        hard_delete_threshold=delete_step_threshold,
        issues=issues,
        stats=stats,
    )
    stats["deleted"] = False
    stats["delete_candidate_by_step_threshold"] = bool(should_delete)

    rgb_files = sorted(rgb_dir.glob("frame_*.png")) if rgb_dir.exists() else []
    stats["rgb_frames"] = len(rgb_files)
    _check_rgb_sequence(rgb_files, scene, traj, issues)
    if rgb_files and len(rgb_files) != len(frames):
        issues.append(TrajectoryIssue(scene, traj, "warning", "frame_count_mismatch", f"uav_frames={len(frames)} rgb_frames={len(rgb_files)}"))

    if target_json.exists():
        try:
            target = _load_json(target_json)
            target_traj = target.get("trajectory") or target.get("target_trajectory") or target.get("target_trajectory_airsim")
            if isinstance(target_traj, list):
                stats["target_frames"] = len(target_traj)
                if len(target_traj) != len(frames):
                    issues.append(TrajectoryIssue(scene, traj, "warning", "target_length_mismatch", f"uav_frames={len(frames)} target_frames={len(target_traj)}"))
                bad = 0
                for j, point in enumerate(target_traj):
                    if _xyz_from_any(point) is None:
                        bad += 1
                        if bad <= 3:
                            issues.append(TrajectoryIssue(scene, traj, "error", "invalid_target_point", f"target[{j}] is not valid finite xyz"))
            else:
                issues.append(TrajectoryIssue(scene, traj, "warning", "invalid_target_trajectory", "target_trajectory.json has no recognized list field"))
        except Exception as exc:
            issues.append(TrajectoryIssue(scene, traj, "error", "target_json_parse_error", str(exc)))
    else:
        issues.append(TrajectoryIssue(scene, traj, "warning", "missing_target_json", str(target_json)))

    if jammer_json.exists():
        try:
            jammer = _load_json(jammer_json)
            jammer_traj = jammer.get("jammer_trajectories") or jammer.get("jammer_trajectories_airsim")
            if isinstance(jammer_traj, dict):
                stats["num_jammers"] = len(jammer_traj)
            elif isinstance(jammer_traj, list):
                stats["num_jammers"] = 1
            else:
                issues.append(TrajectoryIssue(scene, traj, "warning", "invalid_jammer_trajectories", "jammer_trajectories.json has invalid trajectory field"))
        except Exception as exc:
            issues.append(TrajectoryIssue(scene, traj, "error", "jammer_json_parse_error", str(exc)))

    has_warning_or_error = any(issue.severity in {"warning", "error"} for issue in issues)
    if has_warning_or_error and delete_bad_trajectories:
        delete_types = sorted({issue.issue_type for issue in issues if issue.severity in {"warning", "error"}})
        try:
            shutil.rmtree(trajectory_dir)
            stats["deleted"] = True
            issues.append(
                TrajectoryIssue(
                    scene,
                    traj,
                    "error",
                    "trajectory_deleted_by_issue",
                    "trajectory deleted because anomaly check found warning/error issues: "
                    + ", ".join(delete_types),
                )
            )
        except Exception as exc:
            issues.append(TrajectoryIssue(scene, traj, "error", "trajectory_delete_failed", f"failed to delete trajectory dir: {exc}"))

    return issues, stats


def run_dataset_anomaly_check(args, scenes: Sequence[str]) -> Dict[str, Any]:
    dataset_root = Path(args.dataset_base_dir).expanduser().resolve()
    if not dataset_root.exists():
        print(f"[WARN] Dataset anomaly check skipped; dataset root not found: {dataset_root}")
        return {
            "summary": {
                "checked_trajectories": 0,
                "error_count": 1,
                "warning_count": 0,
                "deleted_trajectories": 0,
            },
            "output_path": None,
        }

    all_issues: List[TrajectoryIssue] = []
    all_stats: List[Dict[str, Any]] = []
    checked = 0

    for scene in scenes:
        scene_dir = dataset_root / scene
        if not scene_dir.exists():
            all_issues.append(TrajectoryIssue(scene, "-", "error", "missing_scene_dir", str(scene_dir)))
            continue
        traj_dirs = sorted([d for d in scene_dir.glob("trajectory_*") if d.is_dir()], key=natural_key)
        if int(args.anomaly_head) > 0:
            traj_dirs = traj_dirs[: int(args.anomaly_head)]
        for traj_dir in traj_dirs:
            checked += 1
            issues, stats = check_collected_trajectory(
                traj_dir,
                scene,
                delete_bad_trajectories=bool(args.delete_bad_trajectories),
                delete_step_threshold=float(args.delete_step_threshold),
            )
            all_issues.extend(issues)
            all_stats.append(stats)

    err_n = sum(1 for issue in all_issues if issue.severity == "error")
    warn_n = sum(1 for issue in all_issues if issue.severity == "warning")
    deleted_n = sum(1 for stats in all_stats if bool(stats.get("deleted", False)))
    summary = {
        "dataset_root": str(dataset_root),
        "checked_scenes": list(scenes),
        "checked_trajectories": checked,
        "error_count": err_n,
        "warning_count": warn_n,
        "deleted_trajectories": deleted_n,
        "delete_step_threshold": float(args.delete_step_threshold),
        "delete_bad_trajectories": bool(args.delete_bad_trajectories),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    output_path = (
        Path(args.anomaly_output).expanduser().resolve()
        if str(args.anomaly_output).strip()
        else dataset_root / f"dataset_anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "issues": [asdict(issue) for issue in all_issues],
        "trajectory_stats": all_stats,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[anomaly report saved] {output_path}")
    return {"summary": summary, "output_path": output_path}


def ensure_instruction_files(args, scenes: Sequence[str]) -> int:
    dataset_root = Path(args.dataset_base_dir).expanduser().resolve()
    if not dataset_root.exists():
        return 0

    written = 0
    for scene in scenes:
        scene_dir = dataset_root / scene
        if not scene_dir.exists():
            continue
        for traj_dir in sorted([d for d in scene_dir.glob("trajectory_*") if d.is_dir()], key=natural_key):
            uav_json = traj_dir / "uav_trajectory.json"
            instruction_json = traj_dir / "instruction.json"
            if not uav_json.exists() or instruction_json.exists():
                continue
            num_frames = 0
            try:
                payload = _load_json(uav_json)
                frames = payload.get("trajectory")
                if isinstance(frames, list):
                    num_frames = len(frames)
            except Exception:
                pass
            instruction_payload = {
                "scene_id": scene,
                "trajectory_name": traj_dir.name,
                "num_frames": int(num_frames),
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "instruction": DEFAULT_EPISODE_INSTRUCTION,
                "instructions": DEFAULT_EPISODE_INSTRUCTION,
            }
            tmp_path = traj_dir / "instruction.json.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(instruction_payload, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(instruction_json)
            written += 1
    if written:
        print(f"[instruction] generated missing instruction.json files: {written}")
    return written


def cleanup_executor_connections(executor):
    try:
        if getattr(executor, "sim_client_tool", None) is not None:
            try:
                executor.sim_client_tool._closeConnection()
            except Exception:
                pass
            try:
                executor.sim_client_tool._closeSocketConnection()
            except Exception:
                pass
    finally:
        try:
            executor.disconnect()
        except Exception:
            pass


def close_scene_process(args, scene_id) -> bool:
    try:
        import msgpackrpc
    except Exception as exc:
        print(f"[SCENE-CLOSE-WARN] scene={scene_id}: msgpackrpc unavailable: {exc}")
        return False

    client = None
    try:
        client = msgpackrpc.Client(
            msgpackrpc.Address(args.sim_server_host, args.sim_server_port),
            timeout=30,
        )
        result = client.call("close_scenes", args.sim_server_host, [str(scene_id)])
        ok = bool(result)
        print(f"[SCENE-CLOSE] scene={scene_id} result={result}")
        return ok
    except Exception as exc:
        print(f"[SCENE-CLOSE-WARN] scene={scene_id}: failed to request close_scenes: {exc}")
        return False
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def build_executor(module, args, scene_id, gpu_id):
    target_asset_name = None if args.random_target_asset else args.target_asset_name
    jammer_asset_name = None if args.random_jammer_asset else args.jammer_asset_name

    executor = module.TrajectoryExecutor(
        scene_id=scene_id,
        sim_server_host=args.sim_server_host,
        sim_server_port=args.sim_server_port,
        gpu_id=gpu_id,
        scene_index=args.scene_index,
        uav_vehicle_name=args.uav_vehicle_name,
        target_object_name=args.target_object_name,
        target_asset_name=target_asset_name,
        target_object_scale=tuple(args.target_scale),
        camera_name=args.camera_name,
        auto_start_scene=True,
        deterministic_step_mode=True,
        jammer_enabled=args.jammer_enabled,
        jammer_object_name=args.jammer_object_name,
        jammer_asset_name=jammer_asset_name,
        jammer_object_scale=tuple(args.jammer_scale),
    )
    executor._recover_abnormal_jump = bool(args.recover_abnormal_jump)
    return executor


def run_scene_batch(args, scene_id, gpu_id, progress_position=0):
    trajectory_root = Path(args.trajectory_dir)
    files = discover_trajectory_files(trajectory_root, scene_id, args.trajectory_pattern)

    if not files:
        print(f"[WARN] scene={scene_id} 未找到轨迹文件: root={trajectory_root}, pattern={args.trajectory_pattern}")
        return {
            "scene_id": scene_id,
            "gpu_id": gpu_id,
            "num_files": 0,
        }

    print("=" * 100)
    print(f"scene={scene_id} | gpu={gpu_id} | trajectories={len(files)}")
    print(f"trajectory root : {trajectory_root}")
    print(f"dataset out dir : {args.dataset_base_dir}")
    print(f"target asset    : {'RANDOM(UAV1-UAV20)' if args.random_target_asset else args.target_asset_name}")
    print(f"jammer enabled  : {args.jammer_enabled}")
    print(f"jammer asset    : {('RANDOM(UAV1-UAV20)' if args.random_jammer_asset else args.jammer_asset_name) if args.jammer_enabled else 'DISABLED'}")
    print(f"skip hover      : {args.skip_hover}")
    print("=" * 100)

    try:
        module = dynamic_import_module(Path(args.executor_script))
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise ModuleNotFoundError(
            f"无法导入执行器依赖（缺少模块: {missing}）。"
            f"当前环境需要安装 AirSim Python 包，或切换到包含 airsim 的环境后再运行。"
        ) from e

    executor = build_executor(module, args, scene_id, gpu_id)
    executor._progress_position = progress_position

    try:
        total = len(files)
        for idx, trajectory_file in enumerate(files, start=1):
            print(f"\n[START] scene={scene_id} gpu={gpu_id} [{idx}/{total}] {trajectory_file}")
            try:
                executor.execute_trajectory(
                    trajectory_file=str(trajectory_file),
                    dataset_base_dir=args.dataset_base_dir,
                    save_dataset=args.save_dataset,
                    skip_hover=args.skip_hover,
                    trajectory_index=idx,
                    total_trajectories=total,
                    max_retries=args.max_retries,
                    jump_threshold=args.jump_threshold,
                )
            except Exception as exc:
                print(f"[TRAJECTORY-ERROR] scene={scene_id} gpu={gpu_id} [{idx}/{total}] {trajectory_file}: {exc}")
                try:
                    cleanup_executor_connections(executor)
                except Exception:
                    pass
                executor = build_executor(module, args, scene_id, gpu_id)
                executor._progress_position = progress_position
        return {
            "scene_id": scene_id,
            "gpu_id": gpu_id,
            "num_files": total,
        }
    finally:
        cleanup_executor_connections(executor)
        close_scene_process(args, scene_id)


def worker_entry(args, gpu_id, scene_ids, progress_position=0):
    results = []
    for scene_id in scene_ids:
        results.append(run_scene_batch(args, scene_id, gpu_id, progress_position))
    return results


def count_expected_trajectories(args, scenes: Sequence[str]) -> int:
    trajectory_root = Path(args.trajectory_dir)
    total = 0
    for scene in scenes:
        total += len(discover_trajectory_files(trajectory_root, scene, args.trajectory_pattern))
    return total


def run_collection_round(args, gpu_assignment: Dict[int, List[str]]) -> None:
    if args.multi_worker and len(gpu_assignment) > 1:
        futures = []
        with ThreadPoolExecutor(max_workers=len(gpu_assignment)) as executor_pool:
            for worker_idx, (gpu_id, scene_ids) in enumerate(gpu_assignment.items()):
                futures.append(
                    executor_pool.submit(worker_entry, args, gpu_id, scene_ids, worker_idx)
                )
            for future in as_completed(futures):
                future.result()
    else:
        for worker_idx, (gpu_id, scene_ids) in enumerate(gpu_assignment.items()):
            worker_entry(args, gpu_id, scene_ids, worker_idx)


def print_config(args, gpu_assignment):
    print("=" * 100)
    print("Direct AirSim launcher configuration")
    print("=" * 100)
    print(f"executor script  : {args.executor_script}")
    print(f"trajectory dir   : {args.trajectory_dir}")
    print(f"trajectory patt. : {args.trajectory_pattern}")
    print(f"dataset base dir : {args.dataset_base_dir}")
    print(f"scene ids        : {args.scene_id}")
    print(f"gpu ids          : {normalize_gpu_ids(args.gpu_id)}")
    print(f"gpu assignment   : {gpu_assignment}")
    print(f"sim host         : {args.sim_server_host}")
    print(f"sim port         : {args.sim_server_port}")
    print(f"scene index      : {args.scene_index}")
    print(f"uav vehicle      : {args.uav_vehicle_name}")
    print(f"target object    : {args.target_object_name}")
    print(f"target asset     : {'RANDOM(UAV1-UAV20)' if args.random_target_asset else args.target_asset_name}")
    print(f"target scale     : {tuple(args.target_scale)}")
    print(f"jammer enabled   : {args.jammer_enabled}")
    print(f"jammer asset     : {('RANDOM(UAV1-UAV20)' if args.random_jammer_asset else args.jammer_asset_name) if args.jammer_enabled else 'DISABLED'}")
    print(f"jammer scale     : {tuple(args.jammer_scale)}")
    print(f"camera name      : {args.camera_name}")
    print(f"multi worker     : {args.multi_worker}")
    print(f"save dataset     : {args.save_dataset}")
    print(f"skip hover       : {args.skip_hover}")
    print(f"max retries      : {args.max_retries}")
    print(f"jump threshold   : {args.jump_threshold}")
    print(f"recover jump     : {args.recover_abnormal_jump}")
    print(f"check anomalies  : {args.check_dataset_anomalies}")
    print(f"delete bad traj. : {args.delete_bad_trajectories} (any warning/error)")
    print(f"delete step th.  : {args.delete_step_threshold}")
    print(f"anomaly head     : {args.anomaly_head}")
    print(f"anomaly output   : {args.anomaly_output or 'AUTO'}")
    print(f"repair clean     : {args.repair_until_clean}")
    print(f"repair rounds    : {args.max_repair_rounds}")
    print(f"DAGGER_MULTI_WORKER={os.environ.get('DAGGER_MULTI_WORKER', '0')}")
    print("=" * 100)


def main():
    args = parse_args()
    gpu_ids = normalize_gpu_ids(args.gpu_id)
    gpu_assignment = assign_scenes_to_gpus(args.scene_id, gpu_ids)

    if args.multi_worker:
        os.environ["DAGGER_MULTI_WORKER"] = "1"
    else:
        os.environ.pop("DAGGER_MULTI_WORKER", None)

    print_config(args, gpu_assignment)

    expected_total = count_expected_trajectories(args, args.scene_id)
    max_rounds = max(int(args.max_repair_rounds), 1) if args.repair_until_clean else 1
    last_summary: Dict[str, Any] = {}

    for round_idx in range(1, max_rounds + 1):
        print("=" * 100)
        print(f"[ROUND {round_idx}/{max_rounds}] start collection/check cycle")
        print(f"[ROUND {round_idx}/{max_rounds}] expected trajectories: {expected_total}")
        print("=" * 100)

        run_collection_round(args, gpu_assignment)

        print(f"\n✓ 第 {round_idx} 轮 scene 执行完成")
        if args.save_dataset:
            ensure_instruction_files(args, args.scene_id)

        if args.check_dataset_anomalies and args.save_dataset:
            print("\n[CHECK] 开始检查采集数据异常...")
            report = run_dataset_anomaly_check(args, args.scene_id)
            last_summary = dict(report.get("summary") or {})
        elif args.check_dataset_anomalies and not args.save_dataset:
            print("\n[WARN] anomaly check skipped because --no-save-dataset is active")
            break
        else:
            break

        checked = int(last_summary.get("checked_trajectories", 0))
        errors = int(last_summary.get("error_count", 0))
        warnings = int(last_summary.get("warning_count", 0))
        deleted = int(last_summary.get("deleted_trajectories", 0))
        missing = max(int(expected_total) - checked, 0)
        clean = checked >= int(expected_total) and errors == 0 and warnings == 0 and deleted == 0

        print(
            f"[ROUND {round_idx}/{max_rounds}] checked={checked}/{expected_total}, "
            f"missing={missing}, errors={errors}, warnings={warnings}, deleted={deleted}"
        )
        if clean:
            print("[ROUND] dataset is clean; stop repair loop")
            break
        if not args.repair_until_clean:
            break
        if round_idx >= max_rounds:
            print("[WARN] max repair rounds reached; dataset may still contain missing or problematic trajectories")
            break
        print("[ROUND] dataset still has missing/problematic trajectories; rerun collection for repair")


if __name__ == "__main__":
    main()
