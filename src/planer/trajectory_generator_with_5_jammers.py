import os
import json
import math
import random
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ===================== 配置区 =====================
MAP_DIR = "/data1/ysq/OurVLN/Plandataset/map"
UDF_CACHE_DIR = "/data1/ysq/OurVLN/Plandataset/udf_cache"
OUTPUT_DIR = "/data1/ysq/Worldmodel/Plandataset"

CITY_LIST = [
    "city_1",
    "city_2",
    "city_3",
]

NUM_TRAJECTORIES = 500
MODE_COUNTS = {
    "easy": 167,
    "medium": 167,
    "hard": 166,
}

USE_UDF = True
UDF_RESOLUTION = 1.0

MIN_Z_HEIGHT = 10.0
MAX_Z_HEIGHT = 100.0
BOUND_MARGIN = 2.0
SAFETY_RADIUS = 10.0

TIME_STEP = 1.0
TARGET_SPEED = 1.0
TRACKER_SPEED = 1.0
STEP_LENGTH = 1.0

MEAN_TRAJ_LENGTH = 100
MIN_TRAJ_LENGTH = 50
MAX_TRAJ_LENGTH = 150

START_SEPARATION_MIN = 5.0
START_SEPARATION_MAX = 10.0
TRACKER_FPV_HALF_ANGLE_DEG = 30.0

JAMMER_ENABLED = True
JAMMER_SPEED = 2.0
JAMMER_VIEW_DISTANCE_MIN = 10
JAMMER_VIEW_DISTANCE_MAX = 100
JAMMER_LATERAL_MAX = 25
JAMMER_VERTICAL_MAX = 18
JAMMER_VERTICAL_HALF_ANGLE_DEG = 45
JAMMER_FOV_MARGIN_DEG = 5.0
JAMMER_MIN_TRACKER_DISTANCE = 10
JAMMER_MIN_TARGET_DISTANCE = 10
JAMMER_TARGET_SOFT_BUFFER = 2.0
JAMMER_LINE_OF_SIGHT_RADIUS = 1.0
JAMMER_EVENT_MIN_STEPS = 8
JAMMER_EVENT_MAX_STEPS = 18
JAMMER_HIDDEN_LATERAL = 7.5
JAMMER_CROSS_DEPTHS = (9.0, 10.5, 12.0, 13.5, 15.0)
JAMMER_CROSS_VERTICALS = (0.6, -0.8, 1.1, -1.1, 0.2)
JAMMER_CROSS_SWEEP_RATIO = 0.85

# 加速开关：不改变 STEP_LENGTH，也不关闭 PNG。
# 只加速 jammer active visible segment 的候选搜索。
FAST_JAMMER_VISIBLE_SEGMENT = True
JAMMER_FAST_CANDIDATE_TRIES = 16
JAMMER_FAST_LOS_CHECK = True
JAMMER_FAST_PREV_SEGMENT_CHECK = False


# tracker 构造加速：优先使用“目标后方固定距离跟随”的快速构造，失败后再回退到直追。
# 不改变 STEP_LENGTH，也不关闭 PNG；只是降低 tracker 构造的失败重试率。
FAST_TRACKER_OFFSET_FOLLOW = True
TRACKER_OFFSET_FOLLOW_ATTEMPTS = 24
TRACKER_DIRECT_CHASE_ATTEMPTS = 24
TRACKER_MAX_Z_DROP = 2.0
TRACKER_START_YAW_JITTER_DEG = 10.0
TRACKER_DIRECT_LOOKAHEAD = 1

# hard 模式加速/稳健性：先选建筑，再在建筑外圈采样起点，避免全图随机点导致绕楼失败。
FAST_HARD_START_NEAR_BUILDING = True
HARD_START_BUILDING_ATTEMPTS = 300
HARD_START_EXTRA_RADIUS_MIN = 2.0
HARD_START_EXTRA_RADIUS_MAX = 10.0
HARD_START_Z_ABOVE_BUILDING_MIN = 6.0
HARD_START_Z_ABOVE_BUILDING_MAX = 24.0
HARD_CLIMB_PER_STEP_MIN = 0.03
HARD_CLIMB_PER_STEP_MAX = 0.16

# 如果某个城市无法从 mesh 中提取到建筑，hard 模式退化为高曲率/频繁转向的逃逸轨迹，
# 避免该城市的 hard 样本全部失败。
HARD_FALLBACK_WHEN_NO_BUILDING = True
HARD_FALLBACK_SEGMENT_MIN = 5
HARD_FALLBACK_SEGMENT_MAX = 10
HARD_FALLBACK_YAW_DEG_MIN = 25.0
HARD_FALLBACK_YAW_DEG_MAX = 70.0
HARD_FALLBACK_PITCH_DEG_MAX = 6.0
HARD_FALLBACK_WIGGLE_YAW_DEG = 10.0
HARD_FALLBACK_WIGGLE_PITCH_DEG = 3.0

MAX_TURN_DEG = 15
MAX_PITCH_DEG = 15
SEARCH_TRIES_PER_STEP = 100
MAX_TRAJ_BUILD_RETRY = 100

SAVE_PNG = True
PLOT_EVERY = 1

MIN_BUILDING_HEIGHT = 12.0
MIN_BUILDING_XY_SPAN = 8.0

NUM_WORKERS = max(1, min(8, (os.cpu_count() or 8) - 1))
RANDOM_SEED = 20260420
RESUME_IF_EXISTS = True
# ==================================================


# 进程内缓存：每个 worker 只加载一次当前 city 的地图
_PROCESS_DATA = None


@dataclass
class UDFGrid:
    udf: np.ndarray
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    resolution: float

    @classmethod
    def from_file(cls, udf_path: str) -> "UDFGrid":
        udf = np.load(udf_path)
        meta = np.load(udf_path.replace(".npy", "_meta.npy"), allow_pickle=True).item()
        return cls(
            udf=udf,
            min_xyz=np.array(meta["min_xyz"], dtype=float),
            max_xyz=np.array(meta["max_xyz"], dtype=float),
            resolution=float(meta["resolution"]),
        )

    def world_to_grid(self, p: np.ndarray) -> np.ndarray:
        return ((p - self.min_xyz) / self.resolution).astype(int)

    def in_bounds(self, idx: np.ndarray) -> bool:
        return (
            0 <= idx[0] < self.udf.shape[0]
            and 0 <= idx[1] < self.udf.shape[1]
            and 0 <= idx[2] < self.udf.shape[2]
        )

    def get_distance(self, p: np.ndarray) -> float:
        idx = self.world_to_grid(p)
        if not self.in_bounds(idx):
            return 0.0
        return float(self.udf[idx[0], idx[1], idx[2]])

    def get_distances(self, points: np.ndarray) -> np.ndarray:
        """
        批量查询 UDF 距离，减少 is_segment_free 中大量 Python 循环开销。
        越界点距离置为 0，等价于不可通行。
        """
        points = np.asarray(points, dtype=float)
        idx = ((points - self.min_xyz) / self.resolution).astype(np.int64)

        valid = (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.udf.shape[0]) &
            (idx[:, 1] >= 0) & (idx[:, 1] < self.udf.shape[1]) &
            (idx[:, 2] >= 0) & (idx[:, 2] < self.udf.shape[2])
        )

        dists = np.zeros(points.shape[0], dtype=float)
        valid_idx = idx[valid]
        if valid_idx.size > 0:
            dists[valid] = self.udf[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
        return dists

    def is_point_free(self, p: np.ndarray, safe_radius: float) -> bool:
        return self.get_distance(p) > safe_radius


@dataclass
class BuildingInfo:
    center_xy: np.ndarray
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    orbit_radius: float
    height: float


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=float)


def rotate_direction(base_dir: np.ndarray, yaw_rad: float, pitch_rad: float) -> np.ndarray:
    base_dir = normalize(base_dir)
    R_yaw = rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw_rad)
    d = normalize(R_yaw @ base_dir)
    side = np.cross(d, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    side = normalize(side)
    R_pitch = rotation_matrix_from_axis_angle(side, pitch_rad)
    d = normalize(R_pitch @ d)
    return d


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def estimate_forward_dirs(traj: List[np.ndarray]) -> List[np.ndarray]:
    dirs: List[np.ndarray] = []
    n = len(traj)
    for i in range(n):
        if n == 1:
            d = np.array([1.0, 0.0, 0.0], dtype=float)
        elif i < n - 1:
            d = traj[i + 1] - traj[i]
        else:
            d = traj[i] - traj[i - 1]
        if np.linalg.norm(d) < 1e-6:
            d = np.array([1.0, 0.0, 0.0], dtype=float)
        dirs.append(normalize(d))
    return dirs


def build_camera_basis(forward: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = normalize(forward)
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([1.0, 0.0, 0.0], dtype=float)

    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    right = normalize(right)
    up = normalize(np.cross(right, forward))

    if np.dot(up, world_up) < 0:
        up = -up
        right = -right

    return forward, right, up


def is_point_in_camera_fov(origin: np.ndarray,
                           forward: np.ndarray,
                           right: np.ndarray,
                           up: np.ndarray,
                           point: np.ndarray,
                           horiz_half_angle_deg: float,
                           vert_half_angle_deg: float) -> bool:
    rel = point - origin
    depth = float(np.dot(rel, forward))
    if depth <= 1e-6:
        return False

    horiz = math.degrees(math.atan2(float(np.dot(rel, right)), depth))
    vert = math.degrees(math.atan2(float(np.dot(rel, up)), depth))
    return abs(horiz) <= horiz_half_angle_deg and abs(vert) <= vert_half_angle_deg


def _tracker_view_bounds() -> Tuple[float, float]:
    horiz_half = max(5.0, TRACKER_FPV_HALF_ANGLE_DEG - JAMMER_FOV_MARGIN_DEG)
    vert_half = max(5.0, JAMMER_VERTICAL_HALF_ANGLE_DEG - JAMMER_FOV_MARGIN_DEG)
    return horiz_half, vert_half


def _sample_jammer_candidate_near_desired(tracker_pos: np.ndarray,
                                         tracker_forward: np.ndarray,
                                         prev_jammer: Optional[np.ndarray],
                                         desired_depth: float,
                                         desired_lateral: float,
                                         desired_vertical: float,
                                         prox,
                                         bounds: np.ndarray,
                                         udf_grid: Optional[UDFGrid],
                                         preferred_target: Optional[np.ndarray] = None,
                                         center_weight: float = 1.0,
                                         smooth_weight: float = 0.25,
                                         allow_relaxed: bool = True,
                                         require_in_fov: bool = True,
                                         prefer_out_of_fov: bool = False,
                                         lateral_limit_override: Optional[float] = None,
                                         require_line_of_sight: bool = True) -> Optional[np.ndarray]:
    min_xyz = bounds[0] + BOUND_MARGIN
    max_xyz = bounds[1] - BOUND_MARGIN
    min_xyz[2] = max(min_xyz[2], MIN_Z_HEIGHT)
    max_xyz[2] = min(max_xyz[2], MAX_Z_HEIGHT)

    horiz_half, vert_half = _tracker_view_bounds()
    max_step = max(JAMMER_SPEED * TIME_STEP, STEP_LENGTH * 0.8)
    forward, right, up = build_camera_basis(tracker_forward)
    lateral_limit = JAMMER_LATERAL_MAX if lateral_limit_override is None else max(JAMMER_LATERAL_MAX, lateral_limit_override)

    if lateral_limit <= JAMMER_LATERAL_MAX + 1e-6:
        lateral_offsets = [0.0, -0.6, 0.6, -1.2, 1.2, -2.0, 2.0]
        relaxed_lats = [desired_lateral, 0.0, -1.5, 1.5, -2.5, 2.5]
    else:
        lateral_offsets = [0.0, -0.8, 0.8, -1.6, 1.6, -3.0, 3.0, -4.5, 4.5]
        relaxed_lats = [desired_lateral, 0.0, -0.4 * lateral_limit, 0.4 * lateral_limit, -0.7 * lateral_limit, 0.7 * lateral_limit, -lateral_limit, lateral_limit]

    depth_offsets = [0.0, -1.5, 1.5, -3.0, 3.0]
    vertical_offsets = [0.0, -0.4, 0.4, -0.9, 0.9, -1.4, 1.4]

    best_score = None
    best_point = None

    for d_off in depth_offsets:
        depth = desired_depth + d_off
        if not (JAMMER_VIEW_DISTANCE_MIN <= depth <= JAMMER_VIEW_DISTANCE_MAX):
            continue
        for l_off in lateral_offsets:
            lateral = desired_lateral + l_off
            if abs(lateral) > lateral_limit + 1e-6:
                continue
            for v_off in vertical_offsets:
                vertical = desired_vertical + v_off
                if abs(vertical) > JAMMER_VERTICAL_MAX + 1e-6:
                    continue

                candidate = tracker_pos + depth * forward + lateral * right + vertical * up
                if np.any(candidate < min_xyz) or np.any(candidate > max_xyz):
                    continue
                if float(np.linalg.norm(candidate - tracker_pos)) < JAMMER_MIN_TRACKER_DISTANCE:
                    continue
                if not is_point_free(candidate, prox, SAFETY_RADIUS, udf_grid):
                    continue
                if require_line_of_sight and not is_segment_free(tracker_pos, candidate, prox, JAMMER_LINE_OF_SIGHT_RADIUS, udf_grid):
                    continue

                in_fov = is_point_in_camera_fov(tracker_pos, forward, right, up, candidate, horiz_half, vert_half)
                if require_in_fov and not in_fov:
                    continue
                if prefer_out_of_fov and in_fov:
                    continue

                motion = 0.0
                if prev_jammer is not None:
                    motion = float(np.linalg.norm(candidate - prev_jammer))
                    if motion > max_step + 2.0:
                        continue
                    if not is_segment_free(prev_jammer, candidate, prox, SAFETY_RADIUS, udf_grid):
                        continue

                target_dist = None
                if preferred_target is not None:
                    target_dist = float(np.linalg.norm(candidate - preferred_target))
                    if target_dist < JAMMER_MIN_TARGET_DISTANCE:
                        continue

                center_penalty = center_weight * (abs(lateral) + 1.2 * abs(vertical) + 0.15 * abs(depth - desired_depth))
                smooth_penalty = smooth_weight * motion
                target_penalty = 0.0
                if target_dist is not None:
                    target_penalty = 0.25 * max(
                        0.0,
                        (JAMMER_MIN_TARGET_DISTANCE + JAMMER_TARGET_SOFT_BUFFER) - target_dist,
                    )
                offscreen_bonus = -2.0 if (prefer_out_of_fov and not in_fov) else 0.0
                score = center_penalty + smooth_penalty + target_penalty + offscreen_bonus
                if best_score is None or score < best_score:
                    best_score = score
                    best_point = candidate

    if best_point is not None or not allow_relaxed:
        return None if best_point is None else np.array(best_point, dtype=float)

    relaxed_depths = [desired_depth, 10.0, 12.0, 14.0, 16.0]
    relaxed_verts = [desired_vertical, 0.0, 1.0, -1.0, 1.8, -1.8]
    for depth in relaxed_depths:
        if not (JAMMER_VIEW_DISTANCE_MIN <= depth <= JAMMER_VIEW_DISTANCE_MAX):
            continue
        for lateral in relaxed_lats:
            if abs(lateral) > lateral_limit + 1e-6:
                continue
            for vertical in relaxed_verts:
                if abs(vertical) > JAMMER_VERTICAL_MAX + 1e-6:
                    continue
                candidate = tracker_pos + depth * forward + lateral * right + vertical * up
                if np.any(candidate < min_xyz) or np.any(candidate > max_xyz):
                    continue
                if not is_point_free(candidate, prox, SAFETY_RADIUS, udf_grid):
                    continue
                if require_line_of_sight and not is_segment_free(tracker_pos, candidate, prox, JAMMER_LINE_OF_SIGHT_RADIUS, udf_grid):
                    continue
                if preferred_target is not None:
                    target_dist = float(np.linalg.norm(candidate - preferred_target))
                    if target_dist < JAMMER_MIN_TARGET_DISTANCE:
                        continue
                in_fov = is_point_in_camera_fov(tracker_pos, forward, right, up, candidate, horiz_half, vert_half)
                if require_in_fov and not in_fov:
                    continue
                if prefer_out_of_fov and in_fov:
                    continue
                if prev_jammer is not None:
                    motion = float(np.linalg.norm(candidate - prev_jammer))
                    if motion > max_step + 4.0:
                        continue
                    if not is_segment_free(prev_jammer, candidate, prox, SAFETY_RADIUS, udf_grid):
                        continue
                return np.array(candidate, dtype=float)
    return None


def _fast_sample_visible_jammer_candidate(tracker_pos: np.ndarray,
                                          tracker_forward: np.ndarray,
                                          prev_jammer: Optional[np.ndarray],
                                          desired_depth: float,
                                          desired_lateral: float,
                                          desired_vertical: float,
                                          prox,
                                          bounds: np.ndarray,
                                          udf_grid: Optional[UDFGrid],
                                          preferred_target: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    快速生成 active visible segment 中的 jammer 位置。

    主要加速点：
    1. 直接在 tracker 相机坐标系中构造少量候选点；
    2. 把 lateral / vertical 约束到 FOV 内，避免大量无效搜索；
    3. 默认不检查 prev_jammer -> candidate 的整段碰撞，只限制速度；
    4. LOS 只检查 tracker -> candidate。
    """
    min_xyz = bounds[0] + BOUND_MARGIN
    max_xyz = bounds[1] - BOUND_MARGIN
    min_xyz[2] = max(min_xyz[2], MIN_Z_HEIGHT)
    max_xyz[2] = min(max_xyz[2], MAX_Z_HEIGHT)

    horiz_half, vert_half = _tracker_view_bounds()
    forward, right, up = build_camera_basis(tracker_forward)

    depth = float(np.clip(desired_depth, JAMMER_VIEW_DISTANCE_MIN, JAMMER_VIEW_DISTANCE_MAX))

    max_lateral_by_fov = math.tan(math.radians(horiz_half * 0.92)) * depth
    max_vertical_by_fov = math.tan(math.radians(vert_half * 0.92)) * depth

    lateral = float(np.clip(
        desired_lateral,
        -min(JAMMER_LATERAL_MAX, max_lateral_by_fov),
        min(JAMMER_LATERAL_MAX, max_lateral_by_fov),
    ))
    vertical = float(np.clip(
        desired_vertical,
        -min(JAMMER_VERTICAL_MAX, max_vertical_by_fov),
        min(JAMMER_VERTICAL_MAX, max_vertical_by_fov),
    ))

    candidate_offsets = [
        (0.0, 0.0, 0.0),
        (0.0, -0.6, 0.0),
        (0.0, 0.6, 0.0),
        (0.0, 0.0, -0.5),
        (0.0, 0.0, 0.5),
        (-0.8, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (0.0, -1.2, 0.4),
        (0.0, 1.2, -0.4),
    ]

    for _ in range(max(0, JAMMER_FAST_CANDIDATE_TRIES - len(candidate_offsets))):
        candidate_offsets.append((
            random.uniform(-1.2, 1.2),
            random.uniform(-1.5, 1.5),
            random.uniform(-1.0, 1.0),
        ))

    max_step = JAMMER_SPEED * TIME_STEP + 1.5

    best_candidate = None
    best_score = None

    for d_off, l_off, v_off in candidate_offsets:
        d = float(np.clip(depth + d_off, JAMMER_VIEW_DISTANCE_MIN, JAMMER_VIEW_DISTANCE_MAX))
        l = lateral + l_off
        v = vertical + v_off

        max_lateral_by_fov = math.tan(math.radians(horiz_half * 0.95)) * d
        max_vertical_by_fov = math.tan(math.radians(vert_half * 0.95)) * d

        if abs(l) > min(JAMMER_LATERAL_MAX, max_lateral_by_fov):
            continue
        if abs(v) > min(JAMMER_VERTICAL_MAX, max_vertical_by_fov):
            continue

        candidate = tracker_pos + d * forward + l * right + v * up

        if np.any(candidate < min_xyz) or np.any(candidate > max_xyz):
            continue

        if float(np.linalg.norm(candidate - tracker_pos)) < JAMMER_MIN_TRACKER_DISTANCE:
            continue

        if preferred_target is not None:
            if float(np.linalg.norm(candidate - preferred_target)) < JAMMER_MIN_TARGET_DISTANCE:
                continue

        if not is_point_free(candidate, prox, SAFETY_RADIUS, udf_grid):
            continue

        if not is_point_in_camera_fov(
            tracker_pos,
            forward,
            right,
            up,
            candidate,
            horiz_half,
            vert_half,
        ):
            continue

        if JAMMER_FAST_LOS_CHECK:
            if not is_segment_free(
                tracker_pos,
                candidate,
                prox,
                JAMMER_LINE_OF_SIGHT_RADIUS,
                udf_grid,
            ):
                continue

        if prev_jammer is not None:
            motion = float(np.linalg.norm(candidate - prev_jammer))
            if motion > max_step:
                continue

            if JAMMER_FAST_PREV_SEGMENT_CHECK:
                if not is_segment_free(prev_jammer, candidate, prox, SAFETY_RADIUS, udf_grid):
                    continue
        else:
            motion = 0.0

        score = abs(l - lateral) + 1.2 * abs(v - vertical) + 0.2 * abs(d - depth) + 0.2 * motion

        if best_score is None or score < best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is not None:
        return np.array(best_candidate, dtype=float)

    return None


def build_jammer_from_tracker_view(target_traj: List[np.ndarray],
                                   tracker_traj: List[np.ndarray],
                                   prox,
                                   bounds: np.ndarray,
                                   udf_grid: Optional[UDFGrid]) -> List[np.ndarray]:
    if len(target_traj) != len(tracker_traj):
        raise RuntimeError("jammer 生成失败：target 和 tracker 长度不一致")

    tracker_dirs = estimate_forward_dirs(tracker_traj)
    jammer_traj: List[np.ndarray] = []
    prev_jammer: Optional[np.ndarray] = None

    for i, (target_pos, tracker_pos) in enumerate(zip(target_traj, tracker_traj)):
        candidate = _sample_jammer_candidate_near_desired(
            tracker_pos=tracker_pos,
            tracker_forward=tracker_dirs[i],
            prev_jammer=prev_jammer,
            desired_depth=12.0,
            desired_lateral=0.0,
            desired_vertical=0.0,
            prox=prox,
            bounds=bounds,
            udf_grid=udf_grid,
            preferred_target=target_pos,
            center_weight=1.0,
            smooth_weight=0.25,
            allow_relaxed=True,
        )
        if candidate is None:
            raise RuntimeError(f"第 {i} 帧无法为 jammer-1 找到可见位置")
        jammer_traj.append(candidate)
        prev_jammer = candidate

    return jammer_traj


def _compute_event_window(num_points: int, jammer_id: int) -> Tuple[int, int]:
    event_len = max(JAMMER_EVENT_MIN_STEPS, min(JAMMER_EVENT_MAX_STEPS, int(round(num_points * 0.12))))
    center_fracs = {1: 0.12, 2: 0.28, 3: 0.46, 4: 0.64, 5: 0.82}
    center = int(round(center_fracs.get(jammer_id, 0.5) * max(0, num_points - 1)))
    half = event_len // 2
    start = max(0, center - half)
    end = min(num_points - 1, start + event_len - 1)
    start = max(0, end - event_len + 1)
    return start, end


def _sample_global_random_direction() -> np.ndarray:
    return normalize(np.array([
        random.uniform(-1.0, 1.0),
        random.uniform(-1.0, 1.0),
        random.uniform(-0.18, 0.18),
    ], dtype=float))


def _step_global_random(generator: "SmoothTrajectoryGenerator",
                        curr: np.ndarray,
                        curr_dir: np.ndarray,
                        desired_dir: Optional[np.ndarray] = None,
                        step_len: float = JAMMER_SPEED * TIME_STEP,
                        max_turn_deg: float = 18.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if desired_dir is None or np.linalg.norm(desired_dir) < 1e-6:
        desired_dir = curr_dir
    desired_dir = rotate_direction(
        normalize(desired_dir),
        math.radians(random.uniform(-18.0, 18.0)),
        math.radians(random.uniform(-6.0, 6.0)),
    )
    nxt, nxt_dir = generator._step_with_smoothing(
        curr=curr,
        curr_dir=curr_dir,
        desired_dir=desired_dir,
        max_turn_deg=max_turn_deg,
        step_len=step_len,
    )
    return nxt, nxt_dir, desired_dir


def _build_visible_cross_segment(target_traj: List[np.ndarray],
                                 tracker_traj: List[np.ndarray],
                                 prox,
                                 bounds: np.ndarray,
                                 udf_grid: Optional[UDFGrid],
                                 active_start: int,
                                 active_end: int,
                                 crossing_variant: int) -> List[np.ndarray]:
    tracker_dirs = estimate_forward_dirs(tracker_traj)
    span = max(1, active_end - active_start)
    segment: List[np.ndarray] = []
    prev_jammer: Optional[np.ndarray] = None

    base_depth = JAMMER_CROSS_DEPTHS[crossing_variant % len(JAMMER_CROSS_DEPTHS)]
    base_vertical = JAMMER_CROSS_VERTICALS[crossing_variant % len(JAMMER_CROSS_VERTICALS)]

    horiz_half, _ = _tracker_view_bounds()

    # 原来的 sweep_limit 偏大，可能让 desired_lateral 天然落在 FOV 外，
    # 导致 _sample_jammer_candidate_near_desired 大量无效尝试。
    # 这里将扫描范围压到 FOV 内，仍然保留左右穿越效果。
    fov_lateral_limit = math.tan(math.radians(horiz_half * JAMMER_CROSS_SWEEP_RATIO)) * base_depth
    sweep_limit = max(2.0, min(JAMMER_LATERAL_MAX, fov_lateral_limit))

    direction = -1.0 if (crossing_variant % 2 == 0) else 1.0

    for i in range(active_start, active_end + 1):
        target_pos = target_traj[i]
        tracker_pos = tracker_traj[i]

        progress = (i - active_start) / float(span)

        desired_lateral = direction * (-sweep_limit + 2.0 * sweep_limit * progress)
        desired_vertical = base_vertical + 0.35 * math.sin(2.0 * math.pi * progress + crossing_variant)
        desired_depth = base_depth + 1.0 * math.sin(2.0 * math.pi * progress)

        if FAST_JAMMER_VISIBLE_SEGMENT:
            candidate = _fast_sample_visible_jammer_candidate(
                tracker_pos=tracker_pos,
                tracker_forward=tracker_dirs[i],
                prev_jammer=prev_jammer,
                desired_depth=desired_depth,
                desired_lateral=desired_lateral,
                desired_vertical=desired_vertical,
                prox=prox,
                bounds=bounds,
                udf_grid=udf_grid,
                preferred_target=target_pos,
            )
        else:
            candidate = None

        # 快速采样失败时，回退到原始严格搜索，保证成功率。
        if candidate is None:
            candidate = _sample_jammer_candidate_near_desired(
                tracker_pos=tracker_pos,
                tracker_forward=tracker_dirs[i],
                prev_jammer=prev_jammer,
                desired_depth=desired_depth,
                desired_lateral=desired_lateral,
                desired_vertical=desired_vertical,
                prox=prox,
                bounds=bounds,
                udf_grid=udf_grid,
                preferred_target=target_pos,
                center_weight=0.45,
                smooth_weight=0.12,
                allow_relaxed=True,
                require_in_fov=True,
                prefer_out_of_fov=False,
                lateral_limit_override=max(sweep_limit + 1.0, JAMMER_LATERAL_MAX),
                require_line_of_sight=True,
            )

        if candidate is None:
            raise RuntimeError(f"第 {i} 帧无法为全局事件 jammer 可见段找到位置")

        segment.append(candidate)
        prev_jammer = candidate

    return segment


def _extend_global_random_backward(generator: "SmoothTrajectoryGenerator",
                                   anchor_first: np.ndarray,
                                   anchor_second: np.ndarray,
                                   num_steps: int) -> List[np.ndarray]:
    """
    线性外推 active segment 之前的 jammer 轨迹。
    不再调用 _step_with_smoothing，因此显著减少 jammer 非可见段耗时。
    """
    if num_steps <= 0:
        return []

    direction = normalize(anchor_first - anchor_second)
    if np.linalg.norm(direction) < 1e-6:
        direction = _sample_global_random_direction()

    points: List[np.ndarray] = []
    step_len = JAMMER_SPEED * TIME_STEP

    # 顺序：远离可见段 -> 靠近 anchor_first
    for k in range(num_steps, 0, -1):
        p = anchor_first + direction * step_len * k
        p = generator.clip_to_bounds(p)
        points.append(np.array(p, dtype=float))

    return points


def _extend_global_random_forward(generator: "SmoothTrajectoryGenerator",
                                  anchor_last: np.ndarray,
                                  anchor_prev: np.ndarray,
                                  num_steps: int) -> List[np.ndarray]:
    """
    线性外推 active segment 之后的 jammer 轨迹。
    不再调用 _step_with_smoothing，因此显著减少 jammer 非可见段耗时。
    """
    if num_steps <= 0:
        return []

    direction = normalize(anchor_last - anchor_prev)
    if np.linalg.norm(direction) < 1e-6:
        direction = _sample_global_random_direction()

    points: List[np.ndarray] = []
    step_len = JAMMER_SPEED * TIME_STEP

    for k in range(1, num_steps + 1):
        p = anchor_last + direction * step_len * k
        p = generator.clip_to_bounds(p)
        points.append(np.array(p, dtype=float))

    return points


def build_global_event_jammer(target_traj: List[np.ndarray],
                              tracker_traj: List[np.ndarray],
                              generator: "SmoothTrajectoryGenerator",
                              prox,
                              bounds: np.ndarray,
                              udf_grid: Optional[UDFGrid],
                              jammer_id: int,
                              crossing_variant: int) -> Dict[str, Any]:
    if len(target_traj) != len(tracker_traj):
        raise RuntimeError("全局事件 jammer 生成失败：target 和 tracker 长度不一致")

    num_points = len(tracker_traj)
    active_start, active_end = _compute_event_window(num_points, jammer_id)
    visible_segment = _build_visible_cross_segment(
        target_traj=target_traj,
        tracker_traj=tracker_traj,
        prox=prox,
        bounds=bounds,
        udf_grid=udf_grid,
        active_start=active_start,
        active_end=active_end,
        crossing_variant=crossing_variant,
    )

    prefix = _extend_global_random_backward(
        generator=generator,
        anchor_first=visible_segment[0],
        anchor_second=visible_segment[1] if len(visible_segment) >= 2 else visible_segment[0] + _sample_global_random_direction(),
        num_steps=active_start,
    )
    suffix = _extend_global_random_forward(
        generator=generator,
        anchor_last=visible_segment[-1],
        anchor_prev=visible_segment[-2] if len(visible_segment) >= 2 else visible_segment[-1] - _sample_global_random_direction(),
        num_steps=num_points - active_end - 1,
    )

    trajectory = prefix + visible_segment + suffix
    if len(trajectory) != num_points:
        raise RuntimeError(f"全局事件 jammer-{jammer_id} 轨迹长度错误: {len(trajectory)} != {num_points}")

    visible_mask = [False] * num_points
    for idx in range(active_start, active_end + 1):
        visible_mask[idx] = True

    return {
        "id": int(jammer_id),
        "type": "global_event",
        "trajectory": trajectory,
        "active_start_frame": int(active_start),
        "active_end_frame": int(active_end),
        "visible_expected_mask": visible_mask,
    }


def build_all_jammer_trajectories(target_traj: List[np.ndarray],
                                  tracker_traj: List[np.ndarray],
                                  generator: "SmoothTrajectoryGenerator",
                                  prox,
                                  bounds: np.ndarray,
                                  udf_grid: Optional[UDFGrid],
                                  num_jammers: int) -> List[Dict[str, Any]]:
    if not (1 <= num_jammers <= 5):
        raise ValueError(f"num_jammers 必须在 1~5 之间，当前为 {num_jammers}")

    jammer_trajs: List[Dict[str, Any]] = []

    for jammer_id, variant in zip(range(1, num_jammers + 1), range(num_jammers)):
        jammer_trajs.append(
            build_global_event_jammer(
                target_traj=target_traj,
                tracker_traj=tracker_traj,
                generator=generator,
                prox=prox,
                bounds=bounds,
                udf_grid=udf_grid,
                jammer_id=jammer_id,
                crossing_variant=variant,
            )
        )

    return jammer_trajs


def load_map(obj_path: str, use_udf: bool = True):
    mesh = trimesh.load(obj_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    swap_yz_transform = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    mesh.apply_transform(swap_yz_transform)

    bounds = mesh.bounds
    prox = trimesh.proximity.ProximityQuery(mesh)

    udf_grid = None
    if use_udf and USE_UDF:
        map_name = os.path.splitext(os.path.basename(obj_path))[0]
        udf_filename = f"{map_name}_udf_{UDF_RESOLUTION:.1f}m.npy"
        udf_path = os.path.join(UDF_CACHE_DIR, udf_filename)

        if os.path.exists(udf_path):
            udf_grid = UDFGrid.from_file(udf_path)
            print(f"[UDF] loaded: {udf_path}")
        else:
            raise FileNotFoundError(
                f"USE_UDF=True，但没有找到 UDF 缓存文件:\n"
                f"  {udf_path}\n"
                f"请先生成 UDF，或者设置 USE_UDF=False。"
            )

    return mesh, prox, bounds, udf_grid


def is_point_free(point: np.ndarray, prox, safety_radius: float, udf_grid: Optional[UDFGrid]) -> bool:
    if point[2] < MIN_Z_HEIGHT or point[2] > MAX_Z_HEIGHT:
        return False
    if udf_grid is not None:
        return udf_grid.is_point_free(point, safety_radius)
    closest_point, distance, face_idx = prox.on_surface([point])
    return float(distance[0]) > safety_radius


def is_segment_free(p1: np.ndarray, p2: np.ndarray, prox, safety_radius: float, udf_grid: Optional[UDFGrid]) -> bool:
    """
    UDF 存在时使用批量距离查询，减少逐点函数调用开销。
    """
    length = float(np.linalg.norm(p2 - p1))
    if length < 1e-6:
        return is_point_free(p1, prox, safety_radius, udf_grid)

    num_samples = max(3, int(length / max(1.0, safety_radius * 0.5)))
    ts = np.linspace(0.0, 1.0, num_samples + 1, dtype=float)
    points = p1[None, :] * (1.0 - ts[:, None]) + p2[None, :] * ts[:, None]

    if np.any(points[:, 2] < MIN_Z_HEIGHT) or np.any(points[:, 2] > MAX_Z_HEIGHT):
        return False

    if udf_grid is not None:
        dists = udf_grid.get_distances(points)
        return bool(np.all(dists > safety_radius))

    for p in points:
        if not is_point_free(p, prox, safety_radius, udf_grid):
            return False
    return True


def sample_free_point(bounds: np.ndarray, prox, safety_radius: float,
                      udf_grid: Optional[UDFGrid], max_tries: int = 5000) -> np.ndarray:
    min_xyz, max_xyz = bounds
    low = min_xyz + BOUND_MARGIN
    high = max_xyz - BOUND_MARGIN
    low[2] = max(low[2], MIN_Z_HEIGHT)
    high[2] = min(high[2], MAX_Z_HEIGHT)
    for _ in range(max_tries):
        p = np.array([
            random.uniform(low[0], high[0]),
            random.uniform(low[1], high[1]),
            random.uniform(low[2], high[2]),
        ], dtype=float)
        if is_point_free(p, prox, safety_radius, udf_grid):
            return p
    raise RuntimeError("无法采样到自由空间起点")


def extract_buildings(mesh: trimesh.Trimesh) -> List[BuildingInfo]:
    buildings: List[BuildingInfo] = []
    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        components = [mesh]

    for comp in components:
        if len(comp.vertices) < 20:
            continue
        bmin, bmax = comp.bounds
        span = bmax - bmin
        if span[2] < MIN_BUILDING_HEIGHT:
            continue
        if max(span[0], span[1]) < MIN_BUILDING_XY_SPAN:
            continue
        center_xy = 0.5 * (bmin[:2] + bmax[:2])
        half_extent_xy = 0.5 * max(span[0], span[1])
        orbit_radius = half_extent_xy + SAFETY_RADIUS + 5.0
        buildings.append(BuildingInfo(
            center_xy=center_xy,
            min_xyz=bmin.copy(),
            max_xyz=bmax.copy(),
            orbit_radius=float(orbit_radius),
            height=float(span[2]),
        ))

    filtered = []
    map_span = mesh.bounds[1] - mesh.bounds[0]
    for b in buildings:
        if (b.max_xyz[0] - b.min_xyz[0]) > 0.7 * map_span[0] and (b.max_xyz[1] - b.min_xyz[1]) > 0.7 * map_span[1]:
            continue
        filtered.append(b)
    return filtered


def choose_nearest_building(start: np.ndarray, buildings: List[BuildingInfo]) -> Optional[BuildingInfo]:
    if not buildings:
        return None
    xy = start[:2]
    dists = [np.linalg.norm(b.center_xy - xy) for b in buildings]
    idx = int(np.argmin(dists))
    return buildings[idx]


def sample_target_length() -> float:
    # 50~400，mode=150，平均约 200。
    return float(random.triangular(MIN_TRAJ_LENGTH, MAX_TRAJ_LENGTH, 150.0))


def num_points_from_length(length_m: float) -> int:
    steps = max(20, int(round(length_m / STEP_LENGTH)))
    return steps + 1


class SmoothTrajectoryGenerator:
    def __init__(self, bounds: np.ndarray, prox, udf_grid: Optional[UDFGrid], buildings: List[BuildingInfo]):
        self.bounds = bounds
        self.prox = prox
        self.udf_grid = udf_grid
        self.buildings = buildings
        self.min_xyz = bounds[0] + BOUND_MARGIN
        self.max_xyz = bounds[1] - BOUND_MARGIN
        self.min_xyz[2] = max(self.min_xyz[2], MIN_Z_HEIGHT)
        self.max_xyz[2] = min(self.max_xyz[2], MAX_Z_HEIGHT)

    def in_bounds(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self.min_xyz) and np.all(p <= self.max_xyz))

    def clip_to_bounds(self, p: np.ndarray) -> np.ndarray:
        q = p.copy()
        q[0] = np.clip(q[0], self.min_xyz[0], self.max_xyz[0])
        q[1] = np.clip(q[1], self.min_xyz[1], self.max_xyz[1])
        q[2] = np.clip(q[2], self.min_xyz[2], self.max_xyz[2])
        return q

    def _accept_candidate(self, curr: np.ndarray, nxt: np.ndarray) -> bool:
        if not self.in_bounds(nxt):
            return False
        if not is_point_free(nxt, self.prox, SAFETY_RADIUS, self.udf_grid):
            return False
        if not is_segment_free(curr, nxt, self.prox, SAFETY_RADIUS, self.udf_grid):
            return False
        return True

    def _step_with_smoothing(self,
                             curr: np.ndarray,
                             curr_dir: np.ndarray,
                             desired_dir: np.ndarray,
                             max_turn_deg: float = MAX_TURN_DEG,
                             step_len: float = STEP_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
        curr_dir = normalize(curr_dir)
        desired_dir = normalize(desired_dir)
        if np.linalg.norm(curr_dir) < 1e-6:
            curr_dir = desired_dir.copy()

        turn = angle_deg(curr_dir, desired_dir)
        if turn > max_turn_deg:
            alpha = max_turn_deg / max(turn, 1e-6)
            mixed = normalize((1 - alpha) * curr_dir + alpha * desired_dir)
        else:
            mixed = desired_dir

        search_yaws = np.deg2rad(np.array([0, -6, 6, -12, 12, -18, 18, -25, 25, -35, 35], dtype=float))
        search_pitches = np.deg2rad(np.array([0, -4, 4, -8, 8, -12, 12], dtype=float))

        candidates = []
        for yaw in search_yaws:
            for pitch in search_pitches:
                cand_dir = rotate_direction(mixed, yaw, pitch)
                horiz = np.linalg.norm(cand_dir[:2])
                pitch_deg = abs(math.degrees(math.atan2(cand_dir[2], max(horiz, 1e-8))))
                if pitch_deg > MAX_PITCH_DEG:
                    continue
                nxt = curr + step_len * cand_dir
                nxt = self.clip_to_bounds(nxt)
                candidates.append((cand_dir, nxt, angle_deg(curr_dir, cand_dir)))

        candidates.sort(key=lambda x: x[2])
        for cand_dir, nxt, _ in candidates[:SEARCH_TRIES_PER_STEP]:
            if self._accept_candidate(curr, nxt):
                return nxt, cand_dir

        for _ in range(SEARCH_TRIES_PER_STEP):
            yaw = np.deg2rad(random.uniform(-50, 50))
            pitch = np.deg2rad(random.uniform(-12, 12))
            cand_dir = rotate_direction(curr_dir, yaw, pitch)
            nxt = curr + step_len * cand_dir
            nxt = self.clip_to_bounds(nxt)
            if self._accept_candidate(curr, nxt):
                return nxt, cand_dir

        raise RuntimeError("局部平滑扩展失败")

    def _build_easy(self, start: np.ndarray, num_points: int) -> List[np.ndarray]:
        pts = [start.copy()]
        curr = start.copy()
        curr_dir = normalize(np.array([
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-0.15, 0.15)
        ]))
        desired_dir = curr_dir.copy()
        segment_remaining = random.randint(8, 14)

        for i in range(num_points - 1):
            if segment_remaining <= 0:
                yaw_deg = random.choice([-1, 1]) * random.uniform(15, 45)
                pitch_deg = random.uniform(-5, 5)
                desired_dir = rotate_direction(desired_dir, math.radians(yaw_deg), math.radians(pitch_deg))
                desired_dir = normalize(desired_dir)
                segment_remaining = random.randint(8, 14)

            wiggle_yaw = math.radians(6.0 * math.sin(2 * math.pi * i / 18.0))
            wiggle_pitch = math.radians(2.5 * math.sin(2 * math.pi * i / 24.0))
            local_desired = rotate_direction(desired_dir, wiggle_yaw, wiggle_pitch)
            nxt, curr_dir = self._step_with_smoothing(curr, curr_dir, local_desired, max_turn_deg=12.0)
            pts.append(nxt.copy())
            curr = nxt
            segment_remaining -= 1
        return pts

    def _build_medium(self, start: np.ndarray, num_points: int) -> List[np.ndarray]:
        pts = [start.copy()]
        curr = start.copy()

        radius = random.uniform(18.0, 38.0)
        direction_sign = random.choice([-1.0, 1.0])
        climb_amp = random.uniform(0.12, 0.35)
        base_heading = random.uniform(-math.pi, math.pi)
        center = start[:2] + radius * np.array([math.cos(base_heading + math.pi / 2), math.sin(base_heading + math.pi / 2)])
        angle0 = math.atan2(start[1] - center[1], start[0] - center[0])
        dtheta = direction_sign * (STEP_LENGTH / radius)
        curr_dir = normalize(np.array([
            -direction_sign * math.sin(angle0),
            direction_sign * math.cos(angle0),
            climb_amp,
        ]))

        for i in range(num_points - 1):
            theta = angle0 + (i + 1) * dtheta
            tangent = np.array([
                -direction_sign * math.sin(theta),
                direction_sign * math.cos(theta),
                climb_amp * math.cos(2 * math.pi * i / 22.0),
            ], dtype=float)
            tangent = normalize(tangent)
            nxt, curr_dir = self._step_with_smoothing(curr, curr_dir, tangent, max_turn_deg=10.0)
            pts.append(nxt.copy())
            curr = nxt
        return pts

    def _sample_hard_start_near_building(self, max_tries: int = HARD_START_BUILDING_ATTEMPTS) -> Tuple[np.ndarray, BuildingInfo]:
        """
        hard 模式专用起点采样：
        1. 先随机选择建筑；
        2. 再在建筑外圈的安全半径附近采样起点；
        3. 这样比“全图随机采样起点 -> 找最近建筑”成功率高得多。
        """
        if not self.buildings:
            raise RuntimeError("未找到可用于 hard 模式的建筑")

        buildings = self.buildings[:]
        random.shuffle(buildings)
        tries_per_building = max(1, max_tries // max(1, len(buildings)))

        for building in buildings:
            base_radius = max(building.orbit_radius, 16.0)

            z_low = max(
                self.min_xyz[2],
                MIN_Z_HEIGHT,
                building.min_xyz[2] + HARD_START_Z_ABOVE_BUILDING_MIN,
            )
            z_high = min(
                self.max_xyz[2],
                MAX_Z_HEIGHT,
                building.max_xyz[2] + HARD_START_Z_ABOVE_BUILDING_MAX,
            )

            # 有些组件的 bbox 可能高度异常，避免直接放弃建筑。
            if z_low >= z_high:
                z_low = max(self.min_xyz[2], MIN_Z_HEIGHT)
                z_high = min(self.max_xyz[2], MAX_Z_HEIGHT)
            if z_low >= z_high:
                continue

            for _ in range(tries_per_building):
                theta = random.uniform(-math.pi, math.pi)
                radius = base_radius + random.uniform(HARD_START_EXTRA_RADIUS_MIN, HARD_START_EXTRA_RADIUS_MAX)
                p = np.array([
                    building.center_xy[0] + radius * math.cos(theta),
                    building.center_xy[1] + radius * math.sin(theta),
                    random.uniform(z_low, z_high),
                ], dtype=float)

                if not self.in_bounds(p):
                    continue
                if not is_point_free(p, self.prox, SAFETY_RADIUS, self.udf_grid):
                    continue
                return p, building

        raise RuntimeError("无法在建筑附近采样 hard 起点")

    def _build_hard(self,
                    start: np.ndarray,
                    num_points: int,
                    building: Optional[BuildingInfo] = None) -> List[np.ndarray]:
        if building is None:
            building = choose_nearest_building(start, self.buildings)
        if building is None:
            raise RuntimeError("未找到最近建筑，重采样起点")

        pts = [start.copy()]
        curr = start.copy()

        center = building.center_xy.copy()
        rel0 = curr[:2] - center
        rel0_norm = float(np.linalg.norm(rel0))
        if rel0_norm < 1e-6:
            raise RuntimeError("hard 起点与建筑中心过近，重采样起点")

        # 使用起点所在半径绕楼，避免先强行 approach 到固定 east-side 点导致穿楼或过长路径。
        radius = max(rel0_norm, building.orbit_radius, 16.0)
        theta0 = math.atan2(rel0[1], rel0[0])

        total_steps = num_points - 1
        if total_steps < 8:
            raise RuntimeError("hard 模式有效绕楼步数不足，重采样起点")

        direction_sign = random.choice([-1.0, 1.0])
        dtheta = direction_sign * (STEP_LENGTH / max(radius, 1e-6))

        curr_dir = normalize(np.array([
            -direction_sign * math.sin(theta0),
            direction_sign * math.cos(theta0),
            0.0,
        ], dtype=float))
        if np.linalg.norm(curr_dir) < 1e-6:
            curr_dir = normalize(np.array([1.0, 0.0, 0.0], dtype=float))

        # 关键修正：原来的 climb_total=18~35m 在 STEP_LENGTH=1、短轨迹下太激进，
        # 容易让 desired_dir 的 pitch 超过 MAX_PITCH_DEG，导致候选方向全部被过滤。
        max_pitch_slope = math.tan(math.radians(max(1.0, MAX_PITCH_DEG - 3.0)))
        climb_per_step_high = min(HARD_CLIMB_PER_STEP_MAX, max_pitch_slope * 0.8)
        climb_per_step_low = min(HARD_CLIMB_PER_STEP_MIN, climb_per_step_high)
        climb_per_step = random.uniform(climb_per_step_low, climb_per_step_high)
        if curr[2] + climb_per_step * total_steps > self.max_xyz[2] - 2.0:
            climb_per_step = max(0.0, (self.max_xyz[2] - 2.0 - curr[2]) / max(total_steps, 1))

        for j in range(total_steps):
            theta = theta0 + (j + 1) * dtheta
            desired_xy = np.array([
                center[0] + radius * math.cos(theta),
                center[1] + radius * math.sin(theta),
            ], dtype=float)

            tangent = np.array([
                -direction_sign * math.sin(theta),
                direction_sign * math.cos(theta),
                climb_per_step / max(STEP_LENGTH, 1e-6),
            ], dtype=float)
            radial = np.array([desired_xy[0] - curr[0], desired_xy[1] - curr[1], 0.0], dtype=float)
            if np.linalg.norm(radial) < 1e-6:
                radial = tangent.copy()

            desired_dir = normalize(0.82 * normalize(tangent) + 0.18 * normalize(radial))
            nxt, curr_dir = self._step_with_smoothing(curr, curr_dir, desired_dir, max_turn_deg=10.0)
            pts.append(nxt.copy())
            curr = nxt

        return pts

    def _build_hard_without_building(self, start: np.ndarray, num_points: int) -> List[np.ndarray]:
        """
        无建筑城市的 hard fallback。

        目的：
        - 当 extract_buildings() 没有提取到建筑时，仍然生成 hard 样本；
        - 不再依赖绕楼，而是使用更频繁、更大幅的连续转向、轻微爬升/俯冲和正弦扰动；
        - 仍然通过 _step_with_smoothing() 做边界、UDF 安全距离和段碰撞检查。
        """
        pts = [start.copy()]
        curr = start.copy()

        curr_dir = normalize(np.array([
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-0.05, 0.10),
        ], dtype=float))
        if np.linalg.norm(curr_dir) < 1e-6:
            curr_dir = np.array([1.0, 0.0, 0.0], dtype=float)

        desired_dir = curr_dir.copy()
        segment_remaining = random.randint(HARD_FALLBACK_SEGMENT_MIN, HARD_FALLBACK_SEGMENT_MAX)
        vertical_bias = random.uniform(-0.04, 0.10)

        for i in range(num_points - 1):
            if segment_remaining <= 0:
                yaw_abs = random.uniform(HARD_FALLBACK_YAW_DEG_MIN, HARD_FALLBACK_YAW_DEG_MAX)
                yaw_deg = random.choice([-1.0, 1.0]) * yaw_abs
                pitch_deg = random.uniform(-HARD_FALLBACK_PITCH_DEG_MAX, HARD_FALLBACK_PITCH_DEG_MAX)
                desired_dir = rotate_direction(desired_dir, math.radians(yaw_deg), math.radians(pitch_deg))
                desired_dir[2] += vertical_bias
                desired_dir = normalize(desired_dir)
                segment_remaining = random.randint(HARD_FALLBACK_SEGMENT_MIN, HARD_FALLBACK_SEGMENT_MAX)

                # 防止长时间持续上升/下降导致撞高度边界。
                if curr[2] > self.max_xyz[2] - 12.0:
                    vertical_bias = random.uniform(-0.10, 0.0)
                elif curr[2] < self.min_xyz[2] + 12.0:
                    vertical_bias = random.uniform(0.0, 0.10)
                else:
                    vertical_bias = random.uniform(-0.04, 0.10)

            wiggle_yaw = math.radians(HARD_FALLBACK_WIGGLE_YAW_DEG * math.sin(2 * math.pi * i / 11.0))
            wiggle_pitch = math.radians(HARD_FALLBACK_WIGGLE_PITCH_DEG * math.sin(2 * math.pi * i / 17.0))
            local_desired = rotate_direction(desired_dir, wiggle_yaw, wiggle_pitch)

            # 接近高度边界时主动把方向拉回安全高度区间。
            if curr[2] > self.max_xyz[2] - 6.0:
                local_desired[2] = min(local_desired[2], -0.05)
            elif curr[2] < self.min_xyz[2] + 6.0:
                local_desired[2] = max(local_desired[2], 0.05)
            local_desired = normalize(local_desired)

            nxt, curr_dir = self._step_with_smoothing(curr, curr_dir, local_desired, max_turn_deg=14.0)
            pts.append(nxt.copy())
            curr = nxt
            segment_remaining -= 1

        return pts

    def build_target_trajectory(self, mode: str, num_points: int) -> List[np.ndarray]:
        last_error = None
        error_counts: Dict[str, int] = {}

        for _ in range(MAX_TRAJ_BUILD_RETRY):
            try:
                if mode == "hard" and FAST_HARD_START_NEAR_BUILDING:
                    if self.buildings:
                        start, building = self._sample_hard_start_near_building()
                        traj = self._build_hard(start, num_points, building=building)
                    elif HARD_FALLBACK_WHEN_NO_BUILDING:
                        start = sample_free_point(self.bounds, self.prox, SAFETY_RADIUS, self.udf_grid)
                        traj = self._build_hard_without_building(start, num_points)
                    else:
                        raise RuntimeError("未找到可用于 hard 模式的建筑")
                else:
                    start = sample_free_point(self.bounds, self.prox, SAFETY_RADIUS, self.udf_grid)
                    if mode == "easy":
                        traj = self._build_easy(start, num_points)
                    elif mode == "medium":
                        traj = self._build_medium(start, num_points)
                    elif mode == "hard":
                        traj = self._build_hard(start, num_points)
                    else:
                        raise ValueError(f"未知模式: {mode}")

                if len(traj) == num_points:
                    return traj

            except RuntimeError as e:
                msg = str(e)
                last_error = msg
                error_counts[msg] = error_counts.get(msg, 0) + 1
                continue

        if mode == "hard":
            raise RuntimeError(
                f"{mode} 模式轨迹生成失败，已重试 {MAX_TRAJ_BUILD_RETRY} 次；"
                f"last_error={last_error}; error_counts={error_counts}"
            )

        raise RuntimeError(
            f"{mode} 模式轨迹生成失败，已重试 {MAX_TRAJ_BUILD_RETRY} 次；"
            f"last_error={last_error}; error_counts={error_counts}"
        )


def validate_tracker_fpv(target_traj: List[np.ndarray], tracker_traj: List[np.ndarray],
                         prox, bounds: np.ndarray, udf_grid: Optional[UDFGrid]) -> bool:
    if len(target_traj) != len(tracker_traj):
        return False

    min_xyz = bounds[0] + BOUND_MARGIN
    max_xyz = bounds[1] - BOUND_MARGIN
    min_xyz[2] = max(min_xyz[2], MIN_Z_HEIGHT)
    max_xyz[2] = min(max_xyz[2], MAX_Z_HEIGHT)
    cos_thresh = math.cos(math.radians(TRACKER_FPV_HALF_ANGLE_DEG))

    start_dist = float(np.linalg.norm(target_traj[0] - tracker_traj[0]))
    if not (START_SEPARATION_MIN - 1e-6 <= start_dist <= START_SEPARATION_MAX + 1e-6):
        return False

    tracker_dirs = estimate_forward_dirs(tracker_traj)

    for i, (t, p) in enumerate(zip(target_traj, tracker_traj)):
        if p[2] < MIN_Z_HEIGHT or p[2] > MAX_Z_HEIGHT:
            return False
        if t[2] < MIN_Z_HEIGHT or t[2] > MAX_Z_HEIGHT:
            return False
        if t[2] + 1e-6 < p[2]:
            return False
        if np.any(p < min_xyz) or np.any(p > max_xyz):
            return False
        if np.any(t < min_xyz) or np.any(t > max_xyz):
            return False
        if not is_point_free(p, prox, SAFETY_RADIUS, udf_grid):
            return False
        if i > 0 and not is_segment_free(tracker_traj[i - 1], tracker_traj[i], prox, SAFETY_RADIUS, udf_grid):
            return False

        rel = t - p

        if i < len(target_traj) - 1:
            fwd = tracker_dirs[i]
            rel_dir = normalize(rel)
            if np.linalg.norm(rel_dir) < 1e-6:
                return False
            if float(np.dot(fwd, rel_dir)) < cos_thresh:
                return False

            step_dir = normalize(tracker_traj[i + 1] - tracker_traj[i])
            if np.linalg.norm(step_dir) < 1e-6:
                return False
            if float(np.dot(fwd, step_dir)) <= 0.0:
                return False

    return True



def _horizontal_unit(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    """取水平面方向，避免 target 的竖直运动把 tracker 初始点推到目标上方/下方过多。"""
    h = np.array([float(v[0]), float(v[1]), 0.0], dtype=float)
    if np.linalg.norm(h) >= 1e-6:
        return normalize(h)
    if fallback is not None:
        hf = np.array([float(fallback[0]), float(fallback[1]), 0.0], dtype=float)
        if np.linalg.norm(hf) >= 1e-6:
            return normalize(hf)
    return np.array([1.0, 0.0, 0.0], dtype=float)


def _rotate_horizontal_dir(h: np.ndarray, yaw_rad: float) -> np.ndarray:
    """绕 z 轴旋转水平单位方向。"""
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return normalize(np.array([
        c * h[0] - s * h[1],
        s * h[0] + c * h[1],
        0.0,
    ], dtype=float))


def _tracker_bounds(bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    min_xyz = bounds[0] + BOUND_MARGIN
    max_xyz = bounds[1] - BOUND_MARGIN
    min_xyz[2] = max(min_xyz[2], MIN_Z_HEIGHT)
    max_xyz[2] = min(max_xyz[2], MAX_Z_HEIGHT)
    return min_xyz, max_xyz


def _point_in_tracker_bounds(p: np.ndarray, min_xyz: np.ndarray, max_xyz: np.ndarray) -> bool:
    return bool(np.all(p >= min_xyz) and np.all(p <= max_xyz))


def _build_tracker_offset_follow(target_traj: List[np.ndarray],
                                 prox,
                                 bounds: np.ndarray,
                                 udf_grid: Optional[UDFGrid]) -> List[np.ndarray]:
    """
    快速构造 tracker：让 tracker 以固定 5~10m 距离跟随在 target 运动方向后方。

    这个分支的目的不是关闭 STEP_LENGTH，而是避免原先“逐步直追 + 严格 z 约束”导致大量失败重试。
    生成后仍通过 validate_tracker_fpv() 检查：起始距离、FPV、z、自由空间、段碰撞。
    """
    if len(target_traj) < 2:
        raise RuntimeError("目标轨迹过短，无法构造 tracker")

    min_xyz, max_xyz = _tracker_bounds(bounds)
    target_dirs = estimate_forward_dirs(target_traj)
    base_h = _horizontal_unit(target_dirs[0], target_traj[1] - target_traj[0])

    min_target_z = min(float(p[2]) for p in target_traj)
    max_allowed_z_drop = max(0.0, min(TRACKER_MAX_Z_DROP, min_target_z - min_xyz[2]))

    for _ in range(TRACKER_OFFSET_FOLLOW_ATTEMPTS):
        start_dist = random.uniform(START_SEPARATION_MIN, START_SEPARATION_MAX)
        z_drop = random.uniform(0.0, min(max_allowed_z_drop, start_dist * 0.5))
        xy_dist = math.sqrt(max(start_dist * start_dist - z_drop * z_drop, 0.0))
        yaw_jitter = math.radians(random.uniform(-TRACKER_START_YAW_JITTER_DEG, TRACKER_START_YAW_JITTER_DEG))

        tracker: List[np.ndarray] = []
        ok = True
        prev_h = base_h

        for i, target_pos in enumerate(target_traj):
            h = _horizontal_unit(target_dirs[i], prev_h)
            h = _rotate_horizontal_dir(h, yaw_jitter)
            prev_h = h

            p = np.array([
                target_pos[0] - xy_dist * h[0],
                target_pos[1] - xy_dist * h[1],
                target_pos[2] - z_drop,
            ], dtype=float)

            # 不做 clip。clip 会破坏起始距离，导致最后 validate 才失败，浪费时间。
            if not _point_in_tracker_bounds(p, min_xyz, max_xyz):
                ok = False
                break
            if p[2] > target_pos[2] + 1e-6:
                ok = False
                break
            if not is_point_free(p, prox, SAFETY_RADIUS, udf_grid):
                ok = False
                break
            if tracker and not is_segment_free(tracker[-1], p, prox, SAFETY_RADIUS, udf_grid):
                ok = False
                break

            tracker.append(p)

        if ok and len(tracker) == len(target_traj):
            if validate_tracker_fpv(target_traj, tracker, prox, bounds, udf_grid):
                return tracker

    raise RuntimeError("offset-follow tracker 构造失败")


def _sample_tracker_start(target_traj: List[np.ndarray],
                          init_dir: np.ndarray,
                          min_xyz: np.ndarray,
                          max_xyz: np.ndarray,
                          prox,
                          udf_grid: Optional[UDFGrid]) -> Optional[np.ndarray]:
    """采样满足 5~10m 起始距离、z 不高于 target、且不被边界 clip 破坏的 tracker 起点。"""
    h = _horizontal_unit(init_dir, target_traj[1] - target_traj[0])
    target0 = target_traj[0]

    start_dist = random.uniform(START_SEPARATION_MIN, START_SEPARATION_MAX)
    available_drop = max(0.0, float(target0[2] - min_xyz[2]))
    z_drop = random.uniform(0.0, min(TRACKER_MAX_Z_DROP, available_drop, start_dist * 0.5))
    xy_dist = math.sqrt(max(start_dist * start_dist - z_drop * z_drop, 0.0))

    h = _rotate_horizontal_dir(
        h,
        math.radians(random.uniform(-TRACKER_START_YAW_JITTER_DEG, TRACKER_START_YAW_JITTER_DEG)),
    )

    tracker0 = np.array([
        target0[0] - xy_dist * h[0],
        target0[1] - xy_dist * h[1],
        target0[2] - z_drop,
    ], dtype=float)

    if not _point_in_tracker_bounds(tracker0, min_xyz, max_xyz):
        return None
    if tracker0[2] > target0[2] + 1e-6:
        return None
    start_distance = float(np.linalg.norm(target0 - tracker0))
    if not (START_SEPARATION_MIN - 1e-6 <= start_distance <= START_SEPARATION_MAX + 1e-6):
        return None
    if not is_point_free(tracker0, prox, SAFETY_RADIUS, udf_grid):
        return None
    return tracker0


def _build_tracker_direct_chase(target_traj: List[np.ndarray],
                                prox,
                                bounds: np.ndarray,
                                udf_grid: Optional[UDFGrid]) -> List[np.ndarray]:
    """
    原始直追逻辑的稳健版本：
    - 起点不再 clip，避免破坏起始距离；
    - z_drop 按可用高度自适应，避免 target 接近 MIN_Z_HEIGHT 时无解；
    - 每步用当前 target / lookahead target 作为多个候选方向，减少因为下降或转向导致的失败。
    """
    if len(target_traj) < 2:
        raise RuntimeError("目标轨迹过短，无法构造 tracker")

    init_dir = normalize(target_traj[1] - target_traj[0])
    if np.linalg.norm(init_dir) < 1e-6:
        raise RuntimeError("目标轨迹初始方向无效")

    min_xyz, max_xyz = _tracker_bounds(bounds)
    cos_thresh = math.cos(math.radians(TRACKER_FPV_HALF_ANGLE_DEG))

    for _ in range(TRACKER_DIRECT_CHASE_ATTEMPTS):
        tracker0 = _sample_tracker_start(target_traj, init_dir, min_xyz, max_xyz, prox, udf_grid)
        if tracker0 is None:
            continue

        tracker: List[np.ndarray] = [tracker0]
        ok = True

        for i in range(len(target_traj) - 1):
            curr = tracker[-1]
            target_now = target_traj[i]
            target_next = target_traj[min(i + TRACKER_DIRECT_LOOKAHEAD, len(target_traj) - 1)]

            rel_now = target_now - curr
            rel_dir_now = normalize(rel_now)
            if np.linalg.norm(rel_dir_now) < 1e-6:
                ok = False
                break

            max_next_z = float(target_traj[i + 1][2])

            capped_now = np.array(target_now, dtype=float)
            capped_now[2] = min(capped_now[2], max_next_z)
            capped_next = np.array(target_next, dtype=float)
            capped_next[2] = min(capped_next[2], max_next_z)

            aim_points = [
                target_now,
                target_next,
                capped_now,
                capped_next,
                0.5 * target_now + 0.5 * target_next,
            ]

            candidates = []
            for aim in aim_points:
                move_dir = normalize(np.array(aim, dtype=float) - curr)
                if np.linalg.norm(move_dir) < 1e-6:
                    continue
                # 当前 step 方向应大致朝向当前目标，保证 FPV 不容易失败。
                if float(np.dot(move_dir, rel_dir_now)) < cos_thresh:
                    continue
                nxt = curr + STEP_LENGTH * move_dir
                candidates.append((float(np.dot(move_dir, rel_dir_now)), nxt, move_dir))

            candidates.sort(key=lambda x: -x[0])

            accepted = False
            for _, nxt, _ in candidates:
                if nxt[2] > max_next_z + 1e-6:
                    continue
                if not _point_in_tracker_bounds(nxt, min_xyz, max_xyz):
                    continue
                if not is_segment_free(curr, nxt, prox, SAFETY_RADIUS, udf_grid):
                    continue
                tracker.append(np.array(nxt, dtype=float))
                accepted = True
                break

            if not accepted:
                ok = False
                break

        if ok and len(tracker) == len(target_traj):
            if validate_tracker_fpv(target_traj, tracker, prox, bounds, udf_grid):
                return tracker

    raise RuntimeError("direct-chase tracker 构造失败")


def build_tracker_from_target(target_traj: List[np.ndarray], prox, bounds: np.ndarray,
                              udf_grid: Optional[UDFGrid]) -> List[np.ndarray]:
    """
    构造 tracker 轨迹。

    先用 offset-follow 快速构造，失败后再回退到 direct-chase。
    这样可以显著降低日志里大量 “无法构造 tracker 轨迹” 的外层重试。
    """
    errors = []

    if FAST_TRACKER_OFFSET_FOLLOW:
        try:
            return _build_tracker_offset_follow(target_traj, prox, bounds, udf_grid)
        except RuntimeError as e:
            errors.append(str(e))

    try:
        return _build_tracker_direct_chase(target_traj, prox, bounds, udf_grid)
    except RuntimeError as e:
        errors.append(str(e))

    raise RuntimeError("无法构造 tracker 轨迹: " + " | ".join(errors))

def calculate_trajectory_length(trajectory: List[np.ndarray]) -> float:
    total = 0.0
    for i in range(1, len(trajectory)):
        total += float(np.linalg.norm(trajectory[i] - trajectory[i - 1]))
    return total


def save_visualization(output_png: str,
                       tracker_traj: List[np.ndarray],
                       target_traj: List[np.ndarray],
                       jammer_trajectories: Optional[List[Dict[str, Any]]],
                       mode: str):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    t = np.array(target_traj)
    p = np.array(tracker_traj)

    ax.plot(t[:, 0], t[:, 1], t[:, 2], "r-", linewidth=2.0, label="Evader")
    ax.plot(p[:, 0], p[:, 1], p[:, 2], "b-", linewidth=2.0, label="Tracker")

    jammer_arrays = []
    jammer_colors = {1: ("m-", "magenta", "purple"), 2: ("c-", "cyan", "teal"), 3: ("y-", "yellow", "gold"), 4: ("g-", "lime", "darkgreen"), 5: ("k-", "black", "dimgray")}
    if jammer_trajectories is not None:
        for jammer in jammer_trajectories:
            traj = jammer.get("trajectory")
            if traj is None or len(traj) == 0:
                continue
            arr = np.array(traj)
            jammer_arrays.append(arr)
            line_style, start_color, end_color = jammer_colors.get(jammer["id"], ("m-", "magenta", "purple"))
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], line_style, linewidth=1.8, label=f"J{jammer['id']}-{jammer['type']}")
            ax.scatter(arr[0, 0], arr[0, 1], arr[0, 2], c=start_color, s=45, label=f"J{jammer['id']} Start")
            ax.scatter(arr[-1, 0], arr[-1, 1], arr[-1, 2], c=end_color, s=45, marker="s", label=f"J{jammer['id']} End")

    ax.scatter(t[0, 0], t[0, 1], t[0, 2], c="orange", s=80, label="Evader Start")
    ax.scatter(p[0, 0], p[0, 1], p[0, 2], c="green", s=80, label="Tracker Start")
    ax.scatter(t[-1, 0], t[-1, 1], t[-1, 2], c="red", s=80, marker="s", label="Evader End")
    ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], c="navy", s=80, marker="s", label="Tracker End")

    pts_to_stack = [t, p] + jammer_arrays
    all_pts = np.vstack(pts_to_stack)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    ranges = np.maximum(maxs - mins, np.array([20.0, 20.0, 20.0]))
    margin = 0.1 * ranges

    ax.set_xlim(centers[0] - ranges[0] / 2 - margin[0], centers[0] + ranges[0] / 2 + margin[0])
    ax.set_ylim(centers[1] - ranges[1] / 2 - margin[1], centers[1] + ranges[1] / 2 + margin[1])
    ax.set_zlim(centers[2] - ranges[2] / 2 - margin[2], centers[2] + ranges[2] / 2 + margin[2])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Three-mode escape trajectory | mode={mode}")
    ax.legend(loc="upper left")
    ax.view_init(elev=24, azim=48)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)


def build_record(map_name: str, idx: int, mode: str,
                 target_traj: List[np.ndarray],
                 tracker_traj: List[np.ndarray],
                 jammer_trajectories: Optional[List[Dict[str, Any]]],
                 desired_length: float) -> Dict[str, Any]:
    target_length = calculate_trajectory_length(target_traj)
    tracker_length = calculate_trajectory_length(tracker_traj)
    jammer_lengths = {}
    if jammer_trajectories is not None:
        for jammer in jammer_trajectories:
            jammer_lengths[str(jammer["id"])] = calculate_trajectory_length(jammer["trajectory"])

    start_distance = float(np.linalg.norm(target_traj[0] - tracker_traj[0]))
    frame_distances = [float(np.linalg.norm(t - p)) for t, p in zip(target_traj, tracker_traj)]

    frames = []
    for frame_idx, (target_pos, uav_pos, dist) in enumerate(zip(target_traj, tracker_traj, frame_distances)):
        frame = {
            "frame_id": frame_idx,
            "target_position": [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])],
            "uav_position": [float(uav_pos[0]), float(uav_pos[1]), float(uav_pos[2])],
            "distance": dist,
            "jammers": [],
        }
        if jammer_trajectories is not None:
            for jammer in jammer_trajectories:
                jammer_pos = jammer["trajectory"][frame_idx]
                visible_mask = jammer.get("visible_expected_mask", [False] * len(jammer["trajectory"]))
                jammer_item = {
                    "id": int(jammer["id"]),
                    "type": jammer["type"],
                    "position": [float(jammer_pos[0]), float(jammer_pos[1]), float(jammer_pos[2])],
                    "distance_to_uav": float(np.linalg.norm(jammer_pos - uav_pos)),
                    "distance_to_target": float(np.linalg.norm(jammer_pos - target_pos)),
                    "visible_expected": bool(visible_mask[frame_idx]),
                }
                frame["jammers"].append(jammer_item)

            primary = next((j for j in jammer_trajectories if j["id"] == 1), None)
            if primary is not None:
                jammer_pos = primary["trajectory"][frame_idx]
                visible_mask = primary.get("visible_expected_mask", [False] * len(primary["trajectory"]))
                frame["jammer_position"] = [float(jammer_pos[0]), float(jammer_pos[1]), float(jammer_pos[2])]
                frame["jammer_distance_to_uav"] = float(np.linalg.norm(jammer_pos - uav_pos))
                frame["jammer_distance_to_target"] = float(np.linalg.norm(jammer_pos - target_pos))
                frame["jammer_visible_expected"] = bool(visible_mask[frame_idx])
        frames.append(frame)

    jammers_meta = []
    if jammer_trajectories is not None:
        for jammer in jammer_trajectories:
            jammers_meta.append({
                "id": int(jammer["id"]),
                "type": jammer["type"],
                "num_points": len(jammer["trajectory"]),
                "trajectory_length": jammer_lengths[str(jammer["id"])],
                "active_start_frame": int(jammer.get("active_start_frame", 0)),
                "active_end_frame": int(jammer.get("active_end_frame", len(jammer["trajectory"]) - 1)),
            })

    record = {
        "map_name": map_name,
        "trajectory_id": idx,
        "mode": mode,
        "num_points": len(target_traj),
        "time_step": TIME_STEP,
        "target_speed_nominal": TARGET_SPEED,
        "tracker_speed_nominal": TRACKER_SPEED,
        "jammer_enabled": jammer_trajectories is not None and len(jammer_trajectories) > 0,
        "num_jammers": 0 if jammer_trajectories is None else len(jammer_trajectories),
        "jammer_ids": [] if jammer_trajectories is None else [int(j["id"]) for j in jammer_trajectories],
        "jammer_speed_nominal": JAMMER_SPEED if jammer_trajectories is not None else 0.0,
        "tracker_fpv_half_angle_deg": TRACKER_FPV_HALF_ANGLE_DEG,
        "jammer_vertical_half_angle_deg": JAMMER_VERTICAL_HALF_ANGLE_DEG if jammer_trajectories is not None else 0.0,
        "desired_target_length": desired_length,
        "target_length": target_length,
        "tracker_length": tracker_length,
        "jammer_lengths": jammer_lengths,
        "start_distance": start_distance,
        "jammers_metadata": jammers_meta,
        "frames": frames,
    }
    return record


def init_worker_for_city(obj_path: str):
    global _PROCESS_DATA
    mesh, prox, bounds, udf_grid = load_map(obj_path, use_udf=USE_UDF)
    buildings = extract_buildings(mesh)
    print(f"[buildings] {os.path.basename(obj_path)}: extracted {len(buildings)} usable buildings")
    _PROCESS_DATA = {
        "mesh": mesh,
        "prox": prox,
        "bounds": bounds,
        "udf_grid": udf_grid,
        "buildings": buildings,
    }


def generate_single_trajectory(task: Tuple[int, str, int, str, str, int]) -> Dict[str, Any]:
    idx, mode, num_jammers, map_name, map_output_dir, seed = task
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))

    global _PROCESS_DATA
    if _PROCESS_DATA is None:
        return {
            "success": False,
            "idx": idx,
            "mode": mode,
            "error": "worker 未初始化地图",
        }

    prox = _PROCESS_DATA["prox"]
    bounds = _PROCESS_DATA["bounds"]
    udf_grid = _PROCESS_DATA["udf_grid"]
    buildings = _PROCESS_DATA["buildings"]

    generator = SmoothTrajectoryGenerator(bounds, prox, udf_grid, buildings)

    try:
        for outer_try in range(MAX_TRAJ_BUILD_RETRY):
            try:
                desired_length = sample_target_length()
                num_points = num_points_from_length(desired_length)
                target_traj = generator.build_target_trajectory(mode, num_points)
                tracker_traj = build_tracker_from_target(target_traj, prox, bounds, udf_grid)
                jammer_trajectories = build_all_jammer_trajectories(
                    target_traj, tracker_traj, generator, prox, bounds, udf_grid, num_jammers
                ) if JAMMER_ENABLED else None
                break
            except RuntimeError as e:
                if outer_try % 10 == 0:
                    print(f"[retry] idx={idx}, mode={mode}, num_jammers={num_jammers}, try={outer_try}, error={e}")
                if outer_try == MAX_TRAJ_BUILD_RETRY - 1:
                    raise
                continue

        record = build_record(map_name, idx, mode, target_traj, tracker_traj, jammer_trajectories, desired_length)

        out_json = os.path.join(map_output_dir, f"trajectory_{idx:04d}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        if SAVE_PNG and (idx % PLOT_EVERY == 0 or idx <= 5):
            out_png = os.path.join(map_output_dir, f"trajectory_{idx:04d}.png")
            save_visualization(out_png, tracker_traj, target_traj, jammer_trajectories, mode)

        return {
            "success": True,
            "idx": idx,
            "mode": mode,
            "num_jammers": 0 if jammer_trajectories is None else len(jammer_trajectories),
            "target_length": record["target_length"],
        }
    except Exception as e:
        return {
            "success": False,
            "idx": idx,
            "mode": mode,
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(limit=3),
        }


def list_mode_tasks() -> List[str]:
    mode_list: List[str] = []
    for mode, count in MODE_COUNTS.items():
        mode_list.extend([mode] * count)
    if len(mode_list) != NUM_TRAJECTORIES:
        raise ValueError(f"MODE_COUNTS 总数 {len(mode_list)} != NUM_TRAJECTORIES {NUM_TRAJECTORIES}")
    random.shuffle(mode_list)
    return mode_list


def list_jammer_count_tasks() -> List[int]:
    count_list: List[int] = []
    base = NUM_TRAJECTORIES // 5
    remainder = NUM_TRAJECTORIES % 5

    for jammer_count in range(1, 6):
        count_list.extend([jammer_count] * base)

    if remainder > 0:
        extra_counts = list(range(1, 6))
        random.shuffle(extra_counts)
        count_list.extend(extra_counts[:remainder])

    if len(count_list) != NUM_TRAJECTORIES:
        raise ValueError(f"jammer 数量任务总数 {len(count_list)} != NUM_TRAJECTORIES {NUM_TRAJECTORIES}")

    random.shuffle(count_list)
    return count_list


def maybe_existing_ok(city_output_dir: str, idx: int, expected_num_jammers: Optional[int] = None) -> bool:
    out_json = os.path.join(city_output_dir, f"trajectory_{idx:04d}.json")
    out_png = os.path.join(city_output_dir, f"trajectory_{idx:04d}.png")
    if not os.path.exists(out_json):
        return False
    if SAVE_PNG and not os.path.exists(out_png):
        return False

    # 如果传入 expected_num_jammers，则避免旧数据数量不匹配时被误跳过。
    if expected_num_jammers is not None:
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if int(data.get("num_jammers", -1)) != int(expected_num_jammers):
                return False
        except Exception:
            return False

    return True


def read_target_length_from_json(json_path: str) -> Optional[float]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "target_length" in data:
            return float(data["target_length"])
    except Exception:
        return None
    return None


def summarize_city_output(city_output_dir: str, map_name: str) -> Dict[str, Any]:
    lengths = []
    found = 0
    jammer_count_distribution = {str(i): 0 for i in range(1, 6)}
    for idx in range(1, NUM_TRAJECTORIES + 1):
        json_path = os.path.join(city_output_dir, f"trajectory_{idx:04d}.json")
        if os.path.exists(json_path):
            found += 1
            length = read_target_length_from_json(json_path)
            if length is not None:
                lengths.append(length)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                nj = int(data.get("num_jammers", 0))
                if 1 <= nj <= 5:
                    jammer_count_distribution[str(nj)] += 1
            except Exception:
                pass

    meta = {
        "map_name": map_name,
        "num_trajectories": found,
        "mode_counts": MODE_COUNTS,
        "jammer_count_distribution": jammer_count_distribution,
        "mean_target_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_target_length": float(np.std(lengths)) if lengths else 0.0,
        "config": {
            "MEAN_TRAJ_LENGTH": MEAN_TRAJ_LENGTH,
            "MIN_TRAJ_LENGTH": MIN_TRAJ_LENGTH,
            "MAX_TRAJ_LENGTH": MAX_TRAJ_LENGTH,
            "length_sampling": "triangular(low=50, high=150, mode=150)",
            "STEP_LENGTH": STEP_LENGTH,
            "START_SEPARATION_MIN": START_SEPARATION_MIN,
            "START_SEPARATION_MAX": START_SEPARATION_MAX,
            "SAFETY_RADIUS": SAFETY_RADIUS,
            "MIN_Z_HEIGHT": MIN_Z_HEIGHT,
            "MAX_Z_HEIGHT": MAX_Z_HEIGHT,
            "TARGET_SPEED": TARGET_SPEED,
            "TRACKER_SPEED": TRACKER_SPEED,
            "TRACKER_FPV_HALF_ANGLE_DEG": TRACKER_FPV_HALF_ANGLE_DEG,
            "FAST_JAMMER_VISIBLE_SEGMENT": FAST_JAMMER_VISIBLE_SEGMENT,
            "JAMMER_FAST_CANDIDATE_TRIES": JAMMER_FAST_CANDIDATE_TRIES,
            "JAMMER_FAST_LOS_CHECK": JAMMER_FAST_LOS_CHECK,
            "JAMMER_FAST_PREV_SEGMENT_CHECK": JAMMER_FAST_PREV_SEGMENT_CHECK,
            "FAST_TRACKER_OFFSET_FOLLOW": FAST_TRACKER_OFFSET_FOLLOW,
            "TRACKER_OFFSET_FOLLOW_ATTEMPTS": TRACKER_OFFSET_FOLLOW_ATTEMPTS,
            "TRACKER_DIRECT_CHASE_ATTEMPTS": TRACKER_DIRECT_CHASE_ATTEMPTS,
            "FAST_HARD_START_NEAR_BUILDING": FAST_HARD_START_NEAR_BUILDING,
            "HARD_START_BUILDING_ATTEMPTS": HARD_START_BUILDING_ATTEMPTS,
            "HARD_CLIMB_PER_STEP_MIN": HARD_CLIMB_PER_STEP_MIN,
            "HARD_CLIMB_PER_STEP_MAX": HARD_CLIMB_PER_STEP_MAX,
            "HARD_FALLBACK_WHEN_NO_BUILDING": HARD_FALLBACK_WHEN_NO_BUILDING,
            "HARD_FALLBACK_SEGMENT_MIN": HARD_FALLBACK_SEGMENT_MIN,
            "HARD_FALLBACK_SEGMENT_MAX": HARD_FALLBACK_SEGMENT_MAX,
            "HARD_FALLBACK_YAW_DEG_MIN": HARD_FALLBACK_YAW_DEG_MIN,
            "HARD_FALLBACK_YAW_DEG_MAX": HARD_FALLBACK_YAW_DEG_MAX,
            "NUM_WORKERS": NUM_WORKERS,
            "SAVE_PNG": SAVE_PNG,
            "PLOT_EVERY": PLOT_EVERY,
        },
    }
    with open(os.path.join(city_output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def iter_with_progress(iterable, total: int, desc: str):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, ncols=100)
    # fallback
    def generator():
        done = 0
        print(f"{desc}: 0/{total}")
        for item in iterable:
            done += 1
            if done == 1 or done % max(1, total // 20) == 0 or done == total:
                print(f"{desc}: {done}/{total}")
            yield item
    return generator()


def generate_dataset_for_city(map_name: str):
    obj_path = os.path.join(MAP_DIR, f"{map_name}.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"地图不存在: {obj_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    city_output_dir = os.path.join(OUTPUT_DIR, map_name)
    os.makedirs(city_output_dir, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    mode_list = list_mode_tasks()
    jammer_count_list = list_jammer_count_tasks() if JAMMER_ENABLED else [0] * NUM_TRAJECTORIES

    tasks: List[Tuple[int, str, int, str, str, int]] = []
    skipped = 0
    base_seed = RANDOM_SEED * 100000 + abs(hash(map_name)) % 100000

    for idx, (mode, num_jammers) in enumerate(zip(mode_list, jammer_count_list), start=1):
        if RESUME_IF_EXISTS and maybe_existing_ok(city_output_dir, idx, expected_num_jammers=num_jammers):
            skipped += 1
            continue
        seed = base_seed + idx
        tasks.append((idx, mode, num_jammers, map_name, city_output_dir, seed))

    print(f"\n===== {map_name} =====")
    print(f"输出目录: {city_output_dir}")
    planned_dist = {str(i): jammer_count_list.count(i) for i in range(1, 6)} if JAMMER_ENABLED else {"0": NUM_TRAJECTORIES}
    print(f"总目标数: {NUM_TRAJECTORIES} | 已跳过: {skipped} | 待生成: {len(tasks)} | 进程数: {NUM_WORKERS}")
    print(f"计划 jammer 数量分布: {planned_dist}")
    print(f"FAST_JAMMER_VISIBLE_SEGMENT={FAST_JAMMER_VISIBLE_SEGMENT}, "
          f"JAMMER_FAST_CANDIDATE_TRIES={JAMMER_FAST_CANDIDATE_TRIES}, "
          f"JAMMER_FAST_LOS_CHECK={JAMMER_FAST_LOS_CHECK}, "
          f"JAMMER_FAST_PREV_SEGMENT_CHECK={JAMMER_FAST_PREV_SEGMENT_CHECK}")
    print(f"FAST_TRACKER_OFFSET_FOLLOW={FAST_TRACKER_OFFSET_FOLLOW}, "
          f"TRACKER_OFFSET_FOLLOW_ATTEMPTS={TRACKER_OFFSET_FOLLOW_ATTEMPTS}, "
          f"TRACKER_DIRECT_CHASE_ATTEMPTS={TRACKER_DIRECT_CHASE_ATTEMPTS}")
    print(f"FAST_HARD_START_NEAR_BUILDING={FAST_HARD_START_NEAR_BUILDING}, "
          f"HARD_START_BUILDING_ATTEMPTS={HARD_START_BUILDING_ATTEMPTS}, "
          f"HARD_CLIMB_PER_STEP=({HARD_CLIMB_PER_STEP_MIN}, {HARD_CLIMB_PER_STEP_MAX})")
    print(f"HARD_FALLBACK_WHEN_NO_BUILDING={HARD_FALLBACK_WHEN_NO_BUILDING}, "
          f"HARD_FALLBACK_SEGMENT=({HARD_FALLBACK_SEGMENT_MIN}, {HARD_FALLBACK_SEGMENT_MAX}), "
          f"HARD_FALLBACK_YAW=({HARD_FALLBACK_YAW_DEG_MIN}, {HARD_FALLBACK_YAW_DEG_MAX})")

    results = []
    failures = []

    if tasks:
        with mp.Pool(processes=NUM_WORKERS, initializer=init_worker_for_city, initargs=(obj_path,)) as pool:
            iterator = pool.imap_unordered(generate_single_trajectory, tasks, chunksize=1)
            for res in iter_with_progress(iterator, total=len(tasks), desc=f"{map_name}"):
                results.append(res)
                if not res["success"]:
                    failures.append(res)

    meta = summarize_city_output(city_output_dir, map_name)
    success_count = meta["num_trajectories"]

    print(f"{map_name} 完成 | 总已生成: {success_count}/{NUM_TRAJECTORIES} | 目标均长: {meta['mean_target_length']:.2f} m")
    print(f"{map_name} jammer 分布: {meta['jammer_count_distribution']}")
    if failures:
        err_log = os.path.join(city_output_dir, "failed_cases.json")
        with open(err_log, "w", encoding="utf-8") as f:
            json.dump(failures[:200], f, indent=2, ensure_ascii=False)
        print(f"{map_name} 有失败样本 {len(failures)} 条，已写入: {err_log}")


def generate_all_cities():
    mp.set_start_method("spawn", force=True)

    print("=" * 80)
    print("多城市追逃数据集生成开始")
    print(f"CITY_LIST = {CITY_LIST}")
    print(f"NUM_WORKERS = {NUM_WORKERS}")
    print(f"SAVE_PNG = {SAVE_PNG}, PLOT_EVERY = {PLOT_EVERY}")
    print("=" * 80)

    for city in CITY_LIST:
        generate_dataset_for_city(city)

    print("=" * 80)
    print("全部城市生成完成")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_cities()
