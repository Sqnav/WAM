import json
import math
import os
import random
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

os.environ.setdefault("MPLCONFIGDIR", "/data1/ysq/Worldmodel/.matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ===================== config =====================
MAP_DIR = os.environ.get("MAP_DIR", "/data1/ysq/OurVLN/Plandataset/map")
UDF_CACHE_DIR = os.environ.get("UDF_CACHE_DIR", "/data1/ysq/OurVLN/Plandataset/udf_cache")
OUTPUT_DIR = os.environ.get("RECOVERY_OUTPUT_DIR", "/data1/ysq/Worldmodel/Plandataset")

DEFAULT_CITY_LIST = ",".join(f"city_{i}" for i in range(1, 31))
CITY_LIST = [c.strip() for c in os.environ.get("RECOVERY_CITY_LIST", DEFAULT_CITY_LIST).split(",") if c.strip()]
NUM_TRAJECTORIES = int(os.environ.get("RECOVERY_NUM_TRAJECTORIES", os.environ.get("NUM_TRAJECTORIES", "500")))

RECOVERY_MIN_FRAMES = int(os.environ.get("RECOVERY_MIN_FRAMES", "30"))
RECOVERY_MAX_FRAMES = int(os.environ.get("RECOVERY_MAX_FRAMES", "100"))
RECOVERY_MODE_FRAMES = int(os.environ.get("RECOVERY_MODE_FRAMES", "60"))
RECOVERY_STEPS_MIN = int(os.environ.get("RECOVERY_STEPS_MIN", "6"))
RECOVERY_STEPS_MAX = int(os.environ.get("RECOVERY_STEPS_MAX", "36"))
RECOVERY_STEPS_RATIO_MIN = float(os.environ.get("RECOVERY_STEPS_RATIO_MIN", "0.15"))
RECOVERY_STEPS_RATIO_MAX = float(os.environ.get("RECOVERY_STEPS_RATIO_MAX", "0.45"))

RECOVERY_START_BEARING_DEG_MIN = float(os.environ.get("RECOVERY_START_BEARING_DEG_MIN", "14.0"))
RECOVERY_START_BEARING_DEG_MAX = float(os.environ.get("RECOVERY_START_BEARING_DEG_MAX", "26.0"))
RECOVERY_START_LATERAL_DEG_MIN = float(os.environ.get("RECOVERY_START_LATERAL_DEG_MIN", "8.0"))
RECOVERY_START_LATERAL_DEG_MAX = float(os.environ.get("RECOVERY_START_LATERAL_DEG_MAX", "22.0"))
RECOVERY_TRACKER_STEP_TOL = float(os.environ.get("RECOVERY_TRACKER_STEP_TOL", "1e-4"))
RECOVERY_VERTICAL_HALF_ANGLE_DEG = float(os.environ.get("RECOVERY_VERTICAL_HALF_ANGLE_DEG", "45.0"))

START_SEPARATION_MIN = float(os.environ.get("START_SEPARATION_MIN", "3.0"))
START_SEPARATION_MAX = float(os.environ.get("START_SEPARATION_MAX", "5.0"))
TRACKER_MAX_Z_DROP = float(os.environ.get("TRACKER_MAX_Z_DROP", "2.0"))
TRACKER_FPV_HALF_ANGLE_DEG = float(os.environ.get("TRACKER_FPV_HALF_ANGLE_DEG", "30.0"))

RECOVERY_MAX_JAMMERS = max(1, min(int(os.environ.get("RECOVERY_MAX_JAMMERS", "3")), 3))
JAMMER_ENABLED = os.environ.get("JAMMER_ENABLED", "1") not in ("0", "false", "False")
JAMMER_MIN_TRACKER_DISTANCE = float(os.environ.get("JAMMER_MIN_TRACKER_DISTANCE", "10.0"))
JAMMER_MIN_TARGET_DISTANCE = float(os.environ.get("JAMMER_MIN_TARGET_DISTANCE", "10.0"))
JAMMER_EVENT_MIN_STEPS = int(os.environ.get("JAMMER_EVENT_MIN_STEPS", "6"))
JAMMER_EVENT_MAX_STEPS = int(os.environ.get("JAMMER_EVENT_MAX_STEPS", "12"))
JAMMER_ATTEMPTS = int(os.environ.get("JAMMER_ATTEMPTS", "80"))
JAMMER_MAX_STEP = float(os.environ.get("JAMMER_MAX_STEP", "3.0"))
JAMMER_CROSSING_BUFFER_STEPS = int(os.environ.get("JAMMER_CROSSING_BUFFER_STEPS", "5"))
JAMMER_CROSS_SCREEN_DEG_MIN = float(os.environ.get("JAMMER_CROSS_SCREEN_DEG_MIN", "6.0"))
JAMMER_CROSS_SCREEN_DEG_MAX = float(os.environ.get("JAMMER_CROSS_SCREEN_DEG_MAX", "14.0"))
JAMMER_DEPTH_BEHIND_TARGET_MIN = float(os.environ.get("JAMMER_DEPTH_BEHIND_TARGET_MIN", "12.0"))
JAMMER_DEPTH_BEHIND_TARGET_MAX = float(os.environ.get("JAMMER_DEPTH_BEHIND_TARGET_MAX", "22.0"))
JAMMER_DEPTH_FROM_CAMERA_MIN = float(os.environ.get("JAMMER_DEPTH_FROM_CAMERA_MIN", "14.0"))
JAMMER_DEPTH_FROM_CAMERA_MAX = float(os.environ.get("JAMMER_DEPTH_FROM_CAMERA_MAX", "24.0"))
JAMMER_VERTICAL_SCREEN_OFFSET_DEG = float(os.environ.get("JAMMER_VERTICAL_SCREEN_OFFSET_DEG", "4.0"))
JAMMER_SCREEN_FOV_MARGIN_DEG = float(os.environ.get("JAMMER_SCREEN_FOV_MARGIN_DEG", "1.5"))

USE_UDF = os.environ.get("USE_UDF", "1") not in ("0", "false", "False")
UDF_RESOLUTION = float(os.environ.get("UDF_RESOLUTION", "1.0"))
MIN_Z_HEIGHT = float(os.environ.get("MIN_Z_HEIGHT", "10.0"))
MAX_Z_HEIGHT = float(os.environ.get("MAX_Z_HEIGHT", "100.0"))
BOUND_MARGIN = float(os.environ.get("BOUND_MARGIN", "2.0"))
SAFETY_RADIUS = float(os.environ.get("SAFETY_RADIUS", "10.0"))

TIME_STEP = float(os.environ.get("TIME_STEP", "1.0"))
STEP_LENGTH = float(os.environ.get("STEP_LENGTH", "1.0"))
TARGET_SPEED = STEP_LENGTH / max(TIME_STEP, 1e-6)
TRACKER_SPEED = STEP_LENGTH / max(TIME_STEP, 1e-6)
JAMMER_SPEED = float(os.environ.get("JAMMER_SPEED", "2.0"))

MAX_TRAJ_BUILD_RETRY = int(os.environ.get("MAX_TRAJ_BUILD_RETRY", "120"))
TARGET_STEP_SEARCH_TRIES = int(os.environ.get("TARGET_STEP_SEARCH_TRIES", "80"))
RECOVERY_TRACKER_ATTEMPTS = int(os.environ.get("RECOVERY_TRACKER_ATTEMPTS", "80"))

MIN_BUILDING_HEIGHT = float(os.environ.get("MIN_BUILDING_HEIGHT", "12.0"))
MIN_BUILDING_XY_SPAN = float(os.environ.get("MIN_BUILDING_XY_SPAN", "8.0"))
FAST_HARD_START_NEAR_BUILDING = os.environ.get("FAST_HARD_START_NEAR_BUILDING", "1") not in ("0", "false", "False")
HARD_START_BUILDING_ATTEMPTS = int(os.environ.get("HARD_START_BUILDING_ATTEMPTS", "300"))
HARD_START_EXTRA_RADIUS_MIN = float(os.environ.get("HARD_START_EXTRA_RADIUS_MIN", "2.0"))
HARD_START_EXTRA_RADIUS_MAX = float(os.environ.get("HARD_START_EXTRA_RADIUS_MAX", "10.0"))
HARD_START_Z_ABOVE_BUILDING_MIN = float(os.environ.get("HARD_START_Z_ABOVE_BUILDING_MIN", "6.0"))
HARD_START_Z_ABOVE_BUILDING_MAX = float(os.environ.get("HARD_START_Z_ABOVE_BUILDING_MAX", "24.0"))
HARD_CLIMB_PER_STEP_MIN = float(os.environ.get("HARD_CLIMB_PER_STEP_MIN", "0.03"))
HARD_CLIMB_PER_STEP_MAX = float(os.environ.get("HARD_CLIMB_PER_STEP_MAX", "0.16"))
HARD_FALLBACK_WHEN_NO_BUILDING = os.environ.get("HARD_FALLBACK_WHEN_NO_BUILDING", "1") not in ("0", "false", "False")
HARD_FALLBACK_SEGMENT_MIN = int(os.environ.get("HARD_FALLBACK_SEGMENT_MIN", "5"))
HARD_FALLBACK_SEGMENT_MAX = int(os.environ.get("HARD_FALLBACK_SEGMENT_MAX", "10"))
HARD_FALLBACK_YAW_DEG_MIN = float(os.environ.get("HARD_FALLBACK_YAW_DEG_MIN", "25.0"))
HARD_FALLBACK_YAW_DEG_MAX = float(os.environ.get("HARD_FALLBACK_YAW_DEG_MAX", "70.0"))
HARD_FALLBACK_PITCH_DEG_MAX = float(os.environ.get("HARD_FALLBACK_PITCH_DEG_MAX", "6.0"))
HARD_FALLBACK_WIGGLE_YAW_DEG = float(os.environ.get("HARD_FALLBACK_WIGGLE_YAW_DEG", "10.0"))
HARD_FALLBACK_WIGGLE_PITCH_DEG = float(os.environ.get("HARD_FALLBACK_WIGGLE_PITCH_DEG", "3.0"))
EXTRACT_BUILDINGS = os.environ.get("EXTRACT_BUILDINGS", "0") not in ("0", "false", "False")
MAX_TURN_DEG = float(os.environ.get("MAX_TURN_DEG", "15.0"))
MAX_PITCH_DEG = float(os.environ.get("MAX_PITCH_DEG", "15.0"))
SEARCH_TRIES_PER_STEP = int(os.environ.get("SEARCH_TRIES_PER_STEP", "100"))

SAVE_PNG = os.environ.get("SAVE_PNG", "1") not in ("0", "false", "False")
PLOT_EVERY = int(os.environ.get("PLOT_EVERY", "1"))
RESUME_IF_EXISTS = os.environ.get("RESUME_IF_EXISTS", "1") not in ("0", "false", "False")
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "20260420"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "3"))

MODE_COUNTS = {
    "easy": NUM_TRAJECTORIES // 3,
    "medium": NUM_TRAJECTORIES // 3,
    "hard": NUM_TRAJECTORIES - 2 * (NUM_TRAJECTORIES // 3),
}
# ==================================================

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

    def get_distances(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        idx = ((points - self.min_xyz) / self.resolution).astype(np.int64)
        valid = (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.udf.shape[0])
            & (idx[:, 1] >= 0) & (idx[:, 1] < self.udf.shape[1])
            & (idx[:, 2] >= 0) & (idx[:, 2] < self.udf.shape[2])
        )
        dists = np.zeros(points.shape[0], dtype=float)
        valid_idx = idx[valid]
        if valid_idx.size > 0:
            dists[valid] = self.udf[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
        return dists

    def is_point_free(self, p: np.ndarray, safety_radius: float) -> bool:
        return bool(self.get_distances(np.asarray([p], dtype=float))[0] > safety_radius)


@dataclass
class BuildingInfo:
    center_xy: np.ndarray
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    orbit_radius: float
    height: float


@dataclass
class RecoveryExtra:
    yaws: List[float]
    bearing_yaw_deg: List[float]
    lateral_offset_deg: List[float]
    recovery_scale: List[float]
    recovery_steps: int
    start_bearing_deg: float
    start_lateral_deg: float


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return np.asarray(v, dtype=float) / n


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ],
        dtype=float,
    )


def rotate_direction(base_dir: np.ndarray, yaw_rad: float, pitch_rad: float) -> np.ndarray:
    base_dir = normalize(base_dir)
    if float(np.linalg.norm(base_dir)) < 1e-6:
        base_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    yaw_mat = rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=float), yaw_rad)
    d = normalize(yaw_mat @ base_dir)
    side = np.cross(d, np.array([0.0, 0.0, 1.0], dtype=float))
    if float(np.linalg.norm(side)) < 1e-6:
        side = np.array([1.0, 0.0, 0.0], dtype=float)
    pitch_mat = rotation_matrix_from_axis_angle(normalize(side), pitch_rad)
    return normalize(pitch_mat @ d)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def load_map(obj_path: str, use_udf: bool = True):
    mesh = trimesh.load(obj_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    swap_yz_transform = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    mesh.apply_transform(swap_yz_transform)
    prox = trimesh.proximity.ProximityQuery(mesh)
    bounds = np.asarray(mesh.bounds, dtype=float)

    udf_grid = None
    if use_udf and USE_UDF:
        map_name = os.path.splitext(os.path.basename(obj_path))[0]
        udf_path = os.path.join(UDF_CACHE_DIR, f"{map_name}_udf_{UDF_RESOLUTION:.1f}m.npy")
        if not os.path.exists(udf_path):
            raise FileNotFoundError(f"UDF cache not found: {udf_path}")
        udf_grid = UDFGrid.from_file(udf_path)
        print(f"[UDF] loaded: {udf_path}")

    return mesh, prox, bounds, udf_grid


def tracker_bounds(bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    min_xyz = np.array(bounds[0], dtype=float) + BOUND_MARGIN
    max_xyz = np.array(bounds[1], dtype=float) - BOUND_MARGIN
    min_xyz[2] = max(min_xyz[2], MIN_Z_HEIGHT)
    max_xyz[2] = min(max_xyz[2], MAX_Z_HEIGHT)
    return min_xyz, max_xyz


def point_in_bounds(p: np.ndarray, bounds: np.ndarray) -> bool:
    min_xyz, max_xyz = tracker_bounds(bounds)
    return bool(np.all(p >= min_xyz) and np.all(p <= max_xyz))


def is_point_free(point: np.ndarray, prox, safety_radius: float, udf_grid: Optional[UDFGrid]) -> bool:
    point = np.asarray(point, dtype=float)
    if point[2] < MIN_Z_HEIGHT or point[2] > MAX_Z_HEIGHT:
        return False
    if udf_grid is not None:
        return udf_grid.is_point_free(point, safety_radius)
    _, distance, _ = prox.on_surface([point])
    return bool(float(distance[0]) > safety_radius)


def is_segment_free(p1: np.ndarray, p2: np.ndarray, prox, safety_radius: float, udf_grid: Optional[UDFGrid]) -> bool:
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    length = float(np.linalg.norm(p2 - p1))
    if length < 1e-6:
        return is_point_free(p1, prox, safety_radius, udf_grid)

    num_samples = max(3, int(length / max(1.0, safety_radius * 0.5)))
    ts = np.linspace(0.0, 1.0, num_samples + 1, dtype=float)
    points = p1[None, :] * (1.0 - ts[:, None]) + p2[None, :] * ts[:, None]
    if np.any(points[:, 2] < MIN_Z_HEIGHT) or np.any(points[:, 2] > MAX_Z_HEIGHT):
        return False
    if udf_grid is not None:
        return bool(np.all(udf_grid.get_distances(points) > safety_radius))
    return all(is_point_free(p, prox, safety_radius, udf_grid) for p in points)


def sample_free_point(bounds: np.ndarray, prox, udf_grid: Optional[UDFGrid], max_tries: int = 5000) -> np.ndarray:
    min_xyz, max_xyz = tracker_bounds(bounds)
    for _ in range(max_tries):
        p = np.array(
            [
                random.uniform(min_xyz[0], max_xyz[0]),
                random.uniform(min_xyz[1], max_xyz[1]),
                random.uniform(min_xyz[2], max_xyz[2]),
            ],
            dtype=float,
        )
        if is_point_free(p, prox, SAFETY_RADIUS, udf_grid):
            return p
    raise RuntimeError("failed to sample a free point")


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
        buildings.append(
            BuildingInfo(
                center_xy=np.asarray(center_xy, dtype=float),
                min_xyz=np.asarray(bmin, dtype=float),
                max_xyz=np.asarray(bmax, dtype=float),
                orbit_radius=float(orbit_radius),
                height=float(span[2]),
            )
        )

    filtered: List[BuildingInfo] = []
    map_span = mesh.bounds[1] - mesh.bounds[0]
    for building in buildings:
        if (
            (building.max_xyz[0] - building.min_xyz[0]) > 0.7 * map_span[0]
            and (building.max_xyz[1] - building.min_xyz[1]) > 0.7 * map_span[1]
        ):
            continue
        filtered.append(building)
    return filtered


def choose_nearest_building(start: np.ndarray, buildings: List[BuildingInfo]) -> Optional[BuildingInfo]:
    if not buildings:
        return None
    xy = np.asarray(start, dtype=float)[:2]
    distances = [float(np.linalg.norm(building.center_xy - xy)) for building in buildings]
    return buildings[int(np.argmin(distances))]


def rotate_horizontal_dir(h: np.ndarray, yaw_rad: float) -> np.ndarray:
    h = normalize(np.array([h[0], h[1], 0.0], dtype=float))
    if float(np.linalg.norm(h)) < 1e-6:
        h = np.array([1.0, 0.0, 0.0], dtype=float)
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return normalize(np.array([c * h[0] - s * h[1], s * h[0] + c * h[1], 0.0], dtype=float))


def rotate_pitch(d: np.ndarray, pitch_rad: float) -> np.ndarray:
    d = normalize(d)
    horizontal = normalize(np.array([d[0], d[1], 0.0], dtype=float))
    if float(np.linalg.norm(horizontal)) < 1e-6:
        horizontal = np.array([1.0, 0.0, 0.0], dtype=float)
    z = math.sin(pitch_rad)
    xy = max(0.0, math.cos(pitch_rad))
    return normalize(np.array([horizontal[0] * xy, horizontal[1] * xy, z], dtype=float))


def wrap_angle_rad(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def yaw_to_forward(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)


def camera_basis(forward: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = normalize(forward)
    if float(np.linalg.norm(forward)) < 1e-6:
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    right = normalize(right)
    up = normalize(np.cross(right, forward))
    if float(np.dot(up, world_up)) < 0.0:
        up = -up
        right = -right
    return forward, right, up


def is_point_in_camera_fov(
    origin: np.ndarray,
    forward: np.ndarray,
    point: np.ndarray,
    horiz_half_angle_deg: float,
    vert_half_angle_deg: float,
) -> bool:
    fwd, right, up = camera_basis(forward)
    rel = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
    depth = float(np.dot(rel, fwd))
    if depth <= 1e-6:
        return False
    horiz = math.degrees(math.atan2(float(np.dot(rel, right)), depth))
    vert = math.degrees(math.atan2(float(np.dot(rel, up)), depth))
    return abs(horiz) <= horiz_half_angle_deg and abs(vert) <= vert_half_angle_deg


def camera_screen_angles(
    origin: np.ndarray,
    forward: np.ndarray,
    point: np.ndarray,
) -> Tuple[float, float, float]:
    fwd, right, up = camera_basis(forward)
    rel = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
    depth = float(np.dot(rel, fwd))
    if depth <= 1e-6:
        return depth, float("nan"), float("nan")
    horiz = math.degrees(math.atan2(float(np.dot(rel, right)), depth))
    vert = math.degrees(math.atan2(float(np.dot(rel, up)), depth))
    return depth, horiz, vert


def face_yaw(uav_pos: np.ndarray, target_pos: np.ndarray, fallback_yaw: float = 0.0) -> float:
    rel = np.asarray(target_pos, dtype=float) - np.asarray(uav_pos, dtype=float)
    if float(np.linalg.norm(rel[:2])) < 1e-6:
        return fallback_yaw
    return float(math.atan2(float(rel[1]), float(rel[0])))


def horizontal_unit(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    h = np.array([float(v[0]), float(v[1]), 0.0], dtype=float)
    if float(np.linalg.norm(h)) >= 1e-6:
        return normalize(h)
    if fallback is not None:
        hf = np.array([float(fallback[0]), float(fallback[1]), 0.0], dtype=float)
        if float(np.linalg.norm(hf)) >= 1e-6:
            return normalize(hf)
    return np.array([1.0, 0.0, 0.0], dtype=float)


def estimate_forward_dirs(traj: List[np.ndarray]) -> List[np.ndarray]:
    dirs = []
    for i in range(len(traj)):
        if len(traj) == 1:
            d = np.array([1.0, 0.0, 0.0], dtype=float)
        elif i < len(traj) - 1:
            d = traj[i + 1] - traj[i]
        else:
            d = traj[i] - traj[i - 1]
        if float(np.linalg.norm(d)) < 1e-6:
            d = np.array([1.0, 0.0, 0.0], dtype=float)
        dirs.append(normalize(d))
    return dirs


def recovery_scale(frame_idx: int, recovery_steps: int) -> float:
    if recovery_steps <= 0 or frame_idx >= recovery_steps:
        return 0.0
    x = float(frame_idx) / float(recovery_steps)
    return float(0.5 * (1.0 + math.cos(math.pi * x)))


def sample_num_frames() -> int:
    mode = min(max(RECOVERY_MODE_FRAMES, RECOVERY_MIN_FRAMES), RECOVERY_MAX_FRAMES)
    frames = int(round(random.triangular(RECOVERY_MIN_FRAMES, RECOVERY_MAX_FRAMES, mode)))
    return max(2, frames)


def sample_recovery_steps(num_frames: int) -> int:
    max_available = max(1, int(num_frames) - 1)
    ratio_min = max(0.0, min(1.0, RECOVERY_STEPS_RATIO_MIN))
    ratio_max = max(ratio_min, min(1.0, RECOVERY_STEPS_RATIO_MAX))
    ratio = random.uniform(ratio_min, ratio_max)
    steps = int(round(float(num_frames) * ratio))
    low = min(max_available, max(1, RECOVERY_STEPS_MIN))
    high = min(max_available, max(low, RECOVERY_STEPS_MAX))
    return max(low, min(high, steps))


def sample_mode_tasks() -> List[str]:
    mode_list: List[str] = []
    for mode, count in MODE_COUNTS.items():
        mode_list.extend([mode] * count)
    random.shuffle(mode_list)
    return mode_list


def sample_jammer_count_tasks() -> List[int]:
    if not JAMMER_ENABLED:
        return [0] * NUM_TRAJECTORIES
    counts: List[int] = []
    base_count = NUM_TRAJECTORIES // RECOVERY_MAX_JAMMERS
    remainder = NUM_TRAJECTORIES % RECOVERY_MAX_JAMMERS
    for n in range(1, RECOVERY_MAX_JAMMERS + 1):
        counts.extend([n] * base_count)
    if remainder > 0:
        extra = list(range(1, RECOVERY_MAX_JAMMERS + 1))
        random.shuffle(extra)
        counts.extend(extra[:remainder])
    random.shuffle(counts)
    return counts


class TargetTrajectoryGenerator:
    def __init__(self, bounds: np.ndarray, prox, udf_grid: Optional[UDFGrid], buildings: List[BuildingInfo]):
        self.bounds = bounds
        self.prox = prox
        self.udf_grid = udf_grid
        self.buildings = buildings
        self.min_xyz, self.max_xyz = tracker_bounds(bounds)

    def in_bounds(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self.min_xyz) and np.all(p <= self.max_xyz))

    def clip_to_bounds(self, p: np.ndarray) -> np.ndarray:
        q = np.asarray(p, dtype=float).copy()
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

    def _step_with_smoothing(
        self,
        curr: np.ndarray,
        curr_dir: np.ndarray,
        desired_dir: np.ndarray,
        max_turn_deg: float = MAX_TURN_DEG,
        step_len: float = STEP_LENGTH,
    ) -> Tuple[np.ndarray, np.ndarray]:
        curr_dir = normalize(curr_dir)
        desired_dir = normalize(desired_dir)
        if float(np.linalg.norm(curr_dir)) < 1e-6:
            curr_dir = desired_dir.copy()

        turn = angle_deg(curr_dir, desired_dir)
        if turn > max_turn_deg:
            alpha = max_turn_deg / max(turn, 1e-6)
            mixed = normalize((1.0 - alpha) * curr_dir + alpha * desired_dir)
        else:
            mixed = desired_dir

        search_yaws = np.deg2rad(np.array([0, -6, 6, -12, 12, -18, 18, -25, 25, -35, 35], dtype=float))
        search_pitches = np.deg2rad(np.array([0, -4, 4, -8, 8, -12, 12], dtype=float))
        candidates = []
        for yaw in search_yaws:
            for pitch in search_pitches:
                cand_dir = rotate_direction(mixed, float(yaw), float(pitch))
                horiz = float(np.linalg.norm(cand_dir[:2]))
                pitch_deg = abs(math.degrees(math.atan2(float(cand_dir[2]), max(horiz, 1e-8))))
                if pitch_deg > MAX_PITCH_DEG:
                    continue
                nxt = self.clip_to_bounds(curr + step_len * cand_dir)
                candidates.append((cand_dir, nxt, angle_deg(curr_dir, cand_dir)))

        candidates.sort(key=lambda x: x[2])
        for cand_dir, nxt, _ in candidates[:SEARCH_TRIES_PER_STEP]:
            if self._accept_candidate(curr, nxt):
                return nxt, cand_dir

        for _ in range(SEARCH_TRIES_PER_STEP):
            yaw = math.radians(random.uniform(-50.0, 50.0))
            pitch = math.radians(random.uniform(-12.0, 12.0))
            cand_dir = rotate_direction(curr_dir, yaw, pitch)
            nxt = self.clip_to_bounds(curr + step_len * cand_dir)
            if self._accept_candidate(curr, nxt):
                return nxt, cand_dir

        raise RuntimeError("local smooth extension failed")

    def _build_easy(self, start: np.ndarray, num_points: int) -> List[np.ndarray]:
        pts = [start.copy()]
        curr = start.copy()
        curr_dir = normalize(
            np.array(
                [
                    random.uniform(-1.0, 1.0),
                    random.uniform(-1.0, 1.0),
                    random.uniform(-0.15, 0.15),
                ],
                dtype=float,
            )
        )
        desired_dir = curr_dir.copy()
        segment_remaining = random.randint(8, 14)

        for i in range(num_points - 1):
            if segment_remaining <= 0:
                yaw_deg = random.choice([-1.0, 1.0]) * random.uniform(15.0, 45.0)
                pitch_deg = random.uniform(-5.0, 5.0)
                desired_dir = normalize(rotate_direction(desired_dir, math.radians(yaw_deg), math.radians(pitch_deg)))
                segment_remaining = random.randint(8, 14)

            wiggle_yaw = math.radians(6.0 * math.sin(2.0 * math.pi * i / 18.0))
            wiggle_pitch = math.radians(2.5 * math.sin(2.0 * math.pi * i / 24.0))
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
        center = start[:2] + radius * np.array(
            [math.cos(base_heading + math.pi / 2.0), math.sin(base_heading + math.pi / 2.0)],
            dtype=float,
        )
        angle0 = math.atan2(float(start[1] - center[1]), float(start[0] - center[0]))
        dtheta = direction_sign * (STEP_LENGTH / radius)
        curr_dir = normalize(
            np.array(
                [
                    -direction_sign * math.sin(angle0),
                    direction_sign * math.cos(angle0),
                    climb_amp,
                ],
                dtype=float,
            )
        )

        for i in range(num_points - 1):
            theta = angle0 + (i + 1) * dtheta
            tangent = normalize(
                np.array(
                    [
                        -direction_sign * math.sin(theta),
                        direction_sign * math.cos(theta),
                        climb_amp * math.cos(2.0 * math.pi * i / 22.0),
                    ],
                    dtype=float,
                )
            )
            nxt, curr_dir = self._step_with_smoothing(curr, curr_dir, tangent, max_turn_deg=10.0)
            pts.append(nxt.copy())
            curr = nxt
        return pts

    def _sample_hard_start_near_building(self, max_tries: int = HARD_START_BUILDING_ATTEMPTS) -> Tuple[np.ndarray, BuildingInfo]:
        if not self.buildings:
            raise RuntimeError("no building available for hard mode")

        buildings = self.buildings[:]
        random.shuffle(buildings)
        tries_per_building = max(1, max_tries // max(1, len(buildings)))

        for building in buildings:
            base_radius = max(building.orbit_radius, 16.0)
            z_low = max(self.min_xyz[2], MIN_Z_HEIGHT, building.min_xyz[2] + HARD_START_Z_ABOVE_BUILDING_MIN)
            z_high = min(self.max_xyz[2], MAX_Z_HEIGHT, building.max_xyz[2] + HARD_START_Z_ABOVE_BUILDING_MAX)
            if z_low >= z_high:
                z_low = max(self.min_xyz[2], MIN_Z_HEIGHT)
                z_high = min(self.max_xyz[2], MAX_Z_HEIGHT)
            if z_low >= z_high:
                continue

            for _ in range(tries_per_building):
                theta = random.uniform(-math.pi, math.pi)
                radius = base_radius + random.uniform(HARD_START_EXTRA_RADIUS_MIN, HARD_START_EXTRA_RADIUS_MAX)
                p = np.array(
                    [
                        building.center_xy[0] + radius * math.cos(theta),
                        building.center_xy[1] + radius * math.sin(theta),
                        random.uniform(z_low, z_high),
                    ],
                    dtype=float,
                )
                if not self.in_bounds(p):
                    continue
                if not is_point_free(p, self.prox, SAFETY_RADIUS, self.udf_grid):
                    continue
                return p, building

        raise RuntimeError("failed to sample hard start near building")

    def _build_hard(self, start: np.ndarray, num_points: int, building: Optional[BuildingInfo] = None) -> List[np.ndarray]:
        if building is None:
            building = choose_nearest_building(start, self.buildings)
        if building is None:
            raise RuntimeError("no nearest building for hard mode")

        pts = [start.copy()]
        curr = start.copy()
        center = building.center_xy.copy()
        rel0 = curr[:2] - center
        rel0_norm = float(np.linalg.norm(rel0))
        if rel0_norm < 1e-6:
            raise RuntimeError("hard start is too close to building center")

        radius = max(rel0_norm, building.orbit_radius, 16.0)
        theta0 = math.atan2(float(rel0[1]), float(rel0[0]))
        total_steps = num_points - 1
        if total_steps < 8:
            raise RuntimeError("hard trajectory is too short")

        direction_sign = random.choice([-1.0, 1.0])
        dtheta = direction_sign * (STEP_LENGTH / max(radius, 1e-6))
        curr_dir = normalize(
            np.array(
                [
                    -direction_sign * math.sin(theta0),
                    direction_sign * math.cos(theta0),
                    0.0,
                ],
                dtype=float,
            )
        )
        if float(np.linalg.norm(curr_dir)) < 1e-6:
            curr_dir = np.array([1.0, 0.0, 0.0], dtype=float)

        max_pitch_slope = math.tan(math.radians(max(1.0, MAX_PITCH_DEG - 3.0)))
        climb_high = min(HARD_CLIMB_PER_STEP_MAX, max_pitch_slope * 0.8)
        climb_low = min(HARD_CLIMB_PER_STEP_MIN, climb_high)
        climb_per_step = random.uniform(climb_low, climb_high)
        if curr[2] + climb_per_step * total_steps > self.max_xyz[2] - 2.0:
            climb_per_step = max(0.0, (self.max_xyz[2] - 2.0 - curr[2]) / max(total_steps, 1))

        for j in range(total_steps):
            theta = theta0 + (j + 1) * dtheta
            desired_xy = np.array([center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)], dtype=float)
            tangent = np.array(
                [
                    -direction_sign * math.sin(theta),
                    direction_sign * math.cos(theta),
                    climb_per_step / max(STEP_LENGTH, 1e-6),
                ],
                dtype=float,
            )
            radial = np.array([desired_xy[0] - curr[0], desired_xy[1] - curr[1], 0.0], dtype=float)
            if float(np.linalg.norm(radial)) < 1e-6:
                radial = tangent.copy()
            desired_dir = normalize(0.82 * normalize(tangent) + 0.18 * normalize(radial))
            nxt, curr_dir = self._step_with_smoothing(curr, curr_dir, desired_dir, max_turn_deg=10.0)
            pts.append(nxt.copy())
            curr = nxt
        return pts

    def _build_hard_without_building(self, start: np.ndarray, num_points: int) -> List[np.ndarray]:
        pts = [start.copy()]
        curr = start.copy()
        curr_dir = normalize(
            np.array(
                [
                    random.uniform(-1.0, 1.0),
                    random.uniform(-1.0, 1.0),
                    random.uniform(-0.05, 0.10),
                ],
                dtype=float,
            )
        )
        if float(np.linalg.norm(curr_dir)) < 1e-6:
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

                if curr[2] > self.max_xyz[2] - 12.0:
                    vertical_bias = random.uniform(-0.10, 0.0)
                elif curr[2] < self.min_xyz[2] + 12.0:
                    vertical_bias = random.uniform(0.0, 0.10)
                else:
                    vertical_bias = random.uniform(-0.04, 0.10)

            wiggle_yaw = math.radians(HARD_FALLBACK_WIGGLE_YAW_DEG * math.sin(2.0 * math.pi * i / 11.0))
            wiggle_pitch = math.radians(HARD_FALLBACK_WIGGLE_PITCH_DEG * math.sin(2.0 * math.pi * i / 17.0))
            local_desired = rotate_direction(desired_dir, wiggle_yaw, wiggle_pitch)
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
        last_error = ""
        error_counts: Dict[str, int] = {}
        for _ in range(MAX_TRAJ_BUILD_RETRY):
            try:
                if mode == "hard" and FAST_HARD_START_NEAR_BUILDING:
                    if self.buildings:
                        start, building = self._sample_hard_start_near_building()
                        traj = self._build_hard(start, num_points, building=building)
                    elif HARD_FALLBACK_WHEN_NO_BUILDING:
                        start = sample_free_point(self.bounds, self.prox, self.udf_grid)
                        traj = self._build_hard_without_building(start, num_points)
                    else:
                        raise RuntimeError("no building available for hard mode")
                else:
                    start = sample_free_point(self.bounds, self.prox, self.udf_grid)
                    if mode == "easy":
                        traj = self._build_easy(start, num_points)
                    elif mode == "medium":
                        traj = self._build_medium(start, num_points)
                    elif mode == "hard":
                        traj = self._build_hard(start, num_points)
                    else:
                        raise ValueError(f"unknown mode: {mode}")

                if len(traj) == num_points:
                    return traj
            except RuntimeError as e:
                msg = str(e)
                last_error = msg
                error_counts[msg] = error_counts.get(msg, 0) + 1
                continue
        raise RuntimeError(f"{mode} target trajectory failed; last_error={last_error}; error_counts={error_counts}")


def build_target_trajectory(
    mode: str,
    num_frames: int,
    prox,
    bounds: np.ndarray,
    udf_grid: Optional[UDFGrid],
    buildings: List[BuildingInfo],
) -> List[np.ndarray]:
    generator = TargetTrajectoryGenerator(bounds, prox, udf_grid, buildings)
    return generator.build_target_trajectory(mode, num_frames)


def validate_recovery_tracker(
    target_traj: List[np.ndarray],
    tracker_traj: List[np.ndarray],
    yaws: List[float],
    prox,
    bounds: np.ndarray,
    udf_grid: Optional[UDFGrid],
) -> bool:
    if len(target_traj) != len(tracker_traj) or len(target_traj) != len(yaws):
        return False
    start_dist = float(np.linalg.norm(target_traj[0] - tracker_traj[0]))
    if not (START_SEPARATION_MIN - 1e-6 <= start_dist <= START_SEPARATION_MAX + 1e-6):
        return False

    for i, (target_pos, tracker_pos, yaw) in enumerate(zip(target_traj, tracker_traj, yaws)):
        if target_pos[2] + 1e-6 < tracker_pos[2]:
            return False
        if not point_in_bounds(target_pos, bounds) or not point_in_bounds(tracker_pos, bounds):
            return False
        if not is_point_free(tracker_pos, prox, SAFETY_RADIUS, udf_grid):
            return False
        if i > 0:
            step = float(np.linalg.norm(tracker_pos - tracker_traj[i - 1]))
            if abs(step - STEP_LENGTH) > RECOVERY_TRACKER_STEP_TOL:
                return False
            if not is_segment_free(tracker_traj[i - 1], tracker_pos, prox, SAFETY_RADIUS, udf_grid):
                return False
        if not is_point_in_camera_fov(
            tracker_pos,
            yaw_to_forward(yaw),
            target_pos,
            TRACKER_FPV_HALF_ANGLE_DEG,
            RECOVERY_VERTICAL_HALF_ANGLE_DEG,
        ):
            return False
    return True


def build_recovery_tracker(
    target_traj: List[np.ndarray],
    prox,
    bounds: np.ndarray,
    udf_grid: Optional[UDFGrid],
) -> Tuple[List[np.ndarray], RecoveryExtra]:
    target_dirs = estimate_forward_dirs(target_traj)
    min_xyz, _ = tracker_bounds(bounds)
    min_target_z = min(float(p[2]) for p in target_traj)
    max_z_drop = max(0.0, min(TRACKER_MAX_Z_DROP, min_target_z - min_xyz[2]))

    for _ in range(RECOVERY_TRACKER_ATTEMPTS):
        start_dist = random.uniform(START_SEPARATION_MIN, START_SEPARATION_MAX)
        final_dist = random.uniform(START_SEPARATION_MIN, START_SEPARATION_MAX)
        start_z_drop = random.uniform(0.0, min(max_z_drop, start_dist * 0.5))
        final_z_drop = random.uniform(0.0, min(max_z_drop, final_dist * 0.5))
        start_bearing = random.choice([-1.0, 1.0]) * random.uniform(
            RECOVERY_START_BEARING_DEG_MIN,
            RECOVERY_START_BEARING_DEG_MAX,
        )
        start_lateral = random.choice([-1.0, 1.0]) * random.uniform(
            RECOVERY_START_LATERAL_DEG_MIN,
            RECOVERY_START_LATERAL_DEG_MAX,
        )
        recovery_steps = sample_recovery_steps(len(target_traj))

        tracker: List[np.ndarray] = []
        yaws: List[float] = []
        bearings: List[float] = []
        laterals: List[float] = []
        scales: List[float] = []
        prev_h = horizontal_unit(target_dirs[0], target_traj[1] - target_traj[0])
        prev_yaw = 0.0

        for i, target_pos in enumerate(target_traj):
            scale = recovery_scale(i, recovery_steps)
            h = horizontal_unit(target_dirs[i], prev_h)
            lateral_deg = start_lateral * scale
            h = rotate_horizontal_dir(h, math.radians(lateral_deg))
            prev_h = h

            dist = final_dist + (start_dist - final_dist) * scale
            z_drop = final_z_drop + (start_z_drop - final_z_drop) * scale
            xy_dist = math.sqrt(max(dist * dist - z_drop * z_drop, 0.0))
            desired_tracker_pos = np.array(
                [
                    target_pos[0] - xy_dist * h[0],
                    target_pos[1] - xy_dist * h[1],
                    target_pos[2] - z_drop,
                ],
                dtype=float,
            )

            if tracker:
                delta = desired_tracker_pos - tracker[-1]
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm < 1e-8:
                    break
                tracker_pos = tracker[-1] + delta / delta_norm * STEP_LENGTH
            else:
                tracker_pos = desired_tracker_pos
            bearing = math.radians(start_bearing) * scale
            face = face_yaw(tracker_pos, target_pos, fallback_yaw=prev_yaw)
            yaw = wrap_angle_rad(face - bearing)
            prev_yaw = yaw

            tracker.append(tracker_pos)
            yaws.append(yaw)
            bearings.append(float(math.degrees(bearing)))
            laterals.append(float(lateral_deg))
            scales.append(scale)

        if validate_recovery_tracker(target_traj, tracker, yaws, prox, bounds, udf_grid):
            return tracker, RecoveryExtra(
                yaws=yaws,
                bearing_yaw_deg=bearings,
                lateral_offset_deg=laterals,
                recovery_scale=scales,
                recovery_steps=recovery_steps,
                start_bearing_deg=start_bearing,
                start_lateral_deg=start_lateral,
            )

    raise RuntimeError("failed to build recovery tracker")


def compute_event_window(num_frames: int, jammer_id: int, num_jammers: int) -> Tuple[int, int]:
    event_len = max(JAMMER_EVENT_MIN_STEPS, min(JAMMER_EVENT_MAX_STEPS, int(round(num_frames * 0.22))))
    frac = float(jammer_id) / float(num_jammers + 1)
    center = int(round(frac * max(0, num_frames - 1)))
    half = event_len // 2
    start = max(0, center - half)
    end = min(num_frames - 1, start + event_len - 1)
    start = max(0, end - event_len + 1)
    return start, end


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def build_one_jammer(
    jammer_id: int,
    num_jammers: int,
    target_traj: List[np.ndarray],
    tracker_traj: List[np.ndarray],
    yaws: List[float],
    prox,
    bounds: np.ndarray,
    udf_grid: Optional[UDFGrid],
) -> Dict[str, Any]:
    num_frames = len(target_traj)
    active_start, active_end = compute_event_window(num_frames, jammer_id, num_jammers)
    active_screen_margin = float("inf")
    active_vertical_margin = float("inf")
    for i in range(active_start, active_end + 1):
        target_depth, target_horiz_deg, target_vert_deg = camera_screen_angles(
            tracker_traj[i],
            yaw_to_forward(yaws[i]),
            target_traj[i],
        )
        if target_depth <= 1e-6 or not math.isfinite(target_horiz_deg) or not math.isfinite(target_vert_deg):
            raise RuntimeError(f"failed to build jammer-{jammer_id}: target is behind camera during active window")
        active_screen_margin = min(
            active_screen_margin,
            TRACKER_FPV_HALF_ANGLE_DEG - abs(target_horiz_deg) - JAMMER_SCREEN_FOV_MARGIN_DEG,
        )
        active_vertical_margin = min(
            active_vertical_margin,
            RECOVERY_VERTICAL_HALF_ANGLE_DEG - abs(target_vert_deg) - JAMMER_SCREEN_FOV_MARGIN_DEG,
        )
    active_screen_margin = max(1.0, active_screen_margin)
    active_vertical_margin = max(1.0, active_vertical_margin)

    for _ in range(JAMMER_ATTEMPTS):
        max_cross_deg = max(1.0, min(JAMMER_CROSS_SCREEN_DEG_MAX, active_screen_margin))
        min_cross_deg = min(max_cross_deg, max(1.0, JAMMER_CROSS_SCREEN_DEG_MIN))
        if max_cross_deg <= min_cross_deg + 1e-6:
            cross_screen_deg = max_cross_deg
        else:
            cross_screen_deg = random.uniform(min_cross_deg, max_cross_deg)
        depth_low = max(JAMMER_DEPTH_FROM_CAMERA_MIN, JAMMER_MIN_TRACKER_DISTANCE + 1.0)
        depth_high = max(JAMMER_DEPTH_FROM_CAMERA_MAX, depth_low)
        depth_from_camera = random.uniform(depth_low, depth_high) + 1.5 * float(jammer_id - 1)
        vertical_limit = max(0.0, min(JAMMER_VERTICAL_SCREEN_OFFSET_DEG, active_vertical_margin))
        vertical_screen_deg = random.uniform(-vertical_limit, vertical_limit)
        direction = -1.0 if random.random() < 0.5 else 1.0
        phase = random.uniform(-math.pi, math.pi)
        buffer_steps = min(JAMMER_CROSSING_BUFFER_STEPS, max(1, active_start), max(1, num_frames - 1 - active_end))
        cross_start = max(0, active_start - buffer_steps)
        cross_end = min(num_frames - 1, active_end + buffer_steps)

        trajectory: List[np.ndarray] = []
        visible_mask: List[bool] = []
        ok = True
        prev: Optional[np.ndarray] = None

        for i, (target_pos, tracker_pos, yaw) in enumerate(zip(target_traj, tracker_traj, yaws)):
            fwd, right, up = camera_basis(yaw_to_forward(yaw))
            target_depth, target_horiz_deg, target_vert_deg = camera_screen_angles(tracker_pos, fwd, target_pos)
            if target_depth <= 1e-6 or not math.isfinite(target_horiz_deg) or not math.isfinite(target_vert_deg):
                ok = False
                break
            denom = max(1, cross_end - cross_start)
            progress = smoothstep(float(i - cross_start) / float(denom))
            screen_offset_deg = direction * (-cross_screen_deg + 2.0 * cross_screen_deg * progress)
            horiz_limit = max(1.0, TRACKER_FPV_HALF_ANGLE_DEG - JAMMER_SCREEN_FOV_MARGIN_DEG)
            vert_limit = max(1.0, RECOVERY_VERTICAL_HALF_ANGLE_DEG - JAMMER_SCREEN_FOV_MARGIN_DEG)
            screen_horiz_deg = float(np.clip(target_horiz_deg + screen_offset_deg, -horiz_limit, horiz_limit))
            screen_vert_deg = float(np.clip(
                target_vert_deg + vertical_screen_deg + 1.0 * math.sin(phase + 2.0 * math.pi * progress),
                -vert_limit,
                vert_limit,
            ))
            depth = max(
                JAMMER_MIN_TRACKER_DISTANCE + 1.0,
                depth_from_camera + 0.75 * math.sin(phase + 2.0 * math.pi * progress),
            )
            desired = (
                tracker_pos
                + depth * fwd
                + depth * math.tan(math.radians(screen_horiz_deg)) * right
                + depth * math.tan(math.radians(screen_vert_deg)) * up
            )

            if prev is None:
                candidate = desired
            else:
                delta = desired - prev
                step = float(np.linalg.norm(delta))
                if step > JAMMER_MAX_STEP:
                    candidate = prev + delta / max(step, 1e-6) * JAMMER_MAX_STEP
                else:
                    candidate = desired

            actual_visible = is_point_in_camera_fov(
                tracker_pos,
                fwd,
                candidate,
                TRACKER_FPV_HALF_ANGLE_DEG,
                RECOVERY_VERTICAL_HALF_ANGLE_DEG,
            )
            if not point_in_bounds(candidate, bounds):
                ok = False
                break
            if not is_point_free(candidate, prox, SAFETY_RADIUS, udf_grid):
                ok = False
                break
            if float(np.linalg.norm(candidate - tracker_pos)) < JAMMER_MIN_TRACKER_DISTANCE:
                ok = False
                break
            if float(np.linalg.norm(candidate - target_pos)) < JAMMER_MIN_TARGET_DISTANCE:
                ok = False
                break
            if active_start <= i <= active_end and not actual_visible:
                ok = False
                break
            if prev is not None:
                if float(np.linalg.norm(candidate - prev)) > JAMMER_MAX_STEP + 1e-6:
                    ok = False
                    break
                if not is_segment_free(prev, candidate, prox, SAFETY_RADIUS, udf_grid):
                    ok = False
                    break

            trajectory.append(candidate)
            visible_mask.append(bool(actual_visible and active_start <= i <= active_end))
            prev = candidate

        if ok and len(trajectory) == num_frames:
            return {
                "id": int(jammer_id),
                "type": "screen_space_crossing",
                "trajectory": trajectory,
                "active_start_frame": int(active_start),
                "active_end_frame": int(active_end),
                "crossing_start_frame": int(cross_start),
                "crossing_end_frame": int(cross_end),
                "max_step": float(JAMMER_MAX_STEP),
                "cross_screen_deg": float(cross_screen_deg),
                "depth_from_camera": float(depth_from_camera),
                "depth_offset": float(depth_from_camera - target_depth),
                "vertical_screen_deg": float(vertical_screen_deg),
                "visible_expected_mask": visible_mask,
            }

    raise RuntimeError(f"failed to build jammer-{jammer_id}")


def build_all_jammers(
    target_traj: List[np.ndarray],
    tracker_traj: List[np.ndarray],
    yaws: List[float],
    num_jammers: int,
    prox,
    bounds: np.ndarray,
    udf_grid: Optional[UDFGrid],
) -> List[Dict[str, Any]]:
    if num_jammers <= 0:
        return []
    num_jammers = max(1, min(int(num_jammers), RECOVERY_MAX_JAMMERS))
    return [
        build_one_jammer(jid, num_jammers, target_traj, tracker_traj, yaws, prox, bounds, udf_grid)
        for jid in range(1, num_jammers + 1)
    ]


def trajectory_length(traj: List[np.ndarray]) -> float:
    if len(traj) < 2:
        return 0.0
    return float(sum(np.linalg.norm(traj[i] - traj[i - 1]) for i in range(1, len(traj))))


def build_record(
    map_name: str,
    idx: int,
    mode: str,
    target_traj: List[np.ndarray],
    tracker_traj: List[np.ndarray],
    extra: RecoveryExtra,
    jammer_trajectories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    frame_distances = [float(np.linalg.norm(t - p)) for t, p in zip(target_traj, tracker_traj)]
    frames = []
    for frame_idx, (target_pos, uav_pos, yaw, bearing, lateral, scale, dist) in enumerate(
        zip(
            target_traj,
            tracker_traj,
            extra.yaws,
            extra.bearing_yaw_deg,
            extra.lateral_offset_deg,
            extra.recovery_scale,
            frame_distances,
        )
    ):
        frame = {
            "frame_id": int(frame_idx),
            "target_position": [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])],
            "uav_position": [float(uav_pos[0]), float(uav_pos[1]), float(uav_pos[2])],
            "uav_yaw": float(yaw),
            "uav_yaw_deg": float(math.degrees(yaw)),
            "uav_orientation_euler_planned": {"roll": 0.0, "pitch": 0.0, "yaw": float(yaw)},
            "target_bearing_yaw_deg": float(bearing),
            "tracker_lateral_offset_deg": float(lateral),
            "recovery_scale": float(scale),
            "distance": float(dist),
            "jammers": [],
        }

        for jammer in jammer_trajectories:
            jammer_pos = jammer["trajectory"][frame_idx]
            visible_mask = jammer.get("visible_expected_mask", [False] * len(jammer["trajectory"]))
            item = {
                "id": int(jammer["id"]),
                "type": str(jammer["type"]),
                "position": [float(jammer_pos[0]), float(jammer_pos[1]), float(jammer_pos[2])],
                "distance_to_uav": float(np.linalg.norm(jammer_pos - uav_pos)),
                "distance_to_target": float(np.linalg.norm(jammer_pos - target_pos)),
                "visible_expected": bool(visible_mask[frame_idx]),
            }
            frame["jammers"].append(item)

        primary = next((j for j in jammer_trajectories if int(j["id"]) == 1), None)
        if primary is not None:
            jammer_pos = primary["trajectory"][frame_idx]
            visible_mask = primary.get("visible_expected_mask", [False] * len(primary["trajectory"]))
            frame["jammer_position"] = [float(jammer_pos[0]), float(jammer_pos[1]), float(jammer_pos[2])]
            frame["jammer_distance_to_uav"] = float(np.linalg.norm(jammer_pos - uav_pos))
            frame["jammer_distance_to_target"] = float(np.linalg.norm(jammer_pos - target_pos))
            frame["jammer_visible_expected"] = bool(visible_mask[frame_idx])

        frames.append(frame)

    jammer_lengths = {str(j["id"]): trajectory_length(j["trajectory"]) for j in jammer_trajectories}
    jammers_meta = [
        {
            "id": int(j["id"]),
            "type": str(j["type"]),
            "num_points": len(j["trajectory"]),
            "trajectory_length": jammer_lengths[str(j["id"])],
            "active_start_frame": int(j.get("active_start_frame", 0)),
            "active_end_frame": int(j.get("active_end_frame", len(j["trajectory"]) - 1)),
            "crossing_start_frame": int(j.get("crossing_start_frame", j.get("active_start_frame", 0))),
            "crossing_end_frame": int(j.get("crossing_end_frame", j.get("active_end_frame", len(j["trajectory"]) - 1))),
            "max_step": float(j.get("max_step", JAMMER_MAX_STEP)),
            "cross_screen_deg": float(j.get("cross_screen_deg", 0.0)),
            "depth_from_camera": float(j.get("depth_from_camera", 0.0)),
            "depth_offset": float(j.get("depth_offset", 0.0)),
            "vertical_screen_deg": float(j.get("vertical_screen_deg", 0.0)),
        }
        for j in jammer_trajectories
    ]

    return {
        "map_name": map_name,
        "trajectory_id": int(idx),
        "mode": mode,
        "trajectory_family": "recovery_track",
        "tracker_policy": "standalone_yaw_recovery",
        "tracker_has_planned_yaw": True,
        "num_points": len(target_traj),
        "time_step": TIME_STEP,
        "target_speed_nominal": TARGET_SPEED,
        "tracker_speed_nominal": TRACKER_SPEED,
        "jammer_enabled": bool(len(jammer_trajectories) > 0),
        "num_jammers": int(len(jammer_trajectories)),
        "max_jammers": int(RECOVERY_MAX_JAMMERS),
        "jammer_ids": [int(j["id"]) for j in jammer_trajectories],
        "jammer_speed_nominal": JAMMER_SPEED if jammer_trajectories else 0.0,
        "tracker_fpv_half_angle_deg": TRACKER_FPV_HALF_ANGLE_DEG,
        "desired_target_length": float((len(target_traj) - 1) * STEP_LENGTH),
        "target_length": trajectory_length(target_traj),
        "tracker_length": trajectory_length(tracker_traj),
        "jammer_lengths": jammer_lengths,
        "start_distance": float(frame_distances[0]) if frame_distances else 0.0,
        "recovery_meta": {
            "recovery_steps": int(extra.recovery_steps),
            "start_bearing_yaw_deg": float(extra.start_bearing_deg),
            "start_lateral_offset_deg": float(extra.start_lateral_deg),
            "bearing_yaw_deg_min": float(min(extra.bearing_yaw_deg)) if extra.bearing_yaw_deg else 0.0,
            "bearing_yaw_deg_max": float(max(extra.bearing_yaw_deg)) if extra.bearing_yaw_deg else 0.0,
            "min_frames": int(RECOVERY_MIN_FRAMES),
            "max_frames": int(RECOVERY_MAX_FRAMES),
            "recovery_steps_min": int(RECOVERY_STEPS_MIN),
            "recovery_steps_max": int(RECOVERY_STEPS_MAX),
            "recovery_steps_ratio_min": float(RECOVERY_STEPS_RATIO_MIN),
            "recovery_steps_ratio_max": float(RECOVERY_STEPS_RATIO_MAX),
        },
        "jammers_metadata": jammers_meta,
        "frames": frames,
    }


def save_visualization(
    output_png: str,
    tracker_traj: List[np.ndarray],
    target_traj: List[np.ndarray],
    jammer_trajectories: List[Dict[str, Any]],
    yaws: List[float],
    mode: str,
) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    t = np.asarray(target_traj, dtype=float)
    p = np.asarray(tracker_traj, dtype=float)
    ax.plot(t[:, 0], t[:, 1], t[:, 2], "r-", linewidth=2.0, label="Target")
    ax.plot(p[:, 0], p[:, 1], p[:, 2], "b-", linewidth=2.0, label="UAV")

    stride = max(1, len(p) // 12)
    for i in range(0, len(p), stride):
        fwd = yaw_to_forward(yaws[i])
        ax.quiver(p[i, 0], p[i, 1], p[i, 2], fwd[0], fwd[1], fwd[2], length=3.0, color="navy", alpha=0.45)

    jammer_arrays = []
    colors = {1: "m", 2: "c", 3: "k"}
    for jammer in jammer_trajectories:
        arr = np.asarray(jammer["trajectory"], dtype=float)
        jammer_arrays.append(arr)
        c = colors.get(int(jammer["id"]), "g")
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=c, linewidth=1.5, label=f"J{jammer['id']}")

    ax.scatter(t[0, 0], t[0, 1], t[0, 2], c="orange", s=70, label="Target Start")
    ax.scatter(p[0, 0], p[0, 1], p[0, 2], c="green", s=70, label="UAV Start")
    ax.scatter(t[-1, 0], t[-1, 1], t[-1, 2], c="red", s=70, marker="s", label="Target End")
    ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], c="blue", s=70, marker="s", label="UAV End")

    all_pts = np.vstack([t, p] + jammer_arrays) if jammer_arrays else np.vstack([t, p])
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
    ax.set_title(f"Recovery trajectory | mode={mode}")
    ax.legend(loc="upper left")
    ax.view_init(elev=24, azim=48)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)


def init_worker_for_city(obj_path: str):
    global _PROCESS_DATA
    mesh, prox, bounds, udf_grid = load_map(obj_path, use_udf=USE_UDF)
    if EXTRACT_BUILDINGS:
        buildings = extract_buildings(mesh)
        print(f"[buildings] {os.path.basename(obj_path)}: extracted {len(buildings)} usable buildings")
    else:
        buildings = []
        print(f"[buildings] {os.path.basename(obj_path)}: skipped; hard mode uses fallback")
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

    if _PROCESS_DATA is None:
        return {"success": False, "idx": idx, "mode": mode, "error": "worker map is not initialized"}

    prox = _PROCESS_DATA["prox"]
    bounds = _PROCESS_DATA["bounds"]
    udf_grid = _PROCESS_DATA["udf_grid"]
    buildings = _PROCESS_DATA["buildings"]

    try:
        last_error = ""
        for outer_try in range(MAX_TRAJ_BUILD_RETRY):
            try:
                num_frames = sample_num_frames()
                target_traj = build_target_trajectory(mode, num_frames, prox, bounds, udf_grid, buildings)
                tracker_traj, extra = build_recovery_tracker(target_traj, prox, bounds, udf_grid)
                jammer_trajectories = build_all_jammers(
                    target_traj,
                    tracker_traj,
                    extra.yaws,
                    num_jammers,
                    prox,
                    bounds,
                    udf_grid,
                )
                break
            except RuntimeError as e:
                last_error = str(e)
                if outer_try % 10 == 0:
                    print(f"[retry] idx={idx}, mode={mode}, jammers={num_jammers}, try={outer_try}, error={e}")
                if outer_try == MAX_TRAJ_BUILD_RETRY - 1:
                    raise RuntimeError(last_error)
                continue

        record = build_record(map_name, idx, mode, target_traj, tracker_traj, extra, jammer_trajectories)

        out_json = os.path.join(map_output_dir, f"trajectory_{idx:04d}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        if SAVE_PNG and (idx % PLOT_EVERY == 0 or idx <= 5):
            out_png = os.path.join(map_output_dir, f"trajectory_{idx:04d}.png")
            save_visualization(out_png, tracker_traj, target_traj, jammer_trajectories, extra.yaws, mode)

        return {
            "success": True,
            "idx": idx,
            "mode": mode,
            "num_frames": len(target_traj),
            "num_jammers": len(jammer_trajectories),
            "start_bearing_yaw_deg": extra.start_bearing_deg,
        }
    except Exception as e:
        return {
            "success": False,
            "idx": idx,
            "mode": mode,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
        }


def maybe_existing_ok(city_output_dir: str, idx: int, expected_num_jammers: Optional[int] = None) -> bool:
    out_json = os.path.join(city_output_dir, f"trajectory_{idx:04d}.json")
    out_png = os.path.join(city_output_dir, f"trajectory_{idx:04d}.png")
    if not os.path.exists(out_json):
        return False
    if SAVE_PNG and not os.path.exists(out_png):
        return False
    if expected_num_jammers is None:
        return True
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("num_jammers", -1)) == int(expected_num_jammers)
    except Exception:
        return False


def summarize_city_output(city_output_dir: str, map_name: str) -> Dict[str, Any]:
    lengths = []
    frame_counts = []
    bearing_abs = []
    jammer_dist = {str(i): 0 for i in range(0 if not JAMMER_ENABLED else 1, RECOVERY_MAX_JAMMERS + 1)}
    found = 0

    for idx in range(1, NUM_TRAJECTORIES + 1):
        json_path = os.path.join(city_output_dir, f"trajectory_{idx:04d}.json")
        if not os.path.exists(json_path):
            continue
        found += 1
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        lengths.append(float(data.get("target_length", 0.0)))
        frame_counts.append(int(data.get("num_points", 0)))
        nj = int(data.get("num_jammers", 0))
        key = str(nj)
        jammer_dist[key] = jammer_dist.get(key, 0) + 1
        for frame in data.get("frames", []):
            if "target_bearing_yaw_deg" in frame:
                bearing_abs.append(abs(float(frame["target_bearing_yaw_deg"])))

    meta = {
        "map_name": map_name,
        "num_trajectories": int(found),
        "mode_counts": MODE_COUNTS,
        "jammer_count_distribution": jammer_dist,
        "mean_target_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_target_length": float(np.std(lengths)) if lengths else 0.0,
        "mean_num_frames": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "trajectory_family_counts": {"recovery_track": int(found)},
        "recovery_config": {
            "output_dir": OUTPUT_DIR,
            "city_list": CITY_LIST,
            "min_frames": RECOVERY_MIN_FRAMES,
            "max_frames": RECOVERY_MAX_FRAMES,
            "mode_frames": RECOVERY_MODE_FRAMES,
            "recovery_steps_min": RECOVERY_STEPS_MIN,
            "recovery_steps_max": RECOVERY_STEPS_MAX,
            "recovery_steps_ratio_min": RECOVERY_STEPS_RATIO_MIN,
            "recovery_steps_ratio_max": RECOVERY_STEPS_RATIO_MAX,
            "start_bearing_deg_min": RECOVERY_START_BEARING_DEG_MIN,
            "start_bearing_deg_max": RECOVERY_START_BEARING_DEG_MAX,
            "start_lateral_deg_min": RECOVERY_START_LATERAL_DEG_MIN,
            "start_lateral_deg_max": RECOVERY_START_LATERAL_DEG_MAX,
            "max_jammers": RECOVERY_MAX_JAMMERS,
            "jammer_max_step": JAMMER_MAX_STEP,
            "jammer_crossing_buffer_steps": JAMMER_CROSSING_BUFFER_STEPS,
            "jammer_type": "screen_space_crossing",
            "jammer_cross_screen_deg_min": JAMMER_CROSS_SCREEN_DEG_MIN,
            "jammer_cross_screen_deg_max": JAMMER_CROSS_SCREEN_DEG_MAX,
            "jammer_depth_behind_target_min": JAMMER_DEPTH_BEHIND_TARGET_MIN,
            "jammer_depth_behind_target_max": JAMMER_DEPTH_BEHIND_TARGET_MAX,
            "jammer_depth_from_camera_min": JAMMER_DEPTH_FROM_CAMERA_MIN,
            "jammer_depth_from_camera_max": JAMMER_DEPTH_FROM_CAMERA_MAX,
            "jammer_vertical_screen_offset_deg": JAMMER_VERTICAL_SCREEN_OFFSET_DEG,
            "jammer_screen_fov_margin_deg": JAMMER_SCREEN_FOV_MARGIN_DEG,
            "extract_buildings": EXTRACT_BUILDINGS,
            "mean_abs_target_bearing_yaw_deg": float(np.mean(bearing_abs)) if bearing_abs else 0.0,
            "max_abs_target_bearing_yaw_deg": float(np.max(bearing_abs)) if bearing_abs else 0.0,
        },
        "difficulty_config": {
            "easy": "segmented smooth trajectory with low yaw/pitch wiggle",
            "medium": "arc trajectory with climb oscillation",
            "hard": "building-orbit trajectory when buildings exist; high-curvature fallback otherwise",
            "fast_hard_start_near_building": FAST_HARD_START_NEAR_BUILDING,
            "hard_start_building_attempts": HARD_START_BUILDING_ATTEMPTS,
            "hard_start_extra_radius_min": HARD_START_EXTRA_RADIUS_MIN,
            "hard_start_extra_radius_max": HARD_START_EXTRA_RADIUS_MAX,
            "hard_start_z_above_building_min": HARD_START_Z_ABOVE_BUILDING_MIN,
            "hard_start_z_above_building_max": HARD_START_Z_ABOVE_BUILDING_MAX,
            "hard_climb_per_step_min": HARD_CLIMB_PER_STEP_MIN,
            "hard_climb_per_step_max": HARD_CLIMB_PER_STEP_MAX,
            "hard_fallback_when_no_building": HARD_FALLBACK_WHEN_NO_BUILDING,
            "max_turn_deg": MAX_TURN_DEG,
            "max_pitch_deg": MAX_PITCH_DEG,
        },
    }
    with open(os.path.join(city_output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def iter_with_progress(iterable, total: int, desc: str):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, ncols=100)

    def generator():
        done = 0
        print(f"{desc}: 0/{total}")
        for item in iterable:
            done += 1
            if done == 1 or done % max(1, total // 20) == 0 or done == total:
                print(f"{desc}: {done}/{total}")
            yield item

    return generator()


def generate_dataset_for_city(map_name: str) -> None:
    obj_path = os.path.join(MAP_DIR, f"{map_name}.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"map not found: {obj_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    city_output_dir = os.path.join(OUTPUT_DIR, map_name)
    os.makedirs(city_output_dir, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    mode_list = sample_mode_tasks()
    jammer_count_list = sample_jammer_count_tasks()

    tasks: List[Tuple[int, str, int, str, str, int]] = []
    skipped = 0
    base_seed = RANDOM_SEED * 100000 + abs(hash(map_name)) % 100000
    for idx, (mode, num_jammers) in enumerate(zip(mode_list, jammer_count_list), start=1):
        if RESUME_IF_EXISTS and maybe_existing_ok(city_output_dir, idx, expected_num_jammers=num_jammers):
            skipped += 1
            continue
        tasks.append((idx, mode, num_jammers, map_name, city_output_dir, base_seed + idx))

    print(f"\n===== {map_name} =====")
    print(f"output: {city_output_dir}")
    print(f"target: {NUM_TRAJECTORIES} | skipped: {skipped} | pending: {len(tasks)} | workers: {NUM_WORKERS}")
    print(
        f"frames: {RECOVERY_MIN_FRAMES}..{RECOVERY_MAX_FRAMES} "
        f"(mode={RECOVERY_MODE_FRAMES}) | recovery: "
        f"{RECOVERY_STEPS_RATIO_MIN:.2f}..{RECOVERY_STEPS_RATIO_MAX:.2f}x "
        f"clamped to {RECOVERY_STEPS_MIN}..{RECOVERY_STEPS_MAX}"
    )
    print(f"max_jammers: {RECOVERY_MAX_JAMMERS}")

    results = []
    failures = []
    if tasks:
        with mp.Pool(processes=NUM_WORKERS, initializer=init_worker_for_city, initargs=(obj_path,)) as pool:
            iterator = pool.imap_unordered(generate_single_trajectory, tasks, chunksize=1)
            for res in iter_with_progress(iterator, total=len(tasks), desc=map_name):
                results.append(res)
                if not res.get("success"):
                    failures.append(res)

    meta = summarize_city_output(city_output_dir, map_name)
    print(f"{map_name} done | generated: {meta['num_trajectories']}/{NUM_TRAJECTORIES}")
    print(f"{map_name} jammer distribution: {meta['jammer_count_distribution']}")

    if failures:
        err_log = os.path.join(city_output_dir, "failed_cases.json")
        with open(err_log, "w", encoding="utf-8") as f:
            json.dump(failures[:200], f, indent=2, ensure_ascii=False)
        print(f"{map_name} failures: {len(failures)} | saved: {err_log}")


def generate_all_cities() -> None:
    mp.set_start_method("spawn", force=True)
    print("=" * 80)
    print("Standalone recovery dataset generation")
    print(f"CITY_LIST = {CITY_LIST}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"NUM_TRAJECTORIES = {NUM_TRAJECTORIES}")
    print(f"MAX_JAMMERS = {RECOVERY_MAX_JAMMERS}")
    print("=" * 80)

    for city in CITY_LIST:
        generate_dataset_for_city(city)

    print("=" * 80)
    print("all done")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_cities()
