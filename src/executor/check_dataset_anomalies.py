#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DATASET_ROOT = "/data1/ysq/Worldmodel/Dataset"


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


@dataclass
class TrajectoryIssue:
    scene: str
    trajectory: str
    severity: str
    issue_type: str
    detail: str


def _check_frame_numeric(
    frame: Dict[str, Any], idx: int, scene: str, traj: str, issues: List[TrajectoryIssue]
) -> None:
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

    for k in vec_keys:
        if k in frame and frame[k] is not None and _xyz_from_any(frame[k]) is None:
            issues.append(
                TrajectoryIssue(
                    scene=scene,
                    trajectory=traj,
                    severity="error",
                    issue_type="invalid_vector",
                    detail=f"frame[{idx}] key='{k}' is not a valid finite xyz vector",
                )
            )
    for k in scalar_keys:
        if k in frame and frame[k] is not None and not _is_finite_number(frame[k]):
            issues.append(
                TrajectoryIssue(
                    scene=scene,
                    trajectory=traj,
                    severity="error",
                    issue_type="invalid_scalar",
                    detail=f"frame[{idx}] key='{k}' is not a finite number",
                )
            )


def _parse_frame_index(name: str) -> Optional[int]:
    stem = Path(name).stem
    if not stem.startswith("frame_"):
        return None
    n = stem.replace("frame_", "", 1)
    return int(n) if n.isdigit() else None


def _check_rgb_sequence(
    rgb_files: List[Path], scene: str, traj: str, issues: List[TrajectoryIssue]
) -> None:
    if not rgb_files:
        issues.append(
            TrajectoryIssue(scene, traj, "error", "missing_rgb", "rgb directory has no frame_*.png files")
        )
        return
    idxs = sorted(i for i in (_parse_frame_index(p.name) for p in rgb_files) if i is not None)
    if not idxs:
        issues.append(
            TrajectoryIssue(scene, traj, "error", "invalid_rgb_names", "no valid frame index found in rgb filenames")
        )
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
    """Return True if trajectory should be deleted by hard step threshold."""
    if len(frames) < 2:
        return False
    dists: List[float] = []
    bad_steps: List[Tuple[int, float]] = []
    prev_pos: Optional[Tuple[float, float, float]] = None
    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            prev_pos = None
            continue
        pos = _xyz_from_any(fr.get("uav_position"))
        if pos is None:
            prev_pos = None
            continue
        if prev_pos is not None:
            d = math.dist(prev_pos, pos)
            dists.append(d)
            if abs(d - expected_step) > step_tol:
                bad_steps.append((i, d))
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
        preview = ", ".join([f"(frame={i}, dist={d:.4f})" for i, d in bad_steps[:10]])
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


def check_trajectory(
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
    except Exception as e:
        issues.append(TrajectoryIssue(scene, traj, "error", "uav_json_parse_error", str(e)))
        return issues, stats

    frames = uav.get("trajectory")
    if not isinstance(frames, list):
        issues.append(TrajectoryIssue(scene, traj, "error", "invalid_uav_trajectory", "uav_trajectory.json missing list field 'trajectory'"))
        return issues, stats
    if len(frames) == 0:
        issues.append(TrajectoryIssue(scene, traj, "error", "empty_uav_trajectory", "trajectory list is empty"))
        return issues, stats

    stats["uav_frames"] = len(frames)

    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            issues.append(TrajectoryIssue(scene, traj, "error", "invalid_frame_type", f"frame[{i}] is not a dict"))
            continue
        _check_frame_numeric(fr, i, scene, traj, issues)

        if "frame_idx" in fr and isinstance(fr["frame_idx"], (int, float)):
            if int(fr["frame_idx"]) != i:
                issues.append(
                    TrajectoryIssue(
                        scene,
                        traj,
                        "warning",
                        "frame_idx_mismatch",
                        f"frame[{i}] has frame_idx={fr['frame_idx']}",
                    )
                )

    should_delete = _check_uav_step_distance(
        frames=frames,
        scene=scene,
        traj=traj,
        expected_step=1.0,
        step_tol=0.2,
        hard_delete_threshold=delete_step_threshold,
        issues=issues,
        stats=stats,
    )
    stats["deleted"] = False
    if should_delete and delete_bad_trajectories:
        try:
            shutil.rmtree(trajectory_dir)
            stats["deleted"] = True
            issues.append(
                TrajectoryIssue(
                    scene,
                    traj,
                    "error",
                    "trajectory_deleted_by_step_threshold",
                    (
                        f"trajectory deleted because max step distance="
                        f"{stats.get('uav_step_dist_max', 0.0):.4f}m > {delete_step_threshold:.4f}m"
                    ),
                )
            )
        except Exception as e:
            issues.append(
                TrajectoryIssue(
                    scene,
                    traj,
                    "error",
                    "trajectory_delete_failed",
                    f"failed to delete trajectory dir: {e}",
                )
            )

    rgb_files = sorted(rgb_dir.glob("frame_*.png")) if rgb_dir.exists() else []
    stats["rgb_frames"] = len(rgb_files)
    _check_rgb_sequence(rgb_files, scene, traj, issues)
    if rgb_files and len(rgb_files) != len(frames):
        issues.append(
            TrajectoryIssue(
                scene,
                traj,
                "warning",
                "frame_count_mismatch",
                f"uav_frames={len(frames)} rgb_frames={len(rgb_files)}",
            )
        )

    if target_json.exists():
        try:
            t = _load_json(target_json)
            tgt = t.get("trajectory") or t.get("target_trajectory") or t.get("target_trajectory_airsim")
            if isinstance(tgt, list):
                stats["target_frames"] = len(tgt)
                if len(tgt) != len(frames):
                    issues.append(
                        TrajectoryIssue(
                            scene,
                            traj,
                            "warning",
                            "target_length_mismatch",
                            f"uav_frames={len(frames)} target_frames={len(tgt)}",
                        )
                    )
                bad = 0
                for j, p in enumerate(tgt):
                    if _xyz_from_any(p) is None:
                        bad += 1
                        if bad <= 3:
                            issues.append(
                                TrajectoryIssue(
                                    scene,
                                    traj,
                                    "error",
                                    "invalid_target_point",
                                    f"target[{j}] is not valid finite xyz",
                                )
                            )
            else:
                issues.append(TrajectoryIssue(scene, traj, "warning", "invalid_target_trajectory", "target_trajectory.json has no recognized list field"))
        except Exception as e:
            issues.append(TrajectoryIssue(scene, traj, "error", "target_json_parse_error", str(e)))
    else:
        issues.append(TrajectoryIssue(scene, traj, "warning", "missing_target_json", str(target_json)))

    if jammer_json.exists():
        try:
            j = _load_json(jammer_json)
            jm = j.get("jammer_trajectories") or j.get("jammer_trajectories_airsim")
            if isinstance(jm, dict):
                stats["num_jammers"] = len(jm)
            elif isinstance(jm, list):
                stats["num_jammers"] = 1
            else:
                issues.append(TrajectoryIssue(scene, traj, "warning", "invalid_jammer_trajectories", "jammer_trajectories.json has invalid trajectory field"))
        except Exception as e:
            issues.append(TrajectoryIssue(scene, traj, "error", "jammer_json_parse_error", str(e)))

    return issues, stats


def main() -> None:
    parser = argparse.ArgumentParser("Check dataset anomalies and save report")
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--scene-list", type=str, default="")
    parser.add_argument("--head", type=int, default=0, help="Only check first N trajectories per scene; 0 means all.")
    parser.add_argument("--delete-step-threshold", type=float, default=1.2)
    parser.add_argument("--delete-bad-trajectories", action="store_true", default=True)
    parser.add_argument("--no-delete-bad-trajectories", action="store_false", dest="delete_bad_trajectories")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    if args.scene_list.strip():
        scenes = [s.strip() for s in args.scene_list.split(",") if s.strip()]
    else:
        scenes = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])

    all_issues: List[TrajectoryIssue] = []
    all_stats: List[Dict[str, Any]] = []
    checked = 0

    for scene in scenes:
        sdir = dataset_root / scene
        if not sdir.exists():
            all_issues.append(TrajectoryIssue(scene, "-", "error", "missing_scene_dir", str(sdir)))
            continue
        traj_dirs = sorted([d for d in sdir.glob("trajectory_*") if d.is_dir()])
        if args.head > 0:
            traj_dirs = traj_dirs[: args.head]
        for tdir in traj_dirs:
            checked += 1
            issues, stats = check_trajectory(
                tdir,
                scene,
                delete_bad_trajectories=bool(args.delete_bad_trajectories),
                delete_step_threshold=float(args.delete_step_threshold),
            )
            all_issues.extend(issues)
            all_stats.append(stats)

    err_n = sum(1 for i in all_issues if i.severity == "error")
    warn_n = sum(1 for i in all_issues if i.severity == "warning")
    deleted_n = sum(1 for s in all_stats if bool(s.get("deleted", False)))
    summary = {
        "dataset_root": str(dataset_root),
        "checked_scenes": scenes,
        "checked_trajectories": checked,
        "error_count": err_n,
        "warning_count": warn_n,
        "deleted_trajectories": deleted_n,
        "delete_step_threshold": float(args.delete_step_threshold),
        "delete_bad_trajectories": bool(args.delete_bad_trajectories),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output.strip()
        else dataset_root / f"dataset_anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summary,
        "issues": [asdict(x) for x in all_issues],
        "trajectory_stats": all_stats,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
