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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Sequence


DEFAULT_EXECUTOR_SCRIPT = "/data1/ysq/Worldmodel/code/src/executor/trajectory_executor.py"
DEFAULT_TRAJECTORY_DIR = "/data1/ysq/Worldmodel/Plandataset"
DEFAULT_DATASET_BASE_DIR = "/data1/ysq/Worldmodel/Dataset"
DEFAULT_SCENES = ["City_1", "City_2", "City_3"]
DEFAULT_GPUS = [0]
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
DEFAULT_JUMP_THRESHOLD = 10.0
DEFAULT_JAMMER_ENABLED = True
DEFAULT_JAMMER_OBJECT_NAME = "JammerUAV"
DEFAULT_JAMMER_ASSET_NAME = "UAV1"
DEFAULT_RANDOM_JAMMER_ASSET = True
DEFAULT_JAMMER_SCALE = (1.0, 1.0, 1.0)


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
        help="场景列表，例如 City_1 City_2 City_3 City_4",
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
        return {
            "scene_id": scene_id,
            "gpu_id": gpu_id,
            "num_files": total,
        }
    finally:
        cleanup_executor_connections(executor)


def worker_entry(args, gpu_id, scene_ids, progress_position=0):
    results = []
    for scene_id in scene_ids:
        results.append(run_scene_batch(args, scene_id, gpu_id, progress_position))
    return results


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
        # 串行模式，或者虽然开了 multi-worker 但实际上只有一个 worker
        for worker_idx, (gpu_id, scene_ids) in enumerate(gpu_assignment.items()):
            worker_entry(args, gpu_id, scene_ids, worker_idx)

    print("\n✓ 全部 scene 执行完成")


if __name__ == "__main__":
    main()
