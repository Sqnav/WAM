from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from eval.ortrack_target_heatmap import run
from tracking.runtime import SquareTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Square Tracker heatmap cache with one model load.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes", default="City_1,City_2,City_3")
    parser.add_argument("--trajectory-range", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-real-init-bbox", action="store_true", default=False)
    parser.add_argument("--square-init-bbox", action="store_true", default=False)
    parser.add_argument("--native-first-frame-response", action="store_true", default=False)
    parser.add_argument("--require-direct-area-heatmap", action="store_true", default=False)
    parser.add_argument("--direct-area-heatmap-size", type=int, default=7)
    parser.add_argument("--require-tracker-features", action="store_true", default=False)
    parser.add_argument("--tracker-feature-grid-size", type=int, default=7)
    parser.add_argument("--tracker-feature-dim", type=int, default=192)
    args = parser.parse_args()
    args.direct_area_heatmap_size = max(int(args.direct_area_heatmap_size), 1)
    import torch

    torch.set_num_threads(max(int(os.environ.get("OMP_NUM_THREADS", "2")), 1))

    start_text, end_text = args.trajectory_range.split("-", 1)
    start, end = int(start_text), int(end_text)
    tracker = SquareTracker(
        args.checkpoint,
        args.device,
        feature_grid_size=args.tracker_feature_grid_size,
    )
    expected_checkpoint = str(args.checkpoint.expanduser().resolve())
    checkpoint_stat = args.checkpoint.expanduser().resolve().stat()
    expected_checkpoint_size = int(checkpoint_stat.st_size)
    expected_checkpoint_mtime_ns = int(checkpoint_stat.st_mtime_ns)
    completed = 0
    for scene in [value.strip() for value in args.scenes.split(",") if value.strip()]:
        for index in range(start, end + 1):
            trajectory = args.dataset_root / scene / f"trajectory_{index:04d}"
            summary = args.cache_root / scene / trajectory.name / "summary.json"
            expected_frames = len(list((trajectory / "rgb").glob("frame_*.png")))
            if summary.is_file():
                try:
                    cached = json.loads(summary.read_text(encoding="utf-8"))
                    cache_matches = (
                        cached.get("tracker_backend") == "square"
                        and cached.get("checkpoint") == expected_checkpoint
                        and int(cached.get("checkpoint_size", -1)) == expected_checkpoint_size
                        and int(cached.get("checkpoint_mtime_ns", -1)) == expected_checkpoint_mtime_ns
                    )
                    frames = cached.get("frames")
                    cache_matches = (
                        cache_matches
                        and isinstance(frames, list)
                        and expected_frames > 0
                        and len(frames) == expected_frames
                        and int(cached.get("frame_count", -1)) == expected_frames
                    )
                    if args.require_real_init_bbox:
                        cache_matches = cache_matches and str(
                            cached.get("initialization", {}).get("backend", "")
                        ).startswith("gt_segmentation_bbox")
                    if args.square_init_bbox:
                        cache_matches = cache_matches and str(
                            cached.get("initialization", {}).get("backend", "")
                        ).endswith("_square")
                    if args.native_first_frame_response:
                        initialization = cached.get("initialization", {})
                        delayed_initialization = int(initialization.get("frame", 0)) > 0
                        cache_matches = cache_matches and (
                            cached.get("frame0_heatmap_source") == "tracker_response"
                            or (
                                delayed_initialization
                                and cached.get("frame0_heatmap_source")
                                == "target_absent_before_initialization"
                            )
                        )
                    if args.require_direct_area_heatmap:
                        cache_matches = cache_matches and cached.get("direct_area_heatmap_size") == [
                            args.direct_area_heatmap_size,
                            args.direct_area_heatmap_size,
                        ]
                    if args.require_tracker_features:
                        cache_matches = cache_matches and (
                            int(cached.get("tracker_feature_cache_version", 0)) == 1
                            and cached.get("tracker_feature_grid_size")
                            == [args.tracker_feature_grid_size, args.tracker_feature_grid_size]
                            and int(cached.get("tracker_feature_dim", -1))
                            == args.tracker_feature_dim
                        )
                    if cache_matches:
                        cache_matches = all(
                            isinstance(frame, dict)
                            and isinstance(frame.get("heatmap"), str)
                            and (summary.parent / frame["heatmap"]).is_file()
                            and (
                                not args.require_direct_area_heatmap
                                or (
                                    isinstance(frame.get("heatmap_direct_area"), str)
                                    and (summary.parent / frame["heatmap_direct_area"]).is_file()
                                )
                            )
                            and (
                                not args.require_tracker_features
                                or (
                                    isinstance(frame.get("tracker_features"), str)
                                    and (summary.parent / frame["tracker_features"]).is_file()
                                )
                            )
                            for frame in frames
                        )
                except (AttributeError, OSError, TypeError, ValueError):
                    cache_matches = False
                if cache_matches:
                    print(f"[cache-skip] {scene}/{trajectory.name}", flush=True)
                    completed += 1
                    continue
            run(
                SimpleNamespace(
                    trajectory=trajectory,
                    cache_root=args.cache_root,
                    tracker_backend="square",
                    tracker_checkpoint=args.checkpoint,
                    device=args.device,
                    fov_deg=90.0,
                    camera_offset_body=(0.46, 0.0, 0.0),
                    init_box_frac=0.10,
                    save_debug=False,
                    ortrack_root=Path("."),
                    checkpoint=Path("."),
                    config="square_box_uav_tracker",
                    init_dir_name="init",
                    output_dir_name="heatmaps",
                    summary_name="summary.json",
                    require_real_init_bbox=args.require_real_init_bbox,
                    square_init_bbox=args.square_init_bbox,
                    native_first_frame_response=args.native_first_frame_response,
                    direct_area_heatmap_size=args.direct_area_heatmap_size,
                    save_tracker_features=args.require_tracker_features,
                ),
                tracker=tracker,
            )
            completed += 1
            print(f"[cache] {completed} {scene}/{trajectory.name}", flush=True)
    print(f"[cache-complete] trajectories={completed} range={args.trajectory_range}", flush=True)


if __name__ == "__main__":
    main()
