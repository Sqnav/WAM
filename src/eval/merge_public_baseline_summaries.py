from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


def _mean(rows: list[dict[str, Any]], key: str, boolean: bool = False):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(float(bool(value)) if boolean else float(value))
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--scene-list", required=True)
    parser.add_argument("--trajectory-spec", required=True)
    parser.add_argument("--policy-backend", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    cities = [value.strip() for value in args.scene_list.split(",") if value.strip()]
    summaries: list[dict[str, Any]] = []
    shard_args: dict[str, Any] | None = None
    for city in cities:
        path = model_dir / f"summary_{city}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("summaries", [])
        if any(str(row.get("failure_reason") or "").lower() == "runtime_error" for row in rows):
            raise RuntimeError(f"runtime_error remains in {path}")
        summaries.extend(rows)
        shard_args = payload.get("args", shard_args)

    unique = {
        (str(row.get("scene_id")), str(row.get("trajectory_name"))): row
        for row in summaries
    }
    summaries = [unique[key] for key in sorted(unique)]
    failures = Counter(str(row.get("failure_reason") or "unknown") for row in summaries)
    aggregate_args = dict(shard_args or {})
    aggregate_args.update(
        {
            "scene_list": ",".join(cities),
            "trajectory_range": args.trajectory_spec,
            "policy_backend": args.policy_backend,
        }
    )
    aggregate = {
        "num_trajectories": len(summaries),
        "SR": _mean(summaries, "success", True),
        "success_rate": _mean(summaries, "success", True),
        "ATF": _mean(summaries, "effective_tracked_frames"),
        "average_tracked_frames": _mean(summaries, "effective_tracked_frames"),
        "average_tracked_frame_ratio": _mean(summaries, "effective_tracking_ratio"),
        "CTF": _mean(summaries, "consecutive_tracked_frames_before_failure"),
        "collision_rate": _mean(summaries, "collision", True),
        "mean_final_distance": _mean(summaries, "final_distance"),
        "mean_distance": _mean(summaries, "mean_distance"),
        "failure_reason_counts": dict(failures),
        "args": aggregate_args,
        "summaries": summaries,
    }
    temporary = model_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(model_dir / "summary.json")
    print(
        f"[public-eval-merge] model={args.policy_backend} trajectories={len(summaries)} "
        f"SR={100.0 * float(aggregate['SR'] or 0.0):.2f}%"
    )


if __name__ == "__main__":
    main()
