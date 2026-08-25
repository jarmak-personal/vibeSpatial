#!/usr/bin/env python3
"""Current-revision Q11 parent-vs-component public-path experiment."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CAPSULE = Path(__file__).resolve().parent
REPO = CAPSULE.parents[2]
DATA = Path("/home/picard/datasets/spatialbench/v0.1.0/sf100-geoparquet")
BENCHMARK_DIR = REPO / "benchmarks" / "spatialbench"
sys.path.insert(0, str(REPO))
sys.path.append(str(BENCHMARK_DIR))

os.environ.setdefault("VIBESPATIAL_STRICT_NATIVE", "1")


def _sha256_array(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _query_objects():
    from vibespatial_queries import VibeSpatialQueries

    import vibespatial as gpd
    from benchmarks.spatialbench.run_benchmark import get_data_paths

    gpd.set_execution_mode(gpd.ExecutionMode.AUTO)
    paths = get_data_paths(str(DATA))
    queries = VibeSpatialQueries(
        gpd,
        dissolve_method="coverage",
        distance_pair_rows=1_000_000,
        scan_batch_rows=32_000_000,
    )
    return gpd, paths, queries


def _component_frames(gpd, zones_all, directory: Path):
    """Explode zone components outside timing and reload through public IO."""
    frames = []
    parent_base = 0
    for frame_index, zones in enumerate(zones_all):
        parent_path = directory / f"zones-parent-{frame_index}.parquet"
        component_path = directory / f"zones-components-{frame_index}.parquet"
        zones.to_parquet(parent_path, geometry_encoding="geoarrow", index=False)
        # Importing upstream GeoPandas in this process would replace the
        # compatibility module used by lazily-created GeoSeries spatial
        # indexes. Keep fixture-only decomposition in a separate host process.
        code = """
import geopandas as gpd
import numpy as np
import sys
host = gpd.read_parquet(sys.argv[1])
host['parent_zone_row'] = np.arange(len(host), dtype=np.int64) + int(sys.argv[3])
exploded = host.explode(index_parts=False, ignore_index=True)
exploded = exploded[exploded.geometry.notna() & ~exploded.geometry.is_empty]
exploded.reset_index(drop=True).to_parquet(sys.argv[2], index=False)
"""
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(parent_path),
                str(component_path),
                str(parent_base),
            ],
            check=True,
            cwd=REPO,
        )
        frames.append(gpd.read_parquet(component_path))
        parent_base += len(zones)
    return frames


class _StageRecorder:
    def __init__(self) -> None:
        self._patches = []
        self._records = defaultdict(list)
        self._work_units = defaultdict(int)

    def _patch(self, module, name: str, label: str) -> None:
        import cupy as cp

        original = getattr(module, name)

        def wrapped(*args, **kwargs):
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            wall_start = time.perf_counter()
            if label == "exact_classification" and len(args) >= 5:
                self._work_units[label] += int(args[3].size)
            start.record()
            try:
                return original(*args, **kwargs)
            finally:
                stop.record()
                self._records[label].append(
                    (start, stop, time.perf_counter() - wall_start)
                )

        setattr(module, name, wrapped)
        self._patches.append((module, name, original))

    def __enter__(self):
        import vibespatial.spatial.point_grid_index as grid
        import vibespatial.spatial.spatial_index_device as device

        for module, name, label in (
            (grid, "point_grid_preflight", "grid_preflight"),
            (grid, "prepare_point_grid_index", "grid_prepare"),
            (grid, "point_grid_query_row_partitions", "grid_plan"),
            (grid, "point_grid_superset_query", "grid_candidates"),
            (
                grid,
                "point_grid_candidate_not_in_other_superset",
                "grid_exclusion",
            ),
            (
                device,
                "_classify_homogeneous_reduction_tile",
                "exact_classification",
            ),
        ):
            self._patch(module, name, label)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for module, name, original in reversed(self._patches):
            setattr(module, name, original)

    def summary(self):
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()
        summary = {}
        for label, records in self._records.items():
            item = {
                "calls": len(records),
                "gpu_seconds": sum(
                    float(cp.cuda.get_elapsed_time(start, stop))
                    for start, stop, _wall in records
                )
                / 1000.0,
                "python_wall_seconds": sum(wall for _start, _stop, wall in records),
            }
            if self._work_units[label]:
                item["work_units"] = self._work_units[label]
            summary[label] = item
        return summary


def _dispatch_summary(events):
    counts = Counter(
        (
            event.surface,
            event.operation,
            event.implementation,
            event.selected,
            event.reason,
        )
        for event in events
    )
    return [
        {
            "surface": key[0],
            "operation": key[1],
            "implementation": key[2],
            "selected": key[3],
            "reason": key[4],
            "count": count,
        }
        for key, count in counts.most_common()
    ]


def _first_trip_batch(queries, paths):
    columns = ["t_tripkey", "t_pickuploc", "t_dropoffloc"]
    return next(
        iter(
            queries._spatial_frames(
                paths["trip"],
                columns,
                "t_pickuploc",
                batch_rows=4_000_000,
            )
        )
    )


def run(variant: str, *, zone_frames: int, profile: bool) -> dict[str, object]:
    import vibespatial
    from vibespatial.predicates.point_region_profile import profile_point_region

    gpd, paths, queries = _query_objects()
    zones_parent = queries._zone_frames(paths)[:zone_frames]
    with tempfile.TemporaryDirectory(prefix="vs-r2-q11-components-") as raw_directory:
        zones_component = _component_frames(gpd, zones_parent, Path(raw_directory))
        zones_selected = zones_parent if variant == "parent" else zones_component
        trips = _first_trip_batch(queries, paths)
        pickup = trips.set_geometry("t_pickuploc").geometry
        dropoff = trips.set_geometry("t_dropoffloc").geometry

        # Each arm runs in an isolated process. Keep the cold public path in
        # the measured scope instead of manufacturing an indexed slice whose
        # SpatialIndex carrier differs from the full aligned endpoint frame.
        vibespatial.clear_dispatch_events()

        recorder = _StageRecorder()
        profile_context = (
            profile_point_region(label=f"r2-q11-{variant}")
            if profile
            else contextlib.nullcontext()
        )
        aggregates = []
        started = time.perf_counter()
        with recorder, profile_context as active_profile:
            for zones in zones_selected:
                aggregates.append(
                    queries._zone_pair_aggregates(pickup, dropoff, zones)
                )
            profile_snapshot = active_profile.snapshot() if profile else None
        wall_seconds = time.perf_counter() - started
        stage_timings = recorder.summary()

        left = np.concatenate(
            [item["left_count"].to_numpy(dtype=np.int64) for item in aggregates]
        )
        right = np.concatenate(
            [item["right_count"].to_numpy(dtype=np.int64) for item in aggregates]
        )
        shared = np.concatenate(
            [item["shared_count"].to_numpy(dtype=np.int64) for item in aggregates]
        )
        component_rows = sum(len(frame) for frame in zones_component)
        parent_rows = sum(len(frame) for frame in zones_parent)
        payload = {
            "schema_version": 1,
            "experiment": "q11_component_first",
            "variant": variant,
            "profile_enabled": profile,
            "trip_rows": len(trips),
            "zone_frames": len(zones_selected),
            "parent_zone_rows": parent_rows,
            "component_zone_rows": component_rows,
            "component_to_parent_ratio": component_rows / max(parent_rows, 1),
            "wall_seconds": wall_seconds,
            "stage_timings": stage_timings,
            "left_count_sha256": _sha256_array(left),
            "right_count_sha256": _sha256_array(right),
            "shared_count_sha256": _sha256_array(shared),
            "left_count_sum": int(left.sum()),
            "right_count_sum": int(right.sum()),
            "shared_count_sum": int(shared.sum()),
            "point_region_profile": profile_snapshot,
            "dispatch_summary": _dispatch_summary(
                vibespatial.get_dispatch_events(clear=True)
            ),
        }
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("parent", "component"), required=True)
    parser.add_argument("--zone-frames", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.variant, zone_frames=args.zone_frames, profile=args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
