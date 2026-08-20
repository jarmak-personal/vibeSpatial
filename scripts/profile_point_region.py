#!/usr/bin/env python3
"""Profile exact point/region refinement through public vibeSpatial APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

from vibespatial.api import GeoDataFrame, GeoSeries, points_from_xy, read_parquet
from vibespatial.cuda._runtime import (
    get_d2h_transfer_events,
    get_d2h_transfer_profile,
    reset_d2h_transfer_count,
)
from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
from vibespatial.runtime.materialization import (
    clear_materialization_events,
    get_materialization_events,
)


def _event_dict(event: Any) -> dict[str, Any]:
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    values = getattr(event, "__dict__", None)
    return dict(values) if isinstance(values, dict) else {"event": repr(event)}


def _device_identity() -> dict[str, Any]:
    import cupy as cp

    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return {
        "name": name,
        "compute_capability": [int(properties["major"]), int(properties["minor"])],
        "multiprocessor_count": int(properties["multiProcessorCount"]),
        "total_global_memory_bytes": int(properties["totalGlobalMem"]),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
    }


def _source_identity() -> dict[str, Any]:
    """Identify the imported source tree, including PYTHONPATH worktrees."""
    import vibespatial

    source_file = Path(vibespatial.__file__).resolve()
    repository = source_file.parents[2]

    def _git(*args: str) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                ("git", "-C", str(repository), *args),
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

    revision_result = _git("rev-parse", "HEAD")
    status_result = _git("status", "--porcelain", "--untracked-files=no")
    diff_result = _git(
        "diff",
        "--no-ext-diff",
        "HEAD",
        "--",
        "src",
        "scripts",
        "benchmarks",
    )
    untracked_result = _git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "src",
        "scripts",
        "benchmarks",
    )
    untracked_source_files = (
        sorted(filter(None, untracked_result.stdout.splitlines()))
        if untracked_result is not None and untracked_result.returncode == 0
        else []
    )
    source_fingerprint = None
    if diff_result is not None and diff_result.returncode == 0:
        digest = hashlib.sha256(diff_result.stdout.encode("utf-8"))
        for relative_path in untracked_source_files:
            path = repository / relative_path
            if not path.is_file():
                continue
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
        source_fingerprint = digest.hexdigest()
    return {
        "package_version": str(getattr(vibespatial, "__version__", "unknown")),
        "source_file": str(source_file),
        "git_revision": (
            revision_result.stdout.strip()
            if revision_result is not None and revision_result.returncode == 0
            else None
        ),
        "tracked_worktree_dirty": (
            bool(status_result.stdout.strip())
            if status_result is not None and status_result.returncode == 0
            else None
        ),
        "untracked_source_files": untracked_source_files,
        "worktree_source_sha256": source_fingerprint,
        "python_version": platform.python_version(),
        "cupy_version": __import__("cupy").__version__,
    }


def _circle(vertex_count: int) -> Polygon:
    angles = np.linspace(0.0, 2.0 * math.pi, vertex_count, endpoint=False)
    return Polygon(np.column_stack((np.cos(angles), np.sin(angles))))


def _point_grid(count: int, *, xmin: float, xmax: float, ymin: float, ymax: float):
    side = max(2, int(math.ceil(math.sqrt(count))))
    x_values = np.linspace(xmin, xmax, side, dtype=np.float64)
    y_values = np.linspace(ymin, ymax, side, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return x_grid.reshape(-1)[:count], y_grid.reshape(-1)[:count]


def _shape_cases(point_count: int):
    simple_x, simple_y = _point_grid(
        point_count, xmin=-0.1, xmax=1.1, ymin=-0.1, ymax=1.1
    )
    simple = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.4, 1.1)])
    yield "simple_short_polygon", simple_x, simple_y, [simple]

    circle_x, circle_y = _point_grid(
        point_count, xmin=-0.99, xmax=0.99, ymin=-0.99, ymax=0.99
    )
    yield "long_selected_bin", circle_x, circle_y, [_circle(16_384)]

    part_count = 256
    parts = [
        box(float(part * 4), 0.0, float(part * 4 + 1), 1.0)
        for part in range(part_count)
    ]
    skew_x, skew_y = _point_grid(
        point_count,
        xmin=0.01,
        xmax=float((part_count - 1) * 4 + 0.99),
        ymin=0.01,
        ymax=0.99,
    )
    yield "multipart_envelope_skew", skew_x, skew_y, [MultiPolygon(parts)]

    uniform_x, uniform_y = _point_grid(
        point_count,
        xmin=0.01,
        xmax=float((part_count - 1) * 4 + 0.99),
        ymin=0.01,
        ymax=0.99,
    )
    uniform_polygons = [
        Polygon(
            [
                (float(part * 4), 0.0),
                (float(part * 4 + 1), 0.0),
                (float(part * 4 + 0.5), 1.0),
            ]
        )
        for part in range(part_count)
    ]
    yield "uniform_many_small_polygons", uniform_x, uniform_y, uniform_polygons


def _oracle_counts(x_values, y_values, zones) -> np.ndarray:
    from shapely import contains_xy

    # This is a test-only host oracle, not part of the measured GPU path. Keep
    # it vectorized so large protected shapes do not spend their wall time in
    # Python geometry-object construction.
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    counts = np.zeros(x_values.size, dtype=np.int64)
    for zone in zones:
        counts += contains_xy(zone, x_values, y_values)
    return counts


def _device_resident_regions(zones) -> GeoSeries:
    """Use public GeoParquet IO to reproduce the production native carrier."""
    with tempfile.TemporaryDirectory(prefix="vibespatial-point-region-") as directory:
        path = Path(directory) / "regions.parquet"
        frame = GeoDataFrame(
            {"row_id": np.arange(len(zones), dtype=np.int64)},
            geometry=GeoSeries(zones),
        )
        frame.to_parquet(path, geometry_encoding="geoarrow", index=False)
        return read_parquet(path).geometry


def _aligned_device_points(x_values, y_values):
    """Build aligned endpoint columns through the same public IO contract as Q11."""
    pickup = points_from_xy(x_values, y_values, crs="EPSG:4326")
    dropoff = points_from_xy(x_values[::-1], y_values[::-1], crs="EPSG:4326")
    frame = GeoDataFrame({"dropoff": dropoff}, geometry=pickup).rename_geometry(
        "pickup"
    )
    frame["dropoff"] = frame["dropoff"].set_crs(frame.crs)
    with tempfile.TemporaryDirectory(prefix="vibespatial-aligned-points-") as directory:
        path = Path(directory) / "points.parquet"
        frame.to_parquet(path, geometry_encoding="geoarrow", index=False)
        native = read_parquet(path)
    return native.set_geometry("pickup").geometry, native.set_geometry(
        "dropoff"
    ).geometry


def _run_case(
    name: str,
    x_values,
    y_values,
    zones,
    *,
    repeat: int,
    profile_enabled: bool,
) -> dict[str, Any]:
    pickup, dropoff = _aligned_device_points(x_values, y_values)
    region_series = _device_resident_regions(zones)
    expected_left = _oracle_counts(x_values, y_values, zones)
    expected_right = _oracle_counts(x_values[::-1], y_values[::-1], zones)
    runs: list[dict[str, Any]] = []

    for run_number in range(repeat):
        reset_d2h_transfer_count()
        clear_fallback_events()
        clear_materialization_events()
        if profile_enabled:
            from vibespatial.predicates.point_region_profile import (
                profile_point_region,
            )

            profile_context = profile_point_region(
                label=f"{name}-run-{run_number + 1}",
                force_prepared_index=True,
            )
        else:
            profile_context = nullcontext(None)
        with profile_context as profile:
            started = time.perf_counter()
            result = pickup.sindex.query_pair_aggregate(
                dropoff.sindex,
                region_series,
                predicate="contains",
            )
            wall_seconds = time.perf_counter() - started
            actual_left = result["left_count"].to_numpy()
            actual_right = result["right_count"].to_numpy()
            point_profile = None if profile is None else profile.snapshot()
        transfer_count, transfer_bytes, transfer_seconds = get_d2h_transfer_profile()
        fallback_events = get_fallback_events(clear=True)
        materialization_events = get_materialization_events(clear=True)
        transfer_events = get_d2h_transfer_events(clear=True)
        if not np.array_equal(actual_left, expected_left):
            raise AssertionError(f"{name}: left counts differ from Shapely oracle")
        if not np.array_equal(actual_right, expected_right):
            raise AssertionError(f"{name}: right counts differ from Shapely oracle")
        runs.append(
            {
                "run": run_number + 1,
                "wall_seconds": wall_seconds,
                "point_region_profile": point_profile,
                "d2h_transfer_count": transfer_count,
                "d2h_transfer_bytes": transfer_bytes,
                "d2h_transfer_seconds": transfer_seconds,
                "d2h_transfer_events": [_event_dict(event) for event in transfer_events],
                "fallback_events": [_event_dict(event) for event in fallback_events],
                "materialization_events": [
                    _event_dict(event) for event in materialization_events
                ],
            }
        )
    return {
        "name": name,
        "point_count": int(len(x_values)),
        "region_count": len(zones),
        "variant": "profiled_lane_fp64" if profile_enabled else "production_auto",
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=16_384)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="measure the production path without profiler instrumentation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.points < 1:
        parser.error("--points must be at least 1")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    payload = {
        "schema_version": 2,
        "workload_entrypoint": "GeoSeries.sindex.query_pair_aggregate",
        "prepared_index_forced_for_shape_corpus": not args.measure_only,
        "variant": "production_auto" if args.measure_only else "profiled_lane_fp64",
        "source": _source_identity(),
        "device": _device_identity(),
        "cases": [
            _run_case(
                name,
                x_values,
                y_values,
                zones,
                repeat=args.repeat,
                profile_enabled=not args.measure_only,
            )
            for name, x_values, y_values, zones in _shape_cases(args.points)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
