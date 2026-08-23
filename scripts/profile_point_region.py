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
    get_cuda_runtime,
    get_d2h_transfer_events,
    get_d2h_transfer_profile,
    reset_d2h_transfer_count,
)
from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
from vibespatial.runtime.materialization import (
    clear_materialization_events,
    get_materialization_events,
)

_CASE_NAMES = (
    "simple_short_polygon",
    "long_selected_bin",
    "multipart_envelope_skew",
    "uniform_many_small_polygons",
    "clustered_points_extent_skew",
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


def _device_memory_snapshot() -> dict[str, Any]:
    """Capture live allocator and driver state outside measured regions."""
    import cupy as cp

    runtime = get_cuda_runtime()
    driver_free, driver_total = cp.cuda.runtime.memGetInfo()
    return {
        "driver_free_bytes": int(driver_free),
        "driver_total_bytes": int(driver_total),
        "pool": runtime.memory_pool_stats(),
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


def _shape_cases(point_count: int, *, cluster_extent: float = 1024.0):
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

    # Isolate point-partition skew: almost all points occupy a tiny region, but
    # four finite outliers stretch the global index extent. A fixed global grid
    # groups the dense core into very few cells, so many disjoint query boxes
    # repeatedly refine the same conservative point superset. An adaptive
    # hierarchy should win here if its additional construction/traversal cost
    # is justified.
    core_count = max(point_count - 4, 1)
    clustered_x, clustered_y = _point_grid(
        core_count, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0
    )
    if point_count > core_count:
        outlier_count = point_count - core_count
        outlier_x = np.array(
            [-cluster_extent, cluster_extent, -cluster_extent, cluster_extent],
            dtype=np.float64,
        )
        outlier_y = np.array(
            [-cluster_extent, -cluster_extent, cluster_extent, cluster_extent],
            dtype=np.float64,
        )
        clustered_x = np.concatenate((clustered_x, outlier_x[:outlier_count]))
        clustered_y = np.concatenate((clustered_y, outlier_y[:outlier_count]))
    grid_side = 8
    gap = 0.08 / grid_side
    clustered_zones = [
        box(
            column / grid_side + gap,
            row / grid_side + gap,
            (column + 1) / grid_side - gap,
            (row + 1) / grid_side - gap,
        )
        for row in range(grid_side)
        for column in range(grid_side)
    ]
    yield (
        "clustered_points_extent_skew",
        clustered_x,
        clustered_y,
        clustered_zones,
    )


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


def _point_partition_diagnostics(spatial_index, zones) -> dict[str, Any]:
    """Export diagnostic partition shapes after all timed public runs."""
    _owned, flat_index = spatial_index._owned_flat_sindex()
    native_index = flat_index.to_native_spatial_index()
    from vibespatial.spatial.point_grid_index import PreparedPointGridIndex

    prepared = next(
        (
            value
            for value in native_index.point_partition_cache.values()
            if isinstance(value, PreparedPointGridIndex)
        ),
        None,
    )
    runtime = get_cuda_runtime()
    diagnostics: dict[str, Any] = {"dense_grid_prepared": prepared is not None}
    if prepared is not None:
        cell_counts = runtime.copy_device_to_host(
            prepared.cell_counts,
            reason="point-region benchmark dense-grid occupancy diagnostics",
        )
        occupied = np.asarray(cell_counts, dtype=np.int64)
        occupied = occupied[occupied > 0]
        diagnostics.update(
            {
                "grid_size": int(prepared.grid_size),
                "cell_count": int(prepared.grid_size * prepared.grid_size),
                "occupied_cell_count": int(occupied.size),
                "max_cell_occupancy": int(occupied.max(initial=0)),
                "mean_occupied_cell_occupancy": (
                    float(occupied.mean()) if occupied.size else 0.0
                ),
                "occupied_cell_percentiles": {
                    "p50": float(np.percentile(occupied, 50)) if occupied.size else 0.0,
                    "p95": float(np.percentile(occupied, 95)) if occupied.size else 0.0,
                    "p99": float(np.percentile(occupied, 99)) if occupied.size else 0.0,
                },
                "index_device_bytes": int(prepared.device_bytes),
            }
        )

    from vibespatial.spatial.spatial_index_device import _prepare_morton_range_query

    query_bounds = np.asarray([zone.bounds for zone in zones], dtype=np.float64)
    state = _prepare_morton_range_query(flat_index, query_bounds, query_bounds)
    if state is None:
        diagnostics["morton_range_prepared"] = False
        return diagnostics
    try:
        spans = runtime.copy_device_to_host(
            state.d_ends - state.d_starts,
            reason="point-region benchmark Morton-span diagnostics",
        ).astype(np.int64, copy=False)
    finally:
        state.close()
    diagnostics.update(
        {
            "morton_range_prepared": True,
            "morton_span_total": int(spans.sum()),
            "morton_span_max": int(spans.max(initial=0)),
            "morton_span_per_tree_row": float(
                spans.sum() / max(int(flat_index.size), 1)
            ),
            "morton_span_percentiles": {
                "p50": float(np.percentile(spans, 50)) if spans.size else 0.0,
                "p95": float(np.percentile(spans, 95)) if spans.size else 0.0,
                "p99": float(np.percentile(spans, 99)) if spans.size else 0.0,
            },
        }
    )
    return diagnostics


def _run_case(
    name: str,
    x_values,
    y_values,
    zones,
    *,
    repeat: int,
    profile_enabled: bool,
    point_partition: str,
) -> dict[str, Any]:
    memory_snapshots = {"before_setup": _device_memory_snapshot()}
    pickup, dropoff = _aligned_device_points(x_values, y_values)
    memory_snapshots["after_point_io"] = _device_memory_snapshot()
    region_series = _device_resident_regions(zones)
    pickup_index = pickup.sindex
    dropoff_index = dropoff.sindex
    memory_snapshots["after_public_indexes"] = _device_memory_snapshot()
    expected_left = _oracle_counts(x_values, y_values, zones)
    expected_right = _oracle_counts(x_values[::-1], y_values[::-1], zones)
    memory_snapshots["after_oracle"] = _device_memory_snapshot()
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
            result = pickup_index.query_pair_aggregate(
                dropoff_index,
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
        memory_snapshot = _device_memory_snapshot()
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
                "device_memory": memory_snapshot,
            }
        )
    return {
        "name": name,
        "point_count": int(len(x_values)),
        "region_count": len(zones),
        "oracle_left_hits": int(expected_left.sum()),
        "oracle_right_hits": int(expected_right.sum()),
        "point_partition_diagnostics": _point_partition_diagnostics(
            pickup_index,
            zones,
        ),
        "device_memory_snapshots": memory_snapshots,
        "variant": (
            f"profiled_lane_fp64_{point_partition}"
            if profile_enabled
            else f"production_{point_partition}"
        ),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=16_384)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--cluster-extent",
        type=float,
        default=1024.0,
        help="finite outlier extent for the clustered point-partition canary",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=_CASE_NAMES,
        help="run only the named shape; repeat to select multiple shapes",
    )
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="measure the production path without profiler instrumentation",
    )
    parser.add_argument(
        "--point-partition",
        choices=("auto", "morton", "grid"),
        default="auto",
        help=(
            "candidate partitioning policy; non-auto choices are diagnostic "
            "vS-native floors reached through the public aggregate call"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.points < 1:
        parser.error("--points must be at least 1")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if not np.isfinite(args.cluster_extent) or args.cluster_extent < 1.0:
        parser.error("--cluster-extent must be finite and at least 1")

    from vibespatial.spatial.point_partition import (
        PointPartitionVariant,
        force_point_partition_variant_for_testing,
    )

    forced_variant = (
        None
        if args.point_partition == "auto"
        else PointPartitionVariant(args.point_partition)
    )

    selected_cases = set(args.case or _CASE_NAMES)
    with force_point_partition_variant_for_testing(forced_variant):
        payload = {
            "schema_version": 2,
            "workload_entrypoint": "GeoSeries.sindex.query_pair_aggregate",
            "prepared_index_forced_for_shape_corpus": not args.measure_only,
            "variant": (
                f"production_{args.point_partition}"
                if args.measure_only
                else f"profiled_lane_fp64_{args.point_partition}"
            ),
            "point_partition": args.point_partition,
            "cluster_extent": args.cluster_extent,
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
                    point_partition=args.point_partition,
                )
                for name, x_values, y_values, zones in _shape_cases(
                    args.points,
                    cluster_extent=args.cluster_extent,
                )
                if name in selected_cases
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
