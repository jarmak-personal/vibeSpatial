from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from vibespatial import ExecutionMode, from_shapely_geometries, has_gpu_runtime
from vibespatial.kernels.predicates import point_in_polygon, point_within_bounds
from vibespatial.kernels.predicates.point_in_polygon import _evaluate_point_in_polygon_gpu
from vibespatial.kernels.predicates.point_within_bounds import _normalize_right_input
from vibespatial.testing import SyntheticSpec, generate_points, generate_polygons


def build_inputs(rows: int):
    points = generate_points(
        SyntheticSpec(geometry_type="point", distribution="uniform", count=rows, seed=7)
    ).geometries
    polygons = generate_polygons(
        SyntheticSpec(geometry_type="polygon", distribution="regular-grid", count=max(1, rows // 10), seed=3)
    ).geometries
    tiled = [polygons[index % len(polygons)] for index in range(rows)]
    return from_shapely_geometries(list(points)), from_shapely_geometries(tiled)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark point predicate bootstrap kernels.")
    parser.add_argument("--rows", type=int, default=10_000)
    args = parser.parse_args()

    points, polygons = build_inputs(args.rows)

    started = perf_counter()
    bounds_hits = point_within_bounds(points, polygons, dispatch_mode=ExecutionMode.CPU)
    bounds_elapsed = perf_counter() - started

    started = perf_counter()
    exact_hits = point_in_polygon(points, polygons, dispatch_mode=ExecutionMode.CPU)
    exact_elapsed = perf_counter() - started

    right = _normalize_right_input(polygons, expected_len=points.row_count)
    coarse = point_within_bounds(points, polygons, dispatch_mode=ExecutionMode.CPU)
    candidate_rows = sum(value is True for value in coarse)

    gpu_metrics: dict[str, float | int | None] = {
        "gpu_available": int(has_gpu_runtime()),
        "gpu_auto_elapsed_seconds": None,
        "gpu_dense_elapsed_seconds": None,
        "gpu_compacted_elapsed_seconds": None,
    }
    if has_gpu_runtime():
        points.move_to(
            "device",
            trigger="explicit-runtime-request",
            reason="benchmarking GPU point_in_polygon variants",
        )
        polygons.move_to(
            "device",
            trigger="explicit-runtime-request",
            reason="benchmarking GPU point_in_polygon variants",
        )
        started = perf_counter()
        gpu_auto = _evaluate_point_in_polygon_gpu(points, right, strategy="auto")
        gpu_metrics["gpu_auto_elapsed_seconds"] = perf_counter() - started

        started = perf_counter()
        gpu_dense = _evaluate_point_in_polygon_gpu(points, right, strategy="dense")
        gpu_metrics["gpu_dense_elapsed_seconds"] = perf_counter() - started

        started = perf_counter()
        gpu_compacted = _evaluate_point_in_polygon_gpu(points, right, strategy="compacted")
        gpu_metrics["gpu_compacted_elapsed_seconds"] = perf_counter() - started

        if not np.array_equal(np.asarray(gpu_auto, dtype=object), np.asarray(gpu_dense, dtype=object)):
            raise RuntimeError("GPU auto and dense point_in_polygon strategies diverged")
        if not np.array_equal(np.asarray(gpu_auto, dtype=object), np.asarray(gpu_compacted, dtype=object)):
            raise RuntimeError("GPU auto and compacted point_in_polygon strategies diverged")

    print(
        json.dumps(
            {
                "rows": args.rows,
                "candidate_rows": candidate_rows,
                "candidate_ratio": candidate_rows / max(args.rows, 1),
                "bounds_hits": sum(value is True for value in bounds_hits),
                "point_in_polygon_hits": sum(value is True for value in exact_hits),
                "bounds_elapsed_seconds": bounds_elapsed,
                "point_in_polygon_elapsed_seconds": exact_elapsed,
                **gpu_metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
