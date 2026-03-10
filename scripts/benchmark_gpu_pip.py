from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import shapely

from vibespatial import ExecutionMode, from_shapely_geometries, has_gpu_runtime
from vibespatial.kernels.predicates.point_in_polygon import (
    get_last_gpu_substage_timings,
    point_in_polygon,
)
from vibespatial.testing import SyntheticSpec, generate_points, generate_polygons


def build_inputs(scale: int, *, polygon_base_count: int, hole_probability: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        list(
            generate_points(
                SyntheticSpec(
                    geometry_type="point",
                    distribution="grid",
                    count=scale,
                    seed=0,
                )
            ).geometries
        ),
        dtype=object,
    )
    base_polygons = np.asarray(
        list(
            generate_polygons(
                SyntheticSpec(
                    geometry_type="polygon",
                    distribution="regular-grid",
                    count=max(polygon_base_count, 1),
                    seed=1,
                    vertices=5,
                    hole_probability=hole_probability,
                )
            ).geometries
        ),
        dtype=object,
    )
    polygons = np.resize(base_polygons, scale).astype(object, copy=False)
    return points, polygons


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark exact GPU point-in-polygon against CPU and Shapely baselines.")
    parser.add_argument("--scale", type=int, default=1_000_000)
    parser.add_argument("--polygon-base-count", type=int, default=0)
    parser.add_argument("--hole-probability", type=float, default=0.0)
    parser.add_argument("--skip-shapely", action="store_true")
    args = parser.parse_args()

    if not has_gpu_runtime():
        raise SystemExit("CUDA runtime not available")

    polygon_base_count = args.polygon_base_count or max(args.scale // 8, 1)
    points, polygons = build_inputs(
        args.scale,
        polygon_base_count=polygon_base_count,
        hole_probability=args.hole_probability,
    )
    points_owned = from_shapely_geometries(points.tolist())
    polygons_owned = from_shapely_geometries(polygons.tolist())

    shapely_elapsed = None
    shapely_true_rows = None
    if not args.skip_shapely:
        started = perf_counter()
        shapely_result = np.asarray(shapely.covers(polygons, points), dtype=bool)
        shapely_elapsed = perf_counter() - started
        shapely_true_rows = int(np.count_nonzero(shapely_result))

    started = perf_counter()
    cpu_result = point_in_polygon(points_owned, polygons_owned, dispatch_mode=ExecutionMode.CPU)
    cpu_elapsed = perf_counter() - started
    cpu_values = np.asarray(cpu_result, dtype=object)

    started = perf_counter()
    cold_gpu_result = point_in_polygon(points_owned, polygons_owned, dispatch_mode=ExecutionMode.GPU)
    cold_gpu_elapsed = perf_counter() - started
    cold_gpu_values = np.asarray(cold_gpu_result, dtype=object)

    started = perf_counter()
    warm_gpu_result = point_in_polygon(points_owned, polygons_owned, dispatch_mode=ExecutionMode.GPU)
    warm_gpu_elapsed = perf_counter() - started
    warm_gpu_values = np.asarray(warm_gpu_result, dtype=object)
    gpu_substages = get_last_gpu_substage_timings()

    if not np.array_equal(cold_gpu_values, cpu_values):
        raise RuntimeError("cold GPU point_in_polygon result diverged from CPU result")
    if not np.array_equal(warm_gpu_values, cpu_values):
        raise RuntimeError("warm GPU point_in_polygon result diverged from CPU result")
    if shapely_true_rows is not None:
        if int(np.count_nonzero(cpu_values == True)) != shapely_true_rows:  # noqa: E712
            raise RuntimeError("CPU point_in_polygon result diverged from Shapely covers baseline")

    payload = {
        "benchmark": "gpu_point_in_polygon",
        "scale": args.scale,
        "polygon_base_count": polygon_base_count,
        "hole_probability": args.hole_probability,
        "cpu_true_rows": int(np.count_nonzero(cpu_values == True)),  # noqa: E712
        "gpu_true_rows": int(np.count_nonzero(warm_gpu_values == True)),  # noqa: E712
        "cpu_elapsed_seconds": cpu_elapsed,
        "gpu_cold_elapsed_seconds": cold_gpu_elapsed,
        "gpu_warm_elapsed_seconds": warm_gpu_elapsed,
        "gpu_substage_timings": gpu_substages,
        "gpu_speedup_vs_cpu": (cpu_elapsed / warm_gpu_elapsed) if warm_gpu_elapsed else None,
        "shapely_true_rows": shapely_true_rows,
        "shapely_elapsed_seconds": shapely_elapsed,
        "gpu_speedup_vs_shapely": (
            (shapely_elapsed / warm_gpu_elapsed) if shapely_elapsed and warm_gpu_elapsed else None
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
