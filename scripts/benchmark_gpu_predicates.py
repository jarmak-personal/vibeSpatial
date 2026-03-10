from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import shapely
from shapely.affinity import translate

from vibespatial import ExecutionMode, NullBehavior, evaluate_binary_predicate, from_shapely_geometries
from vibespatial.testing import SyntheticSpec, generate_points, generate_polygons


def build_contains_inputs(scale: int) -> tuple[np.ndarray, np.ndarray]:
    polygons = np.asarray(
        list(generate_polygons(SyntheticSpec(geometry_type="polygon", distribution="regular-grid", count=scale, seed=0)).geometries),
        dtype=object,
    )
    points = np.asarray(
        list(generate_points(SyntheticSpec(geometry_type="point", distribution="grid", count=scale, seed=1)).geometries),
        dtype=object,
    )
    half = scale // 2
    if half < scale:
        shifted = [translate(geometry, xoff=10_000.0, yoff=10_000.0) for geometry in points[half:]]
        points[half:] = np.asarray(shifted, dtype=object)
    return polygons, points


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GPU-backed binary predicates on point-centric workloads.")
    parser.add_argument("--scale", type=int, default=100_000)
    args = parser.parse_args()

    left, right = build_contains_inputs(args.scale)
    left_owned = from_shapely_geometries(left.tolist())
    right_owned = from_shapely_geometries(right.tolist())

    started = perf_counter()
    shapely_result = shapely.contains(left, right)
    shapely_elapsed = perf_counter() - started

    started = perf_counter()
    cpu_result = evaluate_binary_predicate(
        "contains",
        left,
        right,
        dispatch_mode=ExecutionMode.CPU,
        null_behavior=NullBehavior.FALSE,
    )
    cpu_elapsed = perf_counter() - started

    started = perf_counter()
    evaluate_binary_predicate(
        "contains",
        left_owned,
        right_owned,
        dispatch_mode=ExecutionMode.GPU,
        null_behavior=NullBehavior.FALSE,
    )
    cold_gpu_elapsed = perf_counter() - started

    started = perf_counter()
    warm_gpu_result = evaluate_binary_predicate(
        "contains",
        left_owned,
        right_owned,
        dispatch_mode=ExecutionMode.GPU,
        null_behavior=NullBehavior.FALSE,
    )
    warm_gpu_elapsed = perf_counter() - started

    gpu_values = np.asarray(warm_gpu_result.values, dtype=bool)
    print(
        json.dumps(
            {
                "predicate": "contains",
                "scale": args.scale,
                "shapely_true_rows": int(np.count_nonzero(shapely_result)),
                "cpu_true_rows": int(np.count_nonzero(np.asarray(cpu_result.values, dtype=bool))),
                "gpu_true_rows": int(np.count_nonzero(gpu_values)),
                "candidate_rows": int(warm_gpu_result.candidate_rows.size),
                "coarse_false_rows": int(warm_gpu_result.coarse_false_rows.size),
                "shapely_elapsed_seconds": shapely_elapsed,
                "cpu_elapsed_seconds": cpu_elapsed,
                "gpu_cold_elapsed_seconds": cold_gpu_elapsed,
                "gpu_warm_elapsed_seconds": warm_gpu_elapsed,
                "gpu_cold_speedup_vs_shapely": (
                    shapely_elapsed / cold_gpu_elapsed if cold_gpu_elapsed else None
                ),
                "gpu_warm_speedup_vs_shapely": (
                    shapely_elapsed / warm_gpu_elapsed if warm_gpu_elapsed else None
                ),
                "gpu_cold_speedup_vs_cpu": (cpu_elapsed / cold_gpu_elapsed) if cold_gpu_elapsed else None,
                "gpu_warm_speedup_vs_cpu": (cpu_elapsed / warm_gpu_elapsed) if warm_gpu_elapsed else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
