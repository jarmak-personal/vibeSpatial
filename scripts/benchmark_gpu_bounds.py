from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
from shapely.geometry import Point

from vibespatial import ExecutionMode, Residency, compute_geometry_bounds, from_shapely_geometries, has_gpu_runtime


def _sample_points(rows: int) -> list[Point]:
    return [Point(float(index), float((index * 7) % 1000)) for index in range(rows)]


def _time_best(fn, arg, *, repeat: int) -> tuple[float, np.ndarray]:
    best = float("inf")
    last = np.empty((0, 4), dtype=np.float64)
    for _ in range(repeat):
        start = perf_counter()
        last = fn(arg)
        best = min(best, perf_counter() - start)
    return best, last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    host_owned = from_shapely_geometries(_sample_points(args.scale))
    cpu_seconds, cpu_bounds = _time_best(
        lambda values: compute_geometry_bounds(values, dispatch_mode=ExecutionMode.CPU),
        host_owned,
        repeat=args.repeat,
    )

    payload = {
        "rows": args.scale,
        "repeat": args.repeat,
        "cpu_best_seconds": cpu_seconds,
        "cpu_rows_per_second": args.scale / cpu_seconds,
        "checksum": float(np.nan_to_num(cpu_bounds, nan=0.0).sum()),
    }

    if has_gpu_runtime():
        host_gpu_seconds, gpu_bounds = _time_best(
            lambda values: compute_geometry_bounds(values, dispatch_mode=ExecutionMode.GPU),
            host_owned,
            repeat=args.repeat,
        )
        device_owned = from_shapely_geometries(_sample_points(args.scale), residency=Residency.DEVICE)
        device_gpu_seconds, device_gpu_bounds = _time_best(
            lambda values: compute_geometry_bounds(values, dispatch_mode=ExecutionMode.GPU),
            device_owned,
            repeat=args.repeat,
        )
        payload.update(
            {
                "gpu_host_best_seconds": host_gpu_seconds,
                "gpu_host_rows_per_second": args.scale / host_gpu_seconds,
                "gpu_device_best_seconds": device_gpu_seconds,
                "gpu_device_rows_per_second": args.scale / device_gpu_seconds,
                "gpu_matches_cpu": bool(np.allclose(cpu_bounds, gpu_bounds, equal_nan=True)),
                "gpu_device_matches_cpu": bool(
                    np.allclose(cpu_bounds, device_gpu_bounds, equal_nan=True)
                ),
            }
        )
    else:
        payload["gpu_unavailable"] = True

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
