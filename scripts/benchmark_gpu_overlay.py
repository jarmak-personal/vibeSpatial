from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import shapely
from shapely.geometry import box

from vibespatial import ExecutionMode, from_shapely_geometries, overlay_intersection_owned


def _build_rectangle_pairs(pair_count: int) -> tuple[np.ndarray, np.ndarray]:
    grid_width = max(1, int(np.ceil(np.sqrt(pair_count))))
    left = []
    right = []
    for index in range(pair_count):
        row = index // grid_width
        col = index % grid_width
        base_x = float(col * 10.0)
        base_y = float(row * 10.0)
        left.append(box(base_x, base_y, base_x + 4.0, base_y + 4.0))
        right.append(box(base_x + 2.0, base_y + 1.0, base_x + 5.0, base_y + 5.0))
    return np.asarray(left, dtype=object), np.asarray(right, dtype=object)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GPU polygon overlay intersection.")
    parser.add_argument("--operation", choices=("intersection",), default="intersection")
    parser.add_argument("--pairs", type=int, default=10_000)
    args = parser.parse_args()

    left_values, right_values = _build_rectangle_pairs(args.pairs)
    left_owned = from_shapely_geometries(left_values.tolist())
    right_owned = from_shapely_geometries(right_values.tolist())

    started = perf_counter()
    expected = shapely.intersection(left_values, right_values)
    shapely_elapsed = perf_counter() - started

    started = perf_counter()
    gpu_cold = overlay_intersection_owned(left_owned, right_owned, dispatch_mode=ExecutionMode.GPU)
    gpu_cold_elapsed = perf_counter() - started

    started = perf_counter()
    gpu_warm = overlay_intersection_owned(left_owned, right_owned, dispatch_mode=ExecutionMode.GPU)
    gpu_warm_elapsed = perf_counter() - started

    expected_non_empty = int(np.count_nonzero(np.asarray([value is not None and not value.is_empty for value in expected], dtype=bool)))
    actual_non_empty = int(gpu_warm.row_count)

    print(
        json.dumps(
            {
                "operation": args.operation,
                "pairs": args.pairs,
                "expected_non_empty_rows": expected_non_empty,
                "gpu_non_empty_rows": actual_non_empty,
                "shapely_elapsed_seconds": shapely_elapsed,
                "gpu_cold_elapsed_seconds": gpu_cold_elapsed,
                "gpu_warm_elapsed_seconds": gpu_warm_elapsed,
                "gpu_warm_speedup_vs_shapely": (
                    shapely_elapsed / gpu_warm_elapsed if gpu_warm_elapsed else None
                ),
            },
            indent=2,
        )
    )
    del gpu_cold
    del gpu_warm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
