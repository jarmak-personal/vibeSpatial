from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import shapely
from shapely.affinity import translate

from vibespatial import ExecutionMode, clip_by_rect_owned, from_shapely_geometries
from vibespatial.point_constructive import clip_points_rect_owned, point_buffer_owned_array
from vibespatial.testing import SyntheticSpec, generate_points


def build_point_inputs(scale: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
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
    half = scale // 2
    if half < scale:
        shifted = [translate(geometry, xoff=10_000.0, yoff=10_000.0) for geometry in points[half:]]
        points[half:] = np.asarray(shifted, dtype=object)
    in_bounds = points[:half] if half else points
    xs = np.asarray([geom.x for geom in in_bounds], dtype=np.float64)
    ys = np.asarray([geom.y for geom in in_bounds], dtype=np.float64)
    rect = (
        float(xs.min() - 0.25),
        float(ys.min() - 0.25),
        float(xs.max() + 0.25),
        float(ys.max() + 0.25),
    )
    return points, rect


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU constructive kernels on point-centric workloads."
    )
    parser.add_argument("--scale", type=int, default=100_000)
    args = parser.parse_args()

    points, rect = build_point_inputs(args.scale)
    owned = from_shapely_geometries(points.tolist())

    started = perf_counter()
    shapely_clip = shapely.clip_by_rect(points, *rect)
    shapely_clip_elapsed = perf_counter() - started

    started = perf_counter()
    clip_points_rect_owned(
        owned,
        *rect,
        dispatch_mode=ExecutionMode.GPU,
    )
    gpu_clip_kernel_cold_elapsed = perf_counter() - started

    started = perf_counter()
    clip_points_rect_owned(
        owned,
        *rect,
        dispatch_mode=ExecutionMode.GPU,
    )
    gpu_clip_kernel_warm_elapsed = perf_counter() - started

    started = perf_counter()
    gpu_clip_public = clip_by_rect_owned(
        owned,
        *rect,
        dispatch_mode=ExecutionMode.GPU,
    )
    gpu_clip_public_elapsed = perf_counter() - started

    clip_true_rows = int(np.count_nonzero(np.asarray([geom is not None and not geom.is_empty for geom in shapely_clip], dtype=bool)))

    clipped_points = from_shapely_geometries(
        [geom for geom in shapely_clip.tolist() if geom is not None and not geom.is_empty]
    )
    if clipped_points.row_count == 0:
        warm_buffer_elapsed = 0.0
        cold_buffer_elapsed = 0.0
        shapely_buffer_elapsed = 0.0
        buffer_rows = 0
    else:
        started = perf_counter()
        shapely_buffer = shapely.buffer(np.asarray(clipped_points.to_shapely(), dtype=object), 1.0, quad_segs=1)
        shapely_buffer_elapsed = perf_counter() - started

        started = perf_counter()
        point_buffer_owned_array(
            clipped_points,
            1.0,
            quad_segs=1,
            dispatch_mode=ExecutionMode.GPU,
        )
        cold_buffer_elapsed = perf_counter() - started

        started = perf_counter()
        gpu_buffer = point_buffer_owned_array(
            clipped_points,
            1.0,
            quad_segs=1,
            dispatch_mode=ExecutionMode.GPU,
        )
        warm_buffer_elapsed = perf_counter() - started
        buffer_rows = int(gpu_buffer.row_count)
        del gpu_buffer
        del shapely_buffer

    print(
        json.dumps(
            {
                "scale": args.scale,
                "clip": {
                    "candidate_rows": int(gpu_clip_public.candidate_rows.size),
                    "non_empty_rows": clip_true_rows,
                    "rect": rect,
                    "shapely_elapsed_seconds": shapely_clip_elapsed,
                    "gpu_kernel_cold_elapsed_seconds": gpu_clip_kernel_cold_elapsed,
                    "gpu_kernel_warm_elapsed_seconds": gpu_clip_kernel_warm_elapsed,
                    "gpu_public_elapsed_seconds": gpu_clip_public_elapsed,
                    "gpu_public_speedup_vs_shapely": (
                        shapely_clip_elapsed / gpu_clip_public_elapsed if gpu_clip_public_elapsed else None
                    ),
                    "gpu_warm_speedup_vs_shapely": (
                        shapely_clip_elapsed / gpu_clip_kernel_warm_elapsed if gpu_clip_kernel_warm_elapsed else None
                    ),
                },
                "buffer": {
                    "rows": buffer_rows,
                    "shapely_elapsed_seconds": shapely_buffer_elapsed,
                    "gpu_cold_elapsed_seconds": cold_buffer_elapsed,
                    "gpu_warm_elapsed_seconds": warm_buffer_elapsed,
                    "gpu_warm_speedup_vs_shapely": (
                        shapely_buffer_elapsed / warm_buffer_elapsed if warm_buffer_elapsed else None
                    ),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
