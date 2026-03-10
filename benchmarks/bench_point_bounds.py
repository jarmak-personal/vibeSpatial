from __future__ import annotations

import time

from vibespatial import ExecutionMode
from vibespatial.testing import SyntheticSpec, generate_points, generate_polygons
from vibespatial.kernels.predicates.point_bounds import point_bounds


TIER = 4
REFERENCE_SCALE = "100K"


def run_benchmark() -> dict[str, object]:
    points = list(
        generate_points(
            SyntheticSpec(geometry_type="point", distribution="uniform", count=REFERENCE_SCALE, seed=0)
        ).geometries
    )
    polygons = list(
        generate_polygons(
            SyntheticSpec(geometry_type="polygon", distribution="regular-grid", count="10K", seed=0)
        ).geometries
    )
    tiled_polygons = [polygons[index % len(polygons)] for index in range(len(points))]

    started = time.perf_counter()
    try:
        point_bounds(points, tiled_polygons, dispatch_mode=ExecutionMode.CPU)
    except NotImplementedError:
        elapsed = None
    else:
        elapsed = time.perf_counter() - started

    return {
        "kernel": "point_bounds",
        "tier": TIER,
        "scale": REFERENCE_SCALE,
        "requested_runtime": ExecutionMode.CPU.value,
        "selected_runtime": ExecutionMode.CPU.value,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    print(run_benchmark())
