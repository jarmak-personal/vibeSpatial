from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from vibespatial import (
    ExecutionMode,
    Residency,
    compute_geometry_bounds,
    from_shapely_geometries,
    has_gpu_runtime,
)


def _sample_geometries():
    return [
        Point(1, 2),
        None,
        Point(),
        LineString([(0, 0), (2, 4)]),
        Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
        MultiPolygon(
            [
                Polygon([(10, 10), (12, 10), (12, 12), (10, 10)]),
                Polygon([(20, 20), (21, 20), (21, 21), (20, 20)]),
            ]
        ),
    ]


def test_gpu_bounds_matches_cpu_reference() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(_sample_geometries())

    cpu_bounds = compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.CPU)
    gpu_bounds = compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.GPU)

    assert np.allclose(cpu_bounds, gpu_bounds, equal_nan=True)


def test_gpu_bounds_runs_from_device_resident_buffers() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(_sample_geometries(), residency=Residency.DEVICE)
    gpu_bounds = compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.GPU)

    assert owned.residency is Residency.DEVICE
    assert owned.device_state is not None
    assert np.allclose(gpu_bounds[0], np.asarray([1.0, 2.0, 1.0, 2.0]), equal_nan=True)
    assert np.isnan(gpu_bounds[1]).all()
    assert np.isnan(gpu_bounds[2]).all()
