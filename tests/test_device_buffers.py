from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from vibespatial import Residency, TransferTrigger, from_shapely_geometries, has_gpu_runtime
from vibespatial.cuda._runtime import get_cuda_runtime


def _sample_geometries():
    return [
        Point(1, 2),
        Point(),
        LineString([(0, 0), (2, 4)]),
        Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
        None,
    ]


def test_owned_geometry_can_allocate_device_buffers() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(_sample_geometries())
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="device allocation test",
    )

    assert owned.residency is Residency.DEVICE
    assert owned.device_state is not None
    assert owned.diagnostics_report()["device_buffers_allocated"] is True


def test_owned_geometry_round_trips_through_device_residency() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(_sample_geometries())
    baseline = owned.to_shapely()

    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="device round-trip test",
    )
    restored = owned.to_shapely()

    for left, right in zip(restored, baseline, strict=True):
        if left is None or right is None:
            assert left is right
            continue
        assert left.equals(right)


def test_device_bounds_buffers_preserve_cached_values() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(_sample_geometries(), residency=Residency.DEVICE)
    point_family = next(family for family in owned.families if family.value == "point")
    cached = np.asarray([[1.0, 2.0, 1.0, 2.0], [np.nan, np.nan, np.nan, np.nan]], dtype=np.float64)
    owned.cache_bounds(
        np.asarray(
            [
                [1.0, 2.0, 1.0, 2.0],
                [np.nan, np.nan, np.nan, np.nan],
                [0.0, 0.0, 2.0, 4.0],
                [0.0, 0.0, 3.0, 3.0],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        )
    )

    state = owned._ensure_device_state()
    assert state.families[point_family].bounds is not None
    assert np.allclose(
        get_cuda_runtime().copy_device_to_host(state.families[point_family].bounds),
        cached,
        equal_nan=True,
    )
