"""Tests for the GPU polygon-vs-rectangle intersection kernel."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon, box

from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.residency import Residency, TransferTrigger
from vibespatial.testing import build_owned as _make_owned_polygons

try:
    from vibespatial.cuda._runtime import has_cuda_device

    _has_gpu = has_cuda_device()
except (ImportError, ModuleNotFoundError):
    _has_gpu = False

requires_gpu = pytest.mark.skipif(not _has_gpu, reason="GPU not available")


def test_polygon_rect_intersection_can_handle_uses_host_probe_without_move_to(
    monkeypatch: pytest.MonkeyPatch,
):
    left = _make_owned_polygons([box(0.0, 0.0, 2.0, 2.0)])
    right = _make_owned_polygons(
        [
            Polygon(
                [
                    (0.0, 0.0),
                    (2.0, 0.0),
                    (3.0, 1.0),
                    (1.0, 3.0),
                    (0.0, 0.0),
                ]
            )
        ]
    )

    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection_can_handle,
    )

    def _fail_move_to(self, *args, **kwargs):
        raise AssertionError("capability probe should not move operands to the device")

    monkeypatch.setattr(OwnedGeometryArray, "move_to", _fail_move_to)

    assert polygon_rect_intersection_can_handle(left, right) is False


@requires_gpu
def test_polygon_rect_intersection_can_handle_device_take_rows():
    left = _make_owned_polygons(
        [
            Polygon(
                [
                    (0.0, 0.0),
                    (5.0, 12.0),
                    (10.0, 0.0),
                    (0.0, 0.0),
                ]
            ),
            Polygon(
                [
                    (20.0, 0.0),
                    (25.0, 12.0),
                    (30.0, 0.0),
                    (20.0, 0.0),
                ]
            ),
        ]
    )
    right = _make_owned_polygons(
        [
            box(0.0, 0.0, 10.0, 10.0),
            box(20.0, 0.0, 30.0, 10.0),
        ]
    )
    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test device-take capability probe",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test device-take capability probe",
    )

    import cupy as cp

    left_subset = left.device_take(cp.asarray([0, 1], dtype=cp.int64))
    right_subset = right.device_take(cp.asarray([0, 1], dtype=cp.int64))

    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection_can_handle,
    )

    assert polygon_rect_intersection_can_handle(left_subset, right_subset) is True


def _assert_geom_equal(gpu_geom, ref_geom, *, rtol=1e-6, msg=""):
    if ref_geom is None or (hasattr(ref_geom, "is_empty") and ref_geom.is_empty):
        if gpu_geom is not None and hasattr(gpu_geom, "is_empty"):
            assert gpu_geom.is_empty, f"Expected empty/None but got {gpu_geom}. {msg}"
        return

    if gpu_geom is None:
        pytest.fail(f"GPU returned None but expected {ref_geom}. {msg}")

    gpu_area = shapely.area(gpu_geom)
    ref_area = shapely.area(ref_geom)
    if ref_area < 1e-12:
        return
    area_ratio = abs(gpu_area - ref_area) / max(abs(ref_area), 1e-15)
    assert area_ratio < rtol, (
        f"Area mismatch: GPU={gpu_area}, ref={ref_area}, ratio={area_ratio}. {msg}"
    )
    sym_diff = shapely.area(shapely.symmetric_difference(gpu_geom, ref_geom))
    sym_ratio = sym_diff / max(abs(ref_area), 1e-15)
    assert sym_ratio < rtol, (
        f"Symmetric difference too large: {sym_diff} (ratio={sym_ratio}). {msg}"
    )


@requires_gpu
def test_polygon_rect_intersection_handles_buffered_left_polygons():
    left_geoms = [
        Point(0, 0).buffer(5.0),
        Point(20, 0).buffer(5.0),
    ]
    right_geoms = [
        box(-2.0, -2.0, 2.0, 2.0),
        box(18.0, -3.0, 24.0, 3.0),
    ]
    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
    )

    result = polygon_rect_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    expected = shapely.intersection(
        np.asarray(left_geoms, dtype=object),
        np.asarray(right_geoms, dtype=object),
    )

    assert len(result_geoms) == 2
    for i, (got, exp) in enumerate(zip(result_geoms, expected.tolist(), strict=True)):
        _assert_geom_equal(got, exp, msg=f"pair {i}")


@requires_gpu
def test_polygon_rect_intersection_many_pairs():
    rng = np.random.default_rng(42)
    left_geoms = []
    right_geoms = []
    for _ in range(128):
        x, y = rng.uniform(-100.0, 100.0, 2)
        left_geoms.append(Point(x, y).buffer(rng.uniform(2.0, 8.0)))
        xmin = x + rng.uniform(-4.0, 0.0)
        ymin = y + rng.uniform(-4.0, 0.0)
        xmax = xmin + rng.uniform(1.0, 6.0)
        ymax = ymin + rng.uniform(1.0, 6.0)
        right_geoms.append(box(xmin, ymin, xmax, ymax))

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
    )

    result = polygon_rect_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    expected = shapely.intersection(
        np.asarray(left_geoms, dtype=object),
        np.asarray(right_geoms, dtype=object),
    )

    mismatches = 0
    for got, exp in zip(result_geoms, expected.tolist(), strict=True):
        exp_area = shapely.area(exp) if exp is not None else 0.0
        got_area = shapely.area(got) if got is not None else 0.0
        if exp_area < 1e-10 and got_area < 1e-10:
            continue
        if exp_area < 1e-10:
            mismatches += 1
            continue
        ratio = abs(got_area - exp_area) / max(abs(exp_area), 1e-15)
        if ratio > 1e-4:
            mismatches += 1
    assert mismatches == 0


@requires_gpu
def test_polygon_rect_intersection_result_is_device_resident(strict_device_guard):
    left = _make_owned_polygons([Point(0, 0).buffer(3.0)])
    right = _make_owned_polygons([box(-1.0, -1.0, 1.0, 1.0)])

    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
    )

    result = polygon_rect_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    assert result.residency is Residency.DEVICE
    assert result.device_state is not None
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None


@requires_gpu
def test_polygon_rect_intersection_marks_touch_only_rows_empty():
    left = _make_owned_polygons([box(0.0, 0.0, 2.0, 2.0)])
    right = _make_owned_polygons([box(2.0, 0.0, 4.0, 2.0)])

    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
    )

    result = polygon_rect_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()

    assert len(result_geoms) == 1
    assert result_geoms[0] is None or result_geoms[0].is_empty


@requires_gpu
def test_polygon_rect_intersection_emits_boundary_overlap_flag():
    left = _make_owned_polygons(
        [
            Point(0.0, 0.0).buffer(3.0),
            box(0.0, 0.0, 2.0, 2.0),
        ]
    )
    right = _make_owned_polygons(
        [
            box(-1.0, -1.0, 1.0, 1.0),
            box(0.0, 0.0, 1.0, 1.0),
        ]
    )

    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
    )

    result = polygon_rect_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    overlap = getattr(result, "_polygon_rect_boundary_overlap", None)

    assert overlap is not None
    if hasattr(overlap, "get"):
        overlap = overlap.get()
    overlap = np.asarray(overlap, dtype=bool)
    assert overlap.tolist() == [False, True]
