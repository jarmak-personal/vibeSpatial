"""Tests for the GPU polygon-vs-rectangle intersection kernel."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon, box

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.residency import Residency, TransferTrigger
from vibespatial.testing import build_owned as _make_owned_polygons

try:
    from vibespatial.cuda._runtime import has_cuda_device

    _has_gpu = has_cuda_device()
except (ImportError, ModuleNotFoundError):
    _has_gpu = False

requires_gpu = pytest.mark.skipif(not _has_gpu, reason="GPU not available")


def test_polygon_rect_intersection_has_no_raw_cupy_scalar_syncs() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "kernels"
        / "constructive"
        / "polygon_rect_intersection.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    failures: list[str] = []
    cupy_reductions = {
        "all",
        "any",
        "sum",
        "count_nonzero",
        "max",
        "min",
        "nanmax",
        "nanmin",
    }

    def _contains_cupy_reduction(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "cp"
            and child.func.attr in cupy_reductions
            for child in ast.walk(node)
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "item":
                failures.append(f"raw .item() at line {node.lineno}")
            continue
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"bool", "int", "float"}
            and node.args
        ):
            continue
        if _contains_cupy_reduction(node.args[0]):
            failures.append(f"raw {node.func.id}(cp reduction) at line {node.lineno}")

    assert failures == []


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


@requires_gpu
def test_polygon_rect_bounds_clip_ignores_inactive_wide_physical_polygon():
    import cupy as cp

    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection_from_bounds,
    )

    source = _make_owned_polygons([Point(0.0, 0.0).buffer(5.0, quad_segs=96)])
    source.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test inactive wide row-indirected rectangle source",
    )
    polygon = source.device_state.families[GeometryFamily.POLYGON]
    assert int(polygon.ring_offsets[-1] - polygon.ring_offsets[0]) > 257

    result = polygon_rect_intersection_from_bounds(
        source,
        cp.full((1, 4), cp.nan, dtype=cp.float64),
    )

    assert result.residency is Residency.DEVICE
    actual = result.to_shapely()[0]
    assert actual is None or actual.is_empty


@requires_gpu
def test_polygon_rect_bounds_clip_preserves_coincident_boundary_vertices():
    import cupy as cp

    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection_from_bounds,
    )

    source = _make_owned_polygons(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(10.0, 0.0, 12.0, 2.0),
        ]
    )
    source.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test coincident row-indirected rectangle bounds",
    )
    result = polygon_rect_intersection_from_bounds(
        source,
        cp.asarray(
            [
                [0.0, 0.0, 1.0, 1.0],
                [10.0, 0.0, 11.0, 1.0],
            ],
            dtype=cp.float64,
        ),
    )

    expected = np.asarray(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(10.0, 0.0, 11.0, 1.0),
        ],
        dtype=object,
    )
    actual = np.asarray(result.to_shapely(), dtype=object)
    assert shapely.equals_exact(
        shapely.normalize(actual),
        shapely.normalize(expected),
        tolerance=0.0,
    ).all()


@requires_gpu
def test_polygon_box_bounds_accept_device_row_indirected_polygon_rows_from_mixed_base():
    import cupy as cp

    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.spatial.query_box import _extract_owned_polygon_box_bounds

    mixed = from_shapely_geometries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            LineString([(100.0, 0.0), (101.0, 1.0)]),
            LineString([(110.0, 0.0), (111.0, 1.0), (112.0, 1.0)]),
            box(5.0, 1.0, 8.0, 4.0),
        ],
        residency=Residency.DEVICE,
    )
    row_indices = np.tile(np.asarray([0, 3], dtype=np.int64), 600)
    rowset = mixed.device_take(cp.asarray(row_indices, dtype=cp.int64))

    assert rowset.is_indexed_view
    assert set(rowset.families) == {GeometryFamily.POLYGON, GeometryFamily.LINESTRING}

    d_bounds = _extract_owned_polygon_box_bounds(rowset, return_device=True)

    assert d_bounds is not None
    np.testing.assert_allclose(
        cp.asnumpy(d_bounds),
        np.tile(
            np.asarray(
                [
                    [0.0, 0.0, 2.0, 2.0],
                    [5.0, 1.0, 8.0, 4.0],
                ],
                dtype=np.float64,
            ),
            (600, 1),
        ),
    )


@requires_gpu
def test_rectangle_intersection_uses_shape_metadata_without_scalar_fences():
    import cupy as cp

    from vibespatial.constructive.binary_constructive import binary_constructive_owned
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left = _make_owned_polygons(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(3.0, 0.0, 5.0, 2.0),
            box(10.0, 0.0, 11.0, 1.0),
        ]
    )
    right = _make_owned_polygons(
        [
            box(1.0, 1.0, 3.0, 3.0),
            box(4.0, 1.0, 6.0, 3.0),
            box(20.0, 0.0, 21.0, 1.0),
        ]
    )
    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test rectangle shape metadata",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test rectangle shape metadata",
    )
    left_subset = left.device_take(cp.asarray([0, 1, 2], dtype=cp.int64))
    right_subset = right.device_take(cp.asarray([0, 1, 2], dtype=cp.int64))

    reset_d2h_transfer_count()
    result = binary_constructive_owned(
        "intersection",
        left_subset,
        right_subset,
        dispatch_mode=ExecutionMode.GPU,
    )
    transfers = get_d2h_transfer_events(clear=True)
    reasons = {event.reason for event in transfers}

    assert result.residency is Residency.DEVICE
    assert result.row_count == 3
    assert result.device_state is not None
    polygon_buffer = result.device_state.families[GeometryFamily.POLYGON]
    assert polygon_buffer.dense_single_ring_width == 5
    assert polygon_buffer.axis_aligned_rectangles is True
    assert "polygon-rectangle dense single-ring scalar fence" not in reasons
    assert "polygon-rectangle empty-mask scalar fence" not in reasons
    assert "polygon-rectangle ring-offset scalar fence" not in reasons
    assert "polygon-rectangle x-closure scalar fence" not in reasons
    assert "polygon-rectangle y-closure scalar fence" not in reasons
    assert "polygon-rectangle axis-aligned scalar fence" not in reasons
    assert "polygon-rectangle intersection vertex allocation fence" not in reasons
    reset_d2h_transfer_count()


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
def test_polygon_rect_intersection_uses_cached_bounds_matrix_as_contiguous_columns():
    import cupy as cp

    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
    )

    center = (489.9590630158275, 359.7573581617746)
    rectangle_ids = [3247, 3248, 3249, 3250, 3346, 3347, 3348, 3349, 3350, 3351]
    right_geoms = [
        box(
            (idx % 100) * 10.0,
            (idx // 100) * 10.0,
            (idx % 100 + 1) * 10.0,
            (idx // 100 + 1) * 10.0,
        )
        for idx in rectangle_ids
    ]
    left_source = _make_owned_polygons([Point(*center).buffer(35.0)])
    right = _make_owned_polygons(right_geoms)
    left_source.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test cached rectangle bounds matrix",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test cached rectangle bounds matrix",
    )
    left = left_source.device_take(cp.zeros(len(right_geoms), dtype=cp.int64))
    right_buffer = right.device_state.families[GeometryFamily.POLYGON]
    right_buffer.bounds = cp.asarray(
        [[*geom.bounds] for geom in right_geoms],
        dtype=cp.float64,
    )
    right_buffer.axis_aligned_rectangles = True

    result = polygon_rect_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    expected = shapely.intersection(
        np.asarray([Point(*center).buffer(35.0)] * len(right_geoms), dtype=object),
        np.asarray(right_geoms, dtype=object),
    )

    for i, (got, exp) in enumerate(zip(result_geoms, expected.tolist(), strict=True)):
        _assert_geom_equal(got, exp, msg=f"cached bounds pair {i}")


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
def test_rectangle_clipped_difference_uses_native_clip_before_difference():
    from vibespatial.constructive.binary_constructive import (
        _row_aligned_rectangle_clipped_difference_gpu,
    )
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [
        box(0.0, 0.0, 10.0, 10.0),
        box(12.0, 0.0, 22.0, 10.0),
        box(0.0, 12.0, 10.0, 22.0),
    ]
    right_geoms = [
        Point(8.0, 5.0).buffer(5.0, quad_segs=8),
        Point(17.0, 5.0).buffer(20.0, quad_segs=8),
        Point(30.0, 30.0).buffer(3.0, quad_segs=8),
    ]
    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)
    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test rectangle clipped difference left",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test rectangle clipped difference right",
    )

    clear_dispatch_events()
    result = _row_aligned_rectangle_clipped_difference_gpu(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    events = get_dispatch_events(clear=True)

    assert result is not None
    assert result.residency is Residency.DEVICE
    assert result.row_count == len(left_geoms)
    assert any(
        event.implementation == "row_aligned_rectangle_clipped_difference_gpu" for event in events
    )
    result_geoms = result.to_shapely()
    expected = shapely.difference(
        np.asarray(left_geoms, dtype=object),
        np.asarray(right_geoms, dtype=object),
    )
    for i, (got, exp) in enumerate(zip(result_geoms, expected.tolist(), strict=True)):
        _assert_geom_equal(got, exp, msg=f"rectangle clipped difference row {i}")


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
