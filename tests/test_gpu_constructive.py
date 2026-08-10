from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

from vibespatial import ExecutionMode, clip_by_rect_owned, from_shapely_geometries, has_gpu_runtime
from vibespatial.constructive.boundary import boundary_owned
from vibespatial.constructive.exterior import exterior_owned
from vibespatial.constructive.normalize import normalize_owned
from vibespatial.constructive.point import clip_points_rect_owned, point_buffer_owned_array
from vibespatial.constructive.polygon import polygon_buffer_owned_array
from vibespatial.geometry.buffers import GeometryFamily


def test_polygon_boundary_uses_nested_source_capacity_without_allocation_fences() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "constructive" / "boundary.py"
    ).read_text()

    assert "_device_gather_offset_index_ranges(" in source
    assert source.count("allocation_capacity=int(d_x.size)") == 2
    assert "count_scatter_total(" not in source
    assert "boundary coordinate allocation fence" not in source
    assert "boundary multiline ring allocation fence" not in source


def test_segmentize_uses_constructive_precision_and_dispatch_policy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root.joinpath("src/vibespatial/constructive/segmentize.py").read_text()
    kernel_source = repo_root.joinpath(
        "src/vibespatial/constructive/segmentize_kernels.py"
    ).read_text()

    assert source.count("kernel_class=KernelClass.CONSTRUCTIVE") == 2
    assert "kernel_class=KernelClass.COARSE" not in source
    assert "count_scatter_totals(" in source
    assert 'reason="segmentize exact output allocation packet"' in source
    assert 'primary_unit_name="segmentize-input-coordinate"' in source
    assert "coord_capacity = int(device_buf.x.size)" in source
    assert "exclusive_scan_i64" in source
    assert "One lane per physical input coordinate" in kernel_source
    assert "for (int i = start; i < end - 1; i++)" not in kernel_source
    assert "const long long tid" in kernel_source


def _assert_geometries_equal(actual: list[object | None], expected: list[object | None]) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        if left is None or right is None:
            assert left is right
            continue
        assert left.geom_type == right.geom_type
        assert bool(shapely.equals(left, right))


@pytest.mark.gpu
def test_gpu_clip_by_rect_matches_shapely_for_point_only_inputs() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    values = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(3, 3), Point(), None])

    result = clip_by_rect_owned(
        values,
        0.0,
        0.0,
        1.5,
        1.5,
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.clip_by_rect(
        np.asarray([Point(0, 0), Point(1, 1), Point(3, 3), Point(), None], dtype=object),
        0.0,
        0.0,
        1.5,
        1.5,
    )
    _assert_geometries_equal(result.geometries.tolist(), list(expected))
    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.fallback_rows.size == 0


@pytest.mark.gpu
def test_gpu_point_buffer_matches_shapely_for_quad_segs_1() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0), Point(2, 2), Point(-1, 4)])

    gpu = point_buffer_owned_array(points, 1.5, quad_segs=1, dispatch_mode=ExecutionMode.GPU)

    expected = [
        Polygon(((1.5, 0.0), (0.0, -1.5), (-1.5, 0.0), (0.0, 1.5), (1.5, 0.0))),
        Polygon(((3.5, 2.0), (2.0, 0.5), (0.5, 2.0), (2.0, 3.5), (3.5, 2.0))),
        Polygon(((0.5, 4.0), (-1.0, 2.5), (-2.5, 4.0), (-1.0, 5.5), (0.5, 4.0))),
    ]
    _assert_geometries_equal(gpu.to_shapely(), expected)


@pytest.mark.gpu
def test_gpu_clip_by_rect_accepts_device_backed_point_input_without_full_host_materialization() -> (
    None
):
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2), Point(4, 4)])
    device_backed = clip_points_rect_owned(
        points,
        -1.0,
        -1.0,
        5.0,
        5.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert device_backed.families[GeometryFamily.POINT].host_materialized is False

    result = clip_by_rect_owned(
        device_backed,
        0.5,
        0.5,
        3.0,
        3.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.clip_by_rect(
        np.asarray([Point(0, 0), Point(1, 1), Point(2, 2), Point(4, 4)], dtype=object),
        0.5,
        0.5,
        3.0,
        3.0,
    )
    _assert_geometries_equal(result.geometries.tolist(), list(expected))
    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.fallback_rows.size == 0


@pytest.mark.gpu
def test_exterior_gpu_coordinates_stay_device_resident() -> None:
    """Exterior ring extraction keeps coordinates on device."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.runtime.residency import Residency

    polys = from_shapely_geometries(
        [
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            Polygon(
                shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
                holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
            ),
        ]
    )

    result = exterior_owned(polys, dispatch_mode=ExecutionMode.GPU)

    assert result.residency is Residency.DEVICE, "output should be device-resident"
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None
    assert result.device_state is not None
    ls_buf = result.device_state.families[GeometryFamily.LINESTRING]
    assert isinstance(ls_buf.x, cp.ndarray), "x should be CuPy"
    assert isinstance(ls_buf.y, cp.ndarray), "y should be CuPy"

    # Verify correctness: exterior_owned returns LineStrings with the same
    # coordinates as the exterior ring.
    expected_coords = [
        list(Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]).exterior.coords),
        list(
            Polygon(
                shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
                holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
            ).exterior.coords
        ),
    ]
    actual = result.to_shapely()
    if hasattr(actual, "tolist"):
        actual = actual.tolist()
    for a, ec in zip(list(actual), expected_coords, strict=True):
        assert a.geom_type == "LineString"
        assert list(a.coords) == ec


@pytest.mark.gpu
def test_exterior_gpu_moderate_scale_with_nulls() -> None:
    """Exterior ring at moderate scale with interleaved None rows."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    # 100+ polygons with varying ring sizes + None rows
    geoms: list[Polygon | None] = []
    for i in range(120):
        if i % 7 == 0:
            geoms.append(None)
        else:
            s = float(i)
            geoms.append(Polygon([(s, s), (s + 1, s), (s + 1, s + 1), (s, s + 1), (s, s)]))

    polys = from_shapely_geometries(geoms)
    result = exterior_owned(polys, dispatch_mode=ExecutionMode.GPU)

    actual = result.to_shapely()
    if hasattr(actual, "tolist"):
        actual = actual.tolist()
    for i, (a, g) in enumerate(zip(list(actual), geoms, strict=True)):
        if g is None:
            assert a is None, f"row {i} should be None"
        else:
            assert a is not None and a.geom_type == "LineString", f"row {i}"
            assert list(a.coords) == list(g.exterior.coords), f"row {i} coords"


@pytest.mark.gpu
def test_exterior_gpu_all_nonpolygon_rows_stay_device_resident(strict_device_guard) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.residency import Residency

    owned = from_shapely_geometries([Point(0, 0), None], residency=Residency.DEVICE)
    result = exterior_owned(owned, dispatch_mode=ExecutionMode.GPU)

    assert result.residency is Residency.DEVICE
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None


@pytest.mark.gpu
def test_boundary_gpu_metadata_stays_device_resident() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.residency import Residency

    geoms = [
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
        shapely.MultiPolygon(
            [
                Polygon([(10, 0), (14, 0), (14, 4), (10, 4), (10, 0)]),
                Polygon([(20, 0), (24, 0), (24, 4), (20, 4), (20, 0)]),
            ]
        ),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    result = boundary_owned(owned, dispatch_mode=ExecutionMode.GPU)

    assert result.residency is Residency.DEVICE
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None


@pytest.mark.gpu
def test_boundary_gpu_preserves_indexed_polygon_carrier(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.runtime.execution_trace import assert_no_transfers
    from vibespatial.runtime.residency import Residency

    polygons = [
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
        Polygon([(10, 0), (14, 0), (14, 4), (10, 4), (10, 0)]),
    ]
    owned = from_shapely_geometries(polygons, residency=Residency.DEVICE)
    indexed = owned._device_indexed_take(
        cp.asarray([1, 0, 1], dtype=cp.int64),
    )._apply_row_activity(cp.asarray([True, False, True], dtype=cp.bool_))

    def _fail_physicalize(*_args, **_kwargs):
        raise AssertionError("indexed boundary must not physicalize logical rows")

    monkeypatch.setattr(
        OwnedGeometryArray,
        "physicalize_device_rows",
        _fail_physicalize,
    )
    with assert_no_transfers():
        result = boundary_owned(indexed, dispatch_mode=ExecutionMode.GPU)

    assert result.is_indexed_view
    assert result.row_count == indexed.row_count
    assert result._base is not None
    source_polygon = owned.device_state.families[GeometryFamily.POLYGON]
    boundary_line = result._base.device_state.families[GeometryFamily.LINESTRING]
    assert boundary_line.x.data.ptr == source_polygon.x.data.ptr

    monkeypatch.undo()
    actual = result.to_shapely()
    assert actual[0].equals(polygons[1].boundary)
    assert actual[1] is None
    assert actual[2].equals(polygons[1].boundary)


@pytest.mark.gpu
def test_boundary_gpu_mixed_polygon_ring_counts_preserves_row_types() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.execution_trace import assert_no_transfers
    from vibespatial.runtime.residency import Residency

    single_ring = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    with_hole = Polygon(
        [(10, 0), (14, 0), (14, 4), (10, 4), (10, 0)],
        [[(11, 1), (13, 1), (13, 3), (11, 3), (11, 1)]],
    )
    owned = from_shapely_geometries([single_ring, with_hole], residency=Residency.DEVICE)
    state = owned.device_state
    assert state is not None

    from vibespatial.geometry.owned import build_device_resident_owned

    owned = build_device_resident_owned(
        device_families=dict(state.families),
        row_count=owned.row_count,
        tags=state.tags,
        validity=state.validity,
        family_row_offsets=state.family_row_offsets,
        execution_mode="gpu",
    )
    assert not owned.families[GeometryFamily.POLYGON].host_materialized

    with assert_no_transfers():
        result = boundary_owned(owned, dispatch_mode=ExecutionMode.GPU)
    geometries = result.to_shapely()

    assert result.residency is Residency.DEVICE
    assert geometries[0].geom_type == "LineString"
    assert geometries[1].geom_type == "MultiLineString"
    assert bool(shapely.equals(geometries[0], single_ring.boundary))
    assert bool(shapely.equals(geometries[1], with_hole.boundary))


@pytest.mark.gpu
def test_boundary_gpu_empty_lineal_rows_stay_device_resident() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.execution_trace import assert_no_transfers
    from vibespatial.runtime.residency import Residency

    geoms = [
        LineString(),
        LineString([(0, 0), (1, 1)]),
        MultiLineString([]),
        MultiLineString([[(2, 2), (3, 3)]]),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)

    with assert_no_transfers():
        result = boundary_owned(owned, dispatch_mode=ExecutionMode.GPU)

    assert result.residency is Residency.DEVICE
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None
    actual = result.to_shapely()
    _assert_geometries_equal(actual, [geom.boundary for geom in geoms])


@pytest.mark.gpu
def test_normalize_owned_device_resident_polygon_input_uses_device_stats() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    source_poly = Polygon([(0, 0), (3, 0), (3, 2), (0, 2)])
    source = from_shapely_geometries([source_poly])
    device_polys = polygon_buffer_owned_array(source, 0.25, dispatch_mode=ExecutionMode.GPU)

    assert device_polys.device_state is not None
    assert device_polys.families[GeometryFamily.POLYGON].host_materialized is False

    normalized = normalize_owned(device_polys, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.normalize(shapely.buffer(source_poly, 0.25, quad_segs=8))
    actual = shapely.normalize(normalized.to_shapely()[0])

    assert bool(shapely.equals_exact(actual, expected, tolerance=1e-12))


@pytest.mark.gpu
def test_normalize_owned_orders_polygon_hierarchy_like_geos() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.residency import Residency

    left = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8), (0, 0)],
        holes=[
            [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)],
            [(5, 5), (7, 5), (7, 7), (5, 7), (5, 5)],
        ],
    )
    middle = shapely.box(10, 0, 12, 2)
    right = shapely.box(20, 0, 22, 2)
    source = MultiPolygon([left, right, middle])
    owned = from_shapely_geometries([source], residency=Residency.DEVICE)

    actual = normalize_owned(
        owned,
        dispatch_mode=ExecutionMode.GPU,
    ).to_shapely()[0]

    assert shapely.equals_exact(actual, shapely.normalize(source), tolerance=0.0)


@pytest.mark.gpu
def test_normalize_owned_orders_multiple_single_ring_polygons_like_geos() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.residency import Residency

    source = [
        Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]),
        Polygon([(3, 3), (5, 3), (5, 5), (3, 5), (3, 3)]),
    ]
    owned = from_shapely_geometries(source, residency=Residency.DEVICE)

    actual = normalize_owned(
        owned,
        dispatch_mode=ExecutionMode.GPU,
    ).to_shapely()

    assert all(
        shapely.equals_exact(got, shapely.normalize(expected), tolerance=0.0)
        for got, expected in zip(actual, source, strict=True)
    )


@pytest.mark.gpu
def test_normalize_owned_orders_holes_when_empty_rows_balance_ring_count() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.residency import Residency

    polygon = Polygon(
        [(0, 0), (9, 0), (9, 9), (0, 9), (0, 0)],
        holes=[
            [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)],
            [(5, 5), (8, 5), (8, 8), (5, 8), (5, 5)],
        ],
    )
    source = [Polygon(), polygon, Polygon()]
    owned = from_shapely_geometries(source, residency=Residency.DEVICE)

    actual = normalize_owned(owned, dispatch_mode=ExecutionMode.GPU).to_shapely()

    assert all(
        shapely.equals_exact(got, shapely.normalize(expected), tolerance=0.0)
        for got, expected in zip(actual, source, strict=True)
    )


@pytest.mark.gpu
def test_coordinate_stats_ignore_uninitialized_device_capacity_lanes() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.constructive.measurement import _coord_stats_from_owned
    from vibespatial.runtime.residency import Residency

    source = [
        Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]),
        Polygon([(3, 3), (5, 3), (5, 5), (3, 5), (3, 3)]),
    ]
    owned = from_shapely_geometries(source, residency=Residency.DEVICE)
    device_buffer = owned.device_state.families[GeometryFamily.POLYGON]
    active_coordinate_count = int(device_buffer.x.size)
    device_buffer.x = cp.concatenate(
        (device_buffer.x, cp.full(32, cp.inf, dtype=cp.float64))
    )
    device_buffer.y = cp.concatenate(
        (device_buffer.y, cp.full(32, -cp.inf, dtype=cp.float64))
    )
    device_buffer.ring_offsets = cp.concatenate(
        (
            device_buffer.ring_offsets,
            cp.full(8, active_coordinate_count, dtype=cp.int32),
        )
    )

    assert _coord_stats_from_owned(owned) == (5.0, 1.0, 5.0)
    normalized = normalize_owned(owned, dispatch_mode=ExecutionMode.GPU).to_shapely()
    assert all(
        shapely.equals_exact(got, shapely.normalize(expected), tolerance=0.0)
        for got, expected in zip(normalized, source, strict=True)
    )


@pytest.mark.gpu
def test_normalize_owned_orders_mixed_points_and_polygons_like_geos() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.residency import Residency

    source = [
        Point(2, 2),
        Point(3, 3),
        Point(3, 3),
        Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]),
        Polygon([(3, 3), (5, 3), (5, 5), (3, 5), (3, 3)]),
    ]
    owned = from_shapely_geometries(source, residency=Residency.DEVICE)

    actual = normalize_owned(
        owned,
        dispatch_mode=ExecutionMode.GPU,
    ).to_shapely()

    assert all(
        shapely.equals_exact(got, shapely.normalize(expected), tolerance=0.0)
        for got, expected in zip(actual, source, strict=True)
    )


@pytest.mark.gpu
def test_normalize_defers_async_gather_sources_until_terminal_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.runtime.residency import Residency

    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8), (0, 0)],
        holes=[
            [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)],
            [(5, 5), (7, 5), (7, 7), (5, 7), (5, 5)],
        ],
    )
    owned = from_shapely_geometries([source], residency=Residency.DEVICE)
    runtime = get_cuda_runtime()
    original_copy = runtime.copy_device_to_host
    original_free = runtime.free
    events: list[str] = []

    def tracked_copy(*args, **kwargs):
        events.append("copy")
        return original_copy(*args, **kwargs)

    def tracked_free(*args, **kwargs):
        events.append("free")
        return original_free(*args, **kwargs)

    monkeypatch.setattr(runtime, "copy_device_to_host", tracked_copy)
    monkeypatch.setattr(runtime, "free", tracked_free)

    actual = normalize_owned(
        owned,
        dispatch_mode=ExecutionMode.GPU,
    ).to_shapely()[0]

    assert events.count("copy") >= 4
    assert events.index("free") > max(
        position for position, event in enumerate(events) if event == "copy"
    )
    assert shapely.equals_exact(actual, shapely.normalize(source), tolerance=0.0)


@pytest.mark.gpu
def test_normalize_owned_orders_lineal_and_point_multi_hierarchies_like_geos() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.residency import Residency

    multiline = MultiLineString(
        [
            [(2, 0), (2, 1)],
            [(5, 4), (6, 4)],
            [(3, 4), (2, 4), (2, 3)],
        ]
    )
    multipoint = MultiPoint([(2, 0), (5, 4), (2, 3)])
    owned = from_shapely_geometries(
        [multiline, multipoint],
        residency=Residency.DEVICE,
    )

    actual = normalize_owned(
        owned,
        dispatch_mode=ExecutionMode.GPU,
    ).to_shapely()

    assert shapely.equals_exact(actual[0], shapely.normalize(multiline), tolerance=0.0)
    assert shapely.equals_exact(actual[1], shapely.normalize(multipoint), tolerance=0.0)
