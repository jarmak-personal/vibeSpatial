"""Tests for device-side take (GPU gather compaction).

Verifies that ``OwnedGeometryArray.device_take`` produces results identical
to the host ``take``, stays DEVICE-resident, and auto-dispatches correctly.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from vibespatial import (
    Residency,
    TransferTrigger,
    from_shapely_geometries,
    has_gpu_runtime,
)

pytestmark = pytest.mark.gpu


def _require_gpu():
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_points() -> list[object]:
    return [Point(i, i * 2) for i in range(10)]


def _sample_mixed() -> list[object | None]:
    return [
        Point(1, 2),
        None,
        Point(),
        LineString([(0, 0), (2, 4)]),
        Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
        MultiPolygon([
            Polygon([(10, 10), (12, 10), (12, 12), (10, 10)]),
            Polygon([(20, 20), (21, 20), (21, 21), (20, 20)]),
        ]),
    ]


def _all_families() -> list[object]:
    return [
        Point(1, 2),
        LineString([(0, 0), (1, 1), (2, 0)]),
        Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
        MultiPoint([(0, 0), (1, 1)]),
        MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
        MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
        ]),
    ]


def _polygon_with_hole():
    return Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [[(2, 2), (4, 2), (4, 4), (2, 2)]],
    )


def _assert_geometries_equal(left: list, right: list) -> None:
    assert len(left) == len(right), f"length mismatch: {len(left)} vs {len(right)}"
    for i, (lg, rg) in enumerate(zip(left, right, strict=True)):
        if lg is None or rg is None:
            assert lg is rg, f"index {i}: null mismatch"
            continue
        assert lg.equals(rg), f"index {i}: {lg.wkt} != {rg.wkt}"


# ---------------------------------------------------------------------------
# Core correctness: device_take matches host take
# ---------------------------------------------------------------------------


class TestDeviceTakeCorrectness:
    """Device take must produce identical geometries to host take."""

    def test_points_integer_indices(self):
        _require_gpu()
        geoms = _sample_points()
        owned = from_shapely_geometries(geoms)
        host_subset = owned.take(np.array([0, 3, 7, 9]))

        owned_gpu = from_shapely_geometries(geoms)
        owned_gpu.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)
        device_subset = owned_gpu.take(np.array([0, 3, 7, 9]))

        _assert_geometries_equal(device_subset.to_shapely(), host_subset.to_shapely())

    def test_points_boolean_mask(self):
        _require_gpu()
        geoms = _sample_points()
        mask = np.array([True, False, True, False, True, False, True, False, True, False])

        owned = from_shapely_geometries(geoms)
        host_subset = owned.take(mask)

        owned_gpu = from_shapely_geometries(geoms)
        owned_gpu.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)
        device_subset = owned_gpu.take(mask)

        _assert_geometries_equal(device_subset.to_shapely(), host_subset.to_shapely())

    def test_mixed_geometries(self):
        _require_gpu()
        geoms = _sample_mixed()
        indices = np.array([0, 3, 4, 5])

        owned = from_shapely_geometries(geoms)
        host_subset = owned.take(indices)

        owned_gpu = from_shapely_geometries(geoms)
        owned_gpu.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)
        device_subset = owned_gpu.take(indices)

        _assert_geometries_equal(device_subset.to_shapely(), host_subset.to_shapely())

    def test_null_and_empty(self):
        _require_gpu()
        geoms = _sample_mixed()
        indices = np.array([1, 2])  # None, empty Point

        owned = from_shapely_geometries(geoms)
        host_subset = owned.take(indices)

        owned_gpu = from_shapely_geometries(geoms)
        owned_gpu.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)
        device_subset = owned_gpu.take(indices)

        host_shapely = host_subset.to_shapely()
        device_shapely = device_subset.to_shapely()
        assert device_shapely[0] is None
        assert device_shapely[1].is_empty
        _assert_geometries_equal(device_shapely, host_shapely)

    def test_all_families_single_row(self):
        """Each geometry family can be taken individually."""
        _require_gpu()
        geoms = _all_families()
        owned_host = from_shapely_geometries(geoms)
        owned_gpu = from_shapely_geometries(geoms)
        owned_gpu.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)

        for i, geom in enumerate(geoms):
            host_result = owned_host.take(np.array([i])).to_shapely()
            device_result = owned_gpu.take(np.array([i])).to_shapely()
            assert device_result[0].equals(host_result[0]), (
                f"family {geom.geom_type} at index {i} failed device take"
            )

    def test_polygon_with_holes(self):
        _require_gpu()
        poly = _polygon_with_hole()
        geoms = [Point(1, 1), poly, Point(5, 5)]

        owned = from_shapely_geometries(geoms)
        owned.take(np.array([1]))

        owned_gpu = from_shapely_geometries(geoms)
        owned_gpu.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)
        device_subset = owned_gpu.take(np.array([1]))

        device_shapely = device_subset.to_shapely()
        assert device_shapely[0].equals(poly)
        assert len(list(device_shapely[0].interiors)) == 1

    def test_empty_indices(self):
        _require_gpu()
        geoms = _sample_points()
        owned_gpu = from_shapely_geometries(geoms)
        owned_gpu.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)

        subset = owned_gpu.take(np.array([], dtype=np.int64))
        assert subset.row_count == 0
        assert subset.to_shapely() == []

    def test_equal_cardinality_reorder_preserves_metric_row_indirection(self):
        geoms = [
            MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]),
            MultiPolygon(
                [
                    Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
                    Polygon([(3, 0), (4, 0), (4, 1), (3, 0)]),
                ]
            ),
            MultiPolygon([Polygon([(0, 0), (3, 0), (3, 3), (0, 0)])]),
        ]
        owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
        reordered = owned.take(np.asarray([2, 0, 1], dtype=np.int64))

        assert reordered.is_indexed_view

        from vibespatial.constructive.measurement import area_owned, length_owned

        np.testing.assert_allclose(
            area_owned(reordered),
            np.asarray([geom.area for geom in (geoms[2], geoms[0], geoms[1])]),
        )
        np.testing.assert_allclose(
            length_owned(reordered),
            np.asarray([geom.length for geom in (geoms[2], geoms[0], geoms[1])]),
        )


# ---------------------------------------------------------------------------
# Residency and auto-dispatch
# ---------------------------------------------------------------------------


class TestDeviceTakeResidency:
    """Device take preserves residency and auto-dispatches."""

    def test_result_is_device_resident(self):
        _require_gpu()
        geoms = _sample_points()
        owned = from_shapely_geometries(geoms)
        owned.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)

        subset = owned.take(np.array([0, 1, 2]))
        assert subset.residency is Residency.DEVICE
        assert subset.device_state is not None

    def test_host_resident_take_stays_on_host(self):
        """When data is HOST-resident, take uses the host path."""
        _require_gpu()
        geoms = _sample_points()
        owned = from_shapely_geometries(geoms)

        subset = owned.take(np.array([0, 1, 2]))
        assert subset.residency is Residency.HOST
        assert subset.device_state is None

    def test_device_take_then_host_materialization(self):
        """device_take result can be materialized to host on demand."""
        _require_gpu()
        geoms = _all_families()
        owned = from_shapely_geometries(geoms)
        owned.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)

        subset = owned.take(np.array([0, 2, 4, 5]))
        assert subset.residency is Residency.DEVICE

        # Verify host_materialized is False for all family buffers
        for buffer in subset.families.values():
            assert not buffer.host_materialized

        shapely_result = subset.to_shapely()
        assert len(shapely_result) == 4
        assert shapely_result[0].equals(geoms[0])
        assert shapely_result[1].equals(geoms[2])

    def test_auto_dispatch_diagnostic(self):
        """device_take records a diagnostic event."""
        _require_gpu()
        geoms = _sample_points()
        owned = from_shapely_geometries(geoms)
        owned.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)

        subset = owned.take(np.array([0, 1]))
        assert any(
            "device_take" in event.detail
            for event in subset.diagnostics
        )


def test_polygonal_explode_from_capacity_multipolygon_view_retains_capacity():
    _require_gpu()
    import cupy as cp

    from vibespatial.constructive.binary_constructive import (
        _explode_polygonal_rows_to_polygon_capacity_gpu,
    )
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )

    logical_x = np.asarray(
        [0, 0, 1, 0, 2, 2, 3, 2, 10, 10, 11, 10, 12, 12, 13, 12],
        dtype=np.float64,
    )
    logical_y = np.asarray(
        [0, 1, 0, 0, 0, 1, 0, 0, 10, 11, 10, 10, 10, 11, 10, 10],
        dtype=np.float64,
    )
    buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        x=cp.asarray(np.concatenate([logical_x, np.full(80, 999.0)])),
        y=cp.asarray(np.concatenate([logical_y, np.full(80, 999.0)])),
        geometry_offsets=cp.asarray([0, 2, 4], dtype=cp.int32),
        part_offsets=cp.asarray([0, 1, 2, 3, 4], dtype=cp.int32),
        ring_offsets=cp.asarray([0, 4, 8, 12, 16], dtype=cp.int32),
        empty_mask=cp.zeros(2, dtype=cp.bool_),
    )
    owned = build_device_resident_owned(
        device_families={GeometryFamily.MULTIPOLYGON: buffer},
        row_count=2,
        tags=cp.full(2, FAMILY_TAGS[GeometryFamily.MULTIPOLYGON], dtype=cp.int8),
        validity=cp.ones(2, dtype=cp.bool_),
        family_row_offsets=cp.arange(2, dtype=cp.int32),
        execution_mode="gpu",
    )

    view = owned.take(cp.asarray([0, 1], dtype=cp.int64))
    view._device_resolve(allow_capacity_allocation=True)
    exploded = _explode_polygonal_rows_to_polygon_capacity_gpu(view)
    assert exploded is not None
    exploded.geometry._device_resolve(allow_capacity_allocation=True)
    state = exploded.geometry._ensure_device_state()
    polygon = state.families[GeometryFamily.POLYGON]

    assert cp.asnumpy(exploded.source_rows).tolist() == [0, 0, 1, 1]
    assert cp.asnumpy(exploded.logical_count).tolist() == [4]
    assert exploded.capacity == 4
    assert int(polygon.x.size) == int(logical_x.size) + 80
    assert int(polygon.y.size) == int(logical_y.size) + 80
    assert cp.asnumpy(polygon.geometry_offsets).tolist() == [0, 1, 2, 3, 4]
    assert cp.asnumpy(polygon.ring_offsets).tolist() == [0, 4, 8, 12, 16]
    assert np.array_equal(cp.asnumpy(polygon.x[: logical_x.size]), logical_x)
    assert np.array_equal(cp.asnumpy(polygon.y[: logical_y.size]), logical_y)


def test_nullable_homogeneous_device_placeholder_physicalizes_exact_rows():
    _require_gpu()
    import cupy as cp

    from vibespatial.geometry.buffers import GeometryFamily

    polygons = [
        Polygon(
            [
                (float(i), 0.0),
                (float(i + 1), 0.0),
                (float(i + 1), 1.0),
                (float(i), 1.0),
                (float(i), 0.0),
            ]
        )
        for i in range(8)
    ]
    owned = from_shapely_geometries(polygons, residency=Residency.DEVICE)
    activity = cp.asarray(
        [True, True, False, True, True, False, True, True],
        dtype=cp.bool_,
    )

    nullable = owned._device_indexed_take(
        cp.arange(owned.row_count, dtype=cp.int64),
        assume_unique_indices=True,
    )
    nullable._apply_row_activity(activity, assume_active_indices_unique=True)
    first = nullable.physicalize_device_rows(allow_capacity_allocation=True)
    assert not first.families[GeometryFamily.POLYGON].host_materialized

    second_view = first._device_indexed_take(
        cp.arange(first.row_count, dtype=cp.int64),
        assume_unique_indices=True,
    )
    second = second_view.physicalize_device_rows(allow_capacity_allocation=True)
    polygon = second._ensure_device_state().families[GeometryFamily.POLYGON]
    geometry_offsets = cp.asnumpy(polygon.geometry_offsets)
    ring_offsets = cp.asnumpy(polygon.ring_offsets)

    assert np.array_equal(geometry_offsets, [0, 1, 2, 2, 3, 4, 4, 5, 6])
    assert np.array_equal(ring_offsets, [0, 5, 10, 15, 20, 25, 30, 30, 30])
    _assert_geometries_equal(
        second.to_shapely(),
        [
            polygons[0],
            polygons[1],
            None,
            polygons[3],
            polygons[4],
            None,
            polygons[6],
            polygons[7],
        ],
    )


def test_lineal_explode_retains_physical_part_capacity():
    _require_gpu()
    import cupy as cp

    from vibespatial.constructive.binary_constructive import (
        _explode_lineal_rows_to_line_capacity_gpu,
    )
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )

    line_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        x=cp.asarray([0.0, 1.0]),
        y=cp.asarray([0.0, 0.0]),
        geometry_offsets=cp.asarray([0, 2], dtype=cp.int32),
        empty_mask=cp.zeros(1, dtype=cp.bool_),
    )
    multiline_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        x=cp.asarray([2.0, 3.0, 4.0, 5.0]),
        y=cp.asarray([0.0, 0.0, 1.0, 1.0]),
        geometry_offsets=cp.asarray([0, 2], dtype=cp.int32),
        part_offsets=cp.asarray([0, 2, 4], dtype=cp.int32),
        empty_mask=cp.zeros(1, dtype=cp.bool_),
    )
    owned = build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: line_buffer,
            GeometryFamily.MULTILINESTRING: multiline_buffer,
        },
        row_count=2,
        tags=cp.asarray(
            [
                FAMILY_TAGS[GeometryFamily.LINESTRING],
                FAMILY_TAGS[GeometryFamily.MULTILINESTRING],
            ],
            dtype=cp.int8,
        ),
        validity=cp.ones(2, dtype=cp.bool_),
        family_row_offsets=cp.asarray([0, 0], dtype=cp.int32),
        execution_mode="gpu",
    )

    exploded = _explode_lineal_rows_to_line_capacity_gpu(owned)
    assert exploded is not None
    exploded.geometry._device_resolve(allow_capacity_allocation=True)
    state = exploded.geometry._ensure_device_state()
    lines = state.families[GeometryFamily.LINESTRING]

    assert exploded.capacity == 3
    assert cp.asnumpy(exploded.logical_count).tolist() == [3]
    assert cp.asnumpy(exploded.source_rows).tolist() == [0, 1, 1]
    assert cp.asnumpy(lines.geometry_offsets).tolist() == [0, 2, 4, 6]
    assert cp.asnumpy(lines.x).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


# ---------------------------------------------------------------------------
# CuPy indices
# ---------------------------------------------------------------------------


class TestDeviceTakeCuPyIndices:
    """device_take accepts CuPy arrays directly."""

    def test_cupy_integer_indices(self):
        _require_gpu()
        import cupy as cupy_mod

        geoms = _sample_points()
        owned = from_shapely_geometries(geoms)
        owned.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)

        d_indices = cupy_mod.array([0, 3, 7], dtype=cupy_mod.int64)
        subset = owned.device_take(d_indices)

        host_ref = from_shapely_geometries(geoms).take(np.array([0, 3, 7]))
        _assert_geometries_equal(subset.to_shapely(), host_ref.to_shapely())

    def test_cupy_boolean_mask(self):
        _require_gpu()
        import cupy as cupy_mod

        geoms = _sample_points()
        mask = cupy_mod.array(
            [True, False, True, False, True, False, True, False, True, False],
        )

        owned = from_shapely_geometries(geoms)
        owned.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)
        subset = owned.device_take(mask)

        host_ref = from_shapely_geometries(geoms).take(
            np.array([True, False, True, False, True, False, True, False, True, False])
        )
        _assert_geometries_equal(subset.to_shapely(), host_ref.to_shapely())


# ---------------------------------------------------------------------------
# Multi-level offset families (stress tests)
# ---------------------------------------------------------------------------


class TestDeviceTakeMultiLevel:
    """Stress multi-level offset gather for complex geometry families."""

    def test_multiple_polygons(self):
        _require_gpu()
        polys = [
            Polygon([(i, i), (i + 3, i), (i + 3, i + 3), (i, i)])
            for i in range(20)
        ]
        owned = from_shapely_geometries(polys)
        owned.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)

        indices = np.array([0, 5, 10, 15, 19])
        device_subset = owned.take(indices)
        host_subset = from_shapely_geometries(polys).take(indices)
        _assert_geometries_equal(device_subset.to_shapely(), host_subset.to_shapely())

    def test_multiple_multipolygons(self):
        _require_gpu()
        mpolys = [
            MultiPolygon([
                Polygon([(i, i), (i + 1, i), (i + 1, i + 1), (i, i)]),
                Polygon([(i + 5, i + 5), (i + 6, i + 5), (i + 6, i + 6), (i + 5, i + 5)]),
            ])
            for i in range(15)
        ]
        owned = from_shapely_geometries(mpolys)
        owned.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)

        indices = np.array([0, 7, 14])
        device_subset = owned.take(indices)
        host_subset = from_shapely_geometries(mpolys).take(indices)
        _assert_geometries_equal(device_subset.to_shapely(), host_subset.to_shapely())

    def test_multilinestrings(self):
        _require_gpu()
        mlines = [
            MultiLineString([
                [(i, i), (i + 1, i + 1)],
                [(i + 2, i + 2), (i + 3, i + 3), (i + 4, i + 4)],
            ])
            for i in range(10)
        ]
        owned = from_shapely_geometries(mlines)
        owned.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)

        indices = np.array([0, 4, 9])
        device_subset = owned.take(indices)
        host_subset = from_shapely_geometries(mlines).take(indices)
        _assert_geometries_equal(device_subset.to_shapely(), host_subset.to_shapely())

    def test_fixed_multilinestring_device_take_uses_structural_size_metadata(self):
        _require_gpu()
        import cupy as cupy_mod

        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.buffers import GeometryFamily

        mlines = [
            MultiLineString([
                [(i, i), (i + 1, i + 1)],
                [(i + 2, i + 2), (i + 3, i + 3), (i + 4, i + 4)],
            ])
            for i in range(6)
        ]
        owned = from_shapely_geometries(mlines, residency=Residency.DEVICE)
        source_buffer = owned.device_state.families[GeometryFamily.MULTILINESTRING]
        assert source_buffer.fixed_size is not None
        assert source_buffer.fixed_size.first_level_count_per_row == 2
        assert source_buffer.fixed_size.coord_count_per_row == 5

        reset_d2h_transfer_count()
        subset = owned.device_take(cupy_mod.asarray([5, 2, 0], dtype=cupy_mod.int64))
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert "owned geometry device-take nested slice-size allocation fence" not in reasons
        assert "owned geometry device-take slice-size allocation fence" not in reasons
        subset_buffer = subset.device_state.families[GeometryFamily.MULTILINESTRING]
        assert subset_buffer.fixed_size == source_buffer.fixed_size
        _assert_geometries_equal(subset.to_shapely(), [mlines[5], mlines[2], mlines[0]])

    def test_fixed_multipolygon_device_take_uses_structural_size_metadata(self):
        _require_gpu()
        import cupy as cupy_mod

        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.buffers import GeometryFamily

        mpolys = [
            MultiPolygon([
                Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 0)]),
                Polygon([(i + 3, 0), (i + 4, 0), (i + 4, 1), (i + 3, 0)]),
            ])
            for i in range(6)
        ]
        owned = from_shapely_geometries(mpolys, residency=Residency.DEVICE)
        source_buffer = owned.device_state.families[GeometryFamily.MULTIPOLYGON]
        assert source_buffer.fixed_size is not None
        assert source_buffer.fixed_size.first_level_count_per_row == 2
        assert source_buffer.fixed_size.second_level_count_per_row == 2
        assert source_buffer.fixed_size.coord_count_per_row == 8

        reset_d2h_transfer_count()
        subset = owned.device_take(cupy_mod.asarray([4, 2, 0], dtype=cupy_mod.int64))
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert "owned geometry device-take nested slice-size allocation fence" not in reasons
        assert "owned geometry device-take slice-size allocation fence" not in reasons
        subset_buffer = subset.device_state.families[GeometryFamily.MULTIPOLYGON]
        assert subset_buffer.fixed_size == source_buffer.fixed_size
        _assert_geometries_equal(subset.to_shapely(), [mpolys[4], mpolys[2], mpolys[0]])

    def test_variable_polygon_multipolygon_device_take_uses_row_indirection(self, tmp_path):
        _require_gpu()
        import cupy as cupy_mod
        import pandas as pd

        import geopandas as gpd
        from vibespatial.api._native_result_core import GeometryNativeResult, NativeTabularResult
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

        geoms = [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
            MultiPolygon([
                Polygon([(10, 0), (11, 0), (11, 1), (10, 0)]),
                Polygon([(12, 0), (13, 0), (13, 1), (12, 0)]),
            ]),
            Polygon(
                [(20, 0), (24, 0), (24, 4), (20, 0)],
                [[(21, 1), (22, 1), (22, 2), (21, 1)]],
            ),
            MultiPolygon([
                Polygon(
                    [(30, 0), (34, 0), (34, 4), (30, 0)],
                    [[(31, 1), (32, 1), (32, 2), (31, 1)]],
                )
            ]),
            Polygon([(40, 0), (45, 0), (45, 5), (40, 0)]),
        ]
        owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
        order = cupy_mod.asarray([3, 1, 2, 0], dtype=cupy_mod.int64)

        reset_d2h_transfer_count()
        subset = owned.device_take(order)
        take_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert subset.is_indexed_view
        assert "owned geometry device-take nested slice-size allocation fence" not in take_reasons
        assert "owned geometry device-take slice-size allocation fence" not in take_reasons
        expected = [geoms[3], geoms[1], geoms[2], geoms[0]]
        _assert_geometries_equal(subset.to_shapely(), expected)

        reset_d2h_transfer_count()
        bounds = compute_geometry_bounds_device(
            subset,
            preserve_indexed_view=True,
        )
        bounds_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
        assert bounds.shape == (len(expected), 4)
        assert "owned geometry device-take nested slice-size allocation fence" not in bounds_reasons
        assert "owned geometry device-take slice-size allocation fence" not in bounds_reasons

        reset_d2h_transfer_count()
        assert subset.to_wkb() == from_shapely_geometries(expected).to_wkb()
        wkb_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
        assert "owned geometry device-take nested slice-size allocation fence" not in wkb_reasons
        assert "owned geometry device-take slice-size allocation fence" not in wkb_reasons

        native = NativeTabularResult(
            attributes=pd.DataFrame({"value": np.arange(len(expected), dtype=np.int32)}),
            geometry=GeometryNativeResult.from_owned(subset, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
        gdf = native.to_geodataframe()
        np.testing.assert_allclose(
            gdf.total_bounds,
            gpd.GeoSeries(expected).total_bounds,
        )
        output_path = tmp_path / "row_indirected.parquet"
        reset_d2h_transfer_count()
        gdf.to_parquet(output_path)
        parquet_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
        assert "owned geometry device-take nested slice-size allocation fence" not in parquet_reasons
        assert "owned geometry device-take slice-size allocation fence" not in parquet_reasons
        roundtrip = gpd.read_parquet(output_path)
        _assert_geometries_equal(list(roundtrip.geometry), expected)

    def test_variable_polygon_host_index_scatter_uses_row_indirection(self):
        _require_gpu()
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.owned import device_concat_owned_scatter

        base_geoms = [
            Polygon([(0, 0), (4, 0), (3, 3), (0, 0)]),
            Polygon(
                [(10, 0), (14, 0), (14, 4), (10, 0)],
                [[(11, 1), (12, 1), (12, 2), (11, 1)]],
            ),
            Polygon([(20, 0), (24, 0), (23, 2), (20, 0)]),
        ]
        replacement_geoms = [
            Polygon(
                [(100, 0), (105, 0), (105, 5), (100, 0)],
                [[(101, 1), (102, 1), (102, 2), (101, 1)]],
            ),
            Polygon([(200, 0), (204, 0), (203, 3), (200, 0)]),
        ]
        base = from_shapely_geometries(base_geoms, residency=Residency.DEVICE)
        replacement = from_shapely_geometries(
            replacement_geoms,
            residency=Residency.DEVICE,
        )
        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)

        result = device_concat_owned_scatter(
            base,
            replacement,
            np.asarray([1, 2], dtype=np.int64),
        )
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert result.is_indexed_view
        assert "owned geometry device-take nested slice-size allocation fence" not in reasons
        assert "owned geometry device-take slice-size allocation fence" not in reasons
        _assert_geometries_equal(
            result.to_shapely(),
            [base_geoms[0], replacement_geoms[0], replacement_geoms[1]],
        )

    def test_variable_polygon_multi_scatter_uses_single_row_indirection(self):
        _require_gpu()
        import cupy as cupy_mod

        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.owned import device_concat_owned_scatter_many

        base_geoms = [
            Polygon([(0, 0), (4, 0), (3, 3), (0, 0)]),
            Polygon([(10, 0), (14, 0), (14, 4), (10, 0)]),
            Polygon(
                [(20, 0), (24, 0), (24, 4), (20, 0)],
                [[(21, 1), (22, 1), (22, 2), (21, 1)]],
            ),
            Polygon([(30, 0), (34, 0), (33, 3), (30, 0)]),
        ]
        first_geoms = [
            Polygon([(100, 0), (105, 0), (105, 5), (100, 0)]),
            Polygon([(200, 0), (205, 0), (204, 4), (200, 0)]),
        ]
        second_geoms = [
            Polygon(
                [(300, 0), (305, 0), (305, 5), (300, 0)],
                [[(301, 1), (302, 1), (302, 2), (301, 1)]],
            ),
        ]
        base = from_shapely_geometries(base_geoms, residency=Residency.DEVICE)
        first = from_shapely_geometries(first_geoms, residency=Residency.DEVICE)
        second = from_shapely_geometries(second_geoms, residency=Residency.DEVICE)

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        result = device_concat_owned_scatter_many(
            base,
            [
                (first, cupy_mod.asarray([1, 3], dtype=cupy_mod.int64)),
                (second, cupy_mod.asarray([2], dtype=cupy_mod.int64)),
            ],
        )
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert result.is_indexed_view
        assert result._device_scatter_implementation == "device_scatter_row_indirection_many"
        assert "owned geometry device-take nested slice-size allocation fence" not in reasons
        assert "owned geometry device-take slice-size allocation fence" not in reasons
        _assert_geometries_equal(
            result.to_shapely(),
            [base_geoms[0], first_geoms[0], second_geoms[0], first_geoms[1]],
        )

    def test_capacity_partition_selection_uses_one_row_indirection(self):
        _require_gpu()
        import cupy as cupy_mod

        from vibespatial.cuda._runtime import assert_zero_d2h_transfers
        from vibespatial.geometry.owned import (
            build_null_owned_array,
            device_select_owned_capacity_partitions,
        )

        first_geoms = [
            Polygon([(0, 0), (2, 0), (1, 2), (0, 0)]),
            None,
            Polygon([(20, 0), (22, 0), (21, 2), (20, 0)]),
        ]
        second_geoms = [
            None,
            LineString([(10, 0), (12, 2)]),
            None,
        ]
        first = from_shapely_geometries(first_geoms, residency=Residency.DEVICE)
        second = from_shapely_geometries(second_geoms, residency=Residency.DEVICE)
        base = build_null_owned_array(3, residency=Residency.DEVICE)

        with assert_zero_d2h_transfers():
            result = device_select_owned_capacity_partitions(
                base,
                [
                    (first, cupy_mod.asarray([True, False, True])),
                    (second, cupy_mod.asarray([False, True, False])),
                ],
            )

        assert result.is_indexed_view
        assert (
            result._device_scatter_implementation
            == "device_capacity_partition_selection"
        )
        _assert_geometries_equal(
            result.to_shapely(),
            [first_geoms[0], second_geoms[1], first_geoms[2]],
        )

    def test_capacity_selection_scatter_routes_inactive_lanes_to_scratch(self):
        _require_gpu()
        import cupy as cupy_mod

        from vibespatial.api._native_rowset import NativeDeviceSelection
        from vibespatial.cuda._runtime import assert_zero_d2h_transfers
        from vibespatial.geometry.owned import device_scatter_owned_capacity_selection

        base_geoms = [
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(2.0, 0.0),
            Point(3.0, 0.0),
        ]
        replacement_geoms = [
            LineString([(10.0, 0.0), (11.0, 1.0)]),
            LineString([(30.0, 0.0), (31.0, 1.0)]),
            None,
            None,
        ]
        base = from_shapely_geometries(base_geoms, residency=Residency.DEVICE)
        replacement = from_shapely_geometries(
            replacement_geoms,
            residency=Residency.DEVICE,
        )
        selection = NativeDeviceSelection.from_mask(
            cupy_mod.asarray([False, True, False, True]),
            source_row_count=4,
        )

        with assert_zero_d2h_transfers():
            result = device_scatter_owned_capacity_selection(
                base,
                replacement,
                selection,
                active_mask=cupy_mod.asarray([True, False, True, True]),
            )

        assert result.is_indexed_view
        assert (
            result._device_scatter_implementation
            == "device_capacity_selection_scatter"
        )
        _assert_geometries_equal(
            result.to_shapely(),
            [base_geoms[0], replacement_geoms[0], base_geoms[2], base_geoms[3]],
        )

    def test_fused_capacity_scatter_physicalizes_multiple_roots_once(self):
        _require_gpu()
        import cupy as cupy_mod

        from vibespatial.api._native_rowset import NativeDeviceSelection
        from vibespatial.geometry.owned import (
            device_scatter_owned_capacity_selections_many,
        )

        base_geoms = [Point(float(i), 0.0) for i in range(4)]
        first_geoms = [
            LineString([(10.0 + i, 0.0), (10.0 + i, 1.0)]) for i in range(4)
        ]
        second_geoms = [
            LineString([(20.0 + i, 0.0), (20.0 + i, 1.0)]) for i in range(4)
        ]
        base = from_shapely_geometries(base_geoms, residency=Residency.DEVICE)
        first = from_shapely_geometries(first_geoms, residency=Residency.DEVICE)
        second = from_shapely_geometries(second_geoms, residency=Residency.DEVICE)
        first_selection = NativeDeviceSelection.from_mask(
            cupy_mod.asarray([False, True, False, True]),
            source_row_count=4,
        )
        second_selection = NativeDeviceSelection.from_mask(
            cupy_mod.asarray([False, False, True, True]),
            source_row_count=4,
        )

        result = device_scatter_owned_capacity_selections_many(
            base,
            [
                (first, first_selection, None),
                (second, second_selection, None),
            ],
        )

        assert result.is_indexed_view
        assert (
            result._device_scatter_implementation
            == "device_exact_capacity_selection_scatter_many"
        )
        _assert_geometries_equal(
            result.to_shapely(),
            [base_geoms[0], first_geoms[0], second_geoms[0], second_geoms[1]],
        )

    def test_concat_flattens_nested_device_indexed_views_without_take_fence(self):
        _require_gpu()
        import cupy as cupy_mod

        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.owned import OwnedGeometryArray

        geoms = [
            Polygon([(0, 0), (4, 0), (4, 2), (2, 4), (0, 2), (0, 0)]),
            MultiPolygon([
                Polygon([(10, 0), (12, 0), (12, 2), (10, 0)]),
                Polygon([(13, 0), (15, 0), (15, 2), (13, 0)]),
            ]),
            Polygon(
                [(20, 0), (25, 0), (25, 5), (20, 5), (20, 0)],
                [[(21, 1), (23, 1), (23, 3), (21, 1)]],
            ),
            MultiPolygon([
                Polygon(
                    [(30, 0), (35, 0), (35, 5), (30, 5), (30, 0)],
                    [[(31, 1), (33, 1), (33, 3), (31, 1)]],
                )
            ]),
            Polygon([(40, 0), (46, 0), (46, 3), (43, 6), (40, 3), (40, 0)]),
        ]
        base = from_shapely_geometries(geoms, residency=Residency.DEVICE)
        first = base.device_take(cupy_mod.asarray([4, 2, 0, 3], dtype=cupy_mod.int64))
        second = first.device_take(cupy_mod.asarray([2, 0], dtype=cupy_mod.int64))
        sibling = base.device_take(cupy_mod.asarray([1, 3], dtype=cupy_mod.int64))

        assert first.is_indexed_view
        assert second.is_indexed_view
        assert sibling.is_indexed_view

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        result = OwnedGeometryArray.concat([second, sibling])
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert result.is_indexed_view
        assert "owned geometry device-take nested slice-size allocation fence" not in reasons
        assert "owned geometry device-take slice-size allocation fence" not in reasons
        expected = [geoms[0], geoms[4], geoms[1], geoms[3]]
        _assert_geometries_equal(result.to_shapely(), expected)

    def test_chained_device_take(self):
        """Take from a device_take result (successive compaction)."""
        _require_gpu()
        geoms = _all_families()
        owned = from_shapely_geometries(geoms)
        owned.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)

        first = owned.take(np.array([0, 2, 4, 5]))
        assert first.residency is Residency.DEVICE

        second = first.take(np.array([0, 2]))
        assert second.residency is Residency.DEVICE

        result = second.to_shapely()
        assert result[0].equals(geoms[0])  # Point
        assert result[1].equals(geoms[4])  # Polygon
