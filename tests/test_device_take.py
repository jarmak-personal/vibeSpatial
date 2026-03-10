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
