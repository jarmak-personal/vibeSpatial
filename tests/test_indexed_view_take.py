"""Tests for virtual indexed-view take optimisation.

Verifies that ``OwnedGeometryArray.take()`` returns an indexed view (not a
physical copy) when indices have high repetition, and that the indexed view
produces identical results to the physical copy path.
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

from vibespatial import from_shapely_geometries, has_gpu_runtime
from vibespatial.geometry.owned import OwnedGeometryArray

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_geometries() -> list[object | None]:
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


def _make_high_repetition_indices(
    source_size: int, total_size: int
) -> np.ndarray:
    """Create indices with very high repetition (each source index repeated many times)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, source_size, size=total_size, dtype=np.int64)


def _assert_shapely_equal(left: list, right: list) -> None:
    """Assert two lists of Shapely geometries / None are element-wise equal."""
    assert len(left) == len(right), f"Length mismatch: {len(left)} vs {len(right)}"
    for i, (left_g, right_g) in enumerate(zip(left, right)):
        if left_g is None and right_g is None:
            continue
        if left_g is None or right_g is None:
            raise AssertionError(f"Row {i}: one is None, other is not")
        if left_g.is_empty and right_g.is_empty:
            assert left_g.geom_type == right_g.geom_type, (
                f"Row {i}: empty geometry type mismatch {left_g.geom_type} vs {right_g.geom_type}"
            )
            continue
        assert left_g.equals(right_g), f"Row {i}: geometries differ"


# ---------------------------------------------------------------------------
# Test: indexed view is created for high-repetition indices
# ---------------------------------------------------------------------------

class TestIndexedViewCreation:
    """Verify that indexed views are created when repetition is high."""

    def test_high_repetition_creates_indexed_view(self):
        owned = from_shapely_geometries(_sample_geometries())
        # 6 unique geoms, repeat to 2000 rows (well above MIN_ROWS=1000)
        indices = _make_high_repetition_indices(owned.row_count, 2000)
        result = owned.take(indices)
        assert result.is_indexed_view
        assert result.row_count == 2000
        # Base should have at most 6 unique rows
        assert result._base.row_count <= owned.row_count

    def test_low_repetition_does_not_create_indexed_view(self):
        owned = from_shapely_geometries(_sample_geometries())
        # All distinct indices (no repetition)
        indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        result = owned.take(indices)
        assert not result.is_indexed_view

    def test_small_array_does_not_create_indexed_view(self):
        owned = from_shapely_geometries(_sample_geometries())
        # Even with repetition, small arrays should not use indexed views
        indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        result = owned.take(indices)
        assert not result.is_indexed_view

    def test_threshold_boundary(self):
        """Test that at exactly _INDEXED_VIEW_MIN_ROWS with high repetition,
        an indexed view is created."""
        geoms = [Point(i, i) for i in range(10)]
        owned = from_shapely_geometries(geoms)
        # 1000 rows from 10 unique = 10% unique ratio, well under 0.5
        indices = np.tile(np.arange(10, dtype=np.int64), 100)
        assert indices.size == 1000
        result = owned.take(indices)
        assert result.is_indexed_view


# ---------------------------------------------------------------------------
# Test: indexed view produces correct Shapely output
# ---------------------------------------------------------------------------

class TestIndexedViewCorrectness:
    """Verify indexed views produce identical results to physical copies."""

    def test_to_shapely_matches_physical_copy(self):
        owned = from_shapely_geometries(_sample_geometries())
        indices = _make_high_repetition_indices(owned.row_count, 2000)

        # Force physical copy by using _physical_take directly
        physical = owned._physical_take(indices)
        # Indexed view via take()
        virtual = owned.take(indices)

        assert virtual.is_indexed_view
        physical_shapely = physical.to_shapely()
        virtual_shapely = virtual.to_shapely()
        _assert_shapely_equal(physical_shapely, virtual_shapely)

    def test_to_shapely_with_nulls_and_empties(self):
        owned = from_shapely_geometries(_sample_geometries())
        # Include null (index 1) and empty (index 2) repeatedly
        indices = np.array([1, 2, 1, 2, 0, 0] * 200, dtype=np.int64)
        assert indices.size >= 1000

        physical = owned._physical_take(indices)
        virtual = owned.take(indices)

        assert virtual.is_indexed_view
        physical_shapely = physical.to_shapely()
        virtual_shapely = virtual.to_shapely()
        _assert_shapely_equal(physical_shapely, virtual_shapely)

    def test_all_families_round_trip(self):
        owned = from_shapely_geometries(_all_families())
        indices = _make_high_repetition_indices(owned.row_count, 2000)

        physical = owned._physical_take(indices)
        virtual = owned.take(indices)

        assert virtual.is_indexed_view
        _assert_shapely_equal(physical.to_shapely(), virtual.to_shapely())

    def test_polygon_with_holes(self):
        poly_with_hole = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            [[(2, 2), (4, 2), (4, 4), (2, 2)]],
        )
        geoms = [Point(1, 1), poly_with_hole, Point(5, 5)]
        owned = from_shapely_geometries(geoms)
        # Repeat heavily — mostly the polygon with hole
        indices = np.array([1] * 1200 + [0] * 200 + [2] * 200, dtype=np.int64)
        result = owned.take(indices)

        assert result.is_indexed_view
        shapely_out = result.to_shapely()
        assert shapely_out[0].equals(poly_with_hole)
        assert len(list(shapely_out[0].interiors)) == 1


# ---------------------------------------------------------------------------
# Test: row_count property works correctly
# ---------------------------------------------------------------------------

class TestIndexedViewRowCount:
    def test_row_count_reflects_logical_size(self):
        owned = from_shapely_geometries(_sample_geometries())
        indices = _make_high_repetition_indices(owned.row_count, 5000)
        result = owned.take(indices)
        assert result.row_count == 5000

    def test_base_row_count_is_compact(self):
        owned = from_shapely_geometries(_sample_geometries())
        indices = _make_high_repetition_indices(owned.row_count, 5000)
        result = owned.take(indices)
        assert result._base.row_count <= owned.row_count


# ---------------------------------------------------------------------------
# Test: metadata properties expand through index map
# ---------------------------------------------------------------------------

class TestIndexedViewMetadata:
    def test_validity_is_expanded(self):
        owned = from_shapely_geometries(_sample_geometries())
        indices = _make_high_repetition_indices(owned.row_count, 2000)
        result = owned.take(indices)

        # Validity should be a full-size array
        assert result.validity.shape == (2000,)
        # Check against direct indexing
        expected_validity = owned.validity[indices]
        np.testing.assert_array_equal(result.validity, expected_validity)

    def test_tags_is_expanded(self):
        owned = from_shapely_geometries(_sample_geometries())
        indices = _make_high_repetition_indices(owned.row_count, 2000)
        result = owned.take(indices)

        assert result.tags.shape == (2000,)
        expected_tags = owned.tags[indices]
        np.testing.assert_array_equal(result.tags, expected_tags)


# ---------------------------------------------------------------------------
# Test: resolve materialises correctly
# ---------------------------------------------------------------------------

class TestResolve:
    def test_resolve_produces_flat_array(self):
        owned = from_shapely_geometries(_sample_geometries())
        indices = _make_high_repetition_indices(owned.row_count, 2000)
        result = owned.take(indices)

        assert result.is_indexed_view
        result._resolve()
        assert not result.is_indexed_view
        assert result.row_count == 2000

    def test_resolve_is_idempotent(self):
        owned = from_shapely_geometries(_sample_geometries())
        indices = _make_high_repetition_indices(owned.row_count, 2000)
        result = owned.take(indices)

        result._resolve()
        shapely1 = result.to_shapely()
        result._resolve()  # second resolve is a no-op
        shapely2 = result.to_shapely()
        _assert_shapely_equal(shapely1, shapely2)

    def test_ensure_host_state_resolves(self):
        owned = from_shapely_geometries(_sample_geometries())
        indices = _make_high_repetition_indices(owned.row_count, 2000)
        result = owned.take(indices)

        assert result.is_indexed_view
        result._ensure_host_state()
        assert not result.is_indexed_view


# ---------------------------------------------------------------------------
# Test: chained take on indexed views
# ---------------------------------------------------------------------------

class TestChainedTake:
    def test_take_on_indexed_view_composes_maps(self):
        """Taking from an indexed view should compose the index maps,
        not stack views on views."""
        geoms = [Point(i, i) for i in range(10)]
        owned = from_shapely_geometries(geoms)

        # First take: 10 unique -> 2000 rows
        indices1 = _make_high_repetition_indices(10, 2000)
        view1 = owned.take(indices1)
        assert view1.is_indexed_view

        # Get expected output BEFORE second take (since to_shapely is lazy)
        view1_shapely = view1.to_shapely()

        # Second take on the view: draw from only 50 unique logical rows
        # (high repetition: 50/3000 < 0.5) to trigger indexed view path
        rng = np.random.default_rng(99)
        # Pick 50 random logical row indices, then repeat them to 3000
        pool = rng.integers(0, 2000, size=50, dtype=np.int64)
        indices2 = rng.choice(pool, size=3000).astype(np.int64)

        # Re-create view1 (to_shapely may have resolved it)
        view1 = owned.take(indices1)
        assert view1.is_indexed_view

        view2 = view1.take(indices2)
        assert view2.is_indexed_view
        assert view2.row_count == 3000

        # Verify the result is NOT a view-of-a-view
        # (the base should be a flat physical array, not another indexed view)
        assert not view2._base.is_indexed_view

        # Verify correctness
        expected = [view1_shapely[i] for i in indices2]
        actual = view2.to_shapely()
        for i, (e, a) in enumerate(zip(expected, actual)):
            if e is None and a is None:
                continue
            assert e.equals(a), f"Row {i} differs"


# ---------------------------------------------------------------------------
# Test: concat resolves indexed views
# ---------------------------------------------------------------------------

class TestConcatWithIndexedViews:
    def test_concat_two_indexed_views(self):
        geoms = [Point(i, i) for i in range(10)]
        owned = from_shapely_geometries(geoms)

        view1 = owned.take(_make_high_repetition_indices(10, 2000))
        view2 = owned.take(_make_high_repetition_indices(10, 1500))

        assert view1.is_indexed_view
        assert view2.is_indexed_view

        result = OwnedGeometryArray.concat([view1, view2])
        assert result.row_count == 3500
        # After concat, the views should have been resolved
        assert not view1.is_indexed_view
        assert not view2.is_indexed_view

    def test_concat_mixed_indexed_and_flat(self):
        geoms = [Point(i, i) for i in range(10)]
        owned = from_shapely_geometries(geoms)

        view = owned.take(_make_high_repetition_indices(10, 2000))
        flat = owned.take(np.arange(5, dtype=np.int64))  # small, no indexed view

        assert view.is_indexed_view
        assert not flat.is_indexed_view

        result = OwnedGeometryArray.concat([view, flat])
        assert result.row_count == 2005


# ---------------------------------------------------------------------------
# Test: memory savings are real
# ---------------------------------------------------------------------------

class TestMemorySavings:
    def test_indexed_view_uses_compact_base(self):
        """An indexed view should store only unique rows worth of coordinate data."""
        polys = [
            Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)])
            for i in range(20)
        ]
        owned = from_shapely_geometries(polys)

        # 20 unique polygons expanded to 5000 rows
        indices = _make_high_repetition_indices(20, 5000)
        result = owned.take(indices)

        assert result.is_indexed_view
        # The base should have at most 20 rows (the unique ones)
        assert result._base.row_count <= 20
        # The index map should have 5000 entries
        assert result._index_map.shape == (5000,)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestDeviceTakeIndexedView:
    def test_device_take_high_repetition_creates_indexed_view(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        from vibespatial import Residency, TransferTrigger

        geoms = [Point(i, i) for i in range(20)]
        owned = from_shapely_geometries(geoms)
        owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="test",
        )

        indices = _make_high_repetition_indices(20, 3000)
        result = owned.device_take(indices)

        assert result.is_indexed_view
        assert result.row_count == 3000
        assert result._base.row_count <= 20

    def test_device_take_indexed_view_to_shapely(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        from vibespatial import Residency, TransferTrigger

        geoms = _sample_geometries()
        owned = from_shapely_geometries(geoms)
        owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="test",
        )

        indices = _make_high_repetition_indices(len(geoms), 2000)
        result = owned.device_take(indices)

        assert result.is_indexed_view
        shapely_result = result.to_shapely()
        assert len(shapely_result) == 2000

        # Verify correctness by checking a few rows
        for i in range(min(100, len(indices))):
            src_idx = indices[i]
            expected = geoms[src_idx]
            actual = shapely_result[i]
            if expected is None:
                assert actual is None
            elif hasattr(expected, 'is_empty') and expected.is_empty:
                assert actual.is_empty
            else:
                assert actual.equals(expected), f"Row {i} (src={src_idx}) differs"

    def test_device_take_indexed_view_resolves_for_gpu_kernel(self):
        """Ensure that calling _ensure_device_state on an indexed view
        resolves it so GPU kernels get contiguous buffers."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        from vibespatial import Residency, TransferTrigger

        geoms = [Point(i, i) for i in range(20)]
        owned = from_shapely_geometries(geoms)
        owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="test",
        )

        indices = _make_high_repetition_indices(20, 3000)
        result = owned.device_take(indices)

        assert result.is_indexed_view
        # This should resolve the indexed view
        result._ensure_device_state()
        assert not result.is_indexed_view
        assert result.device_state is not None

    def test_device_take_indexed_view_keeps_index_map_on_device(self):
        """The device_take indexed view must keep its index map as a
        CuPy array (no D2H transfer of the inverse map)."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        from vibespatial import Residency, TransferTrigger

        geoms = [Point(i, i) for i in range(20)]
        owned = from_shapely_geometries(geoms)
        owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="test",
        )

        indices = _make_high_repetition_indices(20, 3000)
        result = owned.device_take(indices)

        assert result.is_indexed_view
        # The index map should be a CuPy array, NOT numpy
        assert hasattr(result._index_map, "__cuda_array_interface__"), (
            f"Expected CuPy index map, got {type(result._index_map)}"
        )
        # Host metadata should NOT have been materialized
        assert result._validity is None, (
            "Host validity should be None (lazy) for device-resident indexed view"
        )
        assert result._tags is None, (
            "Host tags should be None (lazy) for device-resident indexed view"
        )

    def test_device_take_indexed_view_device_resolve_no_host_roundtrip(self):
        """Resolving a device-resident indexed view via _ensure_device_state
        must NOT touch host buffers -- the entire resolve stays on GPU."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        from vibespatial import Residency, TransferTrigger

        geoms = [Point(i, i) for i in range(20)]
        owned = from_shapely_geometries(geoms)
        owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="test",
        )

        indices = _make_high_repetition_indices(20, 3000)
        result = owned.device_take(indices)

        assert result.is_indexed_view
        # Before resolve: host metadata is lazy (None)
        assert result._validity is None

        # Resolve on device
        result._ensure_device_state()

        assert not result.is_indexed_view
        assert result.device_state is not None
        assert result.residency is Residency.DEVICE
        # After device resolve: the result should be DEVICE-resident
        # with proper device state, and row_count preserved
        assert result.row_count == 3000


# ---------------------------------------------------------------------------
# Test: DGA take with indexed views
# ---------------------------------------------------------------------------

class TestDGATakeIndexedView:
    def test_dga_take_propagates_indexed_view(self):
        """DeviceGeometryArray.take() should benefit from indexed views."""
        from vibespatial.geometry.device_array import DeviceGeometryArray

        geoms = [Point(i, i) for i in range(20)]
        owned = from_shapely_geometries(geoms)
        dga = DeviceGeometryArray(owned)

        indices = _make_high_repetition_indices(20, 2000)
        result = dga.take(indices)

        assert result._owned.is_indexed_view
        assert len(result) == 2000

    def test_dga_take_allow_fill_works_with_indexed_view(self):
        """allow_fill with negative indices should work correctly
        even when the underlying owned array is an indexed view."""
        from vibespatial.geometry.device_array import DeviceGeometryArray

        geoms = [Point(i, i) for i in range(20)]
        owned = from_shapely_geometries(geoms)
        dga = DeviceGeometryArray(owned)

        # Create indices with some -1 (fill) markers
        indices = np.zeros(2000, dtype=np.int64)
        indices[:1500] = np.tile(np.arange(20, dtype=np.int64), 75)
        indices[1500:] = -1

        result = dga.take(indices, allow_fill=True)
        assert len(result) == 2000

        # Verify fill positions are null
        for i in range(1500, 2000):
            assert result._materialize_row(i) is None

    def test_dga_take_shapely_cache_propagated(self):
        """When shapely_cache exists, DGA take should propagate it."""
        from vibespatial.geometry.device_array import DeviceGeometryArray

        geoms = [Point(i, i) for i in range(20)]
        owned = from_shapely_geometries(geoms)
        dga = DeviceGeometryArray(owned)

        # Force shapely cache to be populated
        _ = dga._ensure_shapely_cache()
        assert dga._shapely_cache is not None

        # Now take with repetition — DGA caches are propagated via indexing
        indices = np.arange(20, dtype=np.int64)  # small, no indexed view
        result = dga.take(indices)
        assert result._shapely_cache is not None
