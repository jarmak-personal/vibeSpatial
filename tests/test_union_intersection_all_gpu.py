"""Tests for GPU tree-reduction global set operations.

Covers:
  - union_all_gpu_owned (vibeSpatial-247.4.7)
  - coverage_union_all_gpu_owned (vibeSpatial-247.4.8)
  - intersection_all_gpu_owned (vibeSpatial-247.4.9)
  - unary_union_gpu_owned (vibeSpatial-247.4.10)
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import box

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries

# ---------------------------------------------------------------------------
# GPU availability check
# ---------------------------------------------------------------------------

def _has_gpu() -> bool:
    try:
        import cupy as cp
        _ = cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_shapely(owned: OwnedGeometryArray):
    """Materialise a 1-row OGA to a single Shapely geometry."""
    geoms = owned.to_shapely()
    assert len(geoms) == 1, f"Expected 1-row OGA, got {len(geoms)}"
    return geoms[0]


def _geom_equiv(a, b, *, tolerance: float = 1e-6) -> bool:
    """Check topological equivalence with tolerance for floating-point noise."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a.is_empty and b.is_empty:
        return True
    # Symmetric difference area should be near-zero for equivalent geometries.
    try:
        sym_diff = a.symmetric_difference(b)
        return sym_diff.area < tolerance
    except Exception:
        return False


# ---------------------------------------------------------------------------
# union_all_gpu tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestUnionAllGPU:
    """Tests for union_all_gpu_owned."""

    def test_basic_polygon_union(self):
        """Union of overlapping polygons matches Shapely."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        polys = [
            box(0, 0, 2, 2),
            box(1, 1, 3, 3),
            box(2, 0, 4, 2),
        ]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result_geom, expected), (
            f"GPU union_all != Shapely union_all\n"
            f"  GPU area={result_geom.area}, expected area={expected.area}"
        )

    def test_non_overlapping_polygons(self):
        """Union of non-overlapping polygons."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        polys = [box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5)]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert abs(result_geom.area - expected.area) < 1e-6

    def test_empty_input(self):
        """Empty input returns an empty geometry."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        owned = from_shapely_geometries([])
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert result_geom.is_empty

    def test_single_row(self):
        """Single-row input returns the input geometry."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        poly = box(0, 0, 10, 10)
        owned = from_shapely_geometries([poly])
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert _geom_equiv(result_geom, poly)

    def test_with_null_rows(self):
        """Null rows are filtered out before union."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        polys = [box(0, 0, 2, 2), None, box(1, 1, 3, 3)]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        # Expected: union of box(0,0,2,2) and box(1,1,3,3)
        valid_polys = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
        arr = np.empty(len(valid_polys), dtype=object)
        arr[:] = valid_polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result_geom, expected)

    def test_with_grid_size(self):
        """grid_size parameter snaps coordinates before union."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        polys = [box(0, 0, 1.5, 1.5), box(1, 1, 2.5, 2.5)]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned, grid_size=1.0)
        result_geom = _to_shapely(result)
        # With grid_size=1.0, coordinates snap to integers.
        # Just verify it's a valid geometry with non-zero area.
        assert result_geom.area > 0

    def test_many_polygons(self):
        """Tree reduction with many polygons (tests multiple levels)."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        # Create 8 overlapping polygons to exercise 3 levels of tree reduction.
        polys = [box(i, 0, i + 2, 2) for i in range(8)]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result_geom, expected)

    def test_odd_count_polygons(self):
        """Odd number of polygons (tests carry-forward of unpaired element)."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        polys = [box(i, 0, i + 2, 2) for i in range(5)]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result_geom, expected)


# ---------------------------------------------------------------------------
# coverage_union_all_gpu tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestCoverageUnionAllGPU:
    """Tests for coverage_union_all_gpu_owned."""

    def test_basic_coverage_union(self):
        """Coverage union of non-overlapping tiles matches Shapely."""
        from vibespatial.constructive.union_all import coverage_union_all_gpu_owned

        # Non-overlapping tiles (coverage property).
        polys = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)]
        owned = from_shapely_geometries(polys)
        result = coverage_union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.coverage_union_all(arr)

        assert _geom_equiv(result_geom, expected), (
            f"GPU coverage_union_all != Shapely\n"
            f"  GPU area={result_geom.area}, expected area={expected.area}"
        )

    def test_empty_input(self):
        """Empty input returns an empty geometry."""
        from vibespatial.constructive.union_all import coverage_union_all_gpu_owned

        owned = from_shapely_geometries([])
        result = coverage_union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert result_geom.is_empty

    def test_single_tile(self):
        """Single tile returns identity."""
        from vibespatial.constructive.union_all import coverage_union_all_gpu_owned

        poly = box(0, 0, 10, 10)
        owned = from_shapely_geometries([poly])
        result = coverage_union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert _geom_equiv(result_geom, poly)


# ---------------------------------------------------------------------------
# intersection_all_gpu tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestIntersectionAllGPU:
    """Tests for intersection_all_gpu_owned."""

    def test_basic_intersection(self):
        """Intersection of overlapping polygons matches Shapely."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        polys = [box(0, 0, 3, 3), box(1, 1, 4, 4), box(2, 2, 5, 5)]
        owned = from_shapely_geometries(polys)
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.intersection_all(arr)

        assert _geom_equiv(result_geom, expected), (
            f"GPU intersection_all != Shapely\n"
            f"  GPU area={result_geom.area}, expected area={expected.area}"
        )

    def test_no_common_region(self):
        """Intersection of non-overlapping polygons is empty."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        polys = [box(0, 0, 1, 1), box(2, 2, 3, 3)]
        owned = from_shapely_geometries(polys)
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        assert result_geom.is_empty or result_geom.area < 1e-10

    def test_early_termination(self):
        """Early termination works when intersection becomes empty mid-way."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        # First two polygons are disjoint, so intersection is empty.
        # Third polygon is huge -- it should never be processed.
        polys = [box(0, 0, 1, 1), box(5, 5, 6, 6), box(-100, -100, 100, 100)]
        owned = from_shapely_geometries(polys)
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        assert result_geom.is_empty or result_geom.area < 1e-10

    def test_empty_input(self):
        """Empty input returns an empty geometry."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        owned = from_shapely_geometries([])
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert result_geom.is_empty

    def test_single_row(self):
        """Single-row input returns the input geometry."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        poly = box(0, 0, 10, 10)
        owned = from_shapely_geometries([poly])
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert _geom_equiv(result_geom, poly)

    def test_with_null_rows(self):
        """Null rows are skipped in intersection."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        polys = [box(0, 0, 3, 3), None, box(1, 1, 4, 4)]
        owned = from_shapely_geometries(polys)
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        # Expected: intersection(box(0,0,3,3), box(1,1,4,4)) = box(1,1,3,3)
        expected = shapely.intersection(box(0, 0, 3, 3), box(1, 1, 4, 4))
        assert _geom_equiv(result_geom, expected)


# ---------------------------------------------------------------------------
# unary_union_gpu tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestUnaryUnionGPU:
    """Tests for unary_union_gpu_owned."""

    def test_basic_unary_union(self):
        """Unary union delegates to union_all_gpu and matches Shapely."""
        from vibespatial.constructive.union_all import unary_union_gpu_owned

        polys = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
        owned = from_shapely_geometries(polys)
        result = unary_union_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result_geom, expected)

    def test_empty_input(self):
        """Empty input returns an empty geometry."""
        from vibespatial.constructive.union_all import unary_union_gpu_owned

        owned = from_shapely_geometries([])
        result = unary_union_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert result_geom.is_empty


# ---------------------------------------------------------------------------
# GeometryArray integration tests (dispatch wiring)
# ---------------------------------------------------------------------------


@requires_gpu
class TestGeometryArrayDispatch:
    """Verify that GeometryArray.union_all/intersection_all dispatch to GPU."""

    def test_union_all_unary_dispatch(self):
        """GeometryArray.union_all(method='unary') dispatches to GPU."""
        from vibespatial.api.geometry_array import GeometryArray

        polys = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
        owned = from_shapely_geometries(polys)
        ga = GeometryArray.from_owned(owned)
        result = ga.union_all(method="unary")

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result, expected)

    def test_union_all_coverage_dispatch(self):
        """GeometryArray.union_all(method='coverage') dispatches to GPU."""
        from vibespatial.api.geometry_array import GeometryArray

        polys = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
        owned = from_shapely_geometries(polys)
        ga = GeometryArray.from_owned(owned)
        result = ga.union_all(method="coverage")

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.coverage_union_all(arr)

        assert _geom_equiv(result, expected)

    def test_intersection_all_dispatch(self):
        """GeometryArray.intersection_all() dispatches to GPU."""
        from vibespatial.api.geometry_array import GeometryArray

        polys = [box(0, 0, 3, 3), box(1, 1, 4, 4)]
        owned = from_shapely_geometries(polys)
        ga = GeometryArray.from_owned(owned)
        result = ga.intersection_all()

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.intersection_all(arr)

        assert _geom_equiv(result, expected)
