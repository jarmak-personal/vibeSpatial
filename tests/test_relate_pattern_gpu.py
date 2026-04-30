"""Tests for GPU-accelerated DE-9IM pattern matching.

Verifies correctness of ``relate_pattern_match`` against Shapely oracle for:
    - All pattern characters: T, F, *, 0, 1, 2
    - Point-Point, Point-Line, Point-Polygon family pairs
    - Non-point families (Polygon-Polygon via Shapely fallback)
    - Null geometry handling (nulls produce False)
    - Empty arrays
    - Pattern validation (bad length, bad characters)
    - GeometryArray.relate_pattern() wiring
    - Scalar broadcast (single geometry against array)
    - Batch size for GPU parallelism
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)

from vibespatial import ExecutionMode, has_gpu_runtime
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.predicates.relate import relate_pattern_match
from vibespatial.runtime.fallbacks import (
    STRICT_NATIVE_ENV_VAR,
    StrictNativeFallbackError,
    clear_fallback_events,
    get_fallback_events,
)

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")


def _assert_pattern_matches_shapely(
    left_geoms,
    right_geoms,
    pattern,
    *,
    label="",
    dispatch_mode=ExecutionMode.AUTO,
):
    """Assert GPU relate_pattern produces the same results as Shapely."""
    left_np = np.array(left_geoms, dtype=object)
    right_np = np.array(right_geoms, dtype=object)
    expected = shapely.relate_pattern(left_np, right_np, pattern)

    left_owned = from_shapely_geometries(list(left_geoms))
    right_owned = from_shapely_geometries(list(right_geoms))
    result = relate_pattern_match(
        left_owned, right_owned, pattern, dispatch_mode=dispatch_mode,
    )

    assert result.dtype == bool, f"{label}: dtype should be bool, got {result.dtype}"
    assert len(result) == len(expected), f"{label}: length mismatch"
    np.testing.assert_array_equal(
        result, expected,
        err_msg=f"{label}: pattern={pattern!r}",
    )


# ---------------------------------------------------------------------------
# Pattern character semantics
# ---------------------------------------------------------------------------

@requires_gpu
class TestPatternCharacters:
    """Verify each DE-9IM pattern character matches correctly."""

    def test_star_matches_anything(self):
        """'*' matches any DE-9IM character."""
        _assert_pattern_matches_shapely(
            [Point(1, 1)],
            [Point(1, 1)],
            "*********",
            label="star_all",
        )

    def test_T_matches_non_F(self):
        """'T' matches 0, 1, or 2 but not F."""
        # Point inside polygon: DE-9IM = "0FFFFF212"
        # T at pos 0 matches '0' (non-F) -> True
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 2)],
            [poly],
            "T********",
            label="T_matches_0",
        )

    def test_T_does_not_match_F(self):
        """'T' does not match 'F'."""
        # Point inside polygon: DE-9IM = "0FFFFF212"
        # T at pos 1 should not match 'F' -> False
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 2)],
            [poly],
            "*T*******",
            label="T_no_match_F",
        )

    def test_F_matches_only_F(self):
        """'F' matches only 'F'."""
        # Point inside polygon: DE-9IM = "0FFFFF212"
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 2)],
            [poly],
            "*F*******",
            label="F_matches_F",
        )

    def test_F_does_not_match_dimension(self):
        """'F' does not match dimension characters."""
        # Point inside polygon: DE-9IM = "0FFFFF212"
        # F at pos 0 should not match '0' -> False
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 2)],
            [poly],
            "F********",
            label="F_no_match_dim",
        )

    def test_exact_0_match(self):
        """'0' matches only '0'."""
        # Point inside polygon: DE-9IM = "0FFFFF212"
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 2)],
            [poly],
            "0********",
            label="exact_0",
        )

    def test_exact_1_match(self):
        """'1' matches only '1'."""
        # Point inside polygon: DE-9IM = "0FFFFF212"
        # Position 7 is '1' (EB dimension)
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 2)],
            [poly],
            "*******1*",
            label="exact_1",
        )

    def test_exact_2_match(self):
        """'2' matches only '2'."""
        # Point inside polygon: DE-9IM = "0FFFFF212"
        # Position 6 is '2' (EI dimension)
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 2)],
            [poly],
            "******2**",
            label="exact_2",
        )

    def test_exact_full_match(self):
        """Exact 9-char match against known DE-9IM string."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 2)],
            [poly],
            "0FFFFF212",
            label="exact_full",
        )

    def test_exact_full_no_match(self):
        """Exact 9-char pattern that doesn't match."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 2)],
            [poly],
            "FF0FFF212",
            label="exact_full_no",
        )


# ---------------------------------------------------------------------------
# Standard predicate patterns
# ---------------------------------------------------------------------------

@requires_gpu
class TestStandardPredicatePatterns:
    """Test well-known predicate patterns from the DE-9IM specification."""

    def test_within_pattern(self):
        """'T*F**F***' is the within predicate pattern."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        # Point inside polygon -> within=True
        # Point outside polygon -> within=False
        _assert_pattern_matches_shapely(
            [Point(2, 2), Point(5, 5)],
            [poly, poly],
            "T*F**F***",
            label="within_pattern",
        )

    def test_contains_pattern(self):
        """'T*****FF*' is the contains predicate pattern."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [poly, poly],
            [Point(2, 2), Point(5, 5)],
            "T*****FF*",
            label="contains_pattern",
        )

    def test_intersects_pattern(self):
        """'T********', '*T*******', '***T*****', '****T****' patterns for intersects."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        # Use the standard Shapely intersects DE-9IM pattern
        _assert_pattern_matches_shapely(
            [Point(2, 2), Point(5, 5)],
            [poly, poly],
            "T********",
            label="intersects_T",
        )

    def test_disjoint_pattern(self):
        """'FF*FF****' is the disjoint predicate pattern."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 2), Point(5, 5)],
            [poly, poly],
            "FF*FF****",
            label="disjoint_pattern",
        )

    def test_touches_pattern(self):
        """Test touches pattern for point on boundary."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 0), Point(2, 2), Point(5, 5)],
            [poly, poly, poly],
            "F0FFFF212",
            label="touches_boundary",
        )


# ---------------------------------------------------------------------------
# Family combinations
# ---------------------------------------------------------------------------

@requires_gpu
class TestFamilyCombinations:
    """Test pattern matching across different geometry family pairs."""

    def test_point_point(self):
        """Point-Point pairs."""
        _assert_pattern_matches_shapely(
            [Point(1, 1), Point(1, 1)],
            [Point(1, 1), Point(3, 3)],
            "0********",
            label="point_point",
        )

    def test_point_linestring(self):
        """Point-LineString pairs."""
        line = LineString([(0, 0), (10, 0)])
        _assert_pattern_matches_shapely(
            [Point(5, 0), Point(0, 5)],
            [line, line],
            "0********",
            label="point_line",
        )

    def test_point_polygon(self):
        """Point-Polygon pairs."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_pattern_matches_shapely(
            [Point(2, 2), Point(5, 5)],
            [poly, poly],
            "0FFFFF212",
            label="point_polygon",
        )

    def test_polygon_polygon_fallback(self):
        """Polygon-Polygon falls back to Shapely, still correct pattern match."""
        poly_a = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        poly_b = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
        _assert_pattern_matches_shapely(
            [poly_a],
            [poly_b],
            "2********",
            label="polygon_polygon",
        )

    def test_line_line_fallback(self):
        """LineString-LineString falls back to Shapely, still correct pattern match."""
        line_a = LineString([(0, 0), (2, 2)])
        line_b = LineString([(0, 2), (2, 0)])
        _assert_pattern_matches_shapely(
            [line_a],
            [line_b],
            "0********",
            label="line_line",
        )

    def test_point_multilinestring(self):
        """Point-MultiLineString pair."""
        mls = MultiLineString([[(0, 0), (2, 0)], [(3, 0), (5, 0)]])
        _assert_pattern_matches_shapely(
            [Point(1, 0)],
            [mls],
            "0********",
            label="point_mls",
        )

    def test_point_multipolygon(self):
        """Point-MultiPolygon pair."""
        mp = MultiPolygon([
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        ])
        _assert_pattern_matches_shapely(
            [Point(1, 1)],
            [mp],
            "0FFFFF212",
            label="point_mpoly",
        )


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------

@requires_gpu
class TestNullHandling:
    """Null geometries produce False (not True or error)."""

    def test_left_null(self):
        """Null left geometry produces False."""
        left_owned = from_shapely_geometries([None])
        right_owned = from_shapely_geometries([Point(1, 2)])
        result = relate_pattern_match(left_owned, right_owned, "*********")
        assert result[0] is np.False_ or result[0] == False  # noqa: E712

    def test_right_null(self):
        """Null right geometry produces False."""
        left_owned = from_shapely_geometries([Point(1, 2)])
        right_owned = from_shapely_geometries([None])
        result = relate_pattern_match(left_owned, right_owned, "*********")
        assert result[0] is np.False_ or result[0] == False  # noqa: E712

    def test_both_null(self):
        """Both null produces False."""
        left_owned = from_shapely_geometries([None])
        right_owned = from_shapely_geometries([None])
        result = relate_pattern_match(left_owned, right_owned, "*********")
        assert result[0] is np.False_ or result[0] == False  # noqa: E712

    def test_mixed_null_and_valid(self):
        """Mix of null and valid geometries."""
        left_owned = from_shapely_geometries([Point(1, 1), None, Point(3, 3)])
        right_owned = from_shapely_geometries([Point(1, 1), Point(2, 2), None])

        # "0FFFFFFF2" is the exact Point-Point equal DE-9IM
        result = relate_pattern_match(left_owned, right_owned, "0FFFFFFF2")
        assert result[0] == True  # equal points match  # noqa: E712
        assert result[1] == False  # left null -> False  # noqa: E712
        assert result[2] == False  # right null -> False  # noqa: E712


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@requires_gpu
class TestEdgeCases:
    def test_empty_arrays(self):
        """Empty input arrays produce empty boolean output."""
        left_owned = from_shapely_geometries([])
        right_owned = from_shapely_geometries([])
        result = relate_pattern_match(left_owned, right_owned, "*********")
        assert len(result) == 0
        assert result.dtype == bool

    def test_row_count_mismatch(self):
        """Mismatched row counts raise ValueError."""
        left_owned = from_shapely_geometries([Point(1, 1)])
        right_owned = from_shapely_geometries([Point(1, 1), Point(2, 2)])
        with pytest.raises(ValueError, match="same row count"):
            relate_pattern_match(left_owned, right_owned, "*********")


# ---------------------------------------------------------------------------
# Pattern validation
# ---------------------------------------------------------------------------

class TestPatternValidation:
    """Pattern validation should reject invalid patterns early."""

    def test_too_short(self):
        """Pattern shorter than 9 characters."""
        left_owned = from_shapely_geometries([Point(1, 1)])
        right_owned = from_shapely_geometries([Point(1, 1)])
        with pytest.raises(ValueError, match="exactly 9 characters"):
            relate_pattern_match(left_owned, right_owned, "T*F")

    def test_too_long(self):
        """Pattern longer than 9 characters."""
        left_owned = from_shapely_geometries([Point(1, 1)])
        right_owned = from_shapely_geometries([Point(1, 1)])
        with pytest.raises(ValueError, match="exactly 9 characters"):
            relate_pattern_match(left_owned, right_owned, "T*F**F***X")

    def test_invalid_character(self):
        """Pattern with invalid character."""
        left_owned = from_shapely_geometries([Point(1, 1)])
        right_owned = from_shapely_geometries([Point(1, 1)])
        with pytest.raises(ValueError, match="Invalid character"):
            relate_pattern_match(left_owned, right_owned, "X********")

    def test_lowercase_t(self):
        """Lowercase 't' is not valid (must be uppercase 'T')."""
        left_owned = from_shapely_geometries([Point(1, 1)])
        right_owned = from_shapely_geometries([Point(1, 1)])
        with pytest.raises(ValueError, match="Invalid character"):
            relate_pattern_match(left_owned, right_owned, "t********")

    def test_lowercase_f(self):
        """Lowercase 'f' is not valid (must be uppercase 'F')."""
        left_owned = from_shapely_geometries([Point(1, 1)])
        right_owned = from_shapely_geometries([Point(1, 1)])
        with pytest.raises(ValueError, match="Invalid character"):
            relate_pattern_match(left_owned, right_owned, "f********")

    def test_digit_3_invalid(self):
        """'3' is not a valid DE-9IM dimension."""
        left_owned = from_shapely_geometries([Point(1, 1)])
        right_owned = from_shapely_geometries([Point(1, 1)])
        with pytest.raises(ValueError, match="Invalid character"):
            relate_pattern_match(left_owned, right_owned, "3********")


# ---------------------------------------------------------------------------
# GeometryArray.relate_pattern() wiring
# ---------------------------------------------------------------------------

@requires_gpu
class TestGeometryArrayWiring:
    def test_relate_pattern_via_geometry_array(self):
        """GeometryArray.relate_pattern() dispatches through GPU path."""
        left_geoms = np.array([Point(1, 1), Point(2, 2)], dtype=object)
        right_geoms = np.array([Point(1, 1), Point(3, 3)], dtype=object)
        expected = shapely.relate_pattern(left_geoms, right_geoms, "0FFFFFFF2")

        ga_left = GeometryArray(left_geoms)
        ga_left.to_owned()
        ga_right = GeometryArray(right_geoms)

        result = ga_left.relate_pattern(ga_right, "0FFFFFFF2")
        np.testing.assert_array_equal(result, expected)

    def test_relate_pattern_with_polygon(self):
        """GeometryArray.relate_pattern() with polygon pair."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        left_geoms = np.array([Point(2, 2), Point(5, 5)], dtype=object)
        right_geoms = np.array([poly, poly], dtype=object)
        expected = shapely.relate_pattern(left_geoms, right_geoms, "T*F**F***")

        ga_left = GeometryArray(left_geoms)
        ga_left.to_owned()
        ga_right = GeometryArray(right_geoms)

        result = ga_left.relate_pattern(ga_right, "T*F**F***")
        np.testing.assert_array_equal(result, expected)

    def test_scalar_broadcast_stays_native_in_strict(self, monkeypatch):
        """Scalar broadcast uses an owned native view before public export."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        left_geoms = np.array([Point(2, 2), Point(5, 5)], dtype=object)
        expected = shapely.relate_pattern(left_geoms, poly, "T*F**F***")

        ga_left = GeometryArray(left_geoms)
        ga_left.to_owned()
        clear_fallback_events()
        monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

        result = ga_left.relate_pattern(poly, "T*F**F***")

        np.testing.assert_array_equal(result, expected)
        assert get_fallback_events(clear=True) == []

    def test_relate_pattern_strict_native_non_point_decline_is_not_swallowed(self, monkeypatch):
        """Strict native must not be hidden by GeometryArray's Shapely fallback."""
        left_geoms = np.array([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])], dtype=object)
        right_geoms = np.array([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])], dtype=object)

        ga_left = GeometryArray(left_geoms)
        ga_left.to_owned()
        ga_right = GeometryArray(right_geoms)
        monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

        with pytest.raises(StrictNativeFallbackError, match="non-point family"):
            ga_left.relate_pattern(ga_right, "T********")


# ---------------------------------------------------------------------------
# Batch tests for GPU parallelism
# ---------------------------------------------------------------------------

@requires_gpu
class TestBatchRelatePattern:
    def test_batch_point_point(self):
        """Batch of 1000 point-point pairs with within pattern."""
        rng = np.random.default_rng(42)
        n = 1000
        left_pts = [Point(rng.random(), rng.random()) for _ in range(n)]
        right_pts = [Point(rng.random(), rng.random()) for _ in range(n)]
        # Make every 10th pair identical.
        for i in range(0, n, 10):
            right_pts[i] = left_pts[i]
        _assert_pattern_matches_shapely(
            left_pts, right_pts, "0FFFFFFF2", label="batch_pp",
        )

    def test_batch_point_polygon(self):
        """Batch of point-polygon pairs with within pattern."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        rng = np.random.default_rng(99)
        n = 100
        points = [Point(rng.uniform(-2, 12), rng.uniform(-2, 12)) for _ in range(n)]
        _assert_pattern_matches_shapely(
            points, [poly] * n, "T*F**F***", label="batch_point_polygon",
        )

    def test_batch_mixed_families(self):
        """Mixed families with a common pattern."""
        line = LineString([(0, 0), (10, 0)])
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        left = [
            Point(5, 0),   # on line interior
            Point(0, 0),   # on line boundary
            Point(5, 5),   # in polygon interior
            Point(5, 0),   # on polygon boundary
            Point(15, 15), # exterior to polygon
        ]
        right = [line, line, poly, poly, poly]
        # T at position 0 matches any non-F first cell
        _assert_pattern_matches_shapely(
            left, right, "T********", label="batch_mixed",
        )


# ---------------------------------------------------------------------------
# Explicit GPU dispatch
# ---------------------------------------------------------------------------

@requires_gpu
class TestExplicitGPUDispatch:
    """Force GPU dispatch to verify the GPU path is actually exercised."""

    def test_gpu_forced_point_point(self):
        """Point-Point with explicit GPU dispatch."""
        _assert_pattern_matches_shapely(
            [Point(1, 2), Point(3, 4), Point(0, 0)],
            [Point(1, 2), Point(5, 6), Point(0, 0)],
            "0FFFFFFF2",
            label="gpu_forced_pp",
            dispatch_mode=ExecutionMode.GPU,
        )

    def test_gpu_forced_point_polygon(self):
        """Point-Polygon with explicit GPU dispatch."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        _assert_pattern_matches_shapely(
            [Point(5, 5), Point(5, 0), Point(15, 15)],
            [poly, poly, poly],
            "T*F**F***",
            label="gpu_forced_ppoly",
            dispatch_mode=ExecutionMode.GPU,
        )

    def test_gpu_forced_mixed_with_fallback(self):
        """Mixed Point-* and Polygon-Polygon with GPU dispatch."""
        poly_a = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        poly_b = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
        _assert_pattern_matches_shapely(
            [Point(1, 1), poly_a, Point(2, 2)],
            [Point(1, 1), poly_b, poly_a],
            "T********",
            label="gpu_forced_mixed",
            dispatch_mode=ExecutionMode.GPU,
        )
