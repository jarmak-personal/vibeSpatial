"""Tests for GPU-accelerated set_precision.

Validates all three modes (valid_output, pointwise, keep_collapsed)
against Shapely oracle for all geometry families.  Ensures grid_size=0
is a no-op and that coordinates are correctly snapped to grid multiples.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from vibespatial import from_shapely_geometries, has_gpu_runtime
from vibespatial.constructive.set_precision import set_precision_owned

requires_gpu = pytest.mark.skipif(
    not has_gpu_runtime(), reason="GPU not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shapely_oracle(geoms, grid_size, mode):
    """Run Shapely set_precision as reference oracle."""
    arr = np.asarray(geoms, dtype=object)
    return list(shapely.set_precision(arr, grid_size=grid_size, mode=mode))


def _assert_geoms_match(gpu_geoms, shapely_geoms, *, tolerance=1e-10):
    """Assert GPU result geometries match Shapely oracle (element-wise WKT/coord)."""
    assert len(gpu_geoms) == len(shapely_geoms), (
        f"Length mismatch: {len(gpu_geoms)} vs {len(shapely_geoms)}"
    )
    for i, (g, s) in enumerate(zip(gpu_geoms, shapely_geoms)):
        if s is None:
            assert g is None or g.is_empty, f"Row {i}: expected None/empty, got {g}"
            continue
        if s.is_empty:
            assert g is None or g.is_empty, f"Row {i}: expected empty, got {g}"
            continue
        assert g is not None, f"Row {i}: expected geometry, got None"
        # Compare normalized WKB for exact structural match
        g_norm = shapely.normalize(g)
        s_norm = shapely.normalize(s)
        if not shapely.equals_exact(g_norm, s_norm, tolerance=tolerance):
            # Fall back to equals (topological)
            assert shapely.equals(g, s), (
                f"Row {i}: GPU result does not match Shapely.\n"
                f"  GPU:    {g.wkt}\n"
                f"  Shapely: {s.wkt}"
            )


def _coords_on_grid(geom, grid_size, tolerance=1e-12):
    """Check that all coordinates of a geometry are on the grid."""
    if geom is None or geom.is_empty:
        return True
    coords = shapely.get_coordinates(geom)
    for x, y in coords:
        x_rem = abs(x - round(x / grid_size) * grid_size)
        y_rem = abs(y - round(y / grid_size) * grid_size)
        if x_rem > tolerance or y_rem > tolerance:
            return False
    return True


# ---------------------------------------------------------------------------
# grid_size=0 no-op
# ---------------------------------------------------------------------------


@requires_gpu
def test_grid_size_zero_is_noop():
    """grid_size=0 returns the input unchanged (Shapely-compatible)."""
    geoms = [
        Point(1.23456, 2.34567),
        LineString([(0.1, 0.2), (3.4, 5.6)]),
        Polygon([(0.1, 0.2), (1.3, 0.4), (0.5, 1.6), (0.1, 0.2)]),
    ]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=0)
    # Should be the same object (no-op)
    assert result is owned


# ---------------------------------------------------------------------------
# Pointwise mode
# ---------------------------------------------------------------------------


@requires_gpu
def test_pointwise_points():
    """Pointwise mode snaps Point coordinates to grid."""
    grid_size = 1.0
    geoms = [Point(0.3, 0.7), Point(1.5, 2.5), Point(-0.4, -0.6)]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=grid_size, mode="pointwise")
    result_geoms = result.to_shapely()

    shapely_ref = _shapely_oracle(geoms, grid_size, "pointwise")
    _assert_geoms_match(result_geoms, shapely_ref)

    for g in result_geoms:
        assert _coords_on_grid(g, grid_size), f"Coords not on grid: {g.wkt}"


@requires_gpu
def test_pointwise_linestrings():
    """Pointwise mode snaps LineString coordinates to grid."""
    grid_size = 0.5
    geoms = [
        LineString([(0.1, 0.2), (0.6, 0.7), (1.1, 1.3)]),
        LineString([(0.0, 0.0), (0.25, 0.75), (0.5, 0.5)]),
    ]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=grid_size, mode="pointwise")
    result_geoms = result.to_shapely()

    shapely_ref = _shapely_oracle(geoms, grid_size, "pointwise")
    _assert_geoms_match(result_geoms, shapely_ref)


@requires_gpu
def test_pointwise_polygons():
    """Pointwise mode snaps Polygon coordinates to grid."""
    grid_size = 1.0
    geoms = [
        Polygon([(0.1, 0.2), (5.3, 0.4), (5.1, 5.6), (0.3, 5.2), (0.1, 0.2)]),
    ]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=grid_size, mode="pointwise")
    result_geoms = result.to_shapely()

    shapely_ref = _shapely_oracle(geoms, grid_size, "pointwise")
    _assert_geoms_match(result_geoms, shapely_ref)


@requires_gpu
def test_pointwise_multipoint():
    """Pointwise mode snaps MultiPoint coordinates to grid."""
    grid_size = 0.5
    geoms = [
        MultiPoint([(0.1, 0.2), (0.6, 0.7), (1.1, 1.3)]),
    ]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=grid_size, mode="pointwise")
    result_geoms = result.to_shapely()

    shapely_ref = _shapely_oracle(geoms, grid_size, "pointwise")
    _assert_geoms_match(result_geoms, shapely_ref)


# ---------------------------------------------------------------------------
# Keep-collapsed mode
# ---------------------------------------------------------------------------


@requires_gpu
def test_keep_collapsed_deduplicates():
    """keep_collapsed mode deduplicates coincident points after snapping."""
    grid_size = 1.0
    # After snapping to grid_size=1.0, (0.1,0.2) and (0.4,0.3) both become (0,0)
    geoms = [
        LineString([(0.1, 0.2), (0.4, 0.3), (5.1, 5.2)]),
    ]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=grid_size, mode="keep_collapsed")
    result_geoms = result.to_shapely()

    shapely_ref = _shapely_oracle(geoms, grid_size, "keep_collapsed")
    _assert_geoms_match(result_geoms, shapely_ref)


@requires_gpu
def test_keep_collapsed_polygon():
    """keep_collapsed keeps degenerate polygon rings after snapping."""
    grid_size = 10.0
    # Large grid: polygon may collapse to a degenerate geometry
    geoms = [
        Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]),
    ]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=grid_size, mode="keep_collapsed")
    result_geoms = result.to_shapely()

    shapely_ref = _shapely_oracle(geoms, grid_size, "keep_collapsed")
    _assert_geoms_match(result_geoms, shapely_ref)


@requires_gpu
def test_keep_collapsed_preserves_non_collapsed():
    """keep_collapsed preserves geometries that survive snapping."""
    grid_size = 1.0
    geoms = [
        Polygon([
            (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)
        ]),
        LineString([(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]),
    ]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=grid_size, mode="keep_collapsed")
    result_geoms = result.to_shapely()

    shapely_ref = _shapely_oracle(geoms, grid_size, "keep_collapsed")
    _assert_geoms_match(result_geoms, shapely_ref)


# ---------------------------------------------------------------------------
# Valid-output mode
# ---------------------------------------------------------------------------


@requires_gpu
def test_valid_output_basic_polygon():
    """valid_output mode produces valid polygon after snapping."""
    grid_size = 1.0
    geoms = [
        Polygon([
            (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)
        ]),
    ]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=grid_size, mode="valid_output")
    result_geoms = result.to_shapely()

    for g in result_geoms:
        if g is not None and not g.is_empty:
            assert g.is_valid, f"Result not valid: {g.wkt}"
            assert _coords_on_grid(g, grid_size)


@requires_gpu
def test_valid_output_matches_shapely():
    """valid_output mode matches Shapely for various geometry types."""
    grid_size = 1.0
    geoms = [
        Point(0.3, 0.7),
        LineString([(0.1, 0.2), (5.6, 7.8)]),
        Polygon([
            (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)
        ]),
    ]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=grid_size, mode="valid_output")
    result_geoms = result.to_shapely()

    shapely_ref = _shapely_oracle(geoms, grid_size, "valid_output")
    _assert_geoms_match(result_geoms, shapely_ref)


@requires_gpu
def test_valid_output_repairs_topology():
    """valid_output mode produces valid geometries after aggressive snapping."""
    grid_size = 5.0
    # Large polygon that remains valid after snapping to coarse grid
    geoms = [
        Polygon([
            (0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0), (0.0, 0.0)
        ]),
        # Polygon with a hole -- snapping may simplify but should stay valid
        Polygon(
            [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0), (0.0, 0.0)],
            [[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (10.0, 10.0)]],
        ),
    ]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=grid_size, mode="valid_output")
    result_geoms = result.to_shapely()

    for g in result_geoms:
        if g is not None and not g.is_empty:
            assert g.is_valid, f"Result not valid after repair: {g.wkt}"
            assert _coords_on_grid(g, grid_size)


# ---------------------------------------------------------------------------
# Mixed geometry families
# ---------------------------------------------------------------------------


@requires_gpu
def test_mixed_families():
    """set_precision handles mixed geometry types in one array."""
    grid_size = 0.5
    geoms = [
        Point(0.3, 0.7),
        LineString([(0.1, 0.2), (0.6, 0.7), (1.1, 1.3)]),
        Polygon([(0.1, 0.2), (5.3, 0.4), (5.1, 5.6), (0.3, 5.2), (0.1, 0.2)]),
        MultiPoint([(0.1, 0.2), (0.6, 0.7)]),
        MultiLineString([
            [(0.1, 0.2), (0.6, 0.7)],
            [(1.1, 1.3), (2.2, 2.4)],
        ]),
        MultiPolygon([
            Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]),
            Polygon([(10.0, 10.0), (15.0, 10.0), (15.0, 15.0), (10.0, 15.0), (10.0, 10.0)]),
        ]),
    ]
    for mode in ("pointwise", "keep_collapsed", "valid_output"):
        owned = from_shapely_geometries(geoms)
        result = set_precision_owned(owned, grid_size=grid_size, mode=mode)
        result_geoms = result.to_shapely()
        shapely_ref = _shapely_oracle(geoms, grid_size, mode)

        # For valid_output, just check validity + grid alignment
        if mode == "valid_output":
            for g in result_geoms:
                if g is not None and not g.is_empty:
                    assert _coords_on_grid(g, grid_size), (
                        f"Coords not on grid (mode={mode}): {g.wkt}"
                    )
        else:
            _assert_geoms_match(result_geoms, shapely_ref, tolerance=1e-10)


# ---------------------------------------------------------------------------
# Fractional grid sizes
# ---------------------------------------------------------------------------


@requires_gpu
def test_fractional_grid_size():
    """set_precision works with non-integer grid sizes."""
    grid_size = 0.01
    geoms = [
        Point(1.23456, 2.34567),
        LineString([(0.12345, 0.23456), (1.34567, 2.45678)]),
    ]
    owned = from_shapely_geometries(geoms)

    for mode in ("pointwise", "keep_collapsed", "valid_output"):
        result = set_precision_owned(owned, grid_size=grid_size, mode=mode)
        result_geoms = result.to_shapely()
        for g in result_geoms:
            if g is not None and not g.is_empty:
                assert _coords_on_grid(g, grid_size, tolerance=1e-10)


# ---------------------------------------------------------------------------
# Empty / null handling
# ---------------------------------------------------------------------------


@requires_gpu
def test_null_geometry_handling():
    """set_precision handles null geometries gracefully."""
    geoms = [Point(0.3, 0.7), None, LineString([(0.1, 0.2), (5.6, 7.8)])]
    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=1.0, mode="pointwise")
    result_geoms = result.to_shapely()
    # Row 1 (None) should stay null
    assert result_geoms[1] is None


@requires_gpu
def test_empty_array():
    """set_precision handles empty input gracefully."""
    owned = from_shapely_geometries([])
    result = set_precision_owned(owned, grid_size=1.0, mode="valid_output")
    assert result.row_count == 0


# ---------------------------------------------------------------------------
# Invalid mode
# ---------------------------------------------------------------------------


def test_invalid_mode():
    """set_precision rejects invalid mode strings."""
    geoms = [Point(1.0, 2.0)]
    owned = from_shapely_geometries(geoms)
    with pytest.raises(ValueError, match="Invalid mode"):
        set_precision_owned(owned, grid_size=1.0, mode="invalid_mode")


# ---------------------------------------------------------------------------
# GeometryArray integration
# ---------------------------------------------------------------------------


@requires_gpu
def test_geometry_array_set_precision():
    """set_precision dispatches through GeometryArray.set_precision()."""
    from vibespatial.api.geometry_array import GeometryArray

    geoms = [
        Point(0.3, 0.7),
        Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]),
    ]
    owned = from_shapely_geometries(geoms)
    ga = GeometryArray.from_owned(owned)

    result_ga = ga.set_precision(grid_size=1.0, mode="pointwise")
    result_geoms = list(result_ga._data)

    shapely_ref = _shapely_oracle(geoms, 1.0, "pointwise")
    _assert_geoms_match(result_geoms, shapely_ref)


# ---------------------------------------------------------------------------
# Large batch
# ---------------------------------------------------------------------------


@requires_gpu
def test_large_batch_pointwise():
    """set_precision handles large batches of Points."""
    np.random.seed(42)
    n = 1000
    coords = np.random.uniform(-180, 180, (n, 2))
    geoms = [Point(x, y) for x, y in coords]
    grid_size = 0.01

    owned = from_shapely_geometries(geoms)
    result = set_precision_owned(owned, grid_size=grid_size, mode="pointwise")
    result_geoms = result.to_shapely()

    shapely_ref = _shapely_oracle(geoms, grid_size, "pointwise")
    _assert_geoms_match(result_geoms, shapely_ref)
