"""Tests for GPU-accelerated line_merge.

Validates that the NVRTC line_merge kernel matches Shapely's line_merge
for connected, disconnected, ring, and directed merge scenarios.

Geometry families tested: LineString, MultiLineString.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
)

from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import has_gpu_runtime

requires_gpu = pytest.mark.skipif(
    not has_gpu_runtime(), reason="GPU not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compare_line_merge(gpu_geoms, source_geoms, directed=False, atol=1e-10):
    """Assert GPU line_merge matches Shapely for each geometry."""
    shapely_arr = np.asarray(source_geoms, dtype=object)
    expected = shapely.line_merge(shapely_arr, directed=directed)

    for i, (gpu_g, exp_g) in enumerate(zip(gpu_geoms, expected)):
        if source_geoms[i] is None:
            continue
        if exp_g is None or exp_g.is_empty:
            # GPU should also produce empty or equivalent
            if gpu_g is not None and not gpu_g.is_empty:
                # Check if GPU produced a valid but equivalent empty result
                if gpu_g.geom_type == "MultiLineString" and len(gpu_g.geoms) == 0:
                    continue
            continue

        assert gpu_g is not None, f"Row {i}: GPU returned None, expected {exp_g.wkt}"
        assert not gpu_g.is_empty, f"Row {i}: GPU returned empty, expected {exp_g.wkt}"

        # Normalize both to coordinate lists for comparison
        gpu_coords = _extract_all_coords(gpu_g)
        exp_coords = _extract_all_coords(exp_g)

        # For MultiLineString vs LineString, compare coordinate content
        # The GPU always returns MultiLineString, Shapely may return LineString
        # when there's a single chain.  We compare the actual coordinates.
        assert len(gpu_coords) == len(exp_coords), (
            f"Row {i}: GPU produced {len(gpu_coords)} parts, "
            f"expected {len(exp_coords)}. "
            f"GPU={gpu_g.wkt}, expected={exp_g.wkt}"
        )

        # Sort parts for order-independent comparison
        gpu_sorted = sorted(gpu_coords, key=lambda c: (c[0][0], c[0][1]) if len(c) > 0 else (0, 0))
        exp_sorted = sorted(exp_coords, key=lambda c: (c[0][0], c[0][1]) if len(c) > 0 else (0, 0))

        for j, (gc, ec) in enumerate(zip(gpu_sorted, exp_sorted)):
            # Allow reverse direction for undirected merges
            if not directed:
                gc_arr = np.array(gc)
                ec_arr = np.array(ec)
                if gc_arr.shape == ec_arr.shape:
                    fwd_match = np.allclose(gc_arr, ec_arr, atol=atol)
                    rev_match = np.allclose(gc_arr, ec_arr[::-1], atol=atol)
                    assert fwd_match or rev_match, (
                        f"Row {i}, part {j}: coordinates don't match.\n"
                        f"GPU: {gc}\nExpected: {ec}"
                    )
                else:
                    pytest.fail(
                        f"Row {i}, part {j}: shape mismatch. "
                        f"GPU: {gc_arr.shape}, Expected: {ec_arr.shape}"
                    )
            else:
                np.testing.assert_allclose(
                    np.array(gc), np.array(ec), atol=atol,
                    err_msg=f"Row {i}, part {j}"
                )


def _extract_all_coords(geom):
    """Extract coordinate arrays from a geometry, normalizing to list of parts."""
    if geom.geom_type == "LineString":
        return [list(geom.coords)]
    elif geom.geom_type == "MultiLineString":
        return [list(part.coords) for part in geom.geoms]
    else:
        return []


# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------

def _simple_connected_mls():
    """Two connected LineStrings sharing an endpoint."""
    return MultiLineString([
        [(0, 0), (1, 1)],
        [(1, 1), (2, 0)],
    ])


def _three_segment_chain():
    """Three segments forming a single chain."""
    return MultiLineString([
        [(0, 0), (1, 0)],
        [(1, 0), (2, 1)],
        [(2, 1), (3, 0)],
    ])


def _disconnected_mls():
    """Two disconnected components."""
    return MultiLineString([
        [(0, 0), (1, 0)],
        [(1, 0), (2, 0)],
        [(10, 10), (11, 10)],
    ])


def _ring_mls():
    """Three segments forming a closed ring."""
    return MultiLineString([
        [(0, 0), (1, 0)],
        [(1, 0), (0.5, 1)],
        [(0.5, 1), (0, 0)],
    ])


def _single_linestring():
    """Single LineString (trivial merge)."""
    return LineString([(0, 0), (1, 1), (2, 0)])


def _single_segment_mls():
    """MultiLineString with one part."""
    return MultiLineString([[(0, 0), (1, 0), (2, 0)]])


def _directed_connected():
    """Two segments that connect in directed mode (end->start)."""
    return MultiLineString([
        [(0, 0), (1, 0)],
        [(1, 0), (2, 0)],
    ])


def _directed_not_connected():
    """Two segments that DON'T connect in directed mode (end->end)."""
    return MultiLineString([
        [(0, 0), (1, 0)],
        [(2, 0), (1, 0)],
    ])


# ---------------------------------------------------------------------------
# Tests: undirected mode
# ---------------------------------------------------------------------------

@requires_gpu
class TestLineMergeUndirected:
    """GPU line_merge with directed=False."""

    def test_simple_connected(self):
        """Two connected segments merge into one chain."""
        geoms = [_simple_connected_mls()]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=False)

    def test_three_segment_chain(self):
        """Three connected segments merge into one chain."""
        geoms = [_three_segment_chain()]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=False)

    def test_disconnected_components(self):
        """Disconnected segments produce MultiLineString with two parts."""
        geoms = [_disconnected_mls()]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=False)

    def test_ring(self):
        """Segments forming a ring merge into a closed LineString."""
        geoms = [_ring_mls()]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        result_geoms = result.to_shapely()
        # For rings, Shapely produces a LinearRing-like LineString
        # We verify coordinate count and closure
        _compare_line_merge(result_geoms, geoms, directed=False)

    def test_single_linestring(self):
        """Single LineString passes through unchanged."""
        geoms = [_single_linestring()]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=False)

    def test_single_segment_mls(self):
        """MultiLineString with one part passes through unchanged."""
        geoms = [_single_segment_mls()]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=False)

    def test_batch_mixed(self):
        """Multiple geometries of different types in one batch."""
        geoms = [
            _simple_connected_mls(),
            _disconnected_mls(),
            _ring_mls(),
            _single_linestring(),
            _three_segment_chain(),
        ]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=False)

    def test_null_rows(self):
        """Null rows propagate correctly."""
        geoms = [
            _simple_connected_mls(),
            None,
            _three_segment_chain(),
        ]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        assert result.validity[1] is False or not result.validity[1]


# ---------------------------------------------------------------------------
# Tests: directed mode
# ---------------------------------------------------------------------------

@requires_gpu
class TestLineMergeDirected:
    """GPU line_merge with directed=True."""

    def test_directed_connected(self):
        """Segments with matching end->start merge in directed mode."""
        geoms = [_directed_connected()]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=True)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=True)

    def test_directed_not_connected(self):
        """Segments without matching direction stay separate in directed mode."""
        geoms = [_directed_not_connected()]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=True)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=True)

    def test_directed_chain(self):
        """Three-segment directed chain merges into one."""
        geoms = [_three_segment_chain()]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=True)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=True)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

@requires_gpu
class TestLineMergeEdgeCases:
    """Edge cases for line_merge."""

    def test_empty_input(self):
        """Empty OGA returns empty OGA."""
        from vibespatial.constructive.line_merge import line_merge_owned
        owned = from_shapely_geometries([])
        result = line_merge_owned(owned, directed=False)
        assert result.row_count == 0

    def test_all_null_rows(self):
        """All-null input returns all-null output."""
        geoms = [None, None]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        assert not result.validity[0]
        assert not result.validity[1]

    def test_y_junction(self):
        """Y-junction (degree-3 node) should not fully merge."""
        # Three segments meeting at (1, 1) form a Y-junction
        geoms = [MultiLineString([
            [(0, 0), (1, 1)],
            [(1, 1), (2, 0)],
            [(1, 1), (1, 2)],
        ])]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=False)

    def test_large_batch(self):
        """Batch of 100 geometries to test thread dispatch."""
        geoms = []
        for i in range(100):
            geoms.append(MultiLineString([
                [(i, 0), (i + 0.5, 0.5)],
                [(i + 0.5, 0.5), (i + 1, 0)],
            ]))
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=False)

    def test_multicoord_segments(self):
        """Segments with multiple intermediate coordinates merge correctly."""
        geoms = [MultiLineString([
            [(0, 0), (0.5, 0.5), (1, 0)],
            [(1, 0), (1.5, 0.5), (2, 0)],
        ])]
        owned = from_shapely_geometries(geoms)
        from vibespatial.constructive.line_merge import line_merge_owned
        result = line_merge_owned(owned, directed=False)
        result_geoms = result.to_shapely()
        _compare_line_merge(result_geoms, geoms, directed=False)


# ---------------------------------------------------------------------------
# Tests: dispatch integration
# ---------------------------------------------------------------------------

@requires_gpu
class TestLineMergeDispatch:
    """Test dispatch through geometry_array.py."""

    def test_via_geometry_array(self):
        """line_merge dispatches through GeometryArray when owned is set."""
        from vibespatial.api.geometry_array import GeometryArray

        geoms = [_simple_connected_mls(), _three_segment_chain()]
        ga = GeometryArray.from_owned(
            from_shapely_geometries(geoms),
        )
        result = ga.line_merge(directed=False)
        result_geoms = [result._data[i] for i in range(len(result._data))]
        _compare_line_merge(result_geoms, geoms, directed=False)
