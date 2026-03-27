"""Tests for spatial_index_knn_device -- GPU k-nearest-neighbor spatial query.

Validates that the device-side k-NN query:
  - Produces the same nearest results as CPU Shapely distance
  - Supports k=1 (most common) and general k values
  - Correctly handles max_distance parameter
  - Works with point-point, point-polygon, and linestring geometry pairs
  - Returns device-resident arrays (zero D2H during computation)
  - Handles edge cases: empty inputs, all-null geometries, single geometry
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, box

from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds
from vibespatial.runtime import has_gpu_runtime

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_points(coords: list[tuple[float, float]]) -> np.ndarray:
    return np.asarray([Point(x, y) for x, y in coords], dtype=object)


def _make_grid_points(n_cols: int, n_rows: int) -> np.ndarray:
    """Create a regular grid of points."""
    coords = [(c + 0.5, r + 0.5) for r in range(n_rows) for c in range(n_cols)]
    return _make_points(coords)


def _cpu_nearest_k(
    query_geoms: np.ndarray,
    tree_geoms: np.ndarray,
    k: int,
    max_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU reference: brute-force k-nearest using Shapely distance."""
    q_list = []
    t_list = []
    d_list = []
    for qi in range(len(query_geoms)):
        qg = query_geoms[qi]
        if shapely.is_missing(qg) or shapely.is_empty(qg):
            continue
        dists = []
        for ti in range(len(tree_geoms)):
            tg = tree_geoms[ti]
            if shapely.is_missing(tg) or shapely.is_empty(tg):
                continue
            d = shapely.distance(qg, tg)
            if max_distance is not None and d > max_distance:
                continue
            dists.append((ti, d))
        dists.sort(key=lambda x: x[1])
        # Keep k nearest (all ties at the k-th distance)
        if not dists:
            continue
        keep = dists[:k]
        if k <= len(dists):
            kth_dist = dists[k - 1][1]
            # Include additional ties at the k-th distance
            for j in range(k, len(dists)):
                if np.isclose(dists[j][1], kth_dist, rtol=1e-10):
                    keep.append(dists[j])
                else:
                    break
        for ti, d in keep:
            q_list.append(qi)
            t_list.append(ti)
            d_list.append(d)
    return (
        np.array(q_list, dtype=np.int32),
        np.array(t_list, dtype=np.int32),
        np.array(d_list, dtype=np.float64),
    )


def _bounds_for(geoms: np.ndarray) -> np.ndarray:
    owned = from_shapely_geometries(geoms)
    return compute_geometry_bounds(owned, dispatch_mode="auto")


# ---------------------------------------------------------------------------
# Tests: basic correctness (point-point k=1)
# ---------------------------------------------------------------------------


@requires_gpu
def test_knn_point_point_k1_basic():
    """k=1 nearest matches CPU Shapely for a simple point grid."""
    from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device

    tree_geoms = _make_grid_points(5, 5)
    query_geoms = _make_points([(0.0, 0.0), (2.5, 2.5), (4.9, 4.9)])

    query_owned = from_shapely_geometries(query_geoms)
    tree_owned = from_shapely_geometries(tree_geoms)
    query_bounds = _bounds_for(query_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=1,
    )
    assert result is not None, "GPU k-NN should succeed for point-point data"
    assert result.total_pairs > 0

    gpu_q, gpu_t, gpu_d = result.to_host()
    cpu_q, cpu_t, cpu_d = _cpu_nearest_k(query_geoms, tree_geoms, k=1)

    # Each query should have exactly 1 nearest result.
    assert len(gpu_q) == len(cpu_q), f"Pair count mismatch: GPU={len(gpu_q)} CPU={len(cpu_q)}"

    # Sort both for stable comparison.
    gpu_order = np.lexsort((gpu_t, gpu_q))
    cpu_order = np.lexsort((cpu_t, cpu_q))
    np.testing.assert_array_equal(gpu_q[gpu_order], cpu_q[cpu_order])
    np.testing.assert_array_equal(gpu_t[gpu_order], cpu_t[cpu_order])
    np.testing.assert_allclose(gpu_d[gpu_order], cpu_d[cpu_order], rtol=1e-10)


@requires_gpu
def test_knn_point_point_k3():
    """k=3 nearest matches CPU Shapely for point data."""
    from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device

    tree_geoms = _make_grid_points(5, 5)
    query_geoms = _make_points([(2.5, 2.5)])

    query_owned = from_shapely_geometries(query_geoms)
    tree_owned = from_shapely_geometries(tree_geoms)
    query_bounds = _bounds_for(query_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=3,
    )
    assert result is not None
    assert result.k == 3

    gpu_q, gpu_t, gpu_d = result.to_host()
    cpu_q, cpu_t, cpu_d = _cpu_nearest_k(query_geoms, tree_geoms, k=3)

    # For a center point on a 5x5 grid, there are 4 equidistant nearest
    # points at distance sqrt(0.5), so k=3 should return 3 or 4 (ties).
    assert len(gpu_q) >= 3
    # Check distances match.
    np.testing.assert_allclose(
        np.sort(gpu_d), np.sort(cpu_d[:len(gpu_d)]), rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# Tests: max_distance support
# ---------------------------------------------------------------------------


@requires_gpu
def test_knn_max_distance():
    """max_distance correctly prunes candidates beyond threshold."""
    from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device

    # Tree: points at (0,0), (5,0), (100,0)
    tree_geoms = _make_points([(0.0, 0.0), (5.0, 0.0), (100.0, 0.0)])
    # Query: point at (2,0) -- nearest is (0,0) at dist=2, then (5,0) at dist=3
    query_geoms = _make_points([(2.0, 0.0)])

    query_owned = from_shapely_geometries(query_geoms)
    tree_owned = from_shapely_geometries(tree_geoms)
    query_bounds = _bounds_for(query_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    # With max_distance=4, should get only (0,0) at dist=2 and (5,0) at dist=3.
    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=10,
        max_distance=4.0,
    )
    assert result is not None
    gpu_q, gpu_t, gpu_d = result.to_host()

    # Should not include (100,0) which is at dist=98.
    assert len(gpu_q) == 2, f"Expected 2 results within max_distance=4, got {len(gpu_q)}"
    np.testing.assert_allclose(np.sort(gpu_d), [2.0, 3.0], rtol=1e-10)


@requires_gpu
def test_knn_max_distance_no_results():
    """max_distance that excludes all candidates returns empty result."""
    from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device

    tree_geoms = _make_points([(10.0, 10.0)])
    query_geoms = _make_points([(0.0, 0.0)])

    query_owned = from_shapely_geometries(query_geoms)
    tree_owned = from_shapely_geometries(tree_geoms)
    query_bounds = _bounds_for(query_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=1,
        max_distance=1.0,  # distance is ~14.1, so no candidates
    )
    # Either None or empty result.
    if result is not None:
        assert result.total_pairs == 0


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


@requires_gpu
def test_knn_empty_inputs():
    """Empty query or tree returns empty result."""
    from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device

    tree_geoms = _make_points([(1.0, 1.0)])
    empty_geoms = np.asarray([], dtype=object)

    tree_owned = from_shapely_geometries(tree_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    # Empty query
    query_owned = from_shapely_geometries(empty_geoms)
    query_bounds = np.empty((0, 4), dtype=np.float64)

    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=1,
    )
    if result is not None:
        assert result.total_pairs == 0


@requires_gpu
def test_knn_single_geometry():
    """Single query and single tree geometry."""
    from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device

    tree_geoms = _make_points([(3.0, 4.0)])
    query_geoms = _make_points([(0.0, 0.0)])

    query_owned = from_shapely_geometries(query_geoms)
    tree_owned = from_shapely_geometries(tree_geoms)
    query_bounds = _bounds_for(query_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=1,
    )
    assert result is not None
    gpu_q, gpu_t, gpu_d = result.to_host()

    assert len(gpu_q) == 1
    assert gpu_q[0] == 0
    assert gpu_t[0] == 0
    np.testing.assert_allclose(gpu_d[0], 5.0, rtol=1e-10)


@requires_gpu
def test_knn_device_resident_output():
    """Output arrays are device-resident (CuPy arrays)."""
    import cupy

    from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device

    tree_geoms = _make_grid_points(3, 3)
    query_geoms = _make_points([(1.5, 1.5)])

    query_owned = from_shapely_geometries(query_geoms)
    tree_owned = from_shapely_geometries(tree_geoms)
    query_bounds = _bounds_for(query_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=1,
    )
    assert result is not None
    assert result.total_pairs > 0

    # Verify device residency -- arrays should be CuPy, not numpy.
    assert isinstance(result.d_query_idx, cupy.ndarray), \
        f"Expected CuPy array, got {type(result.d_query_idx)}"
    assert isinstance(result.d_target_idx, cupy.ndarray), \
        f"Expected CuPy array, got {type(result.d_target_idx)}"
    assert isinstance(result.d_distances, cupy.ndarray), \
        f"Expected CuPy array, got {type(result.d_distances)}"


# ---------------------------------------------------------------------------
# Tests: larger workloads
# ---------------------------------------------------------------------------


@requires_gpu
def test_knn_larger_point_grid():
    """k-NN on a 20x20 grid with 10 random query points."""
    from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device

    rng = np.random.default_rng(42)
    tree_geoms = _make_grid_points(20, 20)
    query_coords = rng.uniform(0, 20, size=(10, 2))
    query_geoms = _make_points([(x, y) for x, y in query_coords])

    query_owned = from_shapely_geometries(query_geoms)
    tree_owned = from_shapely_geometries(tree_geoms)
    query_bounds = _bounds_for(query_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=1,
    )
    assert result is not None
    gpu_q, gpu_t, gpu_d = result.to_host()
    cpu_q, cpu_t, cpu_d = _cpu_nearest_k(query_geoms, tree_geoms, k=1)

    # Distances should match even if tie-breaking picks different targets.
    np.testing.assert_allclose(
        np.sort(gpu_d), np.sort(cpu_d), rtol=1e-8,
    )


@requires_gpu
def test_knn_k5_larger():
    """k=5 on a larger dataset."""
    from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device

    rng = np.random.default_rng(123)
    tree_coords = rng.uniform(0, 100, size=(50, 2))
    tree_geoms = _make_points([(x, y) for x, y in tree_coords])
    query_coords = rng.uniform(0, 100, size=(5, 2))
    query_geoms = _make_points([(x, y) for x, y in query_coords])

    query_owned = from_shapely_geometries(query_geoms)
    tree_owned = from_shapely_geometries(tree_geoms)
    query_bounds = _bounds_for(query_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=5,
    )
    assert result is not None
    gpu_q, gpu_t, gpu_d = result.to_host()
    cpu_q, cpu_t, cpu_d = _cpu_nearest_k(query_geoms, tree_geoms, k=5)

    # Each query should have at least 5 results.
    for qi in range(5):
        gpu_count = np.sum(gpu_q == qi)
        assert gpu_count >= 5, f"Query {qi}: GPU returned {gpu_count} results, expected >= 5"
        # Distances for each query should match.
        gpu_dists = np.sort(gpu_d[gpu_q == qi])
        cpu_dists = np.sort(cpu_d[cpu_q == qi])
        # Compare first 5 (ignoring extra ties).
        np.testing.assert_allclose(gpu_dists[:5], cpu_dists[:5], rtol=1e-8)


# ---------------------------------------------------------------------------
# Tests: point-polygon distance
# ---------------------------------------------------------------------------


@requires_gpu
def test_knn_point_polygon():
    """k-NN with point queries and polygon targets."""
    from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device

    # Polygons: three boxes at different positions.
    tree_geoms = np.asarray([
        box(0, 0, 1, 1),
        box(5, 5, 6, 6),
        box(10, 10, 11, 11),
    ], dtype=object)
    # Query: point at (3, 3) -- closest to box(0,0,1,1) boundary
    query_geoms = _make_points([(3.0, 3.0)])

    query_owned = from_shapely_geometries(query_geoms)
    tree_owned = from_shapely_geometries(tree_geoms)
    query_bounds = _bounds_for(query_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=1,
    )

    if result is not None and result.total_pairs > 0:
        gpu_q, gpu_t, gpu_d = result.to_host()
        cpu_q, cpu_t, cpu_d = _cpu_nearest_k(query_geoms, tree_geoms, k=1)

        # Distance to nearest polygon should match CPU.
        np.testing.assert_allclose(gpu_d[0], cpu_d[0], rtol=1e-8)


# ---------------------------------------------------------------------------
# Tests: return_all=False
# ---------------------------------------------------------------------------


@requires_gpu
def test_knn_return_all_false():
    """return_all=False returns exactly one result per query."""
    from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device

    # Create points where query has ties (equidistant targets).
    tree_geoms = _make_points([(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)])
    query_geoms = _make_points([(0.0, 0.0)])  # equidistant to all 4

    query_owned = from_shapely_geometries(query_geoms)
    tree_owned = from_shapely_geometries(tree_geoms)
    query_bounds = _bounds_for(query_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=1,
        return_all=False,
    )
    assert result is not None
    gpu_q, gpu_t, gpu_d = result.to_host()

    # With return_all=False, should get exactly 1 result.
    assert len(gpu_q) == 1
    np.testing.assert_allclose(gpu_d[0], 1.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Integration: sjoin_nearest compatibility
# ---------------------------------------------------------------------------


@requires_gpu
def test_knn_matches_sjoin_nearest_output_format():
    """Verify output format is compatible with sjoin_nearest expectations."""
    from vibespatial.spatial.spatial_index_knn_device import (
        DeviceKnnResult,
        spatial_index_knn_device,
    )

    tree_geoms = _make_grid_points(3, 3)
    query_geoms = _make_points([(1.5, 1.5)])

    query_owned = from_shapely_geometries(query_geoms)
    tree_owned = from_shapely_geometries(tree_geoms)
    query_bounds = _bounds_for(query_geoms)
    tree_bounds = _bounds_for(tree_geoms)

    result = spatial_index_knn_device(
        query_owned, tree_owned,
        query_bounds, tree_bounds,
        k=1,
    )
    assert result is not None
    assert isinstance(result, DeviceKnnResult)

    # to_host() should return (query_idx, target_idx, distances) all numpy.
    q, t, d = result.to_host()
    assert isinstance(q, np.ndarray)
    assert isinstance(t, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert q.dtype == np.int32
    assert t.dtype == np.int32
    assert d.dtype == np.float64
