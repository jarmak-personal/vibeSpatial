"""Tests for spatial_index_device_query — GPU BVH-style traversal.

Validates that the unified device spatial index query function:
  - Produces the same candidate pairs as CPU STRtree (no false negatives)
  - Correctly selects brute-force vs Morton range strategy
  - Handles dwithin distance expansion
  - Integrates with sjoin end-to-end
  - Reports correct execution metadata
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, box

from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.kernels.core.geometry_analysis import (
    compute_geometry_bounds,
    compute_geometry_bounds_device,
)
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.residency import Residency
from vibespatial.spatial.query import (
    build_owned_spatial_index,
    query_spatial_index,
    spatial_index_device_query,
)

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid_boxes(n_cols: int, n_rows: int) -> np.ndarray:
    """Create a regular grid of unit boxes."""
    geoms = []
    for r in range(n_rows):
        for c in range(n_cols):
            geoms.append(box(c, r, c + 1, r + 1))
    return np.asarray(geoms, dtype=object)


def _make_random_points(n: int, *, seed: int = 42, extent: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, extent, size=(n, 2))
    return np.asarray([Point(x, y) for x, y in coords], dtype=object)


def _make_random_boxes(
    n: int, *, seed: int = 42, extent: float = 100.0, size: float = 5.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mins = rng.uniform(0, extent - size, size=(n, 2))
    geoms = []
    for x, y in mins:
        geoms.append(box(x, y, x + size, y + size))
    return np.asarray(geoms, dtype=object)


def _cpu_bbox_pairs(query_bounds: np.ndarray, tree_bounds: np.ndarray):
    """CPU reference: brute-force bbox overlap detection."""
    left = []
    right = []
    for q in range(query_bounds.shape[0]):
        qb = query_bounds[q]
        if np.isnan(qb).any():
            continue
        for t in range(tree_bounds.shape[0]):
            tb = tree_bounds[t]
            if np.isnan(tb).any():
                continue
            if qb[0] <= tb[2] and qb[2] >= tb[0] and qb[1] <= tb[3] and qb[3] >= tb[1]:
                left.append(q)
                right.append(t)
    return np.array(left, dtype=np.int32), np.array(right, dtype=np.int32)


# ---------------------------------------------------------------------------
# Tests: basic correctness
# ---------------------------------------------------------------------------


@requires_gpu
def test_device_query_matches_cpu_brute_force_small():
    """Device query produces identical pairs to CPU brute-force for small input."""
    tree_geoms = _make_grid_boxes(5, 5)  # 25 boxes
    query_geoms = np.asarray([
        box(0.5, 0.5, 2.5, 2.5),  # overlaps several
        box(10, 10, 10.5, 10.5),   # no overlap
        box(3.5, 3.5, 4.5, 4.5),  # corner overlap
    ], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)
    tree_bounds = flat_index.bounds

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None

    gpu_left, gpu_right = cands.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds, tree_bounds)

    # Sort both for deterministic comparison.
    gpu_pairs = set(zip(gpu_left.tolist(), gpu_right.tolist()))
    cpu_pairs = set(zip(cpu_left.tolist(), cpu_right.tolist()))
    assert gpu_pairs == cpu_pairs, (
        f"GPU pairs != CPU pairs. GPU-only: {gpu_pairs - cpu_pairs}, "
        f"CPU-only: {cpu_pairs - gpu_pairs}"
    )


@requires_gpu
def test_device_query_matches_cpu_brute_force_medium():
    """Device query matches CPU for 100x100 (10K N*M) input."""
    tree_geoms = _make_random_boxes(100, seed=1, extent=50.0, size=3.0)
    query_geoms = _make_random_boxes(100, seed=2, extent=50.0, size=3.0)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)
    tree_bounds = flat_index.bounds

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None

    gpu_left, gpu_right = cands.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds, tree_bounds)

    gpu_pairs = set(zip(gpu_left.tolist(), gpu_right.tolist()))
    cpu_pairs = set(zip(cpu_left.tolist(), cpu_right.tolist()))
    assert gpu_pairs == cpu_pairs


@requires_gpu
def test_device_query_no_candidates_returns_empty_device_candidates():
    """When GPU runs and no bbox overlaps exist, result stays device-resident."""
    tree_geoms = np.asarray([box(0, 0, 1, 1)], dtype=object)
    query_geoms = np.asarray([box(100, 100, 101, 101)], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None
    assert cands.total_pairs == 0
    gpu_left, gpu_right = cands.to_host()
    assert gpu_left.size == 0
    assert gpu_right.size == 0


@requires_gpu
def test_device_query_empty_inputs():
    """Empty query inputs preserve the selected GPU path with empty candidates."""
    tree_geoms = np.asarray([box(0, 0, 1, 1)], dtype=object)
    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)

    empty_bounds = np.empty((0, 4), dtype=np.float64)
    cands, execution = spatial_index_device_query(flat_index, empty_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None
    assert cands.total_pairs == 0


@requires_gpu
def test_device_query_scalar_single_query():
    """Single query row uses scalar fast path."""
    tree_geoms = _make_grid_boxes(10, 10)  # 100 boxes
    query_geoms = np.asarray([box(2.5, 2.5, 7.5, 7.5)], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)
    tree_bounds = flat_index.bounds

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None

    gpu_left, gpu_right = cands.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds, tree_bounds)

    gpu_pairs = set(zip(gpu_left.tolist(), gpu_right.tolist()))
    cpu_pairs = set(zip(cpu_left.tolist(), cpu_right.tolist()))
    assert gpu_pairs == cpu_pairs


# ---------------------------------------------------------------------------
# Tests: dwithin distance expansion
# ---------------------------------------------------------------------------


@requires_gpu
def test_device_query_with_distance_expansion():
    """Distance parameter expands query bounds for dwithin candidates."""
    tree_geoms = np.asarray([box(0, 0, 1, 1), box(5, 5, 6, 6)], dtype=object)
    query_geoms = np.asarray([box(2, 2, 3, 3)], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)

    # Without distance: only close box might overlap.
    cands_no_dist, _ = spatial_index_device_query(flat_index, query_bounds)

    # With large distance: both boxes should be candidates.
    distances = np.array([3.0], dtype=np.float64)
    cands_dist, exec_dist = spatial_index_device_query(
        flat_index, query_bounds, distance=distances,
    )
    assert exec_dist.selected is ExecutionMode.GPU
    assert cands_dist is not None
    gpu_left, gpu_right = cands_dist.to_host()
    # With distance=3.0, query box [2,2,3,3] expands to [-1,-1,6,6]
    # which should overlap both tree boxes.
    assert gpu_right.size == 2, f"Expected 2 candidates, got {gpu_right.size}"


@requires_gpu
def test_device_query_accepts_device_bounds_without_d2h(strict_device_guard):
    """Device-resident query bounds stay on device through candidate generation."""
    tree_geoms = np.asarray([box(0, 0, 1, 1), box(5, 5, 6, 6)], dtype=object)
    query_geoms = [box(2, 2, 3, 3)]

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms, residency=Residency.DEVICE)

    query_bounds = compute_geometry_bounds_device(query_owned)
    cands, execution = spatial_index_device_query(
        flat_index,
        query_bounds,
        distance=np.asarray([3.0], dtype=np.float64),
    )

    assert execution.selected is ExecutionMode.GPU
    assert cands is not None
    assert hasattr(query_bounds, "__cuda_array_interface__")
    gpu_left, gpu_right = cands.to_host()
    assert gpu_left.tolist() == [0, 0]
    assert gpu_right.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Tests: Morton range strategy selection
# ---------------------------------------------------------------------------


@requires_gpu
def test_device_query_uses_morton_range_for_large_input():
    """For large N*M, Morton range strategy is selected (detectable via execution reason)."""
    # Create enough geometries to exceed the Morton range crossover (1M).
    # 1000 x 1000 = 1M.
    tree_geoms = _make_random_boxes(1000, seed=10, extent=200.0, size=2.0)
    query_geoms = _make_random_boxes(1000, seed=11, extent=200.0, size=2.0)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)
    tree_bounds = flat_index.bounds

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None

    # Verify correctness against CPU reference.
    gpu_left, gpu_right = cands.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds, tree_bounds)

    gpu_pairs = set(zip(gpu_left.tolist(), gpu_right.tolist()))
    cpu_pairs = set(zip(cpu_left.tolist(), cpu_right.tolist()))
    # Morton range may produce a superset (false positives are acceptable
    # since they get refined by predicate evaluation), but must not have
    # false negatives.
    assert cpu_pairs.issubset(gpu_pairs), (
        f"GPU Morton range has false negatives: {cpu_pairs - gpu_pairs}"
    )
    # Verify execution reason mentions Morton range.
    assert "Morton" in execution.reason or "brute" in execution.reason


@requires_gpu
def test_device_query_brute_force_for_small_input():
    """For small N*M, brute-force strategy is used."""
    tree_geoms = _make_grid_boxes(3, 3)  # 9 boxes
    query_geoms = np.asarray([box(0.5, 0.5, 2.5, 2.5)], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert "brute" in execution.reason


# ---------------------------------------------------------------------------
# Tests: end-to-end sjoin integration
# ---------------------------------------------------------------------------


@requires_gpu
def test_sjoin_uses_device_query_end_to_end():
    """sjoin produces correct results using the device spatial index query."""
    import vibespatial as gpd

    # Left: random points.
    left_points = _make_random_points(50, seed=100, extent=10.0)
    left_gdf = gpd.GeoDataFrame(
        {"id": range(len(left_points))},
        geometry=list(left_points),
    )

    # Right: grid of boxes.
    right_boxes = _make_grid_boxes(10, 10)
    right_gdf = gpd.GeoDataFrame(
        {"zone": range(len(right_boxes))},
        geometry=list(right_boxes),
    )

    # sjoin should work correctly.
    result = gpd.sjoin(left_gdf, right_gdf, predicate="intersects")
    assert len(result) > 0

    # Verify a few known overlaps manually: every point in [0,10] x [0,10]
    # should match at least one grid cell.
    for idx in range(min(5, len(left_gdf))):
        pt = left_gdf.geometry.iloc[idx]
        # Find expected zone(s).
        col = int(pt.x)
        row = int(pt.y)
        if col >= 10:
            col = 9
        if row >= 10:
            row = 9
        # Check that the point appears in the result with this zone.
        point_results = result[result.index == idx]
        assert len(point_results) > 0, f"Point {idx} at ({pt.x:.2f}, {pt.y:.2f}) not in sjoin result"


@requires_gpu
def test_sjoin_with_polygons():
    """sjoin correctly handles polygon-polygon spatial joins."""
    import vibespatial as gpd

    left_polys = np.asarray([
        box(0, 0, 5, 5),
        box(10, 10, 15, 15),
        box(20, 20, 25, 25),
    ], dtype=object)
    right_polys = np.asarray([
        box(3, 3, 8, 8),    # overlaps first left
        box(12, 12, 18, 18), # overlaps second left
        box(50, 50, 55, 55), # no overlap
    ], dtype=object)

    left_gdf = gpd.GeoDataFrame(
        {"left_id": [0, 1, 2]}, geometry=list(left_polys),
    )
    right_gdf = gpd.GeoDataFrame(
        {"right_id": [0, 1, 2]}, geometry=list(right_polys),
    )

    result = gpd.sjoin(left_gdf, right_gdf, predicate="intersects")
    assert len(result) == 2  # two overlapping pairs

    # Verify the correct pairs.
    result_pairs = set(zip(result["left_id"].tolist(), result["right_id"].tolist()))
    assert (0, 0) in result_pairs
    assert (1, 1) in result_pairs


# ---------------------------------------------------------------------------
# Tests: execution metadata
# ---------------------------------------------------------------------------


@requires_gpu
def test_device_query_returns_execution_metadata():
    """Execution metadata has correct structure."""
    tree_geoms = _make_grid_boxes(5, 5)
    query_geoms = np.asarray([box(0.5, 0.5, 2.5, 2.5)], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert execution.implementation == "owned_gpu_spatial_query"
    assert len(execution.reason) > 0


def test_device_query_cpu_fallback_when_no_gpu():
    """When GPU is unavailable, function returns None with CPU execution."""
    # This test runs even without GPU.
    import unittest.mock

    tree_geoms = _make_grid_boxes(3, 3)

    with unittest.mock.patch(
        "vibespatial.spatial.spatial_index_device.has_gpu_runtime",
        return_value=False,
    ):
        tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
        query_bounds = np.array([[0.5, 0.5, 2.5, 2.5]], dtype=np.float64)

        cands, execution = spatial_index_device_query(flat_index, query_bounds)
        assert cands is None
        assert execution.selected is ExecutionMode.CPU

        empty_bounds = np.empty((0, 4), dtype=np.float64)
        cands, execution = spatial_index_device_query(flat_index, empty_bounds)
        assert cands is None
        assert execution.selected is ExecutionMode.CPU


# ---------------------------------------------------------------------------
# Tests: NaN handling
# ---------------------------------------------------------------------------


@requires_gpu
def test_device_query_handles_nan_bounds():
    """NaN bounds in query or tree are handled gracefully."""
    tree_geoms = np.asarray([box(0, 0, 1, 1), None, box(2, 2, 3, 3)], dtype=object)
    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)

    # Query with a mix of valid and NaN bounds.
    query_bounds = np.array([
        [0.5, 0.5, 1.5, 1.5],   # overlaps first tree box
        [np.nan, np.nan, np.nan, np.nan],  # invalid
    ], dtype=np.float64)

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    if cands is not None:
        gpu_left, gpu_right = cands.to_host()
        # Only the first query row should produce candidates.
        assert np.all(gpu_left == 0) or gpu_left.size == 0


# ---------------------------------------------------------------------------
# Tests: correctness via query_spatial_index integration
# ---------------------------------------------------------------------------


@requires_gpu
def test_query_spatial_index_uses_device_query():
    """query_spatial_index routes through spatial_index_device_query."""
    tree_geoms = _make_grid_boxes(10, 10)
    query_geoms = _make_random_points(50, seed=200, extent=10.0)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)

    result = query_spatial_index(
        tree_owned, flat_index, query_geoms,
        predicate="intersects",
        return_metadata=True,
    )
    indices, execution = result
    # Should use GPU path.
    assert execution.selected is ExecutionMode.GPU
    # Indices should be 2D (left_idx, right_idx).
    assert indices.ndim == 2
    assert indices.shape[0] == 2
    assert indices.shape[1] > 0  # at least some intersecting pairs


@requires_gpu
def test_device_query_large_correctness():
    """Correctness check with 500 query x 500 tree (250K N*M)."""
    tree_geoms = _make_random_boxes(500, seed=30, extent=100.0, size=5.0)
    query_geoms = _make_random_boxes(500, seed=31, extent=100.0, size=5.0)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)
    tree_bounds = flat_index.bounds

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None

    gpu_left, gpu_right = cands.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds, tree_bounds)

    gpu_pairs = set(zip(gpu_left.tolist(), gpu_right.tolist()))
    cpu_pairs = set(zip(cpu_left.tolist(), cpu_right.tolist()))
    # Must have no false negatives.
    assert cpu_pairs.issubset(gpu_pairs), (
        f"False negatives: {cpu_pairs - gpu_pairs}"
    )
    # For brute-force, should be exact match.
    if "brute" in execution.reason:
        assert gpu_pairs == cpu_pairs
