"""Tests for fused ingest + spatial index (packed Hilbert R-tree).

Tests cover:
  - Per-feature bounding box computation
  - Hilbert code computation and spatial ordering
  - R-tree node structure validity (children within parent bounds)
  - Edge cases: empty input, single feature, large datasets
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from vibespatial import has_gpu_runtime

requires_gpu = pytest.mark.skipif(
    not has_gpu_runtime(), reason="GPU not available"
)


def test_fused_index_hilbert_extent_has_no_raw_cupy_scalar_syncs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "vibespatial" / "io" / "gpu_parse" / "indexing.py"
    tree = ast.parse(path.read_text(), filename=str(path))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "item":
            offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}: .item()")
            continue
        if (
            isinstance(func, ast.Name)
            and func.id in {"bool", "int", "float"}
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Attribute)
            and isinstance(node.args[0].func.value, ast.Name)
            and node.args[0].func.value.id == "cp"
        ):
            offenders.append(
                f"{path.relative_to(repo_root)}:{node.lineno}: {func.id}(cp.*)"
            )

    assert offenders == []


def _make_point_data(xs, ys):
    """Create flat coordinate arrays and offsets for simple points.

    Each point is a single coordinate pair, so geometry_offsets = [0, 1, 2, ...].
    """
    import cupy as cp

    n = len(xs)
    d_x = cp.array(xs, dtype=cp.float64)
    d_y = cp.array(ys, dtype=cp.float64)
    offsets = cp.arange(n + 1, dtype=cp.int32)
    return d_x, d_y, offsets


def _make_linestring_data(coords_list):
    """Create flat coordinate arrays and offsets for linestrings.

    coords_list: list of list of (x, y) tuples
    """
    import cupy as cp

    xs = []
    ys = []
    offsets = [0]
    for coords in coords_list:
        for x, y in coords:
            xs.append(x)
            ys.append(y)
        offsets.append(len(xs))

    d_x = cp.array(xs, dtype=cp.float64)
    d_y = cp.array(ys, dtype=cp.float64)
    d_offsets = cp.array(offsets, dtype=cp.int32)
    return d_x, d_y, d_offsets


@requires_gpu
class TestFeatureBounds:
    """Test per-feature bounding box computation kernel."""

    def test_single_point_bounds(self):
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import _compute_bounds_gpu

        d_x, d_y, offsets = _make_point_data([10.0], [20.0])
        bounds = _compute_bounds_gpu(d_x, d_y, offsets, 1)
        result = cp.asnumpy(bounds)
        assert result.shape == (1, 4)
        np.testing.assert_allclose(result[0], [10.0, 20.0, 10.0, 20.0])

    def test_multiple_points_bounds(self):
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import _compute_bounds_gpu

        xs = [1.0, 5.0, -3.0]
        ys = [2.0, -1.0, 8.0]
        d_x, d_y, offsets = _make_point_data(xs, ys)
        bounds = _compute_bounds_gpu(d_x, d_y, offsets, 3)
        result = cp.asnumpy(bounds)
        assert result.shape == (3, 4)
        # Each point's bbox is (x, y, x, y)
        np.testing.assert_allclose(result[0], [1.0, 2.0, 1.0, 2.0])
        np.testing.assert_allclose(result[1], [5.0, -1.0, 5.0, -1.0])
        np.testing.assert_allclose(result[2], [-3.0, 8.0, -3.0, 8.0])

    def test_linestring_bounds(self):
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import _compute_bounds_gpu

        # Two linestrings
        coords_list = [
            [(0.0, 0.0), (10.0, 10.0), (20.0, 5.0)],
            [(-5.0, -5.0), (-1.0, -1.0)],
        ]
        d_x, d_y, offsets = _make_linestring_data(coords_list)
        bounds = _compute_bounds_gpu(d_x, d_y, offsets, 2)
        result = cp.asnumpy(bounds)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 20.0, 10.0])
        np.testing.assert_allclose(result[1], [-5.0, -5.0, -1.0, -1.0])

    def test_empty_feature_gets_nan_bounds(self):
        """Feature with zero coordinates gets NaN bounds."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import _compute_bounds_gpu

        # Two features: first has 2 coords, second has 0
        d_x = cp.array([1.0, 2.0], dtype=cp.float64)
        d_y = cp.array([3.0, 4.0], dtype=cp.float64)
        offsets = cp.array([0, 2, 2], dtype=cp.int32)  # second feature is empty
        bounds = _compute_bounds_gpu(d_x, d_y, offsets, 2)
        result = cp.asnumpy(bounds)
        np.testing.assert_allclose(result[0], [1.0, 3.0, 2.0, 4.0])
        assert np.all(np.isnan(result[1]))


@requires_gpu
class TestHilbertCodes:
    """Test Hilbert code computation."""

    def test_hilbert_codes_deterministic(self):
        """Same input always produces same codes."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import (
            _compute_bounds_gpu,
            _compute_hilbert_codes_gpu,
        )

        xs = [0.0, 50.0, 100.0, 25.0, 75.0]
        ys = [0.0, 50.0, 100.0, 25.0, 75.0]
        d_x, d_y, offsets = _make_point_data(xs, ys)
        bounds = _compute_bounds_gpu(d_x, d_y, offsets, 5)

        codes1 = cp.asnumpy(_compute_hilbert_codes_gpu(bounds, 5))
        codes2 = cp.asnumpy(_compute_hilbert_codes_gpu(bounds, 5))
        np.testing.assert_array_equal(codes1, codes2)

    def test_hilbert_codes_distinct_for_distinct_points(self):
        """Points at different locations should get different codes."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import (
            _compute_bounds_gpu,
            _compute_hilbert_codes_gpu,
        )

        # Spread points across the extent
        xs = [0.0, 25.0, 50.0, 75.0, 100.0]
        ys = [0.0, 25.0, 50.0, 75.0, 100.0]
        d_x, d_y, offsets = _make_point_data(xs, ys)
        bounds = _compute_bounds_gpu(d_x, d_y, offsets, 5)
        codes = cp.asnumpy(_compute_hilbert_codes_gpu(bounds, 5))

        # Not all the same
        assert len(set(codes)) > 1

    def test_hilbert_codes_match_cpu_encode(self):
        """GPU Hilbert codes should match the CPU implementation for the same
        normalized integer coordinates."""
        import cupy as cp

        from vibespatial.api.tools.hilbert_curve import _encode
        from vibespatial.io.gpu_parse.indexing import (
            _compute_bounds_gpu,
            _compute_hilbert_codes_gpu,
        )

        # Use points that map to known integer grid positions
        # Extent: [0, 100] x [0, 100]
        n = 20
        rng = np.random.RandomState(42)
        xs = rng.uniform(0.0, 100.0, n)
        ys = rng.uniform(0.0, 100.0, n)

        d_x, d_y, offsets = _make_point_data(xs.tolist(), ys.tolist())
        bounds = _compute_bounds_gpu(d_x, d_y, offsets, n)
        gpu_codes = cp.asnumpy(_compute_hilbert_codes_gpu(bounds, n))

        # Reproduce normalization on CPU
        minx, miny = xs.min(), ys.min()
        maxx, maxy = xs.max(), ys.max()
        span_x = max(maxx - minx, 1e-12)
        span_y = max(maxy - miny, 1e-12)
        norm_x = np.round(((xs - minx) / span_x) * 65535.0).astype(np.uint32)
        norm_y = np.round(((ys - miny) / span_y) * 65535.0).astype(np.uint32)

        cpu_codes = _encode(16, norm_x, norm_y).astype(np.uint32)
        np.testing.assert_array_equal(gpu_codes, cpu_codes)

    def test_nan_bounds_get_max_hilbert_code(self):
        """Features with NaN bounds should get 0xFFFFFFFF code (sorted last)."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import _compute_hilbert_codes_gpu

        bounds = cp.array([
            [1.0, 2.0, 3.0, 4.0],
            [float("nan"), float("nan"), float("nan"), float("nan")],
        ], dtype=cp.float64)
        codes = cp.asnumpy(_compute_hilbert_codes_gpu(bounds, 2))
        assert codes[1] == 0xFFFFFFFF


@requires_gpu
class TestHilbertSorting:
    """Test that Hilbert-sorted features have spatial locality."""

    def test_sorted_order_valid(self):
        """Sorted indices form a valid permutation of [0, n)."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        xs = [10.0, 80.0, 30.0, 60.0, 5.0]
        ys = [10.0, 80.0, 30.0, 60.0, 5.0]
        d_x, d_y, offsets = _make_point_data(xs, ys)
        index = build_spatial_index(d_x, d_y, offsets)

        sorted_indices = cp.asnumpy(index.d_sorted_indices)
        assert sorted_indices.shape == (5,)
        assert set(sorted_indices) == {0, 1, 2, 3, 4}

    def test_hilbert_codes_monotonic_in_sorted_order(self):
        """Hilbert codes should be non-decreasing in sorted order."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        n = 100
        rng = np.random.RandomState(123)
        xs = rng.uniform(0, 1000, n).tolist()
        ys = rng.uniform(0, 1000, n).tolist()
        d_x, d_y, offsets = _make_point_data(xs, ys)
        index = build_spatial_index(d_x, d_y, offsets)

        codes = cp.asnumpy(index.d_hilbert_codes)
        sorted_indices = cp.asnumpy(index.d_sorted_indices)
        sorted_codes = codes[sorted_indices]
        assert np.all(sorted_codes[:-1] <= sorted_codes[1:])


@requires_gpu
class TestRTreeStructure:
    """Test packed R-tree node structure validity."""

    def test_rtree_node_count(self):
        """Verify node count is consistent with tree structure."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        xs = list(range(100))
        ys = list(range(100))
        d_x, d_y, offsets = _make_point_data(
            [float(x) for x in xs],
            [float(y) for y in ys],
        )
        index = build_spatial_index(d_x, d_y, offsets, node_capacity=16)

        # 100 features / 16 = 7 leaf nodes (ceil)
        assert index.n_leaf_nodes == 7
        assert index.n_nodes > index.n_leaf_nodes  # must have internal nodes
        assert index.node_capacity == 16

        # Node bounds shape matches
        node_bounds = cp.asnumpy(index.d_node_bounds)
        assert node_bounds.shape == (index.n_nodes, 4)

    def test_children_within_parent_bounds(self):
        """Every child node's bounds must be contained within its parent's bounds."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        n = 64
        rng = np.random.RandomState(7)
        xs = rng.uniform(0, 100, n).tolist()
        ys = rng.uniform(0, 100, n).tolist()
        d_x, d_y, offsets = _make_point_data(xs, ys)
        index = build_spatial_index(d_x, d_y, offsets, node_capacity=4)

        node_bounds = cp.asnumpy(index.d_node_bounds)
        node_children = cp.asnumpy(index.d_node_children)

        for parent_idx in range(node_children.shape[0]):
            parent = node_bounds[parent_idx]
            for child_idx in node_children[parent_idx]:
                if child_idx < 0:
                    continue  # unused slot
                child = node_bounds[child_idx]
                if np.any(np.isnan(child)):
                    continue
                # Child minx >= parent minx (with tolerance for floating point)
                assert child[0] >= parent[0] - 1e-12, (
                    f"Child {child_idx} minx {child[0]} < parent {parent_idx} minx {parent[0]}"
                )
                assert child[1] >= parent[1] - 1e-12, (
                    f"Child {child_idx} miny {child[1]} < parent {parent_idx} miny {parent[1]}"
                )
                assert child[2] <= parent[2] + 1e-12, (
                    f"Child {child_idx} maxx {child[2]} > parent {parent_idx} maxx {parent[2]}"
                )
                assert child[3] <= parent[3] + 1e-12, (
                    f"Child {child_idx} maxy {child[3]} > parent {parent_idx} maxy {parent[3]}"
                )

    def test_leaf_bounds_enclose_features(self):
        """Leaf node bounds must enclose the features they contain."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        xs = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        ys = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        d_x, d_y, offsets = _make_point_data(xs, ys)
        index = build_spatial_index(d_x, d_y, offsets, node_capacity=4)

        feature_bounds = cp.asnumpy(index.d_feature_bounds)
        node_bounds = cp.asnumpy(index.d_node_bounds)
        sorted_indices = cp.asnumpy(index.d_sorted_indices)
        B = index.node_capacity

        # Leaf nodes are the last n_leaf_nodes entries in node_bounds
        leaf_start = index.n_nodes - index.n_leaf_nodes
        for leaf_idx in range(index.n_leaf_nodes):
            node_idx = leaf_start + leaf_idx
            leaf_bb = node_bounds[node_idx]

            # Features in this leaf: sorted_indices[leaf_idx*B : (leaf_idx+1)*B]
            feat_start = leaf_idx * B
            feat_end = min(feat_start + B, index.n_features)
            for fi in range(feat_start, feat_end):
                feat_idx = sorted_indices[fi]
                fb = feature_bounds[feat_idx]
                if np.any(np.isnan(fb)):
                    continue
                assert fb[0] >= leaf_bb[0] - 1e-12
                assert fb[1] >= leaf_bb[1] - 1e-12
                assert fb[2] <= leaf_bb[2] + 1e-12
                assert fb[3] <= leaf_bb[3] + 1e-12

    def test_single_feature_tree(self):
        """A single feature should produce a valid tree."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        d_x, d_y, offsets = _make_point_data([42.0], [24.0])
        index = build_spatial_index(d_x, d_y, offsets)

        assert index.n_features == 1
        assert index.n_leaf_nodes == 1
        assert index.n_nodes >= 1
        sorted_indices = cp.asnumpy(index.d_sorted_indices)
        np.testing.assert_array_equal(sorted_indices, [0])


@requires_gpu
class TestBuildSpatialIndex:
    """Integration tests for build_spatial_index."""

    def test_empty_input(self):
        """Empty geometry_offsets (single element) produces empty index."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        d_x = cp.empty(0, dtype=cp.float64)
        d_y = cp.empty(0, dtype=cp.float64)
        offsets = cp.array([0], dtype=cp.int32)
        index = build_spatial_index(d_x, d_y, offsets)

        assert index.n_features == 0
        assert index.n_nodes == 0

    def test_large_dataset(self):
        """Build index for 10K features without error."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        n = 10_000
        rng = np.random.RandomState(0)
        xs = rng.uniform(-180, 180, n).tolist()
        ys = rng.uniform(-90, 90, n).tolist()
        d_x, d_y, offsets = _make_point_data(xs, ys)
        index = build_spatial_index(d_x, d_y, offsets)

        assert index.n_features == n
        sorted_indices = cp.asnumpy(index.d_sorted_indices)
        assert set(sorted_indices) == set(range(n))

    def test_build_index_from_reader_alias(self):
        """build_index_from_reader is functionally identical to build_spatial_index."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import (
            build_index_from_reader,
            build_spatial_index,
        )

        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 4.0, 3.0, 2.0, 1.0]
        d_x, d_y, offsets = _make_point_data(xs, ys)

        idx1 = build_spatial_index(d_x, d_y, offsets)
        idx2 = build_index_from_reader(d_x, d_y, offsets)

        np.testing.assert_array_equal(
            cp.asnumpy(idx1.d_sorted_indices),
            cp.asnumpy(idx2.d_sorted_indices),
        )
        np.testing.assert_array_equal(
            cp.asnumpy(idx1.d_hilbert_codes),
            cp.asnumpy(idx2.d_hilbert_codes),
        )

    def test_custom_node_capacity(self):
        """Non-default node capacity produces a valid tree."""

        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        n = 50
        xs = [float(i) for i in range(n)]
        ys = [float(i) for i in range(n)]
        d_x, d_y, offsets = _make_point_data(xs, ys)
        index = build_spatial_index(d_x, d_y, offsets, node_capacity=8)

        assert index.node_capacity == 8
        assert index.n_leaf_nodes == 7  # ceil(50/8) = 7
        assert index.n_nodes > 0

    def test_all_same_point(self):
        """All features at the same location should not crash."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        n = 20
        d_x, d_y, offsets = _make_point_data([5.0] * n, [5.0] * n)
        index = build_spatial_index(d_x, d_y, offsets)

        assert index.n_features == n
        codes = cp.asnumpy(index.d_hilbert_codes)
        # All points same location -> all codes identical
        assert np.all(codes == codes[0])

    def test_feature_bounds_shape_and_dtype(self):
        """Feature bounds array has correct shape and dtype."""
        import cupy as cp

        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        d_x, d_y, offsets = _make_point_data([1.0, 2.0], [3.0, 4.0])
        index = build_spatial_index(d_x, d_y, offsets)

        assert index.d_feature_bounds.shape == (2, 4)
        assert index.d_feature_bounds.dtype == cp.float64
        assert index.d_hilbert_codes.dtype == cp.uint32
        assert index.d_sorted_indices.dtype == cp.int32
