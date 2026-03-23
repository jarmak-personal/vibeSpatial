"""Tests for the segmented union_all kernel.

Verifies GPU and CPU paths against Shapely oracle, covering:
- Single-element groups (pass-through)
- Empty groups (produce empty polygon)
- Two-element groups (single pairwise union)
- Variable-size groups (1, 2, 5, 100)
- Overlapping vs non-overlapping polygons
- MultiPolygon inputs and outputs
- fp32 vs fp64 precision comparison (observability only for CONSTRUCTIVE)
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.kernels.constructive.segmented_union import segmented_union_all
from vibespatial.runtime import ExecutionMode


def _has_gpu():
    try:
        from vibespatial.cuda._runtime import get_cuda_runtime

        return get_cuda_runtime().available()
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU not available")


def _make_owned(geometries: list) -> OwnedGeometryArray:
    """Build an OwnedGeometryArray from a list of Shapely geometries."""
    return from_shapely_geometries(geometries)


def _shapely_segmented_union(geometries: list, group_offsets: np.ndarray) -> list:
    """Reference implementation using shapely.union_all per group."""
    n_groups = len(group_offsets) - 1
    results = []
    for g in range(n_groups):
        start = int(group_offsets[g])
        end = int(group_offsets[g + 1])
        block = geometries[start:end]
        valid = [g for g in block if g is not None and not shapely.is_empty(g)]
        if len(valid) == 0:
            results.append(Polygon())
        elif len(valid) == 1:
            results.append(valid[0])
        else:
            merged = shapely.union_all(np.asarray(valid, dtype=object))
            if merged is not None and not shapely.is_valid(merged):
                merged = shapely.make_valid(merged)
            results.append(merged if merged is not None else Polygon())
    return results


def _assert_geom_equal(gpu_geom, ref_geom, *, tolerance=1e-8):
    """Assert two geometries are spatially equivalent within tolerance."""
    if gpu_geom is None and ref_geom is None:
        return
    if gpu_geom is None or ref_geom is None:
        # One None, one not: check if the non-None is empty
        non_none = gpu_geom if gpu_geom is not None else ref_geom
        assert shapely.is_empty(non_none), f"Expected empty but got {non_none.wkt}"
        return
    if shapely.is_empty(gpu_geom) and shapely.is_empty(ref_geom):
        return
    # Symmetric difference area should be negligible
    sym_diff = shapely.symmetric_difference(
        shapely.make_valid(gpu_geom), shapely.make_valid(ref_geom)
    )
    sym_area = shapely.area(sym_diff)
    ref_area = shapely.area(ref_geom)
    if ref_area > 0:
        relative_error = sym_area / ref_area
        assert relative_error < tolerance, (
            f"Symmetric difference area ratio {relative_error} exceeds tolerance {tolerance}.\n"
            f"GPU: {gpu_geom.wkt[:200]}\nRef: {ref_geom.wkt[:200]}"
        )
    else:
        assert sym_area < tolerance, (
            f"Symmetric difference area {sym_area} exceeds tolerance {tolerance}"
        )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def overlapping_boxes():
    """Three overlapping boxes suitable for union."""
    return [box(0, 0, 2, 2), box(1, 1, 3, 3), box(2, 0, 4, 2)]


@pytest.fixture
def non_overlapping_boxes():
    """Three non-overlapping boxes."""
    return [box(0, 0, 1, 1), box(3, 3, 4, 4), box(6, 6, 7, 7)]


# ---------------------------------------------------------------------------
# CPU tests
# ---------------------------------------------------------------------------


class TestSegmentedUnionCPU:
    """Tests for the CPU path."""

    def test_single_element_groups(self):
        """Single-element groups should pass through unchanged."""
        geoms = [box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5)]
        owned = _make_owned(geoms)
        offsets = np.array([0, 1, 2, 3], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 3
        for gpu_g, ref_g in zip(result_geoms, ref):
            _assert_geom_equal(gpu_g, ref_g)

    def test_empty_groups(self):
        """Empty groups should produce empty polygons."""
        geoms = [box(0, 0, 1, 1)]
        owned = _make_owned(geoms)
        # Group 0: empty, Group 1: has 1 geom, Group 2: empty
        offsets = np.array([0, 0, 1, 1], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        result_geoms = result.to_shapely()
        assert len(result_geoms) == 3
        assert shapely.is_empty(result_geoms[0])
        assert not shapely.is_empty(result_geoms[1])
        assert shapely.is_empty(result_geoms[2])

    def test_two_element_overlapping(self, overlapping_boxes):
        """Two overlapping boxes should union correctly."""
        geoms = overlapping_boxes[:2]
        owned = _make_owned(geoms)
        offsets = np.array([0, 2], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 1
        _assert_geom_equal(result_geoms[0], ref[0])

    def test_two_element_non_overlapping(self, non_overlapping_boxes):
        """Two non-overlapping boxes should produce a MultiPolygon."""
        geoms = non_overlapping_boxes[:2]
        owned = _make_owned(geoms)
        offsets = np.array([0, 2], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 1
        _assert_geom_equal(result_geoms[0], ref[0])
        # Non-overlapping union should produce MultiPolygon
        assert result_geoms[0].geom_type == "MultiPolygon"

    def test_variable_group_sizes(self):
        """Variable group sizes: 1, 2, 5 elements."""
        geoms = [
            # Group 0: size 1
            box(0, 0, 1, 1),
            # Group 1: size 2
            box(2, 0, 4, 2),
            box(3, 1, 5, 3),
            # Group 2: size 5
            box(0, 5, 2, 7),
            box(1, 5, 3, 7),
            box(2, 5, 4, 7),
            box(3, 5, 5, 7),
            box(4, 5, 6, 7),
        ]
        owned = _make_owned(geoms)
        offsets = np.array([0, 1, 3, 8], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 3
        for i, (gpu_g, ref_g) in enumerate(zip(result_geoms, ref)):
            _assert_geom_equal(gpu_g, ref_g, tolerance=1e-6)

    def test_multipolygon_inputs(self):
        """MultiPolygon inputs should be handled correctly."""
        mp1 = MultiPolygon([box(0, 0, 1, 1), box(3, 3, 4, 4)])
        mp2 = MultiPolygon([box(0.5, 0.5, 1.5, 1.5), box(5, 5, 6, 6)])
        geoms = [mp1, mp2]
        owned = _make_owned(geoms)
        offsets = np.array([0, 2], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 1
        _assert_geom_equal(result_geoms[0], ref[0])

    def test_zero_groups(self):
        """Zero groups should return empty array."""
        geoms = []
        owned = _make_owned(geoms)
        offsets = np.array([0], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        assert result.row_count == 0

    def test_all_empty_groups(self):
        """All empty groups should produce all empty polygons."""
        geoms = []
        owned = _make_owned(geoms)
        offsets = np.array([0, 0, 0], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        result_geoms = result.to_shapely()
        assert len(result_geoms) == 2
        for g in result_geoms:
            assert shapely.is_empty(g)

    def test_many_groups_small(self):
        """Many small groups stress-tests the iteration."""
        n_groups = 50
        geoms = []
        offsets = [0]
        for g in range(n_groups):
            # 2 overlapping boxes per group
            geoms.append(box(g * 10, 0, g * 10 + 2, 2))
            geoms.append(box(g * 10 + 1, 1, g * 10 + 3, 3))
            offsets.append(len(geoms))
        offsets = np.array(offsets, dtype=np.int64)
        owned = _make_owned(geoms)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == n_groups
        for gpu_g, ref_g in zip(result_geoms, ref):
            _assert_geom_equal(gpu_g, ref_g)

    def test_none_in_group(self):
        """None geometries within a group should be skipped."""
        geoms = [box(0, 0, 1, 1), None, box(0.5, 0.5, 1.5, 1.5)]
        owned = _make_owned(geoms)
        offsets = np.array([0, 3], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        # Reference: filter Nones then union
        ref_block = [g for g in geoms if g is not None]
        ref_merged = shapely.union_all(np.asarray(ref_block, dtype=object))

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 1
        _assert_geom_equal(result_geoms[0], ref_merged)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


class TestSegmentedUnionGPU:
    """Tests for the GPU path (skip if no GPU available)."""

    @requires_gpu
    def test_single_element_groups_gpu(self):
        """GPU: single-element groups should pass through."""
        geoms = [box(0, 0, 1, 1), box(2, 2, 3, 3)]
        owned = _make_owned(geoms)
        offsets = np.array([0, 1, 2], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.GPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 2
        for gpu_g, ref_g in zip(result_geoms, ref):
            _assert_geom_equal(gpu_g, ref_g)

    @requires_gpu
    def test_empty_groups_gpu(self):
        """GPU: empty groups should produce empty polygons."""
        geoms = [box(0, 0, 1, 1)]
        owned = _make_owned(geoms)
        offsets = np.array([0, 0, 1], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()
        assert len(result_geoms) == 2
        assert shapely.is_empty(result_geoms[0])
        assert not shapely.is_empty(result_geoms[1])

    @requires_gpu
    def test_two_element_overlapping_gpu(self, overlapping_boxes):
        """GPU: two overlapping boxes should union correctly."""
        geoms = overlapping_boxes[:2]
        owned = _make_owned(geoms)
        offsets = np.array([0, 2], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.GPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 1
        _assert_geom_equal(result_geoms[0], ref[0])

    @requires_gpu
    def test_variable_group_sizes_gpu(self):
        """GPU: variable group sizes (1, 2, 5)."""
        geoms = [
            box(0, 0, 1, 1),
            box(2, 0, 4, 2),
            box(3, 1, 5, 3),
            box(0, 5, 2, 7),
            box(1, 5, 3, 7),
            box(2, 5, 4, 7),
            box(3, 5, 5, 7),
            box(4, 5, 6, 7),
        ]
        owned = _make_owned(geoms)
        offsets = np.array([0, 1, 3, 8], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.GPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 3
        for gpu_g, ref_g in zip(result_geoms, ref):
            _assert_geom_equal(gpu_g, ref_g, tolerance=1e-6)

    @requires_gpu
    def test_non_overlapping_produces_multipolygon_gpu(self, non_overlapping_boxes):
        """GPU: non-overlapping union should produce MultiPolygon."""
        geoms = non_overlapping_boxes[:2]
        owned = _make_owned(geoms)
        offsets = np.array([0, 2], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.GPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 1
        _assert_geom_equal(result_geoms[0], ref[0])

    @requires_gpu
    def test_multipolygon_inputs_gpu(self):
        """GPU: MultiPolygon inputs handled correctly."""
        mp1 = MultiPolygon([box(0, 0, 1, 1), box(3, 3, 4, 4)])
        mp2 = MultiPolygon([box(0.5, 0.5, 1.5, 1.5), box(5, 5, 6, 6)])
        geoms = [mp1, mp2]
        owned = _make_owned(geoms)
        offsets = np.array([0, 2], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.GPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 1
        _assert_geom_equal(result_geoms[0], ref[0])


# ---------------------------------------------------------------------------
# Auto-dispatch tests
# ---------------------------------------------------------------------------


class TestSegmentedUnionAutoDispatch:
    """Tests that auto dispatch falls back gracefully."""

    def test_auto_dispatch_produces_result(self):
        """AUTO dispatch should produce a correct result regardless of backend."""
        geoms = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
        owned = _make_owned(geoms)
        offsets = np.array([0, 2], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.AUTO)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 1
        _assert_geom_equal(result_geoms[0], ref[0])


# ---------------------------------------------------------------------------
# Precision observability tests (CONSTRUCTIVE stays fp64)
# ---------------------------------------------------------------------------


class TestSegmentedUnionPrecision:
    """Verify precision plan is wired through for observability."""

    def test_fp64_precision_accepted(self):
        """Explicit fp64 precision should work."""
        geoms = [box(0, 0, 1, 1)]
        owned = _make_owned(geoms)
        offsets = np.array([0, 1], dtype=np.int64)

        result = segmented_union_all(
            owned, offsets,
            dispatch_mode=ExecutionMode.CPU,
            precision="fp64",
        )
        assert result.row_count == 1

    def test_auto_precision_accepted(self):
        """AUTO precision should work (CONSTRUCTIVE always resolves to fp64)."""
        geoms = [box(0, 0, 1, 1), box(0.5, 0.5, 1.5, 1.5)]
        owned = _make_owned(geoms)
        offsets = np.array([0, 2], dtype=np.int64)

        result = segmented_union_all(
            owned, offsets,
            dispatch_mode=ExecutionMode.CPU,
            precision="auto",
        )
        ref = _shapely_segmented_union(geoms, offsets)
        result_geoms = result.to_shapely()
        _assert_geom_equal(result_geoms[0], ref[0])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSegmentedUnionEdgeCases:
    """Additional edge cases."""

    def test_single_group_all_geoms(self):
        """Single group containing all geometries."""
        geoms = [box(i, 0, i + 2, 2) for i in range(10)]
        owned = _make_owned(geoms)
        offsets = np.array([0, 10], dtype=np.int64)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        ref = _shapely_segmented_union(geoms, offsets)

        result_geoms = result.to_shapely()
        assert len(result_geoms) == 1
        _assert_geom_equal(result_geoms[0], ref[0])

    def test_group_offsets_int32(self):
        """int32 offsets should work."""
        geoms = [box(0, 0, 1, 1), box(0.5, 0.5, 1.5, 1.5)]
        owned = _make_owned(geoms)
        offsets = np.array([0, 2], dtype=np.int32)

        result = segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
        assert result.row_count == 1

    def test_invalid_offsets_raises(self):
        """Empty group_offsets array should raise."""
        geoms = [box(0, 0, 1, 1)]
        owned = _make_owned(geoms)
        offsets = np.array([], dtype=np.int64)

        with pytest.raises(ValueError, match="group_offsets must have length >= 1"):
            segmented_union_all(owned, offsets, dispatch_mode=ExecutionMode.CPU)
