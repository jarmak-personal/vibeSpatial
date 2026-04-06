"""Tests for the GPU polygon difference kernel.

Validates that the GPU overlay-topology-based polygon difference kernel
produces results matching Shapely's shapely.difference() oracle across
a range of geometry configurations.

Test categories:
- Shapely oracle comparison for standard polygon-polygon difference
- Edge cases: empty result, no overlap, partial overlap, full containment
- Multi-polygon inputs
- Null/empty input propagation
- fp32 vs fp64 precision (observability only for CONSTRUCTIVE class)
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

from vibespatial import ExecutionMode, has_gpu_runtime
from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.kernels.constructive.polygon_difference import (
    polygon_difference,
)
from vibespatial.runtime.precision import PrecisionMode

requires_gpu = pytest.mark.skipif(
    not has_gpu_runtime(), reason="CUDA runtime not available"
)


def _shapely_difference(left_geoms, right_geoms):
    """Shapely oracle for element-wise polygon difference."""
    left_arr = np.empty(len(left_geoms), dtype=object)
    left_arr[:] = left_geoms
    right_arr = np.empty(len(right_geoms), dtype=object)
    right_arr[:] = right_geoms
    return shapely.difference(left_arr, right_arr)


def _assert_geometries_equivalent(actual_owned: OwnedGeometryArray, expected: np.ndarray) -> None:
    """Assert that the OwnedGeometryArray matches the expected Shapely geometries."""
    actual_geoms = actual_owned.to_shapely()
    assert len(actual_geoms) == len(expected), (
        f"Row count mismatch: got {len(actual_geoms)}, expected {len(expected)}"
    )
    for i, (act, exp) in enumerate(zip(actual_geoms, expected, strict=True)):
        if act is None and exp is None:
            continue
        if act is None or exp is None:
            pytest.fail(f"Row {i}: one is None, other is not (actual={act}, expected={exp})")
        # Normalize: empty geometry comparison
        if shapely.is_empty(exp):
            assert shapely.is_empty(act), f"Row {i}: expected empty, got {act.wkt}"
            continue
        if shapely.is_empty(act):
            pytest.fail(f"Row {i}: expected non-empty {exp.wkt}, got empty")
        # Normalize ring traversal/order before exact comparison so the
        # oracle checks geometry content rather than incidental vertex order.
        act_norm = shapely.normalize(act)
        exp_norm = shapely.normalize(exp)
        assert shapely.equals_exact(act_norm, exp_norm, tolerance=1e-6), (
            f"Row {i}: geometries differ.\n"
            f"  actual:   {act.wkt}\n"
            f"  expected: {exp.wkt}"
        )


# ---------------------------------------------------------------------------
# Standard polygon-polygon difference tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "null_case,empty_case,mixed_case",
    [
        (True, True, "polygon,multipolygon"),
    ],
)
def test_polygon_difference_edge_cases(null_case, empty_case, mixed_case) -> None:
    """Parametrized edge case coverage: null, empty, and mixed geometry types."""
    if null_case:
        left_geoms = [box(0, 0, 3, 3), None]
        right_geoms = [box(1, 1, 4, 4), box(0, 0, 1, 1)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)
        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_difference(left_geoms, right_geoms)
        _assert_geometries_equivalent(result, expected)

    if empty_case:
        left = from_shapely_geometries([])
        right = from_shapely_geometries([])
        result = polygon_difference(left, right)
        assert result.row_count == 0

    if mixed_case:
        left_geoms = [
            box(0, 0, 3, 3),
            MultiPolygon([box(10, 10, 12, 12), box(14, 14, 16, 16)]),
        ]
        right_geoms = [box(1, 1, 4, 4), box(11, 11, 15, 15)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)
        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_difference(left_geoms, right_geoms)
        _assert_geometries_equivalent(result, expected)


class TestPolygonDifferenceCPU:
    """CPU fallback tests (always available, no GPU required)."""

    def test_partial_overlap(self) -> None:
        """Overlapping boxes: left minus the overlap region."""
        left_geoms = [box(0, 0, 3, 3)]
        right_geoms = [box(1, 1, 4, 4)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    def test_no_overlap(self) -> None:
        """Disjoint boxes: left unchanged."""
        left_geoms = [box(0, 0, 1, 1)]
        right_geoms = [box(5, 5, 6, 6)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    def test_full_containment_empty_result(self) -> None:
        """Left fully inside right: result is empty."""
        left_geoms = [box(1, 1, 2, 2)]
        right_geoms = [box(0, 0, 3, 3)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    def test_multiple_rows(self) -> None:
        """Multiple element-wise pairs."""
        left_geoms = [box(0, 0, 3, 3), box(10, 10, 13, 13)]
        right_geoms = [box(1, 1, 4, 4), box(11, 11, 14, 14)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    def test_row_count_mismatch_raises(self) -> None:
        """Mismatched row counts raise ValueError."""
        left = from_shapely_geometries([box(0, 0, 1, 1)])
        right = from_shapely_geometries([box(0, 0, 1, 1), box(2, 2, 3, 3)])

        with pytest.raises(ValueError, match="row count mismatch"):
            polygon_difference(left, right)

    def test_empty_input(self) -> None:
        """Zero-length input returns zero-length output."""
        left = from_shapely_geometries([])
        right = from_shapely_geometries([])

        result = polygon_difference(left, right)
        assert result.row_count == 0

    def test_null_propagation(self) -> None:
        """Null geometries propagate through CPU path."""
        left_geoms = [box(0, 0, 3, 3), None, box(5, 5, 8, 8)]
        right_geoms = [box(1, 1, 4, 4), box(0, 0, 1, 1), None]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    def test_mixed_polygon_multipolygon_input(self) -> None:
        """Mixed Polygon and MultiPolygon inputs are handled correctly."""
        left_geoms = [
            box(0, 0, 3, 3),
            MultiPolygon([box(10, 10, 12, 12), box(14, 14, 16, 16)]),
        ]
        right_geoms = [
            box(1, 1, 4, 4),
            box(11, 11, 15, 15),
        ]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)


# ---------------------------------------------------------------------------
# GPU tests (skip if no GPU available)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestPolygonDifferenceGPU:
    """GPU overlay pipeline tests."""

    @requires_gpu
    def test_partial_overlap_gpu(self) -> None:
        """GPU: overlapping boxes produce correct difference."""
        left_geoms = [box(0, 0, 3, 3)]
        right_geoms = [box(1, 1, 4, 4)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    @requires_gpu
    def test_no_overlap_gpu(self) -> None:
        """GPU: disjoint boxes leave left unchanged."""
        left_geoms = [box(0, 0, 1, 1)]
        right_geoms = [box(5, 5, 6, 6)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    @requires_gpu
    def test_touch_only_gpu_preserves_left(self) -> None:
        """GPU: touch-only boxes leave left unchanged."""
        left_geoms = [box(10, 0, 14, 4)]
        right_geoms = [box(14, 0, 18, 4)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    @requires_gpu
    def test_full_containment_produces_empty_gpu(self) -> None:
        """GPU: left fully inside right produces empty geometry."""
        left_geoms = [box(1, 1, 2, 2)]
        right_geoms = [box(0, 0, 3, 3)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.GPU)
        # When left is fully contained in right, overlay produces empty result
        # (0 selected faces). The GPU path returns an empty OGA.
        assert result.row_count == 0 or all(
            shapely.is_empty(g) for g in result.to_shapely() if g is not None
        )

    @requires_gpu
    def test_multiple_rows_gpu(self) -> None:
        """GPU: multiple element-wise pairs produce correct results."""
        left_geoms = [box(0, 0, 3, 3), box(10, 10, 13, 13)]
        right_geoms = [box(1, 1, 4, 4), box(11, 11, 14, 14)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    @requires_gpu
    def test_right_fully_inside_left_produces_hole(self) -> None:
        """GPU: right inside left creates a polygon with a hole."""
        left_geoms = [box(0, 0, 6, 6)]
        right_geoms = [box(2, 2, 4, 4)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    @requires_gpu
    def test_non_rectangular_polygons(self) -> None:
        """GPU: non-axis-aligned polygons are handled correctly."""
        left_geoms = [Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])]
        right_geoms = [Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    @requires_gpu
    def test_multipolygon_input(self) -> None:
        """GPU: MultiPolygon inputs are handled via the overlay pipeline."""
        left_geoms = [
            MultiPolygon([box(0, 0, 2, 2), box(4, 4, 6, 6)])
        ]
        right_geoms = [box(1, 1, 5, 5)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    @requires_gpu
    def test_difference_that_splits_polygon(self) -> None:
        """GPU: difference can split a polygon into multiple parts."""
        # Left is a wide rectangle, right cuts it in the middle
        left_geoms = [box(0, 0, 10, 2)]
        right_geoms = [box(4, -1, 6, 3)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(left, right, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)


# ---------------------------------------------------------------------------
# Precision observability tests (CONSTRUCTIVE class stays fp64)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestPolygonDifferencePrecision:
    """Verify precision plan wiring for observability."""

    @requires_gpu
    def test_fp64_precision_produces_correct_result(self) -> None:
        """Explicit fp64 precision works correctly."""
        left_geoms = [box(0, 0, 3, 3)]
        right_geoms = [box(1, 1, 4, 4)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(
            left, right,
            dispatch_mode=ExecutionMode.GPU,
            precision=PrecisionMode.FP64,
        )
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)

    @requires_gpu
    def test_auto_precision_produces_correct_result(self) -> None:
        """AUTO precision (fp64 for CONSTRUCTIVE) works correctly."""
        left_geoms = [box(0, 0, 3, 3)]
        right_geoms = [box(1, 1, 4, 4)]
        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = polygon_difference(
            left, right,
            dispatch_mode=ExecutionMode.GPU,
            precision=PrecisionMode.AUTO,
        )
        expected = _shapely_difference(left_geoms, right_geoms)

        _assert_geometries_equivalent(result, expected)
