from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from vibespatial import (
    ExecutionMode,
    from_shapely_geometries,
    has_gpu_runtime,
    overlay_difference_owned,
    overlay_identity_owned,
    overlay_symmetric_difference_owned,
    overlay_union_owned,
    spatial_overlay_owned,
)


def _normalize(values):
    return [None if value is None else shapely.normalize(value) for value in values]


def _assert_geometry_lists_equal(actual, expected) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(_normalize(actual), _normalize(expected), strict=True):
        if left is None or right is None:
            assert left is right
            continue
        assert bool(shapely.equals(left, right))


@pytest.mark.gpu
def test_gpu_overlay_union_matches_shapely_for_overlapping_rectangles() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 4, 4)])
    right = from_shapely_geometries([box(2, 2, 6, 6)])

    actual = overlay_union_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.union(np.asarray(left.to_shapely(), dtype=object), np.asarray(right.to_shapely(), dtype=object))

    _assert_geometry_lists_equal(actual.to_shapely(), expected.tolist())
    assert actual.runtime_history[-1].selected is ExecutionMode.GPU


@pytest.mark.gpu
def test_gpu_overlay_difference_matches_shapely() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    donut = Polygon(
        shell=[(0, 0), (8, 0), (8, 8), (0, 8), (0, 0)],
        holes=[[(3, 3), (5, 3), (5, 5), (3, 5), (3, 3)]],
    )
    left = from_shapely_geometries([donut])
    right = from_shapely_geometries([box(1, 1, 7, 7)])

    actual = overlay_difference_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.difference(np.asarray(left.to_shapely(), dtype=object), np.asarray(right.to_shapely(), dtype=object))

    _assert_geometry_lists_equal(actual.to_shapely(), expected.tolist())


@pytest.mark.gpu
def test_gpu_overlay_symmetric_difference_groups_row_outputs_into_multipolygon() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 4, 4)])
    right = from_shapely_geometries([box(2, 2, 6, 6)])

    actual = overlay_symmetric_difference_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.symmetric_difference(
        np.asarray(left.to_shapely(), dtype=object),
        np.asarray(right.to_shapely(), dtype=object),
    )

    _assert_geometry_lists_equal(actual.to_shapely(), expected.tolist())
    assert actual.row_count == 1
    assert actual.to_shapely()[0].geom_type == "MultiPolygon"


@pytest.mark.gpu
def test_gpu_overlay_union_of_disjoint_boxes_emits_single_multipolygon_row() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 2, 2)])
    right = from_shapely_geometries([box(4, 0, 6, 2)])

    actual = overlay_union_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.union(np.asarray(left.to_shapely(), dtype=object), np.asarray(right.to_shapely(), dtype=object))

    _assert_geometry_lists_equal(actual.to_shapely(), expected.tolist())
    assert actual.row_count == 1
    assert actual.to_shapely()[0].geom_type == "MultiPolygon"


@pytest.mark.gpu
def test_gpu_overlay_identity_geometry_matches_left_input() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 4, 4)])
    right = from_shapely_geometries([box(2, 2, 6, 6)])

    actual = overlay_identity_owned(left, right, dispatch_mode=ExecutionMode.GPU)

    _assert_geometry_lists_equal(actual.to_shapely(), left.to_shapely())


@pytest.mark.gpu
def test_gpu_overlay_variants_remain_polygon_only() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([shapely.LineString([(0, 0), (1, 1)])])
    right = from_shapely_geometries([box(0, 0, 2, 2)])

    with pytest.raises(NotImplementedError, match="polygon"):
        overlay_union_owned(left, right, dispatch_mode=ExecutionMode.GPU)


# ---------------------------------------------------------------------------
# lyy.11: difference and symmetric_difference containment/disjointness bypass
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_difference_containment_bypass_all_contained_produces_empty() -> None:
    """When all left polygons are fully contained in the corridor,
    difference should produce zero result rows (L_i - R_j = empty)."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    corridor = box(0, 0, 10, 10)
    left = from_shapely_geometries([box(1, 1, 3, 3), box(5, 5, 7, 7), box(2, 4, 4, 6)])
    right = from_shapely_geometries([corridor])

    result = spatial_overlay_owned(left, right, how="difference", dispatch_mode=ExecutionMode.GPU)
    assert result.row_count == 0, (
        f"Expected 0 results for all-contained difference, got {result.row_count}"
    )


@pytest.mark.gpu
def test_difference_containment_bypass_mixed_contained_and_crossing() -> None:
    """Contained polygons produce empty; boundary-crossing polygons produce
    the clipped remainder.  Only the crossing result should appear."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    corridor = box(0, 0, 10, 10)
    contained = box(1, 1, 3, 3)
    crossing = box(8, 8, 12, 12)

    left = from_shapely_geometries([contained, crossing])
    right = from_shapely_geometries([corridor])

    result = spatial_overlay_owned(left, right, how="difference", dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()

    # Only the crossing polygon should produce a result.
    assert result.row_count == 1, (
        f"Expected 1 result (crossing only), got {result.row_count}"
    )

    expected = shapely.difference(crossing, corridor)
    actual = result_geoms[0]
    assert bool(shapely.equals(shapely.normalize(actual), shapely.normalize(expected))), (
        f"Difference result mismatch: {actual} vs {expected}"
    )


@pytest.mark.gpu
def test_difference_containment_bypass_correctness_matches_shapely() -> None:
    """Many-vs-one difference matches Shapely oracle for a mix of
    contained and boundary-crossing polygons."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    corridor = box(0, 0, 10, 10)
    left_geoms = [
        box(1, 1, 3, 3),    # contained -> empty
        box(5, 5, 7, 7),    # contained -> empty
        box(8, 8, 12, 12),  # crossing -> partial
        box(0, 0, 5, 5),    # touching boundary -> may be contained or crossing
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries([corridor])

    result = spatial_overlay_owned(left, right, how="difference", dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()

    # Compute Shapely oracle: per-pair difference, filter out empties.
    expected_geoms = []
    for lg in left_geoms:
        diff = shapely.difference(lg, corridor)
        if diff is not None and not diff.is_empty:
            expected_geoms.append(diff)

    assert result.row_count == len(expected_geoms), (
        f"Result count {result.row_count} != expected {len(expected_geoms)}"
    )

    # Sort by area for stable comparison.
    result_sorted = sorted(result_geoms, key=lambda g: g.area)
    expected_sorted = sorted(expected_geoms, key=lambda g: g.area)

    for actual, expected in zip(result_sorted, expected_sorted, strict=True):
        assert abs(actual.area - expected.area) < 1e-6, (
            f"Area mismatch: {actual.area} vs {expected.area}"
        )


@pytest.mark.gpu
def test_symmetric_difference_bypass_still_produces_correct_results() -> None:
    """Symmetric difference via the owned overlay pipeline produces correct
    results (bypass is not eligible for sym_diff, so full overlay runs)."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    corridor = box(0, 0, 10, 10)
    crossing = box(8, 8, 12, 12)

    left = from_shapely_geometries([crossing])
    right = from_shapely_geometries([corridor])

    result = spatial_overlay_owned(
        left, right, how="symmetric_difference", dispatch_mode=ExecutionMode.GPU,
    )
    result_geoms = result.to_shapely()

    # Shapely oracle: L XOR R.
    expected = shapely.symmetric_difference(crossing, corridor)

    assert result.row_count >= 1
    # The result area should match the expected symmetric difference area.
    total_area = sum(g.area for g in result_geoms if g is not None and not g.is_empty)
    assert abs(total_area - expected.area) < 1e-6, (
        f"Area mismatch: {total_area} vs {expected.area}"
    )


@pytest.mark.gpu
def test_difference_bypass_intersection_still_works() -> None:
    """Verify that the existing intersection containment bypass is unaffected
    by the lyy.11 changes."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    corridor = box(0, 0, 10, 10)
    contained = box(1, 1, 3, 3)
    crossing = box(8, 8, 12, 12)

    left = from_shapely_geometries([contained, crossing])
    right = from_shapely_geometries([corridor])

    result = spatial_overlay_owned(
        left, right, how="intersection", dispatch_mode=ExecutionMode.GPU,
    )
    result_geoms = result.to_shapely()

    # For intersection: contained -> pass-through (L_i), crossing -> clipped.
    assert result.row_count == 2, (
        f"Expected 2 results, got {result.row_count}"
    )

    # Verify areas match Shapely oracle.
    expected_areas = sorted([
        shapely.intersection(contained, corridor).area,
        shapely.intersection(crossing, corridor).area,
    ])
    actual_areas = sorted([g.area for g in result_geoms])

    for actual_a, expected_a in zip(actual_areas, expected_areas, strict=True):
        assert abs(actual_a - expected_a) < 1e-6, (
            f"Area mismatch: {actual_a} vs {expected_a}"
        )
