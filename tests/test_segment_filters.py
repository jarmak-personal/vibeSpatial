from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon

from vibespatial import (
    benchmark_segment_filter,
    extract_segment_mbrs,
    from_shapely_geometries,
    generate_segment_mbr_pairs,
)


def _to_numpy(arr) -> np.ndarray:
    """Convert a device or host array to a numpy array for assertions."""
    try:
        import cupy as cp

        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except ModuleNotFoundError:
        pass
    return np.asarray(arr)


def test_extract_segment_mbrs_captures_line_and_polygon_segments() -> None:
    owned = from_shapely_geometries(
        [
            LineString([(0, 0), (2, 0), (2, 2)]),
            Polygon([(10, 10), (14, 10), (14, 14), (10, 10)]),
        ]
    )

    segments = extract_segment_mbrs(owned)

    assert segments.count == 5
    row_list = _to_numpy(segments.row_indices).tolist()
    assert row_list.count(0) == 2
    assert row_list.count(1) == 3


def test_extract_segment_mbrs_to_host() -> None:
    """Verify to_host() produces numpy arrays with correct values."""
    owned = from_shapely_geometries(
        [
            LineString([(0, 0), (2, 0), (2, 2)]),
            Polygon([(10, 10), (14, 10), (14, 14), (10, 10)]),
        ]
    )

    segments = extract_segment_mbrs(owned).to_host()

    assert isinstance(segments.row_indices, np.ndarray)
    assert isinstance(segments.segment_indices, np.ndarray)
    assert isinstance(segments.bounds, np.ndarray)
    assert segments.count == 5
    assert segments.bounds.shape == (5, 4)


def test_extract_segment_mbrs_bounds_correctness() -> None:
    """Verify segment MBR bounds match expected values for a simple line."""
    owned = from_shapely_geometries(
        [
            LineString([(1, 2), (3, 4), (5, 2)]),
        ]
    )

    segments = extract_segment_mbrs(owned).to_host()

    assert segments.count == 2
    # Segment 0: (1,2) -> (3,4)
    np.testing.assert_allclose(segments.bounds[0], [1.0, 2.0, 3.0, 4.0])
    # Segment 1: (3,4) -> (5,2)
    np.testing.assert_allclose(segments.bounds[1], [3.0, 2.0, 5.0, 4.0])


def test_generate_segment_mbr_pairs_reduces_edge_pair_space() -> None:
    left = from_shapely_geometries(
        [
            Polygon([(0, 0), (20, 0), (20, 20), (10, 10), (0, 20), (0, 0)]),
        ]
    )
    right = from_shapely_geometries(
        [
            Polygon([(15, -5), (25, -5), (25, 25), (15, 25), (15, -5)]),
        ]
    )

    left_segments = extract_segment_mbrs(left)
    right_segments = extract_segment_mbrs(right)
    filtered = generate_segment_mbr_pairs(left, right)

    assert filtered.count < (left_segments.count * right_segments.count)
    assert filtered.count > 0


def test_benchmark_segment_filter_reports_reduction() -> None:
    left = from_shapely_geometries([LineString([(0, 0), (100, 0), (100, 100), (0, 100)])])
    right = from_shapely_geometries([LineString([(50, -10), (50, 110), (60, 110), (60, -10)])])

    benchmark = benchmark_segment_filter(left, right)

    assert benchmark.naive_segment_pairs > benchmark.filtered_segment_pairs
