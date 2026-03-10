from __future__ import annotations

from shapely.geometry import LineString, Polygon

from vibespatial import (
    benchmark_segment_filter,
    extract_segment_mbrs,
    from_shapely_geometries,
    generate_segment_mbr_pairs,
)


def test_extract_segment_mbrs_captures_line_and_polygon_segments() -> None:
    owned = from_shapely_geometries(
        [
            LineString([(0, 0), (2, 0), (2, 2)]),
            Polygon([(10, 10), (14, 10), (14, 14), (10, 10)]),
        ]
    )

    segments = extract_segment_mbrs(owned)

    assert segments.count == 5
    assert segments.row_indices.tolist().count(0) == 2
    assert segments.row_indices.tolist().count(1) == 3


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
