from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from vibespatial import (
    ExecutionMode,
    benchmark_segment_intersections,
    classify_segment_intersections,
    extract_segments,
    from_shapely_geometries,
    has_gpu_runtime,
)


def test_extract_segments_reads_owned_buffers_without_materialization() -> None:
    owned = from_shapely_geometries(
        [
            LineString([(0, 0), (2, 0), (2, 2)]),
            Polygon([(10, 10), (14, 10), (14, 14), (10, 10)]),
        ]
    )

    segments = extract_segments(owned)

    assert segments.count == 5
    assert all(event.kind.value != "materialization" for event in owned.diagnostics)


def test_segment_primitives_classify_proper_cross() -> None:
    left = from_shapely_geometries([LineString([(0, 0), (4, 4)])])
    right = from_shapely_geometries([LineString([(0, 4), (4, 0)])])

    result = classify_segment_intersections(left, right)

    assert result.kind_names() == ["proper"]
    assert result.ambiguous_rows.size == 0
    assert np.allclose([result.point_x[0], result.point_y[0]], [2.0, 2.0])


def test_segment_primitives_classify_shared_vertex_touch() -> None:
    left = from_shapely_geometries([LineString([(0, 0), (2, 2)])])
    right = from_shapely_geometries([LineString([(2, 2), (4, 0)])])

    result = classify_segment_intersections(left, right)

    assert result.kind_names() == ["touch"]
    assert result.ambiguous_rows.tolist() == [0]
    assert np.allclose([result.point_x[0], result.point_y[0]], [2.0, 2.0])


def test_segment_primitives_classify_collinear_overlap() -> None:
    left = from_shapely_geometries([LineString([(0, 0), (5, 0)])])
    right = from_shapely_geometries([LineString([(2, 0), (7, 0)])])

    result = classify_segment_intersections(left, right)

    assert result.kind_names() == ["overlap"]
    assert result.ambiguous_rows.tolist() == [0]
    assert np.allclose(
        [result.overlap_x0[0], result.overlap_y0[0], result.overlap_x1[0], result.overlap_y1[0]],
        [2.0, 0.0, 5.0, 0.0],
    )


def test_segment_primitives_classify_zero_length_piece_as_touch() -> None:
    left = from_shapely_geometries([LineString([(1, 1), (1, 1)])])
    right = from_shapely_geometries([LineString([(0, 0), (2, 2)])])

    result = classify_segment_intersections(left, right)

    assert result.kind_names() == ["touch"]
    assert result.ambiguous_rows.tolist() == [0]
    assert np.allclose([result.point_x[0], result.point_y[0]], [1.0, 1.0])


def test_segment_primitives_preserve_ring_edge_corner_cases() -> None:
    left = from_shapely_geometries([Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])])
    right = from_shapely_geometries([LineString([(4, 0), (4, 4)])])

    result = classify_segment_intersections(left, right)

    assert "overlap" in result.kind_names()
    assert result.candidate_pairs >= 1


@pytest.mark.gpu
def test_segment_primitives_explicit_gpu_request_matches_cpu_for_proper_cross() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([LineString([(0, 0), (4, 4)])])
    right = from_shapely_geometries([LineString([(0, 4), (4, 0)])])

    cpu = classify_segment_intersections(left, right, dispatch_mode=ExecutionMode.CPU)
    gpu = classify_segment_intersections(left, right, dispatch_mode=ExecutionMode.GPU)

    assert gpu.runtime_selection.selected is ExecutionMode.GPU
    assert gpu.kind_names() == cpu.kind_names()
    assert gpu.ambiguous_rows.tolist() == cpu.ambiguous_rows.tolist()
    assert np.allclose([gpu.point_x[0], gpu.point_y[0]], [cpu.point_x[0], cpu.point_y[0]])
    assert gpu.device_state is not None


@pytest.mark.gpu
def test_segment_primitives_explicit_gpu_request_matches_cpu_for_ambiguous_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cases = (
        (
            from_shapely_geometries([LineString([(0, 0), (2, 2)])]),
            from_shapely_geometries([LineString([(2, 2), (4, 0)])]),
        ),
        (
            from_shapely_geometries([LineString([(0, 0), (5, 0)])]),
            from_shapely_geometries([LineString([(2, 0), (7, 0)])]),
        ),
        (
            from_shapely_geometries([LineString([(1, 1), (1, 1)])]),
            from_shapely_geometries([LineString([(0, 0), (2, 2)])]),
        ),
    )

    for left, right in cases:
        cpu = classify_segment_intersections(left, right, dispatch_mode=ExecutionMode.CPU)
        gpu = classify_segment_intersections(left, right, dispatch_mode=ExecutionMode.GPU)
        assert gpu.kind_names() == cpu.kind_names()
        assert gpu.ambiguous_rows.tolist() == cpu.ambiguous_rows.tolist()
        assert np.allclose(gpu.point_x, cpu.point_x, equal_nan=True)
        assert np.allclose(gpu.point_y, cpu.point_y, equal_nan=True)
        assert np.allclose(gpu.overlap_x0, cpu.overlap_x0, equal_nan=True)
        assert np.allclose(gpu.overlap_y0, cpu.overlap_y0, equal_nan=True)
        assert np.allclose(gpu.overlap_x1, cpu.overlap_x1, equal_nan=True)
        assert np.allclose(gpu.overlap_y1, cpu.overlap_y1, equal_nan=True)
        assert gpu.device_state is not None


def test_benchmark_segment_intersections_reports_degenerate_mix() -> None:
    left = from_shapely_geometries([LineString([(0, 0), (4, 4)]), LineString([(0, 0), (5, 0)])])
    right = from_shapely_geometries([LineString([(0, 4), (4, 0)]), LineString([(2, 0), (7, 0)])])

    benchmark = benchmark_segment_intersections(left, right)

    assert benchmark.candidate_pairs >= 2
    assert benchmark.proper_pairs >= 1
    assert benchmark.overlap_pairs >= 1
    assert benchmark.ambiguous_pairs >= 1
