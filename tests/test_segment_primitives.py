from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from vibespatial import (
    ExecutionMode,
    benchmark_segment_intersections,
    classify_segment_intersections,
    extract_segments,
    from_shapely_geometries,
    has_gpu_runtime,
    summarize_exact_local_events,
)
from vibespatial.runtime.hotpath_trace import reset_hotpath_trace, summarize_hotpath_trace
from vibespatial.runtime.residency import Residency


def test_segment_primitives_d2h_exports_are_runtime_accounted() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "vibespatial" / "spatial" / "segment_primitives.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    unnamed_runtime_exports: list[str] = []
    raw_cupy_exports: list[str] = []
    raw_scalar_syncs: list[str] = []
    cupy_reductions = {
        "all",
        "any",
        "sum",
        "count_nonzero",
        "max",
        "min",
        "nanmax",
        "nanmin",
    }

    def _contains_cupy_reduction(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "cp"
            and child.func.attr in cupy_reductions
            for child in ast.walk(node)
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "copy_device_to_host" and not any(
                keyword.arg == "reason" for keyword in node.keywords
            ):
                unnamed_runtime_exports.append(f"{path.relative_to(repo_root)}:{node.lineno}")
            if func.attr == "asnumpy":
                raw_cupy_exports.append(f"{path.relative_to(repo_root)}:{node.lineno}")
            if func.attr == "item":
                raw_scalar_syncs.append(f"{path.relative_to(repo_root)}:{node.lineno}")
            continue
        if (
            isinstance(func, ast.Name)
            and func.id in {"bool", "int", "float"}
            and node.args
            and _contains_cupy_reduction(node.args[0])
        ):
            raw_scalar_syncs.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    assert unnamed_runtime_exports == []
    assert raw_cupy_exports == []
    assert raw_scalar_syncs == []


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


@pytest.mark.gpu
def test_extract_segments_gpu_uses_host_structural_totals() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.spatial.segment_primitives import _extract_segments_gpu

    owned = from_shapely_geometries(
        [
            LineString([(0, 0), (1, 0), (1, 1)]),
            Polygon(
                [(10, 0), (16, 0), (16, 6), (10, 6), (10, 0)],
                [[(11, 1), (12, 1), (12, 2), (11, 1)]],
            ),
            MultiLineString([[(20, 0), (21, 0)], [(22, 0), (22, 2)]]),
            MultiPolygon([Polygon([(30, 0), (33, 0), (33, 3), (30, 0)])]),
        ],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    segments = _extract_segments_gpu(owned)
    events = get_d2h_transfer_events(clear=True)

    assert segments.count > 0
    assert [
        event.reason
        for event in events
        if event.reason == "segment extraction total-segments allocation fence"
    ] == []


@pytest.mark.gpu
def test_extract_segments_gpu_batches_device_only_family_total_fences() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.spatial.segment_primitives import _extract_segments_gpu

    owned = from_shapely_geometries(
        [
            LineString([(0, 0), (1, 0), (1, 1)]),
            Polygon(
                [(10, 0), (16, 0), (16, 6), (10, 6), (10, 0)],
                [[(11, 1), (12, 1), (12, 2), (11, 1)]],
            ),
            MultiLineString([[(20, 0), (21, 0)], [(22, 0), (22, 2)]]),
            MultiPolygon([Polygon([(30, 0), (33, 0), (33, 3), (30, 0)])]),
        ],
        residency=Residency.DEVICE,
    )
    device_only = owned.device_take(cp.arange(owned.row_count, dtype=cp.int64))

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    segments = _extract_segments_gpu(device_only)
    events = get_d2h_transfer_events(clear=True)

    assert segments.count > 0
    assert [
        event.reason
        for event in events
        if event.reason == "segment extraction total-segments allocation fence"
    ] == ["segment extraction total-segments allocation fence"]


@pytest.mark.gpu
def test_extract_segments_gpu_uses_device_structural_totals_for_nested_families() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.spatial.segment_primitives import _extract_segments_gpu

    owned = from_shapely_geometries(
        [
            Polygon(
                [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
                [[(1, 1), (2, 1), (2, 2), (1, 1)]],
            ),
            MultiPolygon(
                [
                    Polygon([(10, 0), (13, 0), (13, 3), (10, 0)]),
                    Polygon([(20, 0), (24, 0), (24, 4), (20, 0)]),
                ]
            ),
        ],
        residency=Residency.DEVICE,
    )
    device_only = owned.device_take(cp.arange(owned.row_count, dtype=cp.int64))

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    segments = _extract_segments_gpu(device_only)
    events = get_d2h_transfer_events(clear=True)

    assert segments.count == 13
    assert [
        event.reason
        for event in events
        if event.reason == "segment extraction total-segments allocation fence"
    ] == []


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


@pytest.mark.gpu
def test_segment_candidate_bounded_capacity_avoids_total_fence() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries(
        [
            LineString([(0, 0), (4, 4)]),
            LineString([(10, 0), (14, 4)]),
            LineString([(20, 0), (24, 4)]),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            LineString([(0, 4), (4, 0)]),
            LineString([(10, 4), (14, 0)]),
            LineString([(20, 4), (24, 0)]),
        ],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = classify_segment_intersections(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    events = get_d2h_transfer_events(clear=True)

    assert result.kind_names() == ["proper", "proper", "proper"]
    assert [
        event.reason
        for event in events
        if event.reason == "segment candidate total allocation fence"
    ] == []


@pytest.mark.gpu
def test_segment_candidate_large_capacity_batches_avoid_total_fence() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    row_count = 1025
    left = from_shapely_geometries(
        [
            LineString([(float(row) * 10.0, 0.0), (float(row) * 10.0 + 1.0, 1.0)])
            for row in range(row_count)
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            LineString([(float(row) * 10.0, 1.0), (float(row) * 10.0 + 1.0, 0.0)])
            for row in range(row_count)
        ],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = classify_segment_intersections(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    events = get_d2h_transfer_events(clear=True)

    assert result.count == row_count
    assert set(result.kind_names()) == {"proper"}
    assert [
        event.reason
        for event in events
        if event.reason == "segment candidate total allocation fence"
    ] == []


@pytest.mark.gpu
def test_segment_primitives_same_row_gpu_fast_path_skips_binary_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries(
        [
            LineString([(0, 0), (3, 3)]),
            LineString([(10, 0), (13, 3)]),
            LineString([(20, 0), (23, 3)]),
        ]
    )
    right = from_shapely_geometries(
        [
            LineString([(0, 3), (3, 0)]),
            LineString([(10, 3), (13, 0)]),
            LineString([(20, 3), (23, 0)]),
        ]
    )

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()
    result = classify_segment_intersections(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _require_same_row=True,
    )

    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.kind_names() == ["proper", "proper", "proper"]
    assert result.left_rows.tolist() == [0, 1, 2]
    assert result.right_rows.tolist() == [0, 1, 2]

    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    assert summary.get("segment.candidates.same_row_fast_path") == 1
    assert "segment.candidates.binary_search" not in summary


@pytest.mark.gpu
def test_segment_same_row_candidate_bounded_capacity_avoids_total_fence() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries(
        [
            LineString([(0, 0), (3, 3)]),
            LineString([(10, 0), (13, 3)]),
            LineString([(20, 0), (23, 3)]),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            LineString([(0, 3), (3, 0)]),
            LineString([(10, 3), (13, 0)]),
            LineString([(20, 3), (23, 0)]),
        ],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = classify_segment_intersections(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _require_same_row=True,
    )
    events = get_d2h_transfer_events(clear=True)

    assert result.kind_names() == ["proper", "proper", "proper"]
    assert [
        event.reason
        for event in events
        if event.reason == "segment same-row candidate total allocation fence"
    ] == []


@pytest.mark.gpu
def test_segment_primitives_same_row_fast_path_allows_large_left_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries(
        [
            LineString([(float(i), 0.0) for i in range(3001)]),
        ]
    )
    right = from_shapely_geometries(
        [
            LineString([(1500.5, -1.0), (1500.5, 1.0)]),
        ]
    )

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()
    result = classify_segment_intersections(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _require_same_row=True,
    )

    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.kind_names() == ["proper"]
    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    assert summary.get("segment.candidates.same_row_fast_path") == 1
    assert "segment.candidates.binary_search" not in summary


@pytest.mark.gpu
def test_segment_primitives_same_row_fast_path_swaps_large_right_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries(
        [
            LineString([(1500.5, -1.0), (1500.5, 1.0)]),
        ]
    )
    right = from_shapely_geometries(
        [
            LineString([(float(i), 0.0) for i in range(3001)]),
        ]
    )

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()
    result = classify_segment_intersections(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _require_same_row=True,
    )

    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.kind_names() == ["proper"]
    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    assert summary.get("segment.candidates.same_row_fast_path") == 1
    assert "segment.candidates.binary_search" not in summary


@pytest.mark.gpu
def test_segment_primitives_same_row_sort_sweep_path_matches_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries(
        [
            LineString([(0, 0), (3, 3)]),
            LineString([(10, 0), (13, 3)]),
            LineString([(20, 0), (23, 3)]),
        ]
    )
    right = from_shapely_geometries(
        [
            LineString([(0, 3), (3, 0)]),
            LineString([(10, 3), (13, 0)]),
            LineString([(20, 3), (23, 0)]),
        ]
    )

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()
    cpu = classify_segment_intersections(
        left,
        right,
        dispatch_mode=ExecutionMode.CPU,
        _require_same_row=True,
    )
    gpu = classify_segment_intersections(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _require_same_row=True,
        _use_same_row_fast_path=False,
    )

    assert gpu.runtime_selection.selected is ExecutionMode.GPU
    assert gpu.kind_names() == cpu.kind_names()
    assert gpu.left_rows.tolist() == cpu.left_rows.tolist()
    assert gpu.right_rows.tolist() == cpu.right_rows.tolist()

    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    assert summary.get("segment.candidates.same_row_fast_path") is None
    assert summary.get("segment.candidates.binary_search") == 1


def test_benchmark_segment_intersections_reports_degenerate_mix() -> None:
    left = from_shapely_geometries([LineString([(0, 0), (4, 4)]), LineString([(0, 0), (5, 0)])])
    right = from_shapely_geometries([LineString([(0, 4), (4, 0)]), LineString([(2, 0), (7, 0)])])

    benchmark = benchmark_segment_intersections(left, right)

    assert benchmark.candidate_pairs >= 2
    assert benchmark.proper_pairs >= 1
    assert benchmark.overlap_pairs >= 1
    assert benchmark.ambiguous_pairs >= 1


def test_exact_local_event_summary_counts_endpoint_and_intersection_events() -> None:
    left = from_shapely_geometries(
        [
            LineString([(0, 0), (4, 4)]),
            LineString([(10, 0), (14, 4)]),
        ]
    )
    right = from_shapely_geometries(
        [
            LineString([(0, 4), (4, 0)]),
            LineString([(10, 4), (14, 0)]),
        ]
    )

    summary = summarize_exact_local_events(left, right, dispatch_mode=ExecutionMode.CPU, _require_same_row=True)

    assert summary.candidate_pairs == 2
    assert summary.point_intersection_count == 2
    assert summary.parallel_or_colinear_candidate_count == 0
    assert summary.row_point_intersection_counts.tolist() == [1, 1]
    assert summary.exact_event_counts.tolist() == [5, 5]
    assert summary.exact_interval_upper_bounds.tolist() == [4, 4]
    assert summary.max_exact_events == 5


@pytest.mark.gpu
def test_exact_local_event_summary_gpu_matches_cpu_for_same_row_workload() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries(
        [
            LineString([(0, 0), (4, 4)]),
            LineString([(10, 0), (14, 4)]),
        ]
    )
    right = from_shapely_geometries(
        [
            LineString([(0, 4), (4, 0)]),
            LineString([(10, 4), (14, 0)]),
        ]
    )

    cpu = summarize_exact_local_events(left, right, dispatch_mode=ExecutionMode.CPU, _require_same_row=True)
    gpu = summarize_exact_local_events(left, right, dispatch_mode=ExecutionMode.GPU, _require_same_row=True)

    assert gpu.runtime_selection.selected is ExecutionMode.GPU
    assert gpu.candidate_pairs == cpu.candidate_pairs
    assert gpu.point_intersection_count == cpu.point_intersection_count
    assert gpu.parallel_or_colinear_candidate_count == cpu.parallel_or_colinear_candidate_count
    assert gpu.row_point_intersection_counts.tolist() == cpu.row_point_intersection_counts.tolist()
    assert gpu.exact_event_counts.tolist() == cpu.exact_event_counts.tolist()
    assert gpu.exact_interval_upper_bounds.tolist() == cpu.exact_interval_upper_bounds.tolist()
