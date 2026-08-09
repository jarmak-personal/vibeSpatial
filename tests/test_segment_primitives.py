from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

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


def test_segment_candidates_use_device_physicalized_bounded_pages() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src" / "vibespatial" / "spatial" / "segment_primitives.py").read_text()

    assert "segment extraction total-segments allocation fence" not in source
    assert "segment same-row candidate total allocation fence" not in source
    assert "segment filtered candidate total allocation fence" not in source
    assert "segment outlier candidate total allocation fence" not in source
    assert "segment candidate total allocation fence" not in source
    assert "segment candidate batch-boundary host export" not in source
    assert "segment candidate batch-boundary-offset host export" not in source
    assert "available contiguous classification budget" not in source
    assert "DeviceSegmentIntersectionCandidatePages" in source
    assert "paged_candidate_classification_gpu" in source
    assert "count_sweep_overlap_candidates" in source
    assert "scatter_sweep_overlap_candidates" in source
    assert "segment sweep compact candidate allocation packet" not in source
    assert "segment sweep compact page-boundary packet" not in source
    assert "d_counts = cp.zeros(left_count" in source
    assert "d_selected_left = compact_indices(" in source
    assert "output_capacity = selected_count * tier_width" in source
    assert "d_counts > tier_start" in source
    assert "d_left_pair = d_batch_left[d_local_left]" not in source
    assert "d_candidate_active = d_local_pair < d_chunk_counts[d_local_left]" not in source


def test_candidate_page_accumulator_switches_before_contiguous_overflow() -> None:
    import vibespatial.spatial.segment_primitives as segment_primitives

    segment_rows = np.arange(8, dtype=np.int32)
    segment_ids = np.arange(100, 108, dtype=np.int32)
    table = SimpleNamespace(
        row_indices=segment_rows,
        segment_indices=segment_ids,
    )
    consumed = []
    accumulator = segment_primitives._DeviceCandidatePageAccumulator(
        left=table,
        right=table,
        contiguous_pair_budget=3,
        page_consumer=consumed.append,
    )

    accumulator.append(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([4, 5], dtype=np.int32),
    )
    assert consumed == []
    accumulator.append(
        np.asarray([2, 3], dtype=np.int32),
        np.asarray([6, 7], dtype=np.int32),
    )
    accumulator.append(
        np.asarray([4], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
    )
    marker = accumulator.finish(runtime=None)

    assert isinstance(
        marker,
        segment_primitives.DeviceSegmentIntersectionCandidatePages,
    )
    assert marker.count == 5
    assert [page.count for page in consumed] == [2, 2, 1]
    assert consumed[0].left_rows.tolist() == [0, 1]
    assert consumed[0].right_segments.tolist() == [104, 105]


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
def test_extract_segments_gpu_uses_device_capacity_for_nonindexed_variable_buffers() -> None:
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
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    segments = _extract_segments_gpu(owned)
    events = get_d2h_transfer_events(clear=True)

    assert segments.count == 13
    assert [
        event.reason
        for event in events
        if event.reason == "segment extraction total-segments allocation fence"
    ] == []


@pytest.mark.gpu
def test_extract_segments_gpu_compacts_single_ring_coordinate_capacity() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )
    from vibespatial.spatial.segment_primitives import _extract_segments_gpu

    x = cp.full(32, cp.nan, dtype=cp.float64)
    y = cp.full(32, cp.nan, dtype=cp.float64)
    x[:5] = cp.asarray([0.0, 4.0, 4.0, 0.0, 0.0])
    y[:5] = cp.asarray([0.0, 0.0, 4.0, 4.0, 0.0])
    owned = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=x,
                y=y,
                geometry_offsets=cp.asarray([0, 1], dtype=cp.int32),
                ring_offsets=cp.asarray([0, 5], dtype=cp.int32),
                empty_mask=cp.asarray([False], dtype=cp.bool_),
            )
        },
        row_count=1,
        tags=cp.asarray([FAMILY_TAGS[GeometryFamily.POLYGON]], dtype=cp.int8),
        validity=cp.asarray([True], dtype=cp.bool_),
        family_row_offsets=cp.asarray([0], dtype=cp.int32),
        execution_mode="gpu",
    )

    segments = _extract_segments_gpu(owned)

    assert segments.count == 4
    assert cp.all(segments.row_indices == 0)
    assert cp.all(cp.isfinite(segments.x0))
    assert cp.all(cp.isfinite(segments.y0))
    assert cp.all(cp.isfinite(segments.x1))
    assert cp.all(cp.isfinite(segments.y1))


@pytest.mark.gpu
def test_extract_segments_gpu_device_only_family_totals_stay_native() -> None:
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
    ] == []


@pytest.mark.gpu
def test_extract_segments_gpu_counts_nested_device_only_families() -> None:
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


@pytest.mark.gpu
def test_extract_segments_gpu_indexed_view_counts_logical_rows() -> None:
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
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            Polygon(
                [(10, 0), (16, 0), (16, 6), (10, 6), (10, 0)],
                [[(11, 1), (12, 1), (12, 2), (11, 1)]],
            ),
        ],
        residency=Residency.DEVICE,
    )
    indices = cp.asarray(np.tile(np.asarray([0, 1], dtype=np.int64), 600))
    indexed = owned.device_take(indices)

    assert indexed.is_indexed_view

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    segments = _extract_segments_gpu(indexed)
    events = get_d2h_transfer_events(clear=True)

    assert segments.count == (4 + 7) * 600
    assert [
        event.reason
        for event in events
        if event.reason == "segment extraction total-segments allocation fence"
    ] == []
    assert int(cp.min(segments.row_indices).get()) == 0
    assert int(cp.max(segments.row_indices).get()) == indexed.row_count - 1


@pytest.mark.gpu
def test_indexed_segment_capacity_uses_variable_width_row_bound() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.spatial.segment_primitives import (
        _device_segment_capacity_for_family,
    )

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            Polygon(
                [(10, 0), (16, 0), (16, 6), (10, 6), (10, 0)],
                [[(11, 1), (12, 1), (12, 2), (11, 1)]],
            ),
        ],
        residency=Residency.DEVICE,
    )
    indexed = owned.device_take(cp.asarray([0, 1] * 600, dtype=cp.int64))
    polygon = indexed.device_state.families[GeometryFamily.POLYGON]
    assert int(polygon.x.size) == 14
    assert polygon.fixed_size.max_coord_count_per_row == 9

    capacity, capacity_per_row = _device_segment_capacity_for_family(
        GeometryFamily.POLYGON,
        polygon,
        indexed_view=True,
        row_count=indexed.row_count,
    )

    assert capacity == indexed.row_count * 9
    assert capacity_per_row is None


@pytest.mark.gpu
def test_indexed_segment_capacity_uses_active_relation_bounds() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.spatial.segment_primitives import (
        _device_segment_capacity_for_family,
    )

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            Polygon(
                [(10, 0), (16, 0), (16, 6), (10, 6), (10, 0)],
                [[(11, 1), (12, 1), (12, 2), (11, 1)]],
            ),
        ],
        residency=Residency.DEVICE,
    )
    relation = owned._device_indexed_take(
        cp.asarray([0, 1, 0, 1, 0, 1], dtype=cp.int64),
        active_family_row_multiplicity_bound=3,
        active_family_row_segment_capacity_bound=7,
    )._apply_row_activity(cp.ones(6, dtype=cp.bool_))
    page = relation._device_indexed_take(
        cp.asarray([0, 1, 2, 3], dtype=cp.int64),
        assume_unique_indices=True,
    )
    polygon = page.device_state.families[GeometryFamily.POLYGON]

    assert page._active_family_row_multiplicity_bound == 3
    assert page._active_family_row_segment_capacity_bound == 7
    capacity, capacity_per_row = _device_segment_capacity_for_family(
        GeometryFamily.POLYGON,
        polygon,
        indexed_view=True,
        row_count=page.row_count,
        active_family_row_multiplicity_bound=(page._active_family_row_multiplicity_bound),
        active_family_row_segment_capacity_bound=(page._active_family_row_segment_capacity_bound),
    )

    assert capacity == page.row_count * 7
    assert capacity_per_row is None


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
def test_segment_primitives_gpu_proper_point_is_symmetric_and_correctly_rounded() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left_line = LineString(
        [
            (399.9999999999999, 326.7949192431123),
            (479.29447639179836, 422.72593389687455),
        ]
    )
    right_line = LineString([(430.0, 370.0), (440.0, 370.0)])
    expected = left_line.intersection(right_line)
    left = from_shapely_geometries([left_line])
    right = from_shapely_geometries([right_line])

    forward = classify_segment_intersections(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    reverse = classify_segment_intersections(
        right,
        left,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert forward.kind_names() == ["proper"]
    assert reverse.kind_names() == ["proper"]
    assert forward.point_x[0] == expected.x
    assert forward.point_y[0] == expected.y
    assert reverse.point_x[0] == expected.x
    assert reverse.point_y[0] == expected.y


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
def test_segment_candidate_strict_upper_capacity_path_avoids_filtered_total_fence() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_cuda_runtime,
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
    runtime = get_cuda_runtime()
    upper_left = runtime.from_host(np.arange(3, dtype=np.int32))
    upper_right = runtime.from_host(np.arange(3, 6, dtype=np.int32))

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = classify_segment_intersections(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _strict_upper_source_rows=(upper_left, upper_right),
        _use_same_row_fast_path=False,
    )
    events = get_d2h_transfer_events(clear=True)

    assert result.count == 3
    assert [
        event.reason
        for event in events
        if event.reason == "segment filtered candidate total allocation fence"
    ] == []


@pytest.mark.gpu
def test_segment_candidate_outlier_capacity_path_avoids_outlier_total_fence() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries(
        [LineString([(float(row) * 10.0, -1.0), (float(row) * 10.0, 1.0)]) for row in range(3)],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            LineString([(200.0 + float(row) * 3.0, 0.5), (201.0 + float(row) * 3.0, 0.5)])
            for row in range(20)
        ]
        + [LineString([(-5.0, 0.0), (25.0, 0.0)])],
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

    assert result.count == 3
    assert [
        event.reason
        for event in events
        if event.reason == "segment outlier candidate total allocation fence"
    ] == []


@pytest.mark.gpu
def test_same_row_small_segment_table_uses_total_span_bound() -> None:
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
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            LineString([(0, 4), (4, 0)]),
            LineString([(10, 4), (14, 0)]),
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

    assert result.count == 2
    assert "segment same-row span summary scalar fence" not in {event.reason for event in events}


@pytest.mark.gpu
def test_same_row_unproved_large_segment_table_uses_generic_native_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    row_count = 2050
    left = from_shapely_geometries(
        [
            LineString([(float(row * 3), 0.0), (float(row * 3 + 1), 1.0)])
            for row in range(row_count)
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            LineString([(float(row * 3), 1.0), (float(row * 3 + 1), 0.0)])
            for row in range(row_count)
        ],
        residency=Residency.DEVICE,
    )

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = classify_segment_intersections(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _require_same_row=True,
    )
    events = get_d2h_transfer_events(clear=True)

    assert result.count == row_count
    assert "segment same-row span summary scalar fence" not in {event.reason for event in events}
    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    assert "segment.candidates.same_row_fast_path" not in summary
    assert summary.get("segment.candidates.binary_search") == 1


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
def test_segment_same_row_fast_path_applies_strict_upper_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.spatial.segment_primitives import DeviceSegmentTable, _extract_segments_gpu

    owned = from_shapely_geometries(
        [
            LineString([(-1, 0), (1, 0)]),
            LineString([(0, -1), (0, 1)]),
            LineString([(-1, -1), (1, 1)]),
        ],
        residency=Residency.DEVICE,
    )
    segments = _extract_segments_gpu(owned)
    grouped_rows = cp.zeros(segments.count, dtype=cp.int32)
    grouped_segments = DeviceSegmentTable(
        row_indices=grouped_rows,
        segment_indices=segments.segment_indices,
        x0=segments.x0,
        y0=segments.y0,
        x1=segments.x1,
        y1=segments.y1,
        count=segments.count,
        part_indices=segments.part_indices,
        ring_indices=segments.ring_indices,
    )
    original_rows = cp.asarray(segments.row_indices, dtype=cp.int32)

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()
    result = classify_segment_intersections(
        owned,
        owned,
        dispatch_mode=ExecutionMode.GPU,
        _cached_left_device_segments=grouped_segments,
        _cached_right_device_segments=grouped_segments,
        _require_same_row=True,
        _strict_upper_source_rows=(original_rows, original_rows),
        _same_row_span_summary=(segments.count, segments.count, 0),
    )

    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.count == 3
    assert result.kind_names() == ["proper", "proper", "proper"]
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
def test_segment_same_row_batched_capacity_avoids_total_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import vibespatial.spatial.segment_primitives as segment_primitives
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    monkeypatch.setattr(segment_primitives, "_compute_max_batch_pairs", lambda: 4)
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
            MultiLineString([[(0, 3), (3, 0)], [(100, 0), (101, 1)]]),
            MultiLineString([[(10, 3), (13, 0)], [(110, 0), (111, 1)]]),
            MultiLineString([[(20, 3), (23, 0)], [(120, 0), (121, 1)]]),
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
        _same_row_span_summary=(1, 2, 2),
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

    summary = summarize_exact_local_events(
        left, right, dispatch_mode=ExecutionMode.CPU, _require_same_row=True
    )

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

    cpu = summarize_exact_local_events(
        left, right, dispatch_mode=ExecutionMode.CPU, _require_same_row=True
    )
    gpu = summarize_exact_local_events(
        left, right, dispatch_mode=ExecutionMode.GPU, _require_same_row=True
    )

    assert gpu.runtime_selection.selected is ExecutionMode.GPU
    assert gpu.candidate_pairs == cpu.candidate_pairs
    assert gpu.point_intersection_count == cpu.point_intersection_count
    assert gpu.parallel_or_colinear_candidate_count == cpu.parallel_or_colinear_candidate_count
    assert gpu.row_point_intersection_counts.tolist() == cpu.row_point_intersection_counts.tolist()
    assert gpu.exact_event_counts.tolist() == cpu.exact_event_counts.tolist()
    assert gpu.exact_interval_upper_bounds.tolist() == cpu.exact_interval_upper_bounds.tolist()
