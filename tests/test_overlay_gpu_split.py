from __future__ import annotations

import importlib
import inspect
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box

from vibespatial import (
    ExecutionMode,
    build_gpu_atomic_edges,
    build_gpu_split_events,
    from_shapely_geometries,
    has_gpu_runtime,
)
from vibespatial.overlay.graph import _largest_power_of_two_block_size
from vibespatial.overlay.types import (
    ComponentOverlayExecutionPlan,
    MicrocellOverlayExecutionPlan,
    PagedOverlayExecutionPlan,
)


def _group_point_counts(source_segment_ids: np.ndarray) -> Counter[int]:
    return Counter(int(value) for value in source_segment_ids.tolist())


def test_row_isolated_topology_page_shape_uses_live_event_work() -> None:
    from vibespatial.overlay.gpu import _row_isolated_topology_page_shape

    shape = _row_isolated_topology_page_shape(
        row_count=10,
        max_left_segments_per_row=4,
        max_right_segments_per_row=4,
        include_same_side_splits=False,
        live_event_budget=500,
    )

    assert shape.worst_live_events_per_row == 80
    assert shape.rows_per_page == 6
    assert shape.page_count == 2
    assert shape.row_spans() == ((0, 6), (6, 10))
    assert not shape.single_row_oversized


def test_row_isolated_topology_page_shape_accounts_for_same_side_events() -> None:
    from vibespatial.overlay.gpu import _row_isolated_topology_page_shape

    shape = _row_isolated_topology_page_shape(
        row_count=3,
        max_left_segments_per_row=3,
        max_right_segments_per_row=2,
        include_same_side_splits=True,
        live_event_budget=49,
    )

    assert shape.worst_live_events_per_row == 50
    assert shape.rows_per_page == 1
    assert shape.page_count == 3
    assert shape.single_row_oversized


def test_live_split_event_budget_tracks_free_memory_and_safety_ceiling(
    monkeypatch,
) -> None:
    import importlib

    gpu = importlib.import_module("vibespatial.overlay.gpu")

    class _Runtime:
        @staticmethod
        def memory_pool_stats():
            return {"free_bytes": 5 * gpu._BYTES_PER_LIVE_SPLIT_EVENT * 1_000_000}

    monkeypatch.setattr(gpu, "get_cuda_runtime", lambda: _Runtime())
    assert gpu._compute_live_split_event_budget() == 1_000_000

    class _LargeRuntime:
        @staticmethod
        def memory_pool_stats():
            return {
                "free_bytes": (
                    10
                    * gpu._BYTES_PER_LIVE_SPLIT_EVENT
                    * gpu._MAX_LIVE_SPLIT_EVENT_BUDGET
                )
            }

    monkeypatch.setattr(gpu, "get_cuda_runtime", lambda: _LargeRuntime())
    assert gpu._compute_live_split_event_budget() == gpu._MAX_LIVE_SPLIT_EVENT_BUDGET


def test_live_split_event_budget_prefers_query_envelope_over_pool_free_bytes(
    monkeypatch,
) -> None:
    import importlib

    gpu = importlib.import_module("vibespatial.overlay.gpu")

    class _Runtime:
        @staticmethod
        def query_memory_remaining_bytes():
            return 5 * gpu._BYTES_PER_LIVE_SPLIT_EVENT * 1_000_000

        @staticmethod
        def memory_pool_stats():
            return {"free_bytes": 1}

    monkeypatch.setattr(gpu, "get_cuda_runtime", lambda: _Runtime())
    assert gpu._compute_live_split_event_budget() == 1_000_000


def test_split_event_consumer_bounds_candidate_pages_by_event_amplification() -> None:
    from vibespatial.overlay import split
    from vibespatial.spatial import segment_primitives

    split_source = Path(split.__file__).read_text()
    segment_source = Path(segment_primitives.__file__).read_text()

    assert "_compute_live_split_event_budget() // 4" in split_source
    assert "_candidate_page_budget=candidate_page_budget" in split_source
    assert "candidate_page_budget=_candidate_page_budget" in segment_source
    assert "accumulator.contiguous_pair_budget" in segment_source


@pytest.mark.gpu
def test_planarity_risk_marks_only_ulp_inconsistent_shared_pair_nodes() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    from vibespatial.overlay.split import _paired_event_planarity_risk

    x = cp.asarray(
        [1.0, 1.0, 2.0, np.nextafter(2.0, np.inf), 3.0, np.nextafter(3.0, np.inf)],
        dtype=cp.float64,
    )
    risk = _paired_event_planarity_risk(
        cp.arange(6, dtype=cp.int32),
        cp.asarray([0, 0, 0, 0, 0, 1], dtype=cp.int32),
        x,
        cp.zeros(6, dtype=cp.float64),
        cp.ones(6, dtype=cp.int8),
    )

    assert cp.asnumpy(risk).tolist() == [False, False, True, True, False, False]


def test_paged_overlay_plan_keeps_page_boundaries_algebraic() -> None:
    plan = PagedOverlayExecutionPlan(
        left=object(),
        right=object(),
        row_count=1_000_001,
        rows_per_page=100_000,
        max_left_segments_per_row=4,
        max_right_segments_per_row=4,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert plan.page_count == 11
    assert plan.row_span(0) == (0, 100_000)
    assert plan.row_span(10) == (1_000_000, 1_000_001)
    assert not hasattr(plan, "page_row_offsets")


def test_paged_overlay_plan_accepts_compact_weighted_row_spans() -> None:
    plan = PagedOverlayExecutionPlan(
        left=object(),
        right=object(),
        row_count=10,
        rows_per_page=6,
        max_left_segments_per_row=4,
        max_right_segments_per_row=40,
        dispatch_mode=ExecutionMode.GPU,
        complete_row_spans=((0, 2), (2, 9), (9, 10)),
    )

    assert plan.page_count == 3
    assert [plan.row_span(index) for index in range(plan.page_count)] == [
        (0, 2),
        (2, 9),
        (9, 10),
    ]


def test_split_consumes_classified_candidate_pages_without_relation_concat() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    split_source = (repo_root / "src/vibespatial/overlay/split.py").read_text()
    primitive_source = (repo_root / "src/vibespatial/spatial/segment_primitives.py").read_text()

    assert "_classified_page_consumer=_consume_classified_page" in split_source
    assert "_classified_page_consumer=_consume_right_right_page" in split_source
    assert "_classified_page_consumer=_consume_side_page" in split_source
    assert "concatenate_paged_segment_intersections_device" not in split_source
    assert "_emit_pair_split_event_batch" in split_source
    assert "_merge_sorted_split_event_runs" in split_source
    assert "_SplitEventRunAccumulator" in split_source
    assert "pair_event_runs.append" not in split_source
    assert "overlay.split.external_merge_events" in split_source
    assert "overlay.split.concat_events" not in split_source
    assert "_stable_radix_order_pass" in split_source
    assert "cp.lexsort" not in split_source
    assert "cp.stack" not in split_source
    assert "bool(cp.any(should_update))" not in split_source
    assert "_classified_page_consumer(classified)" in primitive_source


def test_split_event_run_accumulator_retains_one_run_per_binary_level(
    monkeypatch,
) -> None:
    from vibespatial.overlay import split

    monkeypatch.setattr(
        split._SplitEventRunAccumulator,
        "_merge",
        staticmethod(lambda left, right: (left, right)),
    )
    accumulator = split._SplitEventRunAccumulator()
    for page in range(13):
        accumulator.append(page)
        assert len(accumulator.retained_runs()) == (page + 1).bit_count()

    assert accumulator.finish() == (
        (
            (((0, 1), (2, 3)), ((4, 5), (6, 7))),
            ((8, 9), (10, 11)),
        ),
        12,
    )
    assert accumulator.retained_runs() == ()


def test_oversized_single_row_plan_uses_strict_interval_components() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gpu_source = (repo_root / "src/vibespatial/overlay/gpu.py").read_text()

    assert "ComponentOverlayExecutionPlan" in gpu_source
    assert "MicrocellOverlayExecutionPlan" in gpu_source
    assert "d_sorted_xmin[1:] > d_prefix_xmax[:-1]" in gpu_source
    assert "_pack_polygon_parts_by_component" in gpu_source
    assert "_pack_disjoint_component_result" in gpu_source
    assert "single_row_interval_component_topology_gpu" in gpu_source
    assert "single_row_connected_microcell_boundary_graph_gpu" in gpu_source
    component_start = gpu_source.index("def _single_row_interval_components(")
    component_end = gpu_source.index("\ndef ", component_start + 1)
    component_source = gpu_source[component_start:component_end]
    pack_start = gpu_source.index("def _pack_polygon_parts_by_component(")
    pack_end = gpu_source.index("\ndef ", pack_start + 1)
    pack_source = gpu_source[pack_start:pack_end]
    result_pack_start = gpu_source.index("def _pack_disjoint_component_result(")
    result_pack_end = gpu_source.index("\ndef ", result_pack_start + 1)
    result_pack_source = gpu_source[result_pack_start:result_pack_end]
    assert "_explode_polygonal_rows_to_polygon_capacity_gpu" in component_source
    assert "physicalize_device_rows" not in component_source
    assert "cp.flatnonzero" not in component_source
    assert "_explode_polygonal_rows_to_polygons_gpu" not in component_source
    assert "NativeGroupedSelection" in pack_source
    assert "polygon_parts.selection.active_capacity_mask()" in pack_source
    assert "_assemble_sorted_polygon_part_capacity_gpu" in pack_source
    assert "polygon_parts.ring_capacity" in pack_source
    assert "polygon_parts.coord_capacity" in pack_source
    assert "physicalize_device_rows" not in pack_source
    assert "assume_disjoint=True" in result_pack_source
    assert "_explode_polygonal_rows_to_polygons_gpu" not in result_pack_source


def test_microcell_overlay_plan_is_a_single_row_native_carrier() -> None:
    plan = MicrocellOverlayExecutionPlan(
        left=object(),
        right=object(),
        max_left_segments=100,
        max_right_segments=200,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert plan.row_isolated
    assert plan.max_left_segments == 100
    assert plan.max_right_segments == 200


def test_device_concat_uses_active_offset_carriers_without_scalar_exports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    owned_source = (repo_root / "src/vibespatial/geometry/owned.py").read_text()
    device_concat = owned_source.split("def _concat_device_family_buffers(", 1)[1].split(
        "def _concat_family_buffers(", 1
    )[0]

    assert "_concat_device_xy_compact" in device_concat
    assert "_device_offset_terminal_counts" in device_concat
    assert "int(b.geometry_offsets[-1])" not in device_concat
    assert "int(b.part_offsets[-1])" not in device_concat
    assert "int(b.ring_offsets[-1])" not in device_concat


@pytest.mark.gpu
def test_gpu_row_isolated_topology_pages_preserve_complete_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    overlay_gpu = importlib.import_module("vibespatial.overlay.gpu")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    monkeypatch.setattr(overlay_gpu, "_compute_live_split_event_budget", lambda: 80)
    original_physicalize = overlay_gpu._physicalize_paged_overlay_output
    physicalized_page_rows = []
    runtime = overlay_gpu.get_cuda_runtime()
    original_synchronize_stream = runtime.synchronize_stream
    synchronized_pages = []

    def _record_physicalization(result):
        physicalized_page_rows.append(result.row_count)
        return original_physicalize(result)

    monkeypatch.setattr(
        overlay_gpu,
        "_physicalize_paged_overlay_output",
        _record_physicalization,
    )

    def _record_page_retirement():
        synchronized_pages.append(True)
        original_synchronize_stream()

    monkeypatch.setattr(runtime, "synchronize_stream", _record_page_retirement)
    left = from_shapely_geometries([box(0, 0, 4, 4), box(10, 0, 14, 4), box(20, 0, 24, 4)])
    right = from_shapely_geometries([box(2, 2, 6, 6), box(12, 2, 16, 6), box(22, 2, 26, 6)])

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    plan = overlay_gpu._build_overlay_execution_plan(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
        _same_row_span_summary=(4, 4, 2),
    )
    events = get_d2h_transfer_events(clear=True)

    assert isinstance(plan, PagedOverlayExecutionPlan)
    assert plan.rows_per_page == 1
    assert plan.page_count == 3
    assert [plan.row_span(index) for index in range(plan.page_count)] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert plan.row_isolated
    assert events
    assert {
        event.reason
        for event in events
    } <= {
        "overlay compact topology page-weight planning packet",
        "overlay compact topology work-summary planning packet",
    }
    assert sum(event.item_count for event in events) <= 7

    result, selected = overlay_gpu._materialize_overlay_execution_plan(
        plan,
        operation="intersection",
        requested=ExecutionMode.GPU,
        preserve_row_count=3,
    )
    expected = [
        left_geom.intersection(right_geom)
        for left_geom, right_geom in zip(
            left.to_shapely(),
            right.to_shapely(),
            strict=True,
        )
    ]
    assert selected is ExecutionMode.GPU
    assert result.row_count == 3
    assert physicalized_page_rows == [1, 1, 1]
    assert synchronized_pages == [True, True, True]
    assert all(
        actual.normalize().equals_exact(wanted.normalize(), tolerance=1.0e-9)
        for actual, wanted in zip(result.to_shapely(), expected, strict=True)
    )


@pytest.mark.gpu
def test_gpu_grouped_topology_pages_without_optional_span_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Device row ownership must bound grouped topology without host maxima."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    overlay_gpu = importlib.import_module("vibespatial.overlay.gpu")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    monkeypatch.setattr(overlay_gpu, "_compute_live_split_event_budget", lambda: 400)
    left_geometries = [
        box(0, 0, 10, 10),
        box(20, 0, 30, 10),
        box(40, 0, 50, 10),
    ]
    right_geometries = [
        box(1, -1, 4, 6),
        box(3, 4, 7, 11),
        box(21, -1, 24, 6),
        box(23, 4, 27, 11),
        box(41, -1, 44, 6),
        box(43, 4, 47, 11),
    ]
    left = from_shapely_geometries(left_geometries)
    right = from_shapely_geometries(right_geometries)
    d_group_rows = cp.repeat(cp.arange(3, dtype=cp.int32), 2)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    plan = overlay_gpu._build_overlay_execution_plan(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
        _same_row_span_summary=None,
        _include_same_side_splits=True,
        _right_geometry_source_rows=d_group_rows,
        _right_segment_source_rows=d_group_rows,
    )
    events = get_d2h_transfer_events(clear=True)

    assert isinstance(plan, PagedOverlayExecutionPlan)
    assert plan.max_left_segments_per_row == 4
    assert plan.max_right_segments_per_row == 8
    assert [plan.row_span(index) for index in range(plan.page_count)] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert events
    assert {
        event.reason
        for event in events
    } <= {
        "overlay compact topology page-weight planning packet",
        "overlay compact topology work-summary planning packet",
    }
    assert sum(event.item_count for event in events) <= 7

    result, selected = overlay_gpu._materialize_overlay_execution_plan(
        plan,
        operation="difference",
        requested=ExecutionMode.GPU,
        preserve_row_count=3,
    )
    expected = [
        source.difference(right_geometries[index * 2].union(right_geometries[index * 2 + 1]))
        for index, source in enumerate(left_geometries)
    ]
    assert selected is ExecutionMode.GPU
    assert result.row_count == 3
    assert all(
        actual.normalize().equals_exact(wanted.normalize(), tolerance=1.0e-9)
        for actual, wanted in zip(result.to_shapely(), expected, strict=True)
    )


@pytest.mark.gpu
def test_gpu_paged_overlay_input_physicalization_uses_logical_coordinate_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A padded source buffer must not size a page's coordinate allocation."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    overlay_gpu = importlib.import_module("vibespatial.overlay.gpu")
    api_overlay = importlib.import_module("vibespatial.api.tools.overlay")
    owned_module = importlib.import_module("vibespatial.geometry.owned")
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )

    logical_x = cp.asarray([0, 4, 4, 0, 0, 10, 14, 14, 10, 10], dtype=cp.float64)
    logical_y = cp.asarray([0, 0, 4, 4, 0, 0, 0, 4, 4, 0], dtype=cp.float64)
    coordinate_capacity = 1_000_000
    right = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=cp.pad(logical_x, (0, coordinate_capacity - logical_x.size)),
                y=cp.pad(logical_y, (0, coordinate_capacity - logical_y.size)),
                geometry_offsets=cp.asarray([0, 1, 2], dtype=cp.int32),
                ring_offsets=cp.asarray([0, 5, 10], dtype=cp.int32),
                empty_mask=cp.zeros(2, dtype=cp.bool_),
            )
        },
        row_count=2,
        tags=cp.full(2, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8),
        validity=cp.ones(2, dtype=cp.bool_),
        family_row_offsets=cp.arange(2, dtype=cp.int32),
        execution_mode="gpu",
    )
    left_geometries = [
        box(-1, -1, 15, 5),
        box(20, -1, 25, 5),
        box(30, -1, 35, 5),
    ]
    left = from_shapely_geometries(left_geometries)
    d_right_source_rows = cp.zeros(2, dtype=cp.int32)

    physicalized_coordinate_counts: list[int] = []
    padded_input_indexed_flags: list[bool] = []
    grouped_topology_coordinate_counts: list[int] = []
    original_physicalize = owned_module.device_physicalize_owned_row_selections_exact

    def _record_exact_physicalization(selections, *, reason, **kwargs):
        if reason == "paged overlay input exact-allocation packet":
            for owned, _active in selections:
                state = owned._ensure_device_state(preserve_indexed_view=True)
                polygon = state.families.get(GeometryFamily.POLYGON)
                if polygon is not None and int(polygon.x.size) == coordinate_capacity:
                    padded_input_indexed_flags.append(owned.is_indexed_view)
        physicalized = original_physicalize(selections, reason=reason, **kwargs)
        if reason == "paged overlay input exact-allocation packet":
            for owned in physicalized:
                if owned is None:
                    continue
                state = owned._ensure_device_state(preserve_indexed_view=True)
                polygon = state.families.get(GeometryFamily.POLYGON)
                if polygon is not None:
                    physicalized_coordinate_counts.append(int(polygon.x.size))
        elif reason == "grouped overlay topology input exact-allocation packet":
            for owned in physicalized:
                if owned is None:
                    continue
                state = owned._ensure_device_state(preserve_indexed_view=True)
                polygon = state.families.get(GeometryFamily.POLYGON)
                if polygon is not None:
                    grouped_topology_coordinate_counts.append(int(polygon.x.size))
        return physicalized

    monkeypatch.setattr(
        owned_module,
        "device_physicalize_owned_row_selections_exact",
        _record_exact_physicalization,
    )
    assert not right.is_indexed_view
    _, exact_right = overlay_gpu._physicalize_paged_overlay_inputs(left, right)
    exact_polygon = exact_right._ensure_device_state(
        preserve_indexed_view=True,
    ).families[GeometryFamily.POLYGON]
    assert int(exact_polygon.x.size) == logical_x.size

    grouped_left = left._device_indexed_take(
        cp.asarray([0], dtype=cp.int64),
        assume_unique_indices=True,
    )
    grouped_right = right._device_indexed_take(
        cp.arange(2, dtype=cp.int64),
        assume_unique_indices=True,
    )
    assert grouped_right.is_indexed_view
    grouped_result = api_overlay._grouped_overlay_difference_owned(
        grouped_left,
        grouped_right,
        cp.asarray([0, 2], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
        _skip_containment_union=True,
        _skip_direct_specializations=True,
    )
    mask_geometries = right.to_shapely()
    expected_grouped = left_geometries[0].difference(
        mask_geometries[0].union(mask_geometries[1])
    )
    assert grouped_topology_coordinate_counts == [5, logical_x.size]
    assert grouped_result.to_shapely()[0].normalize().equals_exact(
        expected_grouped.normalize(),
        tolerance=1.0e-9,
    )

    monkeypatch.setattr(overlay_gpu, "_compute_live_split_event_budget", lambda: 160)
    plan = overlay_gpu._build_overlay_execution_plan(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
        _same_row_span_summary=None,
        _right_geometry_source_rows=d_right_source_rows,
        _right_segment_source_rows=d_right_source_rows,
    )

    assert isinstance(plan, PagedOverlayExecutionPlan)
    result, selected = overlay_gpu._materialize_overlay_execution_plan(
        plan,
        operation="difference",
        requested=ExecutionMode.GPU,
        preserve_row_count=3,
    )

    expected = [
        left_geometries[0].difference(mask_geometries[0].union(mask_geometries[1])),
        left_geometries[1],
        left_geometries[2],
    ]
    assert selected is ExecutionMode.GPU
    assert False in padded_input_indexed_flags
    assert physicalized_coordinate_counts
    assert logical_x.size in physicalized_coordinate_counts
    assert max(physicalized_coordinate_counts) < coordinate_capacity
    assert coordinate_capacity not in physicalized_coordinate_counts
    assert all(
        actual.normalize().equals_exact(wanted.normalize(), tolerance=1.0e-9)
        for actual, wanted in zip(result.to_shapely(), expected, strict=True)
    )


@pytest.mark.gpu
def test_gpu_segment_extraction_ignores_inactive_indexed_capacity_lanes() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.spatial.segment_primitives import _extract_segments_gpu

    base = from_shapely_geometries([box(0, 0, 1, 1), box(2, 0, 3, 1)])
    capacity = 4096
    d_indices = cp.arange(capacity, dtype=cp.int64) % cp.int64(2)
    d_active = cp.zeros(capacity, dtype=cp.bool_)
    d_active[0] = True
    d_active[-1] = True
    masked = base._device_indexed_take(d_indices)._apply_row_activity(d_active)

    segments = _extract_segments_gpu(masked)

    assert segments.count == 8
    assert cp.asnumpy(cp.unique(segments.row_indices)).tolist() == [0, capacity - 1]


@pytest.mark.gpu
def test_gpu_oversized_single_row_uses_interval_component_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    overlay_gpu = importlib.import_module("vibespatial.overlay.gpu")

    monkeypatch.setattr(overlay_gpu, "_compute_live_split_event_budget", lambda: 80)
    left_geom = MultiPolygon([box(0, 0, 4, 4), box(100, 0, 104, 4)])
    right_geom = MultiPolygon([box(2, 2, 6, 6), box(102, 2, 106, 6)])
    left = from_shapely_geometries([left_geom])
    right = from_shapely_geometries([right_geom])

    plan = overlay_gpu._build_overlay_execution_plan(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
        _same_row_span_summary=(8, 8, 0),
    )

    assert isinstance(plan, ComponentOverlayExecutionPlan)
    assert plan.component_count == 2
    result, selected = overlay_gpu._materialize_overlay_execution_plan(
        plan,
        operation="intersection",
        requested=ExecutionMode.GPU,
        preserve_row_count=1,
    )
    assert selected is ExecutionMode.GPU
    assert result.row_count == 1
    assert result.to_shapely()[0].equals(left_geom.intersection(right_geom))


@pytest.mark.gpu
def test_gpu_oversized_connected_single_row_uses_microcell_boundary_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    overlay_gpu = importlib.import_module("vibespatial.overlay.gpu")

    monkeypatch.setattr(overlay_gpu, "_compute_live_split_event_budget", lambda: 80)
    left_geom = Polygon([(0, 0), (4, 0), (8, 0), (8, 4), (8, 8), (4, 8), (0, 8), (0, 4)])
    right_geom = Polygon([(2, -1), (6, -1), (10, -1), (10, 3), (10, 7), (6, 7), (2, 7), (2, 3)])
    left = from_shapely_geometries([left_geom])
    right = from_shapely_geometries([right_geom])

    plan = overlay_gpu._build_overlay_execution_plan(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
        _same_row_span_summary=(8, 8, 0),
    )

    assert isinstance(plan, MicrocellOverlayExecutionPlan)
    result, selected = overlay_gpu._materialize_overlay_execution_plan(
        plan,
        operation="intersection",
        requested=ExecutionMode.GPU,
        preserve_row_count=1,
    )
    assert selected is ExecutionMode.GPU
    assert result.to_shapely()[0].equals(left_geom.intersection(right_geom))


@pytest.mark.gpu
def test_gpu_oversized_grouped_difference_uses_interval_component_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp
    import shapely

    overlay_gpu = importlib.import_module("vibespatial.overlay.gpu")

    monkeypatch.setattr(overlay_gpu, "_compute_live_split_event_budget", lambda: 80)
    left_geom = MultiPolygon([box(0, 0, 10, 10), box(100, 0, 110, 10)])
    right_geoms = [
        box(2, -1, 6, 8),
        box(4, 2, 8, 11),
        box(102, -1, 106, 8),
        box(104, 2, 108, 11),
    ]
    left = from_shapely_geometries([left_geom])
    right = from_shapely_geometries(right_geoms)
    d_group_rows = cp.zeros(len(right_geoms), dtype=cp.int32)

    plan = overlay_gpu._build_overlay_execution_plan(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
        _same_row_span_summary=(8, 16, 0),
        _right_geometry_source_rows=d_group_rows,
        _right_segment_source_rows=d_group_rows,
    )

    assert isinstance(plan, ComponentOverlayExecutionPlan)
    assert plan.component_count == 2
    assert plan.include_same_side_splits
    result, selected = overlay_gpu._materialize_overlay_execution_plan(
        plan,
        operation="difference",
        requested=ExecutionMode.GPU,
        preserve_row_count=1,
        valid_empty_rows=cp.ones(1, dtype=cp.bool_),
    )
    expected = left_geom.difference(shapely.union_all(right_geoms))
    assert selected is ExecutionMode.GPU
    assert result.to_shapely()[0].equals(expected)


@pytest.mark.gpu
def test_gpu_split_events_and_atomic_edges_for_proper_cross() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([LineString([(0, 0), (4, 4)])])
    right = from_shapely_geometries([LineString([(0, 4), (4, 0)])])

    split_events = build_gpu_split_events(left, right, dispatch_mode=ExecutionMode.GPU)
    atomic_edges = build_gpu_atomic_edges(split_events)

    assert split_events.runtime_selection.selected is ExecutionMode.GPU
    assert split_events.device_state is not None
    assert atomic_edges.device_state is not None
    assert _group_point_counts(split_events.source_segment_ids) == Counter({0: 3, 1: 3})
    assert atomic_edges.count == 8
    assert np.allclose(split_events.x[[1, 4]], [2.0, 2.0])
    assert np.allclose(split_events.y[[1, 4]], [2.0, 2.0])
    assert all(event.kind.value != "materialization" for event in left.diagnostics)
    assert all(event.kind.value != "materialization" for event in right.diagnostics)


@pytest.mark.gpu
def test_gpu_endpoint_only_split_events_retain_output_allocations() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    left = from_shapely_geometries(
        [LineString([(0.0, 0.0), (1.0, 0.0)])],
    )
    right = from_shapely_geometries(
        [LineString([(0.0, 2.0), (1.0, 2.0)])],
    )
    split_events = build_gpu_split_events(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    device = split_events.device_state
    output_pointers = {
        int(cp.asarray(array).data.ptr)
        for array in (
            device.source_segment_ids,
            device.t,
            device.x,
            device.y,
        )
    }
    pressure = []
    for _ in range(16):
        pressure.extend(
            (
                cp.full(split_events.count, -7, dtype=cp.int32),
                cp.full(split_events.count, 13, dtype=cp.uint64),
                cp.full(split_events.count, cp.nan, dtype=cp.float64),
            )
        )
    pressure_pointers = {int(array.data.ptr) for array in pressure}

    assert output_pointers.isdisjoint(pressure_pointers)
    assert cp.array_equal(device.source_segment_ids, cp.asarray([0, 0, 1, 1]))
    assert cp.allclose(device.t, cp.asarray([0.0, 1.0, 0.0, 1.0]))
    assert cp.allclose(device.x, cp.asarray([0.0, 1.0, 0.0, 1.0]))
    assert cp.allclose(device.y, cp.asarray([0.0, 0.0, 2.0, 2.0]))


@pytest.mark.gpu
def test_gpu_split_events_page_classification_before_global_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    import vibespatial.spatial.segment_primitives as segment_primitives
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    monkeypatch.setattr(segment_primitives, "_compute_max_batch_pairs", lambda: 4)
    left = from_shapely_geometries(
        [
            MultiLineString(
                [
                    [(0, 1), (4, 1)],
                    [(0, 2), (4, 2)],
                    [(0, 3), (4, 3)],
                ]
            )
        ]
    )
    right = from_shapely_geometries(
        [
            MultiLineString(
                [
                    [(1, 0), (1, 4)],
                    [(2, 0), (2, 4)],
                    [(3, 0), (3, 4)],
                ]
            )
        ]
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    split_events = build_gpu_split_events(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        require_same_row=True,
        use_same_row_fast_path=False,
    )
    events = get_d2h_transfer_events(clear=True)

    assert split_events.device_state is not None
    d_source_ids = cp.asarray(split_events.device_state.source_segment_ids, dtype=cp.int32)
    d_t = cp.asarray(split_events.device_state.t, dtype=cp.float64)
    assert bool(
        cp.all(
            (d_source_ids[1:] > d_source_ids[:-1])
            | ((d_source_ids[1:] == d_source_ids[:-1]) & (d_t[1:] >= d_t[:-1]))
        )
    )
    assert split_events.count == 30
    assert events == []


@pytest.mark.gpu
def test_gpu_atomic_edges_derive_pair_count_from_split_event_cardinality() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries([LineString([(0, 0), (4, 4)])])
    right = from_shapely_geometries([LineString([(0, 4), (4, 0)])])
    split_events = build_gpu_split_events(left, right, dispatch_mode=ExecutionMode.GPU)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    atomic_edges = build_gpu_atomic_edges(split_events)
    events = get_d2h_transfer_events(clear=True)

    expected_pairs = (
        split_events.count - split_events.left_segment_count - split_events.right_segment_count
    )
    assert atomic_edges.count == expected_pairs * 2
    assert "overlay split atomic-edge pair-count allocation fence" not in {
        event.reason for event in events
    }


@pytest.mark.gpu
def test_gpu_atomic_edges_can_preserve_source_representative_orientation() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.overlay.split import build_gpu_atomic_edges as build_atomic_edges

    descending = from_shapely_geometries([LineString([(2, 2), (0, 0)])])
    split_events = build_gpu_split_events(
        descending,
        descending,
        dispatch_mode=ExecutionMode.GPU,
    )

    canonical = build_atomic_edges(split_events)
    source_oriented = build_atomic_edges(
        split_events,
        preserve_source_orientation=True,
    )

    canonical_device = canonical.device_state
    source_device = source_oriented.device_state
    assert canonical_device is not None
    assert source_device is not None
    assert cp.asnumpy(canonical_device.src_x[0::2]).tolist() == [0.0]
    assert cp.asnumpy(canonical_device.dst_x[0::2]).tolist() == [2.0]
    assert cp.asnumpy(source_device.src_x[0::2]).tolist() == [2.0]
    assert cp.asnumpy(source_device.dst_x[0::2]).tolist() == [0.0]


@pytest.mark.gpu
def test_gpu_split_events_and_atomic_edges_for_touch_and_overlap() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    touch_left = from_shapely_geometries([LineString([(0, 0), (2, 2)])])
    touch_right = from_shapely_geometries([LineString([(2, 2), (4, 0)])])
    touch_events = build_gpu_split_events(touch_left, touch_right, dispatch_mode=ExecutionMode.GPU)
    touch_edges = build_gpu_atomic_edges(touch_events)
    assert _group_point_counts(touch_events.source_segment_ids) == Counter({0: 2, 1: 2})
    assert touch_edges.count == 4

    overlap_left = from_shapely_geometries([LineString([(0, 0), (5, 0)])])
    overlap_right = from_shapely_geometries([LineString([(2, 0), (7, 0)])])
    overlap_events = build_gpu_split_events(
        overlap_left, overlap_right, dispatch_mode=ExecutionMode.GPU
    )
    overlap_edges = build_gpu_atomic_edges(overlap_events)
    assert _group_point_counts(overlap_events.source_segment_ids) == Counter({0: 3, 1: 3})
    assert overlap_edges.count == 6
    assert np.allclose(overlap_events.x[:3], [0.0, 2.0, 5.0])
    assert np.allclose(overlap_events.x[3:], [2.0, 5.0, 7.0])
    assert np.allclose(overlap_events.y, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.gpu
def test_gpu_split_events_dedup_sorted_runs_without_unique_by_key_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    split_module = importlib.import_module("vibespatial.overlay.split")

    monkeypatch.setattr(
        split_module,
        "unique_sorted_pairs",
        lambda *args, **kwargs: pytest.fail("unique_by_key primitive should not be used"),
        raising=False,
    )

    overlap_left = from_shapely_geometries([LineString([(0, 0), (5, 0)])])
    overlap_right = from_shapely_geometries([LineString([(2, 0), (7, 0)])])

    overlap_events = build_gpu_split_events(
        overlap_left, overlap_right, dispatch_mode=ExecutionMode.GPU
    )

    assert _group_point_counts(overlap_events.source_segment_ids) == Counter({0: 3, 1: 3})
    assert np.allclose(overlap_events.x[:3], [0.0, 2.0, 5.0])
    assert np.allclose(overlap_events.x[3:], [2.0, 5.0, 7.0])


@pytest.mark.gpu
def test_gpu_split_events_handle_empty_segment_tables() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    empty = from_shapely_geometries([])

    split_events = build_gpu_split_events(empty, empty, dispatch_mode=ExecutionMode.GPU)

    assert split_events.count == 0
    assert split_events.source_segment_ids.shape[0] == 0
    assert split_events.x.shape[0] == 0
    assert split_events.runtime_selection.selected is ExecutionMode.GPU


@pytest.mark.gpu
def test_gpu_split_events_preserve_polygon_hole_ring_metadata() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    donut = Polygon(
        shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
        holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
    )
    vertical = LineString([(3, -1), (3, 7)])

    left = from_shapely_geometries([donut])
    right = from_shapely_geometries([vertical])

    split_events = build_gpu_split_events(left, right, dispatch_mode=ExecutionMode.GPU)
    atomic_edges = build_gpu_atomic_edges(split_events)

    left_mask = split_events.source_side == 1
    assert {int(value) for value in split_events.ring_indices[left_mask].tolist()} == {0, 1}
    assert atomic_edges.count > 0
    assert all(event.kind.value != "materialization" for event in left.diagnostics)
    assert all(event.kind.value != "materialization" for event in right.diagnostics)


@pytest.mark.gpu
def test_gpu_atomic_edges_use_dense_metadata_lookup_without_sort_pairs() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    split_module = importlib.import_module("vibespatial.overlay.split")

    left = from_shapely_geometries([Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])])
    right = from_shapely_geometries([Polygon([(2, -1), (5, -1), (5, 3), (2, 3), (2, -1)])])
    split_events = build_gpu_split_events(left, right, dispatch_mode=ExecutionMode.GPU)

    assert not hasattr(split_module, "sort_pairs")

    atomic_edges = build_gpu_atomic_edges(split_events)

    assert atomic_edges.count > 0
    assert atomic_edges.row_indices.shape[0] == atomic_edges.count


def test_split_helper_call_graph_has_no_context_wide_synchronization() -> None:
    split_source = inspect.getsource(importlib.import_module("vibespatial.overlay.split"))
    renode_start = split_source.index("def renode_gpu_atomic_edges(")
    grouped_start = split_source.index("def renode_grouped_boundary_segments_gpu(")

    assert "runtime.synchronize()" not in split_source
    assert "overlay_bool_scalar" not in split_source[grouped_start:]
    assert "cp.int8(127)" in split_source[renode_start:grouped_start]
    assert "cp.int8(127)" in split_source[grouped_start:]


@pytest.mark.gpu
def test_split_helper_call_graph_performs_no_context_wide_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    split_module = importlib.import_module("vibespatial.overlay.split")
    runtime = split_module.get_cuda_runtime()

    class _SyncCountingRuntime:
        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.synchronize_calls = 0

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    counting_runtime = _SyncCountingRuntime(runtime)
    monkeypatch.setattr(
        split_module,
        "get_cuda_runtime",
        lambda: counting_runtime,
    )
    left = from_shapely_geometries([box(0, 0, 2, 2)])
    right = from_shapely_geometries([box(1, 0, 3, 2)])

    split_events = split_module.build_gpu_split_events(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        require_same_row=True,
        use_same_row_fast_path=False,
        include_same_side_splits=True,
    )
    atomic_edges = split_module.build_gpu_atomic_edges(
        split_events,
        isolate_rows=True,
    )
    noded = split_module.renode_gpu_atomic_edges(
        atomic_edges,
        isolate_rows=True,
    )
    split_module._free_split_event_device_state(split_events)
    split_module._free_atomic_edge_excess(noded)

    assert counting_runtime.synchronize_calls == 0


@pytest.mark.gpu
def test_renoding_endpoint_event_is_the_exact_duplicate_representative() -> None:
    import cupy as cp

    from vibespatial.overlay.split import _merge_split_event_runs_balanced

    endpoint = (
        cp.asarray([0], dtype=cp.int32),
        cp.asarray([0.0], dtype=cp.float64),
        cp.asarray([1.0], dtype=cp.float64),
        cp.asarray([2.0], dtype=cp.float64),
        cp.asarray([127], dtype=cp.int8),
    )
    rounded_intersection = (
        cp.asarray([0], dtype=cp.int32),
        cp.asarray([0.0], dtype=cp.float64),
        cp.asarray([cp.nextafter(cp.float64(1.0), cp.float64(2.0))]),
        cp.asarray([2.0], dtype=cp.float64),
        cp.asarray([4], dtype=cp.int8),
    )

    merged = _merge_split_event_runs_balanced([endpoint, rounded_intersection])

    assert merged[0].size == 1
    assert float(merged[2][0]) == 1.0
    assert float(merged[3][0]) == 2.0


@pytest.mark.gpu
@pytest.mark.parametrize("reverse_right", [False, True])
def test_coincident_operand_deltas_aggregate_with_semantic_orientation(
    reverse_right: bool,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    shell = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
    right_shell = list(reversed(shell)) if reverse_right else shell
    split_events = build_gpu_split_events(
        from_shapely_geometries([Polygon(shell)]),
        from_shapely_geometries([Polygon(right_shell)]),
        dispatch_mode=ExecutionMode.GPU,
    )
    atomic_edges = build_gpu_atomic_edges(split_events)
    device = atomic_edges.device_state
    forward = device.direction == 0
    left_delta = device.left_coverage_delta
    right_delta = device.right_coverage_delta

    assert np.all(device.source_membership[forward] == 3)
    assert np.array_equal(left_delta[forward], right_delta[forward])
    assert np.all(np.abs(left_delta[forward]) == 1)
    assert np.array_equal(left_delta[1::2], -left_delta[0::2])
    assert np.array_equal(right_delta[1::2], -right_delta[0::2])


@pytest.mark.gpu
def test_atomic_edge_deltas_are_preserved_through_structural_renoding() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial import RuntimeSelection
    from vibespatial.overlay.split import renode_gpu_atomic_edges
    from vibespatial.overlay.types import AtomicEdgeDeviceState, AtomicEdgeTable

    source_ids = cp.asarray([0, 0, 1, 1, 2, 2], dtype=cp.int32)
    direction = cp.asarray([0, 1, 0, 1, 0, 1], dtype=cp.int8)
    src_x = cp.asarray([0.0, 2.0, 0.0, 2.0, 5.0, 6.0], dtype=cp.float64)
    src_y = cp.asarray([0.0, 2.0, 2.0, 0.0, 0.0, 0.0], dtype=cp.float64)
    dst_x = cp.asarray([2.0, 0.0, 2.0, 0.0, 6.0, 5.0], dtype=cp.float64)
    dst_y = cp.asarray([2.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=cp.float64)
    atomic_edges = AtomicEdgeTable(
        left_segment_count=3,
        right_segment_count=0,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="delta re-node regression",
        ),
        device_state=AtomicEdgeDeviceState(
            source_segment_ids=source_ids,
            direction=direction,
            src_x=src_x,
            src_y=src_y,
            dst_x=dst_x,
            dst_y=dst_y,
            row_indices=cp.zeros(6, dtype=cp.int32),
            part_indices=cp.zeros(6, dtype=cp.int32),
            ring_indices=cp.zeros(6, dtype=cp.int32),
            source_side=cp.ones(6, dtype=cp.int8),
            source_membership=cp.ones(6, dtype=cp.uint8),
            tangent_x=dst_x - src_x,
            tangent_y=dst_y - src_y,
            left_coverage_delta=cp.asarray([2, -2, 3, -3, 4, -4], dtype=cp.int32),
            right_coverage_delta=cp.zeros(6, dtype=cp.int32),
        ),
        _count=6,
    )

    noded = renode_gpu_atomic_edges(atomic_edges, isolate_rows=True)
    device = noded.device_state
    forward = device.direction == 0

    assert noded.count == 10
    assert np.all(device.left_coverage_delta[device.source_segment_ids == 0][forward[device.source_segment_ids == 0]] == 2)
    assert np.all(device.left_coverage_delta[device.source_segment_ids == 1][forward[device.source_segment_ids == 1]] == 3)
    assert np.array_equal(
        device.left_coverage_delta[1::2],
        -device.left_coverage_delta[0::2],
    )
    assert np.all(device.right_coverage_delta == 0)


@pytest.mark.gpu
def test_partial_collinear_renoding_sums_every_parent_occurrence() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial import RuntimeSelection
    from vibespatial.overlay.split import renode_gpu_atomic_edges
    from vibespatial.overlay.types import AtomicEdgeDeviceState, AtomicEdgeTable

    source_ids = cp.asarray([0, 0, 1, 1], dtype=cp.int32)
    direction = cp.asarray([0, 1, 0, 1], dtype=cp.int8)
    src_x = cp.asarray([0.0, 2.0, 1.0, 3.0], dtype=cp.float64)
    dst_x = cp.asarray([2.0, 0.0, 3.0, 1.0], dtype=cp.float64)
    zeros_f64 = cp.zeros(4, dtype=cp.float64)
    atomic_edges = AtomicEdgeTable(
        left_segment_count=2,
        right_segment_count=0,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="partial collinear provenance canary",
        ),
        device_state=AtomicEdgeDeviceState(
            source_segment_ids=source_ids,
            direction=direction,
            src_x=src_x,
            src_y=zeros_f64,
            dst_x=dst_x,
            dst_y=zeros_f64.copy(),
            row_indices=cp.zeros(4, dtype=cp.int32),
            part_indices=cp.zeros(4, dtype=cp.int32),
            ring_indices=cp.zeros(4, dtype=cp.int32),
            source_side=cp.asarray([1, 1, 2, 2], dtype=cp.int8),
            source_membership=cp.asarray([1, 1, 2, 2], dtype=cp.uint8),
            tangent_x=dst_x - src_x,
            tangent_y=zeros_f64.copy(),
            left_coverage_delta=cp.asarray([2, -2, 3, -3], dtype=cp.int32),
            right_coverage_delta=cp.asarray([7, -7, 11, -11], dtype=cp.int32),
        ),
        _count=4,
    )

    noded = renode_gpu_atomic_edges(atomic_edges, isolate_rows=True)
    device = noded.device_state
    forward = cp.asarray(device.direction, dtype=cp.int8) == 0
    actual = cp.asnumpy(
        cp.stack(
            (
                cp.asarray(device.src_x)[forward],
                cp.asarray(device.dst_x)[forward],
                cp.asarray(device.left_coverage_delta)[forward],
                cp.asarray(device.right_coverage_delta)[forward],
                cp.asarray(device.source_membership)[forward],
            ),
            axis=1,
        )
    )

    assert noded.count == 6
    assert np.array_equal(
        actual,
        np.asarray(
            [
                [0.0, 1.0, 2, 7, 1],
                [1.0, 2.0, 5, 18, 3],
                [2.0, 3.0, 3, 11, 2],
            ]
        ),
    )
    assert np.array_equal(
        cp.asnumpy(cp.asarray(device.left_coverage_delta)[1::2]),
        -cp.asnumpy(cp.asarray(device.left_coverage_delta)[0::2]),
    )
    assert np.array_equal(
        cp.asnumpy(cp.asarray(device.right_coverage_delta)[1::2]),
        -cp.asnumpy(cp.asarray(device.right_coverage_delta)[0::2]),
    )


@pytest.mark.gpu
def test_gpu_atomic_edges_collapse_duplicate_overlap_segments() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 5, 2, 7)])
    right = from_shapely_geometries([box(1, 5, 3, 7)])

    split_events = build_gpu_split_events(left, right, dispatch_mode=ExecutionMode.GPU)
    atomic_edges = build_gpu_atomic_edges(split_events)

    forward_mask = atomic_edges.direction == 0
    forward_segments = np.column_stack(
        (
            atomic_edges.src_x[forward_mask],
            atomic_edges.src_y[forward_mask],
            atomic_edges.dst_x[forward_mask],
            atomic_edges.dst_y[forward_mask],
        )
    )
    assert atomic_edges.count == 20
    assert np.unique(np.round(forward_segments, 12), axis=0).shape[0] == forward_segments.shape[0]


@pytest.mark.gpu
def test_grouped_right_right_split_events_use_original_right_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 10, 10)])
    right = from_shapely_geometries([box(2, 2, 7, 7), box(5, 0, 9, 9)])

    split_events = build_gpu_split_events(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        require_same_row=True,
        use_same_row_fast_path=False,
        right_geometry_source_rows=np.asarray([0, 0], dtype=np.int32),
    )

    base_endpoint_count = 2 * (split_events.left_segment_count + split_events.right_segment_count)
    right_extra = (split_events.source_side == 2) & (split_events.t > 0.0) & (split_events.t < 1.0)

    assert split_events.count > base_endpoint_count
    assert right_extra.any()


def test_face_metrics_kernel_block_size_rounds_down_to_power_of_two() -> None:
    assert _largest_power_of_two_block_size(256) == 256
    assert _largest_power_of_two_block_size(224) == 128
    assert _largest_power_of_two_block_size(192) == 128
    assert _largest_power_of_two_block_size(1) == 1
