from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vibespatial.bench import profile_rails as profile_rails_module
from vibespatial.bench.profile_rails import (
    profile_join_kernel,
    profile_overlay_kernel,
    profile_spatial_query_stack,
)
from vibespatial.bench.profiling import StageProfiler
from vibespatial.runtime import ExecutionMode


class _FakeGpuSampler:
    def __init__(self) -> None:
        self.available = True
        self._index = 0
        self._samples = (
            {
                "device_name": "fake-gpu",
                "sm_utilization_pct": 12.0,
                "memory_utilization_pct": 24.0,
                "used_bytes": 100,
                "total_bytes": 1000,
            },
            {
                "device_name": "fake-gpu",
                "sm_utilization_pct": 48.0,
                "memory_utilization_pct": 72.0,
                "used_bytes": 400,
                "total_bytes": 1000,
            },
            {
                "device_name": "fake-gpu",
                "sm_utilization_pct": 36.0,
                "memory_utilization_pct": 60.0,
                "used_bytes": 320,
                "total_bytes": 1000,
            },
        )

    def sample(self):
        from vibespatial.bench.profiling import GpuTelemetrySample

        sample = self._samples[min(self._index, len(self._samples) - 1)]
        self._index += 1
        return GpuTelemetrySample(**sample)


class _FakeGpuEventTimer:
    def __init__(self, elapsed_seconds: float) -> None:
        self.elapsed_seconds = elapsed_seconds
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def summarize(self) -> dict[str, object]:
        return {
            "gpu_event_elapsed_seconds": self.elapsed_seconds,
            "gpu_event_elapsed_display": f"{self.elapsed_seconds * 1_000_000:.0f}us",
            "gpu_timing_source": "cuda_event",
        }


def test_stage_profiler_records_stage_metadata() -> None:
    profiler = StageProfiler(
        operation="join",
        dataset="smoke",
        requested_runtime=ExecutionMode.AUTO,
        selected_runtime=ExecutionMode.CPU,
        gpu_sampler=_FakeGpuSampler(),
        gpu_sample_interval_seconds=0.001,
        retain_gpu_trace=True,
        include_gpu_sparklines=True,
    )

    with profiler.stage(
        "coarse_filter",
        category="filter",
        device=ExecutionMode.CPU,
        rows_in=128,
        detail="candidate pruning",
    ) as stage:
        stage.rows_out = 16
        stage.metadata["pairs_examined"] = 1024

    trace = profiler.finish(metadata={"matched_pairs": 16})

    assert trace.operation == "join"
    assert trace.selected_runtime == "cpu"
    assert trace.metadata["matched_pairs"] == 16
    assert len(trace.stages) == 1
    assert trace.stages[0].category == "filter"
    assert trace.stages[0].rows_in == 128
    assert trace.stages[0].rows_out == 16
    assert trace.stages[0].metadata["pairs_examined"] == 1024
    assert trace.stages[0].metadata["gpu_device_name"] == "fake-gpu"
    assert trace.stages[0].metadata["gpu_utilization_pct_start"] == 12.0
    assert trace.stages[0].metadata["gpu_utilization_pct_end"] in {12.0, 36.0, 48.0}
    assert trace.stages[0].metadata["gpu_utilization_pct_max"] in {12.0, 48.0}
    assert trace.stages[0].metadata["gpu_utilization_pct_avg"] >= 12.0
    assert trace.stages[0].metadata["gpu_vram_used_bytes_start"] == 100
    assert trace.stages[0].metadata["gpu_vram_used_bytes_max"] in {100, 400}
    assert trace.stages[0].metadata["gpu_sample_count"] >= 1
    assert len(trace.stages[0].metadata["gpu_trace"]) >= 1
    assert "|" in trace.stages[0].metadata["gpu_util_sparkline"]
    assert "|" in trace.stages[0].metadata["gpu_memory_util_sparkline"]
    assert "|" in trace.stages[0].metadata["gpu_vram_sparkline"]
    assert trace.stages[0].metadata["elapsed_display"]


def test_stage_profiler_records_cuda_event_timing_for_gpu_stage() -> None:
    profiler = StageProfiler(
        operation="overlay",
        dataset="smoke",
        requested_runtime=ExecutionMode.AUTO,
        selected_runtime=ExecutionMode.GPU,
        gpu_sampler=_FakeGpuSampler(),
        gpu_event_timer_factory=lambda: _FakeGpuEventTimer(0.000245),
    )

    with profiler.stage(
        "refine_intersections",
        category="refine",
        device=ExecutionMode.GPU,
        rows_in=64,
    ) as stage:
        stage.rows_out = 32

    trace = profiler.finish()
    metadata = trace.stages[0].metadata

    assert metadata["gpu_event_elapsed_seconds"] == 0.000245
    assert metadata["gpu_event_elapsed_display"] == "245us"
    assert metadata["gpu_timing_source"] == "cuda_event"
    assert metadata["elapsed_display"]


def test_stage_profiler_skips_cuda_event_metadata_for_cpu_stage() -> None:
    timer = _FakeGpuEventTimer(0.000245)
    profiler = StageProfiler(
        operation="join",
        dataset="smoke",
        requested_runtime=ExecutionMode.AUTO,
        selected_runtime=ExecutionMode.CPU,
        gpu_sampler=_FakeGpuSampler(),
        gpu_event_timer_factory=lambda: timer,
    )

    with profiler.stage(
        "coarse_filter",
        category="filter",
        device=ExecutionMode.CPU,
        rows_in=64,
    ):
        pass

    trace = profiler.finish()
    metadata = trace.stages[0].metadata

    assert "gpu_event_elapsed_seconds" not in metadata
    assert metadata["elapsed_display"]
    assert timer.started is False
    assert timer.stopped is False


def test_profile_join_kernel_distinguishes_sort_filter_and_refine() -> None:
    trace = profile_join_kernel(rows=64, overlap_ratio=0.25, tile_size=32)

    categories = {stage.category for stage in trace.stages}

    assert trace.operation == "join"
    assert {"sort", "filter", "refine"} <= categories
    assert any(stage.name == "coarse_filter" for stage in trace.stages)
    assert any(stage.name == "refine_predicate" for stage in trace.stages)
    assert trace.metadata["matched_pairs"] >= 0


def test_profile_join_kernel_reports_gpu_when_regular_grid_query_fast_path_runs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(profile_rails_module, "has_gpu_runtime", lambda: True)
    monkeypatch.setattr(
        profile_rails_module,
        "_build_join_inputs",
        lambda rows, overlap_ratio: (
            np.asarray([object()] * rows, dtype=object),
            np.asarray([object()] * rows, dtype=object),
        ),
    )

    fake_owned = SimpleNamespace(row_count=8)
    fake_index = SimpleNamespace(
        size=8,
        regular_grid=object(),
        bounds=np.zeros((8, 4), dtype=np.float64),
        geometry_array=fake_owned,
    )

    monkeypatch.setattr(profile_rails_module, "from_shapely_geometries", lambda values: fake_owned)
    monkeypatch.setattr(profile_rails_module, "build_flat_spatial_index", lambda owned: fake_index)
    monkeypatch.setattr(
        profile_rails_module,
        "_extract_box_query_bounds_from_owned",
        lambda predicate, owned: np.zeros((owned.row_count, 4), dtype=np.float64),
    )
    monkeypatch.setattr(
        profile_rails_module,
        "_query_regular_grid_rect_box_index",
        lambda flat_index, bounds, predicate=None: (
            np.asarray([1, 0], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
        ),
    )

    trace = profile_rails_module.profile_join_kernel(rows=8, overlap_ratio=0.25)

    coarse_stage = next(stage for stage in trace.stages if stage.name == "coarse_filter")
    refine_stage = next(stage for stage in trace.stages if stage.name == "refine_predicate")
    sort_stage = next(stage for stage in trace.stages if stage.name == "sort_output")

    assert trace.selected_runtime == "gpu"
    assert trace.metadata["actual_selected_runtime"] == "gpu"
    assert trace.metadata["execution_implementation"] == "owned_gpu_spatial_query"
    assert trace.metadata["selected_path"] == "regular_grid_rect_box"
    assert coarse_stage.device == "gpu"
    assert coarse_stage.metadata["candidate_mode"] == "regular_grid_rect_box"
    assert coarse_stage.metadata["fast_path_hit"] is True
    assert refine_stage.device == "gpu"
    assert refine_stage.metadata["refine_elided"] is True
    assert refine_stage.metadata["refine_selected_runtime"] == "gpu"
    assert sort_stage.device == "gpu"


def test_profile_overlay_kernel_distinguishes_sort_filter_and_refine() -> None:
    trace = profile_overlay_kernel(rows=32, tile_size=64)

    categories = {stage.category for stage in trace.stages}

    assert trace.operation == "overlay"
    assert {"sort", "filter", "refine"} <= categories
    assert any(stage.name == "filter_segment_candidates" for stage in trace.stages)
    assert any(stage.name == "refine_intersections" for stage in trace.stages)
    assert "sort_reconstruction_events" in [stage.name for stage in trace.stages]
    assert trace.metadata["candidate_pairs"] >= 0


def test_profile_overlay_kernel_reports_gpu_when_gpu_segment_pipeline_runs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(profile_rails_module, "has_gpu_runtime", lambda: True)
    monkeypatch.setattr(profile_rails_module, "_sync_gpu_profile_stage", lambda: None)
    monkeypatch.setattr(
        profile_rails_module,
        "cp",
        SimpleNamespace(
            asarray=lambda value: np.asarray(value),
            count_nonzero=lambda value: np.count_nonzero(value),
            lexsort=lambda keys: np.lexsort(keys),
            stack=lambda keys: np.stack(keys),
        ),
    )

    fake_segments = SimpleNamespace(count=6)
    fake_candidates = SimpleNamespace(count=7)
    fake_device_state = SimpleNamespace(
        ambiguous_rows=np.asarray([1, 5], dtype=np.int32),
        kinds=np.asarray([1, 2, 3, 1, 2, 3, 1], dtype=np.int8),
        left_rows=np.arange(7, dtype=np.int32),
        left_segments=np.arange(7, dtype=np.int32),
        right_rows=np.arange(7, dtype=np.int32),
        right_segments=np.arange(7, dtype=np.int32),
    )

    monkeypatch.setattr(
        profile_rails_module,
        "_extract_segments_gpu",
        lambda geometry, compute_type="double": fake_segments,
    )
    monkeypatch.setattr(
        profile_rails_module,
        "_generate_segment_candidates_gpu",
        lambda left, right, require_same_row=False: fake_candidates,
    )
    monkeypatch.setattr(
        profile_rails_module,
        "_classify_segment_intersections_gpu",
        lambda **kwargs: SimpleNamespace(
            count=7,
            device_state=fake_device_state,
        ),
    )

    trace = profile_rails_module.profile_overlay_kernel(rows=8, tile_size=64)

    filter_stage = next(stage for stage in trace.stages if stage.name == "filter_segment_candidates")
    refine_stage = next(stage for stage in trace.stages if stage.name == "refine_intersections")
    sort_stage = next(stage for stage in trace.stages if stage.name == "sort_reconstruction_events")

    assert trace.selected_runtime == "gpu"
    assert trace.metadata["actual_selected_runtime"] == "gpu"
    assert trace.metadata["execution_implementation"] == "gpu_segment_overlay_profile"
    assert filter_stage.device == "gpu"
    assert filter_stage.metadata["candidate_mode"] == "gpu_sort_sweep"
    assert refine_stage.device == "gpu"
    assert refine_stage.metadata["ambiguous_pairs"] == 2
    assert refine_stage.metadata["proper_pairs"] == 3
    assert refine_stage.metadata["touch_pairs"] == 2
    assert refine_stage.metadata["overlap_pairs"] == 2
    assert sort_stage.device == "gpu"


def test_profile_spatial_query_stack_reports_full_path() -> None:
    trace = profile_spatial_query_stack(rows=64, overlap_ratio=0.25)

    stage_names = [stage.name for stage in trace.stages]
    categories = {stage.category for stage in trace.stages}

    assert trace.operation == "spatial_query"
    assert {"setup", "filter", "emit"} <= categories
    assert "build_tree_owned" in stage_names
    assert "build_flat_index" in stage_names
    assert "build_query_owned" in stage_names
    assert "format_output" in stage_names
    assert trace.metadata["matched_pairs"] >= 0
    assert trace.metadata["selected_path"] in {
        "point_box_query",
        "regular_grid_rect_box",
        "regular_grid_point",
        "owned_point_box_query",
        "generic_query_pipeline",
    }


def test_profile_spatial_query_stack_fast_path_does_not_claim_gpu_without_runtime(
    monkeypatch,
) -> None:
    monkeypatch.setattr(profile_rails_module, "has_gpu_runtime", lambda: False)
    monkeypatch.setattr(
        profile_rails_module,
        "_query_point_tree_box_index",
        lambda *args, **kwargs: (np.asarray([0], dtype=np.int32), np.asarray([0], dtype=np.int32)),
    )

    trace = profile_rails_module.profile_spatial_query_stack(rows=8, overlap_ratio=0.25)

    assert trace.metadata["selected_path"] == "point_box_query"
    assert trace.metadata["actual_selected_runtime"] == "cpu"
    assert trace.metadata["execution_implementation"] == "owned_cpu_spatial_query"
    probe_stage = next(stage for stage in trace.stages if stage.name == "probe_query_point_box")
    assert probe_stage.device == "cpu"
