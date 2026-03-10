from __future__ import annotations

from vibespatial.profile_rails import profile_join_kernel, profile_overlay_kernel, profile_spatial_query_stack
from vibespatial.profiling import StageProfiler
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
        from vibespatial.profiling import GpuTelemetrySample

        sample = self._samples[min(self._index, len(self._samples) - 1)]
        self._index += 1
        return GpuTelemetrySample(**sample)


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


def test_profile_join_kernel_distinguishes_sort_filter_and_refine() -> None:
    trace = profile_join_kernel(rows=64, overlap_ratio=0.25, tile_size=32)

    categories = {stage.category for stage in trace.stages}

    assert trace.operation == "join"
    assert {"sort", "filter", "refine"} <= categories
    assert any(stage.name == "coarse_filter" for stage in trace.stages)
    assert any(stage.name == "refine_predicate" for stage in trace.stages)
    assert trace.metadata["matched_pairs"] >= 0


def test_profile_overlay_kernel_distinguishes_sort_filter_and_refine() -> None:
    trace = profile_overlay_kernel(rows=32, tile_size=64)

    categories = {stage.category for stage in trace.stages}

    assert trace.operation == "overlay"
    assert {"sort", "filter", "refine"} <= categories
    assert any(stage.name == "filter_segment_candidates" for stage in trace.stages)
    assert any(stage.name == "refine_intersections" for stage in trace.stages)
    assert "sort_reconstruction_events" in [stage.name for stage in trace.stages]
    assert trace.metadata["candidate_pairs"] >= 0


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
