"""Benchmark orchestration: warmup, repeat/median, pipeline wrapping."""
from __future__ import annotations

from typing import Any

from .catalog import ensure_operations_loaded, get_operation
from .schema import (
    BenchmarkResult,
    GpuUtilSummary,
    SuiteResult,
    TimingSummary,
    TransferSummary,
    timing_from_samples,
)

SCALE_MAP: dict[str, int] = {
    "1k": 1_000,
    "10k": 10_000,
    "100k": 100_000,
    "1m": 1_000_000,
}


def resolve_scale(scale_str: str | None, *, default: int = 100_000) -> int:
    """Convert a scale string like ``'100k'`` to an integer row count."""
    if scale_str is None:
        return default
    return SCALE_MAP.get(scale_str.lower(), default)


# ---------------------------------------------------------------------------
# Operation runner
# ---------------------------------------------------------------------------

def run_operation(
    name: str,
    *,
    scale: int,
    repeat: int = 3,
    compare: str | None = None,
    precision: str = "auto",
    nvtx: bool = False,
    gpu_sparkline: bool = False,
    trace: bool = False,
) -> BenchmarkResult:
    """Run a registered benchmark operation with warmup and repeat/median."""
    ensure_operations_loaded()
    spec = get_operation(name)

    # Warmup run (discarded)
    try:
        spec.callable(
            scale=min(scale, 1_000),
            repeat=1,
            compare=None,
            precision=precision,
            nvtx=False,
            gpu_sparkline=False,
            trace=False,
        )
    except Exception:
        pass  # warmup failures are non-fatal

    # Timed runs
    samples: list[BenchmarkResult] = []
    for _ in range(max(1, repeat)):
        result = spec.callable(
            scale=scale,
            repeat=1,
            compare=compare,
            precision=precision,
            nvtx=nvtx,
            gpu_sparkline=gpu_sparkline,
            trace=trace,
        )
        samples.append(result)

    return _select_median(samples)


def _select_median(samples: list[BenchmarkResult]) -> BenchmarkResult:
    """Return the sample closest to the median elapsed time."""
    if len(samples) == 1:
        return samples[0]
    times = [s.timing.median_seconds for s in samples]
    from statistics import median

    med = median(times)
    return min(samples, key=lambda s: abs(s.timing.median_seconds - med))


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    name: str,
    *,
    suite: str = "ci",
    scale: int | None = None,
    repeat: int = 3,
    nvtx: bool = False,
    gpu_sparkline: bool = False,
    trace: bool = False,
) -> list[BenchmarkResult]:
    """Run a named pipeline benchmark, returning results per scale."""
    from .pipeline import (
        PIPELINE_DEFINITIONS,
        benchmark_pipeline_suite,
    )

    if name not in PIPELINE_DEFINITIONS:
        available = ", ".join(PIPELINE_DEFINITIONS)
        raise KeyError(f"Unknown pipeline: {name!r}. Available: {available}")

    results = benchmark_pipeline_suite(
        suite=suite,
        pipelines=(name,),
        repeat=max(1, repeat),
        enable_nvtx=nvtx,
        retain_gpu_trace=trace,
        include_gpu_sparklines=gpu_sparkline,
    )
    return [_convert_pipeline_result(r) for r in results]


def _convert_pipeline_result(pr: Any) -> BenchmarkResult:
    """Convert a ``PipelineBenchmarkResult`` to unified ``BenchmarkResult``."""
    from .profiling import ProfileTrace

    timing = TimingSummary(
        mean_seconds=pr.elapsed_seconds,
        median_seconds=pr.elapsed_seconds,
        min_seconds=pr.elapsed_seconds,
        max_seconds=pr.elapsed_seconds,
        stddev_seconds=0.0,
        sample_count=1,
    )

    transfers = TransferSummary(
        d2h_count=pr.transfer_count,
        h2d_count=0,
        total_bytes=0,
        total_seconds=0.0,
        offramps=0,
    )

    # Extract GPU util from stage metadata if available
    gpu_util = _extract_gpu_util(pr.stages)

    # Determine pass/fail
    status = "pass" if pr.status == "ok" else "fail"
    status_reason = pr.notes or pr.status

    # Flatten stages to dicts
    stage_dicts: list[dict[str, Any]] = []
    for stage in pr.stages:
        if isinstance(stage, ProfileTrace):
            stage_dicts.append(stage.to_dict())
        elif isinstance(stage, dict):
            stage_dicts.append(stage)

    peak_mem = pr.peak_device_memory_bytes

    return BenchmarkResult(
        operation=pr.pipeline,
        tier=1,
        scale=pr.scale,
        geometry_type="mixed",
        precision="auto",
        status=status,
        status_reason=status_reason,
        timing=timing,
        transfers=transfers,
        gpu_util=gpu_util,
        stages=tuple(stage_dicts),
        metadata={
            "selected_runtime": pr.selected_runtime,
            "planner_selected_runtime": pr.planner_selected_runtime,
            "output_rows": pr.output_rows,
            "materialization_count": pr.materialization_count,
            "fallback_event_count": pr.fallback_event_count,
            "peak_device_memory_bytes": peak_mem,
            "rewrite_event_count": pr.rewrite_event_count,
        },
    )


def _extract_gpu_util(stages: tuple[Any, ...]) -> GpuUtilSummary | None:
    """Try to pull GPU util summary from stage metadata."""
    from .profiling import ProfileTrace

    for stage in stages:
        meta = {}
        if isinstance(stage, ProfileTrace):
            meta = stage.metadata
        elif isinstance(stage, dict):
            meta = stage.get("metadata", {})

        device_name = meta.get("gpu_device_name")
        if device_name:
            return GpuUtilSummary(
                device_name=device_name,
                sm_utilization_pct_avg=meta.get("gpu_utilization_pct_avg", 0.0),
                sm_utilization_pct_max=meta.get("gpu_utilization_pct_max", 0.0),
                memory_utilization_pct_avg=meta.get("gpu_memory_utilization_pct_avg", 0.0),
                vram_used_bytes_max=meta.get("gpu_vram_used_bytes_max", 0),
                vram_total_bytes=meta.get("gpu_vram_total_bytes", 0),
                sparkline=meta.get("gpu_util_sparkline"),
            )
    return None


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_suite(
    level: str,
    *,
    repeat: int = 3,
    compare: str | None = None,
    precision: str = "auto",
    nvtx: bool = False,
    gpu_sparkline: bool = False,
    trace: bool = False,
    pipelines_filter: list[str] | None = None,
) -> SuiteResult:
    """Run a predefined benchmark suite (smoke, ci, full)."""
    from .suites import SUITES

    if level not in SUITES:
        raise KeyError(f"Unknown suite: {level!r}. Available: {', '.join(SUITES)}")
    suite_def = SUITES[level]
    results: list[BenchmarkResult] = []

    # Run operations
    ensure_operations_loaded()
    from .catalog import _OPERATION_REGISTRY

    for op_name in suite_def.operations:
        if op_name not in _OPERATION_REGISTRY:
            continue  # skip unregistered ops (not yet wrapped)
        for scale in suite_def.scales:
            try:
                result = run_operation(
                    op_name,
                    scale=scale,
                    repeat=repeat,
                    compare=compare,
                    precision=precision,
                    nvtx=nvtx,
                    gpu_sparkline=gpu_sparkline,
                    trace=trace,
                )
                results.append(result)
            except Exception as exc:
                results.append(BenchmarkResult(
                    operation=op_name,
                    tier=1,
                    scale=scale,
                    geometry_type="unknown",
                    precision=precision,
                    status="error",
                    status_reason=str(exc),
                    timing=timing_from_samples([]),
                ))

    # Run pipelines
    active_pipelines = suite_def.pipelines
    if pipelines_filter:
        active_pipelines = tuple(p for p in active_pipelines if p in pipelines_filter)

    for pipeline_name in active_pipelines:
        suite_label = level
        try:
            pipeline_results = run_pipeline(
                pipeline_name,
                suite=suite_label,
                repeat=repeat,
                nvtx=nvtx,
                gpu_sparkline=gpu_sparkline,
                trace=trace,
            )
            results.extend(pipeline_results)
        except Exception as exc:
            results.append(BenchmarkResult(
                operation=pipeline_name,
                tier=1,
                scale=0,
                geometry_type="mixed",
                precision="auto",
                status="error",
                status_reason=str(exc),
                timing=timing_from_samples([]),
            ))

    # Tier 2 kernel benchmarks (only if available)
    if suite_def.kernels:
        try:
            from .nvbench_runner import run_kernel_bench

            for kernel_name in suite_def.kernels:
                for scale in suite_def.scales:
                    try:
                        result = run_kernel_bench(kernel_name, scale=scale, precision=precision)
                        results.append(result)
                    except Exception as exc:
                        results.append(BenchmarkResult(
                            operation=kernel_name,
                            tier=2,
                            scale=scale,
                            geometry_type="unknown",
                            precision=precision,
                            status="error",
                            status_reason=str(exc),
                            timing=timing_from_samples([]),
                        ))
        except ImportError:
            pass  # cuda-bench not installed, skip Tier 2

    return SuiteResult(
        suite_name=level,
        results=results,
        metadata={
            "repeat": repeat,
            "precision": precision,
        },
    )
