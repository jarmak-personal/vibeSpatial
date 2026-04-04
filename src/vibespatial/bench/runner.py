"""Benchmark orchestration: warmup, repeat/median, pipeline wrapping."""
from __future__ import annotations

import sys
from dataclasses import replace
from time import perf_counter
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
    input_format: str = "parquet",
    nvtx: bool = False,
    gpu_sparkline: bool = False,
    trace: bool = False,
    operation_args: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run a registered benchmark operation with warmup and repeat/median."""
    ensure_operations_loaded()
    spec = get_operation(name)
    resolved_operation_args = dict(operation_args or {})

    # Clip scale to max_scale if the operation declares one
    effective_scale = min(scale, spec.max_scale) if spec.max_scale else scale

    # Warmup run (discarded)
    try:
        spec.callable(
            scale=min(effective_scale, 1_000),
            repeat=1,
            compare=None,
            precision=precision,
            input_format=input_format,
            nvtx=False,
            gpu_sparkline=False,
            trace=False,
            **resolved_operation_args,
        )
    except Exception:
        pass  # warmup failures are non-fatal

    # Timed runs
    samples: list[BenchmarkResult] = []
    for _ in range(max(1, repeat)):
        result = spec.callable(
            scale=effective_scale,
            repeat=1,
            compare=compare,
            precision=precision,
            input_format=input_format,
            nvtx=nvtx,
            gpu_sparkline=gpu_sparkline,
            trace=trace,
            **resolved_operation_args,
        )
        samples.append(result)

    result = _select_median(samples)
    if not resolved_operation_args:
        return result
    metadata = dict(result.metadata)
    metadata["operation_args"] = resolved_operation_args
    return replace(result, metadata=metadata)


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
    gpu_util = _extract_gpu_util(pr.stages, selected_runtime=pr.selected_runtime)

    # Determine pass/fail/skip
    if pr.status == "ok":
        status = "pass"
    elif pr.status == "deferred":
        status = "skip"
    else:
        status = "fail"
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


def _iter_stage_entries(stages: tuple[Any, ...]):
    from .profiling import ProfileTrace

    for stage in stages:
        if isinstance(stage, ProfileTrace):
            for trace_stage in stage.stages:
                yield trace_stage.device, trace_stage.metadata
        elif isinstance(stage, dict):
            for trace_stage in stage.get("stages", ()):
                yield trace_stage.get("device"), trace_stage.get("metadata", {})


def _extract_gpu_util(
    stages: tuple[Any, ...],
    *,
    selected_runtime: str | None = None,
) -> GpuUtilSummary | None:
    """Try to pull GPU util summary from stage metadata."""
    if selected_runtime not in {"gpu", "hybrid"}:
        return None

    for device, meta in _iter_stage_entries(stages):
        if device != "gpu":
            continue
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

def _fmt_scale(scale: int) -> str:
    if scale >= 1_000_000:
        return f"{scale // 1_000_000}M"
    if scale >= 1_000:
        return f"{scale // 1_000}K"
    return str(scale)


def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}\u00b5s"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f}ms"
    return f"{seconds:.2f}s"


def _progress(result: BenchmarkResult, *, idx: int, total: int) -> None:
    """Print a single-line progress update to stderr.

    Uses ``get_real_stderr()`` so output is visible even while CCCL
    background threads have fd 2 redirected to /dev/null.
    """
    from vibespatial.cuda.cccl_precompile import get_real_stderr

    status = {"pass": "\033[32mPASS\033[0m", "fail": "\033[31mFAIL\033[0m",
              "error": "\033[31mERR\033[0m", "skip": "\033[90mSKIP\033[0m"}.get(
        result.status, result.status)
    speedup = f" {result.speedup:.1f}x" if result.speedup is not None else ""
    time_str = _fmt_time(result.timing.median_seconds) if result.timing.sample_count > 0 else "-"
    print(
        f"  [{idx}/{total}] {status} {result.operation:<22} "
        f"scale={_fmt_scale(result.scale):<4} {time_str:>8}{speedup}",
        file=get_real_stderr(),
        flush=True,
    )


def run_suite(
    level: str,
    *,
    repeat: int = 3,
    compare: str | None = None,
    precision: str = "auto",
    input_format: str = "parquet",
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

    # Count total work items for progress
    ensure_operations_loaded()
    from .catalog import _OPERATION_REGISTRY

    active_pipelines = suite_def.pipelines
    if pipelines_filter:
        active_pipelines = tuple(p for p in active_pipelines if p in pipelines_filter)

    registered_ops = [op for op in suite_def.operations if op in _OPERATION_REGISTRY]
    total_items = (
        len(registered_ops) * len(suite_def.scales)
        + len(active_pipelines)
        + len(suite_def.kernels) * len(suite_def.scales)
    )
    item_idx = 0
    suite_start = perf_counter()

    print(
        f"\033[1mvsbench suite {level}\033[0m — "
        f"{len(registered_ops)} ops × {len(suite_def.scales)} scales + "
        f"{len(active_pipelines)} pipelines "
        f"(repeat={repeat}, compare={compare or 'none'})",
        file=sys.stderr,
        flush=True,
    )

    # Run operations
    for op_name in registered_ops:
        for scale in suite_def.scales:
            item_idx += 1
            try:
                result = run_operation(
                    op_name,
                    scale=scale,
                    repeat=repeat,
                    compare=compare,
                    precision=precision,
                    input_format=input_format,
                    nvtx=nvtx,
                    gpu_sparkline=gpu_sparkline,
                    trace=trace,
                )
                results.append(result)
            except Exception as exc:
                result = BenchmarkResult(
                    operation=op_name,
                    tier=1,
                    scale=scale,
                    geometry_type="unknown",
                    precision=precision,
                    status="error",
                    status_reason=str(exc),
                    timing=timing_from_samples([]),
                )
                results.append(result)
            _progress(result, idx=item_idx, total=total_items)

    # Run pipelines
    for pipeline_name in active_pipelines:
        item_idx += 1
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
            # Filter: only keep results for the requested pipeline (pipeline.py
            # may return raster-to-vector deferred results mixed in)
            own_results = [pr for pr in pipeline_results if pr.operation == pipeline_name]
            other_results = [pr for pr in pipeline_results if pr.operation != pipeline_name]
            results.extend(own_results)
            # Silently collect deferred/other pipeline results without progress noise
            results.extend(other_results)
            for pr in own_results:
                _progress(pr, idx=item_idx, total=total_items)
        except Exception as exc:
            # Use the first suite scale as a best-effort value for the
            # error result (pipeline.py iterates scales internally).
            error_scale = suite_def.scales[0] if suite_def.scales else 0
            result = BenchmarkResult(
                operation=pipeline_name,
                tier=1,
                scale=error_scale,
                geometry_type="mixed",
                precision="auto",
                status="error",
                status_reason=str(exc),
                timing=timing_from_samples([]),
            )
            results.append(result)
            _progress(result, idx=item_idx, total=total_items)

    # Tier 2 kernel benchmarks (only if available)
    if suite_def.kernels:
        try:
            from .nvbench_runner import run_kernel_bench

            for kernel_name in suite_def.kernels:
                for scale in suite_def.scales:
                    item_idx += 1
                    try:
                        result = run_kernel_bench(kernel_name, scale=scale, precision=precision)
                        results.append(result)
                    except Exception as exc:
                        result = BenchmarkResult(
                            operation=kernel_name,
                            tier=2,
                            scale=scale,
                            geometry_type="unknown",
                            precision=precision,
                            status="error",
                            status_reason=str(exc),
                            timing=timing_from_samples([]),
                        )
                        results.append(result)
                    _progress(result, idx=item_idx, total=total_items)
        except ImportError:
            pass  # cuda-bench not installed, skip Tier 2

    elapsed = perf_counter() - suite_start
    passed = sum(1 for r in results if r.status == "pass")
    print(
        f"\n\033[1mDone\033[0m — {passed}/{len(results)} passed in {_fmt_time(elapsed)}",
        file=sys.stderr,
        flush=True,
    )

    return SuiteResult(
        suite_name=level,
        results=results,
        metadata={
            "repeat": repeat,
            "precision": precision,
            "input_format": input_format,
        },
    )
