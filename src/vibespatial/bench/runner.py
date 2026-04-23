"""Benchmark orchestration: warmup, repeat/median, pipeline wrapping."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import replace
from time import perf_counter
from typing import Any

import orjson

from .catalog import ensure_operations_loaded, get_operation
from .schema import (
    BenchmarkResult,
    GpuUtilSummary,
    SuiteResult,
    TimingSummary,
    TransferSummary,
    benchmark_result_from_dict,
    suite_result_from_dict,
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


def _resolve_pipeline_profile_mode(
    profile_mode: str,
    *,
    gpu_sparkline: bool = False,
    trace: bool = False,
) -> str:
    if profile_mode not in {"lean", "audit"}:
        raise ValueError("profile_mode must be 'lean' or 'audit'")
    if gpu_sparkline or trace:
        return "audit"
    return profile_mode


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
    profile_mode: str = "lean",
) -> list[BenchmarkResult]:
    """Run a named pipeline benchmark, returning results per scale."""
    from .pipeline import (
        PIPELINE_DEFINITIONS,
        benchmark_pipeline_suite,
    )
    effective_profile_mode = _resolve_pipeline_profile_mode(
        profile_mode,
        gpu_sparkline=gpu_sparkline,
        trace=trace,
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
        profile_mode=effective_profile_mode,
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

    runtime_d2h_count = getattr(pr, "runtime_d2h_transfer_count", None)
    transfers = TransferSummary(
        d2h_count=(
            pr.transfer_count if runtime_d2h_count is None else runtime_d2h_count
        ),
        h2d_count=0,
        total_bytes=int(getattr(pr, "runtime_d2h_transfer_bytes", 0) or 0),
        total_seconds=float(getattr(pr, "runtime_d2h_transfer_seconds", 0.0) or 0.0),
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
            "profile_mode": getattr(pr, "profile_mode", "lean"),
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


def _scale_flag(scale: int) -> str:
    for flag, value in SCALE_MAP.items():
        if value == scale:
            return flag
    raise ValueError(f"scale {scale!r} is not representable by vsbench --scale")


def _append_common_child_flags(
    command: list[str],
    *,
    repeat: int,
    precision: str,
    input_format: str,
    nvtx: bool,
    gpu_sparkline: bool,
    trace: bool,
) -> None:
    command.extend(
        [
            "--repeat",
            str(repeat),
            "--precision",
            precision,
            "--input-format",
            input_format,
            "--json",
            "--quiet",
        ]
    )
    if nvtx:
        command.append("--nvtx")
    if gpu_sparkline:
        command.append("--gpu-sparkline")
    if trace:
        command.append("--trace")


def _operation_child_command(
    op_name: str,
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str,
    nvtx: bool,
    gpu_sparkline: bool,
    trace: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vibespatial.bench.cli",
        "run",
        op_name,
        "--rows",
        str(scale),
    ]
    if compare:
        command.extend(["--compare", compare])
    _append_common_child_flags(
        command,
        repeat=repeat,
        precision=precision,
        input_format=input_format,
        nvtx=nvtx,
        gpu_sparkline=gpu_sparkline,
        trace=trace,
    )
    return command


def _pipeline_child_command(
    pipeline_name: str,
    *,
    level: str,
    repeat: int,
    nvtx: bool,
    gpu_sparkline: bool,
    trace: bool,
    profile_mode: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vibespatial.bench.cli",
        "pipeline",
        pipeline_name,
        "--suite-level",
        level,
        "--repeat",
        str(repeat),
        "--json",
        "--quiet",
    ]
    if nvtx:
        command.append("--nvtx")
    if gpu_sparkline:
        command.append("--gpu-sparkline")
    if trace:
        command.append("--trace")
    if profile_mode != "lean":
        command.extend(["--profile-mode", profile_mode])
    return command


def _kernel_child_command(
    kernel_name: str,
    *,
    scale: int,
    precision: str,
    gpu_sparkline: bool,
    nvtx: bool,
    trace: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vibespatial.bench.cli",
        "kernel",
        kernel_name,
        "--scale",
        _scale_flag(scale),
        "--precision",
        precision,
        "--json",
        "--quiet",
    ]
    if gpu_sparkline:
        command.append("--gpu-sparkline")
    if nvtx:
        command.append("--nvtx")
    if trace:
        command.append("--trace")
    return command


def _text_tail(text: str, *, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def _extract_json_payload(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError("subprocess stdout did not contain a JSON object")
    return orjson.loads(stdout[start : end + 1])


def _gpu_compute_apps() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    current_pid = str(os.getpid())
    apps: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=2)]
        if len(parts) != 3:
            continue
        if parts[0] == current_pid:
            continue
        apps.append(
            {
                "pid": parts[0],
                "process_name": parts[1],
                "used_memory_mib": parts[2],
            }
        )
    return apps


def _release_local_gpu_memory() -> None:
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _run_child_process(
    command: list[str],
    *,
    item_timeout: int,
) -> tuple[int | None, str, str, bool]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=item_timeout)
        return process.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return None, stdout, stderr, True


def _with_child_metadata(
    result: BenchmarkResult,
    *,
    command: list[str],
    returncode: int | None,
    stderr: str,
    gpu_after: list[dict[str, str]],
    timed_out: bool = False,
    item_timeout: int | None = None,
) -> BenchmarkResult:
    metadata = dict(result.metadata)
    metadata["isolated_subprocess"] = True
    metadata["subprocess_command"] = command
    metadata["subprocess_returncode"] = returncode
    if timed_out and item_timeout is not None:
        metadata["subprocess_timeout_seconds"] = item_timeout
    if stderr and (returncode or timed_out):
        metadata["subprocess_stderr_tail"] = _text_tail(stderr)
    if gpu_after:
        metadata["gpu_compute_apps_after_subprocess"] = gpu_after
    return replace(result, metadata=metadata)


def _parse_child_results(payload: dict[str, Any]) -> list[BenchmarkResult]:
    if "results" in payload:
        return suite_result_from_dict(payload).results
    return [benchmark_result_from_dict(payload)]


def _isolated_error_result(
    *,
    operation: str,
    tier: int,
    scale: int,
    geometry_type: str,
    precision: str,
    status_reason: str,
    command: list[str],
    returncode: int | None,
    stdout: str,
    stderr: str,
    gpu_after: list[dict[str, str]],
    timed_out: bool = False,
    item_timeout: int | None = None,
) -> BenchmarkResult:
    metadata: dict[str, Any] = {
        "isolated_subprocess": True,
        "subprocess_command": command,
        "subprocess_returncode": returncode,
    }
    if timed_out and item_timeout is not None:
        metadata["subprocess_timeout_seconds"] = item_timeout
    if stdout:
        metadata["subprocess_stdout_tail"] = _text_tail(stdout)
    if stderr:
        metadata["subprocess_stderr_tail"] = _text_tail(stderr)
    if gpu_after:
        metadata["gpu_compute_apps_after_subprocess"] = gpu_after
    return BenchmarkResult(
        operation=operation,
        tier=tier,
        scale=scale,
        geometry_type=geometry_type,
        precision=precision,
        status="error",
        status_reason=status_reason,
        timing=timing_from_samples([]),
        metadata=metadata,
    )


def _run_isolated_benchmark_command(
    command: list[str],
    *,
    operation: str,
    tier: int,
    scale: int,
    geometry_type: str,
    precision: str,
    item_timeout: int,
) -> list[BenchmarkResult]:
    returncode, stdout, stderr, timed_out = _run_child_process(
        command,
        item_timeout=item_timeout,
    )
    gpu_after = _gpu_compute_apps()
    if timed_out:
        return [
            _isolated_error_result(
                operation=operation,
                tier=tier,
                scale=scale,
                geometry_type=geometry_type,
                precision=precision,
                status_reason=f"isolated subprocess timed out after {item_timeout}s",
                command=command,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                gpu_after=gpu_after,
                timed_out=True,
                item_timeout=item_timeout,
            )
        ]

    try:
        results = _parse_child_results(_extract_json_payload(stdout))
    except Exception as exc:
        detail = _text_tail(stderr or stdout)
        if detail:
            status_reason = f"isolated subprocess JSON parse failed: {exc}; {detail}"
        else:
            status_reason = f"isolated subprocess JSON parse failed: {exc}"
        return [
            _isolated_error_result(
                operation=operation,
                tier=tier,
                scale=scale,
                geometry_type=geometry_type,
                precision=precision,
                status_reason=status_reason,
                command=command,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                gpu_after=gpu_after,
            )
        ]

    return [
        _with_child_metadata(
            result,
            command=command,
            returncode=returncode,
            stderr=stderr,
            gpu_after=gpu_after,
        )
        for result in results
    ]


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
    profile_mode: str = "lean",
    pipelines_filter: list[str] | None = None,
    isolated: bool = True,
    item_timeout: int = 600,
) -> SuiteResult:
    """Run a predefined benchmark suite (smoke, ci, full)."""
    from .suites import SUITES

    if level not in SUITES:
        raise KeyError(f"Unknown suite: {level!r}. Available: {', '.join(SUITES)}")
    suite_def = SUITES[level]
    results: list[BenchmarkResult] = []

    active_pipelines = suite_def.pipelines
    if pipelines_filter:
        active_pipelines = tuple(p for p in active_pipelines if p in pipelines_filter)
    effective_profile_mode = _resolve_pipeline_profile_mode(
        profile_mode,
        gpu_sparkline=gpu_sparkline,
        trace=trace,
    )

    if isolated:
        registered_ops = list(suite_def.operations)
    else:
        ensure_operations_loaded()
        from .catalog import _OPERATION_REGISTRY

        registered_ops = [
            op
            for op in suite_def.operations
            if op in _OPERATION_REGISTRY and _OPERATION_REGISTRY[op].public_api
        ]
    total_items = (
        len(registered_ops) * len(suite_def.scales)
        + len(active_pipelines)
        + len(suite_def.kernels) * len(suite_def.scales)
    )
    if isolated:
        return _run_suite_isolated(
            level,
            repeat=repeat,
            compare=compare,
            precision=precision,
            input_format=input_format,
            nvtx=nvtx,
            gpu_sparkline=gpu_sparkline,
            trace=trace,
            profile_mode=effective_profile_mode,
            active_pipelines=active_pipelines,
            registered_ops=registered_ops,
            total_items=total_items,
            item_timeout=item_timeout,
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
            finally:
                _release_local_gpu_memory()
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
                profile_mode=effective_profile_mode,
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
        finally:
            _release_local_gpu_memory()

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
                    finally:
                        _release_local_gpu_memory()
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
            "profile_mode": effective_profile_mode,
            "isolated": False,
        },
    )


def _run_suite_isolated(
    level: str,
    *,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str,
    nvtx: bool,
    gpu_sparkline: bool,
    trace: bool,
    profile_mode: str,
    active_pipelines: tuple[str, ...],
    registered_ops: list[str],
    total_items: int,
    item_timeout: int,
) -> SuiteResult:
    """Run suite items in child processes to bound GPU state lifetime."""
    from .suites import SUITES

    suite_def = SUITES[level]
    results: list[BenchmarkResult] = []
    item_idx = 0
    suite_start = perf_counter()

    print(
        f"\033[1mvsbench suite {level}\033[0m — "
        f"{len(registered_ops)} ops × {len(suite_def.scales)} scales + "
        f"{len(active_pipelines)} pipelines "
        f"(repeat={repeat}, compare={compare or 'none'}, isolated=true, "
        f"item_timeout={item_timeout}s)",
        file=sys.stderr,
        flush=True,
    )

    for op_name in registered_ops:
        for scale in suite_def.scales:
            item_idx += 1
            command = _operation_child_command(
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
            item_results = _run_isolated_benchmark_command(
                command,
                operation=op_name,
                tier=1,
                scale=scale,
                geometry_type="unknown",
                precision=precision,
                item_timeout=item_timeout,
            )
            results.extend(item_results)
            _progress(item_results[0], idx=item_idx, total=total_items)

    for pipeline_name in active_pipelines:
        item_idx += 1
        command = _pipeline_child_command(
            pipeline_name,
            level=level,
            repeat=repeat,
            nvtx=nvtx,
            gpu_sparkline=gpu_sparkline,
            trace=trace,
            profile_mode=profile_mode,
        )
        item_results = _run_isolated_benchmark_command(
            command,
            operation=pipeline_name,
            tier=1,
            scale=suite_def.scales[0] if suite_def.scales else 0,
            geometry_type="mixed",
            precision="auto",
            item_timeout=item_timeout,
        )
        own_results = [pr for pr in item_results if pr.operation == pipeline_name]
        other_results = [pr for pr in item_results if pr.operation != pipeline_name]
        results.extend(own_results)
        results.extend(other_results)
        progress_results = own_results or item_results
        for pr in progress_results:
            _progress(pr, idx=item_idx, total=total_items)

    if suite_def.kernels:
        try:
            import vibespatial.bench.nvbench_runner  # noqa: F401
        except ImportError:
            pass
        else:
            for kernel_name in suite_def.kernels:
                for scale in suite_def.scales:
                    item_idx += 1
                    command = _kernel_child_command(
                        kernel_name,
                        scale=scale,
                        precision=precision,
                        gpu_sparkline=gpu_sparkline,
                        nvtx=nvtx,
                        trace=trace,
                    )
                    item_results = _run_isolated_benchmark_command(
                        command,
                        operation=kernel_name,
                        tier=2,
                        scale=scale,
                        geometry_type="unknown",
                        precision=precision,
                        item_timeout=item_timeout,
                    )
                    results.extend(item_results)
                    _progress(item_results[0], idx=item_idx, total=total_items)

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
            "profile_mode": profile_mode,
            "isolated": True,
            "item_timeout_seconds": item_timeout,
            "cleanup_policy": "child-process-exit plus owned-process-group kill on timeout",
            "gpu_compute_apps_after_suite": _gpu_compute_apps(),
        },
    )
