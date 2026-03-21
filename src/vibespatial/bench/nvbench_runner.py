"""NVBench Tier-2 kernel microbenchmark integration.

Spawns NVBench kernel benchmarks as subprocesses (because cuda-bench
has an "owns main" architecture) and parses their JSON output into
unified ``BenchmarkResult`` objects.

Requires the optional ``cuda-bench`` package:
    pip install cuda-bench[cu12]
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import (
    BenchmarkResult,
    KernelTimingSummary,
    TimingSummary,
    timing_from_samples,
)

# ---------------------------------------------------------------------------
# Kernel bench spec and registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelBenchSpec:
    kernel_name: str
    description: str
    script_path: str
    geometry_types: tuple[str, ...]
    default_scale: int


_KERNEL_BENCH_REGISTRY: dict[str, KernelBenchSpec] = {}


def register_kernel_bench(
    kernel_name: str,
    *,
    description: str,
    script_path: str,
    geometry_types: tuple[str, ...] = ("polygon",),
    default_scale: int = 100_000,
) -> None:
    """Register a kernel benchmark script for CLI discovery."""
    _KERNEL_BENCH_REGISTRY[kernel_name] = KernelBenchSpec(
        kernel_name=kernel_name,
        description=description,
        script_path=script_path,
        geometry_types=geometry_types,
        default_scale=default_scale,
    )


def list_kernel_benches() -> tuple[KernelBenchSpec, ...]:
    """Return all registered kernel benchmark specs."""
    _ensure_kernels_loaded()
    return tuple(_KERNEL_BENCH_REGISTRY.values())


def _ensure_kernels_loaded() -> None:
    """Auto-discover and register kernel benchmark scripts."""
    if _KERNEL_BENCH_REGISTRY:
        return
    kernels_dir = Path(__file__).parent / "kernels"
    if not kernels_dir.is_dir():
        return
    for script in sorted(kernels_dir.glob("bench_*.py")):
        name = script.stem.removeprefix("bench_").replace("_", "-")
        # Read a one-line docstring if available
        desc = f"Kernel benchmark: {name}"
        try:
            first_lines = script.read_text(encoding="utf-8").split("\n", 3)
            for line in first_lines:
                stripped = line.strip().strip('"').strip("'")
                if stripped and not stripped.startswith("#") and not stripped.startswith("from"):
                    desc = stripped
                    break
        except Exception:
            pass
        register_kernel_bench(
            name,
            description=desc,
            script_path=str(script),
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_kernel_bench(
    kernel_name: str,
    *,
    scale: int | None = None,
    precision: str = "auto",
    bandwidth: bool = False,
    timeout: int = 300,
) -> BenchmarkResult:
    """Spawn a kernel benchmark as a subprocess, parse JSON, return result."""
    _ensure_kernels_loaded()

    if kernel_name not in _KERNEL_BENCH_REGISTRY:
        available = ", ".join(_KERNEL_BENCH_REGISTRY) or "(none)"
        raise KeyError(
            f"Unknown kernel benchmark: {kernel_name!r}. Available: {available}"
        )

    spec = _KERNEL_BENCH_REGISTRY[kernel_name]
    effective_scale = scale or spec.default_scale

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        output_path = Path(tmp.name)

    cmd = [
        sys.executable,
        spec.script_path,
        "--scale", str(effective_scale),
        "--precision", precision,
        "--output-json", str(output_path),
    ]
    if bandwidth:
        cmd.append("--bandwidth")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return _unavailable_result(
            kernel_name, effective_scale, precision,
            "cuda-bench not installed or kernel script not found",
        )
    except subprocess.TimeoutExpired:
        return _error_result(
            kernel_name, effective_scale, precision,
            f"Kernel benchmark timed out after {timeout}s",
        )
    finally:
        pass  # output_path cleanup happens below

    if proc.returncode != 0:
        return _error_result(
            kernel_name, effective_scale, precision,
            proc.stderr.strip() or f"exit code {proc.returncode}",
        )

    try:
        raw = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        return _error_result(kernel_name, effective_scale, precision, str(exc))
    finally:
        output_path.unlink(missing_ok=True)

    return _parse_nvbench_json(kernel_name, effective_scale, precision, raw)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_nvbench_json(
    kernel_name: str,
    scale: int,
    precision: str,
    raw: dict[str, Any],
) -> BenchmarkResult:
    """Convert NVBench JSON output to unified ``BenchmarkResult``."""
    benchmarks = raw.get("benchmarks", [])
    if not benchmarks:
        return _error_result(kernel_name, scale, precision, "no benchmarks in output")

    # Take the first benchmark's first state
    states = benchmarks[0].get("states", [])
    if not states:
        return _error_result(kernel_name, scale, precision, "no states in benchmark")

    state = states[0]
    summaries = {s.get("tag", ""): s for s in state.get("summaries", [])}

    gpu_time = _extract_summary_value(summaries, "nv/cold/time/gpu/mean", 0.0)
    cpu_time = _extract_summary_value(summaries, "nv/cold/time/cpu/mean", 0.0)

    # Bandwidth (user-defined, may not exist)
    bw_gb = _extract_summary_value(summaries, "nv/cold/bw/global/mean", None)
    bw_pct = _extract_summary_value(summaries, "nv/cold/bw/global/pct_peak", None)

    kernel_timing = KernelTimingSummary(
        gpu_time_seconds=gpu_time,
        cpu_time_seconds=cpu_time,
        bandwidth_gb_per_second=bw_gb,
        bandwidth_pct_of_peak=bw_pct,
        l2_cache_flushed=True,  # NVBench always flushes L2 for cold measurements
        throttle_detected=False,  # TODO: parse from NVBench metadata when available
        convergence_met=True,  # TODO: parse convergence status
    )

    timing = TimingSummary(
        mean_seconds=gpu_time,
        median_seconds=gpu_time,
        min_seconds=gpu_time,
        max_seconds=gpu_time,
        stddev_seconds=0.0,
        sample_count=1,
    )

    return BenchmarkResult(
        operation=kernel_name,
        tier=2,
        scale=scale,
        geometry_type="unknown",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        kernel_timing=kernel_timing,
        metadata={"nvbench_raw": raw.get("meta", {})},
    )


def _extract_summary_value(
    summaries: dict[str, dict],
    tag: str,
    default: Any,
) -> Any:
    """Extract a numeric value from NVBench summary data."""
    s = summaries.get(tag)
    if s is None:
        return default
    value = s.get("value")
    if value is None:
        return default
    return float(value)


# ---------------------------------------------------------------------------
# Error/unavailable result factories
# ---------------------------------------------------------------------------

def _unavailable_result(
    kernel_name: str, scale: int, precision: str, reason: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        operation=kernel_name,
        tier=2,
        scale=scale,
        geometry_type="unknown",
        precision=precision,
        status="skip",
        status_reason=reason,
        timing=timing_from_samples([]),
    )


def _error_result(
    kernel_name: str, scale: int, precision: str, reason: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        operation=kernel_name,
        tier=2,
        scale=scale,
        geometry_type="unknown",
        precision=precision,
        status="error",
        status_reason=reason,
        timing=timing_from_samples([]),
    )
