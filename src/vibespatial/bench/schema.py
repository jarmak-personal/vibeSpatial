"""Unified benchmark result schema for vsbench CLI.

Provides a single ``BenchmarkResult`` dataclass that works across both
Tier 1 (pipeline/operation) and Tier 2 (NVBench kernel) benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import orjson

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimingSummary:
    """Aggregated wall-clock timing across repeated samples."""

    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    stddev_seconds: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "stddev_seconds": self.stddev_seconds,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class TransferSummary:
    """Host/device transfer accounting for a single benchmark run."""

    d2h_count: int
    h2d_count: int
    total_bytes: int
    total_seconds: float
    offramps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "d2h_count": self.d2h_count,
            "h2d_count": self.h2d_count,
            "total_bytes": self.total_bytes,
            "total_seconds": self.total_seconds,
            "offramps": self.offramps,
        }


@dataclass(frozen=True)
class GpuUtilSummary:
    """GPU utilisation snapshot from NVML sampling."""

    device_name: str
    sm_utilization_pct_avg: float
    sm_utilization_pct_max: float
    memory_utilization_pct_avg: float
    vram_used_bytes_max: int
    vram_total_bytes: int
    sparkline: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "device_name": self.device_name,
            "sm_utilization_pct_avg": self.sm_utilization_pct_avg,
            "sm_utilization_pct_max": self.sm_utilization_pct_max,
            "memory_utilization_pct_avg": self.memory_utilization_pct_avg,
            "vram_used_bytes_max": self.vram_used_bytes_max,
            "vram_total_bytes": self.vram_total_bytes,
        }
        if self.sparkline is not None:
            d["sparkline"] = self.sparkline
        return d


@dataclass(frozen=True)
class KernelTimingSummary:
    """Tier-2 only: NVBench kernel-level timing."""

    gpu_time_seconds: float
    cpu_time_seconds: float
    bandwidth_gb_per_second: float | None
    bandwidth_pct_of_peak: float | None
    l2_cache_flushed: bool
    throttle_detected: bool
    convergence_met: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_time_seconds": self.gpu_time_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "bandwidth_gb_per_second": self.bandwidth_gb_per_second,
            "bandwidth_pct_of_peak": self.bandwidth_pct_of_peak,
            "l2_cache_flushed": self.l2_cache_flushed,
            "throttle_detected": self.throttle_detected,
            "convergence_met": self.convergence_met,
        }


# ---------------------------------------------------------------------------
# Core result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkResult:
    """Unified result for any benchmark — Tier 1 or Tier 2."""

    # Identity
    operation: str
    tier: int  # 1 = pipeline/operation, 2 = kernel microbenchmark
    scale: int
    geometry_type: str
    precision: str  # "fp32", "fp64", "auto"

    # Status
    status: str  # "pass", "fail", "error", "skip"
    status_reason: str

    # Timing
    timing: TimingSummary

    # Comparison
    baseline_name: str | None = None
    baseline_timing: TimingSummary | None = None
    speedup: float | None = None

    # Transfer tracking (Tier 1)
    transfers: TransferSummary | None = None

    # GPU utilisation (Tier 1)
    gpu_util: GpuUtilSummary | None = None

    # Kernel-level (Tier 2)
    kernel_timing: KernelTimingSummary | None = None

    # Tier gates
    tier_gate_threshold: float | None = None
    tier_gate_passed: bool | None = None

    # IO format tracking
    input_format: str = "parquet"
    read_seconds: float | None = None

    # Stages (pipeline drilldown)
    stages: tuple[dict[str, Any], ...] = ()

    # Extensible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "operation": self.operation,
            "tier": self.tier,
            "scale": self.scale,
            "geometry_type": self.geometry_type,
            "precision": self.precision,
            "status": self.status,
            "status_reason": self.status_reason,
            "timing": self.timing.to_dict(),
            "input_format": self.input_format,
        }
        if self.read_seconds is not None:
            d["read_seconds"] = self.read_seconds
        if self.baseline_name is not None:
            d["baseline_name"] = self.baseline_name
        if self.baseline_timing is not None:
            d["baseline_timing"] = self.baseline_timing.to_dict()
        if self.speedup is not None:
            d["speedup"] = self.speedup
        if self.transfers is not None:
            d["transfers"] = self.transfers.to_dict()
        if self.gpu_util is not None:
            d["gpu_util"] = self.gpu_util.to_dict()
        if self.kernel_timing is not None:
            d["kernel_timing"] = self.kernel_timing.to_dict()
        if self.tier_gate_threshold is not None:
            d["tier_gate_threshold"] = self.tier_gate_threshold
        if self.tier_gate_passed is not None:
            d["tier_gate_passed"] = self.tier_gate_passed
        if self.stages:
            d["stages"] = list(self.stages)
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_json(self) -> str:
        return orjson.dumps(
            self.to_dict(),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
        ).decode()


# ---------------------------------------------------------------------------
# Suite-level wrapper
# ---------------------------------------------------------------------------

@dataclass
class SuiteResult:
    """Aggregated results from a benchmark suite run."""

    suite_name: str
    results: list[BenchmarkResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "suite": self.suite_name,
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self) -> str:
        return orjson.dumps(
            self.to_dict(),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
        ).decode()


# ---------------------------------------------------------------------------
# Helpers for building timing summaries from sample lists
# ---------------------------------------------------------------------------

def timing_from_samples(seconds: list[float]) -> TimingSummary:
    """Build a ``TimingSummary`` from a list of elapsed-second samples."""
    n = len(seconds)
    if n == 0:
        return TimingSummary(
            mean_seconds=0.0,
            median_seconds=0.0,
            min_seconds=0.0,
            max_seconds=0.0,
            stddev_seconds=0.0,
            sample_count=0,
        )
    from statistics import mean, median, stdev

    return TimingSummary(
        mean_seconds=mean(seconds),
        median_seconds=median(seconds),
        min_seconds=min(seconds),
        max_seconds=max(seconds),
        stddev_seconds=stdev(seconds) if n >= 2 else 0.0,
        sample_count=n,
    )
