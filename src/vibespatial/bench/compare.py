"""Unified regression detection for vsbench CLI.

Supports both v1 (existing pipeline format) and v2 (unified BenchmarkResult) JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Regression thresholds for pipeline benchmark comparisons.
WALL_CLOCK_THRESHOLD = 0.05
DEVICE_MEMORY_THRESHOLD = 0.10
KERNEL_GPU_TIME_THRESHOLD = 0.05
KERNEL_BANDWIDTH_THRESHOLD = 0.10


@dataclass(frozen=True)
class RegressionFinding:
    pipeline: str
    scale: int
    metric: str
    baseline: float | int
    current: float | int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "scale": self.scale,
            "metric": self.metric,
            "baseline": self.baseline,
            "current": self.current,
            "detail": self.detail,
        }


@dataclass
class ComparisonResult:
    baseline_path: str
    current_path: str
    findings: list[RegressionFinding] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return bool(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline_path,
            "current": self.current_path,
            "has_regressions": self.has_regressions,
            "findings": [f.to_dict() for f in self.findings],
        }


def compare(baseline_path: Path, current_path: Path) -> ComparisonResult:
    """Compare two benchmark result files and detect regressions."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    return compare_payloads(
        baseline=baseline,
        current=current,
        baseline_path=str(baseline_path),
        current_path=str(current_path),
    )


def compare_payloads(
    *,
    baseline: dict[str, Any],
    current: dict[str, Any],
    baseline_path: str = "<baseline>",
    current_path: str = "<current>",
) -> ComparisonResult:
    """Compare two already-loaded benchmark payloads."""

    result = ComparisonResult(
        baseline_path=baseline_path,
        current_path=current_path,
    )

    # Normalise: some files store raw lists or non-standard formats
    if not isinstance(current, dict) or not isinstance(baseline, dict):
        return result  # no structured comparison possible

    schema_version = current.get("schema_version", 1)

    if schema_version >= 2:
        result.findings.extend(_compare_v2(baseline, current))
    else:
        result.findings.extend(_compare_v1(baseline, current))

    return result


def compare_results(current: dict[str, Any], baseline: dict[str, Any]) -> list[RegressionFinding]:
    """Compatibility helper for tests that compare in-memory payloads."""
    return compare_payloads(baseline=baseline, current=current).findings


# ---------------------------------------------------------------------------
# v1: existing pipeline format (from suite_to_json)
# ---------------------------------------------------------------------------

def _compare_v1(baseline: dict, current: dict) -> list[RegressionFinding]:
    """Compare v1-format pipeline results (backward compat)."""
    findings: list[RegressionFinding] = []
    base_idx = _index_v1(baseline)
    cur_idx = _index_v1(current)

    for key, base in base_idx.items():
        if key not in cur_idx:
            continue
        cur = cur_idx[key]
        pipeline, scale = key

        # Wall-clock regression
        base_elapsed = float(base["elapsed_seconds"])
        cur_elapsed = float(cur["elapsed_seconds"])
        if base_elapsed > 0 and cur_elapsed > base_elapsed * (1.0 + WALL_CLOCK_THRESHOLD):
            findings.append(RegressionFinding(
                pipeline=pipeline,
                scale=scale,
                metric="wall_clock",
                baseline=base_elapsed,
                current=cur_elapsed,
                detail=f"wall-clock regression exceeds {WALL_CLOCK_THRESHOLD:.0%}",
            ))

        # Transfer count increase. Prefer runtime-layer copy accounting when
        # present; older artifacts only have the owned-array diagnostic count.
        base_transfers = int(
            base.get(
                "runtime_d2h_transfer_count",
                base.get("transfer_count", 0),
            )
        )
        cur_transfers = int(
            cur.get(
                "runtime_d2h_transfer_count",
                cur.get("transfer_count", 0),
            )
        )
        if cur_transfers > base_transfers:
            findings.append(RegressionFinding(
                pipeline=pipeline,
                scale=scale,
                metric="transfer_count",
                baseline=base_transfers,
                current=cur_transfers,
                detail="host/device transfer count increased",
            ))

        # Materialization count increase
        base_mat = int(base.get("materialization_count", 0))
        cur_mat = int(cur.get("materialization_count", 0))
        if cur_mat > base_mat:
            findings.append(RegressionFinding(
                pipeline=pipeline,
                scale=scale,
                metric="materialization_count",
                baseline=base_mat,
                current=cur_mat,
                detail="host materialization count increased",
            ))

        # Device memory regression
        base_mem = base.get("peak_device_memory_bytes")
        cur_mem = cur.get("peak_device_memory_bytes")
        if (
            isinstance(base_mem, int)
            and isinstance(cur_mem, int)
            and base_mem > 0
            and cur_mem > int(base_mem * (1.0 + DEVICE_MEMORY_THRESHOLD))
        ):
            findings.append(RegressionFinding(
                pipeline=pipeline,
                scale=scale,
                metric="peak_device_memory_bytes",
                baseline=base_mem,
                current=cur_mem,
                detail=f"device memory regression exceeds {DEVICE_MEMORY_THRESHOLD:.0%}",
            ))

    return findings


def _index_v1(payload: dict) -> dict[tuple[str, int], dict]:
    return {
        (r["pipeline"], int(r["scale"])): r
        for r in payload.get("results", [])
        if r.get("status") not in ("deferred",)
    }


# ---------------------------------------------------------------------------
# v2: unified BenchmarkResult format
# ---------------------------------------------------------------------------

def _compare_v2(baseline: dict, current: dict) -> list[RegressionFinding]:
    """Compare v2-format unified results."""
    findings: list[RegressionFinding] = []
    base_idx = _index_v2(baseline)
    cur_idx = _index_v2(current)

    for key, base in base_idx.items():
        if key not in cur_idx:
            continue
        cur = cur_idx[key]
        operation, scale = key
        tier = cur.get("tier", 1)

        # Tier 1: wall-clock, transfers
        base_timing = base.get("timing", {})
        cur_timing = cur.get("timing", {})
        base_median = float(base_timing.get("median_seconds", 0))
        cur_median = float(cur_timing.get("median_seconds", 0))
        if base_median > 0 and cur_median > base_median * (1.0 + WALL_CLOCK_THRESHOLD):
            findings.append(RegressionFinding(
                pipeline=operation,
                scale=scale,
                metric="wall_clock",
                baseline=base_median,
                current=cur_median,
                detail=f"wall-clock regression exceeds {WALL_CLOCK_THRESHOLD:.0%}",
            ))

        # Transfer regression
        base_xfer = base.get("transfers", {})
        cur_xfer = cur.get("transfers", {})
        base_total = int(base_xfer.get("d2h_count", 0)) + int(base_xfer.get("h2d_count", 0))
        cur_total = int(cur_xfer.get("d2h_count", 0)) + int(cur_xfer.get("h2d_count", 0))
        if cur_total > base_total:
            findings.append(RegressionFinding(
                pipeline=operation,
                scale=scale,
                metric="transfer_count",
                baseline=base_total,
                current=cur_total,
                detail="host/device transfer count increased",
            ))

        # Tier 2: GPU time, bandwidth
        if tier == 2:
            base_kt = base.get("kernel_timing", {})
            cur_kt = cur.get("kernel_timing", {})

            base_gpu = float(base_kt.get("gpu_time_seconds", 0))
            cur_gpu = float(cur_kt.get("gpu_time_seconds", 0))
            if base_gpu > 0 and cur_gpu > base_gpu * (1.0 + KERNEL_GPU_TIME_THRESHOLD):
                findings.append(RegressionFinding(
                    pipeline=operation,
                    scale=scale,
                    metric="kernel_gpu_time",
                    baseline=base_gpu,
                    current=cur_gpu,
                    detail=f"kernel GPU time regression exceeds {KERNEL_GPU_TIME_THRESHOLD:.0%}",
                ))

            base_bw = base_kt.get("bandwidth_gb_per_second")
            cur_bw = cur_kt.get("bandwidth_gb_per_second")
            if (
                isinstance(base_bw, (int, float))
                and isinstance(cur_bw, (int, float))
                and base_bw > 0
                and cur_bw < base_bw * (1.0 - KERNEL_BANDWIDTH_THRESHOLD)
            ):
                findings.append(RegressionFinding(
                    pipeline=operation,
                    scale=scale,
                    metric="kernel_bandwidth",
                    baseline=base_bw,
                    current=cur_bw,
                    detail=f"kernel bandwidth regression exceeds {KERNEL_BANDWIDTH_THRESHOLD:.0%}",
                ))

    return findings


def _index_v2(payload: dict) -> dict[tuple[str, int], dict]:
    return {
        (r["operation"], int(r["scale"])): r
        for r in payload.get("results", [])
        if r.get("status") not in ("skip",)
    }
