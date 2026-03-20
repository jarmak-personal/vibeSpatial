from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .upstream_native_coverage import (
        DEFAULT_TARGETS,
        NativeCoverageReport,
        parse_pytest_summary,
        run_native_coverage,
    )
except ImportError:
    from upstream_native_coverage import (
        DEFAULT_TARGETS,
        NativeCoverageReport,
        parse_pytest_summary,
        run_native_coverage,
    )
from vibespatial import STRICT_NATIVE_ENV_VAR
from vibespatial.runtime import has_gpu_runtime
from vibespatial.runtime.event_log import EVENT_LOG_ENV_VAR, read_event_records


@dataclass(frozen=True)
class DispatchObservationReport:
    command: str
    targets: tuple[str, ...]
    gpu_available: bool
    counts: dict[str, int]
    failing_tests: tuple[str, ...]
    total_dispatches: int
    gpu_dispatches: int
    cpu_dispatches: int
    fallback_dispatches: int
    returncode: int


@dataclass(frozen=True)
class GPUAccelerationCoverageReport:
    targets: tuple[str, ...]
    gpu_available: bool
    api_compat_pct: float
    api_suite_pct: float
    gpu_accel_pct: float
    total_dispatches: int
    gpu_dispatches: int
    cpu_dispatches: int
    fallback_dispatches: int
    api_compat: NativeCoverageReport
    observed_dispatch: DispatchObservationReport


def summarize_event_records(records: list[dict[str, Any]]) -> dict[str, int]:
    dispatch_records = [record for record in records if record.get("event_type") == "dispatch"]
    return {
        "total_dispatches": len(dispatch_records),
        "gpu_dispatches": sum(1 for record in dispatch_records if record.get("selected") == "gpu"),
        "cpu_dispatches": sum(1 for record in dispatch_records if record.get("selected") == "cpu"),
        "fallback_dispatches": sum(1 for record in records if record.get("event_type") == "fallback"),
    }


def compute_gpu_accel_pct(*, total_dispatches: int, gpu_dispatches: int) -> float:
    if total_dispatches == 0:
        return 0.0
    return 100.0 * gpu_dispatches / total_dispatches


def run_dispatch_observation(targets: tuple[str, ...], *, cwd: Path, timeout: int) -> DispatchObservationReport:
    gpu_available = has_gpu_runtime()
    command = ["uv", "run", "pytest", "-q", *targets]
    env = dict(os.environ)
    env[STRICT_NATIVE_ENV_VAR] = "0"
    with tempfile.TemporaryDirectory() as temp_dir:
        event_log_path = Path(temp_dir) / "dispatch-events.jsonl"
        env[EVENT_LOG_ENV_VAR] = str(event_log_path)
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        combined_output = completed.stdout + "\n" + completed.stderr
        counts, failing_tests = parse_pytest_summary(combined_output)
        summary = summarize_event_records(read_event_records(event_log_path))
    return DispatchObservationReport(
        command=" ".join(command),
        targets=targets,
        gpu_available=gpu_available,
        counts=counts,
        failing_tests=failing_tests,
        total_dispatches=summary["total_dispatches"],
        gpu_dispatches=summary["gpu_dispatches"],
        cpu_dispatches=summary["cpu_dispatches"],
        fallback_dispatches=summary["fallback_dispatches"],
        returncode=completed.returncode,
    )


def build_gpu_acceleration_report(
    api_compat: NativeCoverageReport,
    observed_dispatch: DispatchObservationReport,
) -> GPUAccelerationCoverageReport:
    return GPUAccelerationCoverageReport(
        targets=api_compat.targets,
        gpu_available=observed_dispatch.gpu_available,
        api_compat_pct=api_compat.native_pass_rate_percent,
        api_suite_pct=api_compat.suite_pass_rate_percent,
        gpu_accel_pct=compute_gpu_accel_pct(
            total_dispatches=observed_dispatch.total_dispatches,
            gpu_dispatches=observed_dispatch.gpu_dispatches,
        ),
        total_dispatches=observed_dispatch.total_dispatches,
        gpu_dispatches=observed_dispatch.gpu_dispatches,
        cpu_dispatches=observed_dispatch.cpu_dispatches,
        fallback_dispatches=observed_dispatch.fallback_dispatches,
        api_compat=api_compat,
        observed_dispatch=observed_dispatch,
    )


def run_gpu_acceleration_coverage(
    targets: tuple[str, ...],
    *,
    cwd: Path,
    timeout: int,
) -> GPUAccelerationCoverageReport:
    api_compat = run_native_coverage(targets, cwd=cwd, timeout=timeout)
    observed_dispatch = run_dispatch_observation(targets, cwd=cwd, timeout=timeout)
    return build_gpu_acceleration_report(api_compat, observed_dispatch)


def print_human_summary(report: GPUAccelerationCoverageReport) -> None:
    print("GPU Acceleration Coverage")
    print(f"- GPU available: {'yes' if report.gpu_available else 'no'}")
    print(f"- API compatibility: {report.api_compat_pct:.2f}% native, {report.api_suite_pct:.2f}% suite")
    print(
        f"- dispatches: {report.total_dispatches} total, "
        f"{report.gpu_dispatches} GPU, {report.cpu_dispatches} CPU, "
        f"{report.fallback_dispatches} fallback"
    )
    print(f"- GPU acceleration coverage: {report.gpu_accel_pct:.2f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure repo GPU acceleration coverage separately from strict-native API compatibility."
    )
    parser.add_argument("targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the human summary.")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    report = run_gpu_acceleration_coverage(tuple(args.targets), cwd=root, timeout=args.timeout)
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print_human_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
