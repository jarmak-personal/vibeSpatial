from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .upstream_native_coverage import (
        DEFAULT_TARGETS,
        NativeCoverageReport,
        parse_pytest_summary,
        pytest_worker_args,
        run_native_coverage,
    )
except ImportError:
    from upstream_native_coverage import (
        DEFAULT_TARGETS,
        NativeCoverageReport,
        parse_pytest_summary,
        pytest_worker_args,
        run_native_coverage,
    )
from vibespatial import STRICT_NATIVE_ENV_VAR
from vibespatial.runtime import has_gpu_runtime
from vibespatial.runtime.event_log import EVENT_LOG_ENV_VAR, read_event_records

FAMILY_WEIGHT_UNITS: dict[str, int] = {
    "io_read": 5,
    "io_write": 4,
    "overlay": 5,
    "constructive": 4,
    "dissolve": 5,
    "query": 4,
    "other_public": 2,
    "compat_override": 0,
    "compat_write": 0,
    "measurement": 0,
    "equality": 0,
    "normalization": 0,
    "internal": 0,
}

IO_READ_OPERATIONS = {
    "from_arrow",
    "read_file",
    "read_native",
    "read_owned",
    "read_parquet",
    "row_group_pushdown",
}
IO_WRITE_OPERATIONS = {
    "encode_to_parquet",
    "to_arrow",
    "to_file",
    "to_parquet",
}
QUERY_OPERATIONS = {
    "contains",
    "covered_by",
    "covers",
    "crosses",
    "dwithin",
    "intersects",
    "nearest",
    "overlaps",
    "query",
    "sjoin",
    "sjoin_nearest",
    "touches",
    "within",
}
MEASUREMENT_OPERATIONS = {"area", "boundary", "bounds", "length"}
EQUALITY_OPERATIONS = {"equals", "equals_exact", "equals_identical", "geom_equals", "geom_equals_exact"}
NORMALIZATION_OPERATIONS = {"normalize"}
CONSTRUCTIVE_OPERATIONS = {
    "buffer",
    "centroid",
    "clip_by_rect",
    "convex_hull",
    "difference",
    "identity",
    "intersection",
    "make_valid",
    "representative_point",
    "simplify",
    "symmetric_difference",
    "union",
}

SUPPLEMENTAL_OBSERVATION_TARGETS: tuple[tuple[str, ...], ...] = (
    ("tests/test_gpu_dissolve.py::test_public_dissolve_gpu_coverage_smoke",),
)


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
    deferred_dispatches: int
    raw_dispatch_pct: float
    value_dispatch_pct: float
    value_dispatches: int
    value_gpu_dispatches: int
    weighted_dispatch_units: int
    weighted_gpu_units: int
    family_breakdown: dict[str, dict[str, Any]]
    io_read_breakdown: dict[str, dict[str, dict[str, Any]]]
    compat_read_breakdown: dict[str, dict[str, dict[str, Any]]]
    io_write_breakdown: dict[str, dict[str, dict[str, Any]]]
    compat_write_breakdown: dict[str, dict[str, dict[str, Any]]]
    fallback_reasons: dict[str, int]
    fallback_surfaces: dict[str, int]
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
    deferred_dispatches: int
    raw_dispatch_pct: float
    value_dispatch_pct: float
    value_dispatches: int
    value_gpu_dispatches: int
    weighted_dispatch_units: int
    weighted_gpu_units: int
    family_breakdown: dict[str, dict[str, Any]]
    io_read_breakdown: dict[str, dict[str, dict[str, Any]]]
    compat_read_breakdown: dict[str, dict[str, dict[str, Any]]]
    io_write_breakdown: dict[str, dict[str, dict[str, Any]]]
    compat_write_breakdown: dict[str, dict[str, dict[str, Any]]]
    fallback_reasons: dict[str, int]
    fallback_surfaces: dict[str, int]
    api_compat: NativeCoverageReport
    observed_dispatch: DispatchObservationReport


def classify_dispatch_family(record: dict[str, Any]) -> str:
    surface = str(record.get("surface", ""))
    operation = str(record.get("operation", ""))
    detail = str(record.get("detail", "") or "")
    is_public_surface = surface.startswith("geopandas.") or surface.startswith("DeviceGeometryArray.")

    if surface == "geopandas.read_file" and "compat_override=1" in detail:
        return "compat_override"
    if (
        surface in {"geopandas.geodataframe.to_file", "geopandas.geoseries.to_file"}
        and operation in IO_WRITE_OPERATIONS
        and record.get("selected") == "gpu"
        and "native_arrow_sink=1" in detail
    ):
        return "io_write"
    if surface in {"geopandas.geodataframe.to_file", "geopandas.geoseries.to_file"}:
        return "compat_write"
    if is_public_surface and operation in IO_WRITE_OPERATIONS and "compatibility_writer=1" in detail:
        return "compat_write"
    if is_public_surface and (operation in NORMALIZATION_OPERATIONS or surface == "normalize"):
        return "normalization"
    if is_public_surface and (
        operation in EQUALITY_OPERATIONS or surface in {"geom_equals", "geom_equals_exact"}
    ):
        return "equality"
    if is_public_surface and operation in MEASUREMENT_OPERATIONS:
        return "measurement"
    if is_public_surface and (operation == "dissolve" or surface == "geopandas.geodataframe.dissolve"):
        return "dissolve"
    if is_public_surface and operation in IO_READ_OPERATIONS:
        return "io_read"
    if is_public_surface and operation in IO_WRITE_OPERATIONS:
        return "io_write"
    if is_public_surface and (operation.startswith("overlay_") or surface == "geopandas.overlay"):
        return "overlay"
    if is_public_surface and operation in QUERY_OPERATIONS:
        return "query"
    if is_public_surface and operation in CONSTRUCTIVE_OPERATIONS:
        return "constructive"
    if surface.startswith(("vibespatial.io.geoparquet", "vibespatial.io.wkb", "vibespatial.io.geoarrow")) and (
        operation in IO_WRITE_OPERATIONS or operation == "encode_to_parquet"
    ):
        return "io_write"
    if surface.startswith("geopandas.read_") or surface.startswith("vibespatial.read_"):
        return "io_read"
    if any(surface.endswith(suffix) for suffix in (".to_arrow", ".to_file", ".to_parquet", ".from_arrow")):
        return "io_write" if ".to_" in surface else "io_read"
    if surface.startswith(("geopandas.sindex.", "geopandas.tools.sjoin")):
        return "query"
    if surface.startswith(("geopandas.", "DeviceGeometryArray.")):
        return "other_public"
    return "internal"


def _extract_detail_value(detail: str, key: str) -> str | None:
    match = re.search(rf"(?:^|[,\s]){re.escape(key)}=([^\s,]+)", detail)
    if match is None:
        return None
    return match.group(1)


def classify_io_read_format(record: dict[str, Any]) -> str:
    detail = str(record.get("detail", "") or "")
    detail_format = _extract_detail_value(detail, "format")
    if detail_format:
        return detail_format
    operation = str(record.get("operation", ""))
    surface = str(record.get("surface", ""))
    if operation == "read_parquet" or surface == "geopandas.read_parquet":
        return "geoparquet"
    if operation == "from_arrow" or surface.endswith(".from_arrow"):
        return "geoarrow"
    if surface.startswith("geopandas.read_"):
        return surface.removeprefix("geopandas.read_").replace("_", "-")
    return "other"


def classify_io_read_request_shape(record: dict[str, Any]) -> str | None:
    detail = str(record.get("detail", "") or "")
    request_shape = _extract_detail_value(detail, "request")
    if request_shape:
        return request_shape
    if str(record.get("surface", "")) == "geopandas.read_file":
        return "unspecified"
    return None


def classify_io_write_format(record: dict[str, Any]) -> str:
    detail = str(record.get("detail", "") or "")
    detail_format = _extract_detail_value(detail, "format")
    if detail_format:
        return detail_format
    operation = str(record.get("operation", ""))
    surface = str(record.get("surface", ""))
    if surface.startswith("vibespatial.io.wkb"):
        return "wkb"
    if surface.startswith("vibespatial.io.geoarrow"):
        return "geoarrow"
    if surface.startswith("vibespatial.io.geoparquet"):
        return "geoparquet"
    if operation == "to_parquet" or surface.endswith(".to_parquet"):
        return "geoparquet"
    if operation == "to_arrow" or surface.endswith(".to_arrow"):
        return "geoarrow"
    if surface.endswith(".to_file"):
        return "file"
    return "other"


def classify_compat_write_reason(record: dict[str, Any]) -> str:
    detail = str(record.get("detail", "") or "")
    reason = _extract_detail_value(detail, "reason")
    if reason:
        return reason
    if "compatibility_writer=1" in detail:
        return "compatibility_writer"
    return "unspecified"


def _accumulate_breakdown_count(
    counters: dict[str, Counter[str]],
    *,
    key: str,
    record: dict[str, Any],
    work_units: int,
) -> None:
    counts = counters.setdefault(key, Counter())
    counts["total_dispatches"] += 1
    counts["total_work_units"] += work_units
    if record.get("selected") == "gpu":
        counts["gpu_dispatches"] += 1
        counts["gpu_work_units"] += work_units
    elif record.get("selected") == "cpu":
        counts["cpu_dispatches"] += 1
        counts["cpu_work_units"] += work_units


def _summarize_breakdown_counts(counters: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    breakdown: dict[str, dict[str, Any]] = {}
    for key in sorted(counters):
        counts = counters[key]
        breakdown[key] = {
            "total_dispatches": counts["total_dispatches"],
            "gpu_dispatches": counts["gpu_dispatches"],
            "cpu_dispatches": counts["cpu_dispatches"],
            "gpu_accel_pct": compute_gpu_accel_pct(
                total_dispatches=counts["total_dispatches"],
                gpu_dispatches=counts["gpu_dispatches"],
            ),
            "total_work_units": counts["total_work_units"],
            "gpu_work_units": counts["gpu_work_units"],
            "cpu_work_units": counts["cpu_work_units"],
            "gpu_work_pct": compute_gpu_accel_pct(
                total_dispatches=counts["total_work_units"],
                gpu_dispatches=counts["gpu_work_units"],
            ),
        }
    return breakdown


def summarize_event_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    all_dispatch_records = [record for record in records if record.get("event_type") == "dispatch"]
    dispatch_records = [
        record for record in all_dispatch_records if str(record.get("selected")) in {"cpu", "gpu"}
    ]
    fallback_records = [record for record in records if record.get("event_type") == "fallback"]
    total_dispatches = len(dispatch_records)
    gpu_dispatches = sum(1 for record in dispatch_records if record.get("selected") == "gpu")
    cpu_dispatches = sum(1 for record in dispatch_records if record.get("selected") == "cpu")

    family_counts: dict[str, Counter[str]] = {}
    io_read_format_counts: dict[str, Counter[str]] = {}
    io_read_request_counts: dict[str, Counter[str]] = {}
    io_read_surface_counts: dict[str, Counter[str]] = {}
    compat_read_format_counts: dict[str, Counter[str]] = {}
    compat_read_request_counts: dict[str, Counter[str]] = {}
    compat_read_surface_counts: dict[str, Counter[str]] = {}
    io_write_format_counts: dict[str, Counter[str]] = {}
    io_write_surface_counts: dict[str, Counter[str]] = {}
    compat_write_format_counts: dict[str, Counter[str]] = {}
    compat_write_reason_counts: dict[str, Counter[str]] = {}
    compat_write_surface_counts: dict[str, Counter[str]] = {}
    for record in dispatch_records:
        family = classify_dispatch_family(record)
        counts = family_counts.setdefault(family, Counter())
        work_units = _extract_work_units(record)
        counts["total_dispatches"] += 1
        counts["total_work_units"] += work_units
        if record.get("selected") == "gpu":
            counts["gpu_dispatches"] += 1
            counts["gpu_work_units"] += work_units
        elif record.get("selected") == "cpu":
            counts["cpu_dispatches"] += 1
            counts["cpu_work_units"] += work_units
        if family == "io_read":
            _accumulate_breakdown_count(
                io_read_format_counts,
                key=classify_io_read_format(record),
                record=record,
                work_units=work_units,
            )
            request_shape = classify_io_read_request_shape(record)
            if request_shape is not None:
                _accumulate_breakdown_count(
                    io_read_request_counts,
                    key=request_shape,
                    record=record,
                    work_units=work_units,
                )
            _accumulate_breakdown_count(
                io_read_surface_counts,
                key=f"{record.get('surface', '')}:{record.get('operation', '')}",
                record=record,
                work_units=work_units,
            )
        if family == "compat_override" and str(record.get("surface", "")) == "geopandas.read_file":
            _accumulate_breakdown_count(
                compat_read_format_counts,
                key=classify_io_read_format(record),
                record=record,
                work_units=work_units,
            )
            request_shape = classify_io_read_request_shape(record)
            if request_shape is not None:
                _accumulate_breakdown_count(
                    compat_read_request_counts,
                    key=request_shape,
                    record=record,
                    work_units=work_units,
                )
            _accumulate_breakdown_count(
                compat_read_surface_counts,
                key=f"{record.get('surface', '')}:{record.get('operation', '')}",
                record=record,
                work_units=work_units,
            )
        if family == "io_write":
            _accumulate_breakdown_count(
                io_write_format_counts,
                key=classify_io_write_format(record),
                record=record,
                work_units=work_units,
            )
            _accumulate_breakdown_count(
                io_write_surface_counts,
                key=f"{record.get('surface', '')}:{record.get('operation', '')}",
                record=record,
                work_units=work_units,
            )
        if family == "compat_write":
            _accumulate_breakdown_count(
                compat_write_format_counts,
                key=classify_io_write_format(record),
                record=record,
                work_units=work_units,
            )
            _accumulate_breakdown_count(
                compat_write_reason_counts,
                key=classify_compat_write_reason(record),
                record=record,
                work_units=work_units,
            )
            _accumulate_breakdown_count(
                compat_write_surface_counts,
                key=f"{record.get('surface', '')}:{record.get('operation', '')}",
                record=record,
                work_units=work_units,
            )

    weighted_dispatch_units = 0
    weighted_gpu_units = 0
    value_dispatches = 0
    value_gpu_dispatches = 0
    family_breakdown: dict[str, dict[str, Any]] = {}
    for family in sorted(family_counts):
        counts = family_counts[family]
        total = counts["total_dispatches"]
        gpu = counts["gpu_dispatches"]
        weight = FAMILY_WEIGHT_UNITS.get(family, 0)
        included = weight > 0
        if included:
            value_dispatches += total
            value_gpu_dispatches += gpu
            weighted_dispatch_units += total * weight
            weighted_gpu_units += gpu * weight
        family_breakdown[family] = {
            "weight": weight,
            "included_in_value_metric": included,
            "total_dispatches": total,
            "gpu_dispatches": gpu,
            "cpu_dispatches": counts["cpu_dispatches"],
            "gpu_accel_pct": compute_gpu_accel_pct(total_dispatches=total, gpu_dispatches=gpu),
            "total_work_units": counts["total_work_units"],
            "gpu_work_units": counts["gpu_work_units"],
            "cpu_work_units": counts["cpu_work_units"],
            "gpu_work_pct": compute_gpu_accel_pct(
                total_dispatches=counts["total_work_units"],
                gpu_dispatches=counts["gpu_work_units"],
            ),
        }

    fallback_reasons = Counter(str(record.get("reason", "")) for record in fallback_records)
    fallback_surfaces = Counter(str(record.get("surface", "")) for record in fallback_records)
    return {
        "total_dispatches": total_dispatches,
        "gpu_dispatches": gpu_dispatches,
        "cpu_dispatches": cpu_dispatches,
        "fallback_dispatches": len(fallback_records),
        "deferred_dispatches": len(all_dispatch_records) - len(dispatch_records),
        "raw_dispatch_pct": compute_gpu_accel_pct(total_dispatches=total_dispatches, gpu_dispatches=gpu_dispatches),
        "value_dispatch_pct": compute_gpu_accel_pct(
            total_dispatches=weighted_dispatch_units,
            gpu_dispatches=weighted_gpu_units,
        ),
        "value_dispatches": value_dispatches,
        "value_gpu_dispatches": value_gpu_dispatches,
        "weighted_dispatch_units": weighted_dispatch_units,
        "weighted_gpu_units": weighted_gpu_units,
        "family_breakdown": family_breakdown,
        "io_read_breakdown": {
            "by_format": _summarize_breakdown_counts(io_read_format_counts),
            "by_request_shape": _summarize_breakdown_counts(io_read_request_counts),
            "by_surface": _summarize_breakdown_counts(io_read_surface_counts),
        },
        "compat_read_breakdown": {
            "by_format": _summarize_breakdown_counts(compat_read_format_counts),
            "by_request_shape": _summarize_breakdown_counts(compat_read_request_counts),
            "by_surface": _summarize_breakdown_counts(compat_read_surface_counts),
        },
        "io_write_breakdown": {
            "by_format": _summarize_breakdown_counts(io_write_format_counts),
            "by_surface": _summarize_breakdown_counts(io_write_surface_counts),
        },
        "compat_write_breakdown": {
            "by_format": _summarize_breakdown_counts(compat_write_format_counts),
            "by_reason": _summarize_breakdown_counts(compat_write_reason_counts),
            "by_surface": _summarize_breakdown_counts(compat_write_surface_counts),
        },
        "fallback_reasons": dict(fallback_reasons.most_common()),
        "fallback_surfaces": dict(fallback_surfaces.most_common()),
    }


def compute_gpu_accel_pct(*, total_dispatches: int, gpu_dispatches: int) -> float:
    if total_dispatches == 0:
        return 0.0
    return 100.0 * gpu_dispatches / total_dispatches


def _extract_work_units(record: dict[str, Any]) -> int:
    detail = str(record.get("detail", "") or "")
    for key in ("rows", "total_geoms"):
        match = re.search(rf"(?:^|[,\s]){key}=(\d+)", detail)
        if match is not None:
            return max(int(match.group(1)), 1)
    return 1


def _accumulate_pytest_counts(total: dict[str, int], counts: dict[str, int]) -> None:
    for key in total:
        total[key] += counts.get(key, 0)


def run_dispatch_observation(targets: tuple[str, ...], *, cwd: Path, timeout: int) -> DispatchObservationReport:
    gpu_available = has_gpu_runtime()
    env = dict(os.environ)
    env[STRICT_NATIVE_ENV_VAR] = "0"
    commands: list[str] = []
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    failing_tests: list[str] = []
    returncode = 0
    target_sets = [targets]
    if gpu_available:
        target_sets.extend(
            supplemental
            for supplemental in SUPPLEMENTAL_OBSERVATION_TARGETS
            if supplemental != targets
        )
    with tempfile.TemporaryDirectory() as temp_dir:
        event_log_path = Path(temp_dir) / "dispatch-events.jsonl"
        env[EVENT_LOG_ENV_VAR] = str(event_log_path)
        for target_set in target_sets:
            command = ["uv", "run", "pytest", "-q", *pytest_worker_args(), *target_set]
            commands.append(" ".join(command))
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
            run_counts, run_failing_tests = parse_pytest_summary(combined_output)
            _accumulate_pytest_counts(counts, run_counts)
            failing_tests.extend(run_failing_tests)
            if completed.returncode != 0:
                returncode = completed.returncode
        summary = summarize_event_records(read_event_records(event_log_path))
    return DispatchObservationReport(
        command=" && ".join(commands),
        targets=targets,
        gpu_available=gpu_available,
        counts=counts,
        failing_tests=tuple(failing_tests),
        total_dispatches=summary["total_dispatches"],
        gpu_dispatches=summary["gpu_dispatches"],
        cpu_dispatches=summary["cpu_dispatches"],
        fallback_dispatches=summary["fallback_dispatches"],
        deferred_dispatches=summary["deferred_dispatches"],
        raw_dispatch_pct=summary["raw_dispatch_pct"],
        value_dispatch_pct=summary["value_dispatch_pct"],
        value_dispatches=summary["value_dispatches"],
        value_gpu_dispatches=summary["value_gpu_dispatches"],
        weighted_dispatch_units=summary["weighted_dispatch_units"],
        weighted_gpu_units=summary["weighted_gpu_units"],
        family_breakdown=summary["family_breakdown"],
        io_read_breakdown=summary["io_read_breakdown"],
        compat_read_breakdown=summary["compat_read_breakdown"],
        io_write_breakdown=summary["io_write_breakdown"],
        compat_write_breakdown=summary["compat_write_breakdown"],
        fallback_reasons=summary["fallback_reasons"],
        fallback_surfaces=summary["fallback_surfaces"],
        returncode=returncode,
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
        gpu_accel_pct=observed_dispatch.raw_dispatch_pct,
        total_dispatches=observed_dispatch.total_dispatches,
        gpu_dispatches=observed_dispatch.gpu_dispatches,
        cpu_dispatches=observed_dispatch.cpu_dispatches,
        fallback_dispatches=observed_dispatch.fallback_dispatches,
        deferred_dispatches=observed_dispatch.deferred_dispatches,
        raw_dispatch_pct=observed_dispatch.raw_dispatch_pct,
        value_dispatch_pct=observed_dispatch.value_dispatch_pct,
        value_dispatches=observed_dispatch.value_dispatches,
        value_gpu_dispatches=observed_dispatch.value_gpu_dispatches,
        weighted_dispatch_units=observed_dispatch.weighted_dispatch_units,
        weighted_gpu_units=observed_dispatch.weighted_gpu_units,
        family_breakdown=observed_dispatch.family_breakdown,
        io_read_breakdown=observed_dispatch.io_read_breakdown,
        compat_read_breakdown=observed_dispatch.compat_read_breakdown,
        io_write_breakdown=observed_dispatch.io_write_breakdown,
        compat_write_breakdown=observed_dispatch.compat_write_breakdown,
        fallback_reasons=observed_dispatch.fallback_reasons,
        fallback_surfaces=observed_dispatch.fallback_surfaces,
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
    if report.deferred_dispatches:
        print(f"- deferred dispatch records: {report.deferred_dispatches}")
    print(
        f"- value-weighted public ops: {report.value_dispatch_pct:.2f}% "
        f"({report.value_gpu_dispatches} GPU / {report.value_dispatches} tracked dispatches)"
    )
    print(f"- raw dispatch breadth: {report.raw_dispatch_pct:.2f}%")
    io_read_formats = report.io_read_breakdown.get("by_format", {})
    if io_read_formats:
        print("- io_read by format:")
        for format_name, details in io_read_formats.items():
            print(
                f"  {format_name:<16} {details['gpu_accel_pct']:.2f}% "
                f"({details['gpu_dispatches']}/{details['total_dispatches']} dispatches)"
            )
    compat_read_formats = report.compat_read_breakdown.get("by_format", {})
    if compat_read_formats:
        print("- explicit compat read overrides:")
        for format_name, details in compat_read_formats.items():
            print(
                f"  {format_name:<16} {details['gpu_accel_pct']:.2f}% "
                f"({details['gpu_dispatches']}/{details['total_dispatches']} dispatches)"
            )
    io_write_formats = report.io_write_breakdown.get("by_format", {})
    if io_write_formats:
        print("- io_write by format:")
        for format_name, details in io_write_formats.items():
            print(
                f"  {format_name:<16} {details['gpu_accel_pct']:.2f}% "
                f"({details['gpu_dispatches']}/{details['total_dispatches']} dispatches)"
            )
    compat_write_formats = report.compat_write_breakdown.get("by_format", {})
    if compat_write_formats:
        print("- explicit compat writes by format:")
        for format_name, details in compat_write_formats.items():
            print(
                f"  {format_name:<16} {details['gpu_accel_pct']:.2f}% "
                f"({details['gpu_dispatches']}/{details['total_dispatches']} dispatches)"
            )
    compat_write_reasons = report.compat_write_breakdown.get("by_reason", {})
    if compat_write_reasons:
        print("- explicit compat writes by reason:")
        for reason, details in compat_write_reasons.items():
            print(
                f"  {reason:<48} {details['total_dispatches']} dispatches"
            )


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
