from __future__ import annotations

from pathlib import Path

import pytest

from scripts.gpu_acceleration_coverage import (
    DispatchObservationReport,
    build_gpu_acceleration_report,
    classify_dispatch_family,
    summarize_event_records,
)
from scripts.upstream_native_coverage import NativeCoverageReport
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.event_log import EVENT_LOG_ENV_VAR, read_event_records
from vibespatial.runtime.fallbacks import record_fallback_event


def test_event_log_captures_dispatch_and_fallback_records(tmp_path: Path, monkeypatch) -> None:
    event_log_path = tmp_path / "events.jsonl"
    monkeypatch.setenv(EVENT_LOG_ENV_VAR, str(event_log_path))

    record_dispatch_event(
        surface="geopandas.array.buffer",
        operation="buffer",
        implementation="owned_stroke_kernel",
        reason="test",
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.GPU,
    )
    record_fallback_event(
        surface="geopandas.array.offset_curve",
        reason="test fallback",
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.CPU,
    )

    records = read_event_records(event_log_path)

    assert [record["event_type"] for record in records] == ["dispatch", "fallback"]
    assert records[0]["selected"] == "gpu"
    assert records[1]["selected"] == "cpu"


def test_summarize_event_records_counts_dispatch_modes_and_fallbacks() -> None:
    summary = summarize_event_records(
        [{"event_type": "dispatch", "selected": "cpu", "surface": "geopandas.array.bounds", "operation": "bounds"}]
        * 10
        + [
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "geopandas.read_file",
                "operation": "read_file",
            },
            {
                "event_type": "dispatch",
                "selected": "cpu",
                "surface": "geopandas.geodataframe.to_parquet",
                "operation": "to_parquet",
            },
            {
                "event_type": "fallback",
                "selected": "cpu",
                "surface": "geopandas.array.intersection",
                "reason": "below crossover",
            },
        ],
    )

    assert summary["total_dispatches"] == 12
    assert summary["gpu_dispatches"] == 1
    assert summary["cpu_dispatches"] == 11
    assert summary["fallback_dispatches"] == 1
    assert summary["deferred_dispatches"] == 0
    assert summary["raw_dispatch_pct"] == pytest.approx(100.0 / 12.0)
    assert summary["value_dispatch_pct"] == pytest.approx(100.0 * 5.0 / 9.0)
    assert summary["value_dispatches"] == 2
    assert summary["value_gpu_dispatches"] == 1
    assert summary["family_breakdown"]["measurement"]["included_in_value_metric"] is False
    assert summary["family_breakdown"]["io_read"]["weight"] == 5
    assert summary["family_breakdown"]["io_write"]["weight"] == 4
    assert summary["fallback_reasons"] == {"below crossover": 1}
    assert summary["fallback_surfaces"] == {"geopandas.array.intersection": 1}


def test_classify_dispatch_family_treats_internal_io_planning_as_internal() -> None:
    assert (
        classify_dispatch_family(
            {
                "event_type": "dispatch",
                "selected": "cpu",
                "surface": "vibespatial.io.shapefile",
                "operation": "read_owned",
            }
        )
        == "internal"
    )
    assert (
        classify_dispatch_family(
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "geopandas.read_file",
                "operation": "read_file",
            }
        )
        == "io_read"
    )


def test_build_gpu_acceleration_report_combines_api_and_dispatch_metrics() -> None:
    api_compat = NativeCoverageReport(
        command="uv run pytest -q tests/upstream/geopandas",
        targets=("tests/upstream/geopandas",),
        strict_native=True,
        counts={"passed": 610, "failed": 390, "skipped": 0, "xfailed": 0, "xpassed": 0},
        native_pass_rate_percent=61.0,
        suite_pass_rate_percent=61.0,
        failing_tests=(),
        returncode=1,
    )
    observed_dispatch = DispatchObservationReport(
        command="uv run pytest -q tests/upstream/geopandas",
        targets=("tests/upstream/geopandas",),
        gpu_available=False,
        counts={"passed": 900, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0},
        failing_tests=(),
        total_dispatches=20,
        gpu_dispatches=2,
        cpu_dispatches=18,
        fallback_dispatches=5,
        deferred_dispatches=0,
        raw_dispatch_pct=10.0,
        value_dispatch_pct=40.0,
        value_dispatches=5,
        value_gpu_dispatches=2,
        weighted_dispatch_units=20,
        weighted_gpu_units=8,
        family_breakdown={
            "measurement": {
                "weight": 0,
                "included_in_value_metric": False,
                "total_dispatches": 15,
                "gpu_dispatches": 0,
                "cpu_dispatches": 15,
                "gpu_accel_pct": 0.0,
            },
            "io_read": {
                "weight": 5,
                "included_in_value_metric": True,
                "total_dispatches": 5,
                "gpu_dispatches": 2,
                "cpu_dispatches": 3,
                "gpu_accel_pct": 40.0,
            },
        },
        fallback_reasons={"below crossover": 5},
        fallback_surfaces={"geopandas.array.intersection": 5},
        returncode=0,
    )

    report = build_gpu_acceleration_report(api_compat, observed_dispatch)

    assert report.api_compat_pct == 61.0
    assert report.gpu_accel_pct == 10.0
    assert report.raw_dispatch_pct == 10.0
    assert report.value_dispatch_pct == 40.0
    assert report.total_dispatches == 20
    assert report.gpu_dispatches == 2
    assert report.fallback_dispatches == 5
    assert report.family_breakdown["io_read"]["included_in_value_metric"] is True
