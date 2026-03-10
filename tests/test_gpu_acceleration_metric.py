from __future__ import annotations

from pathlib import Path

from scripts.gpu_acceleration_coverage import (
    DispatchObservationReport,
    build_gpu_acceleration_report,
    summarize_event_records,
)
from scripts.upstream_native_coverage import NativeCoverageReport
from vibespatial.dispatch import record_dispatch_event
from vibespatial.event_log import EVENT_LOG_ENV_VAR, read_event_records
from vibespatial.fallbacks import record_fallback_event
from vibespatial.runtime import ExecutionMode


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
        [
            {"event_type": "dispatch", "selected": "gpu"},
            {"event_type": "dispatch", "selected": "cpu"},
            {"event_type": "dispatch", "selected": "cpu"},
            {"event_type": "fallback", "selected": "cpu"},
        ]
    )

    assert summary == {
        "total_dispatches": 3,
        "gpu_dispatches": 1,
        "cpu_dispatches": 2,
        "fallback_dispatches": 1,
    }


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
        returncode=0,
    )

    report = build_gpu_acceleration_report(api_compat, observed_dispatch)

    assert report.api_compat_pct == 61.0
    assert report.gpu_accel_pct == 10.0
    assert report.total_dispatches == 20
    assert report.gpu_dispatches == 2
    assert report.fallback_dispatches == 5
