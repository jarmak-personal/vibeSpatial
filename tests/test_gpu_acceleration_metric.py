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
    assert summary["family_breakdown"]["io_read"]["gpu_work_pct"] == 100.0
    assert summary["family_breakdown"]["io_read"]["total_work_units"] == 1
    assert summary["family_breakdown"]["measurement"]["total_work_units"] == 10
    assert summary["io_read_breakdown"]["by_format"]["file"]["gpu_dispatches"] == 1
    assert summary["io_read_breakdown"]["by_request_shape"]["unspecified"]["gpu_dispatches"] == 1
    assert summary["io_read_breakdown"]["by_surface"]["geopandas.read_file:read_file"]["gpu_dispatches"] == 1
    assert summary["compat_read_breakdown"]["by_format"] == {}
    assert summary["io_write_breakdown"]["by_format"]["geoparquet"]["cpu_dispatches"] == 1
    assert summary["compat_write_breakdown"]["by_format"] == {}
    assert summary["fallback_reasons"] == {"below crossover": 1}
    assert summary["fallback_surfaces"] == {"geopandas.array.intersection": 1}


def test_summarize_event_records_uses_row_detail_for_family_work_units() -> None:
    summary = summarize_event_records(
        [
            {
                "event_type": "dispatch",
                "selected": "cpu",
                "surface": "geopandas.geodataframe.dissolve",
                "operation": "dissolve",
                "detail": "rows=4, method=unary",
            },
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "geopandas.geodataframe.dissolve",
                "operation": "dissolve",
                "detail": "rows=128, method=unary",
            },
        ]
    )

    assert summary["family_breakdown"]["dissolve"]["gpu_accel_pct"] == 50.0
    assert summary["family_breakdown"]["dissolve"]["gpu_work_pct"] == pytest.approx(100.0 * 128.0 / 132.0)
    assert summary["family_breakdown"]["dissolve"]["gpu_work_units"] == 128
    assert summary["family_breakdown"]["dissolve"]["total_work_units"] == 132


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


def test_explicit_public_read_compat_overrides_are_excluded_from_value_metric() -> None:
    summary = summarize_event_records(
        [
            {
                "event_type": "dispatch",
                "selected": "cpu",
                "surface": "geopandas.read_file",
                "operation": "read_file",
                "detail": (
                    "format=geopackage request=default source=path engine=pyogrio "
                    "explicit_engine=pyogrio compat_override=1"
                ),
            },
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "geopandas.read_file",
                "operation": "read_file",
            },
        ]
    )

    assert summary["family_breakdown"]["compat_override"]["included_in_value_metric"] is False
    assert summary["family_breakdown"]["io_read"]["included_in_value_metric"] is True
    assert summary["value_dispatches"] == 1
    assert summary["value_gpu_dispatches"] == 1
    assert summary["value_dispatch_pct"] == 100.0
    assert summary["compat_read_breakdown"]["by_format"]["geopackage"]["cpu_dispatches"] == 1
    assert summary["compat_read_breakdown"]["by_request_shape"]["default"]["cpu_dispatches"] == 1


def test_explicit_public_read_native_pyogrio_stays_in_io_read_metric() -> None:
    summary = summarize_event_records(
        [
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "geopandas.read_file",
                "operation": "read_file",
                "detail": (
                    "format=geojson request=default source=path engine=pyogrio "
                    "explicit_engine=pyogrio"
                ),
            }
        ]
    )

    assert "compat_override" not in summary["family_breakdown"]
    assert summary["family_breakdown"]["io_read"]["included_in_value_metric"] is True
    assert summary["io_read_breakdown"]["by_format"]["geojson"]["gpu_dispatches"] == 1
    assert summary["compat_read_breakdown"]["by_format"] == {}


def test_summarize_event_records_breaks_io_read_down_by_format_and_request_shape() -> None:
    summary = summarize_event_records(
        [
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "geopandas.read_file",
                "operation": "read_file",
                "detail": "format=geojson request=default source=path engine=auto",
            },
            {
                "event_type": "dispatch",
                "selected": "cpu",
                "surface": "geopandas.read_file",
                "operation": "read_file",
                "detail": (
                    "format=geopackage request=bbox+columns source=path engine=pyogrio "
                    "explicit_engine=pyogrio"
                ),
            },
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "geopandas.read_parquet",
                "operation": "read_parquet",
            },
        ]
    )

    by_format = summary["io_read_breakdown"]["by_format"]
    by_request_shape = summary["io_read_breakdown"]["by_request_shape"]
    by_surface = summary["io_read_breakdown"]["by_surface"]

    assert by_format["geojson"]["gpu_dispatches"] == 1
    assert by_format["geopackage"]["cpu_dispatches"] == 1
    assert by_format["geoparquet"]["gpu_dispatches"] == 1
    assert by_request_shape["default"]["gpu_dispatches"] == 1
    assert by_request_shape["bbox+columns"]["cpu_dispatches"] == 1
    assert by_surface["geopandas.read_file:read_file"]["total_dispatches"] == 2
    assert by_surface["geopandas.read_parquet:read_parquet"]["total_dispatches"] == 1


def test_public_to_file_compatibility_writes_are_excluded_from_value_metric() -> None:
    summary = summarize_event_records(
        [
            {
                "event_type": "dispatch",
                "selected": "cpu",
                "surface": "geopandas.geodataframe.to_file",
                "operation": "to_file",
            },
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "vibespatial.io.geoparquet",
                "operation": "to_parquet",
            },
        ]
    )

    assert summary["family_breakdown"]["compat_write"]["included_in_value_metric"] is False
    assert summary["family_breakdown"]["io_write"]["included_in_value_metric"] is True
    assert summary["compat_write_breakdown"]["by_format"]["file"]["cpu_dispatches"] == 1
    assert summary["compat_write_breakdown"]["by_reason"]["unspecified"]["cpu_dispatches"] == 1
    assert summary["io_write_breakdown"]["by_format"]["geoparquet"]["gpu_dispatches"] == 1
    assert summary["value_dispatches"] == 1
    assert summary["value_gpu_dispatches"] == 1
    assert summary["value_dispatch_pct"] == 100.0


def test_public_host_to_arrow_compatibility_writes_are_excluded_from_value_metric() -> None:
    summary = summarize_event_records(
        [
            {
                "event_type": "dispatch",
                "selected": "cpu",
                "surface": "geopandas.geodataframe.to_arrow",
                "operation": "to_arrow",
                "detail": (
                    "format=geoarrow encoding=wkb row_count=2 geometry_columns=1 "
                    "owned_columns=0 device_columns=0 compatibility_writer=1 "
                    "reason=host_arrow_export"
                ),
            },
            {
                "event_type": "dispatch",
                "selected": "cpu",
                "surface": "geopandas.geodataframe.to_arrow",
                "operation": "to_arrow",
                "detail": (
                    "format=geoarrow encoding=geoarrow row_count=2 geometry_columns=1 "
                    "owned_columns=1 device_columns=1"
                ),
            },
        ]
    )

    assert classify_dispatch_family(
        {
            "event_type": "dispatch",
            "selected": "cpu",
            "surface": "geopandas.geodataframe.to_arrow",
            "operation": "to_arrow",
            "detail": "format=geoarrow compatibility_writer=1 reason=host_arrow_export",
        }
    ) == "compat_write"
    assert summary["compat_write_breakdown"]["by_format"]["geoarrow"]["cpu_dispatches"] == 1
    assert summary["compat_write_breakdown"]["by_reason"]["host_arrow_export"]["cpu_dispatches"] == 1
    assert summary["io_write_breakdown"]["by_format"]["geoarrow"]["cpu_dispatches"] == 1
    assert summary["value_dispatches"] == 1
    assert summary["value_gpu_dispatches"] == 0


def test_public_native_arrow_to_file_writes_count_as_io_write() -> None:
    summary = summarize_event_records(
        [
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "geopandas.geodataframe.to_file",
                "operation": "to_file",
                "detail": "format=geojson engine=pyogrio native_arrow_sink=1 public_device=1 rows=2",
            },
            {
                "event_type": "dispatch",
                "selected": "cpu",
                "surface": "geopandas.geodataframe.to_file",
                "operation": "to_file",
                "detail": "format=geojson engine=pyogrio compatibility_writer=1",
            },
        ]
    )

    assert classify_dispatch_family(
        {
            "event_type": "dispatch",
            "selected": "gpu",
            "surface": "geopandas.geodataframe.to_file",
            "operation": "to_file",
            "detail": "native_arrow_sink=1",
        }
    ) == "io_write"
    assert summary["family_breakdown"]["io_write"]["included_in_value_metric"] is True
    assert summary["family_breakdown"]["io_write"]["gpu_dispatches"] == 1
    assert summary["family_breakdown"]["compat_write"]["included_in_value_metric"] is False
    assert summary["io_write_breakdown"]["by_format"]["geojson"]["gpu_dispatches"] == 1
    assert summary["compat_write_breakdown"]["by_format"]["geojson"]["cpu_dispatches"] == 1
    assert summary["compat_write_breakdown"]["by_reason"]["compatibility_writer"]["cpu_dispatches"] == 1
    assert summary["value_dispatches"] == 1
    assert summary["value_gpu_dispatches"] == 1


def test_summarize_event_records_breaks_compat_writes_down_by_reason() -> None:
    summary = summarize_event_records(
        [
            {
                "event_type": "dispatch",
                "selected": "cpu",
                "surface": "geopandas.geodataframe.to_file",
                "operation": "to_file",
                "detail": (
                    "format=flatgeobuf engine=pyogrio compatibility_writer=1 "
                    "reason=public_geometry_column_is_not_device-backed"
                ),
            },
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "geopandas.geodataframe.to_file",
                "operation": "to_file",
                "detail": "format=flatgeobuf engine=pyogrio native_arrow_sink=1 public_device=1 rows=2",
            },
        ]
    )

    assert summary["io_write_breakdown"]["by_format"]["flatgeobuf"]["gpu_dispatches"] == 1
    assert summary["compat_write_breakdown"]["by_format"]["flatgeobuf"]["cpu_dispatches"] == 1
    assert (
        summary["compat_write_breakdown"]["by_reason"]["public_geometry_column_is_not_device-backed"][
            "cpu_dispatches"
        ]
        == 1
    )


def test_internal_geoparquet_and_wkb_write_events_count_as_io_write() -> None:
    summary = summarize_event_records(
        [
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "vibespatial.io.wkb",
                "operation": "encode_to_parquet",
            }
        ]
    )

    assert (
        classify_dispatch_family(
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "vibespatial.io.geoparquet",
                "operation": "to_parquet",
            }
        )
        == "io_write"
    )
    assert summary["io_write_breakdown"]["by_format"]["wkb"]["gpu_dispatches"] == 1
    assert (
        classify_dispatch_family(
            {
                "event_type": "dispatch",
                "selected": "gpu",
                "surface": "vibespatial.io.wkb",
                "operation": "encode_to_parquet",
            }
        )
        == "io_write"
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
        io_read_breakdown={
            "by_format": {
                "geojson": {
                    "total_dispatches": 5,
                    "gpu_dispatches": 2,
                    "cpu_dispatches": 3,
                    "gpu_accel_pct": 40.0,
                    "total_work_units": 5,
                    "gpu_work_units": 2,
                    "cpu_work_units": 3,
                    "gpu_work_pct": 40.0,
                }
            },
            "by_request_shape": {
                "default": {
                    "total_dispatches": 5,
                    "gpu_dispatches": 2,
                    "cpu_dispatches": 3,
                    "gpu_accel_pct": 40.0,
                    "total_work_units": 5,
                    "gpu_work_units": 2,
                    "cpu_work_units": 3,
                    "gpu_work_pct": 40.0,
                }
            },
            "by_surface": {
                "geopandas.read_file:read_file": {
                    "total_dispatches": 5,
                    "gpu_dispatches": 2,
                    "cpu_dispatches": 3,
                    "gpu_accel_pct": 40.0,
                    "total_work_units": 5,
                    "gpu_work_units": 2,
                    "cpu_work_units": 3,
                    "gpu_work_pct": 40.0,
                }
            },
        },
        compat_read_breakdown={
            "by_format": {},
            "by_request_shape": {},
            "by_surface": {},
        },
        io_write_breakdown={
            "by_format": {},
            "by_surface": {},
        },
        compat_write_breakdown={
            "by_format": {},
            "by_reason": {},
            "by_surface": {},
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
    assert report.io_read_breakdown["by_format"]["geojson"]["gpu_dispatches"] == 2
    assert report.compat_read_breakdown["by_format"] == {}
