from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts import profile_spatialbench_nsight as nsight


def _write_fake_trace(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER NOT NULL, end INTEGER, text TEXT, textId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER NOT NULL, end INTEGER NOT NULL, demangledName INTEGER NOT NULL
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start INTEGER NOT NULL, end INTEGER NOT NULL
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER NOT NULL, end INTEGER NOT NULL, nameId INTEGER NOT NULL
        );
        CREATE TABLE OSRT_API (
            start INTEGER NOT NULL, end INTEGER NOT NULL, nameId INTEGER NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO StringIds VALUES (?, ?)",
        [
            (1, "spatialbench.q1.measured"),
            (2, "kernel_a"),
            (3, "kernel_b"),
            (4, "cudaStreamSynchronize"),
            (5, "cuLaunchKernel"),
            (6, "read"),
        ],
    )
    connection.execute("INSERT INTO NVTX_EVENTS VALUES (0, 100, NULL, 1)")
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
        [(10, 30, 2), (20, 50, 3)],
    )
    connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (70, 80)")
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?)",
        [(5, 15, 4), (15, 17, 5)],
    )
    connection.execute("INSERT INTO OSRT_API VALUES (30, 36, 6)")
    connection.commit()
    connection.close()


def test_summarize_sqlite_reports_union_busy_time_and_sync(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "trace.sqlite"
    result_path = tmp_path / "result.json"
    _write_fake_trace(sqlite_path)
    result_path.write_text(
        json.dumps(
            {
                "query": "q1",
                "status": "success",
                "elapsed_seconds": 1.0,
                "row_count": 100,
            }
        )
    )

    summary = nsight._summarize_sqlite(sqlite_path, result_path, query="q1")
    trace = summary["trace"]

    assert trace["measured_range_ns"] == 100
    assert trace["gpu_busy_union_ns"] == 50
    assert trace["gpu_busy_fraction"] == pytest.approx(0.5)
    assert trace["gpu_idle_union_ns"] == 50
    assert trace["largest_gpu_idle_gap_ns"] == 20
    assert trace["kernel_sum_ns"] == 50
    assert trace["kernel_launch_count"] == 2
    assert trace["cuda_sync_api_ns"] == 10
    assert trace["top_kernels"][0]["name"] == "kernel_b"


def test_build_profile_command_scopes_capture_to_measured_nvtx_range(tmp_path: Path) -> None:
    command = nsight._build_profile_command(
        nsys="/opt/nsys",
        python="/venv/python",
        data_dir=tmp_path / "data",
        query="q11",
        query_dir=tmp_path / "q11",
        warmup_runs=1,
        gpu_metrics=True,
        cuda_memory_usage=False,
    )

    assert "--capture-range=nvtx" in command
    assert "--nvtx-capture=spatialbench.q11.measured" in command
    assert "--capture-range-end=stop" in command
    assert "--sample=process-tree" in command
    assert "--cpuctxsw=process-tree" in command
    assert "--python-sampling=true" in command
    assert "--cuda-memory-usage=false" in command
    assert "--gpu-metrics-devices=cuda-visible" in command
    assert command[-8:] == [
        "--data-dir",
        str(tmp_path / "data"),
        "--query",
        "q11",
        "--warmup-runs",
        "1",
        "--result",
        str(tmp_path / "q11" / "result.json"),
    ]


def _capture_summary(query: str, *, wall: float, busy: float) -> dict:
    return {
        "query": query,
        "status": "success",
        "result": {"elapsed_seconds": wall},
        "trace": {
            "gpu_busy_fraction": busy,
            "kernel_sum_ns": int(wall * 0.4 * 1e9),
            "kernel_launch_count": 10,
            "cuda_sync_api_ns": int(wall * 0.1 * 1e9),
            "largest_gpu_idle_gap_ns": 5_000_000,
        },
    }


def test_comparison_payload_and_markdown_preserve_device_metrics() -> None:
    baseline = {
        "environment": {"source": {"revision": "abc"}},
        "queries": [_capture_summary("q1", wall=2.0, busy=0.25)],
    }
    candidate = {
        "environment": {"source": {"revision": "abc"}},
        "queries": [_capture_summary("q1", wall=1.0, busy=0.75)],
    }

    payload = nsight._comparison_payload(baseline, candidate, "4090", "H200")
    q1 = payload["queries"][0]
    markdown = nsight._comparison_markdown(payload)

    assert q1["wall_speedup"] == pytest.approx(2.0)
    assert q1["baseline_gpu_busy_fraction"] == pytest.approx(0.25)
    assert q1["candidate_gpu_busy_fraction"] == pytest.approx(0.75)
    assert "| q1 | 2.000s | 1.000s | 2.00x | 25.0% | 75.0% |" in markdown
    assert "| q2 | -- | -- |" in markdown


def test_parse_queries_rejects_unknown_query() -> None:
    assert nsight._parse_queries("q1,q12") == ("q1", "q12")
    with pytest.raises(Exception, match="queries must be"):
        nsight._parse_queries("q13")
