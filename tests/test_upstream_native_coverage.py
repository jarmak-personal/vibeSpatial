from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import scripts.upstream_native_coverage as coverage
from scripts.upstream_native_coverage import (
    STRICT_NATIVE_ENV_VAR,
    compute_native_pass_rates,
    discover_group_targets,
    ensure_strict_native_process_env,
    normalize_pytest_returncode,
    parse_pytest_summary,
    pytest_command,
    pytest_worker_args,
    run_native_coverage,
)


def test_parse_pytest_summary_extracts_counts_and_failures() -> None:
    counts, failing = parse_pytest_summary(
        """
10 passed, 2 failed, 3 skipped, 1 xfailed, 1 xpassed in 0.10s
FAILED tests/upstream/geopandas/tests/test_file.py::test_one
FAILED tests/upstream/geopandas/tests/test_file.py::test_two
"""
    )

    assert counts == {"passed": 10, "failed": 2, "skipped": 3, "xfailed": 1, "xpassed": 1}
    assert failing == (
        "tests/upstream/geopandas/tests/test_file.py::test_one",
        "tests/upstream/geopandas/tests/test_file.py::test_two",
    )


def test_compute_native_pass_rates_excludes_skips_from_native_rate() -> None:
    native_rate, suite_rate = compute_native_pass_rates(
        {"passed": 8, "failed": 2, "skipped": 10, "xfailed": 0, "xpassed": 0}
    )

    assert native_rate == 80.0
    assert suite_rate == 40.0


def test_normalize_pytest_returncode_allows_all_skipped_file() -> None:
    counts = {"passed": 0, "failed": 0, "skipped": 1, "xfailed": 0, "xpassed": 0}

    assert normalize_pytest_returncode(5, counts) == 0


def test_normalize_pytest_returncode_preserves_real_failures() -> None:
    counts = {"passed": 0, "failed": 1, "skipped": 1, "xfailed": 0, "xpassed": 0}

    assert normalize_pytest_returncode(5, counts) == 5


def test_pytest_worker_args_are_env_controlled(monkeypatch) -> None:
    monkeypatch.delenv("VIBESPATIAL_GPU_COVERAGE_WORKERS", raising=False)
    assert pytest_worker_args() == []

    monkeypatch.setenv("VIBESPATIAL_GPU_COVERAGE_WORKERS", "auto")
    assert pytest_worker_args() == ["-n", "auto"]

    monkeypatch.setenv("VIBESPATIAL_GPU_COVERAGE_WORKERS", "1")
    assert pytest_worker_args() == []


def test_pytest_command_uses_current_interpreter_not_nested_uv(monkeypatch) -> None:
    monkeypatch.delenv("VIBESPATIAL_GPU_COVERAGE_WORKERS", raising=False)

    command = pytest_command(("tests/upstream/geopandas/tests/test_dissolve.py",))

    assert command == [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/upstream/geopandas/tests/test_dissolve.py",
    ]


def test_ensure_strict_native_process_env_requires_launch_env(monkeypatch) -> None:
    monkeypatch.delenv(STRICT_NATIVE_ENV_VAR, raising=False)

    with pytest.raises(SystemExit, match=f"{STRICT_NATIVE_ENV_VAR}=1 must be set"):
        ensure_strict_native_process_env()

    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    ensure_strict_native_process_env()


def test_discover_group_targets_splits_upstream_tree_by_top_level_area() -> None:
    grouped = discover_group_targets(("tests/upstream/geopandas",), cwd=Path.cwd())

    assert tuple(grouped) == (
        "tests/upstream/geopandas/tests",
        "tests/upstream/geopandas/io",
        "tests/upstream/geopandas/tools",
    )
    assert grouped["tests/upstream/geopandas/tests"] == (
        "tests/upstream/geopandas/tests",
    )
    assert grouped["tests/upstream/geopandas/io"] == ("tests/upstream/geopandas/io",)
    assert grouped["tests/upstream/geopandas/tools"] == (
        "tests/upstream/geopandas/tools",
    )


def test_discover_group_targets_splits_upstream_tree_by_test_file() -> None:
    grouped = discover_group_targets(
        ("tests/upstream/geopandas/tools/tests",),
        cwd=Path.cwd(),
        group_by="file",
    )

    assert tuple(grouped) == (
        "tests/upstream/geopandas/tools/tests/test_clip.py",
        "tests/upstream/geopandas/tools/tests/test_hilbert_curve.py",
        "tests/upstream/geopandas/tools/tests/test_random.py",
        "tests/upstream/geopandas/tools/tests/test_sjoin.py",
        "tests/upstream/geopandas/tools/tests/test_tools.py",
    )
    assert grouped["tests/upstream/geopandas/tools/tests/test_clip.py"] == (
        "tests/upstream/geopandas/tools/tests/test_clip.py",
    )
    assert grouped["tests/upstream/geopandas/tools/tests/test_tools.py"] == (
        "tests/upstream/geopandas/tools/tests/test_tools.py",
    )


def test_run_native_coverage_reports_timeout_as_failure(monkeypatch) -> None:
    def fake_run_command_capture(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["uv", "run", "pytest"],
            timeout=5,
            output=b"1 passed in 0.10s\n",
        )

    monkeypatch.setattr(coverage, "_run_command_capture", fake_run_command_capture)

    report = run_native_coverage(
        ("tests/upstream/geopandas/tests",),
        cwd=Path.cwd(),
        timeout=5,
        progress=False,
    )

    assert report.returncode == 124
    assert report.counts["passed"] == 1
    assert report.failing_tests == (
        "TIMEOUT after 5s: tests/upstream/geopandas/tests",
    )
