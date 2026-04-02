from __future__ import annotations

from pathlib import Path

from scripts.upstream_native_coverage import (
    compute_native_pass_rates,
    discover_group_targets,
    parse_pytest_summary,
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


def test_discover_group_targets_splits_upstream_tree_by_top_level_area() -> None:
    grouped = discover_group_targets(("tests/upstream/geopandas",), cwd=Path.cwd())

    assert tuple(grouped) == ("tests/upstream/geopandas",)
    assert grouped["tests/upstream/geopandas"] == (
        "tests/upstream/geopandas/tests",
        "tests/upstream/geopandas/io",
        "tests/upstream/geopandas/tools",
    )
