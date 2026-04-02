from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from vibespatial.bench.cli import main as vsbench_main


def test_vsbench_shootout_directory_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not available")

    script_dir = Path("benchmarks/shootout")
    expected_scripts = sorted(
        path for path in script_dir.glob("*.py") if not path.name.startswith("_")
    )

    exit_code = vsbench_main(
        [
            "shootout",
            str(script_dir),
            "--scale",
            "200",
            "--repeat",
            "1",
            "--no-warmup",
            "--quiet",
        ]
    )

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    statuses_by_script = {
        Path(line.split()[2]).name: line.split()[0]
        for line in lines
    }
    expected_failures = {
        "corridor_flood_priority.py",
        "parcel_zoning.py",
        "redevelopment_screening.py",
        "transit_service_gap.py",
    }

    # These canaries intentionally stay red until the underlying public-path
    # parity gaps are fixed in the library. When those fixes land, update
    # expected_failures and tighten the benchmark back to all-green.
    assert exit_code == 1
    assert len(lines) == len(expected_scripts)
    assert set(statuses_by_script) == {path.name for path in expected_scripts}
    assert {
        name for name, status in statuses_by_script.items() if status == "[ERR]"
    } == expected_failures
    assert {
        name for name, status in statuses_by_script.items() if status == "[PASS]"
    } == ({path.name for path in expected_scripts} - expected_failures)
