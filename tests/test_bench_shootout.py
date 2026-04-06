from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from vibespatial.bench.cli import main as vsbench_main
from vibespatial.bench.shootout import run_shootout
from vibespatial.runtime import has_gpu_runtime
from vibespatial.testing import strict_native_environment


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
    if has_gpu_runtime():
        expected_failures = {
            "corridor_flood_priority.py",
            "transit_service_gap.py",
        }
    else:
        expected_failures = {
            "flood_exposure.py",
            "network_service_area.py",
            "parcel_zoning.py",
            "redevelopment_screening.py",
            "site_suitability.py",
            "transit_service_gap.py",
            "vegetation_corridor.py",
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


@pytest.mark.gpu
@pytest.mark.parametrize(
    "script_name",
    [
        "network_service_area.py",
    ],
)
def test_strict_native_shootout_scripts_do_not_need_compat_env(
    script_name: str,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU required")

    script = Path("benchmarks/shootout") / script_name
    with strict_native_environment():
        result = run_shootout(
            script,
            repeat=1,
            warmup=False,
            scale="200",
            timeout=300,
            quiet=True,
        )

    assert result.status == "pass"
    assert result.vibespatial.error is None
    assert result.metadata.get("fingerprint") == "match"


@pytest.mark.gpu
def test_strict_native_transit_service_gap_matches_baseline() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU required")

    script = Path("benchmarks/shootout") / "transit_service_gap.py"
    with strict_native_environment():
        result = run_shootout(
            script,
            repeat=1,
            warmup=False,
            scale="10000",
            timeout=300,
            quiet=True,
        )

    assert result.status == "pass"
    assert result.vibespatial.error is None
    assert result.metadata.get("fingerprint") == "match"


@pytest.mark.gpu
def test_strict_native_shootout_recovers_from_nested_launcher_gpu_loss() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU required")

    code = (
        "import json; "
        "from pathlib import Path; "
        "from vibespatial.bench.shootout import run_shootout; "
        "res = run_shootout("
        "Path('benchmarks/shootout/network_service_area.py'), "
        "repeat=1, warmup=False, scale='10k', timeout=300, quiet=True, "
        "baseline_python='/tmp/gpd_bench/bin/python'"
        "); "
        "print(json.dumps({"
        "'status': res.status, "
        "'reason': res.status_reason, "
        "'fingerprint': res.metadata.get('fingerprint'), "
        "'launch': res.metadata.get('vibespatial_launch')"
        "}))"
    )
    proc = subprocess.run(
        [str(Path(".venv/bin/python")), "-c", code],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "VIBESPATIAL_STRICT_NATIVE": "1",
            "UV_CACHE_DIR": "/tmp/uv-cache",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())
    assert payload["status"] == "pass"
    assert payload["fingerprint"] == "match"
    assert payload["launch"] in {"subprocess", "in_process_retry"}
