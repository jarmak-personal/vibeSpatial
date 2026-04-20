from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vibespatial.bench.cli import main as vsbench_main
from vibespatial.bench.output import render_shootout
from vibespatial.bench.schema import timing_from_samples
from vibespatial.bench.shootout import ShootoutResult, ShootoutRun, _run_harness, run_shootout
from vibespatial.cuda.cccl_precompile import SPEC_REGISTRY
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
def test_strict_native_nearby_buildings_matches_baseline() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU required")

    script = Path("benchmarks/shootout") / "nearby_buildings.py"
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
def test_strict_native_accessibility_redevelopment_matches_baseline() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU required")

    script = Path("benchmarks/shootout") / "accessibility_redevelopment.py"
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
def test_strict_native_shootout_handles_nested_launcher_gpu_visibility() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU required")

    code = (
        "import json; "
        "from pathlib import Path; "
        "from vibespatial.bench.shootout import run_shootout; "
        "res = run_shootout("
        "Path('benchmarks/shootout/nearby_buildings.py'), "
        "repeat=1, warmup=False, scale='10k', timeout=300, quiet=True"
        "); "
        "print(json.dumps({"
        "'status': res.status, "
        "'reason': res.status_reason, "
        "'fingerprint': res.metadata.get('fingerprint'), "
        "'launch': res.metadata.get('vibespatial_launch')"
        "}))"
    )
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not available")
    proc = subprocess.run(
        [uv, "run", "python", "-c", code],
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


def test_shootout_postamble_runs_without_strict_native_env(tmp_path: Path) -> None:
    script = tmp_path / "postamble_env_probe.py"
    script.write_text(
        "\n".join(
            [
                "import os",
                "# --- timed work starts here ---",
                "timed_flag = os.environ.get('VIBESPATIAL_STRICT_NATIVE')",
                "# --- timed work ends here ---",
                "print(f'TIMED={timed_flag};POST={os.environ.get(\"VIBESPATIAL_STRICT_NATIVE\")}')",
            ]
        ),
        encoding="utf-8",
    )

    run = _run_harness(
        label="probe",
        python_cmd=[sys.executable],
        script=script,
        repeat=1,
        warmup=False,
        env={**os.environ, "VIBESPATIAL_STRICT_NATIVE": "1"},
        quiet=True,
    )

    assert run.error is None
    assert "TIMED=1;POST=None" in run.stdout


@pytest.mark.gpu
def test_vibespatial_harness_pipeline_warm_drains_deferred_cache(
    tmp_path: Path,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU required")

    script = tmp_path / "warm_probe.py"
    script.write_text(
        "import json\n"
        "import geopandas as gpd\n"
        "from vibespatial.cuda.cccl_precompile import precompile_status\n"
        "print(json.dumps(precompile_status()))\n",
        encoding="utf-8",
    )

    base_env = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"}
    cold = _run_harness(
        label="vibespatial",
        python_cmd=[sys.executable],
        script=script,
        repeat=1,
        warmup=False,
        pipeline_warm=False,
        env=base_env,
        timeout=60,
        quiet=True,
    )
    warm = _run_harness(
        label="vibespatial",
        python_cmd=[sys.executable],
        script=script,
        repeat=1,
        warmup=False,
        pipeline_warm=True,
        env=base_env,
        timeout=60,
        quiet=True,
    )

    cold_status = json.loads(cold.stdout.strip())
    warm_status = json.loads(warm.stdout.strip())
    assert cold.error is None
    assert warm.error is None
    assert cold_status["cccl"]["deferred"] > 0 or cold_status["nvrtc"]["deferred"] > 0
    assert warm_status["cccl"]["submitted"] == len(SPEC_REGISTRY)
    assert warm_status["cccl"]["deferred"] == 0
    assert warm_status["nvrtc"]["deferred"] == 0
    assert warm_status["cccl"]["pending"] == 0
    assert warm_status["nvrtc"]["pending"] == 0


def test_shootout_in_process_retry_keeps_full_precompile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_harness(**kwargs):
        if kwargs["label"] == "geopandas":
            return ShootoutRun(
                label="geopandas",
                timing=timing_from_samples([1.0]),
                stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
            )
        return ShootoutRun(
            label="vibespatial",
            timing=timing_from_samples([]),
            error="GPU execution was requested, but no GPU runtime is available",
        )

    seen: dict[str, object] = {}

    def _fake_run_harness_in_process(**kwargs):
        seen.update(kwargs)
        return ShootoutRun(
            label="vibespatial",
            timing=timing_from_samples([0.5]),
            stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
        )

    monkeypatch.setattr("vibespatial.bench.shootout._run_harness", _fake_run_harness)
    monkeypatch.setattr(
        "vibespatial.bench.shootout._run_harness_in_process",
        _fake_run_harness_in_process,
    )
    monkeypatch.setattr("vibespatial.runtime.has_gpu_runtime", lambda: True)

    result = run_shootout(
        Path("benchmarks/shootout/network_service_area.py"),
        repeat=1,
        warmup=False,
        timeout=30,
        quiet=True,
        baseline_python=sys.executable,
    )

    assert result.status == "pass"
    assert result.metadata["vibespatial_launch"] == "in_process_retry"
    assert seen["pipeline_warm"] is True


def test_run_shootout_baseline_uses_isolated_uv_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = tmp_path / "probe.py"
    script.write_text(
        "print('SHOOTOUT_FINGERPRINT: rows=1')\n",
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def _fake_run_harness(**kwargs):
        calls.append(kwargs)
        return ShootoutRun(
            label=kwargs["label"],
            timing=timing_from_samples([1.0]),
            stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
        )

    monkeypatch.setattr("vibespatial.bench.shootout._find_uv", lambda: "uv")
    monkeypatch.setattr("vibespatial.bench.shootout._run_harness", _fake_run_harness)
    monkeypatch.setattr("vibespatial.runtime.has_gpu_runtime", lambda: False)

    result = run_shootout(
        script,
        repeat=1,
        warmup=False,
        quiet=True,
    )

    assert result.status == "pass"
    baseline_cmd = calls[0]["python_cmd"]
    assert baseline_cmd[:4] == ["uv", "run", "--isolated", "--no-project"]
    assert "--with" in baseline_cmd
    assert "geopandas" in baseline_cmd
    assert "pyarrow" in baseline_cmd


def test_run_shootout_marks_cold_start_probe_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_harness(**kwargs):
        return ShootoutRun(
            label=kwargs["label"],
            timing=timing_from_samples([1.0]),
            stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
        )

    monkeypatch.setattr("vibespatial.bench.shootout._run_harness", _fake_run_harness)
    monkeypatch.setattr("vibespatial.runtime.has_gpu_runtime", lambda: False)

    result = run_shootout(
        Path("benchmarks/shootout/network_service_area.py"),
        repeat=1,
        warmup=False,
        timeout=30,
        quiet=True,
        baseline_python=sys.executable,
    )

    assert result.metadata["measurement_mode"] == "cold_start_probe"
    assert "steady-state parity" in result.metadata["measurement_note"]


def test_render_shootout_marks_cold_start_probe_mode() -> None:
    run = ShootoutRun(
        label="geopandas",
        timing=timing_from_samples([1.0]),
        stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
    )
    result = ShootoutResult(
        script="benchmarks/shootout/network_service_area.py",
        geopandas=run,
        vibespatial=ShootoutRun(
            label="vibespatial",
            timing=timing_from_samples([0.5]),
            stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
        ),
        speedup=2.0,
        status="pass",
        status_reason="ok",
        metadata={
            "measurement_mode": "cold_start_probe",
            "measurement_note": "repeat<3 with warmup disabled is cold-start sensitive",
            "fingerprint": "match",
        },
    )

    quiet = render_shootout(result, mode="quiet")
    human = render_shootout(result, mode="human")

    assert "mode=cold-start" in quiet
    assert "cold-start sensitive" in human
