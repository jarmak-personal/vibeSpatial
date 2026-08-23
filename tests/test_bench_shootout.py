from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vibespatial.bench.cli import main as vsbench_main
from vibespatial.bench.output import render_shootout
from vibespatial.bench.schema import timing_from_samples
from vibespatial.bench.shootout import (
    ShootoutResult,
    ShootoutRun,
    _baseline_environment_sha256,
    _baseline_host_identity,
    _measurement_identity,
    _run_harness,
    load_reusable_geopandas_baseline,
    run_shootout,
    shootout_workload_identity,
)
from vibespatial.cuda.cccl_precompile import SPEC_REGISTRY
from vibespatial.runtime import has_gpu_runtime
from vibespatial.testing import strict_native_environment


def test_site_fixture_subset_matches_full_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial as gpd
    from benchmarks.shootout._data import (
        setup_fixtures,
        setup_site_suitability_fixtures,
    )
    from geopandas.testing import assert_geodataframe_equal

    monkeypatch.setenv("VSBENCH_SCALE", "64")
    full_dir = tmp_path / "full"
    subset_dir = tmp_path / "subset"
    full_dir.mkdir()
    subset_dir.mkdir()
    full = setup_fixtures(full_dir)
    subset = setup_site_suitability_fixtures(subset_dir)

    for name in ("parcels", "exclusion_zones"):
        assert_geodataframe_equal(
            gpd.read_parquet(full[name]),
            gpd.read_parquet(subset[name]),
        )
    assert_geodataframe_equal(
        gpd.read_file(full["transit"]),
        gpd.read_file(subset["transit"]),
    )


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
    gpu_visible = has_gpu_runtime()
    expected_scripts_by_name = {path.name for path in expected_scripts}
    failures = {
        name for name, status in statuses_by_script.items() if status == "[ERR]"
    }

    assert len(lines) == len(expected_scripts)
    assert set(statuses_by_script) == expected_scripts_by_name
    if gpu_visible:
        assert exit_code == 0
        assert failures == set()
    else:
        # Without a visible GPU this smoke test validates CLI shape. Exact
        # cold-start failures are environment-sensitive because each script
        # decides whether it can retry through an in-process GPU-visible path.
        assert exit_code == 1
        assert "transit_service_gap.py" in failures
        assert failures <= expected_scripts_by_name
    assert {
        name for name, status in statuses_by_script.items() if status == "[PASS]"
    } == (expected_scripts_by_name - failures)


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
def test_transit_semijoin_unique_preserves_clipped_left_index_labels() -> None:
    """The transit anti-semijoin must compare public labels, not row positions."""
    if not has_gpu_runtime():
        pytest.skip("GPU required")

    from shapely.geometry import box

    import geopandas as gpd

    source = gpd.GeoDataFrame(
        {"building_id": [0, 1, 2]},
        geometry=[box(index, 0.0, index + 0.5, 0.5) for index in range(3)],
        index=[3000, 3010, 3020],
    )
    admin = gpd.GeoDataFrame(
        {"admin_id": [0]},
        geometry=[box(-1.0, -1.0, 4.0, 2.0)],
    )
    transit = gpd.GeoDataFrame(
        {"station_id": [0, 1]},
        geometry=[box(-1.0, -1.0, 4.0, 2.0)] * 2,
        index=[900, 901],
    )

    with strict_native_environment():
        clipped = gpd.clip(source, admin)
        served = gpd.sjoin(clipped, transit, predicate="intersects")
        served_labels = served.index.unique()
        unserved = clipped.loc[~clipped.index.isin(served_labels)]

    assert served.index.tolist() == [3000, 3000, 3010, 3010, 3020, 3020]
    assert served_labels.tolist() == [3000, 3010, 3020]
    assert unserved.empty


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


def test_shootout_timeout_reaps_harness_descendants(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    script = tmp_path / "timeout_descendant.py"
    script.write_text(
        "\n".join(
            [
                "import os",
                "import subprocess",
                "import sys",
                "import time",
                "from pathlib import Path",
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                "Path(os.environ['VSBENCH_CHILD_PID_PATH']).write_text(str(child.pid))",
                "# --- timed work starts here ---",
                "time.sleep(60)",
                "# --- timed work ends here ---",
                "print('SHOOTOUT_FINGERPRINT: rows=1')",
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
        env={**os.environ, "VSBENCH_CHILD_PID_PATH": str(child_pid_path)},
        timeout=2,
        quiet=True,
    )

    assert run.error is not None
    assert "timeout" in run.error
    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()


def test_vibespatial_shootout_profile_reports_physical_plan_evidence(tmp_path: Path) -> None:
    script = tmp_path / "profile_probe.py"
    script.write_text(
        "\n".join(
            [
                "# --- timed work starts here ---",
                "from vibespatial.runtime.execution_trace import notify_dispatch, notify_transfer",
                "from vibespatial.runtime.fallbacks import record_fallback_event",
                "from vibespatial.runtime.hotpath_trace import hotpath_stage",
                "from vibespatial.runtime.materialization import MaterializationBoundary, record_materialization_event",
                "notify_dispatch(surface='probe', operation='gpu_step', selected='gpu', implementation='probe_gpu')",
                "notify_transfer(direction='d2h', trigger='probe', reason='profile probe')",
                "with hotpath_stage('shootout.profile_probe', category='refine'):",
                "    value = sum(range(8))",
                "record_fallback_event(surface='probe.fallback', reason='profile fallback')",
                "record_materialization_event(",
                "    surface='probe.materialize',",
                "    boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,",
                "    operation='probe_materialize',",
                "    reason='profile materialization',",
                "    d2h_transfer=True,",
                ")",
                "print(f'SHOOTOUT_FINGERPRINT: rows={value // 28}')",
                "# --- timed work ends here ---",
            ]
        ),
        encoding="utf-8",
    )

    run = _run_harness(
        label="vibespatial",
        python_cmd=[sys.executable],
        script=script,
        repeat=1,
        warmup=False,
        pipeline_warm=True,
        env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"},
        timeout=60,
        quiet=True,
        profile=True,
    )

    assert run.error is None
    assert run.profile is not None
    assert run.profile["available"] is True
    assert run.profile["fallback_event_count"] == 1
    assert run.profile["trace_summary"]["gpu_steps"] >= 1
    assert run.profile["trace_summary"]["d2h_transfers"] == 1
    assert run.profile["trace_summary"]["offramps"] >= 1
    assert run.profile["trace_transfers"][0]["reason"] == "profile probe"
    assert run.profile["trace_transfers"][0]["source"] == "semantic"
    assert run.profile["trace_transfers"][0]["bytes_transferred"] == 0
    assert run.profile["trace_transfers"][0]["elapsed_seconds"] == 0.0
    assert run.profile["runtime_d2h_transfer_count"] == 0
    assert run.profile["runtime_d2h_transfer_bytes"] == 0
    assert run.profile["runtime_d2h_transfer_seconds"] == 0.0
    assert run.profile["owned_transfer_count"] == 0
    assert run.profile["materialization_count"] == 1
    assert run.profile["stage_materialization_count"] == 1
    assert run.profile["stage_materialization_d2h_event_count"] == 1
    assert run.profile["stage_materialization_counts_by_boundary"][
        "internal-host-conversion"
    ] == 1
    assert run.profile["stage_materialization_counts_by_surface"][
        "probe.materialize"
    ] == 1
    assert run.profile["fallback_events"][0]["surface"] == "probe.fallback"
    assert run.profile["top_hotpath"][0]["name"] == "shootout.profile_probe"
    assert run.profile["hotpath_total_seconds"] > 0
    assert run.profile["composition_overhead_seconds"] >= 0
    assert run.profile["composition_overhead_ratio"] is not None
    assert run.profile["timed_stages"]
    assert run.profile["stage_total_seconds"] > 0
    assert run.profile["stage_totals_by_tag"]["hotpath"] > 0
    assert run.profile["stage_totals_by_backend"]["gpu"] > 0
    assert all(
        stage.get("runtime_d2h_transfer_seconds_delta", 0.0) == 0.0
        for stage in run.profile["timed_stages"]
    )
    assert any(
        stage["fallback_event_count"] == 1
        for stage in run.profile["timed_stages"]
    )
    materialization_stage = next(
        stage
        for stage in run.profile["timed_stages"]
        if stage["materialization_event_count"] == 1
    )
    event = materialization_stage["materialization_events"][0]
    assert materialization_stage["materialization_d2h_event_count"] == 1
    assert event["pipeline"] == "shootout"
    assert event["dataset"] == "profile_probe.py"
    assert event["stage"] == f"statement_{materialization_stage['index']}"
    assert event["stage_category"] == ",".join(materialization_stage["tags"])
    assert event["surface"] == "probe.materialize"


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


def test_run_shootout_reuses_geopandas_baseline_without_launching_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []

    def _fake_run_harness(**kwargs):
        labels.append(kwargs["label"])
        return ShootoutRun(
            label="vibespatial",
            timing=timing_from_samples([0.5]),
            stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
        )

    monkeypatch.setattr("vibespatial.bench.shootout._run_harness", _fake_run_harness)
    monkeypatch.setattr("vibespatial.runtime.has_gpu_runtime", lambda: False)
    baseline = ShootoutRun(
        label="geopandas",
        timing=timing_from_samples([1.0]),
        stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
        environment={
            "python_version": "test",
            "python_implementation": "cpython",
            "packages": {
                "geopandas": "1",
                "numpy": "1",
                "pandas": "1",
                "pyarrow": "1",
                "pyogrio": "1",
                "shapely": "1",
            },
        },
    )

    result = run_shootout(
        Path("benchmarks/shootout/network_service_area.py"),
        repeat=1,
        warmup=False,
        quiet=True,
        geopandas_baseline=baseline,
        geopandas_baseline_source="baseline.json",
    )

    assert labels == ["vibespatial"]
    assert result.status == "pass"
    assert result.speedup == 2.0
    assert result.metadata["geopandas_baseline"] == "reused"
    assert result.metadata["geopandas_baseline_source"] == "baseline.json"


def test_run_shootout_reused_baseline_requires_current_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_harness(**kwargs):
        return ShootoutRun(
            label="vibespatial",
            timing=timing_from_samples([0.5]),
            stdout="",
        )

    monkeypatch.setattr("vibespatial.bench.shootout._run_harness", _fake_run_harness)
    monkeypatch.setattr("vibespatial.runtime.has_gpu_runtime", lambda: False)
    baseline = ShootoutRun(
        label="geopandas",
        timing=timing_from_samples([1.0]),
        stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
        environment={
            "python_version": "test",
            "python_implementation": "cpython",
            "packages": {
                "geopandas": "1",
                "numpy": "1",
                "pandas": "1",
                "pyarrow": "1",
                "pyogrio": "1",
                "shapely": "1",
            },
        },
    )

    result = run_shootout(
        Path("benchmarks/shootout/network_service_area.py"),
        repeat=1,
        warmup=False,
        quiet=True,
        geopandas_baseline=baseline,
        geopandas_baseline_source="baseline.json",
    )

    assert result.status == "error"
    assert result.speedup is None
    assert result.metadata["fingerprint"] == "missing"
    assert "current vibeSpatial run has no correctness fingerprint" in (
        result.status_reason
    )


def test_load_reusable_geopandas_baseline_validates_workload_identity(
    tmp_path: Path,
) -> None:
    script = tmp_path / "workload.py"
    script.write_text(
        "print('SHOOTOUT_FINGERPRINT: rows=1')\n",
        encoding="utf-8",
    )
    identity = shootout_workload_identity(script)
    baseline_environment = {
        "python_version": "test",
        "python_implementation": "cpython",
        "packages": {
            "geopandas": "1",
            "numpy": "1",
            "pandas": "1",
            "pyarrow": "1",
            "pyogrio": "1",
            "shapely": "1",
        },
    }
    artifact = tmp_path / "baseline.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "type": "shootout",
                "script": str(script),
                "geopandas": {
                    "label": "geopandas",
                    "timing": timing_from_samples([1.0, 1.0, 1.0]).to_dict(),
                    "stdout": "SHOOTOUT_FINGERPRINT: rows=1\n",
                    "environment": baseline_environment,
                },
                "vibespatial": {
                    "label": "vibespatial",
                    "timing": timing_from_samples([0.5]).to_dict(),
                },
                "metadata": {
                    **identity,
                    "scale": "1m",
                    "geopandas_baseline_host": _baseline_host_identity(),
                    "geopandas_baseline_environment_sha256": (
                        _baseline_environment_sha256(baseline_environment)
                    ),
                    "measurement_sha256": _measurement_identity(),
                    "repeat": 3,
                    "warmup": True,
                    "timeout": 300,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_reusable_geopandas_baseline(
        artifact,
        script=script,
        scale="1M",
        repeat=3,
        warmup=True,
        timeout=300,
    )
    assert loaded.timing.median_seconds == 1.0

    tampered_payload = json.loads(artifact.read_text(encoding="utf-8"))
    tampered_payload["geopandas"]["environment"]["python_version"] = "2.7"
    artifact.write_text(json.dumps(tampered_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="environment identity is stale"):
        load_reusable_geopandas_baseline(
            artifact,
            script=script,
            scale="1m",
            repeat=3,
            warmup=True,
            timeout=300,
        )

    tampered_payload["geopandas"]["environment"] = baseline_environment
    artifact.write_text(json.dumps(tampered_payload), encoding="utf-8")

    script.write_text(
        "print('SHOOTOUT_FINGERPRINT: rows=2')\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="workload hash is stale"):
        load_reusable_geopandas_baseline(
            artifact,
            script=script,
            scale="1m",
            repeat=3,
            warmup=True,
            timeout=300,
        )


def test_shootout_cli_reuses_geopandas_with_no_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "workload.py"
    script.write_text(
        "print('SHOOTOUT_FINGERPRINT: rows=1')\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("{}", encoding="utf-8")
    baseline = ShootoutRun(
        label="geopandas",
        timing=timing_from_samples([1.0]),
        stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
    )
    seen: dict[str, object] = {}

    def _load_baseline(path, **kwargs):
        seen["path"] = path
        seen.update(kwargs)
        return baseline

    def _run_shootout(script_path, **kwargs):
        seen["run_script"] = script_path
        seen["run_warmup"] = kwargs["warmup"]
        return ShootoutResult(
            script=str(script_path),
            geopandas=baseline,
            vibespatial=ShootoutRun(
                label="vibespatial",
                timing=timing_from_samples([0.5]),
                stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
            ),
            speedup=2.0,
            status="pass",
            status_reason="ok",
        )

    monkeypatch.setattr(
        "vibespatial.bench.shootout.load_reusable_geopandas_baseline",
        _load_baseline,
    )
    monkeypatch.setattr("vibespatial.bench.shootout.run_shootout", _run_shootout)

    status = vsbench_main(
        [
            "shootout",
            str(script),
            "--repeat",
            "1",
            "--no-warmup",
            "--reuse-geopandas",
            str(baseline_path),
            "--quiet",
        ]
    )

    assert status == 0
    assert seen["warmup"] is False
    assert seen["run_warmup"] is False


def test_shootout_cli_rejects_structurally_invalid_reuse_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = tmp_path / "workload.py"
    script.write_text(
        "print('SHOOTOUT_FINGERPRINT: rows=1')\n",
        encoding="utf-8",
    )
    identity = shootout_workload_identity(script)
    valid_metadata = {
        **identity,
        "scale": "1m",
        "geopandas_baseline_host": _baseline_host_identity(),
        "measurement_sha256": _measurement_identity(),
        "repeat": 3,
        "warmup": True,
        "timeout": 300,
    }
    malformed_payloads = [
        [],
        {"type": "shootout", "metadata": None},
        {
            "type": "shootout",
            "metadata": valid_metadata,
            "geopandas": {
                "timing": {"median_seconds": "fast"},
                "stdout": "SHOOTOUT_FINGERPRINT: rows=1\n",
            },
        },
        {
            "type": "shootout",
            "metadata": valid_metadata,
            "geopandas": {
                "timing": timing_from_samples([1.0]).to_dict(),
                "stdout": "SHOOTOUT_FINGERPRINT: rows=1\n",
                "environment": {
                    "python_version": "test",
                    "python_implementation": "cpython",
                    "packages": {
                        "geopandas": "1",
                        "numpy": "1",
                        "pandas": "1",
                        "pyarrow": "1",
                        "pyogrio": "1",
                        "shapely": "1",
                    },
                },
            },
        },
    ]

    for index, payload in enumerate(malformed_payloads):
        artifact = tmp_path / f"malformed-{index}.json"
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        status = vsbench_main(
            [
                "shootout",
                str(script),
                "--scale",
                "1m",
                "--reuse-geopandas",
                str(artifact),
                "--quiet",
            ]
        )
        captured = capsys.readouterr()
        assert status == 2
        assert captured.out == ""
        assert captured.err.startswith("Error: cannot reuse GeoPandas baseline:")
        assert "Traceback" not in captured.err

    deeply_nested = tmp_path / "deeply-nested.json"
    deeply_nested.write_text("[" * 2_000 + "0" + "]" * 2_000, encoding="utf-8")
    status = vsbench_main(
        [
            "shootout",
            str(script),
            "--scale",
            "1m",
            "--reuse-geopandas",
            str(deeply_nested),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert captured.err.startswith("Error: cannot reuse GeoPandas baseline:")
    assert "Traceback" not in captured.err


def test_run_shootout_measured_comparison_requires_both_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_harness(**kwargs):
        return ShootoutRun(
            label=kwargs["label"],
            timing=timing_from_samples([1.0]),
            stdout=(
                "SHOOTOUT_FINGERPRINT: rows=1\n"
                if kwargs["label"] == "geopandas"
                else ""
            ),
        )

    monkeypatch.setattr("vibespatial.bench.shootout._run_harness", _fake_run_harness)
    monkeypatch.setattr("vibespatial.runtime.has_gpu_runtime", lambda: False)

    result = run_shootout(
        Path("benchmarks/shootout/network_service_area.py"),
        repeat=1,
        warmup=False,
        quiet=True,
        baseline_python=sys.executable,
    )

    assert result.status == "error"
    assert result.speedup is None
    assert result.metadata["fingerprint"] == "missing"
    assert "current vibeSpatial run has no correctness fingerprint" in (
        result.status_reason
    )


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


def test_run_shootout_isolates_upstream_baseline_from_repo_shim(
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

    source_root = Path(__file__).resolve().parents[1] / "src"
    inherited = os.pathsep.join((str(source_root), "/tmp/shootout-extra"))
    monkeypatch.setenv("PYTHONPATH", inherited)
    monkeypatch.setattr("vibespatial.bench.shootout._run_harness", _fake_run_harness)
    monkeypatch.setattr("vibespatial.runtime.has_gpu_runtime", lambda: False)

    result = run_shootout(
        script,
        repeat=1,
        warmup=False,
        quiet=True,
        baseline_python=sys.executable,
    )

    assert result.status == "pass"
    baseline_paths = calls[0]["env"].get("PYTHONPATH", "").split(os.pathsep)
    vibespatial_paths = calls[1]["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(source_root) not in baseline_paths
    assert baseline_paths == ["/tmp/shootout-extra"]
    assert vibespatial_paths[0] == str(source_root)
    assert vibespatial_paths[1:] == ["/tmp/shootout-extra"]


def test_run_shootout_metadata_tags_public_physical_shapes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = tmp_path / "shape_probe.py"
    script.write_text(
        "\n".join(
            [
                "import geopandas as gpd",
                "# --- timed work starts here ---",
                "hits = gpd.sjoin(left, right, predicate='intersects')",
                "kept = left.loc[hits.index.unique()]",
                "safe = kept.loc[~kept.index.isin(hits.index.unique())]",
                "mask = gpd.clip(safe, right)",
                "overlay = gpd.overlay(safe, right, how='intersection')",
                "overlay['area'] = overlay.geometry.area",
                "summary = overlay.dissolve(by='group')",
                "# --- timed work ends here ---",
                "print('SHOOTOUT_FINGERPRINT: rows=1')",
            ]
        ),
        encoding="utf-8",
    )

    def _fake_run_harness(**kwargs):
        return ShootoutRun(
            label=kwargs["label"],
            timing=timing_from_samples([1.0]),
            stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
        )

    monkeypatch.setattr("vibespatial.bench.shootout._run_harness", _fake_run_harness)
    monkeypatch.setattr("vibespatial.runtime.has_gpu_runtime", lambda: False)

    result = run_shootout(
        script,
        repeat=1,
        warmup=False,
        quiet=True,
        baseline_python=sys.executable,
    )

    assert set(result.metadata["physical_shapes"]) >= {
        "semijoin",
        "anti_semijoin",
        "many_few_overlay",
        "grouped_geometry_reduce",
        "area_filter_after_overlay",
    }


def test_vsbench_shootout_directory_json_is_valid_suite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text(
            "print('SHOOTOUT_FINGERPRINT: rows=1')\n",
            encoding="utf-8",
        )

    def _fake_run_shootout(script: Path, **kwargs: object) -> ShootoutResult:
        run = ShootoutRun(
            label="geopandas",
            timing=timing_from_samples([1.0]),
            stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
        )
        return ShootoutResult(
            script=str(script),
            geopandas=run,
            vibespatial=ShootoutRun(
                label="vibespatial",
                timing=timing_from_samples([0.5]),
                stdout="SHOOTOUT_FINGERPRINT: rows=1\n",
            ),
            speedup=2.0,
            status="pass",
            status_reason="ok",
        )

    monkeypatch.setattr("vibespatial.bench.shootout.run_shootout", _fake_run_shootout)

    exit_code = vsbench_main(
        [
            "shootout",
            str(tmp_path),
            "--json",
            "--quiet",
            "--repeat",
            "3",
            "--scale",
            "10k",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["type"] == "shootout_suite"
    assert payload["metadata"]["script_count"] == 2
    assert payload["metadata"]["repeat"] == 3
    assert payload["metadata"]["scale"] == "10k"
    assert [Path(item["script"]).name for item in payload["results"]] == [
        "a.py",
        "b.py",
    ]


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
