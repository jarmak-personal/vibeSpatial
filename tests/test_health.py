from __future__ import annotations

from pathlib import Path

from scripts import health, pre_land


def test_collect_health_report_includes_property_summary_and_signal_statuses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "profile_kernels.py").write_text("", encoding="utf-8")
    src_bench_dir = tmp_path / "src" / "vibespatial" / "bench"
    src_bench_dir.mkdir(parents=True)
    (src_bench_dir / "compare.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        health,
        "collect_coverage_and_tests",
        lambda root: (
            {"percent": 10.0, "covered_lines": 10, "num_statements": 100},
            {"counts": {"passed": 4, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}, "broken": []},
            health.CommandStatus(ok=True, command="pytest", returncode=0),
        ),
    )
    monkeypatch.setattr(health, "collect_lint", lambda root: {"ok": True})
    monkeypatch.setattr(health, "collect_docs", lambda root: {"ok": True})
    monkeypatch.setattr(
        health,
        "collect_gpu_acceleration",
        lambda root, include, timeout: {"ok": True, "available": False, "reason": "not requested"},
    )
    monkeypatch.setattr(
        health,
        "collect_property_summary",
        lambda: {
            "ok": True,
            "status": "ok",
            "clean": 6,
            "debt": 0,
            "regressed": 0,
            "total": 6,
            "total_distance": 0.0,
        },
    )

    report = health.collect_health_report(tmp_path)

    assert report["properties"]["clean"] == 6
    assert report["benchmarks"]["status"] == "configured"
    assert report["transfer_audit"]["status"] == "configured"
    assert report["dispatch_policy"]["status"] == "configured"
    assert report["exit_code"] == 0


def test_collect_health_report_fails_when_properties_regress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        health,
        "collect_coverage_and_tests",
        lambda root: (
            {"percent": 10.0, "covered_lines": 10, "num_statements": 100},
            {"counts": {"passed": 4, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}, "broken": []},
            health.CommandStatus(ok=True, command="pytest", returncode=0),
        ),
    )
    monkeypatch.setattr(health, "collect_lint", lambda root: {"ok": True})
    monkeypatch.setattr(health, "collect_docs", lambda root: {"ok": True})
    monkeypatch.setattr(
        health,
        "collect_gpu_acceleration",
        lambda root, include, timeout: {"ok": True, "available": False, "reason": "not requested"},
    )
    monkeypatch.setattr(
        health,
        "collect_property_summary",
        lambda: {
            "ok": False,
            "status": "regressed",
            "clean": 5,
            "debt": 0,
            "regressed": 1,
            "total": 6,
            "total_distance": 1.2,
        },
    )

    report = health.collect_health_report(tmp_path)

    assert report["exit_code"] == 1


def test_print_summary_omits_coverage_and_surfaces_honest_statuses(capsys) -> None:
    report = {
        "properties": {
            "ok": True,
            "status": "ok",
            "clean": 6,
            "debt": 0,
            "regressed": 0,
            "total": 6,
            "total_distance": 0.0,
        },
        "coverage": {"percent": 9.99, "covered_lines": 10, "num_statements": 100},
        "tests": {
            "counts": {"passed": 26, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0},
            "ok": True,
            "command": "pytest",
        },
        "lint": {"ok": True},
        "docs": {"ok": True},
        "benchmarks": {"status": "configured", "detail": "profiling rails defined: scripts/profile_kernels.py"},
        "transfer_audit": {"status": "configured", "detail": "transfer audit policy defined"},
        "dispatch_policy": {"status": "configured", "detail": "dispatch policy defined"},
        "gpu_acceleration": {"ok": True, "available": False, "reason": "not requested"},
        "exit_code": 0,
    }

    health.print_summary(report)
    captured = capsys.readouterr()

    assert "coverage:" not in captured.out
    assert "Repo Health — bootstrap: PASS" in captured.out
    assert "Property summary:" in captured.out
    assert "Smoke suite: PASS (26/26 passed)" in captured.out
    assert "Additional signals:" in captured.out
    assert "benchmarks: CONFIGURED" in captured.out


def test_pre_land_reports_configured_and_unmeasured_signals(monkeypatch) -> None:
    monkeypatch.setattr(
        pre_land,
        "collect_health_report",
        lambda root: {
            "exit_code": 0,
            "properties": {
                "ok": True,
                "clean": 6,
                "regressed": 0,
                "total": 6,
                "total_distance": 0.0,
            },
            "lint": {"ok": True},
            "docs": {"ok": True},
            "tests": {"ok": True, "command": "pytest"},
            "benchmarks": {"status": "configured", "detail": "profiling rails defined"},
            "transfer_audit": {"status": "configured", "detail": "transfer audit policy defined"},
            "dispatch_policy": {"status": "configured", "detail": "dispatch policy defined"},
        },
    )

    formatted = pre_land.format_results(pre_land.run_pre_land(Path.cwd()))

    assert "PASS properties" in formatted
    assert "CONFIGURED benchmarks" in formatted
    assert "UNMEASURED tier_gate" in formatted
    assert "CONFIGURED dispatch_policy" in formatted
    assert "coverage" not in formatted


def test_load_health_surfaces_parses_toml_matrix(tmp_path: Path) -> None:
    config_path = tmp_path / "health_surfaces.toml"
    config_path.write_text(
        """
[[surface]]
name = "versioning"
owners = ["pyproject.toml"]
command = "uv run pytest -q tests/test_version_consistency.py"
required = true
runtime = "any"
allowed_states = ["pass", "fail"]
baseline_key = "contract.versioning"
""",
        encoding="utf-8",
    )

    surfaces = health.load_health_surfaces(config_path)

    assert len(surfaces) == 1
    assert surfaces[0].name == "versioning"
    assert surfaces[0].owners == ("pyproject.toml",)
    assert surfaces[0].allowed_states == ("pass", "fail")


def test_collect_contract_health_report_skips_gpu_surface_without_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "health_surfaces.toml"
    config_path.write_text(
        """
[[surface]]
name = "versioning"
owners = ["pyproject.toml"]
command = "uv run pytest -q tests/test_version_consistency.py"
required = true
runtime = "any"
allowed_states = ["pass", "fail"]

[[surface]]
name = "overlay"
owners = ["src/vibespatial/api/tools/overlay.py"]
command = "uv run pytest -q tests/test_overlay_api.py"
required = true
runtime = "gpu-preferred"
allowed_states = ["pass", "fail", "skipped-no-gpu"]
""",
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_run_command(command: list[str], *, cwd: Path, timeout: int) -> health.CommandStatus:
        calls.append(command)
        return health.CommandStatus(
            ok=True,
            command=" ".join(command),
            stdout="1 passed in 0.01s\n",
            returncode=0,
        )

    monkeypatch.setattr(health, "has_gpu_runtime", lambda: False)
    monkeypatch.setattr(health, "run_command", fake_run_command)

    report = health.collect_contract_health_report(tmp_path, config_path=config_path)

    assert report["status"] == "pass"
    assert report["required_total"] == 2
    assert report["required_passing"] == 2
    assert report["surfaces"][0]["state"] == "pass"
    assert report["surfaces"][1]["state"] == "skipped-no-gpu"
    assert calls == [["uv", "run", "pytest", "-q", "tests/test_version_consistency.py"]]


def test_main_contract_json_uses_contract_collector(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        health,
        "collect_contract_health_report",
        lambda root, surface_timeout=600: {"tier": "contract", "status": "pass", "exit_code": 0},
    )

    exit_code = health.main(["--tier", "contract", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"tier": "contract"' in captured.out


def test_evaluate_check_report_allows_steady_state_contract_failure() -> None:
    report = {
        "tier": "contract",
        "exit_code": 1,
        "surfaces": [
            {
                "name": "overlay",
                "required": True,
                "state": "fail",
                "passed": 236,
                "total": 260,
                "returncode": 1,
                "ok": False,
            }
        ],
    }
    baseline = {
        "contract": {
            "surfaces": {
                "overlay": {"state": "fail", "passed": 236, "total": 260}
            }
        }
    }

    exit_code, regressions = health.evaluate_check_report(report, baseline)

    assert exit_code == 0
    assert regressions == []


def test_evaluate_check_report_fails_contract_regression() -> None:
    report = {
        "tier": "contract",
        "exit_code": 1,
        "surfaces": [
            {
                "name": "overlay",
                "required": True,
                "state": "fail",
                "passed": 230,
                "total": 260,
                "returncode": 1,
                "ok": False,
            }
        ],
    }
    baseline = {
        "contract": {
            "surfaces": {
                "overlay": {"state": "fail", "passed": 236, "total": 260}
            }
        }
    }

    exit_code, regressions = health.evaluate_check_report(report, baseline)

    assert exit_code == 1
    assert regressions == ["overlay: failures increased from 24 to 30"]


def test_evaluate_check_report_allows_gpu_surface_to_skip_without_runtime() -> None:
    report = {
        "tier": "contract",
        "exit_code": 0,
        "surfaces": [
            {
                "name": "gpu_dispatch_correctness",
                "required": True,
                "runtime": "gpu-required",
                "state": "skipped-no-gpu",
                "passed": 0,
                "total": 0,
                "returncode": 0,
                "ok": True,
            }
        ],
    }
    baseline = {
        "contract": {
            "surfaces": {
                "gpu_dispatch_correctness": {"state": "pass", "passed": 1, "total": 1}
            }
        }
    }

    exit_code, regressions = health.evaluate_check_report(report, baseline)

    assert exit_code == 0
    assert regressions == []


def test_collect_release_health_report_measures_release_checks(monkeypatch) -> None:
    monkeypatch.setattr(
        health,
        "run_command",
        lambda command, *, cwd, timeout: health.CommandStatus(
            ok=True,
            command=" ".join(command),
            stdout="2 passed in 0.01s\n",
            returncode=0,
        ),
    )
    monkeypatch.setattr(health, "collect_docs", lambda root: {"ok": True, "errors": []})
    monkeypatch.setattr(
        health,
        "collect_release_packaging_build",
        lambda root: {"name": "packaging_build", "status": "pass", "detail": "1 wheel, 1 sdist"},
    )
    monkeypatch.setattr(
        health,
        "collect_release_doc_build",
        lambda root: {"name": "doc_build", "status": "pass", "detail": "html docs built"},
    )
    monkeypatch.setattr(
        health,
        "collect_release_notes_check",
        lambda root: {
            "name": "release_notes",
            "status": "pass",
            "detail": "GitHub Release notes auto-generated from commit subjects",
        },
    )
    monkeypatch.setattr(
        health,
        "collect_release_generated_artifacts",
        lambda root: {"name": "generated_artifacts", "status": "pass", "detail": "clean"},
    )

    report = health.collect_release_health_report(Path.cwd())

    assert report["status"] == "pass"
    assert report["exit_code"] == 0
    assert [check["name"] for check in report["checks"]] == [
        "versioning",
        "doc_hygiene",
        "packaging_build",
        "doc_build",
        "release_notes",
        "generated_artifacts",
    ]


def test_collect_release_health_report_fails_dirty_generated_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(
        health,
        "run_command",
        lambda command, *, cwd, timeout: health.CommandStatus(
            ok=True,
            command=" ".join(command),
            stdout="2 passed in 0.01s\n",
            returncode=0,
        ),
    )
    monkeypatch.setattr(health, "collect_docs", lambda root: {"ok": True, "errors": []})
    monkeypatch.setattr(
        health,
        "collect_release_packaging_build",
        lambda root: {"name": "packaging_build", "status": "pass", "detail": "1 wheel, 1 sdist"},
    )
    monkeypatch.setattr(
        health,
        "collect_release_doc_build",
        lambda root: {"name": "doc_build", "status": "pass", "detail": "html docs built"},
    )
    monkeypatch.setattr(
        health,
        "collect_release_notes_check",
        lambda root: {
            "name": "release_notes",
            "status": "pass",
            "detail": "GitHub Release notes auto-generated from commit subjects",
        },
    )
    monkeypatch.setattr(
        health,
        "collect_release_generated_artifacts",
        lambda root: {
            "name": "generated_artifacts",
            "status": "fail",
            "detail": "tracked generated artifact is dirty: docs/ops/intake-index.json",
        },
    )

    report = health.collect_release_health_report(Path.cwd())

    assert report["status"] == "fail"
    assert report["exit_code"] == 1


def test_evaluate_check_report_allows_steady_state_gpu_failure() -> None:
    report = {
        "tier": "gpu",
        "status": "fail",
        "exit_code": 1,
        "gpu_acceleration": {
            "gpu_accel_pct": 20.08,
            "raw_dispatch_pct": 20.08,
            "value_dispatch_pct": 67.25,
            "total_dispatches": 1000,
            "fallback_dispatches": 16,
        },
    }
    baseline = {
        "gpu": {
            "status": "fail",
            "raw_dispatch_pct": 20.08,
            "value_dispatch_pct": 67.25,
            "fallback_rate": 0.016,
        }
    }

    exit_code, regressions = health.evaluate_check_report(report, baseline)

    assert exit_code == 0
    assert regressions == []


def test_evaluate_check_report_fails_gpu_regression() -> None:
    report = {
        "tier": "gpu",
        "status": "fail",
        "exit_code": 1,
        "gpu_acceleration": {
            "gpu_accel_pct": 19.5,
            "raw_dispatch_pct": 19.5,
            "value_dispatch_pct": 66.5,
            "total_dispatches": 1000,
            "fallback_dispatches": 20,
        },
    }
    baseline = {
        "gpu": {
            "status": "fail",
            "raw_dispatch_pct": 20.08,
            "value_dispatch_pct": 67.25,
            "fallback_rate": 0.016,
        }
    }

    exit_code, regressions = health.evaluate_check_report(report, baseline)

    assert exit_code == 1
    assert regressions == [
        "value-weighted gpu acceleration regressed from 67.25% to 66.50%",
        "fallback rate regressed from 0.0160 to 0.0200",
    ]


def test_evaluate_check_report_ignores_gpu_noise_below_display_precision() -> None:
    report = {
        "tier": "gpu",
        "status": "fail",
        "exit_code": 1,
        "gpu_acceleration": {
            "gpu_accel_pct": 20.681345681345682,
            "raw_dispatch_pct": 20.681345681345682,
            "value_dispatch_pct": 65.08094347877969,
            "total_dispatches": 21664,
            "fallback_dispatches": 16,
        },
    }
    baseline = {
        "gpu": {
            "status": "fail",
            "raw_dispatch_pct": 20.681184751287258,
            "value_dispatch_pct": 65.08649830980315,
            "fallback_rate": 0.000735,
        }
    }

    exit_code, regressions = health.evaluate_check_report(report, baseline)

    assert exit_code == 0
    assert regressions == []


def test_evaluate_check_report_ignores_small_gpu_value_dispatch_jitter() -> None:
    report = {
        "tier": "gpu",
        "status": "fail",
        "exit_code": 1,
        "gpu_acceleration": {
            "gpu_accel_pct": 21.425570648368577,
            "raw_dispatch_pct": 21.425570648368577,
            "value_dispatch_pct": 65.9644556509008,
            "total_dispatches": 21423,
            "fallback_dispatches": 31,
        },
    }
    baseline = {
        "gpu": {
            "status": "fail",
            "raw_dispatch_pct": 21.42357712208083,
            "value_dispatch_pct": 65.99764352171617,
            "fallback_rate": 0.001445,
        }
    }

    exit_code, regressions = health.evaluate_check_report(report, baseline)

    assert exit_code == 0
    assert regressions == []


def test_evaluate_check_report_uses_value_dispatch_metric_over_raw_helper_metric() -> None:
    report = {
        "tier": "gpu",
        "status": "fail",
        "exit_code": 1,
        "gpu_acceleration": {
            "gpu_accel_pct": 18.0,
            "raw_dispatch_pct": 18.0,
            "value_dispatch_pct": 67.25,
            "total_dispatches": 1000,
            "fallback_dispatches": 16,
        },
    }
    baseline = {
        "gpu": {
            "status": "fail",
            "raw_dispatch_pct": 20.08,
            "value_dispatch_pct": 67.25,
            "fallback_rate": 0.016,
        }
    }

    exit_code, regressions = health.evaluate_check_report(report, baseline)

    assert exit_code == 0
    assert regressions == []


def test_update_baseline_payload_stores_contract_snapshot() -> None:
    report = {
        "tier": "contract",
        "surfaces": [
            {"name": "versioning", "state": "pass", "passed": 2, "total": 2},
            {"name": "overlay", "state": "fail", "passed": 236, "total": 260},
        ],
    }

    payload = health.update_baseline_payload({}, report)

    assert payload["schema_version"] == 2
    assert payload["contract"]["surfaces"]["versioning"] == {
        "state": "pass",
        "passed": 2,
        "total": 2,
    }
    assert payload["contract"]["surfaces"]["overlay"]["passed"] == 236


def test_update_baseline_payload_stores_value_weighted_gpu_snapshot() -> None:
    report = {
        "tier": "gpu",
        "status": "pass",
        "gpu_acceleration": {
            "gpu_accel_pct": 20.08,
            "raw_dispatch_pct": 20.08,
            "value_dispatch_pct": 67.25,
            "total_dispatches": 1000,
            "fallback_dispatches": 16,
        },
    }

    payload = health.update_baseline_payload({}, report)

    assert payload["schema_version"] == 2
    assert payload["gpu"] == {
        "status": "pass",
        "raw_dispatch_pct": 20.08,
        "value_dispatch_pct": 67.25,
        "fallback_rate": 0.016,
    }
