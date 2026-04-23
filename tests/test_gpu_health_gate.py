from __future__ import annotations

from pathlib import Path

from scripts import gpu_health_gate


def test_parse_pre_push_updates() -> None:
    updates = gpu_health_gate.parse_pre_push_updates(
        "refs/heads/main abc123 refs/heads/main def456\n"
    )

    assert updates == (
        gpu_health_gate.RefUpdate(
            local_ref="refs/heads/main",
            local_oid="abc123",
            remote_ref="refs/heads/main",
            remote_oid="def456",
        ),
    )


def test_plan_skips_docs_only_paths() -> None:
    plan = gpu_health_gate.plan_for_paths(
        ("README.md", "docs/architecture/runtime.md"),
        "cache-key",
    )

    assert not plan.should_run
    assert plan.reason == "all changed paths are documentation"


def test_plan_runs_for_code_paths() -> None:
    plan = gpu_health_gate.plan_for_paths(
        ("README.md", "src/vibespatial/runtime/_runtime.py"),
        "cache-key",
    )

    assert plan.should_run
    assert plan.reason == "changed paths require the heavy gate"


def test_plan_force_overrides_docs_only_paths() -> None:
    plan = gpu_health_gate.plan_for_paths(("README.md",), "cache-key", force=True)

    assert plan.should_run
    assert plan.reason == "forced by environment"


def test_cache_hit_honors_ttl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VIBESPATIAL_GPU_GATE_IGNORE_CACHE", raising=False)
    path = tmp_path / "cache.json"
    gpu_health_gate.write_cache_entry(path, key="abc", label="test", now=100.0)

    assert gpu_health_gate.cache_hit(path, key="abc", now=150.0, ttl_seconds=100)
    assert not gpu_health_gate.cache_hit(path, key="abc", now=250.0, ttl_seconds=100)
