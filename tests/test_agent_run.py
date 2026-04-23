from __future__ import annotations

import json
from pathlib import Path

from scripts import agent_run
from scripts.build_intake_index import build_intake_index


def test_start_run_creates_problem_capsule_and_active_pointer(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        agent_run,
        "plan_request",
        lambda request: {
            "request": request,
            "docs": [{"path": "docs/ops/intake.md"}],
            "files": [{"path": "scripts/intake.py"}],
            "verify": ["uv run pytest tests/test_agent_run.py -q"],
            "risks": ["routing drift"],
            "confidence": "high",
        },
    )

    exit_code = agent_run.main(
        [
            "--root",
            str(tmp_path),
            "--run-id",
            "run-001",
            "start",
            "improve intake routing",
        ]
    )

    assert exit_code == 0
    problem = json.loads((tmp_path / ".agents" / "runs" / "run-001" / "problem.json").read_text())
    assert problem["request"] == "improve intake routing"
    assert problem["intent"] == "implementation"
    assert problem["expected_verification"] == ["uv run pytest tests/test_agent_run.py -q"]
    assert (tmp_path / ".agents" / "runs" / "ACTIVE").read_text().strip() == "run-001"


def test_hypothesis_evidence_learning_and_close_flow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        agent_run,
        "plan_request",
        lambda request: {
            "request": request,
            "docs": [],
            "files": [],
            "verify": ["uv run pytest tests/test_agent_run.py -q"],
            "risks": [],
            "confidence": "medium",
        },
    )

    root_args = ["--root", str(tmp_path)]
    assert agent_run.main([*root_args, "--run-id", "run-002", "start", "track a bug"]) == 0
    assert agent_run.main([*root_args, "hypothesis", "add", "Doc routing is wrong"]) == 0
    assert agent_run.main([*root_args, "evidence", "add", "--hypothesis", "H001", "overlay.py ranked 7th"]) == 0
    assert agent_run.main([*root_args, "hypothesis", "resolve", "H001", "--status", "accepted"]) == 0
    assert (
        agent_run.main(
            [
                *root_args,
                "learning",
                "add",
                "--kind",
                "intake-misroute",
                "--summary",
                "Need a gold query",
                "--repo-action",
                "test",
                "--path",
                "tests/test_intake_quality.py",
            ]
        )
        == 0
    )
    assert agent_run.main([*root_args, "learning", "review", "--require-resolved"]) == 0
    assert (
        agent_run.main(
            [
                *root_args,
                "close",
                "--summary",
                "Fixed router quality",
                "--verification",
                "uv run pytest tests/test_agent_run.py -q",
                "--risk",
                "Did not run full pre-land review",
            ]
        )
        == 0
    )

    closeout = json.loads((tmp_path / ".agents" / "runs" / "run-002" / "closeout.json").read_text())
    assert closeout["summary"] == "Fixed router quality"
    assert closeout["learning"][0]["repo_action"] == "test"
    assert closeout["open_hypotheses"] == []
    closeout_md = (tmp_path / ".agents" / "runs" / "run-002" / "closeout.md").read_text()
    assert "Fixed router quality" in closeout_md


def test_status_json_reports_current_run(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        agent_run,
        "plan_request",
        lambda request: {"request": request, "docs": [], "files": [], "verify": [], "risks": []},
    )

    root_args = ["--root", str(tmp_path)]
    assert agent_run.main([*root_args, "--run-id", "run-003", "start", "measure progress"]) == 0
    assert agent_run.main([*root_args, "hypothesis", "add", "Benchmark regression is real"]) == 0
    capsys.readouterr()
    assert agent_run.main([*root_args, "status", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "run-003"
    assert payload["open_hypotheses"][0]["id"] == "H001"


def test_learning_add_requires_none_reason(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        agent_run,
        "plan_request",
        lambda request: {"request": request, "docs": [], "files": [], "verify": [], "risks": []},
    )
    root_args = ["--root", str(tmp_path)]
    assert agent_run.main([*root_args, "--run-id", "run-004", "start", "close a task"]) == 0

    try:
        agent_run.main(
            [
                *root_args,
                "learning",
                "add",
                "--kind",
                "workflow-gap",
                "--summary",
                "No repo artifact needed",
                "--repo-action",
                "none",
            ]
        )
    except SystemExit as exc:
        assert "--reason is required when --repo-action none" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected SystemExit")


def test_intake_index_ignores_local_agent_runs() -> None:
    index = build_intake_index()
    assert all(not entry["path"].startswith(".agents/runs/") for entry in index["files"])
