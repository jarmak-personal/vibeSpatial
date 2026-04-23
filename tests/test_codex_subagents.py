from __future__ import annotations

import tomllib
from pathlib import Path


def _collect_repo_agents() -> list[Path]:
    source_dir = Path(".codex/agents")
    agents = sorted(source_dir.glob("*.toml"))
    if not agents:
        raise AssertionError(f"No Codex agent files found in {source_dir}")
    return agents


def test_repo_subagent_files_are_valid() -> None:
    agents = _collect_repo_agents()

    assert [agent.stem for agent in agents] == ["cuda-engineer", "python-engineer"]
    for agent in agents:
        payload = tomllib.loads(agent.read_text(encoding="utf-8"))
        assert payload["name"] == agent.stem
        assert payload["description"]
        assert payload["developer_instructions"]


def test_repo_subagents_use_expected_fields() -> None:
    required_fields = {"name", "description", "developer_instructions"}

    for agent in _collect_repo_agents():
        payload = tomllib.loads(agent.read_text(encoding="utf-8"))
        assert required_fields.issubset(payload)
