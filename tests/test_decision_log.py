from __future__ import annotations

from pathlib import Path

from scripts.intake import plan_request
from scripts.new_decision import load_decisions, render_index


def test_decision_index_includes_sample_adr() -> None:
    root = Path(__file__).resolve().parents[1]
    decisions = load_decisions(root)
    rendered = render_index(decisions)

    assert "ADR-0001" in rendered
    assert "vibeSpatial-o17.2.12" in rendered
    assert "docs/decisions/0001-mixed-geometries.md" in rendered


def test_intake_routes_mixed_geometry_request_to_adr() -> None:
    plan = plan_request("mixed geometry design decision")
    doc_paths = [doc["path"] for doc in plan["docs"]]
    file_paths = [entry["path"] for entry in plan["files"]]

    assert "docs/decisions/index.md" in doc_paths
    assert "docs/decisions/0001-mixed-geometries.md" in file_paths
