from __future__ import annotations

import json
import sys

from scripts import intake
from scripts.check_docs import collect_unregistered_header_docs
from scripts.intake import plan_request


def _doc_paths(query: str) -> list[str]:
    return [doc["path"] for doc in plan_request(query)["docs"]]


def _file_paths(query: str) -> list[str]:
    return [entry["path"] for entry in plan_request(query)["files"]]


def _assert_rank(paths: list[str], path: str, max_rank: int) -> None:
    assert path in paths
    assert paths.index(path) < max_rank


def test_intake_router_audit_prefers_router_docs_over_generic_audit_docs() -> None:
    docs = _doc_paths("audit intake router effectiveness repo growth discovery signal noise friction improvements")

    assert docs[0] == "docs/ops/intake.md"
    assert docs[0] != "docs/testing/gpu-performance-checklist.md"


def test_precompile_request_routes_to_kernel_caching_doc_and_precompile_files() -> None:
    query = "precompile_all startup latency CCCL cache"
    docs = _doc_paths(query)
    files = _file_paths(query)

    assert docs[0] == "docs/architecture/gpu-kernel-caching.md"
    _assert_rank(files, "src/vibespatial/cuda/cccl_precompile.py", 3)
    _assert_rank(files, "src/vibespatial/cuda/cccl_cubin_cache.py", 3)


def test_overlay_keep_geom_type_prioritizes_public_overlay_surface() -> None:
    files = _file_paths("overlay function keep_geom_type")

    assert files[0] == "src/vibespatial/api/tools/overlay.py"
    _assert_rank(files, "tests/test_overlay_api.py", 3)


def test_adr_precision_request_keeps_workflow_and_topic_docs() -> None:
    query = "add a new ADR for precision dispatch"
    docs = _doc_paths(query)
    files = _file_paths(query)

    assert docs[0] == "docs/decisions/index.md"
    assert "docs/architecture/precision.md" in docs
    assert files[0] == "scripts/new_decision.py"


def test_registered_growth_docs_route_to_their_own_topics() -> None:
    assert _doc_paths("pylibcudf capabilities cudf geometry")[0] == (
        "docs/architecture/pylibcudf-capabilities.md"
    )
    assert _doc_paths("public API performance roadmap")[0] == (
        "docs/testing/public-api-performance-roadmap.md"
    )


def test_unregistered_generated_header_docs_are_forbidden() -> None:
    assert collect_unregistered_header_docs() == []


def test_cli_json_and_explain_outputs(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["intake.py", "--json", "precompile_all"])
    intake.main()
    data = json.loads(capsys.readouterr().out)

    assert data["docs"][0]["path"] == "docs/architecture/gpu-kernel-caching.md"
    assert "score_details" in data["docs"][0]

    monkeypatch.setattr(sys, "argv", ["intake.py", "--explain", "overlay function keep_geom_type"])
    intake.main()
    output = capsys.readouterr().out

    assert "score=" in output
    assert "direct=" in output
