from __future__ import annotations

from scripts.build_intake_index import evaluate_intake_index
from scripts.intake import plan_request
from scripts.update_doc_headers import evaluate_doc_headers


def test_runtime_request_prioritizes_runtime_doc_and_file() -> None:
    plan = plan_request("Investigate GPU fallback behavior in the runtime selection path")

    assert plan["docs"][0]["path"] == "docs/architecture/runtime.md"
    assert "src/vibespatial/runtime.py" in {entry["path"] for entry in plan["files"]}
    assert "uv run pytest" in plan["verify"]


def test_upstream_request_prioritizes_vendored_tests() -> None:
    plan = plan_request("Refresh vendored GeoPandas fixtures and upstream tests")

    assert plan["docs"][0]["path"] == "tests/upstream/README.md"
    assert "scripts/vendor_geopandas_tests.py" in {entry["path"] for entry in plan["files"]}
    assert "uv run pytest --collect-only tests/upstream/geopandas" in plan["verify"]


def test_generated_docs_are_current() -> None:
    assert evaluate_doc_headers(write=False)["outdated"] == []
    assert evaluate_intake_index(write=False)["outdated"] == []
