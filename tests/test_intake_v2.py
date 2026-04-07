from __future__ import annotations

from scripts.intake import plan_request
from scripts.update_doc_headers import evaluate_doc_headers


def test_runtime_request_prioritizes_runtime_doc_and_file() -> None:
    plan = plan_request("Investigate GPU fallback behavior in the runtime selection path")

    assert plan["docs"][0]["path"] == "docs/architecture/runtime.md"
    file_paths = {entry["path"] for entry in plan["files"]}
    has_runtime_file = any("runtime" in p and p.endswith(".py") for p in file_paths)
    assert has_runtime_file
    assert "uv run pytest" in plan["verify"]


def test_upstream_request_prioritizes_vendored_tests() -> None:
    plan = plan_request("Refresh vendored GeoPandas fixtures and upstream tests")

    assert plan["docs"][0]["path"] == "tests/upstream/README.md"
    assert "scripts/vendor_geopandas_tests.py" in {entry["path"] for entry in plan["files"]}
    assert "uv run pytest --collect-only tests/upstream/geopandas" in plan["verify"]


def test_precompile_request_routes_to_cccl_precompile_surface() -> None:
    plan = plan_request("precompile_all")

    file_paths = {entry["path"] for entry in plan["files"]}

    assert "src/vibespatial/cuda/cccl_precompile.py" in file_paths


def test_overlay_keep_geom_type_request_routes_to_public_overlay_surface() -> None:
    plan = plan_request("overlay function keep_geom_type")

    file_paths = {entry["path"] for entry in plan["files"]}

    assert "src/vibespatial/api/tools/overlay.py" in file_paths


def test_generated_docs_are_current() -> None:
    assert evaluate_doc_headers(write=False)["outdated"] == []
    # Intake index freshness is enforced locally by the pre-commit hook,
    # not in CI — the index drifts between environments.
