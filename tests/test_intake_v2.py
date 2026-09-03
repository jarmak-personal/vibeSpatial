from __future__ import annotations

import subprocess

from scripts.build_intake_index import (
    build_file_entries,
    build_intake_index,
    should_index_file,
)
from scripts.check_docs import _is_header_scan_excluded
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


def test_generated_intake_index_excludes_claude_property_snapshot() -> None:
    index = build_intake_index()
    assert all(entry["path"] != ".claude/.property-before.json" for entry in index["files"])


def test_repository_discovery_excludes_nested_worktrees(tmp_path) -> None:
    nested_doc = tmp_path / ".worktrees" / "branch" / "docs" / "plan.md"
    nested_doc.parent.mkdir(parents=True)
    nested_doc.write_text("# Nested checkout\n")

    assert not should_index_file(nested_doc, tmp_path)
    assert _is_header_scan_excluded(".worktrees/branch/docs/plan.md")


def test_repository_discovery_excludes_ignored_untracked_files(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("reports/*\n")
    tracked_report = tmp_path / "reports" / "tracked.md"
    ignored_report = tmp_path / "reports" / "local.md"
    untracked_source = tmp_path / "local.py"
    deleted_source = tmp_path / "deleted.py"
    tracked_report.parent.mkdir()
    tracked_report.write_text("# Tracked evidence\n")
    ignored_report.write_text("# Local evidence\n")
    untracked_source.write_text('"""Discoverable worktree source."""\n')
    deleted_source.write_text('"""Deleted tracked source."""\n')
    subprocess.run(
        [
            "git",
            "add",
            ".gitignore",
            "-f",
            tracked_report.relative_to(tmp_path),
            deleted_source.relative_to(tmp_path),
        ],
        cwd=tmp_path,
        check=True,
    )
    deleted_source.unlink()

    indexed_paths = {
        entry["path"] for entry in build_file_entries(tmp_path, doc_entries=[])
    }

    assert "reports/tracked.md" in indexed_paths
    assert "reports/local.md" not in indexed_paths
    assert "local.py" in indexed_paths
    assert "deleted.py" not in indexed_paths
