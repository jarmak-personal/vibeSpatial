from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

try:
    from .build_intake_index import build_intake_index, evaluate_intake_index
    from .update_doc_headers import HEADER_START, evaluate_doc_headers, get_repo_root, load_config
except ImportError:  # pragma: no cover - script execution path
    from build_intake_index import build_intake_index, evaluate_intake_index
    from update_doc_headers import HEADER_START, evaluate_doc_headers, get_repo_root, load_config


REQUIRED_ROUTING_SECTIONS = ("Intent", "Request Signals", "Open First", "Verify", "Risks")
HEADER_SCAN_EXCLUDED_DIRS = {".git", ".venv", "docs/_build"}


def _is_header_scan_excluded(relative_path: str) -> bool:
    parts = set(Path(relative_path).parts)
    if parts & {".git", ".venv"}:
        return True
    return relative_path.startswith("docs/_build/")


def collect_unregistered_header_docs(root: Path | None = None) -> list[str]:
    repo_root = root or get_repo_root()
    configured = set(load_config(repo_root)["files"])
    missing: list[str] = []

    for path in sorted(repo_root.rglob("*.md")):
        relative_path = path.relative_to(repo_root).as_posix()
        if _is_header_scan_excluded(relative_path):
            continue
        if relative_path in configured:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if HEADER_START in content:
            missing.append(relative_path)

    return missing


def collect_doc_contract_errors(root: Path | None = None) -> list[str]:
    repo_root = root or get_repo_root()
    index = build_intake_index(repo_root)
    errors: list[str] = []

    for doc in index["docs"]:
        sections = set(doc["sections"])
        for section in REQUIRED_ROUTING_SECTIONS:
            if section not in sections:
                errors.append(f"{doc['path']} missing required section: ## {section}")
        if not doc["request_signals"]:
            errors.append(f"{doc['path']} has no request signals")
        if not doc["open_first"]:
            errors.append(f"{doc['path']} has no open-first paths")
        if not doc["verify"]:
            errors.append(f"{doc['path']} has no verification commands")
        if not doc["risks"]:
            errors.append(f"{doc['path']} has no risks")

    return errors


def stage_paths(root: Path, paths: list[str]) -> None:
    if not paths:
        return
    subprocess.run(["git", "add", *paths], cwd=root, check=True)


def refresh_generated_docs(root: Path, *, stage_generated: bool = False) -> dict[str, Any]:
    header_report = evaluate_doc_headers(root, write=True)
    index_report = evaluate_intake_index(root, write=True)
    updated_paths = [*header_report["updated"], *index_report["updated"]]
    if stage_generated:
        stage_paths(root, updated_paths)
    return {
        "headers": header_report,
        "index": index_report,
        "updated_paths": updated_paths,
    }


def check_docs(root: Path) -> list[str]:
    header_report = evaluate_doc_headers(root, write=False)
    index_report = evaluate_intake_index(root, write=False)
    errors = collect_doc_contract_errors(root)
    for path in collect_unregistered_header_docs(root):
        errors.append(f"{path} has a generated doc header but is missing from docs/doc_headers.json")

    for path in header_report["outdated"]:
        errors.append(f"{path} has an outdated generated header")
    for violation in header_report["over_budget"]:
        errors.append(
            f"{violation['file']} exceeds body line budget ({violation['current']}/{violation['budget']})"
        )
    for path in index_report["outdated"]:
        errors.append(f"{path} is out of date")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh or validate generated docs and intake artifacts.")
    parser.add_argument("--check", action="store_true", help="Fail when docs or generated files are stale.")
    parser.add_argument("--refresh", action="store_true", help="Rewrite generated doc headers and intake index.")
    parser.add_argument(
        "--stage-generated",
        action="store_true",
        help="Stage generated files after refresh. Intended for git hooks.",
    )
    args = parser.parse_args()

    root = get_repo_root()
    should_refresh = args.refresh or args.stage_generated
    if should_refresh:
        report = refresh_generated_docs(root, stage_generated=args.stage_generated)
        if report["updated_paths"]:
            print("Refreshed generated docs:")
            for path in report["updated_paths"]:
                print(f"- {path}")
        else:
            print("Generated docs already up to date.")

    if not args.check:
        return

    errors = check_docs(root)
    if errors:
        print("Documentation checks failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Documentation checks passed.")


if __name__ == "__main__":
    main()
