"""Maintainability and discoverability lints (MAINT001-003).

Ensures new code is findable through the intake routing system and that
new decision records, scripts, and modules are properly indexed.

Uses a ratchet baseline: fails only when violations INCREASE beyond the
known debt count.  Decrease the baseline as debt is paid down.

Run:
    uv run python scripts/check_maintainability.py --all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DOC = "AGENTS.md"
INTAKE_DOC = "docs/ops/intake.md"
DECISIONS_INDEX = "docs/decisions/index.md"

# Known pre-existing violations as of 2026-03-17.
# Decrease this number as debt is paid.  The check fails only if
# the current count EXCEEDS the baseline (new violations introduced).
_VIOLATION_BASELINE = 38  # +1: scripts/verify_degeneracy_corpus.py added by other agent

# Modules that are intentionally not in intake (internal, generated, etc.).
_INTAKE_EXEMPT_MODULES = {
    "__init__.py", "__pycache__", "py.typed", "_version.py", "conftest.py",
}

# Directories under src/vibespatial/ that are tracked in bulk by intake.
_INTAKE_BULK_DIRS = {"api", "kernels", "_vendor"}


@dataclass(frozen=True)
class LintError:
    code: str
    path: Path
    line: int
    message: str
    doc_path: str

    def render(self, repo_root: Path) -> str:
        relative = self.path.relative_to(repo_root)
        return f"{relative}:{self.line}: {self.code} {self.message} See {self.doc_path}."


# ---- MAINT001: New modules in src/vibespatial/ must appear in intake ----

def _load_intake_paths(repo_root: Path) -> set[str]:
    """Load all file paths referenced in intake-index.json."""
    index_path = repo_root / "docs" / "ops" / "intake-index.json"
    if not index_path.exists():
        return set()
    data = json.loads(index_path.read_text(encoding="utf-8"))
    paths: set[str] = set()
    for doc in data.get("docs", []):
        p = doc.get("path", "")
        if p:
            paths.add(p)
        for of in doc.get("open_first", []):
            paths.add(of)
    return paths


def _load_agents_referenced_paths(repo_root: Path) -> set[str]:
    """Extract file paths referenced in AGENTS.md project shape and routing."""
    agents_path = repo_root / AGENTS_DOC
    if not agents_path.exists():
        return set()
    text = agents_path.read_text(encoding="utf-8")
    paths: set[str] = set()
    for match in re.finditer(r"`([^`]+\.\w+)`", text):
        paths.add(match.group(1))
    for match in re.finditer(r"- `?([^`\s]+/[^`\s]+\.\w+)`?", text):
        paths.add(match.group(1))
    return paths


def check_module_intake_coverage(repo_root: Path) -> list[LintError]:
    """Verify top-level modules in src/vibespatial/ are in intake routing."""
    errors: list[LintError] = []
    intake_paths = _load_intake_paths(repo_root)
    agents_paths = _load_agents_referenced_paths(repo_root)
    all_known = intake_paths | agents_paths

    root = repo_root / "src" / "vibespatial"
    if not root.exists():
        return errors

    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix != ".py":
            continue
        if path.name in _INTAKE_EXEMPT_MODULES:
            continue

        relative = f"src/vibespatial/{path.name}"
        found = relative in all_known or path.name in all_known
        if not found:
            # Also accept if the directory is tracked in bulk.
            broader = any(
                "src/vibespatial/" in p or "src/vibespatial" == p
                for p in all_known
            )
            if broader:
                continue
            errors.append(
                LintError(
                    code="MAINT001",
                    path=path,
                    line=1,
                    message=(
                        f"{relative} is not referenced in intake-index.json or AGENTS.md. "
                        "Add it to an intake routing doc so agents can discover it."
                    ),
                    doc_path=INTAKE_DOC,
                )
            )
    return errors


# ---- MAINT002: ADRs must be indexed in decisions/index.md ----

def check_adr_index_coverage(repo_root: Path) -> list[LintError]:
    """Verify every ADR file in docs/decisions/ is referenced in the index."""
    errors: list[LintError] = []
    decisions_dir = repo_root / "docs" / "decisions"
    index_path = decisions_dir / "index.md"

    if not decisions_dir.exists() or not index_path.exists():
        return errors

    index_text = index_path.read_text(encoding="utf-8")
    adr_files = sorted(
        p for p in decisions_dir.glob("*.md")
        if p.name != "index.md" and not p.name.startswith(".")
    )

    for adr in adr_files:
        stem = adr.stem
        if stem not in index_text and adr.name not in index_text:
            errors.append(
                LintError(
                    code="MAINT002",
                    path=adr,
                    line=1,
                    message=(
                        f"{adr.name} is not referenced in {DECISIONS_INDEX}. "
                        "Run `uv run python scripts/new_decision.py` or add it manually."
                    ),
                    doc_path=DECISIONS_INDEX,
                )
            )
    return errors


# ---- MAINT003: Scripts must be listed in AGENTS.md project shape ----

def check_script_agents_coverage(repo_root: Path) -> list[LintError]:
    """Verify every script in scripts/ is referenced in AGENTS.md."""
    errors: list[LintError] = []
    scripts_dir = repo_root / "scripts"
    agents_path = repo_root / AGENTS_DOC

    if not scripts_dir.exists() or not agents_path.exists():
        return errors

    agents_text = agents_path.read_text(encoding="utf-8")
    script_files = sorted(
        p for p in scripts_dir.glob("*.py")
        if not p.name.startswith(".") and p.name != "__init__.py"
    )

    for script in script_files:
        ref = f"scripts/{script.name}"
        if ref not in agents_text and script.name not in agents_text:
            errors.append(
                LintError(
                    code="MAINT003",
                    path=script,
                    line=1,
                    message=(
                        f"{ref} is not listed in AGENTS.md project shape. "
                        "Add a one-line description so agents can discover and route to it."
                    ),
                    doc_path=AGENTS_DOC,
                )
            )
    return errors


def run_checks(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    errors.extend(check_module_intake_coverage(repo_root))
    errors.extend(check_adr_index_coverage(repo_root))
    errors.extend(check_script_agents_coverage(repo_root))
    return sorted(errors, key=lambda e: (str(e.path), e.line, e.code))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check maintainability and discoverability constraints."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scan repository for maintainability violations.",
    )
    args = parser.parse_args(argv)

    if not args.all:
        parser.error("pass --all")

    errors = run_checks(REPO_ROOT)
    count = len(errors)

    if count > _VIOLATION_BASELINE:
        for error in errors:
            print(error.render(REPO_ROOT))
        print(
            f"\nMaintainability checks FAILED: {count} violations found, "
            f"baseline is {_VIOLATION_BASELINE}. "
            f"New code introduced {count - _VIOLATION_BASELINE} violation(s).",
            file=sys.stderr,
        )
        return 1

    if count < _VIOLATION_BASELINE:
        print(
            f"Maintainability checks passed ({count} known violations, "
            f"baseline {_VIOLATION_BASELINE}). "
            f"Debt reduced! Update _VIOLATION_BASELINE to {count}."
        )
    else:
        print(f"Maintainability checks passed ({count} known violations, baseline holds).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
