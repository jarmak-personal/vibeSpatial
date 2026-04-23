"""Install .githooks as the repo's git hooks directory.

Usage:
    uv run python scripts/install_githooks.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    hooks_dir = REPO_ROOT / ".githooks"
    if not hooks_dir.exists():
        print("install_githooks: .githooks not found, skipping.")
        return

    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        print("install_githooks: not a git repository, skipping.")
        return

    current = subprocess.run(
        ["git", "config", "--local", "core.hooksPath"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    if current == ".githooks":
        print("install_githooks: core.hooksPath already set.")
    else:
        source = current or "(default)"
        print(f"install_githooks: setting core.hooksPath from {source} to .githooks.")
        subprocess.run(
            ["git", "config", "--local", "core.hooksPath", ".githooks"],
            cwd=REPO_ROOT,
            check=True,
        )

    print()
    print("Pre-commit hook will run:")
    print("  1. staged path classification")
    print("  2. docs-only fast path: check_docs.py --refresh + --check")
    print("  3. docs-only fast path: check_maintainability.py --all")
    print("  4. full path: ruff check")
    print("  5. full path: check_docs.py --refresh + --check")
    print("  6. full path: check_architecture_lints.py --all")
    print("  7. full path: check_zero_copy.py --all")
    print("  8. full path: check_perf_patterns.py --all")
    print("  9. full path: check_maintainability.py --all")
    print("  10. full path: check_import_guard.py --all")
    print("  11. optional: cached contract/GPU health when VIBESPATIAL_PRECOMMIT_FORCE_GPU=1")
    print("  12. AI review reminder (interactive prompt)")
    print()
    print("Pre-push hook will run:")
    print("  1. cached health.py --tier contract --check")
    print("  2. cached health.py --tier gpu --check")
    print()
    print("AI-powered review skills (run in Codex):")
    print("  $pre-land-review  deterministic checks + AI review gate")
    print("  $gpu-code-review  targeted GPU review when kernel code changes")


if __name__ == "__main__":
    main()
