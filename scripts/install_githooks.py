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
    print("  1. ruff check")
    print("  2. check_docs.py --refresh + --check")
    print("  3. check_architecture_lints.py --all")
    print("  4. check_zero_copy.py --all")
    print("  5. check_perf_patterns.py --all")
    print("  6. check_maintainability.py --all")
    print("  7. check_import_guard.py --all")
    print("  8. health.py --tier contract --check")
    print("  9. health.py --tier gpu --check")
    print("  10. AI review reminder (interactive prompt)")
    print()
    print("AI-powered review skills (run in Codex):")
    print("  $pre-land-review  deterministic checks + AI review gate")
    print("  $gpu-code-review  targeted GPU review when kernel code changes")


if __name__ == "__main__":
    main()
