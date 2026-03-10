from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hooks_dir = repo_root / ".githooks"

    if not hooks_dir.exists():
        print("install_githooks: .githooks not found, skipping.")
        return

    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        print("install_githooks: not a git repository, skipping.")
        return

    current = subprocess.run(
        ["git", "config", "--local", "core.hooksPath"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    if current == ".githooks":
        print("install_githooks: core.hooksPath already set.")
        return

    source = current or "(default)"
    print(f"install_githooks: setting core.hooksPath from {source} to .githooks.")
    subprocess.run(
        ["git", "config", "--local", "core.hooksPath", ".githooks"],
        cwd=repo_root,
        check=True,
    )


if __name__ == "__main__":
    main()
