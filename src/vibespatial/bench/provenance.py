"""Durable source identity for benchmark evidence."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any

_SOURCE_ROOTS = ("src", "scripts", "benchmarks")


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ("git", "-C", str(repository), *args),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def source_identity() -> dict[str, Any]:
    """Identify the imported package and its source-only worktree content."""
    import vibespatial

    source_file = Path(vibespatial.__file__).resolve()
    repository = source_file.parents[2]
    revision_result = _git(repository, "rev-parse", "HEAD")
    diff_result = _git(
        repository,
        "diff",
        "--no-ext-diff",
        "HEAD",
        "--",
        *_SOURCE_ROOTS,
    )
    untracked_result = _git(
        repository,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *_SOURCE_ROOTS,
    )
    untracked_source_files = (
        sorted(filter(None, untracked_result.stdout.splitlines()))
        if untracked_result is not None and untracked_result.returncode == 0
        else []
    )
    source_fingerprint = None
    tracked_source_dirty = None
    if diff_result is not None and diff_result.returncode == 0:
        tracked_source_dirty = bool(diff_result.stdout)
        digest = hashlib.sha256(diff_result.stdout.encode("utf-8"))
        for relative_path in untracked_source_files:
            path = repository / relative_path
            if not path.is_file():
                continue
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
        source_fingerprint = digest.hexdigest()
    return {
        "package_version": str(getattr(vibespatial, "__version__", "unknown")),
        "source_file": str(source_file),
        "git_revision": (
            revision_result.stdout.strip()
            if revision_result is not None and revision_result.returncode == 0
            else None
        ),
        "tracked_source_dirty": tracked_source_dirty,
        "untracked_source_files": untracked_source_files,
        "worktree_source_sha256": source_fingerprint,
        "python_version": platform.python_version(),
    }


__all__ = ["source_identity"]
