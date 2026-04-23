"""Cached contract/GPU health gate for pre-push and forced pre-commit runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .precommit_plan import is_docs_only_path
except ImportError:  # pragma: no cover - script execution path
    from precommit_plan import is_docs_only_path


HEALTH_COMMANDS = (
    ("uv", "run", "python", "scripts/health.py", "--tier", "contract", "--check"),
    (
        "uv",
        "run",
        "python",
        "scripts/health.py",
        "--tier",
        "gpu",
        "--check",
        "--gpu-coverage-timeout",
        "1200",
    ),
)
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class RefUpdate:
    local_ref: str
    local_oid: str
    remote_ref: str
    remote_oid: str


@dataclass(frozen=True)
class GatePlan:
    should_run: bool
    reason: str
    paths: tuple[str, ...]
    cache_key: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _is_zero_oid(oid: str) -> bool:
    return bool(oid) and set(oid) == {"0"}


def parse_pre_push_updates(text: str) -> tuple[RefUpdate, ...]:
    updates: list[RefUpdate] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 4:
            raise ValueError(f"invalid pre-push ref update line: {line!r}")
        updates.append(RefUpdate(*parts))
    return tuple(updates)


def _run_git_bytes(args: list[str], *, root: Path) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _run_git_text(args: list[str], *, root: Path) -> str:
    return _run_git_bytes(args, root=root).decode("utf-8").strip()


def _split_nul_paths(output: bytes) -> tuple[str, ...]:
    return tuple(path.decode("utf-8") for path in output.split(b"\0") if path)


def _paths_for_update(update: RefUpdate, *, root: Path) -> tuple[str, ...]:
    if _is_zero_oid(update.local_oid):
        return ()
    if _is_zero_oid(update.remote_oid):
        output = _run_git_bytes(
            [
                "diff-tree",
                "--root",
                "--no-commit-id",
                "--name-only",
                "-z",
                "-r",
                update.local_oid,
            ],
            root=root,
        )
        return _split_nul_paths(output)

    output = _run_git_bytes(
        ["diff", "--name-only", "-z", update.remote_oid, update.local_oid],
        root=root,
    )
    return _split_nul_paths(output)


def changed_paths_for_updates(updates: tuple[RefUpdate, ...], *, root: Path) -> tuple[str, ...]:
    paths: set[str] = set()
    for update in updates:
        paths.update(_normalize_path(path) for path in _paths_for_update(update, root=root))
    return tuple(sorted(paths))


def staged_changed_paths(*, root: Path) -> tuple[str, ...]:
    output = _run_git_bytes(["diff", "--cached", "--name-only", "-z"], root=root)
    return tuple(sorted(_normalize_path(path) for path in _split_nul_paths(output)))


def cache_key_for_updates(updates: tuple[RefUpdate, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"vibespatial-pre-push-gpu-gate-v1\0")
    for update in updates:
        digest.update(update.local_ref.encode("utf-8"))
        digest.update(b"\0")
        digest.update(update.local_oid.encode("utf-8"))
        digest.update(b"\0")
        digest.update(update.remote_ref.encode("utf-8"))
        digest.update(b"\0")
        digest.update(update.remote_oid.encode("utf-8"))
        digest.update(b"\0")
    return f"pre-push:{digest.hexdigest()}"


def staged_cache_key(*, root: Path) -> str:
    diff = _run_git_bytes(["diff", "--cached", "--binary"], root=root)
    digest = hashlib.sha256()
    digest.update(b"vibespatial-staged-gpu-gate-v1\0")
    digest.update(diff)
    return f"staged:{digest.hexdigest()}"


def plan_for_paths(paths: tuple[str, ...], cache_key: str, *, force: bool = False) -> GatePlan:
    normalized = tuple(sorted({_normalize_path(path) for path in paths if path}))

    if not normalized:
        return GatePlan(
            should_run=False,
            reason="no changed paths require contract/GPU health",
            paths=normalized,
            cache_key=cache_key,
        )

    if not force and all(is_docs_only_path(path) for path in normalized):
        return GatePlan(
            should_run=False,
            reason="all changed paths are documentation",
            paths=normalized,
            cache_key=cache_key,
        )

    reason = "forced by environment" if force else "changed paths require the heavy gate"
    return GatePlan(
        should_run=True,
        reason=reason,
        paths=normalized,
        cache_key=cache_key,
    )


def git_dir(*, root: Path) -> Path:
    raw_path = _run_git_text(["rev-parse", "--git-dir"], root=root)
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def cache_path(*, root: Path) -> Path:
    return git_dir(root=root) / "vibespatial-gpu-health-cache.json"


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"schema_version": 1, "entries": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
        return {"schema_version": 1, "entries": {}}
    return payload


def write_cache_entry(path: Path, *, key: str, label: str, now: float) -> None:
    payload = load_cache(path)
    entries = payload.setdefault("entries", {})
    entries[key] = {
        "status": "passed",
        "timestamp": int(now),
        "label": label,
        "commands": [shlex.join(command) for command in HEALTH_COMMANDS],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cache_hit(path: Path, *, key: str, now: float, ttl_seconds: int) -> bool:
    if os.environ.get("VIBESPATIAL_GPU_GATE_IGNORE_CACHE") == "1":
        return False
    if ttl_seconds <= 0:
        return False
    payload = load_cache(path)
    entry = payload.get("entries", {}).get(key)
    if not isinstance(entry, dict) or entry.get("status") != "passed":
        return False
    timestamp = int(entry.get("timestamp", 0))
    return now - timestamp <= ttl_seconds


def cache_ttl_seconds() -> int:
    raw_value = os.environ.get("VIBESPATIAL_GPU_GATE_CACHE_TTL_SECONDS")
    if raw_value is None:
        return DEFAULT_CACHE_TTL_SECONDS
    try:
        return int(raw_value)
    except ValueError:
        return DEFAULT_CACHE_TTL_SECONDS


def run_health_commands(*, root: Path) -> int:
    for command in HEALTH_COMMANDS:
        print(f"gpu-health-gate: running {shlex.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def run_gate(plan: GatePlan, *, root: Path, label: str) -> int:
    print(f"{label}: {plan.reason}")
    if not plan.should_run:
        return 0

    now = time.time()
    path = cache_path(root=root)
    ttl = cache_ttl_seconds()
    if cache_hit(path, key=plan.cache_key, now=now, ttl_seconds=ttl):
        print(f"{label}: contract/GPU health already passed for this diff; skipping.")
        return 0

    returncode = run_health_commands(root=root)
    if returncode != 0:
        return returncode

    write_cache_entry(path, key=plan.cache_key, label=label, now=now)
    print(f"{label}: contract/GPU health passed and cache was updated.")
    return 0


def _pre_push_plan(*, root: Path) -> GatePlan:
    updates = parse_pre_push_updates(sys.stdin.read())
    paths = changed_paths_for_updates(updates, root=root)
    force = os.environ.get("VIBESPATIAL_PUSH_FORCE_GPU") == "1"
    return plan_for_paths(paths, cache_key_for_updates(updates), force=force)


def _staged_plan(*, root: Path, force: bool) -> GatePlan:
    return plan_for_paths(staged_changed_paths(root=root), staged_cache_key(root=root), force=force)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cached contract/GPU health gates.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pre-push", action="store_true", help="Read pre-push updates from stdin.")
    mode.add_argument("--staged", action="store_true", help="Use the staged diff as the cache key.")
    parser.add_argument("--force", action="store_true", help="Run even when paths are docs-only.")
    args = parser.parse_args(argv)

    root = repo_root()
    if args.pre_push:
        plan = _pre_push_plan(root=root)
        return run_gate(plan, root=root, label="pre-push")

    plan = _staged_plan(root=root, force=args.force)
    return run_gate(plan, root=root, label="pre-commit")


if __name__ == "__main__":
    raise SystemExit(main())
