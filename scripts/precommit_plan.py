"""Pre-commit staged path planner for docs-only fast commits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

DOCS_ONLY_SCOPE = "docs-only"
FULL_SCOPE = "full"


@dataclass(frozen=True)
class PrecommitPlan:
    scope: str
    paths: tuple[str, ...]
    reason: str


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def is_docs_only_path(path: str) -> bool:
    normalized = _normalize_path(path)
    posix_path = PurePosixPath(normalized)

    if normalized.startswith("docs/"):
        return True
    if normalized == "docs/ops/intake-index.json":
        return True
    if posix_path.name in {"README.md", "AGENTS.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE"}:
        return True
    if posix_path.suffix.lower() in {".md", ".rst"}:
        return True

    return False


def classify_paths(paths: list[str] | tuple[str, ...]) -> PrecommitPlan:
    normalized_paths = tuple(_normalize_path(path) for path in paths if path)

    if os.environ.get("VIBESPATIAL_PRECOMMIT_FORCE_FULL") == "1":
        return PrecommitPlan(
            scope=FULL_SCOPE,
            paths=normalized_paths,
            reason="forced by VIBESPATIAL_PRECOMMIT_FORCE_FULL=1",
        )

    if not normalized_paths:
        return PrecommitPlan(
            scope=FULL_SCOPE,
            paths=normalized_paths,
            reason="no staged paths were detected",
        )

    full_check_paths = tuple(path for path in normalized_paths if not is_docs_only_path(path))
    if full_check_paths:
        return PrecommitPlan(
            scope=FULL_SCOPE,
            paths=normalized_paths,
            reason=f"non-doc staged path: {full_check_paths[0]}",
        )

    return PrecommitPlan(
        scope=DOCS_ONLY_SCOPE,
        paths=normalized_paths,
        reason="all staged paths are documentation or generated doc index artifacts",
    )


def staged_paths() -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        check=True,
        capture_output=True,
        text=False,
    )
    return tuple(path.decode("utf-8") for path in completed.stdout.split(b"\0") if path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify staged paths for the pre-commit hook.")
    parser.add_argument("--scope", action="store_true", help="Print only the selected scope.")
    parser.add_argument("--json", action="store_true", help="Print the full plan as JSON.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional paths to classify instead of reading the staged git diff.",
    )
    args = parser.parse_args(argv)

    plan = classify_paths(tuple(args.paths) if args.paths else staged_paths())

    if args.json:
        print(json.dumps(asdict(plan), indent=2))
    elif args.scope:
        print(plan.scope)
    else:
        print(f"{plan.scope}: {plan.reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
