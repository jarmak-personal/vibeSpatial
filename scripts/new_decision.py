from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


DECISIONS_DIR = Path("docs/decisions")
INDEX_PATH = DECISIONS_DIR / "index.md"
INDEX_MARKER = "<!-- DECISION_ROWS -->"


@dataclass(frozen=True)
class DecisionRecord:
    id: str
    numeric_id: int
    status: str
    date: str
    deciders: tuple[str, ...]
    tags: tuple[str, ...]
    title: str
    path: Path


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "decision"


def next_decision_number(root: Path) -> int:
    numbers = []
    decisions_root = root / DECISIONS_DIR
    if decisions_root.exists():
        for path in decisions_root.glob("[0-9][0-9][0-9][0-9]-*.md"):
            numbers.append(int(path.name.split("-", 1)[0]))
    return max(numbers, default=0) + 1


def render_decision(*, title: str, decision_number: int, decision_date: str) -> str:
    decision_id = f"ADR-{decision_number:04d}"
    return f"""---
id: {decision_id}
status: accepted
date: {decision_date}
deciders:
  - vibeSpatial maintainers
tags:
  - architecture
---

# {title}

## Context

Document the design context here.

## Decision

Summarize the decision.

## Consequences

- Record the direct tradeoffs the repo now accepts.

## Alternatives Considered

- Note the main rejected alternatives and why they lost.
"""


def parse_frontmatter(content: str) -> dict[str, Any]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing YAML frontmatter start")
    metadata: dict[str, Any] = {}
    current_key: str | None = None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if line.startswith("  - ") and current_key is not None:
            metadata.setdefault(current_key, []).append(line.split("- ", 1)[1].strip())
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current_key = key.strip()
        raw_value = value.strip()
        metadata[current_key] = raw_value if raw_value else []
    return metadata


def load_decisions(root: Path) -> list[DecisionRecord]:
    decisions: list[DecisionRecord] = []
    decisions_root = root / DECISIONS_DIR
    if not decisions_root.exists():
        return decisions
    for path in sorted(decisions_root.glob("[0-9][0-9][0-9][0-9]-*.md")):
        content = path.read_text(encoding="utf-8")
        metadata = parse_frontmatter(content)
        title_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem
        decisions.append(
            DecisionRecord(
                id=str(metadata["id"]),
                numeric_id=int(str(metadata["id"]).split("-")[-1]),
                status=str(metadata["status"]),
                date=str(metadata["date"]),
                deciders=tuple(metadata.get("deciders", [])),
                tags=tuple(metadata.get("tags", [])),
                title=title,
                path=path.relative_to(root),
            )
        )
    return sorted(decisions, key=lambda item: item.numeric_id)


def render_index(decisions: list[DecisionRecord]) -> str:
    open_first = [
        "- docs/decisions/index.md",
        "- scripts/new_decision.py",
        "- docs/architecture/runtime.md",
    ]
    for decision in decisions[:3]:
        open_first.append(f"- {decision.path.as_posix()}")

    rows = [
        f"| `{decision.id}` | {decision.status} | {decision.date} | "
        f"[{decision.title}]({decision.path.as_posix()}) |"
        for decision in decisions
    ]

    return "\n".join(
        [
            "# Decision Log",
            "",
            "Use this index to find accepted architecture decisions.",
            "",
            "## Intent",
            "",
            "Track architecture decisions in a stable, agent-discoverable format.",
            "",
            "## Request Signals",
            "",
            "- adr",
            "- decision log",
            "- architecture decision",
            "- design decision",
            "- superseded design",
            "",
            "## Open First",
            "",
            *open_first,
            "",
            "## Verify",
            "",
            "- `uv run pytest tests/test_decision_log.py`",
            "- `uv run python scripts/check_docs.py --check`",
            "",
            "## Risks",
            "",
            "- Decisions can drift from implemented code if follow-up changes do not update the log.",
            "- Buried or unindexed ADRs make agents re-litigate settled design choices.",
            "",
            "## Decisions",
            "",
            "| ADR | Status | Date | Title |",
            "|---|---|---|---|",
            INDEX_MARKER,
            *rows,
            "",
        ]
    )


def write_index(root: Path) -> Path:
    decisions = load_decisions(root)
    output = root / INDEX_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_index(decisions), encoding="utf-8")
    return output


def create_decision(root: Path, title: str, slug: str, *, decision_date: str | None = None) -> Path:
    number = next_decision_number(root)
    resolved_date = decision_date or date.today().isoformat()
    output = root / DECISIONS_DIR / f"{number:04d}-{slugify(slug)}.md"
    if output.exists():
        raise FileExistsError(f"decision already exists: {output.relative_to(root)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_decision(title=title, decision_number=number, decision_date=resolved_date),
        encoding="utf-8",
    )
    write_index(root)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a new ADR and refresh the decision index.")
    parser.add_argument("title", nargs="?", help="Title for the new decision record.")
    parser.add_argument("slug", nargs="?", help="URL-friendly slug for the filename.")
    parser.add_argument("--date", dest="decision_date")
    parser.add_argument("--refresh-index", action="store_true")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    if args.refresh_index:
        output = write_index(root)
        print(output.relative_to(root))
        return 0

    if not args.title or not args.slug:
        parser.error("title and slug are required unless --refresh-index is used")

    output = create_decision(root, args.title, args.slug, decision_date=args.decision_date)
    print(output.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
