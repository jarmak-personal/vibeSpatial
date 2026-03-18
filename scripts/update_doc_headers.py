from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HEADER_START = "<!-- DOC_HEADER:START"
HEADER_END = "DOC_HEADER:END -->"
REQUIRED_METADATA_FIELDS = ("scope", "readIf", "stopIf", "sourceOfTruth", "maxBodyLines")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_config_path(root: Path | None = None) -> Path:
    repo_root = root or get_repo_root()
    return repo_root / "docs" / "doc_headers.json"


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n")


def strip_generated_header(content: str) -> str:
    normalized = normalize_newlines(content)
    pattern = re.compile(
        rf"{re.escape(HEADER_START)}[\s\S]*?{re.escape(HEADER_END)}\n*",
        flags=re.MULTILINE,
    )
    stripped = pattern.sub("", normalized)
    compacted = re.sub(r"\n{3,}", "\n\n", stripped)
    return compacted.rstrip() + "\n"


def to_lines(content: str) -> list[str]:
    trimmed = content[:-1] if content.endswith("\n") else content
    return [] if not trimmed else trimmed.split("\n")


def count_body_lines(content: str) -> int:
    return len(to_lines(content))


def parse_sections(content: str) -> list[dict[str, Any]]:
    lines = to_lines(content)
    headings: list[tuple[str, int]] = []

    for index, line in enumerate(lines, start=1):
        match = re.match(r"^##\s+(.+)$", line)
        if match:
            headings.append((match.group(1).strip(), index))

    if not headings:
        return [{"title": "Document", "start": 1, "end": len(lines)}] if lines else []

    sections: list[dict[str, Any]] = []
    if headings[0][1] > 1:
        sections.append({"title": "Preamble", "start": 1, "end": headings[0][1] - 1})

    for position, (title, start) in enumerate(headings):
        next_start = headings[position + 1][1] - 1 if position + 1 < len(headings) else len(lines)
        sections.append({"title": title, "start": start, "end": next_start})

    return sections


def format_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def build_section_table(sections: list[dict[str, Any]], max_rows: int) -> str:
    if not sections:
        return "_No section map available._"

    visible = sections[:max_rows]
    rows = []
    for section in visible:
        title = section["title"].replace("|", "\\|")
        rows.append(f"| {format_range(section['start'], section['end'])} | {title} |")

    hidden = len(sections) - len(visible)
    if hidden > 0:
        rows.append(f"| ... | ({hidden} additional sections omitted; open document body for full map) |")

    return "| Body Lines | Section |\n|---|---|\n" + "\n".join(rows)


def build_header(
    relative_path: str,
    metadata: dict[str, Any],
    body_line_count: int,
    sections: list[dict[str, Any]],
) -> str:
    max_rows = int(metadata.get("maxSectionRows", 14))
    return "\n".join(
        [
            HEADER_START,
            f"Scope: {metadata['scope']}",
            f"Read If: {metadata['readIf']}",
            f"STOP IF: {metadata['stopIf']}",
            f"Source Of Truth: {metadata['sourceOfTruth']}",
            f"Body Budget: {body_line_count}/{metadata['maxBodyLines']} lines",
            f"Document: {relative_path}",
            "",
            "Section Map (Body Lines)",
            build_section_table(sections, max_rows),
            HEADER_END,
        ]
    )


def inject_header(body_content: str, header: str) -> str:
    normalized = normalize_newlines(body_content).rstrip() + "\n"
    lines = to_lines(normalized)
    if not lines:
        return f"{header}\n"

    first_line = lines[0]
    remainder = "\n".join(lines[1:]).lstrip("\n")
    if first_line.startswith("# "):
        if not remainder:
            return f"{first_line}\n\n{header}\n"
        return f"{first_line}\n\n{header}\n\n{remainder.rstrip()}\n"
    return f"{header}\n\n{normalized}"


def load_config(root: Path | None = None) -> dict[str, Any]:
    config_path = get_config_path(root)
    parsed = json.loads(config_path.read_text())
    files = parsed.get("files")
    if not isinstance(files, dict):
        raise ValueError("Invalid docs/doc_headers.json format. Expected {\"files\": {...}}.")
    return parsed


def validate_metadata(relative_path: str, metadata: dict[str, Any]) -> None:
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            raise ValueError(f'{relative_path}: missing required metadata field "{field}"')
    if not isinstance(metadata["maxBodyLines"], int) or metadata["maxBodyLines"] <= 0:
        raise ValueError(f"{relative_path}: maxBodyLines must be a positive integer.")


def evaluate_doc_headers(root: Path | None = None, *, write: bool = False) -> dict[str, Any]:
    repo_root = root or get_repo_root()
    config = load_config(repo_root)
    report: dict[str, Any] = {
        "checked": 0,
        "updated": [],
        "outdated": [],
        "over_budget": [],
    }

    for relative_path, metadata in sorted(config["files"].items()):
        validate_metadata(relative_path, metadata)
        absolute_path = repo_root / relative_path
        if not absolute_path.exists():
            raise FileNotFoundError(f"Configured doc file does not exist: {relative_path}")

        original = normalize_newlines(absolute_path.read_text())
        stripped = strip_generated_header(original)
        body_line_count = count_body_lines(stripped)
        sections = parse_sections(stripped)
        header = build_header(relative_path, metadata, body_line_count, sections)
        rendered = inject_header(stripped, header)

        report["checked"] += 1

        if original != rendered:
            report["outdated"].append(relative_path)
            if write:
                absolute_path.write_text(rendered)
                report["updated"].append(relative_path)

        if body_line_count > metadata["maxBodyLines"]:
            report["over_budget"].append(
                {
                    "file": relative_path,
                    "current": body_line_count,
                    "budget": metadata["maxBodyLines"],
                }
            )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh or validate generated markdown headers.")
    parser.add_argument("--check", action="store_true", help="Fail when headers are stale.")
    args = parser.parse_args()

    report = evaluate_doc_headers(write=not args.check)
    if args.check and (report["outdated"] or report["over_budget"]):
        if report["outdated"]:
            print("Outdated generated headers:")
            for path in report["outdated"]:
                print(f"- {path}")
        if report["over_budget"]:
            print("Docs over body line budget:")
            for violation in report["over_budget"]:
                print(f"- {violation['file']}: {violation['current']}/{violation['budget']}")
        raise SystemExit(1)

    if args.check:
        print(f"Doc headers are up to date ({report['checked']} files).")
        return

    if report["updated"]:
        print(f"Updated doc headers ({len(report['updated'])} files):")
        for path in report["updated"]:
            print(f"- {path}")
    else:
        print(f"Doc headers already up to date ({report['checked']} files).")

    if report["over_budget"]:
        print("Body line budget warnings:")
        for violation in report["over_budget"]:
            print(f"- {violation['file']}: {violation['current']}/{violation['budget']}")


if __name__ == "__main__":
    main()
