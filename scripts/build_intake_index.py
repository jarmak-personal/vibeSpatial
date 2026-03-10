from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from .update_doc_headers import get_repo_root, load_config, strip_generated_header
except ImportError:  # pragma: no cover - script execution path
    from update_doc_headers import get_repo_root, load_config, strip_generated_header


INDEX_OUTPUT = Path("docs/ops/intake-index.json")
TEXT_SUFFIXES = {".md", ".py", ".toml", ".json", ".yaml", ".yml"}
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "path",
    "paths",
    "the",
    "this",
    "to",
    "up",
    "use",
    "with",
    "file",
    "files",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def parse_markdown_sections(content: str) -> dict[str, str]:
    lines = content.splitlines()
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in lines:
        heading = re.match(r"^##\s+(.+)$", line)
        if heading:
            current = heading.group(1).strip()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)

    return {name: "\n".join(body).strip() for name, body in sections.items()}


def clean_list_item(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("`") and cleaned.endswith("`") and len(cleaned) > 1:
        cleaned = cleaned[1:-1]
    return cleaned


def parse_list_section(content: str) -> list[str]:
    items: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            items.append(clean_list_item(line[2:]))
            continue
        numbered = re.match(r"^\d+\.\s+(.*)$", line)
        if numbered:
            items.append(clean_list_item(numbered.group(1)))
    return items


def parse_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def clean_path_candidate(value: str) -> str:
    return value.split(" (", 1)[0].strip("` ")


def path_exists(root: Path, relative_path: str) -> bool:
    return (root / relative_path).exists()


def collect_python_symbols(content: str) -> list[str]:
    symbols = re.findall(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", content, flags=re.MULTILINE)
    return symbols[:24]


def classify_file(relative_path: str) -> str:
    if relative_path.startswith("docs/"):
        return "doc"
    if relative_path.startswith("scripts/"):
        return "script"
    if relative_path.startswith("tests/upstream/"):
        return "upstream-test"
    if relative_path.startswith("tests/"):
        return "test"
    if relative_path.startswith("src/vibespatial/_vendor/"):
        return "vendor"
    if relative_path.startswith("src/"):
        return "source"
    return "repo"


def should_index_file(path: Path, root: Path) -> bool:
    if path.is_dir():
        return False
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False
    if path.suffix not in TEXT_SUFFIXES:
        return False

    relative_path = path.relative_to(root).as_posix()
    if relative_path == INDEX_OUTPUT.as_posix():
        return False
    if "/data/" in relative_path and path.suffix != ".md":
        return False
    if "baseline_images" in relative_path:
        return False
    return True


def build_doc_entries(root: Path) -> list[dict[str, Any]]:
    config = load_config(root)
    entries: list[dict[str, Any]] = []

    for relative_path, metadata in sorted(config["files"].items()):
        content = strip_generated_header((root / relative_path).read_text())
        sections = parse_markdown_sections(content)
        request_signals = parse_list_section(sections.get("Request Signals", ""))
        open_first = [
            clean_path_candidate(item)
            for item in parse_list_section(sections.get("Open First", ""))
            if path_exists(root, clean_path_candidate(item))
        ]
        verify = parse_list_section(sections.get("Verify", ""))
        risks = parse_list_section(sections.get("Risks", ""))
        title = parse_title(content, Path(relative_path).stem.replace("_", " ").title())

        token_source = " ".join(
            [
                title,
                metadata["scope"],
                metadata["readIf"],
                metadata["stopIf"],
                metadata["sourceOfTruth"],
                " ".join(request_signals),
                " ".join(sections.keys()),
            ]
        )

        entries.append(
            {
                "path": relative_path,
                "title": title,
                "metadata": metadata,
                "request_signals": request_signals,
                "open_first": open_first,
                "verify": verify,
                "risks": risks,
                "tokens": sorted(set(tokenize(token_source))),
                "sections": sorted(sections.keys()),
            }
        )

    return entries


def build_file_entries(root: Path, doc_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    referenced_by: dict[str, set[str]] = {}
    for doc in doc_entries:
        for relative_path in doc["open_first"]:
            referenced_by.setdefault(relative_path, set()).add(doc["path"])

    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not should_index_file(path, root):
            continue

        relative_path = path.relative_to(root).as_posix()
        content = path.read_text() if path.suffix == ".py" else ""
        content_tokens = collect_python_symbols(content) if content else []
        token_source = " ".join([relative_path, *content_tokens])

        entries.append(
            {
                "path": relative_path,
                "kind": classify_file(relative_path),
                "tokens": sorted(set(tokenize(token_source))),
                "symbols": content_tokens,
                "referenced_by": sorted(referenced_by.get(relative_path, set())),
            }
        )

    return entries


def build_intake_index(root: Path | None = None) -> dict[str, Any]:
    repo_root = root or get_repo_root()
    doc_entries = build_doc_entries(repo_root)
    file_entries = build_file_entries(repo_root, doc_entries)
    return {
        "version": 2,
        "docs": doc_entries,
        "files": file_entries,
    }


def render_index(index: dict[str, Any]) -> str:
    return json.dumps(index, indent=2, sort_keys=True) + "\n"


def evaluate_intake_index(root: Path | None = None, *, write: bool = False) -> dict[str, Any]:
    repo_root = root or get_repo_root()
    output_path = repo_root / INDEX_OUTPUT
    rendered = render_index(build_intake_index(repo_root))
    original = output_path.read_text() if output_path.exists() else None

    report = {
        "path": INDEX_OUTPUT.as_posix(),
        "updated": [],
        "outdated": [],
    }

    if original != rendered:
        report["outdated"].append(INDEX_OUTPUT.as_posix())
        if write:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered)
            report["updated"].append(INDEX_OUTPUT.as_posix())

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or validate the generated intake index.")
    parser.add_argument("--check", action="store_true", help="Fail when the generated index is stale.")
    args = parser.parse_args()

    report = evaluate_intake_index(write=not args.check)
    if args.check and report["outdated"]:
        print("Outdated generated intake index:")
        for path in report["outdated"]:
            print(f"- {path}")
        raise SystemExit(1)

    if args.check:
        print("Generated intake index is up to date.")
        return

    if report["updated"]:
        print("Updated intake index:")
        for path in report["updated"]:
            print(f"- {path}")
    else:
        print("Generated intake index already up to date.")


if __name__ == "__main__":
    main()
