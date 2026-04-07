from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

try:
    from .build_intake_index import build_intake_index, tokenize
except ImportError:  # pragma: no cover - script execution path
    from build_intake_index import build_intake_index, tokenize


def _score_signal(request_text: str, request_tokens: set[str], text: str, weight: int) -> tuple[int, set[str]]:
    lowered = text.lower()
    tokens = set(tokenize(text))
    matches: set[str] = set()
    score = 0

    if lowered and lowered in request_text:
        score += weight
        matches.add(lowered)

    overlap = request_tokens & tokens
    if overlap:
        score += len(overlap) * max(1, weight // 2)
        matches.update(overlap)

    return score, matches


def _score_doc(request_text: str, request_tokens: set[str], doc: dict[str, Any]) -> dict[str, Any]:
    score = 0
    matches: set[str] = set()

    for signal in doc["request_signals"]:
        delta, delta_matches = _score_signal(request_text, request_tokens, signal, 8)
        score += delta
        matches.update(delta_matches)

    metadata = doc["metadata"]
    for field in ("scope", "readIf", "sourceOfTruth"):
        delta, delta_matches = _score_signal(request_text, request_tokens, metadata[field], 4)
        score += delta
        matches.update(delta_matches)

    delta, delta_matches = _score_signal(request_text, request_tokens, doc["title"], 3)
    score += delta
    matches.update(delta_matches)

    delta, delta_matches = _score_signal(request_text, request_tokens, doc["path"], 2)
    score += delta
    matches.update(delta_matches)

    return {
        "path": doc["path"],
        "title": doc["title"],
        "score": score,
        "matches": sorted(matches),
        "open_first": doc["open_first"],
        "verify": doc["verify"],
        "risks": doc["risks"],
    }


def _score_file(
    request_text: str,
    request_tokens: set[str],
    file_entry: dict[str, Any],
    doc_scores: dict[str, int],
) -> dict[str, Any]:
    direct_score = 0
    matches: set[str] = set()

    delta, delta_matches = _score_signal(request_text, request_tokens, file_entry["path"], 3)
    direct_score += delta
    matches.update(delta_matches)

    token_overlap = request_tokens & set(file_entry["tokens"])
    if token_overlap:
        direct_score += len(token_overlap) * 2
        matches.update(token_overlap)

    doc_boost = sum(doc_scores.get(path, 0) for path in file_entry["referenced_by"])
    score = direct_score + (doc_boost * 2)

    return {
        "path": file_entry["path"],
        "kind": file_entry["kind"],
        "score": score,
        "direct_score": direct_score,
        "matches": sorted(matches),
        "referenced_by": file_entry["referenced_by"],
    }


def _confidence(top_score: int) -> str:
    if top_score >= 18:
        return "high"
    if top_score >= 8:
        return "medium"
    return "low"


MAX_EXPLICIT_FILE_SEEDS = 4


def plan_request(request: str) -> dict[str, Any]:
    index = build_intake_index()
    request_text = request.lower()
    request_tokens = set(tokenize(request))

    doc_scores = [_score_doc(request_text, request_tokens, doc) for doc in index["docs"]]
    doc_scores.sort(key=lambda item: (-item["score"], item["path"]))

    if not doc_scores:
        raise ValueError("Intake index contains no routable docs.")

    if doc_scores[0]["score"] <= 0:
        fallback = next(
            (
                doc
                for doc in doc_scores
                if doc["path"] in {"AGENTS.md", "docs/ops/intake.md"}
            ),
            doc_scores[0],
        )
        doc_scores = [fallback, *[doc for doc in doc_scores if doc["path"] != fallback["path"]]]

    selected_docs = [doc for doc in doc_scores[:3] if doc["score"] > 0] or doc_scores[:1]
    selected_doc_scores = {doc["path"]: doc["score"] for doc in selected_docs}

    explicit_file_paths: list[str] = []
    for doc in selected_docs:
        explicit_file_paths.extend(doc["open_first"])

    file_lookup = {entry["path"]: entry for entry in index["files"]}
    file_scores: list[dict[str, Any]] = []
    for entry in index["files"]:
        if entry["path"].endswith(".md"):
            continue
        scored = _score_file(request_text, request_tokens, entry, selected_doc_scores)
        if scored["score"] > 0 or entry["path"] in explicit_file_paths:
            if entry["path"] in explicit_file_paths:
                boost = max(
                    (
                        selected_doc_scores.get(doc["path"], 0)
                        for doc in selected_docs
                        if entry["path"] in doc["open_first"]
                    ),
                    default=0,
                )
                scored["score"] += boost * 3
            file_scores.append(scored)

    file_scores.sort(key=lambda item: (-item["score"], item["path"]))

    seen_paths: set[str] = set()
    selected_files: list[dict[str, Any]] = []
    direct_match_paths = [
        entry["path"]
        for entry in sorted(
            (entry for entry in file_scores if entry["matches"]),
            key=lambda item: (-item["direct_score"], -item["score"], item["path"]),
        )
    ]
    explicit_seed_paths = explicit_file_paths[:MAX_EXPLICIT_FILE_SEEDS]
    remaining_explicit_paths = explicit_file_paths[MAX_EXPLICIT_FILE_SEEDS:]
    ordered_paths = (
        explicit_seed_paths
        + direct_match_paths
        + remaining_explicit_paths
        + [entry["path"] for entry in file_scores]
    )
    for path in ordered_paths:
        if path in selected_doc_scores:
            continue
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if path in file_lookup:
            selected_files.append(
                next((entry for entry in file_scores if entry["path"] == path), None)
                or {
                    "path": path,
                    "kind": file_lookup[path]["kind"],
                    "score": 0,
                    "matches": [],
                    "referenced_by": file_lookup[path]["referenced_by"],
                }
            )
        if len(selected_files) >= 8:
            break

    verify_counts: dict[str, int] = defaultdict(int)
    risk_counts: dict[str, int] = defaultdict(int)
    for doc in selected_docs:
        for command in doc["verify"]:
            verify_counts[command] += 1
        for risk in doc["risks"]:
            risk_counts[risk] += 1

    verify = sorted(verify_counts, key=lambda item: (-verify_counts[item], item))
    risks = sorted(risk_counts, key=lambda item: (-risk_counts[item], item))

    return {
        "request": request,
        "confidence": _confidence(selected_docs[0]["score"]),
        "docs": selected_docs,
        "files": selected_files,
        "verify": verify,
        "risks": risks,
    }


def print_plan(plan: dict[str, Any]) -> None:
    print(f"request: {plan['request']}")
    print(f"confidence: {plan['confidence']}")
    print("docs:")
    for doc in plan["docs"]:
        detail = f"matched: {', '.join(doc['matches'])}" if doc["matches"] else "matched: none"
        print(f"- {doc['path']} ({detail})")
    print("files:")
    for file_entry in plan["files"]:
        if file_entry["referenced_by"]:
            detail = f"from {', '.join(file_entry['referenced_by'])}"
        elif file_entry["matches"]:
            detail = f"matched: {', '.join(file_entry['matches'])}"
        else:
            detail = "matched: none"
        print(f"- {file_entry['path']} ({detail})")
    print("verify:")
    for command in plan["verify"]:
        print(f"- {command}")
    print("risks:")
    for risk in plan["risks"]:
        print(f"- {risk}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a request to the first files to inspect.")
    parser.add_argument("request", help="Natural language task description")
    args = parser.parse_args()
    print_plan(plan_request(args.request))


if __name__ == "__main__":
    main()
