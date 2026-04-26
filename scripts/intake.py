from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .build_intake_index import build_intake_index, tokenize
except ImportError:  # pragma: no cover - script execution path
    from build_intake_index import build_intake_index, tokenize


COMMON_ROUTE_TOKENS = {
    "audit",
    "benchmark",
    "benchmarks",
    "check",
    "doc",
    "docs",
    "fix",
    "geometries",
    "geometry",
    "gpu",
    "improve",
    "investigate",
    "kernel",
    "kernels",
    "new",
    "performance",
    "plan",
    "repo",
    "review",
    "test",
    "tests",
    "update",
}
INTENT_DOC_BOOST = 32


def _overlap_score(tokens: set[str], base_score: int) -> int:
    score = 0
    for token in tokens:
        score += 1 if token in COMMON_ROUTE_TOKENS else base_score
    return score


def _score_signal(
    request_text: str,
    request_tokens: set[str],
    text: str,
    weight: int,
    *,
    seen_tokens: set[str] | None = None,
) -> tuple[int, set[str]]:
    lowered = text.lower()
    tokens = set(tokenize(text))
    matches: set[str] = set()
    score = 0

    if lowered and lowered in request_text:
        score += weight
        matches.add(lowered)

    overlap = request_tokens & tokens
    if overlap:
        scoring_overlap = overlap - seen_tokens if seen_tokens is not None else overlap
        score += _overlap_score(scoring_overlap, max(1, weight // 2))
        if seen_tokens is not None:
            seen_tokens.update(overlap)
        matches.update(overlap)

    return score, matches


def _record_detail(
    details: list[dict[str, Any]],
    source: str,
    score: int,
    matches: set[str],
) -> None:
    if score <= 0:
        return
    details.append({"source": source, "score": score, "matches": sorted(matches)})


def _intent_doc_boosts(request_text: str, request_tokens: set[str]) -> dict[str, int]:
    boosts: dict[str, int] = {}

    if (
        "adr" in request_tokens
        or "decision record" in request_text
        or "decision log" in request_text
        or "architecture decision" in request_text
    ):
        boosts["docs/decisions/index.md"] = INTENT_DOC_BOOST

    if "intake" in request_tokens or "router" in request_tokens:
        boosts["docs/ops/intake.md"] = INTENT_DOC_BOOST

    if "kernel" in request_tokens and bool(
        {"add", "create", "generate", "new", "scaffold", "write"} & request_tokens
    ):
        boosts["docs/testing/kernel-inventory.md"] = INTENT_DOC_BOOST

    if "precommit" in request_tokens or "pre-commit" in request_text:
        boosts["docs/dev/precommit.md"] = INTENT_DOC_BOOST

    return boosts


def _score_doc(
    request_text: str,
    request_tokens: set[str],
    doc: dict[str, Any],
    intent_boosts: dict[str, int],
) -> dict[str, Any]:
    score = 0
    matches: set[str] = set()
    details: list[dict[str, Any]] = []

    seen_signal_tokens: set[str] = set()
    for signal in doc["request_signals"]:
        delta, delta_matches = _score_signal(
            request_text,
            request_tokens,
            signal,
            8,
            seen_tokens=seen_signal_tokens,
        )
        score += delta
        matches.update(delta_matches)
        _record_detail(details, f"request_signal:{signal}", delta, delta_matches)

    metadata = doc["metadata"]
    seen_metadata_tokens: set[str] = set()
    for field in ("scope", "readIf", "sourceOfTruth"):
        delta, delta_matches = _score_signal(
            request_text,
            request_tokens,
            metadata[field],
            4,
            seen_tokens=seen_metadata_tokens,
        )
        score += delta
        matches.update(delta_matches)
        _record_detail(details, f"metadata:{field}", delta, delta_matches)

    delta, delta_matches = _score_signal(request_text, request_tokens, doc["title"], 3)
    score += delta
    matches.update(delta_matches)
    _record_detail(details, "title", delta, delta_matches)

    delta, delta_matches = _score_signal(request_text, request_tokens, doc["path"], 2)
    score += delta
    matches.update(delta_matches)
    _record_detail(details, "path", delta, delta_matches)

    boost = intent_boosts.get(doc["path"], 0)
    if boost:
        score += boost
        matches.add("intent")
        _record_detail(details, "intent", boost, {"intent"})

    return {
        "path": doc["path"],
        "title": doc["title"],
        "score": score,
        "matches": sorted(matches),
        "score_details": details,
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
    details: list[dict[str, Any]] = []

    delta, delta_matches = _score_signal(request_text, request_tokens, file_entry["path"], 3)
    direct_score += delta
    matches.update(delta_matches)
    _record_detail(details, "path", delta, delta_matches)

    token_overlap = request_tokens & set(file_entry["tokens"])
    if token_overlap:
        token_score = _overlap_score(token_overlap, 2)
        direct_score += token_score
        matches.update(token_overlap)
        _record_detail(details, "symbols_docstring", token_score, token_overlap)

    doc_boost = sum(doc_scores.get(path, 0) for path in file_entry["referenced_by"])
    score = direct_score + (doc_boost * 2)
    if doc_boost:
        _record_detail(details, "referenced_by_docs", doc_boost * 2, set(file_entry["referenced_by"]))

    return {
        "path": file_entry["path"],
        "kind": file_entry["kind"],
        "score": score,
        "direct_score": direct_score,
        "matches": sorted(matches),
        "score_details": details,
        "referenced_by": file_entry["referenced_by"],
    }


def _confidence(top_score: int) -> str:
    if top_score >= 18:
        return "high"
    if top_score >= 8:
        return "medium"
    return "low"


MAX_EXPLICIT_FILE_SEEDS = 8
HIGH_DIRECT_FILE_SCORE = 8
MAX_HIGH_DIRECT_FILE_SEEDS = 3
MAX_DIRECT_MARKDOWN_DOCS = 2
SPECIFIC_DIRECT_FILE_SCORE = 5
MIN_SPECIFIC_DIRECT_MATCHES = 2


def _specific_direct_match_count(file_score: dict[str, Any]) -> int:
    return len(set(file_score["matches"]) - COMMON_ROUTE_TOKENS)


def _is_high_direct_file_seed(
    file_score: dict[str, Any],
    *,
    allow_specific_promotion: bool = True,
) -> bool:
    if file_score["direct_score"] >= HIGH_DIRECT_FILE_SCORE:
        return True
    if not allow_specific_promotion:
        return False
    return (
        file_score["direct_score"] >= SPECIFIC_DIRECT_FILE_SCORE
        and _specific_direct_match_count(file_score) >= MIN_SPECIFIC_DIRECT_MATCHES
    )


def plan_request(request: str) -> dict[str, Any]:
    index = build_intake_index()
    request_text = request.lower()
    request_tokens = set(tokenize(request))
    intent_boosts = _intent_doc_boosts(request_text, request_tokens)

    doc_scores = [_score_doc(request_text, request_tokens, doc, intent_boosts) for doc in index["docs"]]
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

    related_docs: list[dict[str, Any]] = []
    for entry in index["files"]:
        if not entry["path"].endswith(".md"):
            continue
        if entry["path"] in selected_doc_scores:
            continue
        if entry["path"] not in explicit_file_paths:
            continue
        scored = _score_file(request_text, request_tokens, entry, selected_doc_scores)
        if scored["direct_score"] <= 0:
            continue
        title = Path(entry["path"]).stem.replace("-", " ").replace("_", " ").title()
        related_docs.append(
            {
                "path": entry["path"],
                "title": title,
                "score": scored["score"],
                "matches": scored["matches"],
                "score_details": scored["score_details"],
                "open_first": [],
                "verify": [],
                "risks": [],
            }
        )
    related_docs.sort(key=lambda item: (-item["score"], item["path"]))
    related_doc_paths = {doc["path"] for doc in related_docs}

    direct_markdown_docs: list[dict[str, Any]] = []
    for entry in index["files"]:
        if not entry["path"].endswith(".md"):
            continue
        if entry["path"] in selected_doc_scores or entry["path"] in related_doc_paths:
            continue
        scored = _score_file(request_text, request_tokens, entry, selected_doc_scores)
        if not _is_high_direct_file_seed(scored):
            continue
        title = Path(entry["path"]).stem.replace("-", " ").replace("_", " ").title()
        direct_markdown_docs.append(
            {
                "path": entry["path"],
                "title": title,
                "score": scored["score"],
                "matches": scored["matches"],
                "score_details": scored["score_details"],
                "open_first": [],
                "verify": [],
                "risks": [],
            }
        )
    direct_markdown_docs.sort(
        key=lambda item: (
            -len(set(item["matches"]) - COMMON_ROUTE_TOKENS),
            -item["score"],
            item["path"],
        )
    )
    promoted_markdown_docs = direct_markdown_docs[:MAX_DIRECT_MARKDOWN_DOCS]
    if intent_boosts:
        selected_docs = [*selected_docs, *related_docs[:2], *promoted_markdown_docs]
    else:
        selected_docs = [*promoted_markdown_docs, *selected_docs, *related_docs[:2]]
        selected_docs.sort(key=lambda item: (-item["score"], item["path"]))

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

    file_scores.sort(key=lambda item: (-item["score"], -item["direct_score"], item["path"]))

    seen_paths: set[str] = set()
    selected_files: list[dict[str, Any]] = []
    # Workflow-intent queries should keep their doc-provided seed files first.
    allow_specific_promotion = not intent_boosts
    high_direct_match_paths = [
        entry["path"]
        for entry in sorted(
            (
                entry
                for entry in file_scores
                if _is_high_direct_file_seed(
                    entry, allow_specific_promotion=allow_specific_promotion,
                )
            ),
            key=lambda item: (
                -_specific_direct_match_count(item),
                -item["direct_score"],
                -item["score"],
                item["path"],
            ),
        )
    ][:MAX_HIGH_DIRECT_FILE_SEEDS]
    remaining_direct_match_paths = [
        entry["path"]
        for entry in sorted(
            (
                entry
                for entry in file_scores
                if 0 < entry["direct_score"]
                and not _is_high_direct_file_seed(
                    entry, allow_specific_promotion=allow_specific_promotion,
                )
            ),
            key=lambda item: (-item["direct_score"], -item["score"], item["path"]),
        )
    ]
    explicit_non_docs = [path for path in explicit_file_paths if not path.endswith(".md")]
    explicit_seed_paths = explicit_non_docs[:MAX_EXPLICIT_FILE_SEEDS]
    remaining_explicit_paths = explicit_non_docs[MAX_EXPLICIT_FILE_SEEDS:]
    ordered_paths = (
        high_direct_match_paths
        + explicit_seed_paths
        + remaining_explicit_paths
        + remaining_direct_match_paths
        + [entry["path"] for entry in file_scores]
    )
    for path in ordered_paths:
        if path in selected_doc_scores:
            continue
        if path in seen_paths:
            continue
        if path.endswith(".md"):
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


def print_plan(plan: dict[str, Any], *, explain: bool = False) -> None:
    print(f"request: {plan['request']}")
    print(f"confidence: {plan['confidence']}")
    print("docs:")
    for doc in plan["docs"]:
        detail = f"matched: {', '.join(doc['matches'])}" if doc["matches"] else "matched: none"
        if explain:
            detail = f"score={doc['score']}; {detail}"
        print(f"- {doc['path']} ({detail})")
        if explain:
            for score_detail in doc.get("score_details", []):
                matches = ", ".join(score_detail["matches"])
                print(f"  - {score_detail['source']}: +{score_detail['score']} ({matches})")
    print("files:")
    for file_entry in plan["files"]:
        if file_entry["referenced_by"]:
            detail = f"from {', '.join(file_entry['referenced_by'])}"
        elif file_entry["matches"]:
            detail = f"matched: {', '.join(file_entry['matches'])}"
        else:
            detail = "matched: none"
        if explain:
            detail = f"score={file_entry['score']}; direct={file_entry.get('direct_score', 0)}; {detail}"
        print(f"- {file_entry['path']} ({detail})")
        if explain:
            for score_detail in file_entry.get("score_details", []):
                matches = ", ".join(score_detail["matches"])
                print(f"  - {score_detail['source']}: +{score_detail['score']} ({matches})")
    print("verify:")
    for command in plan["verify"]:
        print(f"- {command}")
    print("risks:")
    for risk in plan["risks"]:
        print(f"- {risk}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a request to the first files to inspect.")
    parser.add_argument("--json", action="store_true", help="Emit the plan as machine-readable JSON.")
    parser.add_argument("--explain", action="store_true", help="Include score details in text output.")
    parser.add_argument("request", help="Natural language task description")
    args = parser.parse_args()
    plan = plan_request(args.request)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print_plan(plan, explain=args.explain)


if __name__ == "__main__":
    main()
