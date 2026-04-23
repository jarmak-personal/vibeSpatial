"""Visible local execution state for agentic work.

The files written by this script are intentionally local run artifacts, not
durable project knowledge. Durable learnings should become tests, docs, lints,
benchmarks, intake signals, or skills before landing.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .intake import plan_request
except ImportError:  # pragma: no cover - script execution path
    from intake import plan_request

RUNS_DIR = Path(".agents/runs")
ACTIVE_FILE = RUNS_DIR / "ACTIVE"
LEARNING_ACTIONS = {
    "benchmark",
    "docs",
    "intake-signal",
    "lint",
    "none",
    "skill",
    "test",
}
LEARNING_KINDS = {
    "benchmark-gap",
    "doc-gap",
    "intake-misroute",
    "lint-gap",
    "review-finding",
    "test-gap",
    "workflow-gap",
}
HYPOTHESIS_STATUSES = {"accepted", "blocked", "open", "rejected", "superseded"}


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_dir: Path

    @property
    def problem(self) -> Path:
        return self.run_dir / "problem.json"

    @property
    def hypotheses(self) -> Path:
        return self.run_dir / "hypotheses.jsonl"

    @property
    def evidence(self) -> Path:
        return self.run_dir / "evidence.jsonl"

    @property
    def commands(self) -> Path:
        return self.run_dir / "commands.jsonl"

    @property
    def learning(self) -> Path:
        return self.run_dir / "learning.jsonl"

    @property
    def closeout_json(self) -> Path:
        return self.run_dir / "closeout.json"

    @property
    def closeout_md(self) -> Path:
        return self.run_dir / "closeout.md"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "agent-run"


def _run_paths(root: Path, run_id: str) -> RunPaths:
    return RunPaths(root=root, run_dir=root / RUNS_DIR / run_id)


def _active_run_id(root: Path) -> str:
    active = root / ACTIVE_FILE
    if active.exists():
        run_id = active.read_text(encoding="utf-8").strip()
        if run_id:
            return run_id
    runs_root = root / RUNS_DIR
    if not runs_root.exists():
        raise SystemExit("No active agent run. Start one with: agent_run.py start \"<request>\"")
    candidates = sorted(path.name for path in runs_root.iterdir() if path.is_dir())
    if candidates:
        return candidates[-1]
    raise SystemExit("No active agent run. Start one with: agent_run.py start \"<request>\"")


def _resolve_paths(root: Path, run_id: str | None) -> RunPaths:
    return _run_paths(root, run_id or _active_run_id(root))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _next_id(rows: list[dict[str, Any]], prefix: str) -> str:
    max_seen = 0
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for row in rows:
        value = str(row.get("id", ""))
        match = pattern.match(value)
        if match:
            max_seen = max(max_seen, int(match.group(1)))
    return f"{prefix}{max_seen + 1:03d}"


def _git_status(root: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "paths": []}
    return {
        "available": True,
        "paths": [line for line in completed.stdout.splitlines() if line],
    }


def _infer_intent(request: str) -> str:
    text = request.lower()
    if any(token in text for token in ("commit", "land", "ship")):
        return "landing"
    if any(token in text for token in ("profile", "benchmark", "optimize", "performance")):
        return "performance"
    if any(token in text for token in ("bug", "debug", "fix", "wrong", "fail")):
        return "debug"
    if any(token in text for token in ("doc", "docs", "readme")):
        return "docs"
    if any(token in text for token in ("add", "build", "create", "implement", "improve")):
        return "implementation"
    return "investigation"


def _target_properties(request: str) -> list[str]:
    text = request.lower()
    properties: list[str] = []
    if any(token in text for token in ("intake", "discover", "agent", "workflow")):
        properties.append("discoverability")
    if any(token in text for token in ("perf", "profile", "benchmark", "gpu")):
        properties.append("performance")
    if any(token in text for token in ("fallback", "cpu", "zero-copy", "device")):
        properties.append("gpu-first execution")
    if any(token in text for token in ("test", "verify", "contract")):
        properties.append("verification")
    return properties or ["task-specific correctness"]


def _intake_plan(request: str, *, skip_intake: bool) -> dict[str, Any] | None:
    if skip_intake:
        return None
    return plan_request(request)


def _write_active(root: Path, run_id: str) -> None:
    active = root / ACTIVE_FILE
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(run_id + "\n", encoding="utf-8")


def start_run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    run_id = args.run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{_slugify(args.request)}"
    paths = _run_paths(root, run_id)
    if paths.run_dir.exists():
        raise SystemExit(f"Agent run already exists: {paths.run_dir}")

    intake_plan = _intake_plan(args.request, skip_intake=args.skip_intake)
    payload = {
        "id": run_id,
        "created_at": _now(),
        "request": args.request,
        "intent": _infer_intent(args.request),
        "target_properties": _target_properties(args.request),
        "intake_plan": intake_plan,
        "expected_verification": intake_plan["verify"] if intake_plan else [],
        "risks": intake_plan["risks"] if intake_plan else [],
        "non_goals": [],
        "dirty_worktree": _git_status(root),
    }
    _write_json(paths.problem, payload)
    for path in (paths.hypotheses, paths.evidence, paths.commands, paths.learning):
        path.touch()
    _write_active(root, run_id)
    print(paths.run_dir.relative_to(root))
    return 0


def _hypothesis_states(paths: RunPaths) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(paths.hypotheses):
        hyp_id = row["id"]
        state = states.setdefault(
            hyp_id,
            {
                "id": hyp_id,
                "text": row.get("text", ""),
                "status": "open",
                "created_at": row.get("timestamp"),
                "updated_at": row.get("timestamp"),
                "notes": [],
            },
        )
        if row["event"] == "add":
            state["text"] = row["text"]
            state["status"] = "open"
        elif row["event"] == "resolve":
            state["status"] = row["status"]
            state["updated_at"] = row["timestamp"]
            if row.get("note"):
                state["notes"].append(row["note"])
    return states


def hypothesis_add(args: argparse.Namespace) -> int:
    paths = _resolve_paths(args.root.resolve(), args.run_id)
    hyp_id = _next_id(_read_jsonl(paths.hypotheses), "H")
    _append_jsonl(
        paths.hypotheses,
        {"event": "add", "id": hyp_id, "status": "open", "text": args.text, "timestamp": _now()},
    )
    print(hyp_id)
    return 0


def hypothesis_resolve(args: argparse.Namespace) -> int:
    if args.status not in HYPOTHESIS_STATUSES - {"open"}:
        raise SystemExit(f"Invalid resolved status: {args.status}")
    paths = _resolve_paths(args.root.resolve(), args.run_id)
    states = _hypothesis_states(paths)
    if args.id not in states:
        raise SystemExit(f"Unknown hypothesis: {args.id}")
    _append_jsonl(
        paths.hypotheses,
        {
            "event": "resolve",
            "id": args.id,
            "note": args.note,
            "status": args.status,
            "timestamp": _now(),
        },
    )
    print(f"{args.id}: {args.status}")
    return 0


def evidence_add(args: argparse.Namespace) -> int:
    paths = _resolve_paths(args.root.resolve(), args.run_id)
    if args.hypothesis:
        states = _hypothesis_states(paths)
        if args.hypothesis not in states:
            raise SystemExit(f"Unknown hypothesis: {args.hypothesis}")
    evidence_id = _next_id(_read_jsonl(paths.evidence), "E")
    _append_jsonl(
        paths.evidence,
        {
            "event": "add",
            "hypothesis": args.hypothesis,
            "id": evidence_id,
            "text": args.text,
            "timestamp": _now(),
        },
    )
    print(evidence_id)
    return 0


def command_add(args: argparse.Namespace) -> int:
    paths = _resolve_paths(args.root.resolve(), args.run_id)
    command_id = _next_id(_read_jsonl(paths.commands), "C")
    _append_jsonl(
        paths.commands,
        {
            "command": args.command,
            "event": "add",
            "id": command_id,
            "status": args.status,
            "timestamp": _now(),
        },
    )
    print(command_id)
    return 0


def learning_add(args: argparse.Namespace) -> int:
    if args.kind not in LEARNING_KINDS:
        raise SystemExit(f"Invalid learning kind: {args.kind}")
    if args.repo_action not in LEARNING_ACTIONS:
        raise SystemExit(f"Invalid repo action: {args.repo_action}")
    if args.repo_action == "none" and not args.reason:
        raise SystemExit("--reason is required when --repo-action none")
    if args.repo_action != "none" and not args.path:
        raise SystemExit("--path is required unless --repo-action none")

    paths = _resolve_paths(args.root.resolve(), args.run_id)
    learning_id = _next_id(_read_jsonl(paths.learning), "L")
    _append_jsonl(
        paths.learning,
        {
            "event": "add",
            "id": learning_id,
            "kind": args.kind,
            "path": args.path,
            "reason": args.reason,
            "repo_action": args.repo_action,
            "summary": args.summary,
            "timestamp": _now(),
        },
    )
    print(learning_id)
    return 0


def _learning_rows(paths: RunPaths) -> list[dict[str, Any]]:
    return [row for row in _read_jsonl(paths.learning) if row.get("event") == "add"]


def learning_review(args: argparse.Namespace) -> int:
    paths = _resolve_paths(args.root.resolve(), args.run_id)
    rows = _learning_rows(paths)
    print(f"Learning items: {len(rows)}")
    for row in rows:
        target = row["path"] if row["repo_action"] != "none" else row["reason"]
        print(f"- {row['id']} {row['kind']} -> {row['repo_action']}: {target}")
    missing = [
        row
        for row in rows
        if not row.get("repo_action")
        or (row["repo_action"] == "none" and not row.get("reason"))
        or (row["repo_action"] != "none" and not row.get("path"))
    ]
    if args.require_resolved and missing:
        for row in missing:
            print(f"unresolved learning item: {row['id']}")
        return 1
    return 0


def _summary(paths: RunPaths) -> dict[str, Any]:
    problem = _load_json(paths.problem)
    hypotheses = _hypothesis_states(paths)
    evidence = _read_jsonl(paths.evidence)
    commands = _read_jsonl(paths.commands)
    learning = _learning_rows(paths)
    open_hypotheses = [item for item in hypotheses.values() if item["status"] == "open"]
    return {
        "id": problem["id"],
        "request": problem["request"],
        "intent": problem["intent"],
        "hypotheses": list(hypotheses.values()),
        "open_hypotheses": open_hypotheses,
        "evidence_count": len(evidence),
        "commands": commands,
        "learning": learning,
    }


def status_run(args: argparse.Namespace) -> int:
    paths = _resolve_paths(args.root.resolve(), args.run_id)
    summary = _summary(paths)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"Run: {summary['id']}")
    print(f"Request: {summary['request']}")
    print(f"Intent: {summary['intent']}")
    print(f"Hypotheses: {len(summary['hypotheses'])} ({len(summary['open_hypotheses'])} open)")
    print(f"Evidence: {summary['evidence_count']}")
    print(f"Learning items: {len(summary['learning'])}")
    return 0


def close_run(args: argparse.Namespace) -> int:
    paths = _resolve_paths(args.root.resolve(), args.run_id)
    summary = _summary(paths)
    payload = {
        **summary,
        "closed_at": _now(),
        "residual_risks": args.risk,
        "summary": args.summary,
        "verification": args.verification,
    }
    _write_json(paths.closeout_json, payload)

    lines = [
        f"# Agent Run {payload['id']}",
        "",
        f"Request: {payload['request']}",
        f"Intent: {payload['intent']}",
        "",
        "## Summary",
        args.summary or "No summary provided.",
        "",
        "## Verification",
        *(f"- {item}" for item in args.verification),
        "",
        "## Learning",
        *(f"- {row['id']} {row['kind']} -> {row['repo_action']}" for row in payload["learning"]),
        "",
        "## Residual Risks",
        *(f"- {item}" for item in args.risk),
        "",
    ]
    paths.closeout_md.write_text("\n".join(lines), encoding="utf-8")
    print(paths.closeout_md.relative_to(paths.root))
    return 0


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root for run files.")
    parser.add_argument("--run-id", help="Run id. Defaults to .agents/runs/ACTIVE.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track visible local agent execution state.")
    _add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Create a problem capsule and active run.")
    start.add_argument("request")
    start.add_argument("--skip-intake", action="store_true", help="Do not run intake routing.")
    start.set_defaults(func=start_run)

    status = subparsers.add_parser("status", help="Show active run state.")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=status_run)

    close = subparsers.add_parser("close", help="Write closeout files for the run.")
    close.add_argument("--summary", default="")
    close.add_argument("--verification", action="append", default=[])
    close.add_argument("--risk", action="append", default=[])
    close.set_defaults(func=close_run)

    hypothesis = subparsers.add_parser("hypothesis", help="Manage hypotheses.")
    hypothesis_sub = hypothesis.add_subparsers(dest="hypothesis_command", required=True)
    hyp_add = hypothesis_sub.add_parser("add")
    hyp_add.add_argument("text")
    hyp_add.set_defaults(func=hypothesis_add)
    hyp_resolve = hypothesis_sub.add_parser("resolve")
    hyp_resolve.add_argument("id")
    hyp_resolve.add_argument("--status", required=True, choices=sorted(HYPOTHESIS_STATUSES - {"open"}))
    hyp_resolve.add_argument("--note", default="")
    hyp_resolve.set_defaults(func=hypothesis_resolve)

    evidence = subparsers.add_parser("evidence", help="Record evidence.")
    evidence_sub = evidence.add_subparsers(dest="evidence_command", required=True)
    ev_add = evidence_sub.add_parser("add")
    ev_add.add_argument("text")
    ev_add.add_argument("--hypothesis")
    ev_add.set_defaults(func=evidence_add)

    command = subparsers.add_parser("command-log", help="Record a command run.")
    command_sub = command.add_subparsers(dest="command_log_command", required=True)
    cmd_add = command_sub.add_parser("add")
    cmd_add.add_argument("command")
    cmd_add.add_argument("--status", choices=["blocked", "fail", "pass", "skipped"], required=True)
    cmd_add.set_defaults(func=command_add)

    learning = subparsers.add_parser("learning", help="Record repo-native learnings.")
    learning_sub = learning.add_subparsers(dest="learning_command", required=True)
    learn_add = learning_sub.add_parser("add")
    learn_add.add_argument("--kind", required=True, choices=sorted(LEARNING_KINDS))
    learn_add.add_argument("--summary", required=True)
    learn_add.add_argument("--repo-action", required=True, choices=sorted(LEARNING_ACTIONS))
    learn_add.add_argument("--path")
    learn_add.add_argument("--reason")
    learn_add.set_defaults(func=learning_add)
    learn_review = learning_sub.add_parser("review")
    learn_review.add_argument("--require-resolved", action="store_true")
    learn_review.set_defaults(func=learning_review)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
