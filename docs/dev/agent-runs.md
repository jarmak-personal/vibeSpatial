# Agent Run State

<!-- DOC_HEADER:START
Scope: Visible local agent run capsules, hypothesis ledgers, and repo-native learning loop workflow.
Read If: You are starting a long-running agent task, debugging mid-task reasoning, or closing work with durable learnings.
STOP IF: You only need the normal intake route for a short one-command task.
Source Of Truth: Agent run state and repo-native learning workflow.
Body Budget: 89/220 lines
Document: docs/dev/agent-runs.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-10 | Intent |
| 11-22 | Request Signals |
| 23-31 | Open First |
| 32-36 | Verify |
| 37-44 | Risks |
| 45-84 | Workflow |
| 85-89 | Storage Contract |
DOC_HEADER:END -->

Use `scripts/agent_run.py` when a task needs durable local execution context
across investigation, context compaction, or handoff.

## Intent

Keep active agent reasoning visible in repo-local files while forcing durable
learnings back into normal repo artifacts.

## Request Signals

- problem capsule
- hypothesis ledger
- learning loop
- agent run
- closeout
- execution state
- repo-native learning
- task handoff
- agent workflow

## Open First

- docs/dev/agent-runs.md
- scripts/agent_run.py
- docs/ops/intake.md
- docs/dev/precommit.md
- .agents/skills/pre-land-review/SKILL.md
- .agents/skills/commit/SKILL.md

## Verify

- `uv run pytest tests/test_agent_run.py -q`
- `uv run python scripts/check_docs.py --check`

## Risks

- Treating `.agents/runs/` as durable knowledge recreates hidden memory debt.
- Closing a run without converting lessons into tests, docs, lints, benchmarks,
  intake signals, or skills leaves the next agent to rediscover the same fact.
- Over-instrumenting small tasks makes agents spend more time bookkeeping than
  solving the problem.

## Workflow

Start a run for substantial tasks:

```bash
uv run python scripts/agent_run.py start "fix overlay keep_geom_type routing"
```

This writes a problem capsule under `.agents/runs/<id>/problem.json` and marks
the run active via `.agents/runs/ACTIVE`.

During investigation, record hypotheses and evidence:

```bash
uv run python scripts/agent_run.py hypothesis add "Doc fanout is outranking direct file hits"
uv run python scripts/agent_run.py evidence add --hypothesis H001 "overlay.py ranks below doc seeds"
uv run python scripts/agent_run.py hypothesis resolve H001 --status accepted
```

When a reusable lesson appears, record the repo artifact that captured it:

```bash
uv run python scripts/agent_run.py learning add \
  --kind intake-misroute \
  --summary "keep_geom_type needs a gold routing query" \
  --repo-action test \
  --path tests/test_intake_quality.py
```

Use `--repo-action none --reason "<why>"` only when no repo artifact is
appropriate.

Close the run with verification and residual risks:

```bash
uv run python scripts/agent_run.py close \
  --summary "Added router quality harness" \
  --verification "uv run pytest tests/test_intake_quality.py -q"
```

## Storage Contract

`.agents/runs/` is gitignored local state. It helps the current agent think and
handoff, but it is not project memory. Durable learning must land in tracked
repo artifacts before commit.
