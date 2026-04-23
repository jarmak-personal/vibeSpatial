---
name: pre-land-review
description: "The review gate that must pass before any commit lands. Called automatically by the $commit skill — do NOT invoke directly when the user says \"commit\", \"land\", \"ship it\", etc. (use $commit instead). Invoke directly only when you want to run the review without committing, or when another skill references it. This is a MANDATORY gate — do not create a git commit without completing this checklist."
---

# Pre-Land Review Checklist

This skill is the landing gate for vibeSpatial. Every commit must pass
through it. The checklist has three tiers: deterministic checks (enforced by
the pre-commit hook), AI-powered analysis (run as a single review agent), and
a repo-native learning loop.

## Tier 1: Deterministic Checks

Run each command. ALL must pass before committing.

```bash
uv run ruff check
uv run python scripts/check_docs.py --check
uv run python scripts/check_architecture_lints.py --all
uv run python scripts/check_zero_copy.py --all
uv run python scripts/check_perf_patterns.py --all
uv run python scripts/check_maintainability.py --all
uv run python scripts/check_import_guard.py --all
```

If any fail, fix the issues before proceeding. The pre-commit hook will
also enforce these, but catching them here avoids a failed commit attempt.

## Tier 2: AI-Powered Review

The pre-commit hook CANNOT do this — it requires AI judgment. The review
runs as a **single dedicated agent** that performs all review passes
sequentially in one context, analyzing the code with fresh eyes without
being biased by the context of having written the code.

### Step 1: Gather context

1. Run `git diff --cached --name-only` (or `git diff HEAD~1 --name-only`)
   and save the file list.
2. Run `git diff --cached` (or `git diff HEAD~1`) and save the full diff.
3. If the changes are **test-only**, skip Tier 2 entirely — deterministic
   checks suffice.

### Step 2: Launch review agent

Spawn a **single** fresh-context review `worker` sub-agent with the file list,
the full diff, and the review procedure below. This sub-agent should perform
all applicable passes (GPU code, zero-copy, performance, maintainability, diff
shape) sequentially in one context.

Prompt template:

```
Review the following changes for vibeSpatial.

## Changed Files
{file_list}

## Full Diff
{diff}

## Review Procedure
- Categorize the changed files.
- Re-run the deterministic checks once and stop immediately if any fail.
- If the changes touch GPU kernels or GPU dispatch code, review host/device
  boundaries, synchronization, precision compliance, and memory patterns.
- If the changes touch runtime, pipeline, or dispatch code, review zero-copy,
  performance shape, maintainability, and diff anti-patterns.
- Treat every finding as BLOCKING unless it is pure style with zero
  functional or performance impact.
- Return the report in the format required by the pre-land-review skill.
```

Do not assume the sub-agent has repo-specific reviewer instructions outside
this prompt. If you do not spawn a sub-agent, perform the same review locally.

### Step 3: Collect and report

Wait for the agent to complete, then report its findings.

## Tier 3: Repo-Native Learning Loop

Before writing the review marker, check whether the task exposed anything the
repo should now know. Ask these questions:

- Did the task expose an intake misroute or missing discovery signal?
- Did the task reveal missing or stale docs?
- Did the task reveal a repeated bug pattern that should become a lint,
  hygiene check, benchmark, or test?
- Did the task expose weak verification or a missing regression fixture?
- Did the task require an unexplained workaround or local-only assumption?

For each "yes", either land a tracked repo artifact or record why no artifact
is appropriate:

- test or fixture
- doc or intake signal
- lint, hygiene check, or ratchet
- benchmark or profile rail
- skill/workflow instruction
- explicit `none` reason in `.agents/runs/<id>/learning.jsonl`

If an active run exists, use:

```bash
uv run python scripts/agent_run.py learning review --require-resolved
```

This does not replace engineering judgment. It verifies that captured
learnings point at tracked repo artifacts or have an explicit no-artifact
reason.

## Report Format

```
## Pre-Land Review

### Changed Files
[list with categories]

### Deterministic Checks
[PASS/FAIL for each]

### Agent Review

#### GPU Code Review: [CLEAN / BLOCKING ISSUES]
[findings or "N/A — no GPU code touched"]

#### Zero-Copy Analysis: [CLEAN / LEAKY / BROKEN]
[findings or "N/A"]

#### Performance Analysis: [PASS / FAIL]
[findings or "N/A"]

#### Maintainability: [DISCOVERABLE / GAPS / ORPHANED]
[findings or "N/A"]

#### Diff Shape: [CLEAN / findings]
[findings or "N/A"]

#### Learning Loop: [CAPTURED / NO NEW LEARNINGS / GAPS]
[repo artifacts added, active-run learning summary, or "N/A — no active run and no new learnings"]

### Overall Verdict
[LAND / FIX REQUIRED]

Note: LAND requires zero BLOCKING findings across all passes.
```

## Severity Rules

The review pass follows these severity rules:

- **BLOCKING** — Must fix before landing. Default for all findings.
- **NIT** — Only for pure style preferences with zero functional or
  performance impact.

**"Existing codebase does it too" is NEVER a valid NIT justification.**
If new code builds on a broken upstream pattern, that is BLOCKING — fix
the upstream too.

## Rules

- ALL deterministic checks must pass.
- ANY BLOCKING finding means FIX REQUIRED.
- Test-only changes need only deterministic checks (skip agent).
- Be concise — this is a gate, not a code review.

## After Review

If the verdict is LAND, write the content-addressable review marker:

```bash
printf '{\n  "timestamp": "%s",\n  "staged_hash": "%s",\n  "files": [%s],\n  "verdict": "LAND"\n}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(git diff --cached | sha256sum | cut -d' ' -f1)" \
  "$(git diff --cached --name-only | sed 's/.*/"&"/' | paste -sd,)" \
  > .agents/.review-completed
```

The `commit-msg` hook verifies:
1. The file exists and is less than 1 hour old.
2. `staged_hash` matches a fresh `git diff --cached | sha256sum`.
3. `verdict` is `LAND`.

The marker is single-use — the hook deletes it after a successful commit.

Then proceed with the commit.
