---
name: pre-land-review
description: >
  PROACTIVELY USE THIS SKILL before committing code, landing work, ending a
  session, or when the user says "commit", "land", "done", "ship it", "wrap up",
  or "let's finish". This is a MANDATORY gate — do not create a git commit
  without completing this checklist. The pre-commit hook enforces deterministic
  checks automatically, but the AI-powered review steps require you to run them
  in-session before the commit.
user-invocable: true
argument-hint: "[git-ref range, default HEAD~1]"
---

# Pre-Land Review Checklist

This skill is the landing gate for vibeSpatial. Every commit must pass through
it. The checklist has two tiers: deterministic checks (enforced by the
pre-commit hook) and AI-powered analysis (enforced by YOU running it here).

## Tier 1: Deterministic Checks (verify these pass)

Run each command. ALL must pass before committing.

```bash
uv run ruff check
uv run python scripts/check_docs.py --check
uv run python scripts/check_architecture_lints.py --all
uv run python scripts/check_zero_copy.py --all
uv run python scripts/check_perf_patterns.py --all
uv run python scripts/check_maintainability.py --all
```

If any fail, fix the issues before proceeding. The pre-commit hook will also
enforce these, but catching them here avoids a failed commit attempt.

## Tier 2: AI-Powered Analysis (you must do this)

The pre-commit hook CANNOT do this — it requires your judgment. Gather
context once, then analyze all three domains.

### Gather

1. `git diff --cached --name-only` (or `git diff HEAD~1 --name-only`)
2. `git diff --cached` (or `git diff HEAD~1`)
3. Identify which categories the changes touch:
   - **kernel/pipeline**: needs performance + zero-copy analysis
   - **dispatch/api**: needs zero-copy + maintainability analysis
   - **docs/scripts**: needs maintainability analysis only
   - **tests only**: skip AI analysis (deterministic checks suffice)

### Analyze: Performance (if kernel/pipeline/dispatch code changed)

- Algorithmic complexity: O(n^2) where O(n log n) is achievable?
- GPU utilization: enough parallelism at 1M geometries? branch divergence?
- Host-device boundary: unnecessary syncs? deferrable transfers?
- Tier compliance (ADR-0033): correct GPU primitive tier?
- Regression risk: could this slow an existing benchmark?

### Analyze: Zero-Copy (if runtime code changed)

- Transfer paths: every D/H crossing classified as Necessary/Structural/Avoidable?
- Ping-pongs: any D->H->D round-trips?
- Boundary leaks: functions accepting device data but returning host data?
- Pipeline continuity: data stays on device between stages?
- OwnedGeometryArray contract: lazy host materialization maintained?

### Analyze: Maintainability (if any non-test code changed)

- Intake routing: can an agent discover the changed code via intake.md?
- Doc coherence: do changed behaviors have matching doc updates?
- Cross-references: any dangling references to moved/deleted code?
- AGENTS.md: new scripts/modules listed in project shape?

## Report Format

After analysis, output a concise report:

```
## Pre-Land Review

### Changed Files
[list with categories]

### Deterministic Checks
[PASS/FAIL for each]

### AI Analysis
[Findings by domain, only for applicable domains]

### Overall Verdict
[LAND / FIX REQUIRED / NEEDS PROFILING]
```

## Rules

- ALL deterministic checks must pass.
- ANY critical AI finding means FIX REQUIRED.
- If runtime/kernel/pipeline code changed and no GPU is available for
  benchmarks, verdict is NEEDS PROFILING.
- Test-only changes need only deterministic checks.
- Be concise — this is a gate, not a code review.

## After Review

Once the review passes, proceed with the commit. Include in the commit message:
- Current strict-native GeoPandas coverage from
  `uv run python scripts/upstream_native_coverage.py --json`
- Profile summary if runtime/kernel/pipeline code was changed
