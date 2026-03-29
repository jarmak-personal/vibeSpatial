---
name: pre-land-review
description: "The review gate that must pass before any commit lands. Called automatically by the /commit skill — do NOT invoke directly when the user says \"commit\", \"land\", \"ship it\", etc. (use /commit instead). Invoke directly only when you want to run the review without committing, or when another skill references it. This is a MANDATORY gate — do not create a git commit without completing this checklist."
user-invocable: true
argument-hint: "[git-ref range, default HEAD~1]"
---

# Pre-Land Review Checklist

This skill is the landing gate for vibeSpatial. Every commit must pass
through it. The checklist has two tiers: deterministic checks (enforced by
the pre-commit hook) and AI-powered analysis (run as review agents).

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

## Tier 2: AI-Powered Review Agents

The pre-commit hook CANNOT do this — it requires AI judgment. Reviews run
as **dedicated agents** so they analyze the code with fresh eyes, without
being biased by the context of having written the code.

### Step 1: Gather context

1. Run `git diff --cached --name-only` (or `git diff HEAD~1 --name-only`)
   and save the file list.
2. Run `git diff --cached` (or `git diff HEAD~1`) and save the full diff.
3. Categorize the changed files:
   - **kernel/GPU code** (`*_kernels.py`, `*_gpu.py`, CUDA source strings,
     `cuda_runtime.py`, `cccl_*.py`): needs GPU code review + performance +
     zero-copy analysis
   - **pipeline/dispatch** (`*_pipeline.py`, dispatch logic, runtime):
     needs performance + zero-copy analysis
   - **api** (`api/`, public functions): needs zero-copy + maintainability
   - **docs/scripts**: needs maintainability only
   - **tests only**: skip AI analysis (deterministic checks suffice)

### Step 2: Launch review agents

Based on the categories, spawn the applicable agents **in parallel** using
the Agent tool. Each agent gets the file list and full diff injected into
its prompt. **Launch all applicable agents in a SINGLE message.**

| Category | Agent | When |
|----------|-------|------|
| GPU Code Review | `gpu-code-reviewer` | kernel/GPU/NVRTC/device code changed |
| Zero-Copy | `zero-copy-reviewer` | src/vibespatial/ code changed (not docs/scripts/tests) |
| Performance | `performance-reviewer` | src/vibespatial/ code changed (not docs/scripts/tests) |
| Maintainability | `maintainability-reviewer` | any non-test source code changed |
| Acceleration Angel | `acceleration-angel` | src/vibespatial/ code changed (not docs/scripts/tests) |

Prompt template for each agent:

```
Review the following changes for vibeSpatial.

## Changed Files
{file_list}

## Full Diff
{diff}
```

The agents already know their review procedure and severity rules from
their agent definitions. You only need to provide the diff context.

### Step 3: Collect and report

Wait for all agents to complete, then compile results.

## Report Format

```
## Pre-Land Review

### Changed Files
[list with categories]

### Deterministic Checks
[PASS/FAIL for each]

### Agent Reviews

#### GPU Code Review: [CLEAN / BLOCKING ISSUES]
[findings or "N/A — no GPU code touched"]

#### Zero-Copy Analysis: [CLEAN / LEAKY / BROKEN]
[findings or "N/A"]

#### Performance Analysis: [PASS / FAIL]
[findings or "N/A"]

#### Maintainability: [DISCOVERABLE / GAPS / ORPHANED]
[findings or "N/A"]

#### Acceleration Angel: [PASS / FIX REQUIRED]
[findings or "N/A"]

### Overall Verdict
[LAND / FIX REQUIRED / NEEDS PROFILING]

Note: LAND requires zero BLOCKING findings across all agents.
```

## Severity Rules

All review agents follow these severity rules (defined in their agent
definitions):

- **BLOCKING** — Must fix before landing. Default for all findings.
- **NIT** — Only for pure style preferences with zero functional or
  performance impact.

**"Existing codebase does it too" is NEVER a valid NIT justification.**
If new code builds on a broken upstream pattern, that is BLOCKING — fix
the upstream too.

## Rules

- ALL deterministic checks must pass.
- ANY BLOCKING agent finding means FIX REQUIRED.
- Test-only changes need only deterministic checks (skip agents).
- Be concise — this is a gate, not a code review.

## After Review

If the verdict is LAND, write the content-addressable review marker:

```bash
printf '{\n  "timestamp": "%s",\n  "staged_hash": "%s",\n  "files": [%s],\n  "verdict": "LAND"\n}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(git diff --cached | sha256sum | cut -d' ' -f1)" \
  "$(git diff --cached --name-only | sed 's/.*/"&"/' | paste -sd,)" \
  > .claude/.review-completed
```

The `commit-msg` hook verifies:
1. The file exists and is less than 1 hour old.
2. `staged_hash` matches a fresh `git diff --cached | sha256sum`.
3. `verdict` is `LAND`.

The marker is single-use — the hook deletes it after a successful commit.

Then proceed with the commit.
