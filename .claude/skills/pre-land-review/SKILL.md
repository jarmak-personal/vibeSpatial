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
pre-commit hook) and AI-powered analysis (run as isolated sub-agents for
fresh-eyes review).

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

## Tier 2: AI-Powered Analysis (sub-agent based)

The pre-commit hook CANNOT do this — it requires AI judgment. These reviews
run as **isolated sub-agents** so they analyze the code with fresh eyes,
without being biased by the context of having written the code.

### Step 1: Gather shared context

Collect this once — you will inject it into sub-agent prompts:

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

### Step 2: Launch sub-agent reviews

Based on the categories identified, launch the applicable sub-agents **in
parallel** using the Agent tool. Each sub-agent gets the diff and its domain-
specific review instructions. Use `model: "opus"` for sub-agents — GPU code
is complex and requires deep reasoning.

**IMPORTANT**: Always launch all applicable sub-agents in a SINGLE message
with multiple Agent tool calls so they run in parallel.

#### Sub-agent: GPU Code Review (if kernel/GPU/NVRTC/device code changed)

Launch with `subagent_type: "general-purpose"` and this prompt template:

```
You are a GPU code reviewer for vibeSpatial. You have NOT seen this code
before — review it with fresh eyes. Perform the full 6-pass review procedure.

## Changed Files
{file_list}

## Full Diff
{diff}

## 6-Pass Review Procedure

Pass 1: Host-Device Boundary (CRITICAL)
- No D->H->D ping-pong patterns
- No .get() / cp.asnumpy() in middle of GPU pipeline
- No Python loops over device array elements
- All D2H transfers deferred to pipeline end
- count_scatter_total() used instead of sequential .get() calls

Pass 2: Synchronization (HIGH)
- No runtime.synchronize() between same-stream operations
- No implicit sync from debug prints, scalar conversions
- Stream sync only before host reads of device data
- No cudaMalloc/cudaFree in hot paths (use pool)

Pass 3: Kernel Efficiency (HIGH)
- Block size via occupancy API, not hardcoded
- Grid size sufficient to saturate GPU
- No branch divergence in inner loops
- SoA layout for coordinate data

Pass 4: Precision Compliance (MEDIUM-HIGH)
- compute_t typedef, not hardcoded double
- PrecisionPlan wired through dispatch
- Cache key includes precision variant
- Kahan compensation for METRIC fp32 accumulations

Pass 5: Memory Management (MEDIUM)
- Pool allocation, not raw cudaMalloc
- LIFO deallocation order where possible
- Pre-sized output buffers via count-scatter

Pass 6: NVRTC/Compilation (LOW-MEDIUM)
- SHA1 cache key covers all parameterizations
- Module-scope warmup declared (request_nvrtc_warmup)
- No compilation in hot paths

## Severity Rules
Every finding is BLOCKING unless it meets NIT criteria (see below).
- BLOCKING: Must fix before landing. Includes correctness bugs, avoidable
  transfers, redundant D2H, leaked memory, missing guards, unnecessary syncs,
  and any improvement an agent can fix in minutes.
- NIT: Only for known codebase-wide gaps requiring coordinated migration,
  pure style preferences, or future design-level optimizations.

## Output Format
For each pass, report: CLEAN or list findings with severity (BLOCKING/NIT)
and location. End with overall verdict: CLEAN / BLOCKING ISSUES (list all).
```

#### Sub-agent: Zero-Copy Enforcer (if runtime/kernel/pipeline code changed)

Launch with `subagent_type: "general-purpose"` and this prompt template:

```
You are the zero-copy enforcer for vibeSpatial. Data must stay on device
until the user explicitly materializes it. Every unnecessary D/H transfer
is a performance bug. You have NOT seen this code before — review with
fresh eyes.

## Changed Files
{file_list}

## Full Diff
{diff}

## Analysis Checklist

Transfer Path Analysis:
- Map where device arrays are created, transformed, and consumed
- Identify every point where data crosses the device/host boundary
- Classify each transfer as: Necessary / Structural / Avoidable / Ping-pong

Boundary Leak Detection:
- Do new public functions accept CuPy arrays but return NumPy?
- Do new methods call .get()/.asnumpy() when they could return device arrays?
- Are intermediate results being materialized to host then sent back?

Pipeline Continuity:
- In multi-stage pipelines, does data stay on device between stages?
- Are there stages that force sync when async would work?

OwnedGeometryArray Contract:
- Do changes maintain lazy host materialization?
- Does _ensure_host_state() get called only when truly needed?

## Severity Rules
Every finding is BLOCKING unless it meets NIT criteria (see below).
- BLOCKING: Must fix before landing. Includes avoidable transfers, redundant
  D2H, ping-pong patterns, leaked device memory, and any transfer that an
  agent can eliminate in minutes.
- NIT: Only for known codebase-wide gaps requiring coordinated migration,
  or structural transfers that require significant redesign.
- Test code is exempt (Shapely oracle pattern is expected).

## Output Format
Verdict: CLEAN / LEAKY / BROKEN
For each transfer found: location, direction, classification (BLOCKING/NIT),
recommendation.
```

#### Sub-agent: Performance Analysis (if kernel/pipeline/dispatch code changed)

Launch with `subagent_type: "general-purpose"` and this prompt template:

```
You are the performance analysis enforcer for vibeSpatial, a GPU-first
spatial analytics library. You have NOT seen this code before — review
with fresh eyes.

## Changed Files
{file_list}

## Full Diff
{diff}

## Analysis Checklist

Algorithmic Complexity:
- O(n^2) where O(n log n) is achievable?
- Python loops that should be vectorized or GPU-dispatched?
- Data copied when a view/slice would suffice?

GPU Utilization:
- GPU threads sitting idle (branch divergence, uncoalesced access)?
- Enough parallelism to saturate GPU at 1M geometries?
- Kernel launch overhead amortized?

Host-Device Boundary:
- Unnecessary sync points in hot loops?
- D/H transfers that could be deferred or eliminated?

Regression Risk:
- Could this slow existing benchmarks?
- Allocation patterns that fragment GPU memory pool at scale?
- Shapely/Python round-trip in a previously device-native path?

## Severity Rules
Every finding is BLOCKING unless it meets NIT criteria (see below).
- BLOCKING: Must fix before landing. Includes correctness bugs, avoidable
  transfers, redundant allocations, unnecessary sync points, missing overflow
  guards, suboptimal patterns an agent can fix in minutes, and stale docs.
- NIT: Only for known codebase-wide gaps requiring coordinated migration,
  or future design-level optimizations requiring significant new algorithms.
- Focus on src/vibespatial/, especially kernels and pipeline code.
- Always consider 1M geometry scale.

## Output Format
Verdict: PASS / FAIL
For each finding: severity (BLOCKING/NIT), location, pattern, impact,
recommendation.
```

#### Sub-agent: Maintainability Enforcer (if any non-test source code changed)

Launch with `subagent_type: "general-purpose"` and this prompt template:

```
You are the maintainability enforcer for vibeSpatial, an agent-maintained
spatial analytics project. Code must be discoverable through the intake
routing system. You have NOT seen this code before — review with fresh eyes.

## Changed Files
{file_list}

## Full Diff
{diff}

## Analysis Checklist

Intake Routing:
- Can an agent discover the changed code via the intake routing system?
- Are there new request signals that should route to these files?

Documentation Coherence:
- Do changed behaviors have matching doc updates?
- Are new invariants documented in the right architecture doc?

Cross-Reference Integrity:
- Are there dangling references to moved/deleted code?
- Do ADR references still point to the right places?

Agent Workflow:
- Should AGENTS.md verification commands be updated?
- Are there new verification steps needed?

## Severity Rules
Every finding is BLOCKING unless it meets NIT criteria (see below).
- BLOCKING: Must fix before landing. Includes orphaned files, stale docs
  that contradict new behavior, missing routing signals, and any doc or
  routing fix an agent can make in minutes.
- NIT: Only for style preferences or observations with no functional impact.
- Test files, __init__.py, conftest.py are exempt.
- Files under kernels/ and api/ are covered by directory-level routing.

## Output Format
Verdict: DISCOVERABLE / GAPS / ORPHANED
For each gap: file, severity (BLOCKING/NIT), what's missing, specific fix
needed.
```

### Step 3: Collect and report

Wait for all sub-agents to complete, then compile results into the report.

## Report Format

```
## Pre-Land Review

### Changed Files
[list with categories]

### Deterministic Checks
[PASS/FAIL for each]

### Sub-Agent Reviews
[For each sub-agent that ran, include its verdict and any findings.
 Note: these ran in isolated sub-agents with fresh eyes on the diff.]

#### GPU Code Review: [CLEAN / ISSUES FOUND]
[findings or "N/A — no GPU code touched"]

#### Zero-Copy Analysis: [CLEAN / LEAKY / BROKEN]
[findings or "N/A"]

#### Performance Analysis: [PASS / WARN / FAIL]
[findings or "N/A"]

#### Maintainability: [DISCOVERABLE / GAPS / ORPHANED]
[findings or "N/A"]

### Overall Verdict
[LAND / FIX REQUIRED / NEEDS PROFILING]

Note: LAND requires zero BLOCKING findings across all sub-agents.
```

## Severity Classification

Every sub-agent finding must be classified into one of three tiers:

- **BLOCKING** — Must be fixed before landing. This includes:
  - Correctness bugs
  - Avoidable D/H transfers or ping-pong patterns
  - Redundant D2H transfers
  - Unnecessary sync points
  - Leaked device memory (including CuPy temporaries not explicitly freed)
  - Missing overflow guards for int32 kernel parameters
  - Stale documentation that contradicts new behavior
  - Any improvement that a powerful agent can fix in minutes
- **NIT** — Acceptable to land without fixing. Reserved for:
  - Known codebase-wide gaps (e.g., ADR-0002 fp64 hardcoding) that require
    a coordinated migration, not a per-commit fix
  - Style preferences with no functional or performance impact
  - Observations about future optimization opportunities that require
    significant design work (e.g., alternative algorithms for pathological
    distributions)
- **N/A** — Not applicable to this change

The default classification is BLOCKING. A finding is only a NIT if fixing it
is genuinely impractical in the scope of this commit. "It works fine for now"
is not sufficient — if an agent can fix it in minutes, it is BLOCKING.

## Rules

- ALL deterministic checks must pass.
- ANY BLOCKING sub-agent finding means FIX REQUIRED — fix all blocking items
  before committing. Do not land with known blocking issues.
- If runtime/kernel/pipeline code changed and no GPU is available for
  benchmarks, verdict is NEEDS PROFILING.
- Test-only changes need only deterministic checks (skip sub-agents).
- Be concise — this is a gate, not a code review.

## After Review

If the verdict is LAND, you MUST write the review marker before committing:

```bash
date -u +%Y-%m-%dT%H:%M:%SZ > .claude/.review-completed
```

This marker is checked by the `commit-msg` hook. If Claude is listed as
co-author and the marker is missing or stale (>1 hour), the commit is
blocked. The marker is single-use — the hook deletes it after a successful
commit, so each commit requires its own review. Human-only commits are
not gated.

Then proceed with the commit. Include in the commit message:
- Current strict-native GeoPandas coverage from
  `uv run python scripts/upstream_native_coverage.py --json`
- Profile summary if runtime/kernel/pipeline code was changed
