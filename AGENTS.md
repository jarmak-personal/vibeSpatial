# Agent Instructions

<!-- DOC_HEADER:START
Scope: Repository-wide agent workflow, intake usage, and verification expectations.
Read If: You are starting, routing, or landing work in this repository.
STOP IF: You only need a narrow API detail already covered by a routed doc.
Source Of Truth: Agent workflow and handoff policy for vibeSpatial.
Body Budget: 260/260 lines
Document: AGENTS.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Preamble |
| 7-13 | Intent |
| 14-22 | Request Signals |
| 23-29 | Open First |
| 30-34 | Verify |
| 35-40 | Risks |
| 41-58 | Mission |
| 59-71 | Startup |
| 72-85 | Routing |
| 86-104 | Execution Model |
| 105-135 | Property Convergence |
| 136-147 | Test Strategy |
| 148-156 | Build And Tooling |
| 157-184 | Verification |
| ... | (3 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

GPU-first spatial analytics library. All agent work is measured by
advancement of codebase properties, not task completion. Performance is
the design center. Silent CPU fallbacks are unacceptable.

## Intent

Define the agent workflow for routing requests, verifying changes, and
landing work. This document covers mindset and policy. For file discovery,
use intake routing. For enforcement, trust the pre-commit hook and
property dashboard.

## Request Signals

- intake
- agent workflow
- autonomous mode / PRD execution
- docs-only
- verification
- repo map / handoff

## Open First

- AGENTS.md
- docs/ops/intake.md
- README.md
- pyproject.toml

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "docs update"`

## Risks

- Reading the whole repo before routing increases context noise and slows edits.
- Silent CPU fallback changes can ship if runtime docs and tests drift apart.
- Stale generated headers or intake index can misroute future tasks.

## Mission

Build a new NVIDIA GPU-accelerated spatial analytics library from scratch.

**Hard constraints:**

- PERFORMANCE IS KING. Consider performance implications in every design
  and implementation decision. If an approach has the wrong performance
  shape, throw it out instead of polishing it.
- GPU-first: `cuda-python`, CCCL primitives, NVRTC kernels, CuPy. Never
  NumPy, Shapely, or Rasterio in GPU code paths. No exceptions.
- CPU fallback is available but never silent and never the design center.
  Every fallback must emit an observable event (ADR-0013).
- Use the upstream GeoPandas test suite as the compatibility contract.
- Correctness is non-negotiable. Quick fixes, workarounds, and timeline
  considerations are never acceptable. Always find the root cause, consider
  multiple solutions, and implement the correct one.

## Startup

1. Run `$intake-router` with the task description to find relevant files.
2. Open only the routed files/docs first.
3. Read relevant ADRs from `docs/decisions/index.md` when the task reopens
   a design decision.
4. Make the smallest coherent change that advances target properties.
5. Run the narrowest meaningful verification before expanding scope.
6. Refresh generated docs with `uv run python scripts/check_docs.py --refresh` when doc metadata changes.
7. If the user provides a PRD or tasklist, treat it as the mandate and execute end-to-end without confirmation loops.

For session orientation: `uv run python scripts/health.py --check`

## Routing

Use `$intake-router` or `uv run python scripts/intake.py "<request>"` as
the source of truth for file discovery. Do not manually enumerate files.
Do not front-load full-repo reads. Route, inspect the local area, then
expand.

Legacy Claude agent definitions live in `.claude/agents/` and describe
themselves. Claude skills live in `.claude/skills/`. Repo-local Codex skills
live in `.agents/skills/`.
Scripts live in `scripts/` and are indexed by intake routing.

Do not duplicate their contents here.

## Execution Model

- Default policy is `auto`: use GPU when available, otherwise CPU.
- Any explicit CPU fallback must be observable in code, logs, or test output.
- Do not hide unsupported GPU behavior behind silent host-side execution.
- Prefer data-parallel kernels and bulk columnar transforms over Python loops.
- Reach for CCCL or `cuda-python` building blocks before inventing custom glue.
- CCCL now has a Python interface; review the docs before designing new GPU
  primitives: <https://nvidia.github.io/cccl/unstable/python/compute_api.html>

Hard blocks in GPU code paths (pre-commit hook enforced):

- **NumPy** — use CuPy. NumPy is acceptable only for host-side data that
  never touches the GPU pipeline.
- **Shapely** — use GPU kernels. Shapely is acceptable only at the
  GeoPandas compatibility boundary.
- **Object-dtype arrays** — use typed columnar arrays.
- **Python for-loops over geometry objects** — use vectorized GPU operations.

## Property Convergence

Agent work is measured by the **property dashboard**, not task completion.

```bash
uv run python scripts/property_dashboard.py        # human-readable
uv run python scripts/property_dashboard.py --json  # machine-readable
```

The dashboard aggregates all check scripts into a unified view. Each
property has a distance metric (0.0 = satisfied, >1.0 = regressed).
Ratchet baselines track known debt; new code must not increase it.

**Before starting work:** snapshot the dashboard with
`uv run python scripts/property_dashboard.py --json` if the task is broad
enough that a property baseline will help.

**After completing work:** re-run the dashboard. Your changes are successful
when:
1. At least one target property distance decreased
2. No property distance increased
3. The user's stated goal is achieved

What does NOT count as success:
- A fix that passes tests but advances no property (workaround)
- A change that advances one property but regresses another
- A change that adds new violations to any ratchet baseline

The pre-commit hook enforces all property checks automatically. The
`$pre-land-review` skill validates before commit.

## Test Strategy

- Treat vendored GeoPandas tests as upstream contract coverage.
- Keep the vendored tree refreshable; do not hand-edit copied tests.
- Patch vendored imports only when required for self-containment.
- Add repo-local tests for new runtime, kernel, and fallback behavior.
- New kernel tests should use the Shapely oracle fixture pattern so CPU/GPU
  paths compare against the same mechanical host reference.
- New owned kernels start from:
  `uv run python scripts/generate_kernel_scaffold.py <kernel-name>`
- Prefer targeted smoke runs while bootstrapping, then expand.

## Build And Tooling

- Use `uv` for environment and dependency management.
- Keep the default bootstrap path lightweight enough for test smoke checks.
- Put GPU-specific dependencies behind focused dependency groups.
- If adopting CCCL Python, prefer a first-class refactor to native
  CCCL-compatible device arrays rather than layering a temporary adapter.
- Pre-commit/pre-push hooks install via `uv run python scripts/install_githooks.py`

## Verification

The pre-commit hook runs all deterministic checks automatically:
ruff, doc refresh/validation, ARCH, ZCOPY, VPAT, MAINT, IGRD.
You do not need to run these manually.
The pre-push hook runs cached contract/GPU health before code leaves the workstation.

For targeted verification during development:

| Surface changed | Command |
|----------------|---------|
| Any code | `uv run ruff check` |
| Runtime/package | `uv run pytest` |
| Docs/intake | `uv run python scripts/check_docs.py --check` |
| Pipeline/profiler | `uv run pytest tests/test_pipeline_benchmarks.py -q` |
| Vendored tests | `uv run python scripts/vendor_geopandas_tests.py` |
| Upstream smoke | `uv run pytest tests/upstream/geopandas/tests/test_config.py` |
| All properties | `uv run python scripts/property_dashboard.py` |

For GPU-sensitive verification, do not trust sandboxed "no GPU runtime"
results by default. Re-run the relevant `uv run python ...`, `uv run pytest ... --run-gpu`, or benchmark command outside the sandbox with escalation, and collect `nvidia-smi -L`, `ls /dev/nvidia*`, and `printenv CUDA_VISIBLE_DEVICES`. Missing `/dev/nvidia*` or a failing `nvidia-smi` inside the sandbox is an environment visibility issue first, not a library regression.

If verification cannot run because dependencies, drivers, or local services
are missing, say so concretely and include the command that failed.

The `$commit` skill runs `$pre-land-review` automatically, which orchestrates
deterministic checks and AI-powered sub-agent reviews.

## Working Rules

- Correctness is non-negotiable. Performance is king. UX is queen.
- Never accept a workaround. Never consider timelines. Never optimize for
  appearing done. The only question is: what is the correct solution?
- Quick fixes are ALWAYS wrong. If you're tempted to add a try/except,
  a special case, or a flag — stop. Find the root cause.
- Do not spend time tuning an approach that cannot hit the target
  performance envelope. Throw it out and try a different shape.
- A provided PRD or tasklist is sufficient authority. Do not stop for fallback questions or preference polls about local implementation choices; resolve them from repo principles and continue.
- Interrupt only for real external blockers: missing credentials, required sandbox or network approval, destructive irreversible actions not already authorized, or contradictory requirements.
- Prefer ASCII in new files unless Unicode is already established.
- Keep docs short, specific, and discoverable by agents.
- Avoid broad abstractions before the test suite forces them.
- When fallback behavior changes, update both code and related docs/tests.
- Do not rename or reorganize vendored upstream tests without cause.

If you are uncertain about the right approach, that uncertainty is a
signal to investigate MORE, not to pick the fastest path. Enumerate
at least two structurally different solutions before implementing.

## End-to-End Profile Gate (MANDATORY)

When you modify or add functionality that touches runtime, kernel,
pipeline, IO, or predicate code paths, you MUST run a full end-to-end
profile before work is considered complete:

```bash
uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline
```

After the profile completes:

1. **Review every stage** in the sparkline output. Identify any stage
   that takes a disproportionate share of wall time relative to the
   work it should be doing.
2. **Flag unexpected CPU-heavy stages.** If a stage that should be
   fast (buffer subsetting, mask construction, candidate filtering)
   shows >1s at 1M scale, investigate. Common causes: Python-level
   loops over geometry objects, Shapely round-trips, object-dtype
   array iteration.
3. **Resolve before landing.** Irregularities and unexplained CPU
   bottlenecks are not acceptable as known issues to fix later. Either
   fix them or document in an ADR why the cost is inherent.
4. **Include the profile summary** (stage names + times for the 1M
   scale run) in your commit message or handoff notes.

The goal: never ship a stage that looks fast on paper (dispatched to GPU,
correct results) but is actually dominated by host-side overhead.
See ADR-0032 for the canonical example.

## Landing

All landing is handled by the `$commit` skill. Run `$commit` when work
is complete. It orchestrates:

1. `$pre-land-review` — deterministic checks + AI sub-agent reviews
2. Staging and review marker generation
3. Git commit with content-addressable review verification

**The `$commit` skill will reject your work if:**
- Any deterministic check fails (ruff, ARCH, ZCOPY, VPAT, MAINT, IGRD)
- AI review finds BLOCKING issues (all findings default to BLOCKING)
- The review marker hash doesn't match the staged diff

**After committing, you MUST push:**
```bash
git pull --rebase && git push
```
Work is NOT complete until `git push` succeeds. Never stop before
pushing. Never say "ready to push when you are" — YOU must push.

**Session completion checklist:**
1. File issues for remaining work
2. Push to remote (MANDATORY)
3. Hand off context for next session
