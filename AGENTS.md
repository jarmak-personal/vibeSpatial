# Agent Instructions

<!-- DOC_HEADER:START
Scope: Repository-wide agent workflow, intake usage, and verification expectations.
Read If: You are starting, routing, or landing work in this repository.
STOP IF: You only need a narrow API detail already covered by a routed doc.
Source Of Truth: Agent workflow and handoff policy for vibeSpatial.
Body Budget: 242/260 lines
Document: AGENTS.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Preamble |
| 7-11 | Intent |
| 12-20 | Request Signals |
| 21-27 | Open First |
| 28-32 | Verify |
| 33-38 | Risks |
| 39-47 | Mission |
| 48-57 | Startup |
| 58-79 | Routing |
| 80-105 | Project Shape |
| 106-123 | Execution Model |
| 124-135 | Test Strategy |
| 136-145 | Build And Tooling |
| 146-163 | Verification |
| ... | (4 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

This repository is an agent-maintained spatial analytics project with a
GPU-first execution model. Optimize for fast intake, explicit fallbacks, and
steady progress toward GeoPandas API coverage.

## Intent

Define the agent workflow for routing requests, selecting files, verifying
changes, and landing work without silent fallback regressions.

## Request Signals

- intake
- agent workflow
- docs-only
- verification
- repo map
- handoff

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

- Build a new NVIDIA GPU-accelerated spatial analytics library from scratch.
- Prefer `cuda-python` and CCCL primitives over inheriting cuSpatial design.
- Use the upstream GeoPandas test suite as the compatibility contract.
- Keep CPU fallback available, but never silent and never the design center.
- PERFORMANCE IS KING: consider performance implications in every design and implementation decision.
- If an approach has the wrong performance shape, throw it out instead of polishing it.

## Startup

1. Read `docs/ops/intake.md` or run `uv run python scripts/intake.py "<request>"`.
2. Open only the routed files/docs first.
3. Run `uv run python scripts/health.py --check` when you need repo-wide session orientation.
4. Read relevant ADRs from `docs/decisions/index.md` when the task reopens a design decision.
5. Make the smallest coherent change that improves the target contract.
6. Run the narrowest meaningful verification before expanding scope.
7. Refresh generated docs and intake artifacts with `uv run python scripts/check_docs.py --refresh` when doc metadata changes.

## Routing

Use `docs/ops/intake.md` as the source of truth for:

- request classification
- first files to inspect
- expected verification commands
- when to open architecture or testing docs

Generated headers and `docs/ops/intake-index.json` turn those docs into
machine-readable routing input. Do not front-load full-repo reads. Route,
inspect the local area, then expand.

### Many-vs-One Overlay (N-vs-1 Pattern)

When a task involves N-vs-1 overlay patterns (e.g., clipping many features
against a single corridor or boundary polygon), route to:

- `src/vibespatial/overlay/strategies.py` -- strategy detection (broadcast_right workload shape)
- `src/vibespatial/overlay/gpu.py` -- three-tier GPU dispatch: (1) containment bypass for polygons fully inside the clip polygon, (2) batched Sutherland-Hodgman clip for boundary-crossing simple polygons, (3) per-group overlay for complex remainder
- `src/vibespatial/kernels/predicates/point_in_polygon.py` -- GPU bulk vertex-in-polygon used by containment bypass

## Project Shape

- `src/vibespatial/`: Python package and runtime selection logic.
- `docs/ops/intake.md`: lightweight intake router for agents.
- `docs/architecture/runtime.md`: GPU-first execution and fallback rules.
- `scripts/intake.py`: CLI helper for request routing.
- `scripts/check_docs.py`: refresh and validate generated doc headers and intake index.
- `scripts/health.py`: summarize repo coverage, tests, lint, and docs.
- `scripts/pre_land.py`: run pre-landing verification checks.
- `scripts/generate_kernel_scaffold.py`: create repo-standard kernel, test, benchmark, and manifest stubs.
- `scripts/new_decision.py`: create ADRs and refresh the decision index.
- `scripts/install_githooks.py`: install `.githooks/pre-commit` for auto-refresh.
- `scripts/vendor_geopandas_tests.py`: refresh vendored GeoPandas tests.
- `scripts/extract_vendor_to_api.py`: extract vendored GeoPandas surfaces into the repo-owned API layer.
- `scripts/bench_compact_gather.py`: compact+gather micro-benchmark (CuPy vs CCCL vs NVRTC).
- `vsbench` (entry point): unified benchmarking CLI for operations, pipelines, kernel microbenchmarks, regression detection, and geopandas-vs-vibespatial shootout comparisons. See `src/vibespatial/bench/cli.py`.
- `scripts/benchmark_pipelines.py`: end-to-end pipeline benchmarking and GPU sparkline profiling (legacy; prefer `vsbench suite`).
- `scripts/check_architecture_lints.py`: validate architecture constraints and doc consistency.
- `scripts/check_zero_copy.py`: zero-copy device transfer enforcement (ZCOPY001-003).
- `scripts/check_perf_patterns.py`: performance anti-pattern detection (VPAT001-004).
- `scripts/check_maintainability.py`: intake discoverability enforcement (MAINT001-003).
- `src/geopandas/`: local GeoPandas-compatible package surface owned by this repo.
- `src/vibespatial/api/`: public API dispatch boundary for GeoPandas-facing methods.
- `src/vibespatial/kernels/`: scaffolded owned kernel modules and variant manifest.
- `tests/upstream/geopandas/`: copied upstream GeoPandas contract tests.

## Execution Model

- Default policy is `auto`: use GPU when available, otherwise CPU.
- Any explicit CPU fallback must be observable in code, logs, or test output.
- Do not hide unsupported GPU behavior behind silent host-side execution.
- Prefer data-parallel kernels and bulk columnar transforms over Python loops.
- Reach for CCCL or `cuda-python` building blocks before inventing custom glue.
- CCCL now has a Python interface; review the docs before designing new GPU
  primitives: <https://nvidia.github.io/cccl/unstable/python/compute_api.html>

Borrow from cuDF:

- explicit fast/slow mode selection
- profiler or diagnostics that reveal fallback
- tests that keep CPU and GPU behavior aligned

Do not borrow cuSpatial architecture by default. Re-justify every design.

## Test Strategy

- Treat vendored GeoPandas tests as upstream contract coverage.
- Keep the vendored tree refreshable; do not hand-edit copied tests casually.
- Patch vendored imports only when required to keep the suite self-contained.
- Add repo-local tests for new runtime, kernel, and fallback behavior.
- New kernel tests should use the Shapely oracle fixture pattern so CPU/GPU
  paths compare against the same mechanical host reference.
- New owned kernels should start from `uv run python scripts/generate_kernel_scaffold.py <kernel-name>`.
- Before landing work, run `uv run python scripts/pre_land.py`.
- Prefer targeted smoke runs while bootstrapping, then expand to broader groups.

## Build And Tooling

- Use `uv` for environment and dependency management.
- Keep the default bootstrap path lightweight enough to run test smoke checks.
- Put GPU-specific dependencies behind focused dependency groups when practical.
- If adopting CCCL Python, prefer a first-class repo refactor to native
  CCCL-compatible device arrays and call sites rather than layering a temporary
  adapter over the current raw-pointer runtime.
- Install `.githooks/pre-commit` with `uv run python scripts/install_githooks.py` if you want auto-refreshed generated docs and architecture lint checks before commit.

## Verification

- Packaging or env changes: `uv sync`
- Intake/doc changes (local only, enforced by pre-commit hook, not CI): `uv run python scripts/check_docs.py --check && uv run python scripts/intake.py "<request>"`
- Runtime/package changes: `uv run pytest`
- Pipeline benchmark / profiler changes: `uv run pytest tests/test_pipeline_benchmarks.py tests/test_profiling_rails.py -q && uv run python scripts/benchmark_pipelines.py --suite smoke --repeat 2`
- Vendored test refresh: `uv run python scripts/vendor_geopandas_tests.py`
- Architecture lint: `uv run python scripts/check_architecture_lints.py`
- Zero-copy lint: `uv run python scripts/check_zero_copy.py --all`
- Performance lint: `uv run python scripts/check_perf_patterns.py --all`
- Maintainability lint: `uv run python scripts/check_maintainability.py --all`
- Ruff lint: `uv run ruff check`
- AI pre-land review: `/commit` (runs /pre-land-review automatically)
- Upstream smoke: `uv run pytest tests/upstream/geopandas/tests/test_config.py`

If verification cannot run because dependencies, drivers, or local services are
missing, say so concretely and include the command that failed.

## Working Rules

- Prefer ASCII in new files unless a file already requires Unicode.
- Keep docs short, specific, and discoverable by agents.
- Avoid broad abstractions before the test suite forces them.
- Do not spend time tuning an approach that cannot plausibly hit the target performance envelope.
- Do not rename or reorganize vendored upstream tests without cause.
- When fallback behavior changes, update both code and the related docs/tests.

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
   fix them or document in an ADR why the cost is inherent and cannot
   be avoided.
4. **Include the profile summary** (stage names + times for the 1M
   scale run) in your commit message or handoff notes so the next
   session has a baseline.

The goal is to never ship a stage that looks fast on paper (dispatched
to GPU, correct results) but is actually dominated by host-side
overhead. See ADR-0032 for the canonical example.

## Landing (MANDATORY)

Before committing or ending a session, you MUST complete every applicable step.
This checklist is also enforced by the `/pre-land-review` skill which fires
automatically when you attempt to commit.

1. Refresh vendored tests if upstream copy logic changed.
2. Run the narrow verification gate for the edited surface.
3. Run the end-to-end profile gate if runtime/kernel/pipeline code changed.
4. **Run `/commit`** which orchestrates pre-land review, staging, and commit.
5. Update docs that define the changed workflow or invariant.
7. Report any blockers, especially GPU availability.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
