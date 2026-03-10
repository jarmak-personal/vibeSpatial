# Agent Instructions

<!-- DOC_HEADER:START -->
> [!IMPORTANT]
> This block is auto-generated. Edit metadata in `docs/doc_headers.json`.
> Refresh with `uv run python scripts/check_docs.py --refresh` and validate with `uv run python scripts/check_docs.py --check`.

**Scope:** Repository-wide agent workflow, intake usage, and verification expectations.
**Read If:** You are starting, routing, or landing work in this repository.
**STOP IF:** You only need a narrow API detail already covered by a routed doc.
**Source Of Truth:** Agent workflow and handoff policy for vibeSpatial.
**Body Budget:** 197/260 lines
**Document:** `AGENTS.md`

**Section Map (Body Lines)**
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
| 58-70 | Routing |
| 71-92 | Project Shape |
| 93-110 | Execution Model |
| 111-122 | Test Strategy |
| 123-132 | Build And Tooling |
| 133-146 | Verification |
| ... | (3 additional sections omitted; open document body for full map) |
<!-- DOC_HEADER:END -->

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
- `scripts/upstream_native_coverage.py`: strict-native GeoPandas coverage analysis.
- `scripts/benchmark_pipelines.py`: end-to-end pipeline benchmarking and GPU sparkline profiling.
- `scripts/check_architecture_lints.py`: validate architecture constraints and doc consistency.
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
- Intake/doc changes: `uv run python scripts/check_docs.py --check && uv run python scripts/intake.py "<request>"`
- Runtime/package changes: `uv run pytest`
- Pipeline benchmark / profiler changes: `uv run pytest tests/test_pipeline_benchmarks.py tests/test_profiling_rails.py -q && uv run python scripts/benchmark_pipelines.py --suite smoke --repeat 2`
- Strict-native GeoPandas coverage: `uv run python scripts/upstream_native_coverage.py --json`
- Vendored test refresh: `uv run python scripts/vendor_geopandas_tests.py`
- Architecture lint: `uv run python scripts/check_architecture_lints.py`
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

## Landing

Before ending a session, you must:

1. Refresh vendored tests if upstream copy logic changed.
2. Run the narrow verification gate for the edited surface.
3. Run the end-to-end profile gate if runtime/kernel/pipeline code changed.
4. Update docs that define the changed workflow or invariant.
5. Commit with a clear message and include current strict-native GeoPandas coverage in the commit message from `uv run python scripts/upstream_native_coverage.py --json`.
6. Report any blockers, especially GPU availability.
