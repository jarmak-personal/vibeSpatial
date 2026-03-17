# Pipeline Benchmarks

<!-- DOC_HEADER:START
Scope: End-to-end pipeline benchmark suites, regression thresholds, and CI artifact workflow.
Read If: You are changing pipeline benchmarks, regression gates, or CPU/GPU movement profiling in CI.
STOP IF: You already have the benchmark scripts open and only need a local implementation detail.
Source Of Truth: Phase-1 pipeline benchmark and regression-gate workflow for end-to-end performance tracking.
Body Budget: 141/220 lines
Document: docs/testing/pipeline-benchmarks.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-12 | Intent |
| 13-21 | Request Signals |
| 22-29 | Open First |
| 30-35 | Verify |
| 36-44 | Risks |
| 45-58 | Entry Points |
| 59-73 | Pipelines |
| 74-86 | Suites |
| 87-95 | Regression Rules |
| 96-122 | Trace Contract |
| 123-141 | CI Workflow |
DOC_HEADER:END -->

This repo now has a dedicated end-to-end pipeline benchmark rail for regression
gating.

## Intent

Measure whole-pipeline cost, not just kernel microbenchmarks. The rail is meant
to catch regressions from host<->device movement, materialization, allocation
churn, and bad execution-shape changes that do not show up in isolated kernel
timers.

## Request Signals

- pipeline benchmark
- regression gate
- ci perf
- nvtx
- cpu gpu movement
- benchmark artifact

## Open First

- docs/testing/pipeline-benchmarks.md
- scripts/benchmark_pipelines.py
- scripts/check_pipeline_regressions.py
- src/vibespatial/pipeline_benchmarks.py
- .github/workflows/pipeline-benchmarks.yml

## Verify

- `uv run pytest tests/test_pipeline_benchmarks.py tests/test_profiling_rails.py -q`
- `uv run python scripts/benchmark_pipelines.py --suite smoke --repeat 2`
- `uv run python scripts/check_docs.py --check`

## Risks

- Comparing current results to a stale or missing baseline can hide regressions
  or create false confidence.
- Reporting planner-selected GPU instead of actual hybrid execution hides where
  host materialization or transfer churn still dominates.
- Single-run timings are noisy; median-over-repeats is the local source of
  truth for wall-clock regression checks.

## Entry Points

Run the local smoke suite:

```bash
uv run python scripts/benchmark_pipelines.py --suite smoke --repeat 2
```

Compare a current run against a baseline artifact:

```bash
uv run python scripts/check_pipeline_regressions.py --baseline baseline.json --current current.json
```

## Pipelines

The active benchmarked pipelines are:

- `join-heavy`
  - `read_parquet -> build_index -> sjoin_query -> dissolve -> to_parquet`
- `constructive`
  - `read_parquet -> clip -> buffer -> to_parquet`
- `predicate-heavy`
  - `read_geojson -> load cached polygons -> point_in_polygon -> filter -> DGA-backed to_parquet`
- `predicate-heavy-geopandas`
  - `read_geojson(pyogrio-first) -> covers -> filter -> to_parquet`
- `raster-to-vector`
  - currently emitted as `deferred` until Phase 8 polygonize work lands

## Suites

- `smoke`
  - `1K` rows, local verification only
- `ci`
  - `100K` rows, intended for pull requests
- `full`
  - `100K` and `1M` rows, intended for `main` and manual GPU runs

Each pipeline/scale can be repeated with `--repeat N`. Reported wall-clock is
the median elapsed time across repeats. Device memory and movement counters are
reported conservatively from the worst observed sample.

## Regression Rules

The regression checker currently fails when:

- wall-clock grows by more than `5%`
- peak device memory grows by more than `10%`
- host<->device transfer count increases
- host materialization count increases

## Trace Contract

Each pipeline result includes:

- top-level `selected_runtime`
- `planner_selected_runtime`
- `transfer_count`
- `materialization_count`
- `peak_device_memory_bytes`
- stage traces with per-stage `device`

When a pipeline runs partly on GPU and partly on CPU, `selected_runtime` becomes
`hybrid`. This is intentional. The benchmark rail reports what actually
executed, not what the planner wished would execute.

Each stage may also carry:

- `requested_backend` / `actual_backend`
- `requested_mode` / `actual_mode`
- `fallback_note`
- `transfer_count_delta`
- `materialization_count_delta`
- `peak_device_memory_bytes`

That makes CPU<->GPU movement visible in the same artifact as the wall-clock
timing.

## CI Workflow

`.github/workflows/pipeline-benchmarks.yml` runs the suite in two modes:

- CPU job
  - PRs: `ci` suite
  - `main` / manual: `full` suite
- optional GPU job on a self-hosted NVIDIA runner
  - `full` suite with `--nvtx`

The workflow runs the current commit, attempts the same suite on the base
commit in a detached worktree, stores both artifacts, and diffs them with
`scripts/check_pipeline_regressions.py`.

Bootstrap note:

- if the base commit predates these scripts, the workflow uploads the current
  artifact and records the baseline comparison as unavailable instead of
  pretending the gate ran
