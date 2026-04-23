# Pipeline Benchmarks

<!-- DOC_HEADER:START
Scope: End-to-end pipeline benchmark suites, regression thresholds, and CI artifact workflow.
Read If: You are changing pipeline benchmarks, regression gates, or CPU/GPU movement profiling in CI.
STOP IF: You already have the benchmark scripts open and only need a local implementation detail.
Source Of Truth: Phase-1 pipeline benchmark and regression-gate workflow for end-to-end performance tracking.
Body Budget: 201/220 lines
Document: docs/testing/pipeline-benchmarks.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-12 | Intent |
| 13-29 | Request Signals |
| 30-44 | Open First |
| 45-54 | Verify |
| 55-63 | Risks |
| 64-100 | Entry Points |
| 101-115 | Pipelines |
| 116-133 | Suites |
| 134-142 | Regression Rules |
| 143-182 | Trace Contract |
| 183-201 | CI Workflow |
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
- vsbench
- bench cli
- benchmark operation
- benchmark suite
- benchmark compare
- nvbench kernel
- shootout
- geopandas vs vibespatial

## Open First

- docs/testing/pipeline-benchmarks.md
- src/vibespatial/bench/cli.py
- src/vibespatial/bench/catalog.py
- src/vibespatial/bench/runner.py
- src/vibespatial/bench/schema.py
- src/vibespatial/bench/fixtures.py
- src/vibespatial/bench/fixture_loader.py
- src/vibespatial/bench/pipeline.py
- src/vibespatial/bench/compare.py
- src/vibespatial/bench/shootout.py
- scripts/benchmark_pipelines.py
- .github/workflows/pipeline-benchmarks.yml

## Verify

- `uv run vsbench list operations`
- `uv run vsbench run bounds --scale 1k --repeat 1 --quiet`
- `uv run vsbench fixtures generate --scale 1k --format parquet`
- `uv run vsbench compare baseline.json current.json`
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

Pipeline benchmarks default to `--profile-mode lean`, which keeps wall-clock
stage timing plus runtime D2H count/byte/seconds counters. Use
`--profile-mode audit` when you need NVML samples and CUDA event stage timing.
`--gpu-trace` and `--gpu-sparkline` imply audit mode.

Compare a current run against a baseline artifact:

```bash
uv run vsbench compare baseline.json current.json
```

Discover operation-specific arguments before running a benchmark:

```bash
uv run vsbench list operations --json
uv run vsbench run clip-rect --arg kind=polygon --arg rect=100,100,700,700
uv run vsbench run bounds-pairs --rows 20000 --arg dataset=uniform --arg tile_size=256
```

Default operation listings and suites are public-API benchmarks only. Internal
owned-array or kernel diagnostics are hidden from `vsbench list operations` and
excluded from `vsbench suite`; use `--include-internal` or `vsbench kernel`
when you explicitly want private-path diagnostics.

`vsbench suite` runs serially and isolates each operation, pipeline, or kernel
item in a child process by default. That keeps CUDA allocator state and OOM
failures from bleeding across benchmark items. Use `--in-process` only for
local debugging when you intentionally want the old single-process behavior.

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

The suite CLI enforces per-item timeouts with `--item-timeout N` for isolated
runs. On timeout it kills only the owned child process group and records any
remaining non-orchestrator `nvidia-smi` compute apps in result metadata; it
does not kill unrelated GPU work on the machine.

## Regression Rules

The regression checker currently fails when:

- wall-clock grows by more than `5%`
- peak device memory grows by more than `10%`
- CUDA-runtime D2H transfer count increases
- host materialization count increases

## Trace Contract

Each pipeline result includes:

- top-level `selected_runtime`
- `planner_selected_runtime`
- `transfer_count`
- `owned_transfer_count`
- `runtime_d2h_transfer_count`
- `runtime_d2h_transfer_bytes`
- `runtime_d2h_transfer_seconds`
- `materialization_count`
- `peak_device_memory_bytes`
- stage traces with per-stage `device`

`transfer_count` is the runtime D2H count in current artifacts. Older
artifacts used it for owned-array residency transfer diagnostics, so new
artifacts also include `owned_transfer_count` to keep that semantic boundary
visible without hiding internal runtime copies.

When a pipeline runs partly on GPU and partly on CPU, `selected_runtime` becomes
`hybrid`. This is intentional. The benchmark rail reports what actually
executed, not what the planner wished would execute.

Each stage may also carry:

- `requested_backend` / `actual_backend`
- `requested_mode` / `actual_mode`
- `fallback_note`
- `transfer_count_delta`
- `owned_transfer_count_delta`
- `runtime_d2h_transfer_count_delta`
- `runtime_d2h_transfer_bytes_delta`
- `runtime_d2h_transfer_seconds_delta`
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
`uv run vsbench compare`.

Bootstrap note:

- if the base commit predates these scripts, the workflow uploads the current
  artifact and records the baseline comparison as unavailable instead of
  pretending the gate ran
