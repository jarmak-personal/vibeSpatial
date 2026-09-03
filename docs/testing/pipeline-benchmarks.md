# Pipeline Benchmarks

<!-- DOC_HEADER:START
Scope: End-to-end pipeline benchmark suites, regression thresholds, and CI artifact workflow.
Read If: You are changing pipeline benchmarks, regression gates, or CPU/GPU movement profiling in CI.
STOP IF: You already have the benchmark scripts open and only need a local implementation detail.
Source Of Truth: Phase-1 pipeline benchmark and regression-gate workflow for end-to-end performance tracking.
Body Budget: 229/230 lines
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
| 64-119 | Entry Points |
| 120-134 | Pipelines |
| 135-152 | Suites |
| 153-166 | Regression Rules |
| 167-206 | Trace Contract |
| 207-229 | Automation State |
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

Pipeline benchmarks default to `--profile-mode lean`, retaining wall-clock and
runtime D2H counters; audit mode adds NVML/CUDA timing. GPU traces imply audit.
Nested RMM counters scope peak memory to each sample, excluding process-global
high-water marks left by precompile or earlier pipelines.

Compare a current run against a baseline artifact:

```bash
uv run vsbench compare baseline.json current.json
```

For public shootouts, reuse a static GeoPandas leg while rerunning the current
vibeSpatial source:

```bash
uv run vsbench shootout benchmarks/shootout --scale 10k --repeat 3 \
  --reuse-geopandas baseline.json --json --output current.json
```

Reuse is fail-closed. The artifact must carry the current workload-tree and
measurement-contract hashes, scale, repeat/warmup/timeout settings, host
identity, GeoPandas/Python package identity, and correctness fingerprint.
Refresh the GeoPandas baseline only when one of those inputs changes.

Current shootout and pipeline artifacts also carry `vibespatial_source`: the
imported package path, Git revision, source-only dirty state, untracked source
files, and a SHA-256 over the `src/`, `scripts/`, and `benchmarks/` worktree.
This binds candidate timing to production code without invalidating evidence
when only its report documentation changes.

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

The SpatialBench scale-rail CLI normalizes exact refinement and aggregate D2H by
input rows. Build/reuse transitions are categorical, not slope metrics. Q6 also
requires a measured tier with an ancestor above the one-shot threshold, a
compact derivative below it, and one build followed by reuse.

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

## Automation State

`.github/workflows/pipeline-benchmarks.yml` runs base/current comparison on pull
requests, pushes to `main`, and manual dispatches. The GitHub-hosted CPU rail is
deliberately restricted to the three pipelines whose exact public/reference
implementations are CPU-capable: `constructive`, `predicate-heavy`, and
`predicate-heavy-geopandas`. Pull requests use the `ci` 100K scale; push and
manual runs use `full` at 100K and 1M. Both revisions must report actual runtime
`cpu`, matching result sets, successful statuses, and source-revision
provenance before `uv run vsbench compare` gates them.

The complete Native suite is compared only on a self-hosted NVIDIA runner. It
runs automatically for pushes and same-repository pull requests when the
repository variable `VIBESPATIAL_GPU_RUNNER_AVAILABLE` is `true`, or for an
explicit manual `run_gpu` dispatch. Fork pull requests never execute on the
self-hosted runner. The GPU rail records hardware identity, full sparklines,
actual GPU/hybrid runtime, and base/current comparison artifacts.

GitHub Actions cannot discover whether a labeled self-hosted runner is online,
so runner availability remains an explicit repository contract. If a base
revision predates source-provenance support or otherwise cannot enter the
comparison contract, the workflow writes an `unavailable` comparison artifact
with the reason instead of claiming that a regression gate ran.
