# IO Performance Rails

<!-- DOC_HEADER:START
Scope: IO benchmark rail suites, throughput floors, and format-level performance enforcement.
Read If: You are adding or verifying IO benchmark rails, throughput floors, or format-specific performance gates.
STOP IF: Your task already has the benchmark scripts open and only needs local implementation detail.
Source Of Truth: Phase-6d IO performance rail and floor enforcement workflow.
Body Budget: 123/220 lines
Document: docs/testing/io-performance-rails.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-10 | Intent |
| 11-18 | Request Signals |
| 19-24 | Open First |
| 25-30 | Verify |
| 31-36 | Risks |
| 37-56 | Entry Points |
| 57-68 | Suites |
| 69-98 | Result Contract |
| 99-114 | Current Enforcement |
| 115-123 | Notes |
DOC_HEADER:END -->

This repo has public IO benchmark entry points plus internal component rails for
the accelerated format work in Phase 6d.

## Intent

Turn IO performance claims into repeatable suite artifacts without forcing
private fast paths through user-facing benchmark scripts.

## Request Signals

- io benchmark
- io performance
- throughput floor
- format benchmark
- io rail

## Open First

- docs/testing/io-performance-rails.md
- scripts/benchmark_io_arrow.py
- scripts/benchmark_io_file.py

## Verify

- `uv run python scripts/benchmark_io_arrow.py --suite smoke`
- `uv run python scripts/benchmark_io_file.py --suite smoke`
- `uv run python scripts/check_docs.py --check`

## Risks

- Drift in performance baselines if the rail does not run regularly.
- GeoJSON informational-only status masking real regression.
- Throughput floors set too low to catch meaningful regressions.

## Entry Points

Run the public Arrow/WKB API suite:

```bash
uv run python scripts/benchmark_io_arrow.py --suite all
```

Run the public file-format API suite:

```bash
uv run python scripts/benchmark_io_file.py --suite all
```

Both entry points also accept `--suite smoke` and `--suite ci`. These script
suites call the registered public `vsbench` operations only. Low-level
GeoArrow, WKB, GeoParquet planner/scan, and direct-parser file component rails
remain internal health surfaces under `src/vibespatial/bench/io_benchmark_rails.py`
and `tests/test_io_benchmark_rails.py`.

## Suites

- `smoke`: local verification at `10K`.
- `ci`: `100K`-class runs intended for pull requests.
- `all`: `10K`, `100K`, and `1M` where practical.
- Internal component rails may use smaller high-cost scales for polygon-heavy
  paths.

`10M` is still treated as a manual deep-run scale for the cheapest point-heavy
paths. It is not part of the default `all` suite because the rail is meant to
run often enough to catch drift, not only on rare manual sweeps.

## Result Contract

Public script cases report the standard `BenchmarkResult` schema:

- operation, scale, geometry type, input format, and status
- timing summary
- reference baseline timing and speedup when available
- operation metadata

Internal component rail cases report:

- `rows_input`
- `rows_decoded`
- `bytes_scanned`
- `copies_made`
- `fallback_pool_share`
- baseline and candidate throughput when a speedup target exists
- `target_floor`
- `status`

The status vocabulary is:

- `pass`
- `fail`
- `informational`
- `unavailable`

`informational` is reserved for internal component cases that are tracked but
not currently enforced.

## Current Enforcement

Internal component floors currently cover:

- GeoArrow aligned import and export
- native GeoArrow decode and encode
- WKB decode and encode
- GeoParquet selective decode avoidance
- GeoParquet GPU scan when `pylibcudf` is available
- Shapefile ingest for point-heavy, line-heavy, and polygon-heavy inputs

Informational-only cases currently cover:

- GeoJSON ingest
- mixed-family WKB decode watch runs

## Notes

- `copies_made` is a conservative path-level estimate, not a byte-accurate
  allocator trace.
- GeoJSON is present in the rail so drift stays visible, but it is not part of
  the hard floor gate until the device-rowized tokenizer path actually wins.
- Pair these rails with [profiling-rails.md](profiling-rails.md)
  and [pipeline-benchmarks.md](pipeline-benchmarks.md)
  when tracing CPU/GPU movement through end-to-end workloads.
