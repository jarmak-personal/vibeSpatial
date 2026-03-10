# IO Performance Rails

<!-- DOC_HEADER:START -->
> [!IMPORTANT]
> This block is auto-generated. Edit metadata in `docs/doc_headers.json`.
> Refresh with `uv run python scripts/check_docs.py --refresh` and validate with `uv run python scripts/check_docs.py --check`.

**Scope:** IO benchmark rail suites, throughput floors, and format-level performance enforcement.
**Read If:** You are adding or verifying IO benchmark rails, throughput floors, or format-specific performance gates.
**STOP IF:** Your task already has the benchmark scripts open and only needs local implementation detail.
**Source Of Truth:** Phase-6d IO performance rail and floor enforcement workflow.
**Body Budget:** 117/220 lines
**Document:** `docs/testing/io-performance-rails.md`

**Section Map (Body Lines)**
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-10 | Intent |
| 11-18 | Request Signals |
| 19-24 | Open First |
| 25-30 | Verify |
| 31-36 | Risks |
| 37-52 | Entry Points |
| 53-68 | Suites |
| 69-92 | Result Contract |
| 93-108 | Current Enforcement |
| 109-117 | Notes |
<!-- DOC_HEADER:END -->

This repo now has a standing IO benchmark rail for the accelerated format work in
Phase 6d.

## Intent

Turn IO performance claims into repeatable suite artifacts with explicit floor
comparisons, instead of relying on one-off benchmark snippets in commit notes.

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

Run the Arrow, Parquet, and WKB suite:

```bash
uv run python scripts/benchmark_io_arrow.py --suite all
```

Run the file-format suite:

```bash
uv run python scripts/benchmark_io_file.py --suite all
```

Both entry points also accept `--suite smoke` and `--suite ci`.

## Suites

- `smoke`
  - local verification only
  - keeps scales to `10K` plus the smallest planner case
- `ci`
  - `100K`-class runs intended for pull requests
- `all`
  - `10K`, `100K`, and `1M` where practical
  - polygon-heavy paths use smaller high-cost scales such as `20K`, `100K`,
    and `250K`

`10M` is still treated as a manual deep-run scale for the cheapest point-heavy
paths. It is not part of the default `all` suite because the rail is meant to
run often enough to catch drift, not only on rare manual sweeps.

## Result Contract

Each case reports:

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

`informational` means the case is tracked but not currently enforced. That is
intentional for GeoJSON and mixed-family watch cases where the fastest measured
path is still not the target GPU-dominant design.

## Current Enforcement

Enforced floors currently cover:

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
