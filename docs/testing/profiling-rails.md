# Profiling Rails

<!-- DOC_HEADER:START -->
> [!IMPORTANT]
> This block is auto-generated. Edit metadata in `docs/doc_headers.json`.
> Refresh with `uv run python scripts/check_docs.py --refresh` and validate with `uv run python scripts/check_docs.py --check`.

**Scope:** Stage-timed profiling entrypoints, NVTX guidance, and trace interpretation for join and overlay work.
**Read If:** You are profiling join or overlay kernels, adding benchmark rails, or trying to explain where time is going.
**STOP IF:** You already have the profiling script open and only need a local implementation detail.
**Source Of Truth:** Stage-level profiling workflow for join and overlay kernel development.
**Body Budget:** 123/220 lines
**Document:** `docs/testing/profiling-rails.md`

**Section Map (Body Lines)**
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-10 | Intent |
| 11-20 | Request Signals |
| 21-27 | Open First |
| 28-33 | Verify |
| 34-41 | Risks |
| 42-64 | Entry Point |
| 65-99 | Stage Contracts |
| 100-109 | Trace Interpretation |
| 110-123 | NVTX |
<!-- DOC_HEADER:END -->

This repo now has a dedicated profiling rail for join and overlay kernel work.

## Intent

Provide one local entrypoint that reports stage-level wall-clock time, actual
execution device, row flow, and Nsight-friendly range boundaries for the
current join and overlay hot paths.

## Request Signals

- profiler
- profiling
- nsight
- nvtx
- benchmark rail
- join profile
- overlay profile

## Open First

- docs/testing/profiling-rails.md
- scripts/profile_kernels.py
- src/vibespatial/profiling.py
- src/vibespatial/profile_rails.py

## Verify

- `uv run python scripts/profile_kernels.py --kernel join --rows 1000 --tile-size 256`
- `uv run python scripts/profile_kernels.py --kernel overlay --rows 500 --tile-size 256`
- `uv run python scripts/check_docs.py --check`

## Risks

- Reporting planned GPU selection instead of actual execution device hides CPU
  fallback costs and makes traces misleading.
- End-to-end timers alone blur sort, filter, and refine costs together.
- Profiling rails that are not machine-readable are hard to diff and easy to
  ignore during performance regressions.

## Entry Point

Run:

```bash
uv run python scripts/profile_kernels.py --kernel all --rows 10000
```

Available kernels:

- `join`
- `overlay`
- `all`

Useful flags:

- `--rows`
- `--join-rows`
- `--overlay-rows`
- `--tile-size`
- `--repeat`
- `--nvtx`

## Stage Contracts

The JSON trace must include stage categories that make the current execution
shape obvious:

- `setup`
- `sort`
- `filter`
- `refine`

Join profiling currently records:

- owned-buffer build
- bounds computation
- Morton sort
- coarse candidate filter
- predicate refine
- output sort

Overlay profiling currently records:

- owned-buffer build
- segment extraction
- segment MBR filter
- exact intersection refine
- reconstruction-event sort

Each stage reports:

- `device`
- `elapsed_seconds`
- `rows_in`
- `rows_out`
- stage metadata such as `pairs_examined`, `ambiguous_pairs`, or `tile_size`

## Trace Interpretation

Top-level `selected_runtime` is the device that actually executed the profiled
stages. If `metadata.planner_selected_runtime` differs, the runtime planner
would prefer GPU on this machine but the current implementation surface still
executed on CPU.

This distinction is intentional. The profiling rail is meant to explain real
execution, not aspirational dispatch.

## NVTX

When the optional `nvtx` Python package is installed, the rail emits NVTX
ranges per stage when `--nvtx` is passed. That makes the same stage boundaries
visible inside external profilers such as Nsight Systems.

Example:

```bash
nsys profile --trace=cuda,nvtx uv run python scripts/profile_kernels.py --kernel all --rows 10000 --nvtx
```

The JSON trace remains the source of truth inside the repo. NVTX is an external
augmentation, not a replacement.
