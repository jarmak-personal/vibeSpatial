# GPU Performance Audit Checklist

<!-- DOC_HEADER:START
Scope: Detailed GPU performance audit workflow, anti-pattern checklist, and repo-specific remediation queue.
Read If: You are auditing GPU performance, reviewing CuPy or CCCL usage, or checking whether a path is still CPU-shaped.
STOP IF: You already have the specific hot path open and only need a local implementation detail.
Source Of Truth: Repo-specific GPU performance audit checklist and prioritization guide.
Body Budget: 422/460 lines
Document: docs/testing/gpu-performance-checklist.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-9 | Intent |
| 10-22 | Request Signals |
| 23-32 | Open First |
| 33-39 | Verify |
| 40-53 | Risks |
| 54-69 | When To Use This Checklist |
| 70-93 | Current Verdict |
| 94-111 | Baseline Evidence |
| 112-129 | Severity Model |
| 130-145 | Audit Workflow |
| 146-164 | Fast Triage Commands |
| 165-166 | Checklist |
| 167-187 | Runtime And Dispatch |
| ... | (13 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Provide a detailed, repeatable audit checklist for finding GPU performance
problems in vibeSpatial. This document is meant to be used during deep dives,
pre-land reviews, profiling sessions, and broad "does this code still think
like CPU code?" investigations.

## Request Signals

- gpu performance audit
- performance checklist
- cupy review
- cccl review
- sync audit
- transfer audit
- host orchestration
- cpu-shaped gpu code
- stream audit
- occupancy audit

## Open First

- docs/testing/gpu-performance-checklist.md
- docs/architecture/runtime.md
- docs/architecture/residency.md
- docs/testing/performance-tiers.md
- docs/testing/profiling-rails.md
- src/vibespatial/cuda/_runtime.py
- src/vibespatial/cuda/cccl_primitives.py

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/profile_kernels.py --kernel all --rows 10000 --repeat 1`
- `uv run python scripts/health.py --gpu-coverage`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Fast-looking GPU code can still be host-bound if control flow, allocation,
  or grouping decisions happen in Python.
- Scalar device reads can look harmless in review while silently forcing full
  stream or context synchronization.
- "Device-resident" APIs can still pay hidden D2H costs if they eagerly build
  host mirrors for convenience.
- Stream usage can be misleading when upstream CCCL or NVRTC launch helpers
  still force the null stream.
- Small benchmarks can hide transfer, launch, and orchestration costs.

This checklist was built from a repo audit performed on April 7, 2026.

## When To Use This Checklist

Use this document when any of the following is true:

- a GPU path is slower than expected
- a pipeline looks GPU-native in architecture docs but profiles as CPU-heavy
- an operation uses CuPy, CCCL, NVRTC, or `cuda-python`
- a reviewer suspects excessive synchronization or transfer churn
- a new kernel or dispatch path is being added and you want to avoid copying
  existing mistakes

Use it as both:

- a discovery checklist while auditing existing code
- a pre-merge checklist before landing new GPU work

## Current Verdict

The current repo does have serious GPU-first infrastructure:

- occupancy-based launch sizing exists in `src/vibespatial/cuda/_runtime.py`
- async count-scatter total helpers exist in `src/vibespatial/cuda/_runtime.py`
- hardware-aware precision policy exists in `src/vibespatial/runtime/precision.py`
- parts of overlay already use real stream overlap

The current repo also still has multiple CPU-shaped GPU surfaces:

- host-side orchestration in overlay microcells and grouped overlay
- eager device-to-host metadata mirroring in pylibcudf device builders
- Python loops driven by device-reduced maxima in WKB decode
- unconditional same-stream synchronization in point-in-polygon helpers
- CCCL wrappers that default to null-stream synchronization instead of caller-
  controlled completion

The conclusion is not "the framework is fake GPU." The conclusion is:

- the framework is capable
- adoption is uneven
- the worst problems are structural rather than cosmetic

## Baseline Evidence

As of April 7, 2026:

- `uv run python scripts/profile_kernels.py --kernel all --rows 10000 --repeat 1`
  selected CPU for both join and overlay at 10K rows on the local RTX 4090
- the profiler reported effectively 0% GPU utilization for those profiled runs
- `uv run python scripts/health.py --gpu-coverage` reported:
  - GPU available: `true`
  - total dispatches: `10134`
  - GPU dispatches: `400`
  - CPU dispatches: `9154`
  - fallback dispatches: `361`
  - GPU acceleration rate: `3.95%`

Treat those numbers as a dated snapshot, not a permanent truth. Re-run them
before claiming improvement.

## Severity Model

Use this rubric while filing findings:

- `BLOCKING`
  A structural issue that prevents a path from scaling as a GPU path at all.
  Examples: host loops over device work units, repeated D2H reads inside a hot
  loop, unconditional same-stream syncs between stages, eager full metadata
  materialization on every "device" decode.
- `HIGH`
  A design or implementation issue that will materially cap throughput or
  destroy overlap on realistic workloads.
- `MEDIUM`
  A repeated anti-pattern that may not dominate every dataset, but should be
  cleaned up because it compounds with other costs.
- `LOW`
  A legitimate clean-up item or a suspicious pattern that needs measurement.

## Audit Workflow

Run the audit in this order:

1. Confirm whether the path actually ran on GPU.
2. Confirm whether the path stayed on GPU after it got there.
3. Confirm whether the path avoided same-stream synchronization.
4. Confirm whether the path avoided Python-controlled batching or grouping.
5. Confirm whether kernel launch and primitive selection are appropriate for
   the hardware.
6. Confirm whether the benchmark or profiler surface is measuring the real path,
   not the planner's intent.

If a path fails an earlier step, do not waste time micro-optimizing later
steps first.

## Fast Triage Commands

Use these search rails early:

```bash
rg -n "cp\\.asnumpy\\(|\\.get\\(\\)|\\.item\\(|runtime\\.synchronize\\(|cp\\.cuda\\.Stream\\.null\\.synchronize\\(" src/vibespatial -g'*.py'
rg -n "for .* in range\\(max_|for .* in cp\\.asnumpy|for .* in .*tolist\\(" src/vibespatial -g'*.py'
rg -n "count_scatter_total\\(|launch_config\\(|block = \\(256, 1, 1\\)|block=\\(256, 1, 1\\)" src/vibespatial -g'*.py'
rg -n "copy_device_to_host|to_shapely\\(|to_pandas\\(|to_numpy\\(" src/vibespatial -g'*.py'
```

Interpretation rules:

- a grep hit is not automatically a bug
- a hit inside a hot loop is much more suspicious than a hit at a terminal
  materialization boundary
- `runtime.synchronize()` between same-stream stages is almost always wrong
  unless it is trace-only or required by a host read immediately afterward

## Checklist

## Runtime And Dispatch

- [ ] Confirm the path records both requested runtime and selected runtime.
- [ ] Confirm the profiler or benchmark reports actual execution device, not
  only planner intent.
- [ ] Confirm `auto` does not demote already-device-resident workloads back to
  CPU only because row counts look small.
- [ ] Confirm explicit GPU mode fails loudly instead of silently falling back.
- [ ] Confirm fallback events remain observable.
- [ ] Confirm precision planning matches hardware class.
- [ ] Confirm consumer GPUs are not doing unnecessary fp64 for predicate or
  metric kernels when `PrecisionPlan` would allow fp32.
- [ ] Confirm constructive kernels that stay fp64 do so by policy, not by
  accidental hardcoding.

Pass if:

- the path actually executes on GPU when it should
- the runtime record explains why
- there is no silent fallback

## Memory Pools And Allocation

- [ ] Confirm the path uses the configured pool allocator instead of raw
  allocation churn.
- [ ] Confirm temporary allocations are not repeatedly materialized and freed
  inside Python loops.
- [ ] Confirm count-scatter paths size outputs once from totals instead of
  resizing incrementally.
- [ ] Confirm eager pool trimming is not enabled in hot paths unless debugging.
- [ ] Confirm long-lived and short-lived buffers are not interleaved in a way
  that obviously increases fragmentation.
- [ ] Confirm managed memory is not silently used where a pool-backed VRAM path
  is expected.

Fail immediately if:

- raw allocation or pool flushes happen per work unit
- Python loops cause repeated alloc/free cycles that should have been batched

## Streams And Synchronization

- [ ] Confirm consecutive same-stream stages do not call `runtime.synchronize()`
  between launches unless a host read follows immediately.
- [ ] Confirm CCCL helpers or CuPy wrappers are not forcing
  `Stream.null.synchronize()` when the caller could defer completion.
- [ ] Confirm stream usage is real, not cosmetic.
- [ ] Confirm a stream pool is not feeding work into helpers that still
  serialize on the null stream.
- [ ] Confirm async transfer helpers use pinned memory where overlap matters.
- [ ] Confirm stream synchronization is amortized per batch, not per item.

Mark as `BLOCKING` if:

- null-stream synchronization is hardcoded in a reusable primitive wrapper
- every kernel helper ends with an unconditional full-device sync

## Transfers And Residency

- [ ] Confirm device-backed builders do not eagerly copy structural metadata to
  host unless the public boundary requires it.
- [ ] Confirm host materialization happens only at explicit surfaces such as
  `to_pandas`, `to_numpy`, `values`, `__repr__`, or a visible fallback.
- [ ] Confirm D2H copies are not feeding more GPU work unless absolutely
  unavoidable.
- [ ] Confirm small scalar reads are not happening repeatedly where a single
  batched transfer would work.
- [ ] Confirm host mirrors are lazy and cached instead of rebuilt for every GPU
  path invocation.
- [ ] Confirm zero-copy interop stays zero-copy when layouts already align.

Treat these as suspicious:

- `cp.asnumpy(...)` in a mid-pipeline helper
- `.get()` or `.item()` on device scalars in a hot path
- multiple `copy_device_to_host(...)` calls for related metadata that could be
  transferred together

## CuPy Usage

- [ ] Confirm operations are vectorized and bulk-shaped instead of Python loops
  over row ids, ring ids, or group ids.
- [ ] Confirm `cp.flatnonzero`, `cp.searchsorted`, `cp.cumsum`, and
  `cp.concatenate` are used on whole arrays, not in Python loops over groups.
- [ ] Confirm `cp.unique` results do not immediately trigger host loops over
  large work sets.
- [ ] Confirm boolean masks and family partitions stay on device when possible.
- [ ] Confirm `cp.asarray` is not being used to bounce data host -> device ->
  host inside the same pipeline.
- [ ] Confirm any `cp.column_stack` or dense gather step is justified and not a
  hidden quadratic memory move.

Mark as `HIGH` if:

- Python owns the outer control flow and CuPy only performs the inner slices
- device maxima are read to host just to drive `for range(max_...)`

## CCCL Usage

- [ ] Confirm scan, sort, reduce, and binary-search wrappers accept a caller-
  controlled synchronization policy.
- [ ] Confirm wrappers do not always synchronize the null stream.
- [ ] Confirm wrappers can eventually accept streams when backend support exists.
- [ ] Confirm count-returning primitives do not force immediate host scalar
  reads unless the pipeline truly needs a host integer to allocate.
- [ ] Confirm cold-JIT avoidance logic does not lock the steady-state path into
  slower CuPy fallbacks after warmup.
- [ ] Confirm sort and compaction primitives are used where they improve batch
  shape instead of falling back to host partition logic.

Repo-specific reminder:

- the backend layer already exposes stream arguments in the CCCL cached call
  path
- the wrapper layer is the current bottleneck

## Kernel Launch And Occupancy

- [ ] Confirm kernels use `launch_config()` or another occupancy-aware sizing
  rule instead of hardcoded `(256, 1, 1)` without evidence.
- [ ] Confirm work is launched in bulk instead of many small per-group kernels
  dispatched from Python.
- [ ] Confirm launch geometry matches the dominant work dimension.
- [ ] Confirm launch count is not inflated by host-side binning or slicing that
  could happen on device.
- [ ] Confirm shared-memory requirements are passed into block sizing when
  relevant.

Low-priority only if:

- the hardcoded block size is on a cold path or a tiny helper

## IO And Parsing Pipelines

- [ ] Confirm parsers do not decode structure on GPU only to hand control back
  to Python for ring, part, or polygon walking.
- [ ] Confirm offset construction is device-native count-scatter or segmented
  scan where possible.
- [ ] Confirm format-family assembly does not immediately bounce through host
  family discovery unless the number of families is tiny and the cost is
  demonstrably irrelevant.
- [ ] Confirm legacy count-scatter total sites use `count_scatter_total()`
  rather than `runtime.synchronize()` plus multiple `.get()` calls.
- [ ] Confirm "device decode" surfaces do not eagerly materialize host mirrors
  just because downstream host code currently expects them.

Mark as `BLOCKING` if:

- the parser uses Python loops over per-geometry nested structure on the hot
  path

## Predicate And Spatial Query Pipelines

- [ ] Confirm coarse filtering, candidate compaction, and refine all stay on
  device once the workload is on device.
- [ ] Confirm work estimation and binning do not require candidate rows on host.
- [ ] Confirm dense and compacted helper launches do not synchronize before the
  caller needs the result.
- [ ] Confirm candidate row assembly uses device primitives instead of host
  regrouping.
- [ ] Confirm output scattering back into dense arrays does not force an early
  D2H read.

## Overlay And Constructive Pipelines

- [ ] Confirm grouped overlay does not materialize group boundaries to host
  unless the entire pipeline is already falling back.
- [ ] Confirm stream pools are paired with stream-aware launches and primitive
  calls; otherwise treat them as cosmetic concurrency.
- [ ] Confirm microcell labeling and contraction are not row-by-row host loops.
- [ ] Confirm constructive helpers such as clip, shortest line, line buffer,
  and union-all do not synchronize between same-stream stages without a host
  dependency.
- [ ] Confirm tree reductions do not split large GPU workloads into thousands of
  Python-managed one-row objects if a batched alternative is possible.

Mark as `BLOCKING` if:

- overlay correctness may be fine, but topology assembly or contraction is
  fundamentally host-managed

## Profiling And Evidence

- [ ] Capture at least one profiler or benchmark run on the target machine.
- [ ] Record actual GPU name.
- [ ] Record the command, date, scale, and selected runtime.
- [ ] Record whether the path stayed on device end to end.
- [ ] Record the top 3 longest stages and whether they were CPU or GPU.
- [ ] Record which findings are structural and which are incidental.
- [ ] Record which suspicious host reads are legitimate allocation fences.

Do not accept:

- "the code looks GPU-ish"
- "the planner would select GPU"
- "the kernel itself is fast in isolation"

Accept only:

- measured device execution
- stage-level timing
- transfer-aware reasoning

## Current Repo Hot Spots

Use this list as the first remediation queue, not as a substitute for fresh
auditing.

| Area | Files | Why It Matters | Current Diagnosis |
|---|---|---|---|
| Device WKB decode and OGA builders | `src/vibespatial/io/pylibcudf.py` | Foundational ingest path; hidden host mirrors poison downstream residency | Eager D2H metadata copies and Python loops over nested geometry structure |
| Point in polygon | `src/vibespatial/kernels/predicates/point_in_polygon.py` | Core refine primitive used by predicates and constructive work | Unconditional syncs and host-side work estimation in binned mode |
| CCCL wrapper layer | `src/vibespatial/cuda/cccl_primitives.py` | Shared by scan, sort, compaction, binary search, segmented reduce | Null-stream synchronization is embedded in wrappers |
| Overlay microcells | `src/vibespatial/overlay/microcells.py`, `src/vibespatial/overlay/contract.py` | Structural overlay path; expensive if host-managed | Row-by-row host loops and host-side union-find |
| Grouped overlay orchestration | `src/vibespatial/overlay/gpu.py` | High-value constructive path | Host-materialized grouping and Python-controlled per-group dispatch |
| Legacy count-scatter totals | `src/vibespatial/io/fgb_gpu.py`, `src/vibespatial/io/shp_gpu.py` | Easy wins that remove avoidable syncs | Old sync plus `.get()` pattern still present |
| Hardcoded launch geometry | `src/vibespatial/overlay/assemble.py` | Can cap occupancy or hide resource mismatches | Hardcoded 256-thread block despite launch-config support |
| Python tree reduction | `src/vibespatial/constructive/union_all.py` | Correct but poor performance shape at scale | One-row object splitting and host-managed reduction rounds |

## False Positives To Avoid

Do not file these as bugs without stronger evidence:

- a single scalar host read used strictly to size an output allocation
- synchronization guarded only by hotpath tracing or profiling instrumentation
- a tiny host loop over a small fixed set of geometry families when the data
  payload itself stays on device
- explicit materialization at a public host boundary

Examples already considered legitimate or lower priority during the April 7,
2026 audit:

- trace-only sync guards in overlay hotpath helpers
- allocation-fence scalar reads in output sizing code
- runtime precision detection based on actual hardware ratio

## Sign-Off Checklist

Before closing a performance audit, confirm all of the following:

- [ ] I identified whether the path was actually GPU or CPU at runtime.
- [ ] I separated structural blockers from incidental cleanups.
- [ ] I identified whether synchronization was caller-driven or wrapper-driven.
- [ ] I identified whether host reads were terminal, allocation fences, or
  performance bugs.
- [ ] I named the first 3 remediation targets in priority order.
- [ ] I recorded at least one exact benchmark or profile command with date and
  scale.
- [ ] I recorded whether the result supports or disproves the "CPU-shaped GPU
  code" hypothesis for the audited surface.

The audit is complete only when the final note answers two questions:

- What is forcing this path to behave like CPU code today?
- What is the smallest structural change that moves it back toward a true GPU
  execution shape?
