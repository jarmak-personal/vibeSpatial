# GPU Performance Remediation Plan

<!-- DOC_HEADER:START
Scope: Execution plan for the next GPU performance push, including milestone sequencing and exit criteria.
Read If: You are planning or executing the next GPU performance remediation campaign.
STOP IF: You already have the active milestone surface open and only need local implementation detail.
Source Of Truth: Program plan for fixing structural GPU performance issues found in the audit.
Body Budget: 460/520 lines
Document: docs/testing/gpu-performance-remediation-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-9 | Intent |
| 10-21 | Request Signals |
| 22-30 | Open First |
| 31-37 | Verify |
| 38-50 | Risks |
| 51-65 | Mission |
| 66-82 | Baseline Snapshot |
| 83-98 | Non-Goals |
| 99-110 | Working Principles |
| 111-125 | Program Structure |
| 126-161 | Milestone M0: Baseline And Guardrails |
| 162-200 | Milestone M1: Residency And Metadata Ownership |
| 201-237 | Milestone M2: CCCL And Synchronization Contract |
| ... | (9 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Turn the GPU performance audit into an execution plan for the next major
performance push. This document defines workstreams, sequencing, milestone
checklists, measurement gates, and completion criteria for fixing the known
CPU-shaped GPU behavior in vibeSpatial.

## Request Signals

- gpu remediation plan
- performance push
- fix gpu performance
- execution plan
- milestone plan
- de-host gpu path
- stream and transfer cleanup
- cccl plan
- overlay performance plan

## Open First

- docs/testing/gpu-performance-remediation-plan.md
- docs/testing/gpu-performance-checklist.md
- docs/architecture/runtime.md
- docs/architecture/residency.md
- docs/testing/performance-tiers.md
- docs/testing/profiling-rails.md

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/profile_kernels.py --kernel all --rows 10000 --repeat 1`
- `uv run python scripts/health.py --gpu-coverage`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Fixing low-level synchronization without changing host orchestration can
  produce cleaner code without materially improving throughput.
- Reworking residency and metadata ownership can regress correctness if public
  host boundaries are not revalidated.
- Stream-enabling CCCL wrappers without a caller-level contract can introduce
  races instead of speedups.
- Overlay and microcell work can absorb the entire push unless earlier tracks
  reduce known structural bottlenecks first.
- GPU coverage numbers can improve without end-to-end speed improving if the
  newly-GPU stages are still dominated by host setup and transfer costs.

## Mission

The next performance push is not a general cleanup. It is a structural rewrite
campaign focused on removing the code shapes that make GPU paths behave like CPU
paths.

The push is successful only if it achieves all of the following:

- device-resident paths remain device-resident longer
- reusable GPU helpers stop forcing null-stream completion
- profiling rails begin selecting GPU for important benchmark surfaces at
  meaningful scales
- end-to-end pipeline profiles show less CPU-dominated orchestration
- GPU acceleration coverage improves materially from the April 7, 2026 baseline

## Baseline Snapshot

This plan starts from the audit snapshot recorded on April 7, 2026:

- `profile_kernels.py --kernel all --rows 10000 --repeat 1`
  selected CPU for both join and overlay on the local RTX 4090
- that run showed effectively 0% GPU utilization for those profiled surfaces
- `health.py --gpu-coverage` reported:
  - total dispatches: `10134`
  - GPU dispatches: `400`
  - CPU dispatches: `9154`
  - fallback dispatches: `361`
  - GPU acceleration rate: `3.95%`

These values are the baseline to beat, not a permanent target. Re-capture them
at the start of the push and again at every milestone boundary.

## Non-Goals

This push is not done when:

- a few `.get()` calls have been removed
- a handful of kernels have nicer launch parameters
- profiler output looks cleaner but execution is still mostly CPU
- GPU dispatch count rises only because tiny helper stages moved to device
- docs claim GPU-first behavior without corresponding runtime evidence

This push is also not about:

- changing user-facing APIs unless required by explicit fallback visibility
- polishing cold-start latency before hot-path execution shape is fixed
- broad architectural abstraction work not tied to measured bottlenecks

## Working Principles

Apply these principles throughout the push:

- fix structural blockers before localized micro-optimizations
- prefer batching, scans, compaction, and segmented primitives over Python loops
- prefer caller-controlled synchronization over helper-controlled
  synchronization
- prefer device-native metadata ownership over convenience host mirrors
- verify with profiler rails and pipeline benchmarks after each milestone
- do not accept "planner would choose GPU" as evidence of real improvement

## Program Structure

This push is divided into six workstreams. They are intentionally ordered. Do
not start with overlay micro-optimizations while residency, CCCL wrappers, and
device-native decode are still structurally wrong.

| Milestone | Name | Primary Surfaces | Why First |
|---|---|---|---|
| M0 | Baseline And Guardrails | profiling, health, docs, benchmark rails | Prevents the push from drifting without evidence |
| M1 | Residency And Metadata Ownership | `io/pylibcudf.py`, owned geometry state | Removes hidden D2H taxes from "device" paths |
| M2 | CCCL And Synchronization Contract | `cuda/cccl_primitives.py`, runtime helper call sites | Unlocks overlap and same-stream composition |
| M3 | Device-Native Decode And Compaction | WKB and related count-scatter paths | Removes host-driven nested decode loops |
| M4 | Predicate And Query Execution Shape | PIP, candidate assembly, work estimation | Fixes a core refine primitive and query path |
| M5 | Overlay And Constructive De-Hosting | grouped overlay, microcells, contraction, union-all | Fixes the highest-value structural CPU orchestration |

## Milestone M0: Baseline And Guardrails

### Goal

Establish current measurements, ensure the rails report actual execution
device, and define hard acceptance gates for the rest of the push.

### Primary Surfaces

- `scripts/profile_kernels.py`
- `src/vibespatial/bench/profiling.py`
- `src/vibespatial/bench/profile_rails.py`
- `scripts/benchmark_pipelines.py`
- `scripts/health.py --gpu-coverage`

### Checklist

- [ ] Re-run the baseline profiler on the target machine.
- [ ] Re-run GPU coverage and record the exact percentages.
- [ ] Capture `nvidia-smi -L`, `/dev/nvidia*`, and `CUDA_VISIBLE_DEVICES`.
- [ ] Confirm profiler rails report actual selected runtime, not only planned
  runtime.
- [ ] Confirm pipeline benchmark stage names are sufficient to identify CPU
  orchestration bottlenecks.
- [ ] Write down the baseline for:
  - GPU acceleration coverage
  - join profiler selected runtime
  - overlay profiler selected runtime
  - pipeline sparkline stage times for the 1M run

### Exit Criteria

- documented baseline is captured for the target machine
- profiler rails and health rails are trusted as evidence sources
- later milestones can compare against a stable before-state

## Milestone M1: Residency And Metadata Ownership

### Goal

Stop calling paths "device-resident" when they eagerly materialize host
metadata during construction.

### Primary Surfaces

- `src/vibespatial/io/pylibcudf.py`
- owned geometry builders and host-state helpers
- residency diagnostics and transfer visibility surfaces

### Known Problems To Fix

- `_build_device_single_family_owned` eagerly copies validity, tags,
  family-row offsets, geometry offsets, empty masks, and optional part/ring
  offsets to host
- `_build_device_mixed_owned` does the same for mixed-family cases
- decode helpers build host mirrors before downstream GPU work requests them

### Checklist

- [ ] Inventory which host arrays are truly required at construction time.
- [ ] Split mandatory public-boundary metadata from convenience mirrors.
- [ ] Make host structural metadata lazy where possible.
- [ ] Preserve explicit materialization events and diagnostics.
- [ ] Re-run transfer audits for any path that starts from pylibcudf decode.
- [ ] Re-check downstream callers that may have been relying on implicit host
  mirrors.
- [ ] Update tests so device-resident outputs do not require host metadata
  unless explicitly materialized.

### Exit Criteria

- device-backed builders no longer force broad D2H copies by default
- downstream GPU consumers can continue from decode without hidden host setup
- transfer counts for decode-to-GPU pipelines decrease measurably

## Milestone M2: CCCL And Synchronization Contract

### Goal

Move synchronization ownership to callers and stop null-stream completion from
being baked into reusable primitive wrappers.

### Primary Surfaces

- `src/vibespatial/cuda/cccl_primitives.py`
- `src/vibespatial/cuda/_runtime.py`
- CCCL helper call sites in indexing, queries, overlay, and constructive code

### Known Problems To Fix

- primitive wrappers default to `Stream.null.synchronize()`
- count-returning wrappers read scalar results on host immediately
- wrappers do not expose a stream-aware contract even though lower layers can
  already accept streams

### Checklist

- [ ] Review every CCCL helper for hardcoded null-stream synchronization.
- [ ] Add or normalize `synchronize=` behavior so callers can defer completion.
- [ ] Decide which primitives need stream parameters immediately and which can
  stay same-stream but caller-synchronized first.
- [ ] Replace legacy call sites that still expect helper-owned completion.
- [ ] Verify count-scatter, sort, search, and segmented reduce stages compose
  without extra host syncs.
- [ ] Keep correctness by adding tests around deferred completion paths.

### Exit Criteria

- CCCL wrappers no longer force completion by default in hot reusable paths
- same-stream pipelines can chain primitives without repeated barriers
- stream-aware extension path is clear and minimally invasive

## Milestone M3: Device-Native Decode And Compaction

### Goal

Replace Python-controlled nested decode logic with device-native staged scans
and count-scatter passes.

### Primary Surfaces

- `src/vibespatial/io/pylibcudf.py`
- legacy count-scatter total sites in `io/shp_gpu.py` and `io/fgb_gpu.py`
- any adjacent decode helpers still driven by host maxima

### Known Problems To Fix

- WKB polygon, multilinestring, and multipolygon decode walk nested structure
  with Python `for range(max_...)` loops
- repeated `cp.asnumpy(...max())` reads drive control flow
- some count-scatter totals still use `runtime.synchronize()` plus multiple
  `.get()` calls instead of the async helper

### Checklist

- [ ] Replace host maxima loops with segmented device passes where feasible.
- [ ] Batch byte-start discovery and nested offset generation on device.
- [ ] Use `count_scatter_total()` or `count_scatter_total_with_transfer()` for
  legacy total sites.
- [ ] Verify mixed, polygon, and multipolygon decode correctness against
  existing fixtures.
- [ ] Re-measure ingest surfaces that were previously paying hidden host loops.

### Exit Criteria

- decode control flow is no longer Python-driven for nested WKB structure
- legacy sync-plus-`.get()` count-scatter sites are removed from major IO paths
- decode throughput and transfer shape improve on realistic mixed geometry data

## Milestone M4: Predicate And Query Execution Shape

### Goal

Fix point-in-polygon and adjacent query paths so refine logic behaves like a
GPU pipeline, not a host-managed dispatch loop.

### Primary Surfaces

- `src/vibespatial/kernels/predicates/point_in_polygon.py`
- candidate assembly helpers in query code
- any work estimation or binning code that pulls candidate rows to host

### Known Problems To Fix

- dense and compacted helpers end with unconditional `runtime.synchronize()`
- binned mode copies candidate rows to host for work estimation
- same-stream launch chains synchronize before the caller actually needs host
  data

### Checklist

- [ ] Remove helper-level syncs where same-stream ordering already guarantees
  correctness.
- [ ] Move work estimation and bin selection onto device where possible.
- [ ] Keep candidate rows device-side through coarse filter and refine.
- [ ] Re-check dense, compacted, binned, and fused paths separately.
- [ ] Re-run profiler rails and inspect selected runtime and stage times.

### Exit Criteria

- PIP helpers synchronize only at real host boundaries
- work estimation no longer requires candidate rows on host
- query and predicate rails show a more GPU-shaped refine stage

## Milestone M5: Overlay And Constructive De-Hosting

### Goal

Remove the highest-value remaining host orchestration from grouped overlay,
microcells, contraction, and constructive reduction flows.

### Primary Surfaces

- `src/vibespatial/overlay/gpu.py`
- `src/vibespatial/overlay/microcells.py`
- `src/vibespatial/overlay/contract.py`
- `src/vibespatial/overlay/assemble.py`
- `src/vibespatial/constructive/union_all.py`
- adjacent constructive helpers still forcing same-stream syncs

### Known Problems To Fix

- grouped overlay materializes group boundaries to host and iterates in Python
- current stream pool is limited by null-stream NVRTC and wrapper behavior
- microcell labeling loops row-by-row over host materialized row ids
- contraction moves full band arrays to host and runs union-find in Python
- some constructive helpers still synchronize after same-stream scatter
- union-all still performs Python tree reduction over single-row objects

### Checklist

- [ ] Replace host grouping loops with device-side grouping or batched planning
  where correctness allows.
- [ ] Revisit stream pool usefulness after M2 lands.
- [ ] Move microcell labeling control flow off host.
- [ ] Replace host union-find contraction with a device-friendly contraction
  plan or a clearly bounded fallback path.
- [ ] Audit overlay assembly for remaining hardcoded launch geometry and avoidable
  host scalar control decisions.
- [ ] Reassess union-all reduction shape once overlay batching improves.
- [ ] Re-run the full pipeline benchmark and inspect the 1M sparkline.

### Exit Criteria

- grouped overlay no longer depends on host-materialized per-group ranges in
  its main GPU path
- microcells and contraction are no longer structurally host-managed
- overlay pipeline stages show materially less CPU orchestration in profiling

## Cross-Cutting Cleanup Sweep

After M1 through M5 land, do a sweep for smaller but repeated anti-patterns.

### Sweep Targets

- `src/vibespatial/io/shp_gpu.py`
- `src/vibespatial/io/fgb_gpu.py`
- `src/vibespatial/constructive/clip_rect.py`
- `src/vibespatial/constructive/linestring.py`
- `src/vibespatial/constructive/shortest_line.py`
- `src/vibespatial/overlay/assemble.py`
- any residual hardcoded `(256, 1, 1)` launch patterns not justified by data

### Sweep Checklist

- [ ] replace sync-plus-scalar-read totals with the runtime helper
- [ ] remove same-stream syncs that only guard later device work
- [ ] switch obvious hardcoded launch sizes to occupancy-aware launch config
- [ ] collapse repeated small D2H reads into one batched transfer where host
  reads remain necessary

## Measurement Gates

Every milestone must report:

- exact commands used
- machine and GPU model
- selected runtime
- before and after timing
- before and after transfer shape if the milestone touches residency

At minimum, re-run these after each milestone:

```bash
uv run python scripts/profile_kernels.py --kernel all --rows 10000 --repeat 1
uv run python scripts/health.py --gpu-coverage
uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline
```

If a milestone changes join, overlay, IO, predicate, or runtime surfaces, also
run the narrowest relevant pytest slice before the broad rails.

## Program-Level Exit Criteria

The full push is complete only when all of the following are true:

- GPU acceleration coverage is materially above the April 7, 2026 baseline of
  `3.95%`
- profiler rails select GPU for important benchmark surfaces at useful scales,
  not just tiny helper stages
- end-to-end pipeline sparkline shows reduced CPU-heavy orchestration relative
  to the starting baseline
- major device-resident pipelines no longer pay eager host metadata mirroring
- major reusable GPU helpers no longer force null-stream synchronization by
  default
- grouped overlay and microcells are no longer fundamentally host-managed in
  their mainline GPU execution shape

## What Counts As Failure

This push fails if it ends with:

- more GPU dispatches but no meaningful end-to-end improvement
- more stream code but the same null-stream serialization
- cleaner helper APIs but unchanged host-driven decode and grouping
- overlay still structurally controlled by Python loops
- new correctness regressions accepted as the price of performance

## Recommended Delivery Order

Use this order unless measurement proves otherwise:

1. M0 baseline and measurement hardening
2. M1 residency and metadata ownership
3. M2 CCCL synchronization contract
4. M3 device-native decode and compaction
5. M4 predicate and query execution shape
6. M5 overlay and constructive de-hosting
7. cross-cutting cleanup sweep

The ordering matters because:

- M1 removes hidden D2H taxes that pollute later measurements
- M2 makes later stream and composition work possible
- M3 and M4 fix shared primitives used by many higher-level paths
- M5 is the hardest and should start after the foundations stop fighting back

## Session Checklist

Use this at the start of each session in the push:

- [ ] Which milestone is active?
- [ ] What exact baseline numbers am I trying to beat?
- [ ] What structural blocker am I removing first?
- [ ] What profiler and benchmark commands will prove the change mattered?
- [ ] What correctness slices must stay green while I change the execution
  shape?

Use this at the end of each session:

- [ ] What changed in execution shape, not just code structure?
- [ ] What measurements improved?
- [ ] What measurements did not move?
- [ ] What blocker remains for the current milestone?
- [ ] What is the next smallest structural step?
