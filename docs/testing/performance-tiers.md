# Performance Tiers

<!-- DOC_HEADER:START
Scope: Performance tier gates, reference datasets, and benchmark acceptance policy.
Read If: You are defining kernel success criteria, benchmark rails, or performance gates.
STOP IF: Your task is limited to a single benchmark implementation detail already routed elsewhere.
Source Of Truth: Phase-0 performance gate policy for GPU kernel work.
Body Budget: 158/240 lines
Document: docs/testing/performance-tiers.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-10 | Intent |
| 11-22 | Request Signals |
| 23-29 | Open First |
| 30-34 | Verify |
| 35-40 | Risks |
| 41-51 | Denominator |
| 52-70 | Reference Scale |
| 71-83 | Tier Table |
| 84-108 | Transient Latency Gates |
| 109-122 | Tier Rules |
| 123-134 | Mapping To Roadmap |
| 135-150 | Acceptance Policy |
| 151-158 | Verification |
DOC_HEADER:END -->

Define the minimum performance gates for GPU-first kernel work before the
benchmark harness and synthetic datasets are fully implemented.

## Intent

Set explicit speedup floors, aspirational targets, and reference benchmark
rules so each kernel can declare success against the same denominator.

## Request Signals

- benchmark
- performance
- perf gates
- speedup
- throughput
- latency
- transient native work
- scalar fence budget
- kernel tier

## Open First

- docs/testing/performance-tiers.md
- docs/implementation-order.md
- docs/testing/upstream-inventory.md
- src/vibespatial/runtime/_runtime.py

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "define performance tier gates for GPU kernels"`

## Risks

- Speedup targets can become meaningless if the reference denominator shifts.
- Small synthetic cases can overstate GPU wins by hiding transfer and setup costs.
- Gates that are too strict too early can block correct-but-incomplete kernel landings.

## Denominator

- Baseline comparisons are against single-threaded GeoPandas or Shapely on the
  same machine unless a benchmark explicitly documents a different host-side
  denominator.
- The gate measures steady-state kernel-path speedup at the reference scale,
  not cold-start import time or one-time environment setup.
- CPU fallback paths do not count as passing a GPU benchmark gate.
- A kernel must state both its expected tier and the benchmark command or
  harness entrypoint that will eventually enforce the gate.

## Reference Scale

- Required scales: `10K`, `100K`, and `1M` geometries.
- The default gate scale is `100K` mixed polygons unless the operation is
  clearly point- or IO-dominated.
- `10K` exists to observe crossover behavior and dispatch thresholds.
- `1M` exists to catch memory-pressure and batching regressions once kernels
  move beyond smoke status.

Use these reference dataset families:

- uniform grids for regular work and predictable memory access
- polygon-heavy parcel-like subdivisions for overlay, clip, and dissolve
- point clouds for joins, distance, and coarse-filter workloads
- admin-boundary style polygon sets for real-world irregularity

`o17.1.8` should generate license-free versions of these families instead of
checking in sourced benchmark data.

## Tier Table

| Tier | Parallelism shape | Gate | Aspirational | Example operations |
|---|---|---:|---:|---|
| 5 | embarrassingly parallel | 100x | 1000x | bounds, centroid, area, length, affine transforms, SFC keys, coordinate access |
| 4 | per-geometry parallel | 20x | 100x | buffer, simplify, convex hull, point-in-polygon, unary predicates |
| 3 | filtered parallel | 10x | 50x | sjoin, sindex query, nearest, `dwithin`, binary predicates after coarse filtering |
| 2 | structured parallel | 5x | 20x | clip, intersection, union, difference, dissolve |
| 1 | external-bound | 1x | 3x to 5x | file IO, GDAL-mediated reads, format parsing |

CRS transforms are out of scope for these tiers because that work is expected
to route through cuProj policy later in `o17.6.3`.

## Transient Latency Gates

ADR-0045 adds latency budgets for small native work items inside larger
workflows. These gates complement throughput tiers; they do not replace
speedup measurements for large kernels.

Every transient-shape canary should report runtime D2H count/bytes,
materialization count, scalar allocation fences when distinguishable, launch or
stage count when observable, input/output residency, and the public export
boundary if one exists.

Default gates for admitted native transient stages:

| Metric | Gate |
|---|---:|
| Hidden materialization | 0 |
| Device output residency | required |
| Runtime D2H for rowset/filter/export consumers | 0 |
| Runtime D2H for scalar allocation totals | ratchet, batched, documented |
| Scalar-only D2H payload | <=64 bytes per native stage unless justified |

Small-shape canaries should run at `10K` and at representative transient
shapes such as many groups of size 2-8 or 2-16. Workflow shootouts may observe
impact, but the acceptance gate belongs to the reusable transient shape.

## Tier Rules

- Tier 5 kernels should usually be memory-bandwidth bound. If they do not beat
  the gate, treat the implementation as suspect until profiling proves
  otherwise.
- Tier 4 kernels may have variable geometry complexity, but they should still
  scale mostly with per-geometry independence.
- Tier 3 kernels must include the coarse-filter stage in benchmark accounting.
  Reporting only the refine pass is not acceptable.
- Tier 2 kernels may land below aspirational targets early, but the minimum
  gate still applies once correctness and batching stabilize.
- Tier 1 work is allowed to land with parity-only performance if the bottleneck
  is dominated by host parsing, legacy libraries, or disk.

## Mapping To Roadmap

- Phase 2 geometry-buffer kernels should mostly declare Tier 5 or Tier 4.
- Phase 3 indexing work should declare Tier 5 for pair generation and Tier 3
  for query paths.
- Phase 4 predicates and joins should mostly declare Tier 3, with some Tier 4
  unary predicate coverage.
- Phase 5 overlay and constructive geometry should declare Tier 2 unless a
  narrower fast path clearly fits Tier 4.
- Phase 6b IO work should default to Tier 1 unless a GPU-native scanner moves
  parsing and filtering onto the device.

## Acceptance Policy

- Every kernel-oriented task must name its tier in the description or notes.
- Every benchmark result should report:
  - dataset family
  - scale
  - requested runtime mode
  - selected runtime mode
  - speedup versus the documented host baseline
- Gates are enforced first in docs and manual benchmark runs, then in
  `o17.1.3` and `o17.1.7` once benchmark rails and CI are in place.
- Falling below the gate is allowed only with an explicit blocker or follow-up
  follow-up explaining why the kernel still needs to land.
- `o17.2.7` should use the `10K` and `100K` scales to reason about dispatch
  crossover thresholds.

## Verification

Use this doc as the policy source until benchmark rails exist:

```bash
uv run python scripts/check_docs.py --check
uv run python scripts/intake.py "define performance tier gates for GPU kernels"
```
