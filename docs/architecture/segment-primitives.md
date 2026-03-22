# Segment Intersection Primitives

<!-- DOC_HEADER:START
Scope: Robust segment extraction, candidate generation, and exact intersection-classification strategy for constructive kernels.
Read If: You are changing segment intersection math, degeneracy handling, or constructive-kernel foundations.
STOP IF: You already have the segment primitive implementation open and only need local implementation detail.
Source Of Truth: Phase-5 segment intersection primitive policy before overlay assembly.
Body Budget: 114/220 lines
Document: docs/architecture/segment-primitives.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Preamble |
| 7-15 | Request Signals |
| 16-22 | Open First |
| 23-30 | Verify |
| 31-36 | Risks |
| 37-41 | Intent |
| 42-52 | Options Considered |
| 53-74 | Decision |
| 75-95 | CCCL Mapping |
| 96-109 | Degeneracy Semantics |
| 110-114 | Consequences |
DOC_HEADER:END -->

`o17.5.1` establishes the first constructive-kernel primitive layer: direct
segment extraction from owned buffers, cheap candidate generation, and robust
intersection classification.

## Request Signals

- segment intersection
- overlay primitives
- degeneracy
- collinear overlap
- zero-length segment
- cccl

## Open First

- docs/architecture/segment-primitives.md
- docs/architecture/robustness.md
- src/vibespatial/spatial/segment_primitives.py
- tests/test_segment_primitives.py

## Verify

- `uv run pytest tests/test_segment_primitives.py tests/test_segment_filters.py`
- `uv run pytest tests/test_segment_primitives.py -q --run-gpu`
- `uv run python scripts/benchmark_segment_intersections.py --rows 10000`
- `uv run python scripts/check_architecture_lints.py --all`
- `uv run python scripts/check_docs.py --check`

## Risks

- Wrong segment classification poisons every later overlay kernel.
- Exact arithmetic on every row would destroy GPU throughput.
- Routing candidate generation through materialized Shapely geometry would lock Phase 5 to the host path we are trying to replace.

## Intent

Choose a segment-intersection primitive shape that preserves robustness while
staying compatible with later CCCL-backed GPU execution.

## Options Considered

1. Delegate segment intersection to Shapely/GEOS.
   Correct on CPU, but it hides the primitive boundaries and forces future GPU
   work to reverse-engineer the operation graph.
2. Run exact scalar arithmetic on every candidate pair.
   Robust, but throughput-hostile and a poor fit for SIMT hardware.
3. Stage the work: direct segment extraction, vectorized fast-path orientation,
   ambiguous-row compaction, and exact fallback only on compacted rows.
   This matches the repo's robustness policy and maps onto reusable primitives.

## Decision

Use option 3.

The landed primitive is a fully GPU-native 3-kernel pipeline:

- extracts segments from owned coordinate buffers via GPU count-scatter kernels
- generates candidate pairs via O(n log n) sort-sweep with P95 OOM protection
- classifies pairs on GPU with fast orientation filter and Shewchuk adaptive refinement
- no host round-trip for ambiguous rows -- all refinement stays on device

The kernel registers as `KernelClass.PREDICATE` (not CONSTRUCTIVE) because
on-GPU Shewchuk adaptive refinement handles precision internally, removing the
need for forced fp64 compute on consumer GPUs.

The implementation has three GPU kernel stages:

- segment extraction via NVRTC count-scatter kernels (per-family dispatch)
- candidate generation via CCCL radix sort + binary search sweep
- classification with warp-cooperative MBR skip and on-GPU Shewchuk refinement
- result buffers stay on device for later reconstruction work

## CCCL Mapping

The GPU path uses a three-tier primitive strategy (ADR-0033):

- **Segment extraction** (Tier 1 NVRTC): two-pass count-scatter kernel with
  per-family dispatch extracts segments directly from owned coordinate buffers.
- **Candidate generation** (CCCL Tier 3a + CuPy Tier 2): O(n log n)
  sort-sweep using CCCL `radix_sort` on right-segment x-midpoints, then
  CCCL `lower_bound`/`upper_bound` binary search per left segment.
  A **P95 half-width strategy** prevents OOM from outlier segments:
  the binary-search window uses the 95th-percentile right half-width
  (tight windows for 95% of segments), with a separate brute-force MBR
  pass for the <=5% outlier right segments whose half-width exceeds P95.
  Main and outlier candidates merge via `cp.concatenate` on device.
  When right.count < 20 or P95 == max, the outlier pass is skipped.
- **Fast classification** (Tier 1 NVRTC): warp-cooperative MBR skip,
  fast orientation filter, Shewchuk adaptive refinement on GPU.
  No ambiguous rows are sent to the host; all refinement stays on device.
- **Compaction** (CuPy Tier 2): `compact_indices` via CuPy `flatnonzero`
  for MBR overlap filtering and ambiguous-row isolation.

## Degeneracy Semantics

- strict interior crossing: `proper`
- shared endpoints or point-on-segment hits: `touch`
- collinear shared span: `overlap`
- otherwise: `disjoint`

The exact fallback path must cover:

- collinear overlap
- shared vertices
- zero-length segments
- polygon ring-edge corner cases

## Consequences

- Phase 5 now has a reusable segment primitive instead of ad hoc scalar logic
- the owned-buffer path stays visible from extraction through classification
- later overlay work can consume candidate pairs and classifications directly
