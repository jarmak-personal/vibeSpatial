# Segment Intersection Primitives

<!-- DOC_HEADER:START
Scope: Robust segment extraction, candidate generation, and exact intersection-classification strategy for constructive kernels.
Read If: You are changing segment intersection math, degeneracy handling, or constructive-kernel foundations.
STOP IF: You already have the segment primitive implementation open and only need local implementation detail.
Source Of Truth: Phase-5 segment intersection primitive policy before overlay assembly.
Body Budget: 103/220 lines
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
| 53-70 | Decision |
| 71-84 | CCCL Mapping |
| 85-98 | Degeneracy Semantics |
| 99-103 | Consequences |
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
- src/vibespatial/segment_primitives.py
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

The landed primitive now does the following:

- extracts segments directly from owned coordinate buffers
- generates candidate segment pairs from segment MBR overlap
- classifies clear non-degenerate pairs in vectorized fp64 arithmetic
- compacts ambiguous rows and reclassifies them with exact-style rational math

The implementation now has an explicit GPU classifier path for candidate pairs:

- fast orientation and proper-cross detection run in a CUDA kernel
- ambiguous rows are compacted on device
- the exact-style lane still handles only the compacted minority rows
- result buffers can stay mirrored on device for later reconstruction work

## CCCL Mapping

The intended GPU path should remain primitive-oriented:

- direct segment extraction: gather/scatter from owned coordinate buffers
- candidate generation: tiled MBR overlap plus CCCL `DeviceSelect`
- fast classification: transform-style orientation/sign evaluation
- ambiguity isolation: CCCL `DeviceSelect`
- exact fallback: compacted minority pass only
- output restoration: scatter back to original candidate order

This does not land raw CUDA kernels. It lands the robust primitive contract
that a future CCCL-backed implementation should preserve.

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
