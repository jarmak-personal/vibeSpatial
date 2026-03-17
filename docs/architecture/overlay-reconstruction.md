# Overlay Reconstruction

<!-- DOC_HEADER:START
Scope: Overlay reconstruction staging, face-labeling plan, and CCCL-oriented output assembly policy.
Read If: You are changing union, difference, symmetric difference, or overlay output reconstruction.
STOP IF: You already have the reconstruction planner open and only need local implementation detail.
Source Of Truth: Phase-5 reconstruction plan from segment primitives to public overlay outputs.
Body Budget: 92/220 lines
Document: docs/architecture/overlay-reconstruction.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-13 | Request Signals |
| 14-20 | Open First |
| 21-25 | Verify |
| 26-31 | Risks |
| 32-36 | Intent |
| 37-46 | Options Considered |
| 47-60 | Decision |
| 61-74 | CCCL Mapping |
| 75-92 | Consequences |
DOC_HEADER:END -->

`o17.5.3` fixes the constructive assembly shape before full overlay kernels land.

## Request Signals

- union reconstruction
- difference reconstruction
- symmetric difference
- overlay assembly
- face labeling
- cccl

## Open First

- docs/architecture/overlay-reconstruction.md
- docs/architecture/segment-primitives.md
- src/vibespatial/overlay_reconstruction.py
- tests/test_overlay_reconstruction.py

## Verify

- `uv run pytest tests/test_overlay_reconstruction.py tests/test_segment_primitives.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Monolithic overlay kernels would hide the exact stage boundaries needed for CCCL.
- If stable ordering is not explicit, later dissolve and overlay assembly will drift from pandas-facing semantics.
- Union, difference, and symmetric difference should not each invent separate topology assembly logic.

## Intent

Define one staged reconstruction plan from classified segments to geometry output
buffers so later constructive kernels share the same graph.

## Options Considered

1. One bespoke kernel per overlay operation.
   Fast to sketch, but it duplicates topology assembly and destroys reuse.
2. Keep using host-side Shapely for all reconstruction.
   Correct on CPU, but it provides no owned assembly seam for GPU work.
3. Shared staged reconstruction.
   Classify segments once, emit nodes once, build directed edges once, label
   faces once, then apply operation-specific selection at the end.

## Decision

Use option 3.

The shared plan is:

- classify candidate segment pairs
- emit nodes and split segments at intersections
- stable-sort directed half-edges
- walk rings and open chains
- label faces and chains by source coverage
- select union/difference/symmetric-difference outputs
- emit geometry buffers in deterministic order

## CCCL Mapping

The intended GPU execution is:

- compaction of ambiguous segment rows
- device-side split-event emission for endpoints, touches, proper crosses, and overlaps
- stable sort for half-edge grouping
- prefix sums for segment splitting and output sizing
- reduce-by-key for face labeling and chain aggregation
- scatter/gather for output restoration

That keeps the expensive topology decisions localized and reusable across
overlay operations.

## Consequences

- Phase 5 now has one reconstruction graph instead of operation-specific glue
- `o17.9.6.2` lands the device split-event and directed-edge primitive that
  feeds the later half-edge graph work
- `o17.9.6.3` adds canonical node ids, deterministic `next_edge` traversal,
  and bounded-face labeling on top of that directed-edge table
- `o17.9.6.4` is allowed to route simple axis-aligned rectangle batches into a
  dedicated fast path when the generic shared graph would have the wrong
  performance shape for pairwise box intersections
- `o17.5.5` can build dissolve on top of the same union reconstruction
- `o17.9.6.6` reuses the same bounded-face labels for `union`,
  `difference`, `symmetric_difference`, and geometry-only `identity` selectors
  instead of forking new topology assembly code
- current GPU overlay output support remains polygon-only at the input seam;
  mixed-family constructive overlay stays explicitly unsupported until a later
  a later change widens the kernel contract
- later GPU overlay work has an explicit CCCL-friendly assembly seam
