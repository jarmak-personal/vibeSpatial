# Overlay Reconstruction

<!-- DOC_HEADER:START
Scope: Overlay reconstruction staging, face-labeling plan, and CCCL-oriented output assembly policy.
Read If: You are changing union, difference, symmetric difference, or overlay output reconstruction.
STOP IF: You already have the reconstruction planner open and only need local implementation detail.
Source Of Truth: Phase-5 reconstruction plan from segment primitives to public overlay outputs.
Body Budget: 122/220 lines
Document: docs/architecture/overlay-reconstruction.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-7 | Preamble |
| 8-16 | Request Signals |
| 17-23 | Open First |
| 24-28 | Verify |
| 29-34 | Risks |
| 35-39 | Intent |
| 40-49 | Options Considered |
| 50-63 | Decision |
| 64-77 | CCCL Mapping |
| 78-122 | Consequences |
DOC_HEADER:END -->

`o17.5.3` fixes the constructive assembly shape before full overlay kernels land.

For debugging workflow, pinned-pair isolation, and regression strategy, see
`docs/testing/overlay-debugging-playbook.md`.

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
- src/vibespatial/overlay/reconstruction.py
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

- public `overlay()` now chooses an execution family before heavy work starts
  (`clip_rewrite`, `broadcast_right_intersection`,
  `broadcast_right_difference`, `coverage_union`, `grouped_union`, or
  `generic_reconstruction`) and records that family in dispatch telemetry
- every constructive family still lowers through one canonical
  `NativeTabularResult` boundary, so planner selection changes execution
  shape without reintroducing host-side composition trees
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
- **Memory-Safe Difference Batching:** overlay difference splits large
  workloads into VRAM-safe batches to prevent OOM at scale. The strategy:
  1. Estimate per-pair byte cost from right-side coordinate density.
  2. Query free VRAM via `cupy.cuda.Device().mem_info`.
  3. Budget = `free_bytes * _VRAM_BUDGET_FRACTION` (0.3, conservative
     because segmented_union and binary_constructive each allocate
     comparable working memory).
  4. Compute groups per batch from budget / (avg_pairs * bytes_per_pair),
     clamped to `[_MIN_GROUPS_PER_BATCH=64, _MAX_GROUPS_PER_BATCH=10K]`.
  5. Below `_PAIR_THRESHOLD` (200K total pairs), skip batching entirely
     and process in a single dispatch (lower overhead).
  6. Each batch gathers its left/right slices, runs the GPU overlay
     difference kernel, and appends results. Final assembly concatenates
     batch outputs with consistent index tracking.
- many-vs-one (N-vs-1) overlay uses a three-tier strategy that bypasses the
  full reconstruction graph for the common broadcast_right workload shape:
  (1) containment bypass identifies polygons fully inside the clip polygon
  and returns them unchanged, (2) batched Sutherland-Hodgman clip handles
  boundary-crossing simple polygons on GPU, (3) only complex remainder
  polygons fall through to the full per-group overlay pipeline
