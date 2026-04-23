# Exact GPU Overlay Debugging Playbook

<!-- DOC_HEADER:START
Scope: Repeatable workflow for debugging exact GPU overlay and dissolve correctness bugs without losing the real performance target.
Read If: You are investigating a union, intersection, dissolve, face-selection, split-event, or half-edge graph bug in the GPU overlay stack.
STOP IF: You already have the exact failing stage pinned and only need local implementation detail in that module.
Source Of Truth: Repo playbook for exact GPU overlay debugging and regression lock-in.
Body Budget: 179/220 lines
Document: docs/testing/overlay-debugging-playbook.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-15 | Intent |
| 16-34 | Request Signals |
| 35-46 | Open First |
| 47-54 | Verify |
| 55-62 | Risks |
| 63-80 | Core Rules |
| 81-97 | Debugging Workflow |
| 98-130 | Stage-Isolation Ladder |
| 131-149 | Structural Smells |
| 150-165 | Regression Lock-In |
| 166-179 | Session Lessons |
DOC_HEADER:END -->

## Intent

Capture the repeatable workflow for debugging exact GPU overlay bugs in
vibeSpatial. This is not a generic geometry-debug guide. It is the repo-local
playbook for situations where:

- the public overlay or dissolve result is wrong
- the GPU path looks active but the output is still incorrect
- the bug may live anywhere from segment classification through face selection

Use it to keep correctness work structured and to avoid wasting time in the
wrong layer.

## Request Signals

- overlay debugging
- overlay debugging playbook
- dissolve debugging
- exact gpu overlay
- union miss
- intersection miss
- valid but wrong
- symmetric difference oracle
- disconnected overlap
- bad face selection
- split-event bug
- atomic edge dedup
- collinear overlap
- half-edge graph
- buffered-line dissolve
- exact overlay regression

## Open First

- docs/testing/overlay-debugging-playbook.md
- docs/architecture/overlay-reconstruction.md
- docs/architecture/segment-primitives.md
- docs/architecture/dissolve.md
- src/vibespatial/overlay/split.py
- src/vibespatial/overlay/graph.py
- src/vibespatial/overlay/faces.py
- src/vibespatial/constructive/binary_constructive.py
- tests/test_overlay_assembly_debug.py

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run pytest -q tests/test_overlay_assembly_debug.py`
- `uv run pytest -q tests/test_gpu_dissolve.py -k buffered_line_dissolve_gpu_result_is_valid`
- `env VSBENCH_SCALE=10000 uv run python benchmarks/shootout/vegetation_corridor.py`
- `env VSBENCH_SCALE=10000 uv run python benchmarks/shootout/nearby_buildings.py`

## Risks

- `valid` output can still be exactly wrong.
- Public dissolve bugs often originate below the dissolve wrapper.
- Tiny synthetic fixtures can miss the real structural bug.
- A correctness fix can remove one topology bug while leaving the broader perf
  debt untouched.

## Core Rules

- Always measure exactness, not only validity.
  Use symmetric-difference area against a Shapely oracle.
- Always pin one real failing pair from the real workload.
  Do not stay in abstract “some rows are bad” territory.
- Always isolate stages before editing kernels.
  Prove whether the miss is in:
  - public wrapper
  - union repair orchestration
  - split-event emission
  - atomic-edge dedup
  - half-edge graph build
  - face selection
  - polygon assembly
- Never treat a fallback or wrapper rewrite as the fix until the pinned exact
  oracle is clean.

## Debugging Workflow

1. Reproduce the bug on the real workflow first.
   Record the fingerprint, elapsed time, and the exact public API call shape.
2. Pin one failing pair or one failing row.
   Keep that reproducer alive for the entire session.
3. Compare against a Shapely oracle.
   Record:
   - result validity
   - geometry type
   - symmetric-difference area
4. Re-run the same pinned case through lower layers.
   Reuse the same geometry pair while changing only one stage at a time.
5. Do not widen scope until the pinned case is explained.
6. After the exact miss is fixed, return to workflow perf and check whether the
   broader product bottleneck is actually the same bug.

## Stage-Isolation Ladder

Walk this ladder in order. Stop as soon as the first wrong stage is found.

1. Public operation:
   - `binary_constructive_owned(...)`
   - public `GeoDataFrame.dissolve(...)`
   - public `overlay(...)`
2. Shared execution plan:
   - `_build_overlay_execution_plan(...)`
   - `_materialize_overlay_execution_plan(...)`
3. Direct per-op overlay materialization:
   - `_dispatch_overlay_gpu(...)`
   - `_dispatch_polygon_intersection_overlay_rowwise_gpu(...)`
4. Split topology:
   - `build_gpu_split_events(...)`
   - inspect overlap rows, event payload, and per-segment `t/x/y`
5. Atomic edges:
   - `build_gpu_atomic_edges(...)`
   - check whether duplicate geometric segments survive
6. Half-edge graph:
   - `build_gpu_half_edge_graph(...)`
   - inspect coincident or repeated directed edges
7. Face coverage / selection:
   - `build_gpu_overlay_faces(...)`
   - `_select_overlay_face_indices_gpu(...)`
8. Output assembly:
   - GPU face assembly
   - host face assembly from the same selected faces

If both GPU and host assembly are wrong from the same selected faces, the bug
is upstream of assembly.

## Structural Smells

These are high-probability causes for exact overlay misses:

- orientation-sensitive dedup of coincident overlap spans
- duplicate same-direction half-edges after reverse-pair regeneration
- shared-span provenance collapsed before both sides carry the same split set
- label-point sampling on a non-atomic face
- valid-but-non-exact `MultiPolygon` outputs that hide an extra selected face
- workflow-level fixes that only move the bug deeper into the reducer

When overlap is involved, inspect both:

- segment-classifier overlap rows
- post-split atomic edges

Do not assume a bad selected face means face labeling is wrong. It may mean the
face was never atomic because topology was corrupted earlier.

## Regression Lock-In

Add two regressions whenever possible:

- a public or user-visible exactness regression
- a low-level structural regression

Examples:

- exact pinned `intersection` or `union` equals Shapely
- no duplicate oriented overlap segments remain after atomic-edge dedup
- selected faces for the pinned case are all valid polygons

This prevents future work from “fixing” the public output while reintroducing
the same low-level graph bug.

## Session Lessons

The April 18, 2026 disconnected-overlap session established four durable rules:

- The first useful oracle was a real buffered-line reduction pair, not a toy
  synthetic polygon fixture.
- `valid` was a weak signal; the decisive metric was symmetric-difference area.
- The bug was not in dissolve wrapping or face assembly. It was in
  orientation-sensitive atomic-edge dedup inside `src/vibespatial/overlay/split.py`.
- The right regression shape was both:
  - exact pinned overlay equality
  - structural “no duplicate overlap segment survives dedup”

Treat those as defaults for future overlay correctness sessions.
