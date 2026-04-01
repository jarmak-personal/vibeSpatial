# Point Predicate Pipeline

<!-- DOC_HEADER:START
Scope: Point-versus-bounds and point-in-polygon pipeline shape, stage boundaries, and fallback contract.
Read If: You are changing point predicates, candidate refinement, or the first exact spatial query path.
STOP IF: Your task already has the staged point-predicate contract open and only needs local implementation detail.
Source Of Truth: Phase-4 point predicate architecture policy before sindex and join assembly.
Body Budget: 105/220 lines
Document: docs/architecture/point-predicates.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-11 | Request Signals |
| 12-19 | Open First |
| 20-27 | Verify |
| 28-36 | Risks |
| 37-41 | Intent |
| 42-54 | Options Considered |
| 55-75 | Decision |
| 76-89 | CCCL Mapping |
| 90-96 | Semantics |
| 97-105 | Consequences |
DOC_HEADER:END -->

## Request Signals

- point in polygon
- point versus bounds
- point predicate
- bounds filter
- cccl
- fallback

## Open First

- docs/architecture/point-predicates.md
- docs/architecture/runtime.md
- docs/architecture/robustness.md
- src/vibespatial/kernels/predicates/point_within_bounds.py
- src/vibespatial/kernels/predicates/point_in_polygon.py

## Verify

- `uv run pytest tests/test_point_within_bounds.py tests/test_point_in_polygon.py`
- `uv run vsbench run point-predicates --scale 10k`
- `uv run vsbench run gpu-pip --scale 1m`
- `uv run python scripts/check_architecture_lints.py --all`
- `uv run python scripts/check_docs.py --check`

## Risks

- Monolithic predicate kernels would bypass the coarse-filter and adaptive-runtime work already landed.
- Silent CPU fallback would hide the absence of a real GPU predicate variant.
- Predicate boundary semantics can drift if bounds checks and exact refine are tested separately.

`o17.4.1` establishes the first owned predicate pipeline for point queries against
polygonal inputs.

## Intent

Choose a point-predicate implementation shape that is compatible with the
repo's existing precision, robustness, indexing, and fusion policy.

## Options Considered

1. Monolithic ray-casting kernels.
   Fast to prototype, but it bakes traversal, compaction, and reduction into one
   bespoke kernel and leaves little room for reuse.
2. Pre-triangulate polygons and answer point location from triangle lookup.
   Attractive for repeated queries, but expensive to build, awkward for mutable
   inputs, and too much machinery for the first predicate landing.
3. Staged point-predicate pipeline.
   Use a cheap bounds pass, compact candidates, then refine only the surviving
   rows. This matches the existing Phase-3 coarse-filter work and keeps the GPU
   path expressible in reusable primitives.

## Decision

Use option 3.

The owned predicate surface is:

- `point_within_bounds`: coarse bounds predicate for aligned point and
  polygon-or-bounds inputs
- `point_in_polygon`: exact point-in-polygon result for aligned point and
  polygon or multipolygon inputs

The current implementation keeps correctness first:

- bounds checks run directly on owned point coordinates and row-aligned bounds
- point-in-polygon uses `point_within_bounds` as the coarse pass
- surviving candidates are compacted and refined on GPU with cuda-python kernels
- the GPU implementation keeps both dense-row and compacted-candidate kernels,
  but current measurements keep `auto` on the compacted path
- `auto` mode records explicit CPU fallback only when the GPU runtime is unavailable
- explicit `gpu` mode now executes the owned GPU variant when CUDA is available

## CCCL Mapping

The intended GPU path should stay staged and primitive-oriented:

- row-aligned bounds predicate: transform-style compare over point coordinates
- candidate compaction: CCCL `DeviceSelect`
- segment/ring crossing accumulation: CCCL scan and reduction primitives
- parity aggregation by row: CCCL `reduce_by_key`
- selective ambiguous-row refine: compacted fallback pass only

Phase 9b lands the first live cuda-python predicate implementation while
preserving the same staged contract that later CCCL compaction and scan
primitives can replace.

## Semantics

- null on either side yields `None`
- empty point or empty polygon yields `False`
- boundary hits count as `True` for `point_in_polygon`
- bounds predicates are inclusive on all four edges

## Consequences

- correctness and visible fallback are in place now
- the owned GPU kernel now replaces the Shapely refine stage without changing
  the public contract
- the staged design reuses Phase-3 coarse filters instead of bypassing them
- the planned convergence point with the spatial query/index stack lets
  point-vs-polygon candidates reuse this dedicated predicate path where
  the query surface and semantics line up
