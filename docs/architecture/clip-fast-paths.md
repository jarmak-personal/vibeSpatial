# Rectangle Clip Fast Paths

<!-- DOC_HEADER:START
Scope: Rectangle clip fast-path strategy, owned constructive dataflow, and GeoPandas adapter policy.
Read If: You are changing clip_by_rect, rectangle clip performance, or early constructive fast paths.
STOP IF: You already have the rectangle clip engine open and only need local implementation detail.
Source Of Truth: Phase-5 rectangle clip fast-path policy before broader overlay assembly.
Body Budget: 107/220 lines
Document: docs/architecture/clip-fast-paths.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-14 | Request Signals |
| 15-21 | Open First |
| 22-30 | Verify |
| 31-36 | Risks |
| 37-41 | Intent |
| 42-53 | Options Considered |
| 54-70 | Decision |
| 71-87 | GeoPandas Adapter Policy |
| 88-100 | CCCL Mapping |
| 101-107 | Consequences |
DOC_HEADER:END -->

`o17.5.2` lands the first owned constructive fast path through axis-aligned
rectangle clipping.

## Request Signals

- clip_by_rect
- rectangle clip
- constructive fast path
- clip performance
- overlay first fast path
- cccl

## Open First

- docs/architecture/clip-fast-paths.md
- docs/architecture/segment-primitives.md
- src/vibespatial/constructive/clip_rect.py
- tests/test_clip_rect.py

## Verify

- `uv run pytest tests/test_clip_rect.py tests/test_degeneracy_corpus.py`
- `uv run python scripts/benchmark_clip_rect.py --kind line --rows 5000`
- `uv run python scripts/benchmark_clip_rect.py --kind polygon --rows 5000`
- `uv run pytest tests/upstream/geopandas/tests/test_geom_methods.py -k clip_by_rect`
- `uv run pytest tests/upstream/geopandas/tools/tests/test_clip.py -k "test_clip_poly or test_clip_line_keep_slivers or test_clip_multipoly_keep_slivers"`
- `uv run python scripts/check_docs.py --check`

## Risks

- General polygon intersection is too broad for the first constructive fast path.
- Forcing a slower owned host implementation onto GeoPandas would ship a performance regression.
- Hole, multipolygon, and invalid-input behavior can drift if the fast path is not checked against the degeneracy corpus.

## Intent

Choose the first constructive surface that is genuinely GPU-shaped and useful to
GeoPandas, without overcommitting to a full overlay implementation.

## Options Considered

1. Full polygon or line intersection against arbitrary geometries.
   Too broad for the first landing and too much assembly before we have a GPU
   variant.
2. Keep using Shapely only and postpone constructive kernels entirely.
   Safe on CPU, but it leaves no owned execution seam for Phase 5.
3. Rectangle clip first.
   Bounds filtering, candidate compaction, and per-family clipping all map
   cleanly onto reusable primitives and the GeoPandas `clip` / `clip_by_rect`
   surfaces already expose it.

## Decision

Use option 3.

The owned rectangle-clip engine now handles:

- `Point` and `MultiPoint`
- `LineString` and `MultiLineString`
- `Polygon` and `MultiPolygon`

It uses:

- owned buffer conversion
- row bounds filtering
- direct line clipping and ring clipping for candidate rows
- row-level fallback for unsupported or invalid geometry cases

## GeoPandas Adapter Policy

The repo now has an explicit adapter seam at `GeometryArray.clip_by_rect`, but
the GeoPandas host path remains on direct Shapely today.

Current state:

- the owned CPU path is correct and benchmarked
- the owned GPU point-only path is now faster than Shapely on the benchmark
  harness and can re-enter from device-backed point arrays without materializing
  the full source batch
- non-point families still do not have a public GPU clip path

So the adapter still records an explicit fallback event and leaves the host path
on Shapely until broader constructive GPU coverage exists, not because the
point-only path is still too slow.

## CCCL Mapping

The intended GPU path stays staged:

- bounds filter over row envelopes
- candidate compaction with `DeviceSelect`
- per-family clip kernels over compacted rows
- output restoration by row scatter

Rectangle clip is the right first constructive fast path because it preserves
that staged structure instead of hiding everything inside a monolithic overlay
kernel.

## Consequences

- Phase 5 now has a real owned constructive engine to optimize further
- the owned point-only GPU path can keep clipped coordinate payloads on device
  across constructive chains
- GeoPandas keeps current host performance while the adapter seam stays visible
- later overlay work can reuse the same candidate/filter/restore structure
