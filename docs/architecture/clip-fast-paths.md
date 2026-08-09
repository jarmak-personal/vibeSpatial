# Rectangle Clip Fast Paths

<!-- DOC_HEADER:START
Scope: Rectangle clip fast-path strategy, owned constructive dataflow, and GeoPandas adapter policy.
Read If: You are changing clip_by_rect, rectangle clip performance, or early constructive fast paths.
STOP IF: You already have the rectangle clip engine open and only need local implementation detail.
Source Of Truth: Phase-5 rectangle clip fast-path policy before broader overlay assembly.
Body Budget: 130/220 lines
Document: docs/architecture/clip-fast-paths.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-14 | Request Signals |
| 15-21 | Open First |
| 22-30 | Verify |
| 31-37 | Risks |
| 38-42 | Intent |
| 43-52 | Options Considered |
| 53-73 | Decision |
| 74-109 | GeoPandas Adapter Policy |
| 110-123 | CCCL Mapping |
| 124-130 | Consequences |
DOC_HEADER:END -->

Rectangle clipping is an owned constructive surface with native rowset and
topology carriers for device-resident inputs.

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
- `uv run vsbench run clip-rect --arg kind=line`
- `uv run vsbench run clip-rect --arg kind=polygon`
- `uv run pytest tests/upstream/geopandas/tests/test_geom_methods.py -k clip_by_rect`
- `uv run pytest tests/upstream/geopandas/tools/tests/test_clip.py -k "test_clip_poly or test_clip_line_keep_slivers or test_clip_multipoly_keep_slivers"`
- `uv run python scripts/check_docs.py --check`

## Risks

- Polygon clipping must preserve holes, disconnected parts, and multipart output;
  ring-independent clipping is not a valid general topology model.
- Forcing a slower owned host implementation onto GeoPandas would ship a performance regression.
- Hole, multipolygon, and invalid-input behavior can drift if the fast path is not checked against the degeneracy corpus.

## Intent

Keep rectangle clipping GPU-shaped while sharing exact polygon topology with the
binary constructive planner instead of maintaining a second polygon assembler.

## Options Considered

1. Clip every polygon ring independently and rebuild geometry metadata after the
   kernel. This cannot represent a concave polygon clipped into disconnected
   parts and requires row-shaped host metadata.
2. Add a second clip-specific exact topology implementation. This duplicates
   boundary repair, multipart regrouping, and output assembly.
3. Select polygon rows natively, build rectangle rows on device, and use the
   shared rectangle/SH/exact binary constructive planner.

## Decision

Use option 3. Dense proven rows can still use the bounded rectangle kernel, but
the shared planner owns boundary-split repair and exact completion.

The owned rectangle-clip engine now handles:

- `Point` and `MultiPoint`
- `LineString` and `MultiLineString`
- `Polygon` and `MultiPolygon`

It uses:

- owned buffer conversion
- row bounds filtering
- source-row-capacity point selection and direct line clipping for their
  admitted family shapes
- device rowset selection plus exact constructive topology for polygonal rows
- identity device row maps plus validity-backed dynamic output selection
- observable compatibility boundaries for unsupported or invalid input

## GeoPandas Adapter Policy

The repo now has an explicit adapter seam at `GeometryArray.clip_by_rect`, and
the public `clip(...)` path opportunistically stays inside the owned/native
boundary when the resulting family mix is representable there.

Current state:

- the owned CPU path is correct and benchmarked
- the owned GPU Point path filters coordinates in source-row capacity and can
  re-enter from device-backed arrays without candidate compaction or host
  materialization
- polygon families use one device rowset plus device-built rectangle rows; the
  binary constructive planner partitions rectangle, SH-eligible, and exact
  topology work and returns Polygon/MultiPolygon output without host metadata
- the former host polygon-ring extractor, per-ring clip assembler, surviving-ring
  count export, output-offset export, and duplicate scalar rectangle path are
  deleted
- line families use fused fp64 Liang-Barsky count/scatter directly over owned
  LineString/MultiLineString buffers; source rows and part boundaries remain
  device-shaped and bounded output capacity avoids cardinality reads
- point, polygon, and line GPU paths return source-row-capacity
  ``Residency.DEVICE`` OwnedGeometryArrays, and mixed public rectangle clip
  invokes the combined carrier once instead of partitioning families
- default public `clip(..., keep_geom_type=False)` now builds a row-preserving
  native result first; valid/nonempty, positive-area, and keep-type cleanup
  remain one capacity-backed device selection, and only unsupported public
  collection typing exits the native family model
- host normalization preserves owned backing for representable host-side results,
  so later `area` / `length` probes stay on the owned measurement path instead
  of silently dropping to Shapely
- unsupported public outputs such as `GeometryCollection` slivers stay on the
  explicit compatibility path instead of being forced into the owned model
- public ordering now matches GeoPandas again: `sort=False` preserves encounter
  order, and `sort=True` sorts by index

## CCCL Mapping

The intended GPU path stays staged:

- bounds filter over row envelopes
- source-row masks and capacity-preserving family partitions
- per-family clip/topology kernels over source-row capacity
- output selection through one device row-indirection map

Polygon rectangle clip shares the native constructive topology stages; point and
line clip retain their narrower family kernels. Point, polygon, and line outputs
compose as source-row capacity partitions. Host row-map reads occur only when a
caller explicitly asks for the public Shapely result.

## Consequences

- Phase 5 now has a real owned constructive engine to optimize further
- the owned Point GPU path can keep selected coordinate payloads on device
  across constructive chains
- GeoPandas keeps current host performance while the adapter seam stays visible
- overlay and clip use the same polygon topology and output carriers
