# Spatial Index Backends

<!-- DOC_HEADER:START
Scope: Operation capabilities, cache lineage and precision for native spatial index backends.
Read If: You are selecting spatial-index backends or changing public nearest integration.
STOP IF: You only need historical experiment timings.
Source Of Truth: Native spatial-index layout and query-strategy contracts.
Body Budget: 157/200 lines
Document: docs/architecture/spatial-index-backends.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-15 | Request Signals |
| 16-24 | Open First |
| 25-30 | Verify |
| 31-37 | Risks |
| 38-68 | Public Contract |
| 69-100 | Layouts and Strategies |
| 101-147 | Physical Shape and Precision |
| 148-157 | Evidence |
DOC_HEADER:END -->

## Intent

Keep one public spatial-index API while choosing reusable execution state by
operation. Packed STR nearest completes NativeSpatialIndex functionality;
flat envelope queries and bounded fixed-k queries retain their own strategies.

## Request Signals

- sindex backends
- packed STR nearest
- native spatial index capabilities
- nearest cache invalidation

## Open First

- src/vibespatial/spatial/index_backends.py
- src/vibespatial/api/sindex.py
- src/vibespatial/api/_native_metadata.py
- src/vibespatial/kernels/spatial/packed_str_index.py
- src/vibespatial/kernels/spatial/segment_bvh.py
- docs/dev/strtree-integration-ledger.md

## Verify

- `uv run pytest tests/test_packed_str_index.py tests/test_segment_bvh.py --run-gpu -q`
- `uv run pytest tests/upstream/geopandas/tests/test_sindex.py tests/upstream/geopandas/tools/tests/test_sjoin.py -k nearest --run-gpu -q`
- `uv run python scripts/benchmark_generalized_strtree.py --help`

## Risks

- A single global backend switch obscures different operation contracts.
- Segment boundaries alone do not capture polygon containment or holes.
- Cached state must follow geometry lineage and producing CUDA streams.
- Nearest ties require exact terminal distances, not a tolerance-based tie test.

## Public Contract

```python
import vibespatial as vs

roads = vs.GeoSeries.from_wkb(road_wkb)
buildings = vs.GeoSeries.from_wkb(building_point_wkb)
indices, distances = roads.sindex.nearest(
    buildings, return_all=True, return_distance=True
)
```

The public result is the usual `(2, n)` index array and optional distance array.
Native consumers use `nearest_relation()` and keep indices/distances on device.
Public `size`/`len(index)` count valid nonempty features; internal row domains
include every logical position so null/empty rows remain aligned in aggregates.
Scalar Shapely geometries and arrays of Shapely geometries enter through the
compatibility boundary. Null/empty rows produce no matches. Row IDs refer to
logical input positions, including duplicate rows in indexed views.

`return_all=True` preserves every exact minimum-distance tie. Output is ordered
by query row, then tree row; `return_all=False` chooses the lowest tree row
among ties. `max_distance` is a positive distance ceiling. `exclusive=True`
excludes geometrically equal features, including reversed/redundant line
vertices and alternative multipart representations; it does not merely exclude
matching row numbers or every intersecting geometry.

`roads.sindex.backend_info` reports capabilities and cached state without
building anything. Dispatch events identify the implementation actually used.
There is no public tuning flag for leaf size, segment tiles, or backend choice.

## Layouts and Strategies

| Backend | Operation | Geometry admission | Reusable state |
|---|---|---|---|
| `flat-morton` | envelope/predicate query, aggregates | canonical families | bounds, Morton order, optional regular grid |
| `packed-str` | k=1 nearest, all ties/exclusive/radius | six canonical families, mixed arrays | packed envelopes, optional segment BVHs |
| `bounded-knn` | k>1 with `return_all=False` | existing homogeneous-family refiners | flat state, optional point partitions |
| `strtree-host` | host query/nearest | Shapely-admitted input | lazy or host-built Shapely STRtree |

The capability registry describes contracts, not interchangeable storage
classes. Regular grids, point partitions, and segment BVHs are accelerators
within those contracts. For example, a segment BVH refines an STR candidate; it
is not a replacement for a feature index. Fixed-k exclusion remains explicitly
unsupported by its existing native contract.

`NativeSpatialIndex.kind` continues to identify its flat layout. Its backend
cache can retain packed STR alongside that layout. GeometryArray invalidation
replaces the flat index, which changes the native cache identity and discards
all derived state together. A changed source token also invalidates reuse.
Packed state is keyed by bounds precision; fanout and tile sizes are private
implementation constants rather than user-selected cache parameters.

A lock protects backend construction. Each packed tree serializes access to its
prepared-query cache and records a CUDA event after construction and querying;
consumers on another stream wait for that event. Independent index objects can
execute independently. SpatialIndex weakly references its public GeometryArray
while retaining the underlying geometry independently, so the index can outlive
the array without a cache cycle. RMM stream wrappers borrow handles with one-way
references. The native CuPy allocator roots each RMM allocation and its stream
through a memory-owner finalizer, preserving asynchronous deallocation order
even when user code puts device arrays in Python reference cycles.

## Physical Shape and Precision

Construction packs feature envelopes into an eight-way STR hierarchy. Original
row IDs remain separate from physical leaf order. Reused FP64 metadata bounds
are authoritative; the COARSE PrecisionPlan chooses FP64 traversal or outward
FP32 envelope compression. Sort/scan operations use CCCL and typed device arrays.

Query envelopes are spatially ordered, then processed in batches of at most
65,536 rows with eight candidate slots each. A resumable DFS keeps bounded
scratch. The minimum pass first refines one promising leaf to tighten pruning.
Count/scan/scatter passes produce exact ties without storing all candidates.
Dense overlap remains output-sensitive. A scalar convergence fence per wave
and an output-allocation fence per batch are explicit host control boundaries;
geometry, candidate and result columns remain on device until public export.

Simple candidates use existing all-family GPU refiners. Complex boundaries
use cached segment endpoints and per-row binary bounds trees. One warp refines
a candidate by distributing query segments over indexed segment subtrees.
Admission uses cached maximum coordinates per row, deriving missing bounds
from device offsets once. Unrelated simple/null rows cannot hide a large row.
Containment considers component/ring anchors in both directions, including
holes, and is skipped only when feature envelopes are disjoint. Point-family
candidates in mixed batches retain the existing point refiners.
Segment distance certifies intersection, then minimizes four endpoint-to-segment
projections. Point and segment refiners share the same precision-parametric
relative-residual helper; no near-parallel determinant or tiny-edge cutoff is used.

Both traversal precisions use directed lower-bound arithmetic; FP32 leaf boxes
round outward. Metric ordering and tie membership use the selected FP64 final
refinement context. No FP32 metric value decides an exact tie. The pruning
margin broadens search only; the terminal predicate remains `distance == best`.
The margin is a numerical guard, not a claim of exact real-arithmetic distance
for arbitrary extreme coordinates. Coordinate-offset and narrow-gap canaries
complement the Shapely pair/distance oracle.

Current integer admission is fewer than 2**24 feature/query rows, fewer than
2**29 non-point coordinates for segment extraction, fewer than 2**24 segments
in a single row, and fewer than 2**29 BVH nodes (four int32-indexed bound planes).
These bounds prevent
stack/offset overflow; admission precedes the relevant extraction or traversal
launch. Result offsets use int64. Null/empty feature envelopes
are omitted from the index while preserving original positions.

Device candidate counts also bound predicate classification, DE-9IM decoding
and transposition. Scratch capacity beyond the count is never consumed;
full predicate masks define inactive entries as false for reduction consumers.

## Evidence

The integration ledger records rejected cooperative cross-products, segment
tiling experiments, complex shape canaries, public 100K/1M comparisons and
verification. CPU comparator files are immutable. A lazy public sindex accessor
can be nearly free: tree preparation is included in the first-query timing and
must not be presented as a free index build. Reused-query timings include public
index/distance export. Historical prototype timings are not public API claims.
Current measurements and complete 1M profile stages are recorded in
`docs/testing/strtree-integration-results.md` with a durable evidence manifest.
