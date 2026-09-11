# Nearest Index Acceleration Recommendation

<!-- DOC_HEADER:START
Scope: Measured GPU nearest acceleration recommendation, build/reuse tradeoffs and public API graduation work.
Read If: You need the conclusion of the nearest experiments or the implementation route for NativeSpatialIndex k=1.
STOP IF: You need individual experiment history rather than the final recommendation.
Source Of Truth: Empirical nearest acceleration recommendation and limits on its production applicability.
Body Budget: 238/240 lines
Document: docs/dev/nearest-index-recommendation.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-9 | Intent |
| 10-18 | Request Signals |
| 19-27 | Open First |
| 28-32 | Verify |
| 33-39 | Risks |
| 40-71 | Recommendation |
| 72-97 | Confirmed MA Results / E14 |
| 98-120 | Build, Reuse and Memory |
| 121-136 | Alternatives and Generalization |
| 137-194 | 2026-09-10 / E17 / General Geometry Direction |
| 195-220 | Production Graduation |
| 221-238 | Reproduction and Verification |
DOC_HEADER:END -->

## Intent

Record the best measured path from the local nearest experiments, its limits,
and the production work needed to expose it through the existing public API.
This is an experimental recommendation, not a shipped performance claim.
Current public integration is recorded in [the integration ledger](strtree-integration-ledger.md).

## Request Signals

- nearest acceleration recommendation
- GPU STRtree
- road snapping
- NativeSpatialIndex k=1
- generalized STR hierarchy
- mixed geometry nearest

## Open First

- docs/dev/generalized-strtree-experiment-ledger.md
- docs/dev/nearest-index-experiment-ledger.md
- docs/dev/nearest-index-strategy-ledger.md
- docs/dev/nearest-index-refinement-ledger.md
- docs/testing/nearest-index-strategy-evidence.json
- docs/testing/osm-nearest-performance.md

## Verify

- `uv run pytest tests/test_experimental_nearest_hierarchy.py tests/test_benchmark_osm_nearest.py --run-gpu -q`
- `uv run python scripts/check_docs.py --check`

## Risks

- Public vS nearest dispatch has not been changed by these experiments.
- Finite 2D LineStrings and points are the experimental admission domain.
- One GPU and one real dataset do not establish a universal optimum.
- Retained index bytes exclude construction scratch, input buffers and outputs.

## Recommendation

Implement a reusable, geometry-independent GPU STR hierarchy in
NativeSpatialIndex. Point-to-line is the first measured specialization:
build on physical segments internally, preserve original road IDs, and return
all equally nearest roads through a NativeRelation. Keep the user-facing
operation on the ordinary spatial index:

```python
indices, distances = roads.sindex.nearest(
    building_points, return_all=True, return_distance=True
)
```

This is the intended existing API integration surface; it does not currently
use the experimental hierarchy. Users should not split roads, choose a grid
resolution, or tune tree fanout to get this behavior. A separate Shapely-style
constructor would be an API decision after this path works through vS.

The leading structure uses recursive STR packing, eight-way branching,
nearest-child-first traversal, Morton ordering of query batches, and a cheap
grid seed that supplies an initial upper bound. Traversal still certifies the
minimum and exact tie membership. Outward-rounded FP32 boxes and enclosed
query coordinates provide conservative coarse filtering; original coordinate
storage and distance refinement remain FP64. No candidate cross-product or
approximate nearest answer is materialized.

Yes, GPU STR trees make sense for this workload. The advantage comes from the
whole path: spatial packing, coherent traversal, early bounds, small leaves,
precision-aware bandwidth use, and reuse. Merely removing the current `k > 1`
cache gate leaves the all-match progressive-radius algorithm unchanged.

## Confirmed MA Results / E14

RTX 4090, i9-13900K, Shapely 2.1.2 / GEOS 3.13.1. Every query scale searches
the full 991,441-road / 8,728,450-segment MA index in EPSG:26986. Query rows
are shuffled prefixes of building representative points. Three fresh-input
warm trials follow a separately recorded first-process trial. Timings include
GPU synchronization, query ordering and int64/float64 NumPy output export;
fixture preparation, imports, file reads and oracle comparison are excluded.

Smallest leaves maximize index reuse throughput:

| Input layout | Queries | Shapely build ms | Shapely repeated ms | GPU build ms | GPU repeated ms | Shapely WKB-to-first ms | GPU WKB-to-first ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Whole roads | 100K | 153.5 | 1789.4 | 22.7 | 3.44 | 2125.9 | 183.7 |
| Whole roads | 1M | 154.0 | 17815.9 | 22.7 | 12.59 | 18356.7 | 224.3 |
| Segments | 100K | 1375.4 | 801.4 | 29.3 | 3.18 | 4117.3 | 457.4 |
| Segments | 1M | 1376.7 | 7987.2 | 29.4 | 12.67 | 11367.0 | 502.7 |

All first and repeated outputs match the Shapely pair multiset; observed
maximum distance error is zero. Segment and road outputs have different ID
domains and tie multiplicities: compare only within a layout. The original
segment Shapely comparator is reused with its validated identity; E10 supplies
the missing whole-road three-trial comparator. No public vS speedup is claimed.
WKB-to-first is the median of each trial's ingress + build + first query,
not a sum of independent medians. It excludes the shared PBF preparation.

## Build, Reuse and Memory

Final conservative-FP32 whole-road results, same full index at both scales:

| Segments per leaf | Build ms | Repeated 100K ms | Repeated 1M ms | Build + first 100K ms | Build + first 1M ms | Retained index MB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 22.7 | 3.44 | 12.59 | 26.23 | 34.63 | 530.5 |
| 2 | 18.0 | 3.98 | 15.84 | 21.92 | 32.86 | 430.8 |
| 4 | 15.8 | 4.86 | 20.17 | 20.53 | 34.87 | 380.9 |

Start production work with leaf width 2 as the balanced candidate: it gives
the lowest measured build + first query at 1M and uses 100 MB less than width
1. Width 1 is the candidate for an index expected to be reused, while width 4
is preferable for smaller one-shot batches or tighter memory budgets. These
are measured tradeoffs, not a reason to expose experimental tuning knobs.
Validate policy on more datasets before making an automatic selector.

Whole-road WKB ingress at 1M is about 190 ms, already much larger than build
or query. Avoiding repeated conversions, retaining the index and query buffers,
and consuming device relations are now the next end-to-end opportunities.
The WKB benchmark does not establish zero-cost ingress from existing Shapely
objects; that convenience boundary needs its own measurement.

## Alternatives and Generalization

The ledgers preserve the full progression and failed attempts. Established
alternatives include the existing tiled fixed-k index, Morton binary and wide
hierarchies, STR leaf tiling, recursive STR, and a direct uniform grid. Exploratory
variants include grid-seeded exact search, conservative FP32 search, and virtual
subdivision of bounds while retaining original geometry arithmetic.

Morton query ordering nearly halves the initial traversal time. Recursive STR
and seed bounds reduce visited nodes and refined segments further. A repaired
direct grid reaches 41.8 ms at 1M but retains 502 MB, compared with 12.7 ms and
531 MB for leaf-1 STR or 20.0 ms and 381 MB for leaf-4 STR. Grid replication
also needs explicit capacity admission. Virtual bounds cost more on MA because
they require an extra count pass and deduplication; do not enable them by
default. See the refinement ledger for independent and overlap stress cases.

## 2026-09-10 / E17 / General Geometry Direction

The user clarified that the production index must generalize beyond roads.
STR packing operates on axis-aligned bounding boxes; it does not require road
semantics. Shapely likewise permits mixed geometry types in its
[STRtree](https://shapely.readthedocs.io/en/stable/strtree.html). Target the same
2D Cartesian semantics first; this proposal does not add geodesic/3D distance.

Separate three contracts inside the existing NativeSpatialIndex owner:

1. A packed bounds hierarchy: conservative boxes, child references, leaf
   references and lifecycle/precision metadata. Share construction and bounds
   pruning across nearest, envelope queries, dwithin and join candidates.
2. Geometry views: map leaves to points, line segments or polygon components,
   with family, part and original-row provenance. Cache secondary primitive
   hierarchies where geometry complexity justifies their build/memory cost.
3. Query/refinement policies: nearest maintains distance bounds and ties;
   envelope search tests overlap; exact predicates and distances consume typed
   device geometry views. They share index state, not identical search logic.

| Logical geometry | Physical representation and exact nearest requirement |
|---|---|
| Point / MultiPoint | Point leaves; reduce multipart results to original rows |
| LineString / MultiLineString | Segment leaves; point-segment or segment-segment refinement; original-row reduction |
| Polygon / MultiPolygon | Component envelopes plus shell/hole topology; containment/intersection and boundary distance, optionally accelerated by boundary hierarchies |
| Mixed / GeometryCollection | Typed component references with original-row lineage; family-aware execution and logical-row reduction |

Boundary segments alone do not cover polygon interiors. A point inside a
polygon or a polygon contained in another has distance zero even when all
boundaries are separated. A point inside a hole usually has positive distance
to the polygon. An envelope-only coarse stage is safe but can be loose;
boundary-only nearest without containment is incorrect. Preserve a complete
coverage/containment path when accelerating polygon boundaries.

Mechanical Shapely checks on a 10x10 square with a central 2x2 hole: point
(2,2) has polygon distance 0 and boundary distance 2; point (5,5) in the hole
has distance 1; an interior box has polygon distance 0 and boundary distance
1. These are semantic examples, not generalized GPU benchmark results.

For mixed workloads, evaluate family partitioning/bounded device refinement
queues; retain fused typed traversal for common homogeneous batches. Any
staged queue must feed refined upper bounds back into pruning without
materializing an unbounded candidate relation. Large query geometries may
benefit from hierarchy-pair traversal or component work with per-query minima.
Both indexed and query geometries must be supported, not only point queries.

Multipart decomposition preserves distance minima, but k counts distinct
logical features, and all-match output deduplicates feature pairs. Predicates
such as contains/within require whole-geometry semantics, not merely a hit on
one fragment. Original IDs and hole/ring relationships remain authoritative.

Follow-on E18-E26 results are in the generalized STR ledger linked above.
Shared construction and six-family device refinement now pass both query
directions and win the simple 100K/1M screening fixtures. MA performance holds.
Complex polygons expose a segment cross-product bottleneck even after fixing
launch sizing; component/segment hierarchies are the next experiment. Public
integration, collection ingestion and extreme-coordinate proof remain open.

## Production Graduation

1. Add a segment-shaped index payload to NativeSpatialIndex with parent IDs,
   readiness, cache ownership/invalidation, and build work estimates. Reuse
   NativeGeometryMetadata for finite/empty admission and coordinate statistics.
2. Connect all-match k=1 through the normal public nearest path. Use
   count/scan/scatter and parent-pair deduplication to produce NativeRelation;
   export only at the requesting boundary. Make fallback decisions observable.
3. Replace experimental precision policy with normal PrecisionPlan dispatch.
   Prove or refine the FP64 metric-error pruning guard, test extreme magnitudes
   and cancellation, and preserve strict tie identity. Conservative FP32
   bounds alone do not prove every GEOS distance corner case.
4. Complete empty/null, mixed-family, MultiLineString, max-distance, exclusive,
   return-all, stream/lifetime, capacity and mutation contracts. The present
   generalized canaries extend family coverage; public contracts remain open.
5. Register kernels and cached CCCL primitives with normal warmup; budget
   scratch as well as retained buffers. Replace experimental scalar admission
   synchronizations where metadata can supply the answer.
6. Re-run the unchanged public benchmark at 100K/1M, with exact all-match
   parity and separately reported conversion, cold/build and reused-query
   time. Profile raw Shapely ingress and native downstream consumption too.

Production work should complete the existing Native* coverage mandate, with
the experimental ledgers and retained source snapshots as evidence. No
production dispatch changes, commit or push are part of this research phase.

## Reproduction and Verification

With the validated MA fixture cache already present, reproduce the balanced
1M whole-road candidate using a new output path:

```bash
uv run python scripts/experiment_nearest_strategies.py \
  --layout roads --rows 1000000 --repeat 3 --packing str-recursive \
  --fanout 8 --leaf-width 2 --query-order morton --seed-resolution 2048 \
  --bounds-precision fp32-outward --output /tmp/nearest-roads-balanced-new.json
```

E17 verification: 57 targeted tests pass; CUDA memcheck and initcheck each
report zero errors across all 53 GPU canaries. Ruff and documentation checks
pass. The full pipeline profile passes 22 executed cases, with two existing
raster deferrals; every stage was reviewed. The refinement ledger records the
51 stage names and times at the 1M scale label, while the evidence JSON retains
all 102 stages, source identities, trial records and failure dispositions.
