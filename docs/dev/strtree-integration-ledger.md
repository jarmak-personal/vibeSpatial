# STRtree Integration Ledger

<!-- DOC_HEADER:START
Scope: Complex polygon refinement experiments and NativeSpatialIndex backend integration ledger.
Read If: You are accelerating complex geometry nearest or integrating spatial index backends.
STOP IF: You only need historical MA or simple generalized STR screening results.
Source Of Truth: Current complex-refinement and sindex integration work and evidence.
Body Budget: 255/260 lines
Document: docs/dev/strtree-integration-ledger.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-15 | Request Signals |
| 16-25 | Open First |
| 26-31 | Verify |
| 32-38 | Risks |
| 39-66 | E28 / Complex Refinement Shape |
| 67-78 | Integration Sequence |
| 79-95 | E28 / Hierarchy Beats Cooperative Cross-Products |
| 96-124 | E29-E32 / Large-Single, Skew, Holes and Dense Overlap |
| 125-145 | E33 / Native Integration and Stream Lifetime |
| 146-170 | E34-E41 / Public Nearest and MA Correctness |
| 171-188 | E42-E43 / Cyclic Allocation Ownership |
| 189-204 | E46-E50 / Complex Scale and Public Screening |
| ... | (3 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Continue the generalized STR experiments through complex-geometry refinement
and public sindex integration. This is NativeSpatialIndex completion work.
Keep local changes and immutable comparator evidence; no commit/push is requested.

## Request Signals

- complex polygon nearest
- sindex backends
- STRtree integration
- segment hierarchy refinement

## Open First

- docs/dev/generalized-strtree-experiment-ledger.md
- scripts/experiment_segment_bvh.py
- scripts/experimental_segment_bvh_kernels.py
- src/vibespatial/api/sindex.py
- src/vibespatial/api/_native_metadata.py
- src/vibespatial/spatial/nearest.py
- src/vibespatial/spatial/segment_distance.py

## Verify

- `uv run pytest tests/test_generalized_strtree_experiment.py tests/test_spatial_query.py --run-gpu -q`
- `uv run pytest tests/upstream/geopandas/tests/test_sindex.py -q`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Boundary distance alone misses polygon containment and hole semantics.
- Small row counts can conceal millions of segment-pair tests.
- A backend choice must preserve cache lineage, precision and query semantics.
- Existing nullable Arrow ingestion has an independently reproduced cuDF initcheck issue.

## E28 / Complex Refinement Shape

Compare two structures: warp-cooperative segment cross-products, and a cached
binary bounds hierarchy per geometry with cooperative segment-to-tree traversal.
The former exposes parallelism but retains quadratic work; the latter can prune
primitive work. Keep the same top-level STR and cached Shapely E21 oracle.

Physical shape: bounded candidate relation -> candidate/segment work -> per-pair
distance -> nearest NativeRelation. Reuse owned geometry and device segment
extraction. Typed SoA endpoints and row offsets preserve original row identity;
temporary bounds trees are execution state, not a new geometry storage format.
Count/scan and sorting use existing CuPy/CCCL wrappers; geometry traversal and
exact refinement use NVRTC. No Python geometry loops or host coordinate scans.

Start with the four non-point canonical families. Each warp refines a candidate;
lanes traverse independent query segments against the indexed row hierarchy.
This distributes work for many complex rows; large-single and skewed geometries
also need segment-tile canaries before production admission. Containment checks
retain component/ring anchors and the existing polygonal-family predicate.
Sparse candidates stay in bounded tiles; dense ties remain output-sized.

Reference metric uses explicit FP64 PrecisionPlan and shared exact-contact math;
conservative outward FP32 bounds only prune. Production precision selection,
readiness and cache invalidation must be wired before public integration.
Measure ingress, index preparation, first query, reused query, primitive/node
work and retained/scratch memory separately. Never substitute historical GPU
timings for the new candidate. Reuse CPU only after identity/oracle validation.

## Integration Sequence

1. Prove complex refinement correctness and improved work shape against E21,
   then broaden complexity, overlap, multipart, null/empty and tie canaries.
2. Define backend capabilities and selection inside NativeSpatialIndex. Existing
   flat-Morton, regular-grid and point-partition state should remain reusable;
   STR nearest state should share source lineage and stream readiness.
3. Wire k=1 nearest through cached native state, preserve k>1 behavior and all
   public options, and expose backend decisions through existing diagnostics.
4. Validate public sindex/sjoin_nearest, cache mutation/lifetime, strict-native
   behavior, unchanged MA 100K/1M comparator, mixed fixtures and full profiles.

## E28 / Hierarchy Beats Cooperative Cross-Products

Same E21 fixtures, CPU environment and hashed oracle; one first and one fresh
warm query, milliseconds. These are screening trials, not stable medians.

| Rows / vertices per ring | Shapely query | Previous GPU | Cooperative | Segment BVH |
|---|---:|---:|---:|---:|
| 10K / 16 | 191.118 | 69.558 | 32.694 | 16.057 |
| 10K / 64 | 1234.032 | 958.475 | 340.087 | 45.861 |
| 1K / 256 | 1155.572 | 4908.185 | 457.256 | 21.315 |

Each polygon has a shell and hole. The last case therefore has 512 boundary
segments per row. Segment tests fall from 2.89 billion in the cooperative
cross-product to 11.1 million with bounds pruning. BVH builds cost 4.7-5.5 ms;
retained tree buffers are 20.4/81.9/32.8 MB. All oracle comparisons pass.
Raw evidence: `/tmp/vibespatial-osm/nearest-experiments/e28-*-complexity.*`.

## E29-E32 / Large-Single, Skew, Holes and Dense Overlap

Four deterministic fixtures broaden the physical shape: one 16,384-edge polygon,
128 rows with one 16,384-edge outlier, 256 polygons with 16 holes each, and
128 overlapping 256-edge polygons producing 16,384 nearest ties.
E29 owns the immutable Shapely timings/oracles. Subsequent trials validate and
reuse those comparators without retiming CPU. All reported cases pass.

| Case | Shapely | E29 BVH | E31 overlap guard | E32 first-candidate seed |
|---|---:|---:|---:|---:|
| Single | 0.769 | 25.833 | 3.541 | 3.503 |
| Skew | 25.338 | 53.418 | 24.894 | 3.697 |
| Many holes | 83.723 | 10.589 | 3.211 | 3.020 |
| Dense overlap | 33.482 | 28.237 | 28.279 | 28.808 |

Times are reused-query milliseconds, one trial each. GPU remains slower for
one geometry; the target remains 100K/1M, not small-input parity. Segment tiling
without a seed caused redundant traversal; seeding restored work efficiency
but extra launches generally lost to the warp-per-candidate schedule. Retain
tiling as an experiment, not the production default.

E31 skips polygon containment when feature envelopes are disjoint. E32 refines
one promising candidate before filling wider slots in the minimum-distance
pass. The skew case then needs only 384 candidate refinements across all three
passes. Dense zero ties remain output-sensitive and see no such reduction.
E32 verification: 60 tests passed. Shared metric helper extraction afterward:
38 segment/existing-BVH tests passed. Raw cases, source snapshots, fixture hashes
and oracles are in `e29-*`, `e30-*`, `e31-*`, and `e32-*` artifacts.

## E33 / Native Integration and Stream Lifetime

Promoted packed STR state and cooperative segment BVHs into
`src/vibespatial/kernels/spatial/`. The k=1 public/native route now caches STR
under NativeSpatialIndex lineage, with both conservative bounds precisions,
FP64 final refinement, sorted NativeRelation output, max-distance admission,
exact ties and device topological equality for `exclusive=True`. Existing flat
queries and bounded fixed-k paths remain distinct operation contracts.
`SpatialIndex.backend_info` reports capabilities and retained layouts without
building them. Host query inputs enter through the existing owned boundary.

Initial public tests caught a physical-work-estimate row-count mismatch; fixed
before measuring. A later run passed all 28 assertions but crashed during GC.
GDB traced this to RMM `cuMemFreeAsync`, not compilation. The existing stream
wrapper cache formed CuPy stream -> RMM wrapper -> CuPy stream cycles, allowing
GC to clear a stream before asynchronous buffer destruction. Fresh borrowed
RMM wrappers preserve one-way ownership and allocate no new CUDA streams.
The ordinary stream-reuse/mutation test reproduces the old crash; after removal,
61 native/public/memory-pool tests pass and the process exits successfully.
Raw failures, GDB trace, pre-fix runtime and passing logs: `e33-*` artifacts.

## E34-E41 / Public Nearest and MA Correctness

Public synthetic screening preserves the large-scale advantage; E40 validates
all 14 E19 comparator identities (seven family directions, 100K/1M). E41 repeats
the four complex shape fixtures through `sindex.nearest`, all passing. These
measurements precede the final allocator-lifetime changes and are historical.
The public accessor is lazy; first-query time includes actual index preparation.

Upstream nearest initially exposed tie-order differences. Native output now
orders `(query_row, tree_row)` and selects the lowest tree ID for a single tie;
81 nearest/sjoin_nearest tests then pass. Full upstream sindex/sjoin coverage
after lifetime integration: 416 pass, 59 optional/version skips, one existing
xfail. No vendored tests were edited.

E37 MA segments/all found one different nearest ID among 2,834,045 pairs.
At building row 806049, projection onto segment 655134 is just inside its end,
3.8e-13 m closer than segment 655135's shared endpoint. Reconstructing the
closest point in world coordinates erased that improvement. Point-segment
metrics now compute relative residuals, with direct endpoint distances and
exact-zero degeneracy checks. A literal-coordinate regression reproduces it.
E39 all eight MA cases then match every saved pair/distance oracle: full roads
0.950 s versus 48.111 s CPU; segments 0.730 s versus 21.464 s CPU. These are
reused-query screening times for all 2,680,339 building points, before final
allocator ownership changes. E38 native/distance suite: 252 pass, one SciPy skip.

## E42-E43 / Cyclic Allocation Ownership

Broader transient-stream tests reproduced the deallocation crash again. A
standalone RMM/CuPy cycle, without vibeSpatial, also exits with SIGSEGV. CuPy
stream weakref finalization can run before native buffer destruction; a Python
`__del__` alone is too late. The native allocator now roots the original RMM
allocation and stream through the CuPy memory owner's weakref finalizer. It adds
no device allocations, copies, streams or synchronization. Eight standalone
cycles and 110 native/public/memory-pool tests then pass with normal process
exit. Failed alternatives and the independent reproducer remain in `e42-*` and
`e43-*`; earlier cycle-removal claims alone were insufficient.

SpatialIndex also borrows its public GeometryArray through a weak reference,
retaining independent owned buffers when the public array is released. GPU
index access for host geometry arrays now defers the host STRtree too. Lazy
index size counts valid nonempty features, including owned-backed inputs.
Convergence and allocation scalar fences are named in runtime transfer events.

## E46-E50 / Complex Scale and Public Screening

Added two immutable holed-polygon CPU comparators: 100K rows with 128 boundary
edges per polygon and 1M rows with 32 edges. E50 public reused queries take
130.858 ms / 688.774 ms versus 12,747.051 ms / 28,104.619 ms Shapely, with
identical pair sets. GPU first queries include hierarchy preparation and take
196.055 ms / 869.494 ms; WKB ingress is separately 272.868 ms / 668.559 ms.
Both query and tree have the stated row count; these are fresh-input screening
trials, not confidence intervals. Fixture, oracle and environment hashes are saved.

E47 MA all eight cases, E48 all 14 simple/mixed family cases, and E49 all four
complex shape canaries pass. These measurements precede the review corrections
below and remain historical. E50 full profile: 22 runnable cases pass, two raster
deferrals, 102 stages reviewed, zero fallbacks. Maximum stage is 82.001 ms;
no stage over 5 ms regresses more than 1.35x against E45.

## E51-E55 / Fresh Review and Root Corrections

Independent review found two blockers. Ericson's closest-approach denominator
cancels for nearly parallel segments, and its 1e-30 cutoff collapses nonzero
short segments. Both can select the wrong nearest feature. The shared FP64
segment refiner now certifies contact then minimizes four endpoint projections;
the precision-parametric point helper uses relative residuals and exact-zero
degeneracy. Regressions exercise simple and BVH refiners in both bounds precisions.

Average complexity also hid expensive outliers when unrelated simple/null rows
were added. Admission now reuses maximum per-row structural bounds, with a
single named device-offset planning packet when proof is missing. Null/simple
padding canaries assert both oracle parity and hierarchy admission.

Broader tests exposed internal use of public index `size` as a row domain.
Internal allocations, reductions and shape validation now use metadata-only
logical row counts; public size still excludes null/empty features. A paired
aggregate regression preserves all four positions despite unequal valid counts.
The nearest attribute-filter test now checks the documented seven scalar fences
(14 bytes total for its one-wave fixture), including exact reasons and sizes,
while forbidding relation/geometry exports. It no longer assumes the old engine's
single fence. Metric/runtime suite: 337 pass, one optional SciPy skip.

## E60-E64 / Initialization and Typed Primitive Contracts

Memcheck passes 113 tests. Initcheck then exposes existing predicate consumers
reading uninitialized capacity beyond device candidate counts. DE-9IM decoding
and transposition now reuse count-aware grouped kernels. Point classification
uses direct indexed input for device-counted prefixes, avoiding full-capacity
row-ID gathers; predicate decoding also respects the prefix. Full predicate
masks define inactive entries as false while relation/index scratch stays unread.
Owned nearest/sort initcheck: 38 pass, zero errors. Dedicated count-zero,
partial-prefix and swapped-family regressions: 18 pass, zero errors.

Unfiltered public initcheck passes 94 assertions but reports eight reads in
libcudf `count_set_bits_kernel`, matching the separately reproduced nullable
Arrow dependency issue. It is not a clean end-to-end initcheck result.
Failed runs and root corrections remain in E55/E61/E62/E63 artifacts.
Added typed CCCL warmup for uint32/int32 Morton sorting and int64/int64 result
sorting; high-bit key/value tests require the radix path and preserve exact IDs.

## E56-E59 and E65 / Final Integration Evidence

Final public runs pass all 20 synthetic/complex and eight MA oracle comparisons.
Holed polygons: 100K reused 121.561 ms vs 12,747.051 ms CPU; 1M reused
675.971 ms vs 28,104.619 ms CPU. First queries include hierarchy preparation.
The final profile reviews all 102 stages: 22 runnable cases, two raster
deferrals, zero fallbacks, maximum stage 75.540 ms. All seven deterministic
checks pass. Detailed timings, 1M stages and validation limits are in
`docs/testing/strtree-integration-results.md`; durable source/oracle fingerprints
and review records are in `docs/testing/strtree-integration-evidence.json`.
