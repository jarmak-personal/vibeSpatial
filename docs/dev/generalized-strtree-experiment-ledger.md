# Generalized STR Hierarchy Experiment Ledger

<!-- DOC_HEADER:START
Scope: Generalized GPU STR hierarchy experiments, bounded all-family refinement and measured regression evidence.
Read If: You are generalizing the nearest hierarchy across geometry families or testing bounded refinement.
STOP IF: You only need the initial MA segment performance comparison.
Source Of Truth: Local generalized STR experiment contracts, results and limitations.
Body Budget: 248/260 lines
Document: docs/dev/generalized-strtree-experiment-ledger.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-10 | Intent |
| 11-16 | Request Signals |
| 17-30 | Open First |
| 31-36 | Verify |
| 37-43 | Risks |
| 44-96 | 2026-09-10 / E18 / Shared Construction and Bounded Refinement |
| 97-170 | E19-E24 / Independent Geometry Screening |
| 171-208 | Reproduction and Remaining Contracts |
| 209-248 | Verification and Full Pipeline Profile |
DOC_HEADER:END -->

## Intent

Continue the nearest experiments with geometry-independent construction and
mixed-family queries. Preserve the measured segment specialization and expose
limitations through evidence before production NativeSpatialIndex integration.
Current public integration and complex-boundary results continue in
[the integration ledger](strtree-integration-ledger.md).

## Request Signals

- generalized STR hierarchy
- mixed geometry nearest
- bounded nearest refinement

## Open First

- docs/dev/nearest-index-recommendation.md
- scripts/experimental_strtree_bounds.py
- scripts/experiment_generalized_strtree.py
- scripts/experimental_strtree_kernels.py
- scripts/experimental_strtree_bounds_kernels.py
- scripts/benchmark_generalized_strtree.py
- tests/test_generalized_strtree_experiment.py
- scripts/experiment_nearest_strategies.py
- src/vibespatial/spatial/nearest.py
- src/vibespatial/spatial/distance_owned.py
- docs/testing/generalized-strtree-evidence.json

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run pytest tests/test_experimental_nearest_hierarchy.py --run-gpu -q`
- `uv run pytest tests/test_generalized_strtree_experiment.py tests/test_segment_distance.py --run-gpu -q`

## Risks

- Existing distance refiners may have different tie arithmetic from Shapely.
- Broad envelopes increase traversal; high-complexity rows increase refinement.
- GeometryCollection is not a canonical OwnedGeometryArray family today.
- Prototype performance is not a public API performance claim.

## 2026-09-10 / E18 / Shared Construction and Bounded Refinement

Two structures considered: fused typed traversal over physical components,
and resumable bounds traversal feeding bounded candidate tiles to existing
device refiners. Start with the latter to separate coverage from specialization;
retain the fused segment path using the same extracted construction methods.
Do not wire a slower experimental path into production merely for coverage.

Physical contract: generic finite 2D geometry envelopes -> packed STR nodes ->
resumable per-query DFS -> bounded candidate relation -> family-specific exact
distance refinement -> nearest NativeRelation. Input geometry stays owned on
device. Null/empty envelopes are skipped with original row IDs preserved.
All six existing owned families are in scope; collections require separate
component ingestion and must not silently enter a host fallback.

Work units are index boxes/nodes, query-node visits, candidate pairs, geometry
coordinates/segments/rings, output pairs and bytes. Query batches bound the
DFS arena; fixed candidate slots bound scratch independently of tree size.
Minimum, fixed-minimum count, and scatter traversals avoid retaining candidate
history. The result is output-sized, including all distinct feature ties.

Shared construction uses existing CCCL sorting and geometry bounds kernels.
New traversal is geometry-specific Tier 1 NVRTC; gather/reduction is device
array work. FP64 metric PrecisionPlan is explicit for the reference experiment;
coarse FP64 and conservative outward FP32 bounds share the same tests.
Only progress/allocation scalars cross the host boundary during execution;
NumPy indices/distances are a terminal convenience export. Scalar fences,
refiner launches and candidate occupancy are measured before optimization.

Large batches provide parallel query work. Large-single and heavily overlapping
geometry expose the row-refiner and DFS limitations and need separate canaries;
they are not assumed to inherit the segment benchmark result. No road-name,
dataset, CRS, or fixed spatial grid assumptions enter the shared builder.

### Refiner Correctness Finding

The first six-by-six family matrix had three failures involving MultiPolygon
queries. The new tree omitted genuine zero-distance ties. Directly calling the
existing device refiner on missing oracle pairs reproduced distances
3.55e-18 and 7.55e-18 where Shapely returns exactly zero, independent of bounds
traversal. Closest-approach segment arithmetic left rounding residuals at
intersections; another intersecting feature returned zero and won the minimum.

Considered replacing all segment metric arithmetic versus adding exact
intersection classification before the existing disjoint-distance calculation.
Chose the latter: bounding-box rejection avoids unnecessary predicates and
the existing adaptive `vs_orient2d` certifies crossings, touches and overlaps.
No distance tolerance creates ties. This is a necessary shared kernel fix in
`src/vibespatial/spatial/segment_distance_kernels.py`, not a dispatch change.
The regression checks exact zero and a positive 1e-10 gap. The subsequent
117-test run passes, including all 36 family combinations, polygon holes,
containment, null/empty rows, mixed ties and the earlier segment experiments.

## E19-E24 / Independent Geometry Screening

RTX 4090, i9-13900K, Python 3.13.12, Shapely 2.1.2 / GEOS 3.13.1.
Each scale has equally many indexed and query features. Independent seeded
fixtures use uniform centres, points, two-vertex lines, square polygons,
two-part MultiPolygons and round-robin mixtures. These are low-complexity
features, not representative of every polygon workload.

Both backends receive identical WKB. Ingress, completed index construction,
first query and repeated query are separate; terminal NumPy indices/distances
and synchronization are included. Fixture generation, imports and oracle work
are excluded. Each case has one first trial and one fresh-input warm trial:
screening evidence, not three-trial medians or process-cold latency claims.
E19 supplies immutable Shapely results and hashed oracles. E20 extracts shared
construction kernels; E24 reuses the exact E19 identity after launch sizing.
Reports retain both first/reused validations, source hashes and trial times.

Repeated query milliseconds, after the launch correction:

| Query -> indexed family | Shapely 100K | GPU 100K | Shapely 1M | GPU 1M |
|---|---:|---:|---:|---:|
| Point -> Point | 257.01 | 25.37 | 4012.80 | 162.06 |
| Point -> Polygon | 639.43 | 19.93 | 8744.72 | 169.84 |
| Polygon -> Point | 523.59 | 19.99 | 7361.89 | 165.58 |
| LineString -> Polygon | 655.96 | 22.03 | 10000.66 | 178.89 |
| Polygon -> Polygon | 883.12 | 24.89 | 10932.50 | 211.26 |
| MultiPolygon -> MultiPolygon | 1568.11 | 42.69 | 18168.19 | 383.38 |
| Mixed -> Mixed | 878.60 | 37.56 | 11634.95 | 308.98 |

All 14 cases preserve exact all-match pair multisets. Maximum distance
difference is 2.71e-12 coordinate units; validation permits 1e-6 absolute plus
1e-10 relative error. Tie selection itself uses exact distance equality.
Warm build is 3.2-8.7 ms; retained bounds/children/IDs occupy 2.69 MB at 100K,
26.86 MB at 1M, excluding geometry, scratch and output. Same-WKB ingress +
build + first-query speedups range 9.5-23.6x at 100K and 18.8-34.4x at 1M.
These are experimental API measurements, not a production `.sindex` claim.

### Complex Geometry Canary and Launch Finding

Regular polygons have one shell and one hole. E21 established the comparator
and the pre-correction GPU timings; E24 reuses its identical fixtures/oracles.
Ring vertex counts exclude the repeated closure coordinate.

| Rows per side | Vertices per ring | Shapely reused ms | GPU before ms | GPU after ms |
|---:|---:|---:|---:|---:|
| 10,000 | 16 | 191.12 | 281.86 | 69.56 |
| 10,000 | 64 | 1234.03 | 4302.80 | 958.48 |
| 1,000 | 256 | 1155.57 | 49154.94 | 4908.18 |

The existing mixed-family refiner divided launch capacity by all 36 schema
family pairs even for homogeneous inputs. For 8,000 candidate slots this
launched only 223 threads, forcing serial work through large polygons. It now
sizes launches by the family pairs present in the inputs; partition storage
still covers the complete tag domain. Grid-stride bounds and device logical
counts remain authoritative. Tests cover full homogeneous parallelism, no
present families, and FP32 ambiguity selection with only one active FP64 pair.

This correction improves throughput but does not change refinement complexity.
The two 10K cases have almost identical traversal (576K node visits / 133K
active candidates); quadrupling edges still increases time almost fourteenfold.
The 512-edge polygon case remains 4.25x slower than Shapely. Stop tuning this
row-shaped segment cross-product: the next experiment should cache per-geometry
segment hierarchies and use cooperative candidate/segment work, preserving
polygon containment and holes. The top-level bounds tree is reusable as-is.

### MA Regression

E22 repeats the previous fused strategy after shared construction extraction,
using the same MA fixture and three fresh warm trials. At 1M queries the
whole-road path takes 12.78 ms reused / 22.71 ms build; explicit segments take
12.54 ms reused / 29.31 ms build. Every first/reused output passes exact pair
and distance checks. This is within variation of the prior ~12.6 ms results.
The generalized refiner is not substituted into that specialized path.

## Reproduction and Remaining Contracts

The experiment accepts vS GeoSeries on either side and returns a NativeRelation
from `query_relation`, or Shapely-shaped array results from `query_nearest`:

```python
# Start Python with PYTHONPATH=scripts.
from experiment_generalized_strtree import GeneralizedSTRtree
tree = GeneralizedSTRtree(indexed_geometries)
indices, distances = tree.query_nearest(
    query_geometries, all_matches=True, return_distance=True
)
```

Run a fresh independent comparison with a new output path:

```bash
uv run python scripts/benchmark_generalized_strtree.py \
  --tree-family polygon --query-family polygon \
  --tree-rows 1000000 --rows 1000000 --repeat 1 \
  --output /tmp/generalized-strtree-new.json
```

Use `--reuse-comparator <matching-report.json>` for GPU-only reruns. The
durable evidence includes trial records, source/oracle identities, the complex
fixture reproducer and local raw artifact manifest under
`/tmp/vibespatial-osm/nearest-experiments/`.

Next contracts: component views and parent deduplication; complex-geometry
refinement; mixed-family work imbalance; public max-distance/exclusive/scalar
semantics; GeometryCollection ingestion; cache invalidation and stream lifetime;
normal PrecisionPlan/warmup dispatch; proof or refinement of the metric-error
pruning guard at extreme magnitudes. Prototype limits each side to fewer than
2**24 rows and batches queries to bound its 128-entry DFS stacks/candidate tiles.
Dense all-tie output still requires output-sized storage. Bounds filtering
skips nonfinite envelopes; arbitrary nonfinite geometry semantics are unproven.
Production NativeSpatialIndex/public nearest integration remains separate work.

## Verification and Full Pipeline Profile

E26: 285 tests pass; one spatial-query test skips because SciPy is not installed.
E27 adds two passing translated positive-distance tie tests across all families.
Coverage includes all 36 family pairs, translated holes/nulls/mixed ties, empty
inputs, 129-way ties across candidate tiles, prior MA canaries and production
spatial-query/refiner regressions. Memcheck reports zero errors for 123 GPU
canaries plus two launch tests and two positive-tie tests. Owned-ingress initcheck
passes all 48 generalized canaries with zero errors.

The unfiltered initcheck does fail: 20 uninitialized reads inside libcudf
`count_set_bits_kernel` during nullable Arrow import. A standalone
`plc.Column.from_arrow(pa.array([None, "a"]))` reproduces two errors without
importing vS (pylibcudf/libcudf 26.2.1, PyArrow 23.0.1). Roundtrip values pass.
Retain the upstream-boundary reproducer; do not claim clean end-to-end initcheck
or suppress it. The owned-ingress check changes only test fixture construction,
not runtime dispatch. Root cause beyond libcudf mask import remains upstream
investigation; bitmask padding is a hypothesis, not a certified diagnosis.

E23 full profile completes 22 cases with two existing raster deferrals. All
102 stages were reviewed; no unexpected >1s CPU stage. At the 1M suite label,
maximum stage is read_geojson at 70.91 ms; CPU index setup is 0.30 ms. Grouped
union/capacity scenarios retain 8/12 compute D2H transfers (8,048/12,480 bytes),
zero fallback events; other scenarios have zero compute D2H. Some constructive
fixtures cap actual rows below the suite scale; evidence retains stage row counts.
The following names and milliseconds include every stage at the 1M label.

| Pipeline | Stages, wall milliseconds |
|---|---|
| join-heavy | read_points 3.69; read_polygons 6.19; build_index 0.30; sjoin_query 0.68; assemble_join_rows 0.52; dissolve_groups 2.34; write_output 16.33 |
| relation-semijoin | read_inputs 10.29; build_index 0.21; sjoin_relation 0.61; semijoin_rowset 0.37; subset_rows 0.90; write_output 3.47 |
| small-grouped-constructive-reduce | build_device_grouped_polygons 39.96; native_grouped_union 48.73; native_reference_check 0.03 |
| grouped-capacity-partitions | build_grouped_partition_fixtures 48.58; mixed_strip_exact_union 65.80; positive_degenerate_union 62.30; native_reference_check 0.03 |
| grouped-disjoint-constructive-reduce | build_device_disjoint_groups 65.63; native_grouped_disjoint_subset 1.31; native_reference_check 0.02 |
| grouped-difference-constructive | build_device_grouped_difference_inputs 26.73; native_grouped_difference 9.64; native_reference_check 0.03 |
| constructive-output-native | build_device_pairwise_boxes 3.75; native_constructive_intersection 3.73; constructive_area_expression 1.24; constructive_expression_consumers 1.50; native_reference_check 0.02 |
| overlay-relation-constructive | build_native_overlay_inputs 6.32; build_spatial_index 0.10; candidate_relation 0.73; refine_relation 0.22; constructive_intersection 2.59; native_tabular_projection 0.89; native_reference_check 0.03 |
| constructive | read_points 3.33; clip_points 0.58; buffer_points 1.59; write_output 16.66 |
| predicate-heavy | read_geojson 70.91; load_polygons 6.89; point_in_polygon 0.32; filter_points 0.36; write_output 1.40 |
| zero-transfer | read_input 7.57; predicate_filter 0.40; subset_rows 1.08; write_output 3.41 |
