# libcuSpatial Quadtree PIP Benchmark Plan

<!-- DOC_HEADER:START
Scope: Benchmark plan for comparing pinned libcuSpatial quadtree point-in-polygon CUDA execution with vibeSpatial's public point-region shapes.
Read If: You are building, running, interpreting, or reviewing the cuSpatial quadtree PIP comparison, precision variants, or SF100 Q10/Q11 point-region evidence.
STOP IF: You only need current production point-region behavior or the consolidated cross-device SF100 results.
Source Of Truth: Benchmark contract for the external libcuSpatial CUDA algorithm study; raw run artifacts remain authoritative measurements.
Body Budget: 363/390 lines
Document: docs/dev/libcuspatial-quadtree-pip-benchmark-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-14 | Intent |
| 15-25 | Request Signals |
| 26-37 | Open First |
| 38-46 | Verify |
| 47-61 | Risks |
| 62-83 | Pinned Reference |
| 84-104 | Reference Algorithm |
| 105-142 | Compatibility Gap |
| 143-203 | Benchmark Lanes |
| 204-217 | Precision Variants |
| 218-248 | Workload Matrix |
| 249-264 | Quadtree Search Space |
| 265-299 | Required Measurements |
| ... | (5 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Use libcuSpatial's CUDA quadtree point-in-polygon implementation as an external
algorithmic floor for vibeSpatial point-region execution. Measure whether its
specialized point-quadtree shape remains advantageous after MultiPolygon
normalization, precision handling, parent-row restoration, and the bounded
reductions required by public vibeSpatial APIs.

This is a benchmark and algorithm study. It does not adopt cuSpatial as a
runtime dependency and does not create a SpatialBench-specific production
path.

## Request Signals

- cuSpatial quadtree PIP
- libcuspatial benchmark
- point-in-polygon floor
- SF100 Q10
- SF100 Q11
- float point-in-polygon
- quadtree spatial join
- point-region optimization

## Open First

- `docs/dev/sf100-cross-device-optimization-audit.md`
- `docs/dev/evidence-first-point-region-execution-plan.md`
- `docs/dev/point-region-execution-evidence.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `docs/decisions/0002-dual-precision-dispatch.md`
- `src/vibespatial/predicates/point_location_index.py`
- `src/vibespatial/predicates/point_location_index_kernels.py`
- `../cuspatial/cpp/include/cuspatial/detail/join/quadtree_point_in_polygon.cuh`
- `../cuspatial/cpp/include/cuspatial/detail/algorithm/is_point_in_polygon.cuh`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "libcuspatial quadtree PIP benchmark"`
- run the point-region public correctness and protected-shape commands in
  `docs/dev/evidence-first-point-region-execution-plan.md`
- run the full SF100, public 10K/1M, and pipeline-profile gates before any
  production implementation is admitted

## Risks

- Comparing only exact-kernel time hides quadtree build, bbox join, coordinate
  conversion, MultiPolygon restoration, output materialization, and reduction.
- Comparing cuSpatial's flat Polygon contract directly with vibeSpatial's
  parent-row MultiPolygon contract is not semantically equivalent.
- Raw fp32 PIP is not exact at boundaries; a fast mismatching result is not an
  admissible vS path.
- A hand-selected quadtree depth or leaf size can overfit one GPU or SF100's
  zone distribution.
- Stock libcuSpatial materializes hit pairs, while Q10/Q11 require bounded
  reductions. That output cost must be reported, not silently omitted.
- The archived branch uses RAPIDS 25.04 while vibeSpatial uses newer RAPIDS
  components. Environment and source differences must remain explicit.

## Pinned Reference

The local reference checkout is:

```text
repository: ../cuspatial
branch: branch-25.04
commit: 126ef134df17350dd7ac9d700dd35555f575b039
license: Apache-2.0
```

cuSpatial 25.04 was its final release and the repository is archived. The
study therefore treats it as a stable reference algorithm, not a dependency
whose API or compatibility will continue evolving.

The public Python API is not the target. The target is the templated C++/CUDA
implementation in `cuspatial::quadtree_on_points`,
`cuspatial::join_quadtree_and_bounding_boxes`, and
`cuspatial::quadtree_point_in_polygon`. The C++ tests instantiate both `float`
and `double`, even though ordinary Python geometry construction tends to expose
the fp64 path.

## Reference Algorithm

The pinned source performs these stages:

1. Convert point coordinates to Morton keys and build a point quadtree with a
   configurable depth and maximum leaf occupancy.
2. Compute Polygon bounding boxes and join each box to intersecting quadtree
   leaves, producing `(polygon, quadrant)` pairs.
3. Prefix-sum the point counts of those quadrants. This defines an implicit
   point-polygon candidate space without first materializing every candidate.
4. For each candidate ordinal, binary-search the prefix offsets to recover its
   polygon/quadrant descriptor, recover its point, and run a crossings-multiply
   PIP test over every ring edge of that Polygon.
5. Use Thrust `copy_if` to materialize `(polygon_index, point_index)` hits.

The implementation chunks candidate iteration at `INT32_MAX`, but it first
tries to allocate output capacity for every candidate. On RMM OOM it reruns the
exact predicate with `count_if`, allocates exact hit capacity, and runs
`copy_if` again. That recovery shape is useful reference behavior but is not
admissible as a vS production memory strategy.

## Compatibility Gap

### Geometry

- The header implementation requires one Polygon per MultiPolygon. SF100 zones
  contain true MultiPolygons.
- The benchmark adapter must flatten Polygon parts, retain ring/hole structure,
  and carry `flat_part -> parent_zone` on device.
- Parent membership is the union of part hits. Valid MultiPolygon interiors
  should not overlap, but the reducer must still deduplicate parent-point hits
  to preserve arbitrary valid input and boundary behavior.
- Nulls, empties, invalid offsets, and non-polygon families must be handled by
  the benchmark contract rather than dropped invisibly.

### Coordinates And Precision

- libcuSpatial requires points and polygon coordinates to have the same `T`.
  Its C++ wrapper uses iterator adapters over separate x/y columns, so an AoS
  interleave is not inherently required.
- vS storage remains fp64. A native `float` cuSpatial run therefore includes an
  explicit conversion boundary unless a transform iterator can center and cast
  without materialization.
- A valid vS-inspired fp32 experiment subtracts an fp64 center before casting,
  runs a tri-state fp32 predicate, and selectively refines ambiguous candidates
  in fp64. Native cuSpatial `float` is a performance floor, not an exactness
  claim.
- Quadtree bounds and scale must remain conservative. A float index may not
  shrink boxes or omit a candidate that the fp64 contract would retain.

### Result Shape

- The returned point offset indexes the quadtree's sorted point-index array and
  must be mapped again to the original public point row.
- The returned Polygon row must be mapped through the flattened-part parent.
- Q2 needs one scalar count. Q10 needs per-zone size and weighted sums. Q11
  needs aligned pickup, dropoff, and shared parent-zone counts. None needs the
  stock full hit relation as its terminal physical shape.

## Benchmark Lanes

Every result reports all earlier lanes needed to reach it. No lane may be
presented as end-to-end performance by itself.

### L0: Predicate Floor

Run the exact pinned crossings predicate over precomputed candidate descriptors
and resident coordinates:

- `float` and `double` template instantiations;
- cuSpatial's native coordinate layout and a zero-copy iterator over vS SoA;
- no quadtree build, bbox join, result remap, or public construction;
- predicate-only boolean/count output alongside stock `copy_if` output.

This answers whether the edge traversal itself has a compelling throughput
floor. It does not answer whether the quadtree algorithm wins.

### L1: Stock Quadtree Pipeline

Measure the unmodified pinned stages independently and together:

- quadtree build;
- Polygon bbox construction;
- bbox-to-leaf join;
- prefix-offset construction;
- PIP plus stock hit-pair materialization;
- hit-pair shrink/copy and peak allocation.

Use admitted shards when stock candidate-capacity allocation cannot fit. Record
that as a stock-shape limit rather than triggering and timing OOM recovery.

### L2: vS-Shaped Adapter

Add the unavoidable work needed to consume vS buffers:

- flatten and parent mapping;
- point and polygon coordinate iterator or conversion;
- centered-fp32 metadata where selected;
- sorted-point and parent-row restoration;
- deduplication required by parent semantics;
- direct scalar/grouped/aligned reduction.

Run both relation materialization and direct reduction. The latter is a
benchmark-only consumer of the same cuSpatial candidate/PIP algorithm, clearly
labelled as modified.

### L3: Public vS Comparison

Compare against existing public APIs only:

- `SpatialIndex.query_aggregate` for Q2/Q10-shaped count and weighted sums;
- `SpatialIndex.query_pair_aggregate` for Q11's aligned endpoints;
- the same public GeoParquet inputs, partitions, memory ceiling, and result
  semantics used by SF100;
- both current prepared part-Y execution and any later admitted vS quadtree
  implementation.

No benchmark helper may invoke private vS executors to produce the public vS
timing.

## Precision Variants

| Variant | Storage | Compute | Correctness role |
|---|---|---|---|
| C64 | fp64 | cuSpatial fp64 | exact external reference candidate |
| C32-native | fp32 | cuSpatial fp32 | raw CUDA throughput floor only |
| C32-centered | fp64 source, centered cast | cuSpatial-like fp32 | measures coordinate-conditioning value |
| C32-refine | fp64 source | centered tri-state fp32 plus selective fp64 | only potentially admissible vS precision shape |
| VS64 | fp64 | current prepared fp64 | current public production comparison |

The benchmark must report C32 mismatch and ambiguous-refinement rates. It may
not compare C32 performance with VS64 without putting those rates beside the
timing.

## Workload Matrix

### SF100-derived shapes

| Shape | Why it matters | Required consumer |
|---|---|---|
| Q2 point batch versus one Polygon | best-case quadtree reuse and scalar result | direct count |
| Q4 1,000 points versus many zone geometries | tests whether build cost amortizes | grouped count |
| Q10 4M points versus each zone partition | primary many-point/many-region workload | size plus two weighted sums |
| Q11 pickup/dropoff versus each zone partition | aligned dual-index and shared-membership workload | three row-aligned counts |

### Protected synthetic shapes

- many points versus a simple short Polygon;
- one long ring whose bbox overlaps many leaves;
- many small Polygon parts under one parent MultiPolygon;
- a highly skewed MultiPolygon envelope;
- holes, nested rings, touching parts, and shared boundaries;
- points exactly on and within one/two ULPs of an edge or vertex;
- coordinates near zero and large translated coordinates with small local
  deltas;
- sparse, moderate, and dense hit rates;
- candidate work distributions with low and high coefficient of variation.

### Scale

Use small correctness runs followed by resident 16K, 1M, 4M, and 8M point
floors. Full Q10/Q11 runs retain their existing 4M public batches. A larger
point batch is admitted only by the common memory budget, never because H200 is
named explicitly.

## Quadtree Search Space

Do not select one remembered cuSpatial tuning. Sweep:

- `max_depth`: 6, 8, 10, 12, 15, subject to coordinate scale validity;
- target/max leaf occupancy: 32, 64, 125, 256, 512;
- scale derived conservatively from extent and depth, plus admitted nearby
  values that cannot exclude candidates;
- point-tree reuse count: 1, 2, 5, and the full partition consumer count;
- Polygon partitioning: current SF100 partitions and bounded alternative
  partitions based on flat parts/edges rather than rows.

The selected configuration is a function of observed leaf occupancy,
polygon-leaf pairs, candidate points, edges, output bytes, build cost, reuse,
and memory. GPU model names are not planner inputs.

## Required Measurements

### Shape counters

- input points, parent geometries, flat Polygons, rings, and edges;
- quadtree nodes/leaves, depth distribution, and leaf occupancy percentiles;
- Polygon-leaf pairs and points represented by those pairs;
- candidate point-Polygon pairs and parent-deduplicated candidates;
- full-ring edge visits, orientation/crossing tests, and boundary tests;
- true hits, flat-part duplicate hits, and final output rows;
- ambiguous fp32 candidates and fp64 refinements.

### Time and memory

- every L1 stage plus adapter, remap, dedup, reduction, and public result;
- warm and cold quadtree build, including reusable index lifetime;
- allocated, peak live, pool reserved, zeroed, copied, and materialized bytes;
- H2D, D2H, D2D, synchronization, and allocation fences;
- stock maximum candidate output request versus direct-reducer scratch;
- candidates, edges, and final results per second.

### Hardware efficiency

Targeted Nsight Compute captures on 4090 and H200 must report:

- achieved occupancy, eligible warps, issue rate, and warp stall reasons;
- branch efficiency and loop-tail divergence;
- registers, spills/local memory, and active blocks per SM;
- L1/L2 hit rate, global-load efficiency, DRAM bytes, and bandwidth;
- integer/binary-search cost versus crossing arithmetic;
- fp32/fp64 instruction mix and useful results per edge/byte.

Use Nsight Systems for stage chronology and gaps. Do not replay a full Q10/Q11
through Nsight Compute.

## Correctness Contract

- Compare C64 and all potentially admissible variants with current vS fp64 and
  a mechanical Shapely oracle on bounded samples.
- Validate full result fingerprints for Q2/Q4/Q10/Q11 before using SF100 timing.
- Preserve `contains` semantics: boundary points are false, hole points are
  false, and null/empty inputs do not match.
- Preserve parent MultiPolygon identity and multiplicity after flattening.
- Test both ring closure forms and winding directions.
- A C32 mismatch is measured evidence for refinement, never an accepted error
  tolerance for a boolean predicate.

## Hypotheses And Decision Rules

1. **Quadtree pruning may dominate part-Y edge pruning.** Confirm only if the
   complete L2 pipeline reduces candidate-edge work and wall on Q10/Q11.
2. **Native fp32 may expose a very low consumer-GPU floor.** Treat it only as
   motivation until C32-refine preserves exactness and retains most of the gain.
3. **MultiPolygon restoration may erase the stock advantage.** Reject the
   shape if flattening, duplicate work, or parent reduction dominates.
4. **Relation materialization may erase the stock advantage.** A direct reducer
   is required for production relevance, but stock and modified results remain
   separate in every table.
5. **Build amortization determines admission.** A quadtree path must win after
   build at the observed reuse count; a kernel-only win is insufficient.

Production work is admitted only if the same general shape wins on both GPUs,
preserves protected shapes, and improves a public operator floor after all
adapter and reducer costs. The final implementation may borrow the algorithmic
ideas under Apache-2.0, but it must use vS native carriers, memory admission,
precision policy, and public APIs.

## Milestones

1. Build a standalone CUDA/NVBench target in vibeSpatial that includes the
   pinned sibling libcuSpatial headers without modifying `../cuspatial`.
2. Add deterministic synthetic generators and C64/C32 stock stage timings.
3. Export or generate Q2/Q4/Q10/Q11 physical fixtures with flat-parent maps and
   authoritative fingerprints.
4. Add shape counters, stock relation memory telemetry, and zero-copy SoA
   iterator comparison.
5. Add benchmark-only direct count/group/aligned consumers.
6. Add centered tri-state fp32 plus selective fp64 refinement experiments.
7. Capture 4090 and H200 Systems/Compute evidence and decide whether a vS
   production implementation is justified.

## Evidence Outputs

Write immutable artifacts under:

```text
benchmark_results/point_region/cuspatial_quadtree/<date>-<device>/
```

Each run includes source commit, compiler/CUDA/RAPIDS versions, device facts,
fixture fingerprint, quadtree parameters, stage JSON, correctness JSON, memory
JSON, and profiler paths. A human-readable checkpoint must distinguish stock
cuSpatial, benchmark-modified cuSpatial, and public vibeSpatial columns.

## External References

- [cuSpatial 25.04 spatial API](https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/)
- [libcuSpatial spatial-join API](https://docs.rapids.ai/api/libcuspatial/stable/group__spatial__join)
- [Archived cuSpatial repository](https://github.com/rapidsai/cuspatial)
