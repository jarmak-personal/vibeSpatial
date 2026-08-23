# libcuSpatial Quadtree PIP Source Study And vS Benchmark Plan

<!-- DOC_HEADER:START
Scope: Source-study and benchmark plan for testing libcuSpatial-inspired point-space partitioning ideas inside vibeSpatial's public point-region shapes.
Read If: You are studying the archived cuSpatial quadtree PIP algorithm, testing equivalent vS physical shapes, or reviewing SF100 Q10/Q11 point-region evidence.
STOP IF: You only need current production point-region behavior or the consolidated cross-device SF100 results.
Source Of Truth: Benchmark contract for source-derived vibeSpatial experiments; raw vS run artifacts remain authoritative measurements.
Body Budget: 509/520 lines
Document: docs/dev/libcuspatial-quadtree-pip-benchmark-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-15 | Intent |
| 16-26 | Request Signals |
| 27-38 | Open First |
| 39-47 | Verify |
| 48-62 | Risks |
| 63-83 | Pinned Reference |
| 84-104 | Reference Algorithm |
| 105-142 | Source-Derived Decisions |
| 143-176 | Semantic Gap |
| 177-251 | vS Benchmark Lanes |
| 252-264 | Precision Variants |
| 265-295 | Workload Matrix |
| 296-315 | Spatial-Partition Search Space |
| ... | (9 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Use libcuSpatial's archived CUDA quadtree point-in-polygon implementation as a
source-level algorithm reference for vibeSpatial point-region execution. Test
whether its reusable physical ideas improve vS after MultiPolygon semantics,
precision handling, bounded memory, and direct public-result reductions are
included.

The archived implementation is not compiled, packaged, or run. This is a
source study followed by vS-native experiments. It does not adopt cuSpatial as
a runtime dependency and does not create a SpatialBench-specific production
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
- run the vS point-region public correctness and protected-shape commands in
  `docs/dev/evidence-first-point-region-execution-plan.md`
- run the full SF100, public 10K/1M, and pipeline-profile gates before any
  production implementation is admitted

## Risks

- Treating source structure as performance evidence would substitute inference
  for measurement. All performance claims must come from current vS code.
- Comparing a flat Polygon algorithm with vibeSpatial's parent-row
  MultiPolygon contract is not semantically equivalent.
- Raw fp32 PIP is not exact at boundaries; a fast mismatching result is not an
  admissible vS path.
- A hand-selected quadtree depth or leaf size can overfit one GPU or SF100's
  zone distribution.
- libcuSpatial materializes hit pairs, while Q10/Q11 require bounded direct
  reductions. Copying that terminal shape would be a regression.
- A quadtree adds build, traversal, parameter, and skew costs. The source's
  existence is not evidence that it beats vS's current dense point grid.

## Pinned Reference

The local reference checkout is:

```text
repository: ../cuspatial
branch: branch-25.04
commit: 126ef134df17350dd7ac9d700dd35555f575b039
license: Apache-2.0
```

cuSpatial 25.04 was its final release and the repository is archived. The
study therefore treats it as a stable source artifact, not as executable
benchmark software or a dependency whose API will continue evolving.

The relevant source is the templated C++/CUDA implementation in
`cuspatial::quadtree_on_points`,
`cuspatial::join_quadtree_and_bounding_boxes`, and
`cuspatial::quadtree_point_in_polygon`. Its float and double templates inform
precision hypotheses only; they do not become measured comparison columns.

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

## Source-Derived Decisions

| Reference idea | vS status | Decision |
|---|---|---|
| spatially reorder points once | present in dense point-grid and Morton indexes | retain and measure reuse/locality |
| join Polygon boxes to point leaves | present as conservative cell/Morton candidate generation | compare fixed-grid waste with adaptive subdivision |
| enumerate candidates from descriptor offsets | partly present in bounded point-grid tiles | test descriptor consumption without a full pair relation |
| one lane performs complete Polygon crossings | replaced by exact part-y edge pruning | use only as a full-ring floor, not a production default |
| one Polygon per MultiPolygon | incompatible | reject; retain native parent/part semantics |
| allocate output for all candidates, retry after OOM | incompatible | reject; preserve pre-admitted bounded reduction |
| materialize every hit pair | wrong Q10/Q11 terminal shape | reject; reduce to native scalar/grouped/aligned outputs |
| raw float/double template switch | insufficient for exact predicates | test only centered fp32 tri-state plus selective fp64 |

The source has no authoritative performance result for current vS, RTX 4090,
or H200. It supplies algorithm hypotheses and failure modes, not a speed floor.

### Frozen Source Anchors

The pinned line references supporting the audit are:

- quadtree candidate prefix construction and its scalar total read:
  `quadtree_point_in_polygon.cuh:121-166`;
- sorted-point indirection and implicit candidate ordinal mapping:
  `quadtree_point_in_polygon.cuh:168-207`;
- candidate-sized result allocation, compaction, and OOM rerun:
  `quadtree_point_in_polygon.cuh:193-234`;
- single-Polygon-per-MultiPolygon restriction:
  `quadtree_point_in_polygon.cuh:118-119`;
- full-ring crossings-multiply traversal and boundary rejection:
  `is_point_in_polygon.cuh:45-100`;
- level-by-level bbox/leaf traversal, resizing, and final leaf sort:
  `quadtree_bbox_filtering.cuh:49-187`;
- Morton point ordering and adaptive tree construction:
  `point_quadtree.cuh:190-230`.

These anchors are provenance for the hypotheses. They are not runtime or
throughput evidence.

## Semantic Gap

### Geometry

- The header implementation requires one Polygon per MultiPolygon. SF100 zones
  contain true MultiPolygons.
- Any vS experiment must retain ring/hole structure and carry part-to-parent
  identity on device.
- Parent membership is the union of part hits. Valid MultiPolygon interiors
  should not overlap, but the reducer must still deduplicate parent-point hits
  to preserve arbitrary valid input and boundary behavior.
- Nulls, empties, invalid offsets, and non-polygon families must be handled by
  the benchmark contract rather than dropped invisibly.

### Coordinates And Precision

- libcuSpatial requires points and polygon coordinates to have the same `T`.
  Its C++ wrapper uses iterator adapters over separate x/y columns, so an AoS
  interleave is not inherently required.
- vS storage remains fp64. A valid fp32 experiment subtracts an fp64 center
  before casting, runs a tri-state fp32 predicate, and selectively refines
  ambiguous candidates in fp64. Raw fp32 crossings are an arithmetic
  diagnostic, not an exactness claim.
- Spatial-index bounds and scale must remain conservative. A float index may not
  shrink boxes or omit a candidate that the fp64 contract would retain.

### Result Shape

- Any sorted point offset must map back to the original public point row, and
  any part identifier must reduce through its parent geometry.
- Q2 needs one scalar count. Q10 needs per-zone size and weighted sums. Q11
  needs aligned pickup, dropoff, and shared parent-zone counts. None needs the
  stock full hit relation as its terminal physical shape.

## vS Benchmark Lanes

Every result reports all earlier lanes needed to reach it. No lane may be
presented as end-to-end performance by itself.

### V0: Current Public Baseline

Measure current public point-region execution with resident native coordinates:

- public pair, existential, grouped, and aligned reductions;
- dense point-grid preparation, candidate tiles, exact part-y traversal, and
  direct reduction separated where bounded instrumentation allows;
- production fp64 and any already-admitted precision variant;
- relation and transfer counters proving the terminal shape.

This is the only end-to-end baseline. Private floors never replace it.

### V1: Point-Partition Candidate Floor

Benchmark vS-native candidate generation variants over identical resident
points and query bounds:

- current fixed dense point grid;
- current Morton-range path;
- an adaptive hierarchy only after fixed-grid waste is measured;
- build, reuse, query, descriptor/pair bytes, and direct-consumer costs.

The candidate set must be conservative and identical in public semantics, not
identical in internal ordering.

#### Adaptive-quadtree physical contract

The first vS quadtree experiment is a benchmark-gated `NativeSpatialIndex`
candidate-refine shape. It reuses device Morton order, forms variable-depth
prefix leaves until an occupancy target or the 16-bit key depth is reached and
stores exact fp64 point bounds per leaf. Query boxes join to leaves as bounded
`(query, leaf, sorted-point-span)` work. Existing exact Polygon/MultiPolygon
reducers consume each tile immediately; no full-query relation is retained.

Two designs were considered. A breadth-first node frontier permits arbitrary
splits but repeatedly compacts queues. The selected sparse-prefix design uses
one sorted order and a smaller persistent carrier. Query-by-leaf traversal and
maximum-depth collisions can still lose, so automatic dispatch is not admitted.

### V2: Implicit Descriptor Consumption

Test the strongest reusable source idea without copying its output shape:

- group spatially ordered point spans by query/leaf descriptor;
- enumerate candidate ordinals from bounded descriptor offsets;
- refine and reduce directly into native scalar/grouped/aligned carriers;
- compare per-candidate descriptor lookup with pre-expanded bounded tiles;
- include remap, deduplication, scratch, and synchronization.

Relation materialization may be measured as a diagnostic floor but is never an
admissible Q10/Q11 production result.

### V3: Exact-Refinement Shape

Compare exact vS refinement shapes after candidate generation is held fixed:

- current one-lane-per-candidate part-y traversal;
- full-ring crossings as a diagnostic lower-complexity control;
- the one evidence-selected skew alternative from the active point-region
  plan;
- centered fp32 tri-state plus fp64 refinement only in a separate precision
  experiment.

### V4: Public Validation

Compare any admitted alternative through `SpatialIndex.query_aggregate` and
`SpatialIndex.query_pair_aggregate` with the same public GeoParquet inputs,
partitions, memory ceiling, and result semantics used by SF100. No benchmark
helper may invoke a private executor and claim public acceleration.

## Precision Variants

| Variant | Storage | Compute | Correctness role |
|---|---|---|---|
| VS64 | fp64 | current prepared fp64 | authoritative public baseline |
| X32-native | fp32 fixture | raw fp32 crossings | diagnostic arithmetic floor only |
| X32-centered | fp64 source, centered cast | fp32 tri-state | measures conditioning and ambiguity |
| X32-refine | fp64 source | centered tri-state fp32 plus selective fp64 | only potentially admissible staged shape |

The benchmark must report X32 mismatch and ambiguous-refinement rates. It may
not compare fp32 performance with VS64 without putting those rates beside the
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

## Spatial-Partition Search Space

Do not copy one remembered cuSpatial tuning. First sweep existing vS controls
and observed shapes:

- fixed-grid target occupancy and size caps;
- Morton span and dense-grid candidate inflation;
- point-tree reuse count: 1, 2, 5, and the full partition consumer count;
- Polygon partitioning: current SF100 partitions and bounded alternative
  partitions based on flat parts/edges rather than rows.

Only if V1 shows material fixed-grid waste should an adaptive hierarchy sweep
depth 6, 8, 10, 12, and 15 with target leaf occupancies 32, 64, 125, 256, and
512. Bounds remain conservative and parameter selection uses observed shape,
never a GPU product name.

The selected configuration is a function of observed leaf occupancy,
polygon-leaf pairs, candidate points, edges, output bytes, build cost, reuse,
and memory. GPU model names are not planner inputs.

## Required Measurements

### Shape counters

- input points, parent geometries, flat Polygons, rings, and edges;
- occupied cells or hierarchy leaves, depth distribution, and occupancy
  percentiles;
- Polygon-cell/leaf descriptors and points represented by those descriptors;
- candidate point-Polygon pairs and parent-deduplicated candidates;
- full-ring edge visits, orientation/crossing tests, and boundary tests;
- true hits, flat-part duplicate hits, and final output rows;
- ambiguous fp32 candidates and fp64 refinements.

### Time and memory

- every V1-V4 stage plus remap, dedup, reduction, and public result;
- warm and cold index build, including reusable index lifetime;
- allocated, peak live, pool reserved, zeroed, copied, and materialized bytes;
- H2D, D2H, D2D, synchronization, and allocation fences;
- materialized candidate relation bytes versus descriptor/direct-reducer
  scratch;
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

- Compare VS64 and all potentially admissible variants with current vS fp64 and
  a mechanical Shapely oracle on bounded samples.
- Validate full result fingerprints for Q2/Q4/Q10/Q11 before using SF100 timing.
- Preserve `contains` semantics: boundary points are false, hole points are
  false, and null/empty inputs do not match.
- Preserve parent MultiPolygon identity and multiplicity after flattening.
- Test both ring closure forms and winding directions.
- An X32 mismatch is measured evidence for refinement, never an accepted error
  tolerance for a boolean predicate.

## Hypotheses And Decision Rules

1. **Adaptive point partitioning may reduce fixed-grid candidate inflation.**
   Confirm only if the complete V2/V4 pipeline reduces work and public wall on
   skewed shapes.
2. **Centered fp32 may expose a low consumer-GPU floor.** Treat it only as
   motivation until X32-refine preserves exactness and retains most of the gain.
3. **MultiPolygon parent work may dominate point partitioning.** Reject a new
   index if considered parts and edge visits remain unchanged.
4. **Implicit descriptors may beat candidate relations.** Require a direct
   reducer and include ordinal lookup, remap, and bounded scratch in the result.
5. **Build amortization determines admission.** A quadtree path must win after
   build at the observed reuse count; a kernel-only win is insufficient.

Production work is admitted only if the same general shape wins on both GPUs,
preserves protected shapes, and improves a public operator floor after all
adapter and reducer costs. The final implementation may borrow the algorithmic
ideas under Apache-2.0, but it must use vS native carriers, memory admission,
precision policy, and public APIs.

## First vS Canary

The first source-derived canary isolates point-partition skew rather than PIP
arithmetic. At 1M points, almost all points occupy `[0, 1] x [0, 1]`, four
finite outliers stretch the global extent, and 64 disjoint query boxes cover
the dense core.

At extent 1024 on the RTX 4090, current public execution produced:

| Metric | Result |
|---|---:|
| endpoint oracle hits | 1,479,200 |
| dense-grid exact candidate lanes | 134,217,472 |
| dense-grid candidates per endpoint hit | 90.74 |
| dense-grid exact-kernel time | 91.24 ms |
| dense-grid warm public wall | 203.37 ms |
| forced-Morton exact candidate lanes | 2,218,800 |
| forced-Morton exact-kernel time | 2.00 ms |
| forced-Morton warm public wall | 13.44 ms |
| public speedup | 15.1x |

The protected 1M corpus rejects a global Morton default: forced Morton was
111x slower for a simple Polygon, 14x slower for the long-bin Polygon, 22x
slower for multipart skew, and 344x slower for uniform small Polygons.

The extent sweep also rejects max-cell occupancy as a selector. Morton range
span work is quantized and non-monotonic: at extents 80, 112, and 160, it was
22x, 34x, and 4.6x slower than the dense grid despite producing the same 2.22M
exact candidates as the winning extent-256 case. The diagnostic discriminator
must include both dense-grid candidate work and pre-refinement Morton span
work. A conservative rule is not admitted until the same sweep runs on H200.

The audit also found that `NativeSpatialIndex.to_flat_index()` reconstructed a
transitional carrier and lost its prepared point-grid cache. The implementation
now retains the originating `FlatSpatialIndex`, preserving device index state
across repeated native public reductions. This is a Native* completeness fix,
not evidence for either candidate strategy.

## First Adaptive Quadtree Evidence

The source-derived quadtree is now measured separately from the earlier
Morton-range proxy. On the RTX 4090 1M extent-1024 canary, its first public call
was 42.2 ms and warm calls were 1.86-2.04 ms. Current warm dense-grid and
Morton floors are about 191.5 ms and 13.4 ms. The paired aggregate ran 3.00M
exact classifications, matched both endpoint oracles, and used 8.05 MB of
persistent quadtree state without retaining a full relation.

With leaf target 64, maximum depths 10, 12, 14, and 16 took 160.3, 19.5, 2.23,
and 1.92 ms warm. At depth 16 the skew fixture has 1,028 leaves and maximum
occupancy 1,024 because its 1M points exceed the 16-bit Morton resolution.
Leaf targets 64-512 consequently produce the same skew hierarchy.

| Protected 1M shape | auto | quadtree-256 |
|---|---:|---:|
| simple short Polygon | 3.52 ms | 1.92 ms |
| long selected bin | 590.92 ms | 589.42 ms |
| multipart envelope skew | 60.48 ms | 56.63 ms |
| uniform small Polygons | 1.99 ms | 2.82 ms |
| clustered extent skew | 191.54 ms | 1.87 ms |

A leaf target of 64 reduces the uniform case to about 2.20 ms but misses the
protected 5% rail. These synthetic results justified a fuller physical-shape
experiment; they did not justify production selection.

## Final Production Decision (2026-08-21)

The complete SF100 experiment is preserved under the ignored
`benchmark_results/experiments/2026-08-21-q11-q12-physical-shapes/` capsule.
Forced dense grid completed Q11 in 241.79 seconds; the production adaptive
leaf-all-pairs implementation took 478.21 seconds. Replacing the universal leaf
scan with a true hierarchy fixed that diagnosed traversal defect, but did not
create a winning warmed Q11 region: grid took 7.7366 seconds for the measured
late-batch slice, while the best hierarchy setting (leaf capacity 128) took
8.9879 seconds, 16.17% slower.

The production provider and selector are therefore archived. The durable work
is the shared NativeSpatialIndex ownership/readiness substrate, complete-stage
memory admission, sealed query slices, guarded scatters, repaired dense-grid
construction, and grid-to-Morton selection. A future quadtree proposal must
start from the true hierarchy, demonstrate a public end-to-end win on more than
one relevant shape/device, and identify a selection region before re-entering
production code.

## Milestones

1. Preserve a line-cited source audit of the archived algorithm and map each
   idea to retain, reject, already-present, or experiment.
2. Extend deterministic vS shape generators with candidate-inflation and
   locality controls.
3. Export or generate Q2/Q4/Q10/Q11 physical fixtures with parent maps and
   authoritative fingerprints.
4. Add cell/leaf occupancy, descriptor, candidate-relation, remap, and direct
   reduction telemetry.
5. Compare fixed-grid, Morton, and adaptive-quadtree vS floors; the measured
   hierarchy remains benchmark-gated pending cross-device evidence.
6. Add centered tri-state fp32 plus selective fp64 refinement experiments.
7. Capture 4090 and H200 Systems/Compute evidence and decide whether a
   production change is justified.

## Evidence Outputs

Write immutable artifacts under:

```text
benchmark_results/point_region/source_derived_partitioning/<date>-<device>/
```

Each run includes vS source commit, CUDA/RAPIDS versions, device facts, fixture
fingerprint, partition parameters, stage JSON, correctness JSON, memory JSON,
and profiler paths. The archived cuSpatial commit appears only as provenance
for the source-study hypotheses; no column may imply it was executed.

The first canary artifacts are:

- `2026-08-20-rtx4090-v0-profiled-16k.json`;
- `2026-08-20-rtx4090-v0-profiled-128k.json`;
- `2026-08-20-rtx4090-v0-retained-grid-profiled-all-1m.json`;
- `2026-08-20-rtx4090-v1-<mode>-extent-<extent>-1m.json`;
- matching `*-profiled-1m.json` files for the crossover discontinuities.

## External References

- [cuSpatial 25.04 spatial API](https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/)
- [libcuSpatial spatial-join API](https://docs.rapids.ai/api/libcuspatial/stable/group__spatial__join)
- [Archived cuSpatial repository](https://github.com/rapidsai/cuspatial)
