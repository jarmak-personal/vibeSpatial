# Nearest Index Refinement Ledger

<!-- DOC_HEADER:START
Scope: Nearest shape tuning, conservative precision, virtual bounds and independent distribution evidence.
Read If: You are checking nearest precision, build/query tradeoffs or generalization beyond MA OSM.
STOP IF: You only need the original public performance gap or the initial algorithm comparison.
Source Of Truth: Late-stage nearest experiment evidence and unresolved graduation conditions.
Body Budget: 232/260 lines
Document: docs/dev/nearest-index-refinement-ledger.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-15 | Request Signals |
| 16-25 | Open First |
| 26-30 | Verify |
| 31-38 | Risks |
| 39-53 | 2026-09-10 / E9 / Shape Parameter Screening |
| 54-69 | 2026-09-10 / E10 / Fresh Whole-Road Comparator |
| 70-90 | 2026-09-10 / E11 / Virtual Bounds, Original Geometry |
| 91-104 | 2026-09-10 / E12 / Independent Distributions |
| 105-125 | 2026-09-10 / E13 / Conservative FP32 Coarse Search |
| 126-153 | 2026-09-10 / E14 / Confirmed Build and Reuse Tradeoff |
| 154-194 | 2026-09-10 / E15 / Independent Confirmation and Overlap Exception |
| 195-232 | 2026-09-10 / E16 / Verification and Profile Review |
DOC_HEADER:END -->

## Intent

Continue the [strategy ledger](nearest-index-strategy-ledger.md) with leaf-size,
precision, adversarial-shape and independent-distribution experiments. Identify
the measured build/query/memory tradeoff before writing the recommendation.

## Request Signals

- nearest precision experiments
- nearest index experiments
- conservative FP32 bounds
- virtual segment bounds

## Open First

- docs/dev/nearest-index-recommendation.md
- docs/testing/nearest-index-strategy-evidence.json
- scripts/experiment_nearest_strategies.py
- scripts/nearest_strategy_kernels.py
- scripts/experiment_nearest_distributions.py
- docs/dev/nearest-index-strategy-ledger.md
- tests/test_experimental_nearest_hierarchy.py

## Verify

- `uv run pytest tests/test_experimental_nearest_hierarchy.py tests/test_benchmark_osm_nearest.py --run-gpu -q`
- `uv run python scripts/check_docs.py --check`

## Risks

- The fastest repeated query can require a larger, slower-building index.
- Virtual bounds cover only part of the geometry used in exact refinement.
- Coarse FP32 must enclose both stored boxes and query coordinates.
- Dense synthetic overlap can make CPU oracle generation expensive.
- Screening results need confirmation before setting a default.

## 2026-09-10 / E9 / Shape Parameter Screening

Same MA segment fixture, 1M queries, one warm trial, all four outputs checked.
Recursive STR with eight-way branching remains preferable to 2/4-way branching
in the tested seeded variants. At leaf width 8, seed resolutions 1024/2048/4096
are close (roughly 34 ms); 512 is slower (40 ms). Seed table sizes grow
quadratically, so a larger table needs a measured benefit.

With an eight-way tree and 2048 seed resolution, leaf widths 1/2/4/8/16/32 give
repeated-query times of 21.2/24.1/28.1/34.4/43.3/60.1 ms. Build times for
widths 1/2/4 are 28.8/24.3/21.3 ms: build + first query is close across them.
The smaller leaves require more nodes and memory. Do not identify query-only
winning parameters with an unconditional public default. E9 and E11 raw
snapshots are in `/tmp/vibespatial-osm/nearest-experiments/`.

## 2026-09-10 / E10 / Fresh Whole-Road Comparator

Fresh Shapely measurements use the original immutable benchmark script,
the same full road WKB fixture, and verified prefixes of the full-state oracle.
One first-process trial then three fresh-input warm trials, each with first
and repeated queries. Both scales pass every output check.

| Queries | WKB ingress ms | Build ms | First query ms | Repeated ms | Ingress + build + first ms |
|---|---:|---:|---:|---:|---:|
| 100K | 188.2 | 153.5 | 1792.6 | 1789.4 | 2125.9 |
| 1M | 325.7 | 154.0 | 17873.8 | 17815.9 | 18356.7 |

This refresh supplies the previously absent 1M whole-road timing and matched
three-trial contract. The separate E3 segment comparator remains unchanged.
Raw artifacts: `nearest-experiments/e10-shapely-roads-*.json`.

## 2026-09-10 / E11 / Virtual Bounds, Original Geometry

Exploratory alternative: split long segment parameter intervals into smaller
index boxes, but always refine against the original segment coordinates.
Directed FP64 subtraction/division/multiplication/addition enclose each
parameter interval. This avoids changing distance arithmetic through geometric
subdivision and retains original segment/road IDs.

An initial two-pass implementation failed the adversarial oracle: the first
pass could count references whose virtual boxes were later pruned during
emission. This left unfilled output slots. Root cause: a virtual box bounds
only part of the geometry used for distance refinement. Fixed the experiment
with minimum, fixed-minimum count, then emission passes, followed by logical
ID deduplication. All 35 tests then passed, including translated crossing-line
and duplicate-geometry canaries with exactly equal distances.

On MA, max spans 64/256/1024 m take about 38/36/36 ms at 1M, versus 28 ms for
the same leaf-width-4 hierarchy without subdivision. Extra construction,
traversal and deduplication outweigh pruning improvements. Reject as the MA
default; retain as an experiment for very long, overlapping primitives.

## 2026-09-10 / E12 / Independent Distributions

Independent uniform fixture: 1M segments and 1M queries, seed 90217. Three
warm trials, all results passing. Repeated queries: Shapely 6984.7 ms,
STR leaf 4 17.3 ms, STR leaf 1 13.4 ms, direct grid 18.7 ms. The hierarchy
advantage is not specific to the OSM extract.

The 1M-query dense-cluster worker hit its 240-second timeout before the first
CPU query completed. No CPU query duration or GPU result exists for that
worker. Its geometry has 1M random 10 m segments in just 16 clusters of 50 m
standard deviation, deliberately creating heavy overlap. A 100K prefix on the
same generated fixture provides the E15 bounded stress comparison; do not call
it a completed 1M result. Preserve the failed artifact and worker trace.

## 2026-09-10 / E13 / Conservative FP32 Coarse Search

Add a COARSE FP32 PrecisionPlan while retaining FP64 geometry storage and
metric refinement. Bounds are built in FP64, then converted with directed
rounding: minima down and maxima up. Query x/y values become FP32 intervals
via directed conversion. Coordinate gaps are subtracted downward; nonnegative
squared terms and their sum are also rounded downward.

For finite ordinary inputs this coarse value cannot exceed the exact distance
to the original box: the box is expanded, the point is enclosed, and every
nonnegative arithmetic operation rounds downward. This is a conservative
filter, not approximate output. The original FP64 metric/tie test is unchanged.
The reference metric-error guard still needs production robustness analysis;
this coarse-filter argument does not prove every GEOS corner case.

A dedicated conversion kernel avoids extra array passes and unnecessary
additional ULP expansion. All 57 tests pass across FP64/FP32 bounds, packing,
fanout, seeds, virtual references, duplicates, far queries and large offsets.
One-warm-trial screening reduced 1M repeated query time to 12.3 ms at leaf
width 1, versus 21.2 ms with FP64 boxes. The final repeat matrix follows.

## 2026-09-10 / E14 / Confirmed Build and Reuse Tradeoff

Twelve independent sequential workers cover segments/whole roads, 100K/1M
queries and leaf widths 1/2/4. Every worker uses eight-way recursive STR,
Morton query order, a 2048-square seed table, and outward FP32 bounds. One
first-process trial then three fresh-input warm trials; all eight outputs
per worker pass, with exactly equal observed distances.

| Input | Queries | Leaf width | Build ms | First ms | Repeated ms | Index MB |
|---|---:|---:|---:|---:|---:|---:|
| Roads | 100K | 1 | 22.71 | 3.48 | 3.44 | 530.5 |
| Roads | 100K | 2 | 17.97 | 3.94 | 3.98 | 430.8 |
| Roads | 100K | 4 | 15.77 | 4.74 | 4.86 | 380.9 |
| Roads | 1M | 1 | 22.72 | 11.93 | 12.59 | 530.5 |
| Roads | 1M | 2 | 17.97 | 14.89 | 15.84 | 430.8 |
| Roads | 1M | 4 | 15.78 | 19.09 | 20.17 | 380.9 |
| Segments | 100K | 1 | 29.31 | 3.11 | 3.18 | 530.5 |
| Segments | 100K | 2 | 24.54 | 3.53 | 3.59 | 430.8 |
| Segments | 100K | 4 | 22.40 | 4.66 | 4.69 | 380.9 |
| Segments | 1M | 1 | 29.36 | 10.51 | 12.67 | 530.5 |
| Segments | 1M | 2 | 24.59 | 13.73 | 15.93 | 430.8 |
| Segments | 1M | 4 | 22.44 | 18.09 | 20.05 | 380.9 |

Index MB are decimal retained bytes, not peak allocation. The width-2 candidate
has the lowest measured build + first query at 1M; width 1 improves reuse;
width 4 saves memory and construction time. See the recommendation for totals
and policy. Every raw worker has source snapshots and an environment packet.

## 2026-09-10 / E15 / Independent Confirmation and Overlap Exception

Refreshed the uniform comparator once because the original E12 synthetic
artifact lacked an explicit environment packet. The new harness records host,
Python, packages, GEOS, lock, fixture hash, source hashes and source snapshots.
It checkpoints each stage, so future timeouts can identify the active stage.
All backends run sequentially in one worker, each with a separate first trial;
this is not process-cold timing for every backend. The MA workers remain
separate processes. Uniform has three warm trials; stress cases have one.

| Fixture | Index segments | Queries | Shapely repeated ms | STR leaf 4 FP64 ms | STR leaf 1 FP64 ms | STR leaf 4 FP32 ms | STR leaf 1 FP32 ms | Direct grid ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 1M | 1M | 7076.6 | 17.11 | 13.73 | 11.92 | 7.86 | 18.84 |
| Dense clusters | 1M | 100K | 26606.9 | 52.32 | 33.74 | 40.21 | 18.11 | 40.74 |

All outputs pass with zero observed distance error. Clustered uses the exact
1M-query generator invocation from E12, then takes the first 100K queries;
`--fixture-rows 1000000` prevents query count from changing the random tree.
The 100K result does not resolve the timed-out 1M CPU measurement.

Long crossing segments expose a different workload shape: 4096 segments,
5000 queries, one warm trial. Shapely takes 2155.7 ms; leaf-4 FP64 STR takes
141.0 ms; the same STR with virtual 25 m boxes takes 13.8 ms, with exact
original segment IDs and distances. This controlled subdivision comparison
uses FP64 boxes for both variants; it is not a comparison against the leading
FP32 MA configuration. The 256-square direct grid declines admission because
it would need 67,900,844 replicated entries, above the experimental 64M cap.

Decision: retain virtual bounds as an overlap-specific research path. It is
slower on MA, useful on this deliberately pathological shape, and needs a
measured admission policy before any production selection. The raw failed
E12 worker, successful E15 cases and grid admission reason remain preserved.

Reproduce independently:

```bash
uv run python scripts/experiment_nearest_distributions.py --distribution uniform --output /tmp/nearest-uniform-new.json
uv run python scripts/experiment_nearest_distributions.py --distribution clustered --rows 100000 --fixture-rows 1000000 --repeat 1 --output /tmp/nearest-clustered-new.json
uv run python scripts/experiment_nearest_distributions.py --distribution long --tree-rows 4096 --rows 5000 --repeat 1 --output /tmp/nearest-long-new.json
```

## 2026-09-10 / E16 / Verification and Profile Review

Final targeted run: 57 passed in 5.42 s. CUDA memcheck and initcheck each run
all 53 GPU experiment canaries and report zero errors. Full-repository ruff,
documentation refresh/check, intake routing and diff whitespace checks pass.
The original public benchmark and the user's existing `.gitignore` edits
retain their pre-experiment hashes. No production source changes or landing.

The mandatory full pipeline profile completes: 22 cases pass and the two
existing raster-to-vector cases remain deferred. Reviewed all 102 stages,
including 51 under the 1M scale label. No stage exceeds 1 s; the largest is
74.37 ms of device fixture construction. The only CPU-labeled 1M stage is
the 0.316 ms join index wrapper. Setup/export dominate several short cases;
there is no unexplained multi-second CPU stage. Zero-transfer cases pass.
Some suite fixtures cap physical rows; this is repository health evidence,
not a substitute for the full-size MA nearest experiments.

1M stage wall times in ms (full precision and row counts retained in evidence):

| Pipeline | Stages and ms |
|---|---|
| join-heavy | read_points 3.723; read_polygons 6.118; build_index 0.316; sjoin_query 0.653; assemble_join_rows 0.526; dissolve_groups 2.686; write_output 16.499 |
| relation-semijoin | read_inputs 9.819; build_index 0.220; sjoin_relation 0.552; semijoin_rowset 0.360; subset_rows 0.923; write_output 2.971 |
| small-grouped-constructive-reduce | build_device_grouped_polygons 40.095; native_grouped_union 46.380; native_reference_check 0.028 |
| grouped-capacity-partitions | build_grouped_partition_fixtures 61.540; mixed_strip_exact_union 65.889; positive_degenerate_union 60.730; native_reference_check 0.025 |
| grouped-disjoint-constructive-reduce | build_device_disjoint_groups 74.370; native_grouped_disjoint_subset 1.323; native_reference_check 0.025 |
| grouped-difference-constructive | build_device_grouped_difference_inputs 25.734; native_grouped_difference 9.427; native_reference_check 0.029 |
| constructive-output-native | build_device_pairwise_boxes 3.392; native_constructive_intersection 3.326; constructive_area_expression 1.088; constructive_expression_consumers 1.546; native_reference_check 0.026 |
| overlay-relation-constructive | build_native_overlay_inputs 6.597; build_spatial_index 0.100; candidate_relation 0.687; refine_relation 0.219; constructive_intersection 2.687; native_tabular_projection 0.900; native_reference_check 0.026 |
| constructive | read_points 3.753; clip_points 0.606; buffer_points 1.546; write_output 16.663 |
| predicate-heavy | read_geojson 70.770; load_polygons 7.206; point_in_polygon 0.384; filter_points 0.372; write_output 1.626 |
| zero-transfer | read_input 7.010; predicate_filter 0.406; subset_rows 1.002; write_output 3.518 |

Durable rollup: `docs/testing/nearest-index-strategy-evidence.json`; raw logs,
JSON, HTML profile and measured source snapshots remain in
`/tmp/vibespatial-osm/nearest-experiments/e16-*` and their experiment prefixes.
The [recommendation](nearest-index-recommendation.md) records the empirical
choice, measured tradeoffs, and remaining production graduation work.
