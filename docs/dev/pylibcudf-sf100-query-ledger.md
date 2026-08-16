# pylibcudf SF100 Query Evidence Ledger

<!-- DOC_HEADER:START
Scope: Per-query semantic, physical-shape, memory, correctness, and benchmark evidence for the pylibcudf-backed SpatialBench SF100 program.
Read If: You are changing an SF100 query path, native tabular primitive, memory estimate, export boundary, or performance claim.
STOP IF: You only need the program milestones or architecture; open the execution plan instead.
Source Of Truth: Query-level evidence ledger required by the pylibcudf SF100 execution plan.
Body Budget: 201/260 lines
Document: docs/dev/pylibcudf-sf100-query-ledger.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-7 | Intent |
| 8-14 | Request Signals |
| 15-21 | Open First |
| 22-27 | Verify |
| 28-34 | Risks |
| 35-67 | Measurement Contract And Shared Evidence |
| 68-181 | Q1-Q12 Ledger |
| 182-192 | Rejected Physical Shapes |
| 193-201 | Artifact Map |
DOC_HEADER:END -->

## Intent

Freeze query-level SQL semantics, physical shapes, memory contracts, and
reproducible evidence for the SF100 pylibcudf performance program.

## Request Signals

- SF100 query evidence
- SpatialBench query semantics
- pylibcudf query shape
- benchmark provenance

## Open First

- `docs/dev/pylibcudf-sf100-execution-plan.md`
- `benchmark_results/spatialbench/sf100/2026-08-14-final-median/final_benchmark.json`
- `benchmark_results/spatialbench/sf100/2026-08-14-final-median/telemetry_summary.json`
- `../sedona-spatialbench/docs/queries.md`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "pylibcudf SF100 query evidence ledger"`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Clean aggregate timing cannot be presented as kernel-only stage timing.
- A suite total can hide a query regression or correctness failure.
- An underestimated candidate or sort shape can invalidate an otherwise fast run.
- Benchmark-specific shortcuts do not establish a reusable public-API contract.

## Measurement Contract And Shared Evidence

This ledger freezes SQL intent from SpatialBench rather than preserving behavior
from the old GeoPandas translation. The final comparison uses the same converted
SF100 GeoParquet dataset, one isolated process per engine/query, one untimed
warmup, and the median of three measured runs. Timed work includes scan,
compute, and public result construction. Result CSV serialization is excluded.

The post-implementation profile artifact reports aggregate query wall and each
explicit D2H export because internal query scopes were not enabled during clean
timing. Thus `device pipeline` below is the measured scan/compute/public-result
aggregate, not an inferred kernel-only time. The earlier stage-attribution
profile remains the before-state for Q3, Q4, Q7, and Q12.

The completion audit reran all twelve queries at commit `c74a773` with one
warmup and three measured runs. The 641.76 s VS sum is 12.742x faster than the
unchanged 8,177.23 s optimized-GPD baseline; all outputs pass the SF100
same-data oracle and every per-query timing remains within the 5% gate.

All queries pass the SF1 canonical-answer oracle and the same-data SF100
optimized-GeoPandas comparison at `rtol=1e-6`, `atol=1e-9`. Q10 differs only by
admitted sub-ULP reduction order. Every final profile has zero library fallback
events. The compatibility contract preserves SQL null semantics, stable stated
tie ordering, canonical column order, pandas/GeoPandas public result types, and
the source key columns rather than an incidental RangeIndex. Unsupported native
dtypes or operations decline to an observable exact pandas path; strict-native
mode rejects that boundary.

Memory values below are process peak / RMM pool-reserved peak / largest admitted
operation. `N`, `C`, `P`, and `G` denote input rows, grid cells, candidate pairs,
and groups. Scans are row-group bounded and use decoded-byte admission; explicit
trip batches are 8M rows for Q2/Q6, 4M for Q10, and at most 32M otherwise.

## Q1-Q12 Ledger

### Q1: Sedona-Center Distance Top 100

- Semantics/schema/order: trips within 0.45 degrees; `t_tripkey, pickup_lon, pickup_lat, t_pickuptime, distance_to_center`; distance then key ascending; 100 rows.
- Projection/pushdown: trip key, pickup time, pickup geometry; distance predicate is evaluated after row-group scan.
- Shapes/chain/export: trip rows -> distance `NativeExpression` -> filtered `NativeRowSet` -> bounded top-k -> one terminal pandas frame.
- Primitive/budget: fused custom distance plus pylibcudf selection; scan decoded bytes + `N * (64 + source_width + 5 * expanded_key_width) + 1 MiB` top-k scratch, shard-local top-k merged to 100.
- Measured after: profiled pipeline 13.07 s; D2H 0.0045 s / 96,000 B; 12.01 / 11.56 / 11.17 GiB; clean GPD 106.03 s versus VS 12.51 s (8.48x).
- Correctness/fallback: finite fp64 distance, datetime/key dtype and exact tie order preserved; SF1 and SF100 pass; no fallback.

### Q2: Coconino Pickup Count

- Semantics/schema/order: exact point/intersects selected Coconino County polygon; scalar `trip_count_in_coconino_county`; one row.
- Projection/pushdown: trip pickup geometry; zone name and boundary, with `z_name == 'Coconino County'` pushed to the small zone table.
- Shapes/chain/export: 8M point batches -> cached point grid / polygon refinement -> `NativeRelation` count -> scalar pandas result; measured result is 53,348.
- Primitive/budget: point-grid candidate generation plus exact fp64 custom predicate; grid build is `192*N + 96*C + 1 MiB`, relation capacity is admitted and batches resize before launch.
- Measured after: profiled pipeline 8.35 s; D2H 0.4002 s / 389,527 B; 19.57 / 19.13 / 2.79 GiB; clean GPD 113.25 s versus VS 7.89 s (14.35x).
- Correctness/fallback: boundary points follow `intersects`, null geometry is false, count is int64; SF1/SF100 pass; no fallback.

### Q3: Monthly Buffered-Box Statistics

- Semantics/schema/order: pickups within 0.045 of the fixed box; `pickup_month, total_trips, avg_distance, avg_duration, avg_fare`; month ascending; 84 rows.
- Projection/pushdown: trip key, pickup/dropoff timestamps, distance, fare, pickup geometry; fixed spatial bounds prune row groups where metadata permits.
- Shapes/chain/export: rows -> distance rowset -> native datetime/month expressions -> `NativeGrouped` reductions -> terminal pandas frame.
- Primitive/budget: custom distance, pylibcudf gather/group/reduce; decoded scan bytes plus selected rows and group state, with row-group shards and compact `G`-sized merge state.
- Measured after: profiled pipeline 13.39 s; D2H 0.0029 s / 70,016 B; 12.62 / 12.18 / 11.17 GiB; clean GPD 405.03 s versus VS 12.81 s (31.62x).
- Before/after stages: before read 217.75 s, distance 15.62 s, host take 13.47 s, group 0.43 s; after total 12.79 s, with no intermediate public frame.
- Correctness/fallback: UTC/naive local-calendar components are native; non-UTC timestamps observably use pandas; decimal/duration/null contracts and month order pass SF1/SF100.

### Q4: Zone Distribution Of Top Tips

- Semantics/schema/order: top 1,000 trips by tip descending/key ascending, exact `within` zone join; `z_zonekey, z_name, trip_count`; count descending/key ascending; 614 rows.
- Projection/pushdown: trip key, tip, pickup geometry; zone key, name, boundary; top-k precedes the spatial join.
- Shapes/chain/export: native numeric-cast expression -> 1,000-row `NativeRowSet` -> point-location `NativeRelation` -> grouped count -> pandas.
- Primitive/budget: pylibcudf top-k/gather/group plus exact custom point-location; top-k formula above, then bounded 1,000-by-zone candidates.
- Measured after: profiled pipeline 9.11 s; D2H 1.0529 s / 665,680 B; 9.62 / 9.18 / 11.17 GiB; clean GPD 231.98 s versus VS 8.53 s (27.20x).
- Before/after stages: before read 107.03 s, host top-k 30.12 s, take/sort 9.86 s, joins 4.33 s; after total 8.28 s.
- Correctness/fallback: decimal source is explicitly public-cast to float before public `nlargest`; missing values last and signed-zero/tie order match pandas; SF1/SF100 pass, no fallback.

### Q5: Repeat-Customer Monthly Dropoff Hull

- Semantics/schema/order: customer/month groups with count >5; `c_custkey, customer_name, pickup_month, monthly_travel_hull_area, dropoff_count`; area descending, key/month ascending; 100 rows.
- Projection/pushdown: trip customer key, pickup time, dropoff geometry; customer key/name; HAVING applies after grouping.
- Shapes/chain/export: packed customer/month codes -> `NativeGrouped` point collections -> grouped convex hull/area -> bounded top-k -> terminal pandas.
- Primitive/budget: pylibcudf grouping plus custom segmented hull; input coordinates + group codes/offsets + admitted hull output, processed by source shard and merged as compact group/hull state; top-k formula applies to `G`.
- Measured after: profiled pipeline 135.06 s; D2H 1.4817 s / 10.34 GB at an explicit public arithmetic boundary; 16.21 / 12.83 / 11.17 GiB; clean GPD 834.69 s versus VS 126.94 s (6.58x).
- Correctness/fallback: all groups compute hull before ranking, degenerate hulls have exact area semantics, customer/name/month/null/tie contracts pass SF1/SF100; no library fallback.

### Q6: Sedona-Radius Zone Statistics

- Semantics/schema/order: pickups inside zones whose geometry intersects the 0.45-degree Sedona circle; `z_zonekey, z_name, total_pickups, avg_distance, avg_duration`; count descending/key ascending; 19 rows.
- Projection/pushdown: trip key, pickup geometry, distance, pickup/dropoff timestamps; zone key/name/boundary; zone bounds prefilter applies before 8M point batches.
- Shapes/chain/export: point-grid/location `NativeRelation` -> pylibcudf joined reductions -> terminal pandas.
- Primitive/budget: point grid `192*N + 96*C + 1 MiB`, exact fp64 refinement, bounded relation and shard-local sum/count state; resize below remaining query budget.
- Measured after: public relation reduction plus device-backed Series accumulation profiles at 17.07 s; D2H 0.3129 s / 15.51 MB; peak 19.59 GiB; clean GPD 371.42 s versus VS 16.32 s (22.76x).
- Correctness/fallback: exact zone/pickup predicate, duration/decimal averages, null exclusion and stable order pass SF1/SF100; no fallback.

### Q7: Detour Ratio Top 100

- Semantics/schema/order: reported distance metres divided by nonzero line distance; `t_tripkey, reported_distance_m, line_distance_m, detour_ratio`; ratio descending/key ascending; 100 rows.
- Projection/pushdown: trip key, reported decimal distance, pickup and dropoff geometry; zero geometric distances are excluded.
- Shapes/chain/export: dual-geometry distance expression + decimal cast/ratio expression -> bounded top-k `NativeRowSet` -> pandas.
- Primitive/budget: custom point distance and pylibcudf expression/top-k; decoded scan bytes plus the top-k formula, shard-local candidates merged to 100.
- Measured after: profiled pipeline 4.71 s; D2H 0.0026 s / 64,000 B; 17.78 / 17.34 / 11.17 GiB; clean GPD 337.44 s versus VS 4.25 s (79.40x).
- Before/after stages: before read 127.14 s, decimal conversion 27.63 s, distance 6.84 s, host ratio/top-k 5.20 s; after total 4.50 s.
- Correctness/fallback: decimal-cast lineage, division-by-zero null semantics, fp64 distances and ties pass SF1/SF100; no fallback.

### Q8: Nearby Pickups Per Building

- Semantics/schema/order: pickups within 0.0045 degrees of each building; `b_buildingkey, b_name, nearby_pickup_count`; count descending/key ascending; 100 rows.
- Projection/pushdown: building key/name/boundary and trip pickup geometry; building-expanded bounds constrain candidate generation.
- Shapes/chain/export: reusable building spatial index -> bounded distance relation -> relation count -> top-k -> pandas.
- Primitive/budget: custom index/candidate/refinement plus pylibcudf reduction/top-k; `P * (left_index + right_index + optional distance)` is admitted per shard, while only per-building counts persist.
- Measured after: profiled pipeline 17.49 s; D2H 0.3626 s / 5,853,512 B; 13.21 / 12.77 / 11.17 GiB; clean GPD 285.06 s versus VS 16.98 s (16.79x).
- Correctness/fallback: distance threshold includes the boundary, null geometry contributes no pair, zero-count buildings are absent; SF1/SF100 and tie order pass; no fallback.

### Q9: Building Conflation IoU

- Semantics/schema/order: distinct candidate building pairs with positive intersection, IoU = intersection/union; `building_1, building_2, iou`; IoU descending then both keys ascending; 100 rows.
- Projection/pushdown: building key and boundary only; self/duplicate pairs removed before exact overlay.
- Shapes/chain/export: building spatial index -> candidate `NativeRelation` -> exact intersection/union expressions -> bounded top-k -> pandas.
- Primitive/budget: custom spatial candidate/overlay kernels plus pylibcudf top-k; candidate pair bytes and geometry output capacity are admitted and shardable by left rows.
- Measured after: profiled pipeline 0.64 s; D2H 0.0006 s / 109,212 B; 2.24 / 0.10 / 0.08 GiB; clean GPD 0.19 s versus VS 0.14 s (1.36x).
- Correctness/fallback: valid positive union, unique ordered pair keys and fp64 IoU pass SF1/SF100; no fallback. Q9 is exempt from per-query 10x but remains in suite/correctness/regression gates.

### Q10: All-Zone Pickup Statistics

- Semantics/schema/order: point-in-zone pickup join; `z_zonekey, pickup_zone, avg_duration, avg_distance, num_trips`; duration descending nulls last/key ascending; 100 rows.
- Projection/pushdown: zone key/name/boundary; trip key, pickup geometry, distance, pickup/dropoff timestamps; zone partitions bound each 4M-row trip pass.
- Shapes/chain/export: cached point grid -> per-zone-shard location relation -> streamed sum/count reducers -> top-k -> pandas.
- Primitive/budget: custom grid/refinement plus pylibcudf reducers; grid formula plus bounded `P`, five zone partitions, and compact per-zone state avoid eager relation consolidation.
- Measured after: public relation reduction plus five device-backed partition accumulators profiles at 125.93 s; D2H 0.2949 s / 33.09 MB; peak 13.20 GiB; clean GPD 1,738.00 s versus VS 126.43 s (13.75x).
- Correctness/fallback: exact point-in-polygon, null-aware averages/durations and stable nulls-last order pass SF1/SF100; no fallback.

### Q11: Cross-Zone Trip Count

- Semantics/schema/order: pickup and dropoff each locate to a zone, count rows whose zone keys differ; scalar `cross_zone_trip_count`; one row.
- Projection/pushdown: zone key/boundary; trip key, pickup and dropoff geometry; endpoints are processed separately to keep candidate shapes bounded.
- Shapes/chain/export: cached point grid -> pickup/dropoff location relations -> key inequality/count reduction -> scalar pandas.
- Primitive/budget: custom grid/refinement plus pylibcudf relation reducer; each endpoint uses the grid/relation budget independently, with compact keyed partial counts merged across shards.
- Measured after: profiled pipeline 269.22 s; D2H 1.6361 s / 14.10 GB at explicit public reducers; 13.20 / 12.76 / 5.62 GiB; clean GPD 3,127.50 s versus VS 266.42 s (11.74x).
- Correctness/fallback: rows with an unlocated/null endpoint do not join, multiplicity follows SQL join semantics, int64 count passes SF1/SF100; no fallback.

### Q12: Five-Nearest-Building Isolation Top 100

- Semantics/schema/order: average exact distance from each pickup to its five nearest buildings; `t_tripkey, avg_distance_to_5_nearest`; average descending/key ascending; 100 rows.
- Projection/pushdown: trip key/pickup geometry and building key/boundary; Hilbert ordering supplies public coarse locality, not an approximate answer.
- Shapes/chain/export: Hilbert `NativeExpression` -> certified public `take` -> exact k=5 KNN `NativeRelation`/distance -> bounded top-k -> pandas.
- Primitive/budget: custom Hilbert/KNN plus pylibcudf take/reduce/top-k; candidate `P * (indices + fp64 distance)` is admitted per trip shard and only five pairs/row plus 100 global candidates persist.
- Measured after: profiled pipeline 23.51 s; D2H 0.0065 s / 7,524,656 B; 13.34 / 12.64 / 11.17 GiB; clean GPD 626.64 s versus VS 21.20 s (29.56x).
- Before/after stages: before ten-shard upper bounds 39.81 s, CPU exact KNN 34.40 s, reads 1.86 s, setup 1.27 s; full before 917.71 s, after 21.48 s.
- Correctness/fallback: exactly five neighbors, fp64 distance/average, certified Hilbert index domain and stable ties pass SF1/SF100; no fallback.

## Rejected Physical Shapes

- A 32M Q2 relation requested a single 12-GiB block; 8M batches retain exactness and fit the planner.
- A 16M Q6 batch exceeded remaining budget by 28.3 MiB; deterministic 8M resizing is used.
- An eager Q5 32M grouped sort estimated 44.44 GiB; segmented grouped hulls replace it.
- Eager Q10 zone-partition consolidation reserved 20.71 GB; five streamed partitions retain 5.474 GB.
- Combined Q11 endpoint relations duplicate the largest candidate shape; endpoint relations remain separate.
- One Morton interval per query bbox scans 14.04B positions for 28.07M pairs; a cached exact point grid replaces it.
- A 0.1-degree membership grid creates 14.1M memberships and 34.7M coarse pairs for only 2.23M exact pairs.
- Interval-fp32 point-in-polygon was slower than exact fp64 on the RTX 4090 and risked boundary ambiguity; the exact refinement kernel remains fp64.

## Artifact Map

- Before-stage evidence: `docs/dev/pylibcudf-sf100-execution-plan.md#locked-sf100-baseline-2026-08-13`.
- Final medians: `benchmark_results/spatialbench/sf100/2026-08-14-final-median/final_benchmark.json`.
- Runtime telemetry: `benchmark_results/spatialbench/sf100/2026-08-14-final-median/final_telemetry.json` and `telemetry_summary.json`.
- Correctness: `benchmark_results/spatialbench/sf100/2026-08-14-final-median/same_data_correctness.json`.
- End-to-end profile: `benchmark_results/spatialbench/sf100/2026-08-14-final-median/pipeline_profile_summary.json`.
- Reproducibility: `benchmark_results/spatialbench/sf100/2026-08-14-final-median/provenance.json`.
- Public relation reducer checkpoint: `benchmark_results/spatialbench/sf100/2026-08-15-public-relation-reducer/checkpoint.json`.
