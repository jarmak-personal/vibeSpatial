# STRtree Integration Results

<!-- DOC_HEADER:START
Scope: Final public nearest timings, complex polygon canaries, correctness evidence and full pipeline profile.
Read If: You need current public STRtree integration performance or its verification limits.
STOP IF: You need implementation details rather than measured integration results.
Source Of Truth: Final local public nearest integration measurements and validation summary.
Body Budget: 172/200 lines
Document: docs/testing/strtree-integration-results.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-7 | Intent |
| 8-13 | Request Signals |
| 14-19 | Open First |
| 20-24 | Verify |
| 25-30 | Risks |
| 31-83 | Public Measurements |
| 84-96 | One-Shot WKB Comparison |
| 97-107 | Verification |
| 108-172 | Full 1M Pipeline Profile |
DOC_HEADER:END -->

## Intent

Record current public comparisons, correctness checks and the complete 1M pipeline profile.
Older experiments remain in the integration ledger; all timings below use the final implementation.

## Request Signals

- public nearest performance
- complex polygon benchmark
- STRtree integration profile

## Open First

- docs/dev/strtree-integration-ledger.md
- docs/architecture/spatial-index-backends.md
- docs/testing/strtree-integration-evidence.json

## Verify

- `uv run pytest tests/test_packed_str_index.py tests/test_segment_bvh.py --run-gpu -q`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- First-query time includes lazy index preparation; accessor time is not build cost.
- Single screening trials do not provide statistical confidence intervals.
- Unfiltered initcheck retains a separately reproduced nullable Arrow dependency finding.

## Public Measurements

RTX 4090 / i9-13900K; Shapely 2.1.2 / GEOS 3.13.1. Times are milliseconds.
Same WKB input and public pair/distance output. Imports, fixture/oracle work excluded.
The table uses the fresh-input trial. Ingress is separate; first query includes actual
hierarchy preparation. CPU values are reused from validated immutable comparators.

| Query / indexed geometry | Rows per side | CPU reused | GPU first | GPU reused | Speedup |
|---|---:|---:|---:|---:|---:|
| line / polygon | 100,000 | 655.962 | 29.428 | 23.996 | 27.3x |
| line / polygon | 1,000,000 | 10000.657 | 187.475 | 176.243 | 56.7x |
| mixed / mixed | 100,000 | 878.601 | 48.595 | 40.431 | 21.7x |
| mixed / mixed | 1,000,000 | 11634.948 | 344.745 | 337.003 | 34.5x |
| multipart / multipart | 100,000 | 1568.112 | 45.708 | 39.075 | 40.1x |
| multipart / multipart | 1,000,000 | 18168.190 | 363.702 | 351.169 | 51.7x |
| point / point | 100,000 | 257.009 | 25.988 | 22.272 | 11.5x |
| point / point | 1,000,000 | 4012.801 | 167.987 | 161.492 | 24.8x |
| point / polygon | 100,000 | 639.434 | 26.653 | 23.689 | 27.0x |
| point / polygon | 1,000,000 | 8744.718 | 178.830 | 167.165 | 52.3x |
| polygon / point | 100,000 | 523.594 | 27.984 | 22.595 | 23.2x |
| polygon / point | 1,000,000 | 7361.895 | 175.560 | 170.642 | 43.1x |
| polygon / polygon | 100,000 | 883.123 | 31.515 | 25.958 | 34.0x |
| polygon / polygon | 1,000,000 | 10932.498 | 220.599 | 212.420 | 51.5x |
| Holed polygons (128 edges) | 100,000 | 12747.051 | 190.174 | 121.561 | 104.9x |
| Holed polygons (32 edges) | 1,000,000 | 28104.619 | 859.397 | 675.971 | 41.6x |

All 20 synthetic/complex cases match saved nearest pairs and distance oracles.
Each complex polygon in the scale comparison has a shell and hole.

| Shape canary | CPU reused | GPU first | GPU reused |
|---|---:|---:|---:|
| holes | 83.723 | 19.731 | 5.751 |
| overlap | 33.482 | 35.473 | 28.576 |
| single | 0.769 | 14.204 | 5.078 |
| skew | 25.338 | 15.961 | 5.042 |

The single 16,384-edge polygon remains a small-input loss. Dense overlap also
loses on the first query; returning every nearest tie is inherently output-sized.

| MA layout | Building queries | CPU reused | GPU first | GPU reused |
|---|---:|---:|---:|---:|
| roads | 1,000 | 16.619 | 23.437 | 16.491 |
| roads | 10,000 | 177.949 | 31.667 | 24.533 |
| roads | 100,000 | 1778.848 | 71.575 | 62.387 |
| roads | 2,680,339 | 48111.705 | 974.259 | 987.450 |
| segments | 1,000 | 8.620 | 52.835 | 20.417 |
| segments | 10,000 | 80.114 | 53.749 | 20.945 |
| segments | 100,000 | 800.311 | 78.345 | 45.807 |
| segments | 2,680,339 | 21463.824 | 783.299 | 780.217 |

MA uses all 991,441 roads or 8,728,450 segments. Every one of the eight cases
matches its saved pair/distance oracle. Full input has 2,680,339 building points.

## One-Shot WKB Comparison

Ingress plus index preparation plus first public query, in milliseconds.
These are fresh inputs after process initialization, not cold startup timings.

| Workload | CPU total | GPU total | Speedup |
|---|---:|---:|---:|
| Holed polygons 100,000 | 12944.906 | 480.444 | 26.9x |
| Holed polygons 1,000,000 | 29313.753 | 1525.287 | 19.2x |
| Mixed 1,000,000 | 12758.887 | 480.593 | 26.5x |
| MA roads-all | 48923.948 | 1209.624 | 40.4x |
| MA segments-all | 24899.436 | 1264.716 | 19.7x |

## Verification

Final native/metric/runtime/predicate suite: 444 passed, one optional SciPy skip.
Upstream sindex/sjoin: 416 passed, 59 optional/version skips, one existing xfail.
Native metadata/nearest/Arrow stream selection: 30 passed. Memcheck: 113 tests
plus 20 updated sort/predicate cases, zero errors. Owned nearest/sort initcheck:
38 passed; count-prefix/transposition initcheck: 18 passed; both zero errors.
Unfiltered public initcheck: 94 assertions pass, eight libcudf bitmap reads remain.
The standalone PyArrow/pylibcudf/CuPy control reproduces that dependency issue
without importing vibeSpatial. Failure logs are retained; no suppression is used.

## Full 1M Pipeline Profile

All 22 runnable pipeline cases pass; two raster cases retain existing deferrals.
All 102 stages were reviewed, with zero fallback events. The 51 stages below
are the 1M-scale entries; some constructive fixtures intentionally cap actual
rows below the scale label. Raw row counts and transfer packets are in the JSON.

| Pipeline | Stage | Milliseconds |
|---|---|---:|
| join-heavy | read_points | 3.218 |
| join-heavy | read_polygons | 5.500 |
| join-heavy | build_index | 0.261 |
| join-heavy | sjoin_query | 0.674 |
| join-heavy | assemble_join_rows | 0.526 |
| join-heavy | dissolve_groups | 2.394 |
| join-heavy | write_output | 16.427 |
| relation-semijoin | read_inputs | 12.319 |
| relation-semijoin | build_index | 0.237 |
| relation-semijoin | sjoin_relation | 0.592 |
| relation-semijoin | semijoin_rowset | 0.393 |
| relation-semijoin | subset_rows | 1.015 |
| relation-semijoin | write_output | 3.150 |
| small-grouped-constructive-reduce | build_device_grouped_polygons | 40.733 |
| small-grouped-constructive-reduce | native_grouped_union | 53.170 |
| small-grouped-constructive-reduce | native_reference_check | 0.030 |
| grouped-capacity-partitions | build_grouped_partition_fixtures | 49.838 |
| grouped-capacity-partitions | mixed_strip_exact_union | 75.540 |
| grouped-capacity-partitions | positive_degenerate_union | 69.648 |
| grouped-capacity-partitions | native_reference_check | 0.026 |
| grouped-disjoint-constructive-reduce | build_device_disjoint_groups | 64.636 |
| grouped-disjoint-constructive-reduce | native_grouped_disjoint_subset | 1.393 |
| grouped-disjoint-constructive-reduce | native_reference_check | 0.024 |
| grouped-difference-constructive | build_device_grouped_difference_inputs | 27.426 |
| grouped-difference-constructive | native_grouped_difference | 10.768 |
| grouped-difference-constructive | native_reference_check | 0.028 |
| constructive-output-native | build_device_pairwise_boxes | 3.813 |
| constructive-output-native | native_constructive_intersection | 3.980 |
| constructive-output-native | constructive_area_expression | 1.198 |
| constructive-output-native | constructive_expression_consumers | 1.721 |
| constructive-output-native | native_reference_check | 0.027 |
| overlay-relation-constructive | build_native_overlay_inputs | 6.900 |
| overlay-relation-constructive | build_spatial_index | 0.105 |
| overlay-relation-constructive | candidate_relation | 0.791 |
| overlay-relation-constructive | refine_relation | 0.243 |
| overlay-relation-constructive | constructive_intersection | 3.213 |
| overlay-relation-constructive | native_tabular_projection | 1.055 |
| overlay-relation-constructive | native_reference_check | 0.026 |
| constructive | read_points | 3.523 |
| constructive | clip_points | 0.582 |
| constructive | buffer_points | 1.513 |
| constructive | write_output | 16.883 |
| predicate-heavy | read_geojson | 70.696 |
| predicate-heavy | load_polygons | 7.188 |
| predicate-heavy | point_in_polygon | 0.392 |
| predicate-heavy | filter_points | 0.386 |
| predicate-heavy | write_output | 1.540 |
| zero-transfer | read_input | 8.107 |
| zero-transfer | predicate_filter | 0.436 |
| zero-transfer | subset_rows | 1.244 |
| zero-transfer | write_output | 3.661 |

Compute transfers remain bounded metadata packets in grouped topology; other
pipelines retain zero compute D2H bytes. No unexplained CPU-heavy stage was found.
Full identities, source fingerprints, raw artifact hashes, review reports and
failed alternatives are retained in `strtree-integration-evidence.json`.
