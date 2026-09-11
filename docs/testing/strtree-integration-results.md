# STRtree Integration Results

<!-- DOC_HEADER:START
Scope: Final public nearest timings, complex polygon canaries, correctness evidence and full pipeline profile.
Read If: You need current public STRtree integration performance or its verification limits.
STOP IF: You need implementation details rather than measured integration results.
Source Of Truth: Final local public nearest integration measurements and validation summary.
Body Budget: 173/200 lines
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
| 97-108 | Verification |
| 109-173 | Full 1M Pipeline Profile |
DOC_HEADER:END -->

## Intent

Record current public comparisons, correctness checks and the complete 1M pipeline profile.
Older experiments remain in the integration ledgers; these timings include lazy backend admission.

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
| line / polygon | 100,000 | 655.962 | 28.656 | 22.640 | 29.0x |
| line / polygon | 1,000,000 | 10000.657 | 182.010 | 173.908 | 57.5x |
| mixed / mixed | 100,000 | 878.601 | 47.698 | 40.613 | 21.6x |
| mixed / mixed | 1,000,000 | 11634.948 | 339.697 | 333.744 | 34.9x |
| multipart / multipart | 100,000 | 1568.112 | 44.432 | 38.103 | 41.2x |
| multipart / multipart | 1,000,000 | 18168.190 | 349.844 | 343.092 | 53.0x |
| point / point | 100,000 | 257.009 | 25.395 | 20.733 | 12.4x |
| point / point | 1,000,000 | 4012.801 | 176.029 | 161.229 | 24.9x |
| point / polygon | 100,000 | 639.434 | 25.561 | 21.358 | 29.9x |
| point / polygon | 1,000,000 | 8744.718 | 170.306 | 164.751 | 53.1x |
| polygon / point | 100,000 | 523.594 | 25.753 | 21.069 | 24.9x |
| polygon / point | 1,000,000 | 7361.895 | 178.226 | 172.288 | 42.7x |
| polygon / polygon | 100,000 | 883.123 | 30.359 | 25.268 | 35.0x |
| polygon / polygon | 1,000,000 | 10932.498 | 215.986 | 208.352 | 52.5x |
| Holed polygons (128 edges) | 100,000 | 12747.051 | 188.718 | 121.225 | 105.2x |
| Holed polygons (32 edges) | 1,000,000 | 28104.619 | 850.222 | 677.595 | 41.5x |

All 20 synthetic/complex cases match saved nearest pairs and distance oracles.
Each complex polygon in the scale comparison has a shell and hole.

| Shape canary | CPU reused | GPU first | GPU reused |
|---|---:|---:|---:|
| holes | 83.723 | 15.505 | 4.119 |
| overlap | 33.482 | 35.034 | 28.200 |
| single | 0.769 | 13.331 | 4.750 |
| skew | 25.338 | 15.335 | 4.962 |

The single 16,384-edge polygon remains a small-input loss. Dense overlap also
loses on the first query; returning every nearest tie is inherently output-sized.

| MA layout | Building queries | CPU reused | GPU first | GPU reused |
|---|---:|---:|---:|---:|
| roads | 1,000 | 16.619 | 23.271 | 16.797 |
| roads | 10,000 | 177.949 | 29.979 | 23.795 |
| roads | 100,000 | 1778.848 | 67.242 | 61.812 |
| roads | 2,680,339 | 48111.705 | 986.104 | 984.966 |
| segments | 1,000 | 8.620 | 43.862 | 19.690 |
| segments | 10,000 | 80.114 | 45.005 | 20.147 |
| segments | 100,000 | 800.311 | 70.352 | 46.042 |
| segments | 2,680,339 | 21463.824 | 775.960 | 753.922 |

MA uses all 991,441 roads or 8,728,450 segments. Every one of the eight cases
matches its saved pair/distance oracle. Full input has 2,680,339 building points.

## One-Shot WKB Comparison

Ingress plus index preparation plus first public query, in milliseconds.
These are fresh inputs after process initialization, not cold startup timings.

| Workload | CPU total | GPU total | Speedup |
|---|---:|---:|---:|
| Holed polygons 100,000 | 12944.906 | 466.361 | 27.8x |
| Holed polygons 1,000,000 | 29313.753 | 1534.890 | 19.1x |
| Mixed 1,000,000 | 12758.887 | 476.129 | 26.8x |
| MA roads-all | 48923.948 | 1218.160 | 40.2x |
| MA segments-all | 24899.436 | 1274.217 | 19.5x |

## Verification

Final native/metric/runtime/predicate/contract suite: 525 passed, one optional SciPy skip.
Upstream sindex/sjoin: 416 passed, 59 optional/version skips, one existing xfail.
Native metadata/nearest/Arrow stream selection: 30 passed. Latest packed STR
memcheck: 83 passed, zero errors. Earlier memcheck covered 113 tests
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
| join-heavy | read_points | 3.717 |
| join-heavy | read_polygons | 6.115 |
| join-heavy | build_index | 0.263 |
| join-heavy | sjoin_query | 0.662 |
| join-heavy | assemble_join_rows | 0.550 |
| join-heavy | dissolve_groups | 2.403 |
| join-heavy | write_output | 15.899 |
| relation-semijoin | read_inputs | 12.221 |
| relation-semijoin | build_index | 0.209 |
| relation-semijoin | sjoin_relation | 0.590 |
| relation-semijoin | semijoin_rowset | 0.408 |
| relation-semijoin | subset_rows | 0.953 |
| relation-semijoin | write_output | 3.389 |
| small-grouped-constructive-reduce | build_device_grouped_polygons | 40.004 |
| small-grouped-constructive-reduce | native_grouped_union | 53.713 |
| small-grouped-constructive-reduce | native_reference_check | 0.025 |
| grouped-capacity-partitions | build_grouped_partition_fixtures | 50.610 |
| grouped-capacity-partitions | mixed_strip_exact_union | 74.039 |
| grouped-capacity-partitions | positive_degenerate_union | 68.286 |
| grouped-capacity-partitions | native_reference_check | 0.020 |
| grouped-disjoint-constructive-reduce | build_device_disjoint_groups | 68.754 |
| grouped-disjoint-constructive-reduce | native_grouped_disjoint_subset | 1.312 |
| grouped-disjoint-constructive-reduce | native_reference_check | 0.022 |
| grouped-difference-constructive | build_device_grouped_difference_inputs | 27.506 |
| grouped-difference-constructive | native_grouped_difference | 10.798 |
| grouped-difference-constructive | native_reference_check | 0.028 |
| constructive-output-native | build_device_pairwise_boxes | 3.329 |
| constructive-output-native | native_constructive_intersection | 3.863 |
| constructive-output-native | constructive_area_expression | 1.227 |
| constructive-output-native | constructive_expression_consumers | 1.781 |
| constructive-output-native | native_reference_check | 0.038 |
| overlay-relation-constructive | build_native_overlay_inputs | 6.350 |
| overlay-relation-constructive | build_spatial_index | 0.092 |
| overlay-relation-constructive | candidate_relation | 0.809 |
| overlay-relation-constructive | refine_relation | 0.210 |
| overlay-relation-constructive | constructive_intersection | 3.002 |
| overlay-relation-constructive | native_tabular_projection | 0.883 |
| overlay-relation-constructive | native_reference_check | 0.024 |
| constructive | read_points | 4.160 |
| constructive | clip_points | 0.906 |
| constructive | buffer_points | 1.537 |
| constructive | write_output | 20.016 |
| predicate-heavy | read_geojson | 70.070 |
| predicate-heavy | load_polygons | 7.037 |
| predicate-heavy | point_in_polygon | 0.370 |
| predicate-heavy | filter_points | 0.376 |
| predicate-heavy | write_output | 1.541 |
| zero-transfer | read_input | 7.651 |
| zero-transfer | predicate_filter | 0.470 |
| zero-transfer | subset_rows | 1.094 |
| zero-transfer | write_output | 4.149 |

Compute transfers remain bounded metadata packets in grouped topology; other
pipelines retain zero compute D2H bytes. No unexplained CPU-heavy stage was found.
Full identities, source fingerprints, raw artifact hashes, review reports and
failed alternatives are retained in `strtree-backend-landing-evidence.json`.
