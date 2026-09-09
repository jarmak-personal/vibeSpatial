# OSM PBF Performance Evidence

<!-- DOC_HEADER:START
Scope: Massachusetts PBF timings, native GeoParquet comparison, memory pressure, and regression evidence.
Read If: You are comparing PBF performance or reproducing capacity and relation-shape measurements.
STOP IF: You only need the PBF API or implementation contract in osm-pbf-native.md.
Source Of Truth: Measured OSM PBF performance and validation evidence for the 4090 implementation.
Body Budget: 188/240 lines
Document: docs/testing/osm-pbf-performance.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-7 | Intent |
| 8-14 | Request Signals |
| 15-21 | Open First |
| 22-28 | Verify |
| 29-35 | Risks |
| 36-61 | Dataset and Measurement |
| 62-88 | Read Times |
| 89-118 | Capacity and Work Shape |
| 119-180 | End-to-End Profile |
| 181-188 | Regression Coverage |
DOC_HEADER:END -->

## Intent

Record the measured native PBF implementation, same-data comparator, capacity
canaries, and end-to-end regression profile from 2026-09-09.

## Request Signals

- Massachusetts OSM benchmark
- PBF capacity pressure
- GeoArrow GeoParquet comparison
- PBF performance evidence

## Open First

- docs/architecture/osm-pbf-native.md
- docs/testing/osm-pbf-evidence.json
- scripts/benchmark_osm_pbf.py
- scripts/benchmark_osm_pbf_capacity.py

## Verify

- `uv run python scripts/benchmark_osm_pbf.py DATA.osm.pbf --cache /tmp/osm-comparator --output /tmp/osm-current.json --repeat 3`
- `uv run python scripts/benchmark_osm_pbf_capacity.py pressure DATA.osm.pbf --baseline /tmp/osm-comparator/baseline.json --headroom-mib 1280 --output /tmp/osm-pressure.json`
- `uv run python scripts/benchmark_osm_pbf_capacity.py junction --output /tmp/osm-junction.json`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- These are warm-filesystem, synchronized reads; first-process calls are separate.
- Managed pages consume host backing; RMM counters exclude managed allocations.
- Passing one extract is evidence, not a guarantee for every planet-scale shape.
- A native GeoParquet GeometryCollection comparison is unavailable in version 1.1.

## Dataset and Measurement

Geofabrik Massachusetts complete extract, 310,023,499 bytes, downloaded from
[Geofabrik](https://download.geofabrik.de/north-america/us/massachusetts-latest.osm.pbf).
OSM data is copyright OpenStreetMap contributors, available under ODbL.
SHA256: `d7f2220cf2d0ed6b6eeb432f1986c27799a7248bd5ec6f24746fdf1d53c17263`.
There are 5,237 DataBlobs and 729,777,162 decompressed bytes.

Host: RTX 4090 24 GiB, i9-13900K, driver 580.173.02, Linux x86-64.
GDAL 3.11.4 / pyogrio 0.12.1, pylibcudf 26.2.1, nvCOMP 5.1.0.21,
PyArrow 23.0.1. The JSON records complete package, machine, workload and
candidate-source hashes. The user's 13s/43s RTX 6000 measurements have a
different machine and measurement boundary; they are not the speedup denominator.

Each backend runs in a separate process. Imports and conversion are excluded;
CUDA is synchronized at the return boundary. Three warm repetitions follow
the first read. GDAL comparators were measured once and reused after identity
validation. The comparator's preparation history is preserved in the JSON.
No cached candidate timings are used.

Preconversion uses native GeoArrow GeoParquet 1.1, Snappy, and 1M-row groups.
All attributes and all geometries are fingerprinted against GDAL, outside
timing. Polygon fingerprints normalize ring starts and orientation, preserving
invalid rings; other geometries preserve exact coordinate/member order.
Every layer matches. This includes inherited tags and ordered collections.

## Read Times

Warm medians in seconds. Public means a returned GeoDataFrame. Native means
the private result before public export; ordinary families retain device
geometry, while collections use the terminal Shapely export boundary.

| Layer | Rows | GDAL public | GPU PBF public | Native GeoParquet public | PBF speedup |
|---|---:|---:|---:|---:|---:|
| points | 518,420 | 1.281 | 0.515 | 0.014 | 2.49x |
| lines | 1,168,444 | 4.502 | 0.766 | 0.060 | 5.87x |
| multilinestrings | 1,935 | 7.017 | 0.975 | 0.014 | 7.20x |
| multipolygons | 2,953,572 | 14.020 | 1.792 | 0.140 | 7.82x |
| other_relations | 13,797 | 13.950 | 1.118 | unavailable | 12.48x |

| Layer | Native PBF | First process read | Peak RMM MiB | Final RMM MiB |
|---|---:|---:|---:|---:|
| points | 0.509 | 0.655 | 1142 | 69 |
| lines | 0.761 | 1.016 | 1246 | 342 |
| multilinestrings | 0.979 | 1.114 | 798 | 26 |
| multipolygons | 1.784 | 2.292 | 3485 | 1045 |
| other_relations | 1.039 | 1.160 | 772 | 14 |

First-process measurements can reuse the on-disk kernel cache. All ordinary
timed runs report zero managed-workspace events. GeoParquet remains the
appropriate repeated-read target: it skips protobuf, decompression of unrelated
OSM records, reference resolution, and relation topology reconstruction.

## Capacity and Work Shape

The physical-pressure canary reserved 23,444,127,744 bytes outside RMM before
the reader grew its pool. Only 1,340,145,664 bytes (about 1.25 GiB) remained
free. The full polygon read completed in 10.045s with an identical fingerprint.
Normal final output is about 1.02 GiB. Blob, output, and attribute-assembly
workspaces paged through CUDA managed memory. RMM peak was 426,353,290 bytes;
this excludes managed pages and must not be presented as total memory usage.

Admission now consults current driver free memory as well as allocator limits.
Expanded inherited strings account for repeated way tags and both libcudf
gather/scatter copies. Tests force this path with many polygons inheriting one
large tag. Geometry scratch can page when one relation alone exceeds budget.
Final owned int32 offsets and libcudf string capacity remain explicit limits;
these measurements do not establish planet-file support.

A relation with all open members sharing one junction tests endpoint skew:

| Members | Endpoint states | Whole assembly seconds |
|---:|---:|---:|
| 10,000 | 20,000 | 0.010 |
| 100,000 | 200,000 | 0.074 |
| 1,000,000 | 2,000,000 | 1.203 |

The group directory and monotone cursors eliminate repeated degree-squared
endpoint scans. Ordering and pointer jumping retain their sort/logarithmic
costs. Greedy branch traversal follows source order on the GPU. Ring nesting
prunes parent tests by relation, integer area and bounds, then applies OGR's
binary64 boundary semantics. It does not allocate a dense ring-pair matrix.

## End-to-End Profile

The mandatory full suite completed with both 100K and 1M scales. Every stage
was reviewed: no unexpected CPU-heavy stage exceeds 1s. The largest 1M stage
is GeoJSON input at 70.54ms; constructive fixture/union stages are 27-65ms.
Read, indexing, selection and output stages have no unexplained wall-time gap.
The 1M stage wall times below are milliseconds.

| Pipeline | Stage | Wall ms |
|---|---|---:|
| join-heavy | read_points | 3.550 |
| join-heavy | read_polygons | 6.110 |
| join-heavy | build_index | 0.280 |
| join-heavy | sjoin_query | 0.687 |
| join-heavy | assemble_join_rows | 0.595 |
| join-heavy | dissolve_groups | 2.350 |
| join-heavy | write_output | 16.130 |
| relation-semijoin | read_inputs | 10.140 |
| relation-semijoin | build_index | 0.229 |
| relation-semijoin | sjoin_relation | 0.596 |
| relation-semijoin | semijoin_rowset | 0.375 |
| relation-semijoin | subset_rows | 0.980 |
| relation-semijoin | write_output | 3.300 |
| small-grouped-constructive-reduce | build_device_grouped_polygons | 40.700 |
| small-grouped-constructive-reduce | native_grouped_union | 46.270 |
| small-grouped-constructive-reduce | native_reference_check | 0.028 |
| grouped-capacity-partitions | build_grouped_partition_fixtures | 47.150 |
| grouped-capacity-partitions | mixed_strip_exact_union | 64.490 |
| grouped-capacity-partitions | positive_degenerate_union | 61.730 |
| grouped-capacity-partitions | native_reference_check | 0.023 |
| grouped-disjoint-constructive-reduce | build_device_disjoint_groups | 64.630 |
| grouped-disjoint-constructive-reduce | native_grouped_disjoint_subset | 1.290 |
| grouped-disjoint-constructive-reduce | native_reference_check | 0.026 |
| grouped-difference-constructive | build_device_grouped_difference_inputs | 27.670 |
| grouped-difference-constructive | native_grouped_difference | 9.410 |
| grouped-difference-constructive | native_reference_check | 0.026 |
| constructive-output-native | build_device_pairwise_boxes | 3.730 |
| constructive-output-native | native_constructive_intersection | 3.590 |
| constructive-output-native | constructive_area_expression | 1.090 |
| constructive-output-native | constructive_expression_consumers | 1.510 |
| constructive-output-native | native_reference_check | 0.026 |
| overlay-relation-constructive | build_native_overlay_inputs | 6.620 |
| overlay-relation-constructive | build_spatial_index | 0.109 |
| overlay-relation-constructive | candidate_relation | 0.726 |
| overlay-relation-constructive | refine_relation | 0.201 |
| overlay-relation-constructive | constructive_intersection | 2.630 |
| overlay-relation-constructive | native_tabular_projection | 0.937 |
| overlay-relation-constructive | native_reference_check | 0.024 |
| constructive | read_points | 3.410 |
| constructive | clip_points | 0.489 |
| constructive | buffer_points | 1.470 |
| constructive | write_output | 19.750 |
| predicate-heavy | read_geojson | 70.540 |
| predicate-heavy | load_polygons | 7.360 |
| predicate-heavy | point_in_polygon | 0.400 |
| predicate-heavy | filter_points | 0.393 |
| predicate-heavy | write_output | 1.590 |
| zero-transfer | read_input | 7.240 |
| zero-transfer | predicate_filter | 0.412 |
| zero-transfer | subset_rows | 1.040 |
| zero-transfer | write_output | 3.170 |

## Regression Coverage

The focused reader/file suite passed 178 tests. A further 68 selected native
composition, collection, metadata, Arrow and precision tests passed. Coverage
includes malformed protobuf/zlib, source replacement, signed offsets, missing
members, duplicate way IDs, degenerate/duplicate rings, tag inheritance,
collection row operations, empty schemas, strict-native ingress and completion
after reprojection on a nondefault stream.
