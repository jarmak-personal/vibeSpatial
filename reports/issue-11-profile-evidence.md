# Issue 11 Profile Evidence

This is the tracked evidence packet for the bounded exact fixed-`k` nearest
implementation. Raw audit JSON stays outside git because it contains GPU
sparklines and transfer-event detail; the hashes below bind those raw files,
and this packet records their complete source identity and every active 1M
stage. SF1000 was not run on this machine.

## Execution identity

- Date: 2026-09-01 UTC
- Host: `picard-4090`
- GPU: NVIDIA GeForce RTX 4090
- GPU UUID: `GPU-39aa5702-5aec-2729-8fb7-412d84ca1cbe`
- Driver: `580.173.02`
- VRAM: 24,564 MiB
- Package: vibeSpatial `0.5.3`
- Python: `3.13.12`
- Base revision: `bc022feeb7f621a0b25630ef45740ae4f883d492`
- Tracked source dirty: `true`
- Untracked source files: none
- Exact worktree source SHA-256:
  `f048d61c37e358a38b7b609fc768cda766dcabd5d70896d1015b8381d68e664e`

## Mandatory full profile

Command:

```text
uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 \
  --gpu-sparkline \
  --output /tmp/vibespatial-issue-11-full-profile-final4.json
```

Raw JSON SHA-256:
`0f629750efbb505bf7870eed5d047d11bcce2d3109d3296dd378636f18820b65`.
The 11 active 1M pipelines completed with zero fallback events, zero compute
materializations, and 20,528 bytes of compute D2H. The deferred
`raster-to-vector` entries have no stages. No active stage exceeded one second;
the slowest was `grouped-capacity-partitions/mixed_strip_exact_union` at
73.212 ms.

All active 1M stages, using wall elapsed seconds from the audit artifact:

| Pipeline | Stage | Boundary | Seconds |
|---|---|---:|---:|
| join-heavy | read_points | compute | 0.003030147 |
| join-heavy | read_polygons | compute | 0.005726841 |
| join-heavy | build_index | compute | 0.000260143 |
| join-heavy | sjoin_query | compute | 0.000637833 |
| join-heavy | assemble_join_rows | compute | 0.000451107 |
| join-heavy | dissolve_groups | compute | 0.002382145 |
| join-heavy | write_output | terminal | 0.015904649 |
| relation-semijoin | read_inputs | compute | 0.012044850 |
| relation-semijoin | build_index | compute | 0.000207869 |
| relation-semijoin | sjoin_relation | compute | 0.000575789 |
| relation-semijoin | semijoin_rowset | compute | 0.000387296 |
| relation-semijoin | subset_rows | compute | 0.000947377 |
| relation-semijoin | write_output | terminal | 0.003315081 |
| small-grouped-constructive-reduce | build_device_grouped_polygons | compute | 0.038494692 |
| small-grouped-constructive-reduce | native_grouped_union | compute | 0.055012884 |
| small-grouped-constructive-reduce | native_reference_check | reference | 0.000030676 |
| grouped-capacity-partitions | build_grouped_partition_fixtures | compute | 0.049704405 |
| grouped-capacity-partitions | mixed_strip_exact_union | compute | 0.073212167 |
| grouped-capacity-partitions | positive_degenerate_union | compute | 0.068004265 |
| grouped-capacity-partitions | native_reference_check | reference | 0.000021154 |
| grouped-disjoint-constructive-reduce | build_device_disjoint_groups | compute | 0.065518488 |
| grouped-disjoint-constructive-reduce | native_grouped_disjoint_subset | compute | 0.001310934 |
| grouped-disjoint-constructive-reduce | native_reference_check | reference | 0.000021255 |
| grouped-difference-constructive | build_device_grouped_difference_inputs | compute | 0.026236812 |
| grouped-difference-constructive | native_grouped_difference | compute | 0.010407847 |
| grouped-difference-constructive | native_reference_check | reference | 0.000025946 |
| constructive-output-native | build_device_pairwise_boxes | compute | 0.003536413 |
| constructive-output-native | native_constructive_intersection | compute | 0.005110804 |
| constructive-output-native | constructive_area_expression | compute | 0.003175700 |
| constructive-output-native | constructive_expression_consumers | compute | 0.003283209 |
| constructive-output-native | native_reference_check | reference | 0.000044294 |
| overlay-relation-constructive | build_native_overlay_inputs | compute | 0.006410345 |
| overlay-relation-constructive | build_spatial_index | compute | 0.000092709 |
| overlay-relation-constructive | candidate_relation | compute | 0.000686947 |
| overlay-relation-constructive | refine_relation | compute | 0.000229928 |
| overlay-relation-constructive | constructive_intersection | compute | 0.002892533 |
| overlay-relation-constructive | native_tabular_projection | compute | 0.000900694 |
| overlay-relation-constructive | native_reference_check | reference | 0.000024943 |
| constructive | read_points | compute | 0.003452052 |
| constructive | clip_points | compute | 0.000595782 |
| constructive | buffer_points | compute | 0.001257459 |
| constructive | write_output | terminal | 0.017916444 |
| predicate-heavy | read_geojson | compute | 0.069031620 |
| predicate-heavy | load_polygons | compute | 0.007046916 |
| predicate-heavy | point_in_polygon | compute | 0.000354256 |
| predicate-heavy | filter_points | compute | 0.000391782 |
| predicate-heavy | write_output | terminal | 0.001591432 |
| zero-transfer | read_input | compute | 0.007000098 |
| zero-transfer | predicate_filter | compute | 0.000402713 |
| zero-transfer | subset_rows | compute | 0.001043123 |
| zero-transfer | write_output | terminal | 0.003430969 |

## Nearest relation canary

Command:

```text
uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 \
  --gpu-sparkline --pipeline nearest-relation-producer \
  --output /tmp/vibespatial-issue-11-nearest-profile-final4.json
```

Raw JSON SHA-256:
`76af2f4de6bf02dc9feeba46db7d74df3b346aa81b1fc3348504626de22480ce`.
The 1M shape canary selected the GPU planner with no fallback, no compute
materialization, 16 bytes of scalar compute D2H, and an operation-local peak
of 1,290,256 bytes. Its complete 1M stages were:

| Stage | Boundary | Seconds |
|---|---:|---:|
| build_nearest_relation | compute | 0.027464146 |
| native_distance_consume | compute | 0.001373030 |
| native_attribute_match_filter | compute | 0.000286897 |
| build_right_nearest_relation | compute | 0.002570096 |
| public_reference_export | reference | 0.000437283 |

## SF100 comparator identity packet

The immutable comparator is
`benchmark_results/spatialbench/sf100/accepted-geopandas-comparator.json`,
SHA-256
`a75e20baa894f4609034c3f1440450bcd2120b778b6fda1d6330f22aaf5069a3`.
Its bound identity is:

- Schema/kind/status/date: `1`, `spatialbench_sf100_geopandas_optimized`,
  `accepted_immutable`, 2026-08-17.
- Engine: optimized GeoPandas `1.1.4`, Python `3.13.12`.
- Host: `picard-4090`, Intel i9-13900K, ext4 WD_BLACK SN770 NVMe.
- Dataset: SF100 GeoParquet 1.1/native GeoArrow, 179 files,
  37,940,233,646 bytes; manifest SHA-256
  `a970d40adc2f754e781197d9ec9af1f03627351ceb8d01e92e9c09ff6c5080d3`;
  inventory SHA-256
  `ad526c085c09b50410402945c24a9b6162b4467ceb4daa3c1f24d3db9848a35c`.
- Lock: `uv.lock` SHA-256
  `60942838f44a9792dfc8a49371e8a1367a59e8d858e39ed827dfa735e0d615fe`;
  pandas `3.0.1`, PyArrow `23.0.1`.
- Workload: SpatialBench `0.1.0`, 12 queries. Source hashes:
  `geopandas_optimized_queries.py`
  `8325d9bb2288b49008bc81eca8a2f9ff9eff6efdc2d472a9c7bce2e7877a2b35`,
  `geoparquet_public_api_queries.py`
  `b34099f0665ece2f23fa5044698eda82c962e1cf875313cbbbcd8b27080e63f9`,
  and `public_api_queries.py`
  `36e230f02f8f8e21b1f3542d6c9e76005bd1b7c5b3422e572a82ea8f08930e65`.
- Measurement: isolated process per engine/query, one warmup, three measured
  runs, median, `scan_compute_public_result`, serialization excluded.
- Correctness: exact rows/columns and exact integer/string/datetime cells;
  floating point `rtol=1e-6`, `atol=1e-9`.
- Accepted comparator total: 8,086.00 seconds; minimum suite speedup 10x.
- Identity-source hashes: provenance
  `1e3057c7e0e236bb1f5355d90033c7ae168bf61980a1fcff6fcd1d7caae96e63`,
  same-data correctness
  `bca547bce14685ad8903a9558161a801ff4dea8efab7fbcb0b588e002de40d8a`,
  cross-device report
  `43ff4c33222dc36e0b44c8f096cdac9c95469c8a925c7f4f05a9bad5191beed4`.
- Q12 oracle: 100 rows at
  `benchmark_results/spatialbench/sf100/2026-08-14-final-median/results/geopandas_optimized_q12_result.csv`,
  SHA-256
  `ff7f7155a0c47292de28dd5d92b96ce37cfca0f66e17f7df97e133c9f98828fe`;
  accepted median 626.64 seconds from 635.27, 625.19, and 626.64 seconds.

## Current SF100 Q12 result

Command used strict native mode, SF100, one measured run, result export, and
telemetry:

```text
env VIBESPATIAL_STRICT_NATIVE=1 UV_CACHE_DIR=/tmp/uv-cache \
  uv run python -m benchmarks.spatialbench.run_benchmark \
  --data-dir /home/picard/datasets/spatialbench/v0.1.0/sf100-geoparquet \
  --engines vibespatial --queries q12 --scale-factor 100 \
  --warmup-runs 0 --runs 1 --statistic median --timeout 7200 \
  --profile-telemetry \
  --output /tmp/vibespatial-issue-11-sf100-q12-final4.json \
  --result-dir /tmp/vibespatial-issue-11-sf100-results-final4
```

It did not run SF10 or SF1000. The run completed in 34.48 seconds with 100
rows and zero fallback events.

- Benchmark JSON SHA-256:
  `69a9729b1a6052b8675b2b216fa41a15eead14eea31ef18996712719f2208ce0`.
- Normalized result SHA-256:
  `4c9b9f08cf2a800d64412a4c0444162a81a088286e677963662844e0ac52d00d`.
- Ordered key SHA-256:
  `5e53fb7a9f4e26c6cbbb71c93ee8c1eaf3faf0aef8e15a8c9c6f1311cdc4a32d`.
- Exact ordered keys matched the frozen oracle; max absolute numeric delta was
  `1.4210854715202004e-14`, max relative delta was
  `1.729905677985445e-16`.
- Telemetry: peak VRAM 12,060,721,152 bytes; operation-local RMM peak
  6,008,950,876 bytes; 46,602 allocations; 1,783,277,721,131 allocated bytes;
  874 tracked D2H transfers totaling 7,662,452 bytes and 0.019254 seconds;
  pool reserve 11,363,306,624 bytes; final pool live 96,630,867 bytes; largest
  admitted allocation 22,291,953,856 bytes.

SF1 and an independent GeoPandas oracle were also checked during development;
SF100 is the final exact-source correctness and capacity run. SF10 was not
needed as an intermediate diagnostic. SF1000 was explicitly not run.
