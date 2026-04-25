# ADR0044 Rich Baseline 2026-04-25

<!-- DOC_HEADER:START
Scope: Post-ADR0044/0045 performance baseline for non-IO workflows, operations, and pipelines.
Read If: You are comparing public workflow speed, operation-level speed, or transient D2H health after the ADR0044 native substrate checkpoint.
STOP IF: You need IO shootout baselines; this baseline intentionally excludes the shootout IO directory.
Source Of Truth: Measurement checkpoint for committed SHA 0ff2a07 on 2026-04-25.
Body Budget: 166/180 lines
Document: docs/testing/adr0044-rich-baseline-2026-04-25.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-12 | Intent |
| 13-20 | Request Signals |
| 21-27 | Open First |
| 28-33 | Verify |
| 34-42 | Risks |
| 43-54 | Environment |
| 55-68 | Commands |
| 69-80 | Summary |
| 81-99 | Workflow Shootouts |
| 100-119 | Operation Baselines |
| 120-136 | Pipeline Baselines |
| 137-149 | Profile Gate |
| 150-166 | Interpretation |
DOC_HEADER:END -->

Use this baseline as the ADR0044/0045 checkpoint before the next
generalized performance pass.

## Intent

Capture where the repo stands after the private native execution substrate
checkpoint and the benchmark harness repeat/JSON fixes. The target is not to
optimize these specific workflows. They are measurement canaries for generic
public API composition, transient work, and host/device transfer shape.

## Request Signals

- adr0044 baseline
- rich baseline
- workflow shootout
- operation benchmark
- transient D2H

## Open First

- docs/testing/adr0044-rich-baseline-2026-04-25.md
- docs/testing/pipeline-benchmarks.md
- docs/testing/performance-tiers.md
- docs/dev/private-native-execution-substrate-plan.md

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run vsbench shootout benchmarks/shootout --scale 10k --repeat 3`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Raw artifacts live under ignored `benchmark_results/working/` and are not
  durable unless this note is updated.
- Sandbox GPU visibility can make timings meaningless; rerun performance
  commands outside the sandbox before comparing.
- Workflow wins are canaries only; do not overfit implementation strategy to
  a single shootout script.

## Environment

- Date: 2026-04-25.
- Commit: `0ff2a07` (`Fix vsbench repeat and shootout JSON baselines`).
- GPU: NVIDIA GeForce RTX 4090.
- `CUDA_VISIBLE_DEVICES`: unset.
- Device nodes visible outside sandbox: `/dev/nvidia0`, `/dev/nvidiactl`,
  `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`, `/dev/nvidia-modeset`.
- Raw artifacts: `benchmark_results/working/rich_baseline_2026_04_25_head_0ff2a07/`.
- Artifact policy: `benchmark_results/` is ignored, so this tracked note is
  the durable repo record.

## Commands

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync vsbench shootout benchmarks/shootout --scale 10k --repeat 3 --json --output benchmark_results/working/rich_baseline_2026_04_25_head_0ff2a07/shootouts_10k_repeat3.json --timeout 900
```

Operation baselines used `vsbench run <operation> --scale 10k --repeat 3
--json --quiet --output ...` for the operation variants listed below.

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline --output benchmark_results/working/rich_baseline_2026_04_25_head_0ff2a07/pipelines_full_repeat1_gpu_sparkline.json
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/benchmark_pipelines.py --suite full --repeat 3 --profile-mode lean --output benchmark_results/working/rich_baseline_2026_04_25_head_0ff2a07/pipelines_full_repeat3_lean.json
```

## Summary

- Workflow shootouts: 14/14 passed fingerprint checks at 10K, repeat 3.
- Workflow geomean speedup: 1.12x vs GeoPandas.
- Workflow total median speedup: 1.39x vs GeoPandas, 3.425s GeoPandas vs
  2.465s vibeSpatial across all 14 workflows.
- Operation baselines now report real repeat summaries (`sample_count=3`).
- Pipeline full repeat-3 lean stays healthy at 1M: zero-transfer is 21.7ms
  with zero runtime D2H; join-heavy is 95.4ms with 24 runtime D2H transfers.
- Main warning signal: `gpu-dissolve method=disjoint_subset` is 4.44s vs
  70.6ms baseline, with repeated D2H warnings.

## Workflow Shootouts

| Workflow | GeoPandas ms | vibeSpatial ms | Speedup | Runtime D2H | Materializations | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| `accessibility_redevelopment.py` | 218.4 | 252.4 | 0.87x | 247 | 12 | 0 |
| `corridor_flood_priority.py` | 160.1 | 189.6 | 0.84x | 192 | 10 | 0 |
| `emergency_response_catchments.py` | 95.4 | 157.8 | 0.60x | 226 | 9 | 0 |
| `flood_exposure.py` | 37.7 | 32.1 | 1.18x | 41 | 3 | 0 |
| `habitat_corridor_compliance.py` | 124.1 | 147.3 | 0.84x | 174 | 12 | 1 |
| `insurance_flood_screening.py` | 40.7 | 109.0 | 0.37x | 244 | 7 | 0 |
| `nearby_buildings.py` | 97.1 | 49.5 | 1.96x | 27 | 1 | 0 |
| `network_service_area.py` | 92.2 | 87.0 | 1.06x | 50 | 4 | 0 |
| `parcel_zoning.py` | 65.1 | 58.4 | 1.12x | 51 | 6 | 0 |
| `redevelopment_screening.py` | 692.0 | 367.4 | 1.88x | 179 | 10 | 1 |
| `retail_trade_area_screening.py` | 615.5 | 303.2 | 2.03x | 81 | 9 | 0 |
| `site_suitability.py` | 656.0 | 268.1 | 2.45x | 73 | 7 | 0 |
| `transit_service_gap.py` | 231.5 | 232.9 | 0.99x | 185 | 8 | 1 |
| `vegetation_corridor.py` | 298.9 | 209.9 | 1.42x | 92 | 6 | 1 |

## Operation Baselines

| Operation variant | vibeSpatial ms | Baseline | Baseline ms | Speedup | Samples |
|---|---:|---|---:|---:|---:|
| `binary-predicates-contains` | 1.161 | shapely | 0.414 | 0.36x | 3 |
| `binary-predicates-covered-by` | 1.155 | shapely | 0.432 | 0.37x | 3 |
| `binary-predicates-intersects` | 1.157 | shapely | 1.104 | 0.95x | 3 |
| `bounds` | 0.223 | shapely | 0.097 | 0.44x | 3 |
| `clip-rect-line` | 3.186 | shapely | 3.736 | 1.17x | 3 |
| `clip-rect-polygon` | 0.963 | shapely | 2.318 | 2.41x | 3 |
| `gpu-dissolve-coverage` | 4.667 | shapely-coverage | 13.606 | 2.92x | 3 |
| `gpu-dissolve-disjoint-subset` | 4443.660 | shapely-disjoint_subset | 70.558 | 0.02x | 3 |
| `gpu-dissolve-unary` | 71.940 | shapely-unary | 68.368 | 0.95x | 3 |
| `gpu-overlay` | 58.761 | shapely-strtree-intersection | 233.800 | 3.98x | 3 |
| `make-valid` | 12.717 | baseline | 21.414 | 1.68x | 3 |
| `spatial-query-overlap02` | 1.413 | shapely_strtree | 3.040 | 2.15x | 3 |
| `spatial-query-overlap08` | 1.479 | shapely_strtree | 9.473 | 6.41x | 3 |
| `stroke-offset-curve` | 213.764 | shapely | 309.179 | 1.45x | 3 |
| `stroke-point-buffer` | 0.541 | shapely | 33.626 | 62.17x | 3 |

## Pipeline Baselines

| Pipeline | Scale | Elapsed ms | Runtime | Runtime D2H | Runtime D2H MB | Materializations | Fallbacks |
|---|---:|---:|---|---:|---:|---:|---:|
| `join-heavy` | 100000 | 39.3 | hybrid | 24 | 2.63 | 0 | 0 |
| `relation-semijoin` | 100000 | 13.6 | gpu | 3 | 0.00 | 0 | 0 |
| `small-grouped-constructive-reduce` | 100000 | 117.9 | hybrid | 11 | 0.33 | 2 | 0 |
| `constructive` | 100000 | 10.8 | hybrid | 3 | 0.60 | 0 | 0 |
| `predicate-heavy` | 100000 | 12.4 | gpu | 6 | 1.20 | 0 | 0 |
| `zero-transfer` | 100000 | 10.0 | gpu | 0 | 0.00 | 0 | 0 |
| `join-heavy` | 1000000 | 95.4 | hybrid | 24 | 25.97 | 0 | 0 |
| `relation-semijoin` | 1000000 | 25.6 | gpu | 3 | 0.00 | 0 | 0 |
| `small-grouped-constructive-reduce` | 1000000 | 120.9 | hybrid | 11 | 0.33 | 2 | 0 |
| `constructive` | 1000000 | 22.9 | hybrid | 3 | 6.00 | 0 | 0 |
| `predicate-heavy` | 1000000 | 42.1 | gpu | 6 | 12.00 | 0 | 0 |
| `zero-transfer` | 1000000 | 21.7 | gpu | 0 | 0.00 | 0 | 0 |

## Profile Gate

Top 1M audit stages from `--gpu-sparkline`:

| Pipeline | Total ms | Runtime D2H | Runtime D2H MB | Largest stages |
|---|---:|---:|---:|---|
| `join-heavy` | 105.3 | 24 | 25.97 | `dissolve_groups` 51.5ms CPU, `assemble_join_rows` 18.7ms GPU, `sjoin_query` 17.4ms GPU |
| `relation-semijoin` | 26.6 | 3 | 0.00 | `read_inputs` 16.4ms GPU, `write_output` 5.4ms GPU, `subset_rows` 1.6ms GPU |
| `small-grouped-constructive-reduce` | 130.3 | 11 | 0.33 | `shapely_reference` 74.2ms CPU, `build_device_grouped_polygons` 35.2ms GPU, `native_grouped_union` 20.2ms GPU |
| `constructive` | 27.8 | 3 | 6.00 | `write_output` 19.7ms GPU, `read_points` 3.5ms GPU, `buffer_points` 1.7ms GPU |
| `predicate-heavy` | 88.3 | 6 | 12.00 | `read_geojson` 60.5ms GPU, `load_polygons` 9.8ms GPU, `point_in_polygon` 3.3ms GPU |
| `zero-transfer` | 27.7 | 0 | 0.00 | `read_input` 17.8ms GPU, `write_output` 5.5ms GPU, `subset_rows` 1.8ms GPU |

## Interpretation

The ADR0044 substrate is helping where workflows can stay in larger native
chunks: site suitability, retail, nearby buildings, redevelopment, and
vegetation corridor are all faster than GeoPandas. The slow workflows are not
missing "GPU work" in a simple sense; they are dominated by many small public
API transitions, runtime D2H checks, and compatibility materializations.

The operation table shows the same shape. High-throughput native kernels are
excellent once the work is large enough (`stroke-point-buffer`, spatial query,
overlay), while tiny bounds and binary predicate calls are still slower than
Shapely because launch/composition overhead dominates.

The highest priority generic fixes remain: remove transient runtime D2H from
predicate/bounds/buffer setup paths, make public rowset/copy/filter composition
consume private native state without compatibility exports, and quarantine or
rewrite bad-shape modes such as `dissolve(method="disjoint_subset")`.
