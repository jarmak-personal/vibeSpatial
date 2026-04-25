# ADR0044 Rich Baseline 2026-04-25

<!-- DOC_HEADER:START
Scope: Post-ADR0044/0045 performance baseline for non-IO workflows, operations, and pipelines.
Read If: You are comparing public workflow speed, operation-level speed, or transient D2H health after the ADR0044 native substrate checkpoint.
STOP IF: You need IO shootout baselines; this baseline intentionally excludes the shootout IO directory.
Source Of Truth: Measurement checkpoint for committed SHA bae6767 on 2026-04-25.
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
- Commit: `bae6767` (`Honor public CRS on native Arrow exports`).
- GPU: NVIDIA GeForce RTX 4090.
- `CUDA_VISIBLE_DEVICES`: unset.
- Device nodes visible outside sandbox: `/dev/nvidia0`, `/dev/nvidiactl`,
  `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`, `/dev/nvidia-modeset`.
- Raw artifacts: `benchmark_results/working/rich_baseline_2026_04_25_head_bae6767/`.
- Artifact policy: `benchmark_results/` is ignored, so this tracked note is
  the durable repo record.

## Commands

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync vsbench shootout benchmarks/shootout --scale 10k --repeat 3 --json --output benchmark_results/working/rich_baseline_2026_04_25_head_bae6767/shootouts_10k_repeat3.json --timeout 900
```

Operation baselines used `vsbench run <operation> --scale 10k --repeat 3
--json --quiet --output ...` for the operation variants listed below.

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline --output benchmark_results/working/rich_baseline_2026_04_25_head_bae6767/pipelines_full_repeat1_gpu_sparkline.json
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/benchmark_pipelines.py --suite full --repeat 3 --profile-mode lean --output benchmark_results/working/rich_baseline_2026_04_25_head_bae6767/pipelines_full_repeat3_lean.json
```

## Summary

- Workflow shootouts: 14/14 passed fingerprint checks at 10K, repeat 3.
- Workflow geomean speedup: 1.11x vs GeoPandas.
- Workflow total median speedup: 1.38x vs GeoPandas, 3.411s GeoPandas vs
  2.468s vibeSpatial across all 14 workflows.
- Operation baselines now report real repeat summaries (`sample_count=3`).
- Pipeline full repeat-3 lean stays healthy at 1M: zero-transfer is 21.6ms
  with zero runtime D2H; join-heavy is 111.4ms with 24 runtime D2H transfers.
- Main warning signal: `gpu-dissolve method=disjoint_subset` is 4.50s vs
  71.2ms baseline, with repeated D2H warnings.

## Workflow Shootouts

| Workflow | GeoPandas ms | vibeSpatial ms | Speedup | Runtime D2H | Materializations | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| `accessibility_redevelopment.py` | 217.1 | 257.4 | 0.84x | 247 | 12 | 0 |
| `corridor_flood_priority.py` | 160.0 | 193.7 | 0.83x | 192 | 10 | 0 |
| `emergency_response_catchments.py` | 95.6 | 162.8 | 0.59x | 226 | 9 | 0 |
| `flood_exposure.py` | 38.0 | 32.6 | 1.16x | 41 | 3 | 0 |
| `habitat_corridor_compliance.py` | 124.3 | 145.5 | 0.85x | 174 | 12 | 1 |
| `insurance_flood_screening.py` | 39.8 | 110.3 | 0.36x | 244 | 7 | 0 |
| `nearby_buildings.py` | 97.5 | 47.3 | 2.06x | 27 | 1 | 0 |
| `network_service_area.py` | 91.0 | 88.9 | 1.02x | 50 | 4 | 0 |
| `parcel_zoning.py` | 64.7 | 58.8 | 1.10x | 51 | 6 | 0 |
| `redevelopment_screening.py` | 688.9 | 373.9 | 1.84x | 179 | 10 | 1 |
| `retail_trade_area_screening.py` | 612.9 | 296.2 | 2.07x | 81 | 9 | 0 |
| `site_suitability.py` | 660.2 | 267.2 | 2.47x | 73 | 7 | 0 |
| `transit_service_gap.py` | 227.0 | 222.1 | 1.02x | 185 | 8 | 1 |
| `vegetation_corridor.py` | 294.0 | 211.2 | 1.39x | 92 | 6 | 1 |

## Operation Baselines

| Operation variant | vibeSpatial ms | Baseline | Baseline ms | Speedup | Samples |
|---|---:|---|---:|---:|---:|
| `binary-predicates-contains` | 1.171 | shapely | 0.421 | 0.36x | 3 |
| `binary-predicates-covered-by` | 1.156 | shapely | 0.428 | 0.37x | 3 |
| `binary-predicates-intersects` | 1.140 | shapely | 1.065 | 0.93x | 3 |
| `bounds` | 0.225 | shapely | 0.100 | 0.44x | 3 |
| `clip-rect-line` | 3.199 | shapely | 3.880 | 1.21x | 3 |
| `clip-rect-polygon` | 0.972 | shapely | 2.387 | 2.46x | 3 |
| `gpu-dissolve-coverage` | 4.709 | shapely-coverage | 13.620 | 2.89x | 3 |
| `gpu-dissolve-disjoint-subset` | 4502.373 | shapely-disjoint_subset | 71.228 | 0.02x | 3 |
| `gpu-dissolve-unary` | 70.557 | shapely-unary | 68.657 | 0.97x | 3 |
| `gpu-overlay` | 63.008 | shapely-strtree-intersection | 233.404 | 3.70x | 3 |
| `make-valid` | 12.441 | baseline | 20.491 | 1.65x | 3 |
| `spatial-query-overlap02` | 1.420 | shapely_strtree | 3.034 | 2.14x | 3 |
| `spatial-query-overlap08` | 1.527 | shapely_strtree | 9.388 | 6.15x | 3 |
| `stroke-offset-curve` | 214.995 | shapely | 311.595 | 1.45x | 3 |
| `stroke-point-buffer` | 0.560 | shapely | 35.265 | 62.96x | 3 |

## Pipeline Baselines

| Pipeline | Scale | Elapsed ms | Runtime | Runtime D2H | Runtime D2H MB | Materializations | Fallbacks |
|---|---:|---:|---|---:|---:|---:|---:|
| `join-heavy` | 100000 | 40.1 | hybrid | 24 | 2.63 | 0 | 0 |
| `relation-semijoin` | 100000 | 14.2 | gpu | 3 | 0.00 | 0 | 0 |
| `small-grouped-constructive-reduce` | 100000 | 119.7 | hybrid | 11 | 0.33 | 2 | 0 |
| `constructive` | 100000 | 10.7 | hybrid | 3 | 0.60 | 0 | 0 |
| `predicate-heavy` | 100000 | 12.1 | gpu | 6 | 1.20 | 0 | 0 |
| `zero-transfer` | 100000 | 10.1 | gpu | 0 | 0.00 | 0 | 0 |
| `join-heavy` | 1000000 | 111.4 | hybrid | 24 | 25.97 | 0 | 0 |
| `relation-semijoin` | 1000000 | 25.1 | gpu | 3 | 0.00 | 0 | 0 |
| `small-grouped-constructive-reduce` | 1000000 | 119.7 | hybrid | 11 | 0.33 | 2 | 0 |
| `constructive` | 1000000 | 22.8 | hybrid | 3 | 6.00 | 0 | 0 |
| `predicate-heavy` | 1000000 | 42.1 | gpu | 6 | 12.00 | 0 | 0 |
| `zero-transfer` | 1000000 | 21.6 | gpu | 0 | 0.00 | 0 | 0 |

## Profile Gate

Top 1M audit stages from `--gpu-sparkline`:

| Pipeline | Total ms | Runtime D2H | Runtime D2H MB | Largest stages |
|---|---:|---:|---:|---|
| `join-heavy` | 98.4 | 24 | 25.97 | `dissolve_groups` 47.9ms CPU, `assemble_join_rows` 18.1ms GPU, `sjoin_query` 16.7ms GPU |
| `relation-semijoin` | 24.0 | 3 | 0.00 | `read_inputs` 14.7ms GPU, `write_output` 4.7ms GPU, `subset_rows` 1.5ms GPU |
| `small-grouped-constructive-reduce` | 128.3 | 11 | 0.33 | `shapely_reference` 72.7ms CPU, `build_device_grouped_polygons` 35.4ms GPU, `native_grouped_union` 19.3ms GPU |
| `constructive` | 28.8 | 3 | 6.00 | `write_output` 21.0ms GPU, `read_points` 3.3ms GPU, `buffer_points` 1.7ms GPU |
| `predicate-heavy` | 89.3 | 6 | 12.00 | `read_geojson` 60.9ms GPU, `load_polygons` 10.1ms GPU, `point_in_polygon` 3.4ms GPU |
| `zero-transfer` | 22.1 | 0 | 0.00 | `read_input` 13.8ms GPU, `write_output` 4.6ms GPU, `subset_rows` 1.6ms GPU |

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
