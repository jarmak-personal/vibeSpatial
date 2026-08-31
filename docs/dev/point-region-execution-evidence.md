# Point-Region Execution Evidence

<!-- DOC_HEADER:START
Scope: Current-revision evidence for the point-region profiler and classification-once paired point-grid reduction.
Read If: You are changing prepared point-region refinement, paired spatial aggregation, point-grid candidate reduction, or validating SF100 Q11 performance.
STOP IF: You only need public predicate semantics or the superseded generic device-planning proposal.
Source Of Truth: Measurement record for docs/dev/evidence-first-point-region-execution-plan.md.
Body Budget: 255/260 lines
Document: docs/dev/point-region-execution-evidence.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-9 | Intent |
| 10-19 | Request Signals |
| 20-29 | Open First |
| 30-37 | Verify |
| 38-48 | Risks |
| 49-74 | Environment And Commands |
| 75-95 | Datacenter Handoff |
| 96-123 | E0 Baseline |
| 124-152 | E1 Attribution |
| 153-167 | Alternatives Falsified |
| 168-194 | Selected Alternative |
| 195-233 | Validation Status |
| 234-255 | 2026-08-31 Wider Directory And Coverage Result |
DOC_HEADER:END -->

## Intent

Record the evidence used to select one exact execution improvement without
introducing a generic device planner. SF100 Q11 is the motivating public
workflow, while deterministic public shape cases protect the broader
point-region problem space.

## Request Signals

- point-region evidence
- point-in-polygon profile
- Q11 performance
- classification once
- paired point-grid reduction
- RTX 4090 measurements
- H100 completion gate

## Open First

- `docs/dev/evidence-first-point-region-execution-plan.md`
- `docs/dev/cross-device-performance-report.md`
- `docs/decisions/0032-point-in-polygon-gpu-utilization-diagnosis.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `scripts/profile_point_region.py`
- `src/vibespatial/predicates/point_region_profile.py`
- `src/vibespatial/spatial/point_grid_index.py`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run pytest tests/test_point_region_profile.py tests/test_spatial_query.py -q`
- `uv run python scripts/profile_point_region.py --points 16384 --repeat 5 --measure-only --output benchmark_results/point_region/current.json`
- run the public 10K, 1M, and SF100 gates named in the active plan
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Instrumented and production kernels can drift if their entry points are not
  verified together.
- A local synthetic improvement can regress the complete public Q11 workflow.
- Consumer-only timing is insufficient evidence for a cross-device selector.
- Benchmark artifacts can become stale after runtime, allocator, or kernel
  changes.
- Candidate-level counters can distort the kernel if aggregation is moved into
  the inner edge loop.

## Environment And Commands

Local consumer evidence was collected on 2026-08-18 with:

- NVIDIA GeForce RTX 4090, compute capability 8.9, 128 SMs, 24 GiB
- SpatialBench v0.1.0 SF100 converted to GeoParquet
- one warmup and one measured run for the uninstrumented Q11 comparison
- the repository allocator defaults and strict-native public execution

The Q11 command was:

```bash
uv run python benchmarks/spatialbench/run_benchmark.py \
  --data-dir /home/picard/datasets/spatialbench/v0.1.0/sf100-geoparquet \
  --engines vibespatial --queries q11 --scale-factor 100 \
  --warmup-runs 1 --runs 1 --statistic median --timeout 1200
```

The deterministic shape corpus is reached only through public GeoParquet IO
and `SpatialIndex.query_pair_aggregate`:

```bash
uv run python scripts/profile_point_region.py --points 16384 --repeat 5 \
  --measure-only --output benchmark_results/point_region/current.json
```

## Datacenter Handoff

The shape-corpus artifact records the imported package path and Git revision,
tracked-worktree state, Python/CuPy/CUDA versions, and logical device facts.
On H100 hardware, run the production corpus once with clean baseline commit
`1fbd892c9d33a56c6e2b6362689c77759d0ce6a1` on `PYTHONPATH` and once with the
current source:

```bash
PYTHONPATH=/path/to/baseline/src uv run python scripts/profile_point_region.py \
  --points 16384 --repeat 5 --measure-only \
  --output benchmark_results/point_region/h100-baseline.json
uv run python scripts/profile_point_region.py \
  --points 16384 --repeat 5 --measure-only \
  --output benchmark_results/point_region/h100-current.json
```

Then run the current source once without `--measure-only` to capture bounded
physical counters. All three invocations use public GeoParquet IO and
`SpatialIndex.query_pair_aggregate`; no private executor is benchmarked.

## E0 Baseline

The fresh clean-HEAD public Q11 baseline was 311.46 seconds and returned one row with
`cross_zone_trip_count=1511054981`. There were no fallback events. The
reference profile attributed nearly all useful wall time to bounded paired
spatial reduction rather than GeoParquet IO or result export.

The first bounded profiler run recorded across five prepared MultiPolygon
groups:

| Metric | Baseline |
|---|---:|
| exact candidates | 11,265,678,037 |
| candidate-parts considered | 2,412,190,273,900 |
| active parts | 8,974,833,230 |
| selected-bin edge visits | 6,827,917,811,140 |
| orientation calls | 156,770,351 |
| exact-kernel time | 290.165 s |
| prepared-index builds | 5 |
| peak pool live bytes | 11,689,640,959 |
| pool reserved bytes | 17,044,435,712 |

The original percentile sample was launch-front biased. Aggregate counters,
maxima, timings, preparation, and memory remain valid, but percentile values
from the existing artifacts are withdrawn. Schema 2 distributes each launch's
actual sample quota across its full logical range; future percentile evidence
must come from a schema-2 recapture.

## E1 Attribution

The profiler is opt-in and uses separate instrumented kernels. Disabled
profiling leaves the production CUDA kernels unchanged. It records one
block-reduced fixed summary and at most 65,536 distributed samples per
prepared region group. No atomic is placed in the candidate edge loop.

The evidence rejects fp64 orientation throughput as the primary limiter:
156.8 million orientation calls are small compared with 11.3 billion
candidates, 2.4 trillion considered parts, and 6.8 trillion selected-bin edge
visits. Preparation also builds only once per region partition and is reused
for every shard.

The post-change profile recorded these unaffected aggregate measurements:

| Metric | Baseline | Classification once | Change |
|---|---:|---:|---:|
| exact candidates | 11.266 B | 7.981 B | -29.2% |
| exact-kernel time | 290.165 s | 210.904 s | -27.3% |
| instrumented Q11 wall | 330.21 s | 252.41 s | -23.6% |
| prepared-index builds | 5 | 5 | unchanged |
| peak pool live | 11.690 GB | 11.656 GB | -0.3% |
| pool reserved | 17.044 GB | 17.044 GB | unchanged |

Each region group had 462 launches and 461 cache hits after its single build.
Exact group maxima ranged from 1,404 to 18,933 considered parts and from 37,652
to 193,292 edge visits. Existing median and p99 values are intentionally not
used because they predate the schema-2 distributed sampler.

## Alternatives Falsified

Three independently exact fp64 traversal shapes were measured and removed
because complete Q11 performance falsified their local wins:

- edge-warp refinement improved the synthetic long-edge case by 65.7%, but
  Q11 regressed to 445.12 seconds
- whole-warp candidate-part refinement improved multipart skew by 39.4%, but
  Q11 exceeded eight minutes and was aborted
- block-local hybrid thresholds of 8 and 128 both exceeded the baseline and
  six-minute stop boundary

These experiments show why a 4090 microbenchmark crossover is not sufficient
selection evidence. None remains in the production registry or dispatch path.

## Selected Alternative

Repeated classification, not a new point-in-polygon kernel shape, was the
supported hypothesis. In a paired aggregate, the pickup candidate pass already
classifies the aligned dropoff point for the shared count. The implementation
now retains those exact dropoff results and classifies only dropoff candidate
pairs absent from the pickup grid's conservative superset.

The exclusion relation is exact at the conservative-candidate level:

- normal rows compare the aligned pickup point's grid cell with the identical
  query grid window used by candidate generation
- oversized pickup rows are already exhaustive and therefore exclude every
  aligned row from the second exact pass
- empty, invalid, non-finite, or out-of-window pickup points exclude nothing
- a device selection produces a stable active prefix and rejected zero tail
  without reading cardinality on the host

The physical tile remains bounded by the existing 64-byte-per-lane budget and
four-times-free-memory headroom. The additional mask, scan, positions, gathered
row vectors, exact output, and relation scratch remain inside that bound.
Allocation failure occurs before the new exact launch and propagates; it does
not retry after submission or silently fall back to CPU.

Uninstrumented public Q11 improved from 311.46 to 238.32 seconds, a 23.5%
end-to-end gain, with a byte-identical normalized result.

## Validation Status

Current consumer-device results:

- asymmetric pickup/dropoff candidate oracle: exact, no fallback
- normal and oversized bounded point-grid partitions: exact
- spatial-query GPU suite: 132 passed, one optional SciPy skip
- point-in-polygon, binary predicate, adaptive, and precision suites: 75 passed
- upstream predicate smoke: 4 passed
- public 10K shootout: 14/14 matching fingerprints
- public 10K VS subtotal: 2.6728 seconds versus 2.6645 before, +0.31%
- public 1M closure workflows: redevelopment 404.98 seconds (+0.27%), retail
  7.01 seconds (-0.67%), site 4.14 seconds (+0.26%), and transit 60.83
  seconds (+0.49%), all with unchanged fingerprints
- SF100: 12/12 within the established numeric oracle, zero fallbacks; eleven
  byte-identical outputs and Q6 differing only below 1e-19 serialization noise
- full 1M pipeline profile: 11 successful, one pre-existing raster deferment,
  zero compute D2H, zero compute materialization, zero fallback, and no stage
  above one second
- maintained binary-predicate public benchmark: 1.01 ms at 10K and 17.36 ms
  at 1M
- clean-HEAD versus current automatic 128K protected medians across 21 runs,
  above the point-grid activation threshold: simple -0.68%, long-bin +0.14%,
  multipart +0.26%, and uniform +0.38%

Raw local artifacts are under `benchmark_results/point_region/` and
`benchmark_results/spatialbench/sf100/2026-08-18-point-region-final*`.
The tracked handoff includes the current profile, 10K/1M shootouts, full
pipeline profile, normalized SF100 results, and a compact checkpoint.

Real datacenter Hopper evidence is now recorded in
`docs/dev/cross-device-performance-report.md`. On H200, the four protected
shape medians changed by +0.27% to +0.61%, while public Q11 improved from
116.43 to 107.16 seconds (8.0%). This clears the cross-device safety gate and
also shows that the 4090's 23.5% Q11 gain is not a universal device ratio.
The implementation uses no product name, compute capability, SM count, or
optional performance attribute for selection, so absent optional device facts
leave the permanent baseline selection contract unchanged.

## 2026-08-31 Wider Directory And Coverage Result

The SF100 follow-up used the same RTX 4090 and strict-native public Q10/Q11
paths. Device capacity selected 64 y bins and an admitted `8x8` conservative
coverage grid for each of five prepared region groups. One warmup plus one
measured profiled run produced:

| Metric | Wider y only | With coverage | Change |
|---|---:|---:|---:|
| Q10 wall | 59.38 s | 56.13 s | -5.5% |
| Q10 exact kernel | 24.60 s | 20.67 s | -16.0% |
| Q10 edges visited | 288.58 B | 137.54 B | -52.3% |
| Q11 wall | 89.53 s | 85.15 s | -4.9% |
| Q11 exact kernel | 39.11 s | 34.07 s | -12.9% |
| Q11 edges visited | 511.66 B | 274.92 B | -46.3% |

Both runs reported zero fallback events. Q11 remained bit-exact at
`1511054981`; Q10 keys, names, counts, and durations were exact, with four
averaged-distance cells differing by at most `1.01e-16`. Profiler schema 4
records y-bin and coverage-grid widths per prepared group. Dense concave,
hole, boundary, every compiled width, and coverage-memory-decline tests remain
exact against the Shapely oracle.
