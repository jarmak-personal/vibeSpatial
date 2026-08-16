# pylibcudf SF100 Execution Plan

<!-- DOC_HEADER:START
Scope: Execution plan for using pylibcudf and unified RMM memory management to make public vibeSpatial APIs exceed optimized GeoPandas by 10x on SpatialBench SF100.
Read If: You are changing native attribute execution, relational primitives, GeoParquet scans, GPU memory policy, or the SpatialBench public-API engines.
STOP IF: You only need an operation-local geometry kernel detail already routed by intake.
Source Of Truth: Program plan for the pylibcudf-backed SF100 performance push.
Body Budget: 420/420 lines
Document: docs/dev/pylibcudf-sf100-execution-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-13 | Intent |
| 14-22 | Request Signals |
| 23-38 | Open First |
| 39-47 | Risks |
| 48-67 | Mission And Success Contract |
| 68-80 | Constraints |
| 81-99 | Target Architecture |
| 100-113 | Backend Policy |
| 114-140 | Memory And Stream Contract |
| 141-159 | Required Evidence Ledger |
| 160-307 | Locked SF100 Baseline: 2026-08-13 |
| 308-323 | M0: Freeze Semantics And Measurement |
| 324-337 | M1: Make Allocation A Query Contract |
| ... | (6 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Use pylibcudf as the private tabular and relational engine beneath exact public
GeoPandas-compatible APIs. Keep geometry and spatial execution in vibeSpatial's
Native* carriers, share one RMM allocation domain, and materialize pandas only
at an explicit terminal compatibility boundary.

SpatialBench is the proof workload, not a source of production special cases.
The SQL queries define intent. Every optimization must be expressed as a
reusable physical-shape contract and exercised by an independent canary.

## Request Signals

- pylibcudf execution
- unified RMM allocator
- SF100 or SpatialBench
- device attributes and relational primitives
- GPU memory budget, contention, or fragmentation
- public-API performance

## Open First

- `docs/decisions/0042-device-native-result-boundary.md`
- `docs/decisions/0044-private-native-execution-substrate.md`
- `docs/dev/private-native-execution-substrate-plan.md`
- `docs/dev/native-format-library-plan.md`
- `docs/architecture/pylibcudf-capabilities.md`
- `docs/dev/pylibcudf-sf100-query-ledger.md`
- `src/vibespatial/cuda/_runtime.py`
- `src/vibespatial/api/_native_result_core.py`
- `src/vibespatial/api/_native_relation.py`
- `src/vibespatial/io/geoparquet.py`
- `src/vibespatial/io/pylibcudf.py`
- `../sedona-spatialbench/spatialbench-queries/geopandas_optimized_queries.py`
- `../sedona-spatialbench/spatialbench-queries/vibespatial_queries.py`

## Risks

- A second allocator pool can strand free VRAM and turn fragmentation into OOM.
- Immediate gather can multiply geometry and attributes before selectivity is known.
- A broad cuDF frame layer can repeat the rejected public-planner architecture.
- Arrow or pandas intermediates can conceal bulk D2H and synchronization.
- Suite totals can hide a failed or severely regressed individual query.
- Benchmark-specific branches do not generalize and are forbidden.

## Mission And Success Contract

Primary goal: at SF100, vibeSpatial's sum of median query wall times must be at
most 10% of optimized GeoPandas for the same twelve correct queries.
Q9 is exempt from per-query 10x because it is already sub-second; it remains in the suite-total, correctness, and 5% no-regression gates.

Measurement contract:

- convert legacy WKB Parquet to equivalent GeoParquet once, outside timing
- give both engines the same GeoParquet dataset and projected source columns
- use one isolated process per engine/query and one untimed warmup
- report the median of at least three measured runs per query
- include scan, compute, and public result construction in query time
- exclude one-time conversion and result serialization from query time
- verify all results against canonical SpatialBench answers before comparison
- publish per-query time, suite total, peak VRAM, D2H bytes, and fallback events
- do not hide slow or failed queries behind a geometric mean

Secondary gates: no correctness-admitted query may regress against its prior vibeSpatial baseline by more than 5%, and every changed native stage must retain its dedicated microbenchmark or pipeline canary.

## Constraints

- Benchmark query modules may import and call only public GeoPandas, pandas, or
  vibeSpatial APIs. They may not import private Native* or pylibcudf helpers.
- Public objects and results remain real GeoPandas/pandas-compatible objects.
- No public cuDF dataframe, lazy dataframe, or proxy Series API is introduced.
- No query-name checks, SF100 branches, precomputed answers, or benchmark-only
  production paths are permitted.
- CPU fallback remains exact and observable; strict-native rejects hidden host
  execution or materialization.
- Unknown pandas operations invalidate native state.
- Performance claims require end-to-end profiles, not dispatch counts.

## Target Architecture

The admitted hot path is:

```text
GeoParquet scan
  -> pylibcudf attribute columns + OwnedGeometryArray
  -> NativeFrameState
  -> NativeRowSet / NativeRelation / NativeGrouped / NativeExpression
  -> pylibcudf gather, filter, join, sort, top-k, groupby, reduction
  -> vibeSpatial spatial kernels and reusable index/metadata carriers
  -> small NativeFrameState or scalar result
  -> explicit public pandas/GeoPandas export
```

Row, candidate, pair, segment, group, and output-byte shapes stay distinct.
Public row alignment never forces an intermediate operation to become
pandas-row-shaped.

## Backend Policy

pylibcudf is the preferred internal backend because it exposes libcudf
primitives without replacing vibeSpatial's carriers or index semantics. Use
cuDF only as a development oracle or when a measured capability is unavailable
from pylibcudf and the cuDF call can consume and return the same RMM-owned
buffers without exposing cuDF publicly.

Use CCCL for compact/scan/sort/reduce shapes that pylibcudf does not express
efficiently. Use custom CUDA kernels for spatial work or fused physical shapes
whose intermediate cardinality would make generic tabular primitives
structurally too expensive. Record the rejected alternative and measurements
before adding a custom implementation.

## Memory And Stream Contract

- Install one RMM device resource before the first CuPy or pylibcudf
  allocation. Configure CuPy to allocate from that resource.
- Treat split CuPy/RMM pools as an initialization failure for native execution,
  not a silent fallback.
- Derive a per-query device budget from free VRAM minus an explicit reserve for
  the CUDA context, modules, library workspaces, and terminal output.
- Give every admitted physical operation a conservative peak-byte estimate
  covering inputs retained, scratch, candidate growth, and output capacity.
- Resize or shard work before launch when the estimate exceeds the remaining
  query budget. OOM callbacks are recovery rails, not planning.
- Track live bytes, pool-reserved bytes, high-water bytes, allocation count,
  D2H bytes, and the largest single allocation by stage.
- Preserve producer readiness when CuPy, cuda-python, and pylibcudf objects
  cross carrier boundaries. A buffer remains owned or borrowed until its
  completion event makes release safe.
- Avoid device-wide synchronization for ownership transfer. Use stream/event
  ordering and test per-thread default stream behavior.
- Keep pools hot between stages. Eager pool flushing is a diagnostic escape
  hatch because it synchronizes and increases allocator churn.
- Retain indirection (`row_positions`, relation pairs, group offsets) until a
  consumer requires physical gather. Do not duplicate large geometry buffers.
- Make spill or managed-memory migration explicit and measured. Prefer bounded
  shards and compact merge state; do not rely on opaque oversubscription to hit
  the primary performance target.

## Required Evidence Ledger

Create a checked-in SF100 query ledger before implementation. Each Q1-Q12 row
must record:

- SQL semantics and canonical output schema/order
- projected input columns and pushdown predicates
- physical shapes and their measured cardinalities
- current CPU, GPU, synchronization, allocation, and export stage times
- current peak live/reserved VRAM and largest allocation
- Native* producer/consumer chain and explicit export boundary
- intended pylibcudf, CCCL, or custom-kernel primitive
- conservative peak-byte formula and sharding strategy
- correctness oracle, dtype/null/index contracts, and fallback behavior
- before/after benchmark artifact paths

The ledger controls priority: fix the largest reusable wall-time shape first,
not the query with the most visible Python code.

## Locked SF100 Baseline: 2026-08-13

This is the initial before-state for the pylibcudf push. Preserve it in future
reports until a replacement baseline is captured under the same measurement
contract.

Environment and method:

- NVIDIA GeForce RTX 4090 with 24 GiB VRAM
- SpatialBench v0.1.0 SF100 converted to native GeoParquet
- 154 trip shards for Q3, Q4, and Q7
- one isolated process and one measured run for stage attribution
- low-overhead scoped wall timers plus runtime D2H/materialization/fallback
  counters and 100-250 ms `nvidia-smi` sampling
- clean query totals come from the uninstrumented benchmark artifacts
- Q12 stage attribution uses one- and ten-shard subsets; its full stage split is
  an inference, not a direct full-query stage trace

### Clean Query Totals

| Query | Optimized GeoPandas | vibeSpatial | Current speedup |
|---|---:|---:|---:|
| Q3 | 396.02 s | 252.36 s | 1.57x |
| Q4 | 224.60 s | 157.50 s | 1.43x |
| Q7 | 333.55 s | 174.60 s | 1.91x |
| Q12 | 591.28 s | 917.71 s | 0.64x |
| Four-query subtotal | 1,545.45 s | 1,502.17 s | 1.03x |

The broader seven-query run measured 1,761.73 s for optimized GeoPandas and
1,600.51 s for vibeSpatial, or 1.10x. These four queries account for about 94%
of that measured vibeSpatial subset and therefore control the first major
performance moves. This partial-suite number is a waypoint, not the final
twelve-query success contract.

### Stage Attribution

| Query | Profile wall | Stage | Time | Share |
|---|---:|---|---:|---:|
| Q3 | 260.68 s | GeoParquet read/public frame materialization | 217.75 s | 83.5% |
|  |  | point-to-polygon distance | 15.62 s | 6.0% |
|  |  | host selected-row take | 13.47 s | 5.2% |
|  |  | pandas grouped reduction | 0.43 s | 0.2% |
| Q4 | 159.93 s | trip read/public frame materialization | 107.03 s | 66.9% |
|  |  | host NumPy top-k partition | 30.12 s | 18.8% |
|  |  | pandas take/sort/head | 9.86 s | 6.2% |
|  |  | all 23 spatial joins | 4.33 s | 2.7% |
| Q7 | 174.59 s | GeoParquet read/public frame materialization | 127.14 s | 72.8% |
|  |  | host `t_distance` float conversion | 27.63 s | 15.8% |
|  |  | point-to-point distance | 6.84 s | 3.9% |
|  |  | host ratio/partition/frame construction | 5.20 s | 3.0% |
| Q12, 10 shards | 77.55 s | five-pass upper-bound calculation | 39.81 s | 51.3% |
|  |  | exact CPU GeoPandas KNN refinement | 34.40 s | 44.4% |
|  |  | all eleven GeoParquet reads | 1.86 s | 2.4% |
|  |  | building/grid setup | 1.27 s | 1.6% |

Q12's ten-shard upper-bound rate projects to roughly 613 s of the measured
917.71 s full run. After allowing about 25-30 s for full-scale scan/setup, the
remaining roughly 275-300 s is consistent with exact CPU refinement and final
selection. Re-measure this split directly when the benchmark profiler can
capture a full query without material overhead.

### Residency And Utilization

| Query | Avg/max GPU | Peak VRAM | Runtime D2H | D2H events |
|---|---:|---:|---:|---:|
| Q3 | 0.4% / 16% | 2,483 MiB | 8.40 GB | 616 |
| Q4 | 8.5% / 100% | 4,505 MiB | 3.74 MB | 1,501 |
| Q7 | 3.8% / 19% | 1,473 MiB | 12.00 GB | 1,078 |
| Q12, 10 shards | 1.1% / 27% | 1,249 MiB | 2.84 GB | 1,527 |

Q4's low D2H byte count does not imply device-native attributes: those columns
are already converted from Arrow to pandas during the timed public read. Its
high event count comes from small coordinate exports, relation exports, and
scalar fences. Successful profiles are well below total VRAM; the earlier Q12
GPU attempt still proved the need for admission control when one candidate
shape requested an additional 7.27 GiB allocation.

No library fallback events occurred in these profiles. Q12's exact refinement
constructs CPU GeoPandas objects explicitly in the benchmark query, so that
host path is outside vibeSpatial fallback telemetry. It remains part of timed
query execution and must be removed from the optimized engine.

### Source-Level Diagnosis

- Public `read_parquet` calls `payload.to_geodataframe()` for every shard in
  `src/vibespatial/io/geoparquet.py`.
- Compatibility assembly reaches `NativeAttributeTable.to_pandas()`, commonly
  `arrow_table.to_pandas()`, in `src/vibespatial/api/_native_result_core.py`.
- Q3 proves pandas groupby is not currently the bottleneck; scan, native rowset
  consumption, and selected attribute gather come first.
- Q4 proves spatial join optimization is not currently the bottleneck; native
  top-k and deferred gather come first.
- Q7 proves the distance kernel is not currently the bottleneck; preserving
  typed attributes and keeping ratio/top-k device-resident come first.
- Q12 repeatedly exports coordinates and performs five public distance passes
  in `_q12_upper_bounds`, then converts retained rows and buildings to CPU
  GeoPandas in `vibespatial_queries.py::_knn5_batch`.

### Milestone Measurement Hypotheses

- M1 should make memory behavior deterministic; it is not credited with speedup
  unless measured wall time also improves.
- M2 directly targets about 452 s of combined Q3/Q4/Q7 scan/materialization
  time. Preserve scan columns as a pylibcudf-backed `NativeAttributeTable`.
- M3 should eliminate Q3's host row take, Q7's typed attribute conversion, and
  repeated relation/row-position exports before terminal output.
- M4 should remove Q4's roughly 40 s host partition/take/sort path. Do not spend
  early effort replacing Q3's 0.43 s grouped reduction.
- M5 must redesign Q12's upper-bound and exact relation shapes; scan tuning
  alone cannot repair a query whose scan share is about 2%.
- Kernel-only wins that leave these composition stages intact cannot hit the
  program goal.

### Checkpoint Record

After every milestone, save raw artifacts under
`benchmark_results/spatialbench/sf100/<date>-<milestone>/` and append one
checkpoint to this section containing:

- code revision, GPU, CUDA/RAPIDS versions, and relevant environment settings
- correctness status and result artifact for every measured query
- clean median query times and comparison with this locked baseline
- named stage times using the same stage boundaries above
- GPU average/maximum utilization and peak live/reserved VRAM
- D2H count, bytes, and seconds; materialization and fallback counts
- largest allocation request, OOM/admission events, and shard cardinalities
- an explanation for any stage, transfer, memory, or correctness regression

### Active Checkpoint: 2026-08-14

| Shape | Evidence / status |
|---|---|
| Allocator / scan | CuPy and pylibcudf share one fail-closed RMM pool; metadata plans decoded bytes. A 3,896,103-row warm scan is 0.57-0.64 s plus 0.034 s public assembly. |
| Q1 / Q2 | Final medians are 12.39 s / 7.84 s versus GPD 106.03 s / 113.25 s. Q2 is exact at 53,348 after the short-edge tolerance/FP32 fix; an admitted 8M public point-index relation replaces a 4.0-GB predicate-mask export and reaches 14.45x. The rejected 32M relation requested one 12-GiB block. |
| Q3 / Q4 | Final medians are 12.86 s / 8.55 s versus GPD 405.03 s / 231.98 s, or 31.50x / 27.13x. Native selected gather and exact top-k replace host row materialization/full sort. |
| Q5 | Final median is 126.18 s versus GPD 834.69 s (6.62x): packed codes, standards-valid external GeoParquet partitions, grouped point hulls, and bounded top-k avoid a 44.44-GiB global sort. Public `to_parquet` supplies GeoParquet metadata because `to_arrow` carries only field-level GeoArrow metadata. |
| Q6 | Locked median is 21.13 s versus GPD 371.42 s (17.58x) at an admitted 8M target; a 16M plan exceeded remaining budget by 28.3 MiB. |
| Q7 | Preserving decimal-cast lineage and projecting the consumed secondary geometry keeps fused distance/ratio/top-k native. The final median is 4.23 s versus GPD 337.44 s (79.77x), and D2H is only 64,000 terminal bytes. |
| Q8 / Q9 | Final medians are 16.94 s / 0.14 s versus GPD 285.06 s / 0.19 s. Q9 is explicitly exempt from per-query 10x while retaining correctness, suite-total, and no-regression gates. |
| Q10 / Q11 | Final medians are 143.33 s / 268.90 s versus GPD 1,738.00 s / 3,127.50 s, or 12.13x / 11.63x. Exact per-part y directories and the cached 1024-square point grid replace 14.04B Morton scan lanes; five zone partitions retain 5.474 GB instead of 20.71 GB reserved by eager consolidation. |
| Q12 | Public exact k=5 plus Hilbert bounds has a final 21.28 s median versus GPD 626.64 s (29.45x); keys/order match GPD and metrics pass canonical tolerance. |
| Full suite / profile | The final warmup-plus-three median comparison is VS 643.77 s versus optimized GPD 8,177.23 s, or 12.702x. VS Q1-Q12 are 12.39, 7.84, 12.86, 8.55, 126.18, 21.13, 4.23, 16.94, 0.14, 143.33, 268.90, and 21.28 s. The post-push-fix Q4 median is 8.53 s with byte-identical output. The mandatory final full 1M pipeline profile has zero fallbacks, zero compute D2H/materializations, and no stage above 71.1 ms. The 2026-08-15 public shootout regression gates add 14/14 exact 10K workflows with zero fallbacks and summed medians of VS 2.614 s versus GPD 3.540 s. At 1M, nine current GPD-comparable workflows pass the fingerprint contract: eight have identical fingerprints, while accessibility is an unchanged tolerance match at 8,034 GPD rows versus 8,035 VS rows. The ten workflows with VS timings in both checkpoints improve from 205.09 s to 104.63 s, with no individual regression above 5%. Redevelopment, retail, site, and transit retain their pre-existing VS capacity failures; transit now passes correct grid admission and reaches a later 932 MiB overlay allocation instead of returning a false Morton subset. |
| Telemetry | Q1-Q12 process peaks are 12.01, 19.57, 12.62, 9.62, 16.21, 19.59, 17.78, 13.21, 2.24, 13.20, 13.20, and 13.34 GiB; every query has zero fallback events. Q2/Q7 are down to 0.37/0.06 MiB D2H. Q5/Q6/Q10/Q11 still explicitly export 9.63/13.43/19.99/13.14 GiB through public arithmetic or relation-result reducers; these are measured compatibility boundaries, not silent fallback, and a future reusable public relation-reducer contract must replace them rather than a private benchmark hook or broad planner. |
| Correctness | All twelve pass the committed SF1 oracle and the same-data SF100 GPD comparison at `rtol=1e-6`, `atol=1e-9`; Q10 differs only by admitted sub-ULP reduction order. SpatialBench has no committed SF100 answers. |
| Rejected shapes | 64M/16M admission failures, eager 32M grouped sort (44.44 GiB), eager zone-partition consolidation (20.71 GB reserved), duplicated polygon `take`, interval-fp32, combined Q11 endpoints, and one Morton interval per query bbox have the wrong shape. The latter scans 14.04B point positions to emit 28.07M pairs. A 0.1-degree zone-membership grid also loses: 14.1M memberships, 34.7M coarse pairs, and 15.82 s for only 2.23M exact pairs. |

Durable evidence is summarized here; raw public-shootout JSON is local, gitignored diagnostic output. Per-query semantics, physical shapes, memory formulas, and measurement evidence are checked in at `docs/dev/pylibcudf-sf100-query-ledger.md`.

## M0: Freeze Semantics And Measurement

- Derive each public implementation from the SQL rather than the legacy
  GeoPandas translation.
- Verify optimized GeoPandas and vibeSpatial against all canonical answers at
  SF1, then SF100.
- Capture cold and warm SF100 baselines in isolated processes.
- Add stage timing, D2H, fallback, RMM high-water, and allocation telemetry to
  benchmark artifacts.
- Populate the query evidence ledger and rank dominant reusable shapes.
- Confirm GeoParquet conversion preserves row count, CRS, geometry encoding,
  nulls, and non-geometry dtypes for every shard.

Exit: twelve correct queries, reproducible measurements, and no unexplained
time outside named stages.

## M1: Make Allocation A Query Contract

- Harden initialization so CuPy and pylibcudf demonstrably share one RMM
  resource.
- Add configurable reserve and query-budget calculation.
- Add shape-level peak estimators and budget admission to scan, index, relation,
  distance, sort, group, and constructive output stages.
- Add deterministic shard resizing before large allocation requests.
- Test fragmentation, concurrent CuPy/pylibcudf allocation, OOM recovery,
  carrier release, and cross-stream lifetime.

Exit: no unbudgeted large allocation, no split allocator state, and repeated
SF100 queries return to a stable live-byte baseline without eager trimming.

## M2: Preserve Device Tables From Scan

- Push projection, row-group pruning, and supported filters into pylibcudf
  GeoParquet reads.
- Attach the resulting pylibcudf table directly to `NativeAttributeTable`.
- Decode or adopt geometry on device and attach one coherent
  `NativeFrameState` with schema and index plan.
- Preserve chunk-reader device tables across sanctioned downstream consumers.
- Remove Arrow/pandas reconstruction from admitted scan-to-native paths.

Exit: scan canaries report zero bulk D2H and no pandas/Arrow attribute frame
between file input and the first native consumer.

## M3: Complete Rowset And Relation Consumption

- Make pylibcudf projection and gather consume `NativeRowSet` without host row
  positions.
- Keep deferred row indirection through consecutive filters and projections.
- Consume `NativeRelation` pairs directly for semijoin, antijoin, equality
  attribute filters, joined projection, and relation reductions.
- Reuse sorted/grouped relation metadata rather than sorting pairs repeatedly.
- Preserve exact duplicate, null, ordering, and `NativeIndexPlan` behavior.

Exit: admitted filter/join pipelines have zero intermediate public frame
assembly and zero bulk D2H before terminal export.

## M4: Complete Tabular Analytics

- Audit the installed pylibcudf surface and add guarded adapters for stable
  sort, bounded top-k, hash join, groupby, and required reductions.
- Keep numeric, boolean, string, categorical, datetime, and nullable contracts
  explicit; movement support does not imply compute support.
- Lower grouped keys and aggregations through `NativeGrouped` and preserve
  output index semantics.
- Merge shard-local top-k and grouped states on device with bounded memory.
- Add CCCL/custom implementations only when profiles prove the generic
  primitive has the wrong physical shape.

Exit: no admitted SpatialBench aggregation, ordering, or bounded selection is
dominated by pandas execution or a full intermediate result materialization.

## M5: Bound Large Physical Shapes

- Add row-group/shard planning that uses metadata and the query budget.
- Bound spatial candidate generation before exact predicates or distance.
- Reuse `NativeSpatialIndex` and `NativeGeometryMetadata` across shards when
  lineage and parameters remain valid.
- Stream relation reductions so full pair sets need not coexist in memory.
- Keep only compact global state for top-k, grouped reducers, and scalar
  aggregates.
- Treat unexpectedly dense relations as an observable plan decision with an
  exact bounded alternative.

Exit: every SF100 query completes without allocator thrash, opaque spill, or a
single allocation capable of exhausting the query budget.

## M6: Close The SF100 Gap

- Work down the evidence ledger in descending wall-time order.
- Require a reusable shape canary and public correctness test for every change.
- Re-run the full suite after each material milestone and reject cross-query
  regressions above the stated gate.
- Remove transitional host helpers from admitted paths once their native
  consumers are complete.
- Run the mandatory full pipeline profile and investigate every disproportionate
  or CPU-heavy stage.
- Publish hardware, software versions, environment, per-query distributions,
  suite totals, speedups, peak VRAM, transfers, and fallbacks.

Exit: all twelve answers are correct and the measured SF100 suite satisfies the
10x primary goal under the frozen measurement contract.

## Verify

- `uv run ruff check`
- `uv run pytest tests/test_gpu_memory_pool.py -q`
- `uv run pytest tests/test_pylibcudf_capabilities.py -q`
- `uv run pytest tests/test_private_native_substrate.py -q`
- `uv run pytest tests/test_strict_native_mode.py -q`
- `uv run pytest tests/test_pipeline_benchmarks.py -k "native or relation or grouped" -q`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`
- `uv run python scripts/check_docs.py --check`
- `python ../sedona-spatialbench/benchmark/run_benchmark.py --data-dir <sf100-geoparquet> --engines geopandas_optimized,vibespatial --scale-factor 100 --warmup-runs 1 --runs 3 --statistic median --timeout 3600 --result-dir <results>`
