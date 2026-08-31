# vibeSpatial SF1000 optimization investigation and implementation

Date: 2026-08-30
Updated: 2026-08-31

The original measurements below were collected before implementation changes. The y-bin experiments varied one physical index parameter before NVRTC compilation; public APIs, query plans, and query semantics remained unchanged during that investigation. The implementation outcome is recorded separately below.

All comparative measurements below were collected in the same environment. One-shard probes are directional engineering measurements rather than release-grade full-dataset trials. Tracked D2H covers copies reported by the vibeSpatial runtime and does not include any transfers internal to the storage transport. Before these measurements are used as landing or publication evidence, preserve a durable identity packet containing the source revision and worktree diff, device/driver/allocator state, dataset and shard fingerprints, commands, warmup/repeat policy, measurement boundaries, raw timings, and output hashes.

## Bottom line

1. **Q10/Q11 have an immediately actionable engine improvement.** Expanding the reusable polygon part-y edge directory from 8 to 64 bins reduced ten-shard SF1000 Q10 from 21.06s to 7.77s and Q11 from 27.91s to 9.02s. At 128 bins they reached 6.63s and 7.47s. Outputs were identical at every tested width from 8 through 256.
2. **Q5's warm SF100 path is mostly scan, host externalization, and frame/index orchestration—not convex hull compute.** A device-native partition-clustered Parquet spill was faster than the current host partition writer in a shard-scale proof and eliminated tracked D2H.
3. **The original WKB Parquet can be made standard GeoParquet without CPU geometry conversion.** A full 12-column shard was rewritten as WKB GeoParquet in 0.74s with zero tracked D2H and byte-identical WKB in both geometry columns.
4. **pylibcudf execution and physical GDS are separate claims.** These probes establish device-native Parquet decode/encode without Python or pandas materialization, but they do not establish direct GPU-to-storage DMA.

## Implementation outcome (2026-08-31)

- The prepared part-Y directory now compiles and caches uniform
  `8/16/32/64/128/256` variants. Reported VRAM selects the first attempted
  tier (`8/16/32/64/128/256` at the documented class boundaries), nominal
  capacities snap within five percent, and concrete peak-memory admission
  descends before immutable publication. A 48 GiB device therefore attempts
  128 first; an 8 GiB device retains a first-class 8-bin path.
- The width-scaled per-part scatter cursor was replaced by an
  edge/bin-membership count/scan/scatter builder. Width participates in NVRTC,
  warmup, prepared-cache, readiness, and telemetry identities.
- Full SF100 profiling showed exact traversal still dominant enough to trigger
  the conditional next step. The implemented conservative part coverage grid
  reduced Q10 edge visits by 52.3% and Q11 by 46.3%; measured walls improved
  from 59.38s to 56.13s and 89.53s to 85.15s, respectively. Certified cells are
  exact fp64 interior/exterior states; edge-touched or invalid cells use the
  existing exact predicate.
- `NativePartitionedParquetSink` is wired through public
  `write_geoparquet` batch iteration into Q5's large-domain plan. It clusters
  rows on device, writes one persistent file with bounded homogeneous row
  groups, and routes ordinary partition equality reads through its manifest.
  The SF100 native branch was exact but slower (46.70s) than the existing
  host-clustered SF100 plan (23.21s cold with telemetry), so a measured
  group-domain crossover keeps SF100 on the faster plan and selects the native
  bounded spill only once the dense group domain reaches one billion slots.
  This is a workload-shape tier, not a predicted-time or GPU-name model.
- The final native Q5 branch uses WKB. A tested GeoArrow alternative reached
  44.99s but transferred 5.60 GB to host, so it was rejected despite the small
  wall-time gain. Empty pushed-down WKB results now remain device-native.
- The Q5 orchestration audit is closed: sanctioned `assign` and `reset_index`
  transitions retain private native state, device-index reset stays zero-D2H,
  and the reconstruction interval is split into prototype and device-index
  materialization stages. The final warm SF100 public plan completed in 17.54s.
- Q11 now retains an owner-local capacity workspace across batches. Point-grid
  candidate scatter writes directly into its left/right row buffers; exact
  locations, packed keys, radix outputs, dedup masks, and reduction counts are
  reused with stream readiness. A strict SF100 timing pass completed in 84.87s,
  returned `1511054981` exactly, and reported zero fallbacks. Across 770 calls
  per side, exact classification consumed 17.33/17.27s, candidate generation
  1.68/1.66s, parent-key construction 0.15/0.14s, radix sort 0.08/0.08s,
  deduplication 0.10/0.10s, and pair intersection 0.75s total. The evidence does
  not justify replacing the radix reducer.

All final SF100 Q5/Q10/Q11 outputs match the frozen oracle within its numeric
tolerances and report zero fallbacks. No product implementation is deferred.
The only unavailable evidence is a fresh full SF1000 rerun, which requires a
machine with the dataset and storage capacity; no SF1000 result is inferred
from the SF100 measurements.

## Baseline evidence

| Workload | SF1000 wall time | Main evidence |
|---|---:|---|
| WKB → native-GeoArrow conversion | 11,184.71s (3h 06m 25s) | PyArrow → pandas → GeoPandas WKB decode → GeoArrow/Zstd, four CPU workers |
| Q5 | 4,588.72s (76m 29s) | 77.90 GB tracked D2H; 6.35 TB cumulative RMM allocation |
| Q10 | 3,025.27s (50m 25s) | 15.36 TB cumulative RMM allocation; 1.77M allocations |
| Q11 | 4,124.18s (68m 44s) | 22.83 TB cumulative RMM allocation; 1.30M allocations |

At SF100, exact point-region refinement accounts for 83.63s of Q10's 120.88s (69.2%) and 145.51s of Q11's 196.15s (74.2%). Q10 visits 2.28 trillion polygon edges and Q11 visits 4.03 trillion.

## Q10/Q11: where the superlinear scaling comes from

The zone datasets have almost identical row counts: 1,033,509 at SF100 and 1,033,728 at SF1000. SF1000 has only 5.4% more polygon parts and 1.9% more prepared edge memberships. The trip-point distribution is materially harder, however.

Ten evenly spaced shards from each scale produced:

| Metric | Q10 SF100 | Q10 SF1000 | Ratio | Q11 SF100 | Q11 SF1000 | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Wall time | 11.40s | 23.51s | 2.06× | 16.30s | 30.38s | 1.86× |
| Candidate lanes | 242.75M | 324.66M | 1.34× | 461.23M | 569.99M | 1.24× |
| Parts considered | 52.23B | 221.47B | 4.24× | 461.23M | 624.80M | 1.35× |
| Edges visited | 147.85B | 804.29B | 5.44× | 261.74B | 1.071T | 4.09× |
| Exact-kernel time | 5.37s | 17.30s | 3.22× | 9.36s | 23.18s | 2.48× |
| Cumulative allocation | 106.8 GB | 122.8 GB | 1.15× | 146.5 GB | 163.4 GB | 1.12× |

The whole SF1000 query has about ten times as many shards and each sampled shard takes roughly twice as long. Multiplying the shard count by the sampled wall ratios predicts about 20.6× Q10 and 18.6× Q11 growth, versus the observed 25.0× and 21.0×. The measured work amplification therefore explains most, but not all, of the superlinear growth; the remaining full-run overhead requires aggregate evidence rather than extrapolation.

### Wider prepared y-edge directory

The current index assigns every polygon edge to one or more of eight y bins per polygon part. Point-in-polygon refinement then visits only the selected bin. The tested variant changes only that physical bin width.

SF1000 ten-shard results:

| Bins per part | Persistent index | Q10 | Q10 vs 8 | Q11 | Q11 vs 8 |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.51 GiB | 21.06s | baseline | 27.91s | baseline |
| 16 | 1.73 GiB | 13.71s | 1.54× | 17.37s | 1.61× |
| 32 | 2.17 GiB | 9.92s | 2.12× | 12.11s | 2.31× |
| 64 | 3.04 GiB | 7.77s | 2.71× | 9.02s | 3.09× |
| 128 | 4.78 GiB | 6.63s | 3.18× | 7.47s | 3.74× |
| 256 | 8.26 GiB | 6.17s | 3.42× | 6.73s | 4.15× |

All six configurations produced the same complete Q10 and Q11 result hashes. A separate SF100 ten-shard check also held exactly: 64 bins improved Q10 from 9.07s to 5.33s (1.70×) and Q11 from 13.81s to 6.98s (1.98×).

The curve has the expected shape: more bins continue to improve refinement, but performance returns diminish while VRAM cost accelerates. More bins reject additional irrelevant edges, but the remaining edges increasingly span several bins or still require exact evaluation. At the same time, every part gains more count/offset slots and long edges are duplicated into more membership lists. For `P` polygon parts, `B` bins, and `M(B)` edge memberships, the current carrier occupies approximately:

```text
index_bytes(B) = 16P + 12PB + 4M(B)
```

The `12PB` count/offset term grows directly with bin width, while `M(B)` also grows as edges cross additional bins. This explains why 128→256 saves much less time than 64→128 while adding 3.48 GiB in this workload.

This is not a Q10/Q11 specialization. It improves the existing reusable exact point-in-region index for any sufficiently large polygon side. The production implementation should compile and cache a bounded family of uniform 8/16/32/64/128/256-bin variants rather than hard-code one width. `PreparedPolygonPartYIndex` already carries `bin_count`; the width must also become part of the NVRTC source/cache identity, warmup request, prepared-index cache identity, and telemetry.

The current builder assigns one thread to each polygon part, serially traverses its rings and edges, and keeps a width-sized cursor array per thread during scatter. That physical shape becomes increasingly hostile to occupancy and local memory at 128 or 256 bins. Production wider-bin work therefore includes an edge/bin-membership-shaped count/scan/scatter builder; it is not only a selector change. Prepared publication must remain immutable, atomic, and completion-ready for cross-stream consumers.

### VRAM-class selector with capacity decline

The first production selector should be deliberately simple. It should not attempt to predict wall time, infer future workflow reuse, run online comparisons, or key policy by GPU product name. Once the existing workload gate decides that a prepared part-y index is worthwhile, driver-reported total device memory selects an aggressive target width:

| Nominal device VRAM | First attempted width |
|---:|---:|
| 8 GB or less | 8 |
| More than 8 GB through 16 GB | 16 |
| More than 16 GB but less than 24 GB | 32 |
| 24 GB but less than 48 GB | 64 |
| 48 GB but less than 100 GB | 128 |
| 100 GB or more | 256 |

The implementation must normalize reported bytes into tolerant nominal classes. An advertised 8, 24, 48, or 100 GB device can report slightly less usable memory and must not fall into the next lower class because of allocator or binary/decimal accounting. The thresholds remain internal policy constants backed by cross-device evidence, not public tuning API.

The 48 GB class intentionally attempts 128 first. In this sweep, 64→128 reduced Q10 by 14.7% and Q11 by 17.2% for 1.74 GiB of additional persistent index memory. The 128→256 step still reduced their combined time by about 8.5% for another 3.48 GiB, so 256 should also be tested on 48 GB devices with substantial observed headroom; 128 is the initial policy, not a permanent ceiling. Likewise, 8 bins is a first-class constrained-device tier so an 8 GB consumer GPU can retain bounded GPU acceleration.

Before committing the target index, the operation must account for the complete simultaneously-live envelope: input carriers, persistent index, index-build temporaries, candidate relation, exact-refinement and reduction scratch, other prepared state, outputs, and a safety margin. The remaining amount becomes the index budget:

```text
index_budget = available_device_memory
             - required_input_and_output
             - candidate_and_refinement_scratch
             - other_persistent_state
             - safety_margin
```

Starting from the VRAM-class target, admission descends deterministically through smaller compiled tiers until the full envelope fits. If even 8 bins cannot fit, execution retains the exact GPU baseline with bounded candidate/refinement tiles; it must not silently fall back to CPU or rely on managed-memory oversubscription. Observed free memory is advisory rather than a reservation, so the first implementation should use conservative operation-private peak accounting, bounded construction, and actual allocation before prepared state is published.

The selector exports nominal capacity class, target width, admitted width, decline reason, exact membership count, persistent and peak-build bytes, cache hits, and profiling-only refinement edge visits. This provides evidence for changing the class ladder without introducing a generic analytic cost model or a library-wide lifecycle controller.

A practical implementation can initially select one uniform width per prepared polygon index. The longer-term representation may allow a variable bin count per polygon part: small/simple parts keep a narrow directory, while heavy-tail parts responsible for tens of thousands of edge visits receive wider directories. That should capture more high-memory-device benefit without paying the maximum slot cost for all 1.37 million parts.

Wider bins did leave point-region refinement dominant after the full SF100 run,
so the reusable conservative coverage grid described above is now implemented.
It certifies cell/part pairs as interior or exterior and sends every
boundary/ambiguous cell through the exact part-Y kernel. Invalid or unsupported
geometry declines to the existing path. The previously rejected point-side
adaptive quadtree is not evidence against this region-side prepared classifier.

Q11 also needs stage-separated CUDA timing around classification, parent-key construction, radix sort, deduplication, and pair intersection. The current `component_parent.key_sort` interval contains queued classification work, so it cannot yet justify a sort rewrite. After measurement, a general segmented parent-match reducer could avoid global packed-key sorting when candidate topology is already point-major.

## Q5: where the time goes

The saved warm in-process SF100 profile completes in 17.92s. Cumulative timings overlap because CUDA work is asynchronous, but the expensive surfaces are clear:

| Surface | Calls | Cumulative time | Observation |
|---|---:|---:|---|
| PyArrow partitioned dataset write | 20 | 4.67s | Host externalization; 640 output files |
| pylibcudf GeoParquet scan | 72 | 4.08s | Necessary scan/decode work |
| Frame/index reconstruction interval (`RangeIndex`/`numpy.arange`) | 582 | 2.11s | Split constant-time index construction from array materialization and validation before optimizing |
| Native frame → Arrow | 20 | 1.76s | Host boundary before spill |
| Runtime deferred-owner retirement | 328 | 1.12s | Repeated small-batch lifecycle overhead |
| Dissolve preparation/execution | 32 | 1.03s | Warm hull compute itself is only 0.12s |
| Temporary-file deletion | 640 | 0.49s | Many-file spill layout overhead |

The cold profile additionally pays about 4.4s for one-time CCCL segmented-sort compilation. Dense-count updates are only about 0.20–0.26s at SF100; replacing that kernel is not the priority.

### Native clustered spill proof

One 3.9M-row shard was partitioned 32 ways by customer key:

| Spill shape | Write | Bytes | Tracked D2H | Readback |
|---|---:|---:|---:|---:|
| Current host Arrow directory dataset | 0.174s | 133.0 MB | 31.2 MB minimum | not measured |
| 32 independent pylibcudf sinks | 1.216s | 105.8 MB | 0 | not measured |
| One device-partitioned, clustered-row-group file | 0.132s | 108.3 MB | 0 | 0.190s for all 32 filters |

The clustered file read back exactly 3,896,103 rows across its 32 filter-pushed partitions. Row-total equality alone cannot detect duplicate, omitted, or mispartitioned records, and the single timing comparison is directional. Production validation must use repeated measurements under the same compression, cache, and durability contract plus per-partition row/key/attribute/null/WKB fingerprints. The negative independent-sink result still matters: replacing PyArrow with many short-lived device writers is slower. The general primitive should instead be a `NativePartitionedParquetSink`-style internal carrier that:

1. computes the partition map with `pylibcudf.partitioning.partition` or `hash_partition`;
2. clusters rows by partition on device;
3. writes bounded row groups through one or a small number of persistent `ChunkedParquetWriter`s;
4. records partition-to-file/row-group metadata; and
5. reads selected partitions with pylibcudf projection/filter pushdown.

WKB is a good internal spill encoding today. A single native WKB GeoParquet write took 0.177s, transferred only a 24-byte planning packet to host, and was only 2.2% larger than the host GeoArrow file. The current public native-GeoArrow write of a partitioned geometry composition is not ready for this role: it took 2.43s and copied 245 MB to host. That path should either consume the composition directly or deliberately choose native WKB.

A second, algorithmically different Q5 direction is a mergeable grouped convex-hull reducer. Because `hull(A ∪ B) = hull(hull(A) ∪ hull(B))`, shard-local group hulls can be merged hierarchically instead of externalizing every qualifying point. This is exact and general, but it needs a spill-volume crossover study before implementation; low-cardinality groups may not shrink enough per shard.

## Native WKB input and conversion

The existing original-WKB benchmark control scans through PyArrow, annotates Arrow fields on host, then calls `GeoDataFrame.from_arrow`. The production I/O stack already owns pylibcudf GeoParquet scanning, the device WKB decoder, native WKB GeoParquet writing, and a persistent `ChunkedParquetWriter`. The missing work is a metadata-only legacy-WKB-Parquet → WKB-GeoParquet transcode and benchmark preparation path that compose those existing primitives; it is not a second scanner/writer stack.

For two geometry columns from one 3.9M-row shard:

| Read path | Warm | Cache-evicted |
|---|---:|---:|
| Current host PyArrow WKB → public frame | 0.258s | 0.274s |
| Direct pylibcudf WKB scan | 0.028s | 0.046s |
| Direct pylibcudf WKB scan + GPU decode | 0.032s | 0.048s |
| Direct pylibcudf native-GeoArrow scan | 0.015s | 0.033s |
| Full public native-GeoArrow frame read | 0.138s | 0.155s |

The source WKB format is therefore not intrinsically slow. The host scanner/frame bridge is.

A full 12-column source shard was also rewritten with pylibcudf as standards-compliant GeoParquet 1.1 using WKB geometry encoding:

- 3,896,103 rows
- 256.95 MB source → 254.42 MB output
- 0.737s scan + Snappy rewrite
- zero tracked D2H
- public `vibespatial.read_parquet` readback: 0.228s
- every pickup and dropoff WKB value byte-identical across the full shard

This is the recommended fast **step 0** when a standard GeoParquet artifact is desired but native GeoArrow conversion is not worth its cost. It only needs declared geometry columns, CRS, and optional geometry-family metadata; it does not parse geometry on CPU. Production validation must additionally cover all attributes and dtypes, null masks, row order, both geometry metadata entries, primary geometry designation, CRS, and independent GeoParquet metadata validation. Extrapolating one cached shard is not a replacement for a full SF1000 run, but it establishes the correct architecture and suggests a much smaller conversion envelope than the current 3h06m CPU path.

## Device-native I/O versus physical GDS

There are two different claims to track:

- **Logical device-native:** Parquet decode/encode and geometry processing stay in device columns without Python/pandas materialization. The direct scanner, WKB transcode, native WKB writer, and clustered spill proofs demonstrate this.
- **Physical GPU↔storage DMA:** cuFile transfers directly between GPU memory and supported storage without compatibility-mode host bounce buffers. That requires separate transport-level validation.

vS should expose transport provenance separately from scan backend: for example, `pylibcudf+cufile-gds`, `pylibcudf+cufile-compat`, or `pyarrow-host`. A pylibcudf backend label or successful `SourceInfo(path)` call alone does not prove GDS; direct mode should require supporting cuFile logs or counters.

## Validation environment constraint

The current development workstation does not have storage capacity for a fresh SF1000 dataset or full SF1000 rerun. Local implementation and acceptance gates therefore use SF1 and full SF100, including complete public Q5/Q10/Q11 stages and device-memory envelopes. A fresh SF1000 run is a separate validation step on a machine with the required dataset and storage capacity. It is required before publishing new full-SF1000 performance claims, but no product work is postponed behind it.

## Recommended implementation order

1. **Evidence freeze (complete):** preserve the report's identity packet and result hashes before changing the measured path.
2. **VRAM-class prepared part-y index (complete):** compile all six variants, use the edge-shaped builder, and include width in cache/readiness/telemetry identity.
3. **Full local Q10/Q11 validation (complete):** constrained, 48 GiB, and simulated high-memory policies are covered; full SF100 outputs, memory, fallback, and physical-work evidence are recorded above. Run SF1000 separately without extrapolation.
4. **Metadata-only WKB-GeoParquet transcode (complete):** the existing scan, GPU WKB, native attributes, metadata, and persistent writer are composed without a duplicate scanner stack.
5. **Partition-clustered native spill (complete):** the generalized internal primitive is wired into the large-domain public Q5 plan; the 32-independent-sink design remains rejected.
6. **Q5 orchestration cleanup (complete):** sanctioned assign/reset operations preserve native state, device-index reset remains zero-D2H, and reconstruction stages are separately attributable.
7. **Persistent point-region workspace and Q11 stage timing (complete):** candidate/sort/reduction buffers are owner-local and reusable; separated SF100 evidence leaves exact classification dominant, so the radix reducer remains.
8. **Prepared coverage grid (complete):** full-SF100 evidence triggered it; the exact conservative classifier and decline path are implemented and measured above.
9. **GDS validation:** rerun cold read, write, and overlapped probes on a supported storage stack; do not claim direct GDS without transport-level evidence.

Every engine change must select from reusable physical-work eligibility, reported resource capacity, and concrete allocation admission. No query number, SpatialBench column name, GPU product name, or benchmark identity belongs in the implementation.
