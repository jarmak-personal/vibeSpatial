# SF100 Cross-Device Optimization Audit

<!-- DOC_HEADER:START
Scope: Per-query RTX 4090 and H200 SF100 diagnosis and reusable public-API optimization paths.
Read If: You are selecting, designing, profiling, or reviewing SF100-driven performance work across IO, spatial reductions, point location, grouped geometry, or nearest search.
STOP IF: You only need the consolidated benchmark numbers or an operation-local implementation detail already routed by intake.
Source Of Truth: Current cross-device interpretation and prioritized optimization audit; raw Nsight and benchmark artifacts remain authoritative measurements.
Body Budget: 445/460 lines
Document: docs/dev/sf100-cross-device-optimization-audit.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-17 | Request Signals |
| 18-25 | Open First |
| 26-32 | Verify |
| 33-44 | Risks |
| 45-61 | Evidence Contract |
| 62-83 | Query Summary |
| 84-369 | Query Audits |
| 370-398 | Cross-Query Program |
| 399-417 | Required Efficiency Profiles |
| 418-430 | Acceptance Rails |
| 431-445 | Evidence |
DOC_HEADER:END -->

## Intent

Explain what the RTX 4090 and H200 evidence says about every SF100 query and
identify reusable vibeSpatial performance work. This is an optimization audit,
not a claim that GPU busy time proves hardware efficiency.

## Request Signals

- SF100 query optimization
- RTX 4090 versus H200
- GPU utilization
- SpatialBench profiling
- point-in-polygon performance
- public API physical plans

## Open First

- `docs/dev/cross-device-performance-report.md`
- `docs/dev/point-region-execution-evidence.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `docs/decisions/0044-private-native-execution-substrate.md`
- `docs/testing/spatialbench-nsight.md`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "SF100 cross-device optimization audit"`
- `(cd benchmark_results/nsight/sf100/2026-08-19-rtx4090-comparable && sha256sum --check SHA256SUMS)`
- `(cd benchmark_results/nsight/sf100/2026-08-20-h200-comparable/nsight-sf100 && sha256sum --check SHA256SUMS)`

## Risks

- GPU busy fraction measures occupied time, not useful work per byte, edge, or
  candidate.
- Nsight Systems kernel sums expose device work but not occupancy, cache
  efficiency, memory bandwidth, or instruction efficiency.
- The H200 profile wall is perturbed by a shared CPU quota and FUSE storage.
  Clean wall timings, not H200 trace wall, are the end-to-end comparison.
- Summed kernel duration may overlap across streams and is not additive to wall.
- A query-specific rewrite is not acceptable unless it becomes an admitted
  public physical shape under ADR-0046.

## Evidence Contract

The clean timings are the same public implementation and 38.096 GB GeoParquet
dataset on both machines. The 4090 host is an i9-13900K with local NVMe. The
H200 host had 12 vCPUs and network/FUSE-backed storage. Both returned 12/12
correct answers. Cross-machine clean timings are directional because each is
one warmup plus one measured run.

The kernel columns below come from matched Nsight Systems captures. They are
useful for separating device work from scan and orchestration, but the H200
trace wall is not a clean benchmark. Local CPU sampling was available; the
H200 host fixed `perf_event_paranoid=4`, so that trace has no CPU call chains.

`H200 kernel gain` is 4090 summed kernel time divided by H200 summed kernel
time. A value above one means the same captured device work was faster on H200.
`Clean winner` compares end-to-end clean wall.

## Query Summary

| Query | Clean wall: 4090 / H200 | Kernel sum: 4090 / H200 | H200 kernel gain | Clean winner | Dominant observed shape | Primary reusable path |
|---|---:|---:|---:|---|---|---|
| Q1 | 12.43 / 23.75 s | 1.072 / 1.095 s | 0.98x | 4090 1.91x | multi-file decode and host orchestration | persistent active-source scan plus streamed top-k |
| Q2 | 7.82 / 12.15 s | 4.299 / 3.146 s | 1.37x | 4090 1.55x | repeated point-to-one-polygon count | relation-free count plus reuse-aware polygon preparation |
| Q3 | 12.73 / 23.00 s | 1.189 / 0.843 s | 1.41x | 4090 1.81x | scan, host uniqueness, then grouped scalar expressions | pipelined scan and fused filtered grouped reduction |
| Q4 | 8.49 / 9.88 s | 5.728 / 4.375 s | 1.31x | 4090 1.16x | index/preparation work for only 1,000 points | candidate-work admission and edge-shaped index build |
| Q5 | 17.46 / 31.95 s | 3.087 / 2.782 s | 1.11x | 4090 1.83x | two-pass grouping plus Parquet spill and allocation | persistent dense group state and native bounded grouped hull |
| Q6 | 16.25 / 21.53 s | 8.504 / 7.937 s | 1.07x | 4090 1.33x | expensive polygon/box prefilter plus point-zone reduction | bounds-first many-few filter and fused weighted relation reduction |
| Q7 | 4.20 / 6.64 s | 1.608 / 1.305 s | 1.23x | 4090 1.58x | scan/decode around a cheap row expression/top-k | streamed scan-to-expression-to-top-k |
| Q8 | 16.95 / 11.07 s | 13.917 / 5.248 s | 2.65x | H200 1.53x | count then scatter the same Morton candidates | direct `dwithin` count; do not materialize pairs |
| Q9 | 0.15 / 0.19 s | 0.086 / 0.090 s | 0.96x | 4090 1.27x | fixed/process overhead around a tiny overlay | preserve as correctness and launch-overhead canary |
| Q10 | 125.84 / 68.58 s | 111.802 / 40.505 s | 2.76x | H200 1.83x | trillions-scale prepared point-location traversal | reduce edge visits, then fuse predicate and grouped sums |
| Q11 | 237.59 / 107.16 s | 223.134 / 77.662 s | 2.87x | H200 2.22x | prepared point-location plus aligned endpoint reductions | better point-location index and dual-endpoint fused reduction |
| Q12 | 21.15 / 22.14 s | 14.608 / 6.105 s | 2.39x | effectively tied | global radix/sort and exported fixed-k relation | fixed-k nearest aggregate with segmented local selection |

The table rejects a product-name planner. H200 is much faster at long exact and
sort kernels, yet loses or ties eight end-to-end queries because their device
work is too small or surrounded by host and storage work. Conversely, Q10 and
Q11 can keep a GPU busy while still doing far too many exact edge visits.

## Query Audits

### Q1: Sedona-Center Distance Top 100

**Observed.** Only 1.07 seconds of 12.43-second local wall is kernels; ZSTD
decode is 63% of that kernel sum. The trace moves 17.59 GB H2D. Local CPU
samples are led by numeric uniqueness/hash, FNV hashing, and memory copies.
The multi-file scan emits 6,160 KvikIO file opens over 20 read calls: the 154
sources are reopened for every chunk. `_iter_geoparquet_native_impl` passes the
complete source list, including empty row-group selections, into a newly built
`SourceInfo` for each chunk.

**Interpretation.** More distance-kernel throughput cannot materially change
the query. The physical unit is a row-group stream ending in a 100-row bounded
state, not a sequence of complete public frames.

**vS path.** Compact every chunk to active sources and aligned nonempty
row-group lists. Then introduce a persistent multi-source reader/open-handle
cache and two-buffer scan/decode/compute overlap. Keep distance, predicate, and
shard top-k as native expressions and merge only the 100-row states. This is a
general GeoParquet streaming shape, not a Q1 path.

**Proof.** Track opens per source, read calls, decoded bytes, overlap, kernel
sum, terminal D2H, and clean wall on local NVMe and network storage.

### Q2: Coconino Pickup Count

**Observed.** Device work is material: 4.30 seconds locally. The compacted
point-in-polygon kernel is 1.125 seconds on 4090 but only about 0.156 seconds on
H200, a 7.19x kernel-specific gain. The current public query asks
`sindex.query(...).size`, so it constructs a relation when only one scalar count
is required. Preparation admission is driven mainly by polygon coordinate
count, not candidate-edge work or repeated query reuse.

**Interpretation.** There are two inefficiencies: unnecessary relation output
and a preparation decision that does not model one-polygon reuse. The H200
kernel win is hidden by scan/host costs, but it proves the exact work itself is
architecture-sensitive.

**vS path.** Lower public `query_aggregate(target, {"count": "size"},
predicate="intersects")` directly to a scalar device reduction. Admit prepared
point location using estimated candidates, selected-bin edges, polygon reuse,
and build amortization rather than a fixed coordinate threshold. Reuse the
prepared polygon across all streamed point batches.

**Proof.** Require zero `NativeRelation` pair capacity, one persistent prepared
index, exact boundary semantics, and shape floors spanning small/large polygons
and sparse/dense point batches on both GPUs.

### Q3: Monthly Buffered-Box Statistics

**Observed.** Kernel time is only 9% of clean local wall. ZSTD is the largest
single kernel. CPU samples resemble Q1: uniqueness/hash, FNV hashing, copies,
and file input. H200 kernels are 1.41x faster while clean wall is 1.81x slower.

**Interpretation.** This is a streaming relational expression problem with a
spatial filter, not primarily a geometry-kernel problem. Repeated column/index
normalization around chunks is more important than raw distance throughput.

**vS path.** Use the Q1 active-source/persistent reader. Lower distance-to-box,
month extraction, duration, filter, and group sums into one admitted native
streaming group state. Merge the 84 month states on device. Add spatial
row-group bbox pruning where metadata certifies it.

**Proof.** Count public frame assemblies, uniqueness/hash work, active sources,
row groups pruned, group-state bytes, and terminal exports. The result remains
an ordinary public DataFrame.

### Q4: Zone Distribution Of Top Tips

**Observed.** The SQL plan correctly reduces trips to 1,000 points before the
zone join, yet kernels consume 5.73 seconds. No exact kernel dominates. Morton
count/scatter, part-Y count/scatter, and ZSTD preparation are collectively the
cost; the final prepared PIP launches have grid size one. H200 improves total
kernels only 1.31x.

**Interpretation.** vS is paying large-zone index construction costs for a tiny
query side. The current part-Y builder assigns one thread to each polygon part
and serially scans every ring/edge, which is the wrong physical shape for long
parts. Preparation may not amortize for only 1,000 points.

**vS path.** Base admission on predicted candidate-edge visits and reuse. A
direct bounds/candidate/refine path should win when query rows are tiny; a
prepared path remains available when reuse amortizes its build. Rebuild the
part-Y index as edge-count, scan, and scatter work so large parts expose
edge-shaped parallelism. Preserve the top-k rowset as a native point carrier
instead of reconstructing public points before the join.

**Proof.** Separate build from query time. Sweep query rows, part count, edge
count, skew, and reuse. The planner must decline safely and must not use GPU
model names.

### Q5: Repeat-Customer Monthly Dropoff Hull

**Observed.** Only 3.09 seconds of 17.46-second clean wall is kernels. The trace
moves 37.52 GB H2D, 15.14 GB D2H, and zeros about 34.34 GB. CPU samples are led
by Arrow hash/dictionary work, Parquet encoding, Snappy compression, memset,
and allocator activity. The public implementation intentionally uses a
count-first pass and a temporary partitioned GeoParquet spill before grouped
hulls.

**Interpretation.** The current bounded shape solved the 24 GiB capacity
problem, but serialization is now the tax. H200's memory capacity can avoid the
spill, while the same plan must remain bounded on a 4090. This is a memory
admission decision, not a device-name decision.

**vS path.** Keep one persistent fixed-domain count accumulator and update it
in place across batches rather than allocate/zero/add dense state repeatedly.
After eligibility is known, stream compact `(group_code, point)` columns into a
native grouped-member carrier. Select between in-memory segmented grouped hull
and external partitions from measured available pool, decoded bytes, eligible
rows, and group skew. External mode should spill typed native columns without a
pandas/Arrow geometry round trip.

**Proof.** Measure zeroed bytes, serialization bytes/time, eligible fraction,
peak live/reserved memory, and hull edge work. Verify degenerate hull and full
group-before-top-k semantics.

### Q6: Sedona-Radius Zone Statistics

**Observed.** Kernel sum is 8.50 seconds, but H200 gains only 1.07x. One
`polygon_multipolygon_de9im_from_owned` launch used about 3.02 seconds locally
with 110 registers/thread. It comes from testing every zone geometry against a
fixed rectangle before the point-zone aggregate. The later point-zone public
`query_aggregate` already avoids exporting pairs.

**Interpretation.** A low-cardinality selector is taking a general exact
polygon/polygon path too early. The expensive launch is neither a good H200
workload nor necessary for most zones.

**vS path.** Make public spatial-index bounds query/semijoin the normal
many-polygons/few-rectangles prefilter, then exact-refine only ambiguous bounds.
Provide an admitted rectangle-vs-geometry predicate specialization. In the
point-zone phase, fuse exact membership with size/distance/duration reductions
so no boolean relation scratch survives the tile.

**Proof.** Report zones before bounds, after bounds, and exact-refined; DE-9IM
launch rows; candidate pairs; scratch bytes; and weighted reducer throughput.

### Q7: Detour Ratio Top 100

**Observed.** Clean local wall is 4.20 seconds and kernel sum 1.61 seconds; ZSTD
is 44.5% of kernels. H200 improves kernels 1.23x but loses clean wall 1.58x.
Local samples are mostly file input, CUDA/runtime waits, and host array setup.

**Interpretation.** The row-wise point distance, division, null handling, and
top-k are already cheap. Scan cadence and public chunk composition set the
floor.

**vS path.** Apply the persistent active-source reader and overlap decode with a
single native distance/ratio/validity expression feeding bounded top-k. Preserve
both geometry columns as native columns through scan so changing active
geometry does not assemble a new frame.

**Proof.** Compare scan-only and scan-plus-expression floors; record source
opens, frame assemblies, geometry carrier reuse, and 100-row merge time.

### Q8: Nearby Pickups Per Building

**Observed.** This is the clearest avoidable device-work result. Morton range
count and scatter consume about 12.46 of 13.92 local kernel seconds. The public
workflow materializes `dwithin` pairs, exports tree indices with `np.asarray`,
and performs host `np.bincount`, even though only per-building counts survive.
H200 cuts kernel time 2.65x and wins clean wall.

**Interpretation.** Count and scatter repeat candidate traversal, then the
relation is reduced immediately. High busy time measures an unnecessarily
materialized relation.

**vS path.** Index each streamed pickup batch and express the workflow with
public `pickups.sindex.query_aggregate(buildings.geometry,
{"nearby_pickup_count": "size"}, predicate="dwithin", distance=...)`, which
returns one count per building. Lower it to count-only candidate traversal plus
exact refinement and device counts.
Where possible, fuse Morton candidate count, distance refine, and grouped count;
never allocate the pair scatter output. Carry counts into public top-k without
host export.

**Proof.** Pair capacity and pair D2H must be zero. The floor must compare
relation-then-count with direct count across density, threshold, and geometry
family, including one-element distance broadcast and exact boundary cases.

### Q9: Building Conflation IoU

**Observed.** Clean work is 0.15 seconds locally; the 0.73-second trace envelope
is mostly process/profiler overhead. Kernel work is only 86 ms and essentially
unchanged on H200. Q9 is already exempt from the 10x target.

**Interpretation.** There is little recoverable absolute time. Optimizing this
query risks measurement noise and benchmark specialization.

**vS path.** Keep Q9 as a canary for `NativeRelation -> pair filter ->
intersection/union area -> bounded top-k`. Only improve it through reusable
carrier preservation or launch aggregation that also wins a larger shape.

**Proof.** Use many repeated in-process trials and operator floors; never infer
a win from the single-query process wall.

### Q10: All-Zone Pickup Statistics

**Observed.** Kernels are 112 of 126 clean local seconds. The prepared part-Y
point-location kernel alone is 90.32 seconds on 4090 and 22.01 seconds on H200,
a 4.10x gain, across 770 large launches. The whole trace contains 227,583
kernel launches and about 31.48 GB of memset. CUDA API durations overlap the
long same-stream kernels and must not be added to wall.

**Interpretation.** This is real device work, but not necessarily efficient
work. The current prepared kernel assigns one thread to a candidate and
serially visits candidate parts and selected-bin edges. Q11 counters show that
the broader point-location family performs trillions of edge visits for far
fewer orientation calls. H200's fp64/cache/memory strengths reward this shape;
they do not validate it.

**vS path.** First redesign the point-location index to reduce selected-bin
edge visits: more adaptive y subdivision, compact per-part edge intervals, and
large-part handling selected from measured skew. Build it with edge-shaped
parallel count/scan/scatter. Second, fuse exact membership with group size and
weighted sums so candidate booleans, capacity-tail masks, and repeated
`add.at`/bincount state are not materialized. Preserve cached point grids and
prepared polygon metadata across all trip batches.

**Proof.** The primary efficiency measures are candidates, candidate parts,
selected-bin edge visits, orientation calls, bytes zeroed, and exact time per
million visits—not GPU busy. Run targeted Nsight Compute on representative
kernels rather than replaying all 770 launches.

### Q11: Cross-Zone Trip Count

**Observed.** Kernels are 223 of 238 clean local seconds. The same prepared
point-location kernel consumes 197.06 seconds on 4090 and 56.35 seconds on
H200, a 3.50x gain, across 2,310 launches. The landed classify-once change
reduced exact candidates 29.2% and local wall 23.5%, but H200 wall only 8.0%.
The post-change profile still shows 7.981 billion exact candidates. The
baseline bounded profiler recorded
2.412 trillion candidate parts, 6.828 trillion selected-bin edge visits, and
only 156.8 million orientation calls; those three counters were not repeated
in the post-change comparison. Distributed post-change samples still show an
extreme per-candidate heavy tail. The trace zeros about 165.56 GB.

**Interpretation.** PIP is embarrassingly parallel only after the work unit is
made balanced. One-thread-per-candidate serial traversal produces massive
variance and redundant endpoint work. High busy time is compatible with poor
load balance and excess edge visits. Prior whole-warp, edge-warp, and simple
block-hybrid experiments regressed 39-66% on synthetic shapes or pushed Q11
beyond six to eight minutes; repeating them without a new index/work queue is
rejected.

**vS path.** Share Q10's better point-location index. Then add an admitted
aligned dual-endpoint operator for common candidate rows: reuse polygon
metadata/cache and reduce pickup count, dropoff count, and shared membership
directly. Compact only genuinely heavy candidate-part work into a second-stage
edge work queue; keep ordinary one/few-part candidates on the scalar-thread
path. Fuse exact results into row counts and use reusable scratch arenas or
selection-sized initialization instead of zeroing launch-capacity tails.

**Proof.** Partition evidence by ordinary and heavy-tail work: parts/candidate,
edges/candidate, queue admission, warp efficiency, atomics, zeroed bytes, and
endpoint metadata reuse. Protected simple, long-bin, multipart-skew, and
many-small-polygon shapes are mandatory on both 4090 and H200.

### Q12: Five-Nearest-Building Isolation Top 100

**Observed.** H200 kernels are 2.39x faster but clean wall is within 5%. On
4090, radix sort alone costs 4.24 seconds; a bbox-overlap count kernel also
shows a large Hopper gain. The public workflow exports fixed-k nearest indices
and distances to NumPy, runs `bincount`, loops over unresolved batches, and
uses small in-memory GeoParquet round trips to establish device-native frames
with multiple geometry columns.

**Interpretation.** The public `nearest(k=5)` relation is more general than the
consumer, which needs one row-aligned mean. Global sorts and relation exports
erase the H200 kernel advantage. The physical shape is fixed-k segmented
selection and reduction, not a global pair relation.

**vS path.** Add a public eager nearest aggregate that returns input-sized
count/sum/mean columns without exporting neighbor pairs. Lower k=5 point-to-
polygon work to cell candidate gather, exact distance, per-query segmented
select-five, and mean. Replace global radix sorts with bounded local selection.
Make multiple owned geometry columns first-class in public frame construction
so no in-memory serialization round trip is needed.

**Proof.** Measure candidates/query, unresolved rounds, sort elements, relation
bytes, geometry-column round trips, and exact distance time. Verify ties,
exclusive/max-distance behavior, fewer-than-k results, and polygon distance
semantics.

## Cross-Query Program

Priority is based on reusable wall opportunity, confidence in the diagnosis,
and architectural leverage—not ease of producing a benchmark win.

1. **Direct spatial reducers.** Complete relation-free count, weighted group,
   aligned pair, and fixed-k nearest reductions. Q8 is the cleanest first proof;
   the same substrate removes scratch and exports from Q10-Q12.
2. **Point-location index v2.** Reduce edge visits before tuning the prepared
   kernel. Use edge-shaped construction, measured skew, reuse-aware admission,
   and a heavy-tail queue. This is the largest SF100 wall opportunity in Q10
   and Q11 and also serves Q2/Q4.
3. **Persistent multi-source GeoParquet streaming.** Compact active sources,
   retain readers/handles, and overlap decode with native expression/reduction.
   Q1/Q3/Q7 are direct beneficiaries; every SF100 scan shares the substrate.
4. **Bounded native grouped state.** Remove Q5's serialization when admitted by
   memory and improve its typed spill path when not. Reuse the same state and
   scratch admission for other high-cardinality grouped geometry workloads.
5. **Fixed-k nearest physical operator.** Build segmented local selection and
   row-aligned reductions for Q12 rather than accelerating global relation
   materialization.
6. **Many-few selector planning.** Bounds-first and rectangle-specialized
   refinement prevents Q4/Q6 from paying general index or DE-9IM setup costs.

All work remains behind public GeoPandas-compatible objects and explicit public
vibeSpatial extensions such as `query_aggregate`, `query_pair_aggregate`, and a
future nearest aggregate. Native carriers are private lowering details. No
SpatialBench detection or private benchmark entrypoint is admissible.

## Required Efficiency Profiles

Nsight Systems established chronology and device time, not full utilization.
Before kernel redesign, isolate representative operator floors and collect
Nsight Compute metrics on both devices:

- achieved and theoretical occupancy, eligible warps, active warps, and issue
  stalls;
- branch efficiency and per-source warp stall reasons;
- L1/L2 hit rates, DRAM bytes and throughput, and global load efficiency;
- register count, spills/local-memory traffic, and shared-memory use;
- candidates, parts, edges, output rows, and bytes moved per launch;
- useful results per candidate/edge/byte and not merely elapsed time.

The profiles should cover scan decode, direct `dwithin` count, ordinary and
heavy-tail prepared point location, fused grouped point-location reduction,
and fixed-k nearest selection. Full Q10/Q11 Nsight Compute replay is too costly
and noisy; capture bounded representative kernels selected from Systems ranges.

## Acceptance Rails

- Same public query source and 12/12 SQL-derived correctness on both devices.
- No new CPU fallback, compute D2H, candidate relation export, or public-frame
  materialization in an optimized native shape.
- Shape-level counters and operator floors accompany clean end-to-end timing.
- Decisions use candidate/edge/group/output/memory estimates, never GPU model
  names or fixed row count alone.
- 4090 capacity remains bounded; H200 memory may admit an in-memory plan through
  the same general memory-budget rule.
- Every change reruns SF100 Q1-Q12, public 10K/1M shootouts, protected point-
  region shapes, and the full pipeline profile required by `AGENTS.md`.

## Evidence

- Clean cross-device results: `docs/dev/cross-device-performance-report.md`
- 4090 traces: `benchmark_results/nsight/sf100/2026-08-19-rtx4090-comparable/`
- H200 traces: `benchmark_results/nsight/sf100/2026-08-20-h200-comparable/nsight-sf100/`
- Point-region counters and rejected shapes:
  `docs/dev/point-region-execution-evidence.md`
- External CUDA algorithm floor:
  `docs/dev/libcuspatial-quadtree-pip-benchmark-plan.md`
- Public query source: `benchmarks/spatialbench/vibespatial_queries.py` and
  `benchmarks/spatialbench/geoparquet_public_api_queries.py`
- GeoParquet scan source: `src/vibespatial/io/geoparquet.py`
- Prepared point-location source:
  `src/vibespatial/predicates/point_location_index.py` and
  `src/vibespatial/predicates/point_location_index_kernels.py`
