

## Summary

An end-to-end SF1000 run remained 12/12 correct and improved total query time,
but it exposed one severe engine regression and three smaller or less certain
performance signals:

- **P0: Q6 is a proven 9.77x engine regression.** A controlled 100-source-shard
  A/B localizes it to eager physicalization of a device-resident,
  variable-width query-geometry view at `SpatialIndex._owned_query_input()`.
  Reverting only that hunk restores the prior rate.
- **Q1 has a variable, scale-dependent gap.** A same-data, same-query-code
  full-scale control was 9.6% slower on current source than on the v0.5.3 tag,
  but bounded and intermediate-revision controls did not reproduce a stable
  slowdown. Treat this as a regression rail and profiling target, not a
  localized bug.
- **Q7 has a smaller, repeatable full-scale gap.** The same-code full-scale
  control was 10.0% slower than v0.5.3, with 12.2% more allocations. The change
  first appears with the expanded exact multi-key top-k implementation, but a
  function-level cause has not yet been proven.
- **Q3 is not an engine regression in the controlled A/B.** Current and
  v0.5.3 source take 205.90s and 206.06s with identical query code and
  telemetry. The larger historical delta compares different query plans and
  should drive a public device-resident composition improvement, not a source
  revert.

These findings define two implementation commits, not an umbrella follow-up
issue. The first fixes the proven Q6 regression. The second improves the shared
Q1/Q7 top-k shape and the Q3 device-resident grouped-reduction shape. Pinned
query provenance, physical-stage telemetry, and large-streaming canaries are
acceptance evidence inside those commits rather than a separate bookkeeping
stage.

The implementations must remain general and shape-driven. SpatialBench is the
measurement surface, not the source of production policy: no production branch
may inspect a benchmark query number, dataset column, scale factor, or device
model. Each optimization must first demonstrate its physical-work improvement
with an independent synthetic shape test, then show the expected benefit in
SpatialBench.

## Why Q6 is the priority

The current full run spends 5,941.23 seconds in Q6, or 56.4% of the complete
Q1-Q12 query time. The current query total excluding Q6 is 2.83x faster than the
prior run, so restoring Q6 would dominate any near-term improvement from the
other three workstreams.

Q6 exercises a general public workload shape:

```text
large point batches with aligned numeric values
  -> point spatial index
  -> repeated query by one small selected Polygon/MultiPolygon view
  -> exact contains refinement
  -> count plus multiple weighted sums by query geometry
  -> small terminal result
```

This is not a benchmark-only shape. It is the normal implementation target for
repeated point-region analytics through `SpatialIndex.query_aggregate()`.

## Evidence quality and provenance

There are two comparisons below and they answer different questions.

1. The **historical integrated comparison** motivated the investigation, but
   its earlier source tree was not fully pinned and its Q1/Q3/Q7 query code was
   different. It is useful as an end-user performance signal, not as causal
   evidence.
2. The **controlled source comparison** loads identical current benchmark code
   and identical prepared data against pinned vibeSpatial source revisions.
   This is the authoritative engine-regression evidence.

All controlled runs used one consistent environment, one prepared dataset,
isolated query processes, one measured cold execution, and telemetry. Every
path recorded zero fallback. Q1/Q3/Q7 outputs are byte-identical; Q6 integer
counts are identical and its fp64 reductions pass the established tolerance.
Full-scale numbers are single executions rather than medians; bounded warm
medians are noted where available.

The SF1000 evidence was produced on a machine with an RTX 6000 Ada, a newer
processor, and more system memory than the local development machine. The
claims below are proven for that source machine. This machine cannot execute
SF1000 for verification, and its RTX 4090 also selects a different memory
envelope for some point-location plans. Local bounded tests establish
correctness and physical behavior; paired source-machine runs establish the
SF1000 performance result. Absolute timings and admission choices are not
expected to match across the two machines.

### Historical signal versus controlled result

| Query | Earlier integrated run | New integrated run | Isolated current confirmation | Same-code v0.5.3 control | Current / control | Classification |
|---|---:|---:|---:|---:|---:|---|
| Q1 | 166.08s | 206.90s | 173.69s | 158.55s | 1.10x | Variable small gap; add rail and profile |
| Q3 | 116.13s | 208.12s | 205.90s | 206.06s | 1.00x | Query-plan/methodology gap, not engine regression |
| Q6 | 626.00s | 5,941.23s | 410.56s on bounded control | 42.01s on bounded control | **9.77x** | Proven engine regression |
| Q7 | 70.31s | 89.93s | 86.94s | 79.03s | 1.10x | Repeatable small engine gap plus query-plan delta |

The earlier Q1/Q3/Q7 implementation used lower-level WKB/device helpers and a
selective host refinement. The current implementation is intentionally built
from public vibeSpatial APIs. Recovering performance must therefore improve the
public physical paths or express a better plan through public APIs; reinstating
private benchmark helpers or making selective host refinement an engine fast
path is not an acceptable fix.

### Same-code telemetry

| Query | Source | Time | Cumulative allocation | Allocations | Tracked D2H | Fallbacks |
|---|---|---:|---:|---:|---:|---:|
| Q1 | v0.5.3 (`2ad5a86`) | 158.55s | 1.236 TB | 61,554 | 0.93 MB | 0 |
| Q1 | current (`03981d5`) | 173.69s | 1.236 TB | 64,835 | 0.93 MB | 0 |
| Q3 | v0.5.3 (`2ad5a86`) | 206.06s | 2.338 TB | 62,125 | 0.70 MB | 0 |
| Q3 | current (`03981d5`) | 205.90s | 2.338 TB | 62,125 | 0.70 MB | 0 |
| Q7 | v0.5.3 (`2ad5a86`) | 79.03s | 2.991 TB | 52,111 | 0.62 MB | 0 |
| Q7 | current (`03981d5`) | 86.94s | 3.025 TB | 58,480 | 0.62 MB | 0 |

These are not transfer regressions: D2H volume is unchanged and small, and no
path falls back. Q1 and Q7 instead warrant stage timing around scan, distance,
expression evaluation, top-k, synchronization, and terminal export.

Peak VRAM is deliberately omitted from this comparison because allocator-pool
reservation policy differed between revisions; cumulative allocation, event
count, and transfer volume are comparable.

## P0 diagnosis: Q6 prepared execution is lost after query-view compaction

Commit `2487aff` added `compact_indexed_spatial_input()` at the owned spatial
query boundary. When the query side is a device-resident indexed view over
variable-width geometry, it gathers the selected rows into a physically compact
carrier before indexing and exact-predicate preparation. The objective is
correct: downstream coordinate-shaped work should be sized from selected
geometry rather than ancestral capacity.

The Q6 workload repeatedly passes the same small selected polygon view to
point-index aggregate queries. With eager compaction, execution no longer uses
the prepared point-location y-edge directory observed on v0.5.3 and spends its
time in exact predicate-pair filtering. Reverting only the compaction call
restores the old performance shape.

### Single-change control

The same 100 source shards, complete region table, query code, data, and output
were used for every row:

| Source/configuration | Time | Relative to v0.5.3 | Cumulative allocation | Allocations | Tracked D2H |
|---|---:|---:|---:|---:|---:|
| v0.5.3 (`2ad5a86`) | 42.01s | baseline | 259.56 GB | 20,910 | 7.24 MB |
| `2487aff`, compaction hunk reverted | 42.19s | 1.00x | 259.56 GB | 20,915 | 7.24 MB |
| `2487aff` | 410.41s | **9.77x slower** | 257.90 GB | 20,999 | 7.24 MB |
| current (`03981d5`) | 410.56s | **9.77x slower** | - | - | - |

Integer counts are identical. Weighted averages differ only by floating
reduction order around 1e-19 and pass the established tolerances. A second
physical copy and storage path changed current runtime by only 1.9%, ruling out
GeoParquet conversion and storage throughput as the cause. Sustained device
utilization during the slow control also rules out an idle IO wait.

Allocation volume, allocation count, and D2H are effectively unchanged across
the decisive control. This is a change in exact-refinement work, not a bulk
movement or allocator regression.

### Instrumentation gap

The existing point-region profiler records no stage groups for this public
`query_aggregate()` path. A stack sample can identify exact pair filtering, but
the normal profile cannot currently answer how much time or work belongs to:

- query-input compaction;
- prepared-index construction and reuse;
- candidate generation;
- exact point-region refinement;
- grouped count and weighted reduction;
- synchronization; or
- terminal export.

That gap allowed a nearly 10x public-path regression to coexist with healthy
high-level allocation and transfer counters.

## Required physical contracts

The implementation should declare and test these reusable shapes under
ADR-0046 rather than optimize four named queries.

| Motivating query | Reusable physical shape | Native terminal state |
|---|---|---|
| Q6 | candidate-refine point/region relation consumed by segmented count and weighted sums | small `NativeGroupedAttributeReduction` |
| Q1 | broadcast point-to-scalar distance, predicate rowset, bounded lexicographic top-k | ordered `NativeRowSet` plus selected columns |
| Q3 | broadcast point-to-one-region distance, temporal bucket, grouped numeric reductions | small `NativeGrouped` result |
| Q7 | aligned point-to-point distance, numeric expression chain, bounded lexicographic top-k | ordered `NativeRowSet` plus selected columns |

Canonical owned geometry remains storage; each operator may choose a compact
carrier, ancestral row view, prepared derivative, bounded candidate set, or
workspace only through shape-level work and live-memory admission.

## Two-commit implementation plan

### Commit 1: restore the general Q6 physical shape

Commit subject: `fix(spatial): preserve prepared execution for selected query views`

#### Add the Q6 regression canary before changing execution

Add a public `SpatialIndex.query_aggregate()` GPU test with:

- a large indexed point side;
- a non-identity selected view of variable-width Polygon/MultiPolygon input;
- `predicate="contains"`;
- count and at least two aligned fp64 sums;
- repeated calls using the same selected query owner;
- a brute-force oracle for counts and sums; and
- observable counters for prepared construction/reuse and exact work.

Run the canary with both an ample and constrained memory envelope. It must be
large enough to expose the physical-path difference without requiring SF1000.

#### Preserve exact preparation across Q6 compaction

Evaluate two legitimate engine designs against the canary:

**A. Compact derivative with retained preparation.** Physicalize the selected
geometry once, attach it to a stable owner, and build/cache prepared
point-location state keyed by immutable geometry lineage and row selection.
Repeated aggregate consumers reuse that derivative. Stream readiness and cache
invalidation must be explicit.

**B. Admitted ancestral-view execution.** Keep the compact carrier for cases
where ancestral coordinate capacity would violate memory admission, but allow
the indexed ancestral view plus its prepared state when the row map and
coordinate work fit the live envelope. The selection must compare total
candidate/refinement/preparation/transient work, not only geometry bytes.

A direct aggregate consumer that fuses exact classification with count/sum
reduction may complement either design, provided it preserves the same exact
predicate semantics and remains a general relation-consumer path.

Do not fix Q6 by globally deleting compaction. The compaction has a valid
memory-shaping role for sparse selections from very large variable-width
owners. The defect is losing the better prepared execution shape, not the
existence of a compact representation.

#### Add stage-separated spatial aggregate telemetry

Extend the point-region profile to cover public `query_aggregate()` and record:

- ancestral and selected geometry rows, parts, edges, and bytes;
- chosen carrier and reason;
- prepared-state cache hit/miss/build bytes;
- candidate pairs and surviving candidate-parts;
- selected-bin memberships or equivalent exact-edge work;
- count/sum reduction rows and groups;
- per-stage duration, launches, synchronizations, and transient peak; and
- D2H, materialization, and fallback events.

Instrumentation must reduce to bounded packets and compile out of production
kernels; it must not add an atomic to the inner edge loop. Commit 1 is complete
after the canary, targeted suites, SF1/SF100 where available, and the full 1M
pipeline profile pass. The paired 100-source-shard Q6 control on the RTX 6000
Ada machine remains useful follow-up evidence, but it is not a delivery gate
for this implementation because that machine is not available for validation.

### Commit 2: improve the general Q1/Q3/Q7 physical shapes

Commit subject: `perf(tabular): optimize streamed top-k and grouped reductions`

#### Profile and optimize the common Q1/Q7 top-k shape

Q1 and Q7 both scan large resident batches, compute continuous fp64 metrics,
and retain 100 rows using multiple ordering keys. Starting with `2487aff`, the
exact top-k implementation expanded support for nulls and `keep=first|last|all`
using iterative boundary refinement across keys. That is correct and more
general, but Q7's stable full-scale slowdown and higher allocation count make
this the first shared subsystem to profile.

Add timings and work counters for:

- key normalization and missing masks;
- primary `top_k` selection;
- each boundary reduction and host-visible scalar fence;
- active-position compaction and key gathers;
- final selected-key sort; and
- rowset take/export.

Then evaluate two exact shapes:

**A. Discriminative-primary boundary path.** Select the primary threshold,
count strict winners and the boundary-equal span, then stable-sort only those
candidates by the complete key and original row position. Admit this path only
when the measured boundary cardinality and workspace fit. This is especially
appropriate for continuous primary metrics, but selection must depend on
observed key shape rather than a column or query name.

**B. Fused iterative refinement.** Retain the current general tie-capable
algorithm but fuse masks/counts where possible, reuse position/mask workspace,
and remove per-key synchronization fences that are not semantically required.
This remains the fallback for large tie spans and `keep="all"` growth.

Required top-k tests include nulls, NaNs, signed zero, ascending and descending
keys, multi-key ties, all-equal primary keys, `keep="first"`, `keep="last"`,
`keep="all"`, `n<=0`, and `n>=rows`. The result and ordering must match pandas.

If operator time is no longer material after this work, do not create a
streaming dataframe planner solely for the benchmark. If repeated terminal
100-row exports remain material, implement a reusable streaming top-k
accumulator in this commit only when its independent shape tests justify it.

#### Re-express Q3 as a public device-resident grouped reduction

The current Q3 benchmark explicitly exports selected temporal/numeric columns
to pandas for every batch and performs a host `Period` conversion plus groupby.
It also emits one timezone-loss warning per non-empty batch. The controlled
engine A/B is flat, so this is not evidence for reverting an engine change.

First express the equivalent plan through existing public native operations:

```text
distance predicate -> NativeRowSet
  -> calendar-month code + duration expression
  -> count and three sums by month
  -> merge bounded monthly state
  -> one terminal host export
```

If current public APIs can already retain this state, only the benchmark plan
needs to change. If a public operation materializes or is unsupported, fill the
general Native expression/grouped-reduction continuity gap and add an ordinary
API test. Do not teach the engine about months, fares, or this query; temporal
component extraction and small-domain grouped numeric reduction are the
general contracts.

Suppressing the warning alone is not a performance fix. The target is removal
of repeated host materialization and duplicate per-batch aggregation work.

#### Make benchmark provenance and regression rails durable

Every performance artifact used for comparison should record:

- exact vibeSpatial commit and dirty-state digest;
- exact benchmark query-module digest;
- prepared-data manifest and geometry encoding;
- cold/warm status and run count;
- execution/fallback mode;
- device-memory policy, without using device identity for dispatch; and
- correctness fingerprint or oracle result.

Add three tiers of rails:

1. deterministic small correctness tests for every semantic edge;
2. bounded large-shape medians for point/region aggregate, scalar distance plus
   top-k, aligned distance plus top-k, and filtered temporal grouping; and
3. the existing SF100 correctness/performance suite followed by one SF1000
   confirmation when a candidate passes the bounded rails.

Cold and warm results must be reported separately. In the bounded Q7 control,
one cold observation suggested a large delta while a one-warmup, three-run
median was 3.15s current versus 3.11s at the pre-latest revision. That is a
useful warning against promoting single cold observations into root causes.

Commit 2 is complete after the Q1/Q3/Q7 shape tests, the cross-query SF1/SF100
checks where available, and the full 1M pipeline profile pass. A later full
SF1000 run on the RTX 6000 Ada machine may confirm the extrapolation, but it is
not required to land either commit.

## Acceptance criteria

### P0 Q6

- The bounded selected-view canary demonstrates reuse of prepared execution
  under an ample envelope and compact execution under a constrained envelope.
- Current versus pinned output matches exactly for integer counts and within
  the established fp64 reduction tolerances for sums/averages.
- Repeated selected-view `query_aggregate()` records one reusable prepared
  build or an alternative path with demonstrably equivalent bounded exact work.
- Both ample-memory and constrained-memory canaries pass without OOM, silent
  CPU fallback, or unbounded relation materialization.
- Telemetry attributes carrier selection, preparation, candidate generation,
  exact refinement, reduction, and synchronization.

### Q1 and Q7

- Same-code paired large-shape medians are within 5% of the pinned accepted
  baseline, or a documented physical-work improvement establishes a new lower
  baseline.
- Exact ordering matches pandas across the top-k semantic matrix.
- The optimized path is selected by key/tie/workspace shape, never query or
  column identity.
- Full-row allocation and synchronization do not grow linearly with the number
  of ordering keys after a discriminative primary boundary is known.
- Strict-native execution records zero fallback and no non-terminal bulk D2H.

### Q3

- A public device-resident plan performs spatial filtering, temporal bucketing,
  and grouped numeric reduction with one bounded terminal export.
- The result matches the established oracle, including month keys, counts,
  durations, sums/means, null behavior, and timezone semantics.
- The controlled same-code engine rail remains within 5% across accepted
  revisions.
- No per-batch timezone warning, host groupby, or selected-row bulk export
  remains in the timed plan.

### Cross-query no-regression gate

- SF1 and SF100 remain 12/12 correct.
- The current Q5 persistent dense-count/spill path, Q10/Q11 prepared
  point-region improvements, and Q12 bounded fixed-k path regress by no more
  than 5% in paired medians.
- Any later SF1000 confirmation should remain 12/12 correct, improve total wall
  time, and show that Q6 no longer dominates the suite; this is additional
  source-machine evidence rather than a local landing requirement.
- The implementation adds no benchmark-, scale-, data-column-, path-, or
  device-model-specific production branch.

## Delivery and completion

1. Land Commit 1 after its local correctness, bounded-shape, SF100, and full
   1M profile gates pass.
2. Land Commit 2 after its local semantic matrix, cross-query SF100, and full
   1M profile gates pass.

This ordering isolates the 9.77x proven regression before spending time on the
smaller signals. Work is complete when both commits and their local evidence
pass. A defect or unexplained regression found by a gate is fixed before
completion. Create an issue only for concrete work deliberately excluded from
these two commits; describe the current problem directly rather than preserving
unfinished items from this plan.

## Local implementation evidence

Commit 1 evidence was collected on 2026-09-02 on the RTX 4090 host.
The imported source was base revision `03981d5` plus source-worktree digest
`e63bfa61ff34b2e2f6917d679dcd9d686069bb2f91acc381466522b481bbb2bc`.
The streamed query module, shared query module, lockfile, and SF100 GeoParquet
manifest digests were respectively `b34099f0`, `36e230f0`, `60942838`, and
`a970d40a` (SHA-256 prefixes).

- The selected-view ample/constrained canaries passed with exact counts and
  two fp64 sum oracles, zero fallback, one reusable prepared build after reuse
  was observed, and explicit carrier/reduction work packets.
- The targeted point-region, spatial-query, and geometry-slicing suites passed:
  234 passed and one optional SciPy test skipped.
- Strict-native SF100 Q6 completed with one warmup and three measured runs:
  13.47s median from 13.47s, 13.34s, and 13.54s. The 19 result rows match the
  accepted local artifact exactly for keys and integer counts; the maximum
  relative fp64 difference is `3.429e-16` and passes the established tolerance.
- The full 1M pipeline profile completed. Every non-deferred stage was reviewed;
  the largest 1M stage was `predicate-heavy.read_geojson` at 81.41ms. Other
  notable maxima were `grouped-capacity-partitions.mixed_strip_exact_union` at
  68.53ms, `grouped-disjoint-constructive-reduce.build_device_disjoint_groups`
  at 65.33ms, `small-grouped-constructive-reduce.native_grouped_union` at
  48.93ms, and `join-heavy.write_output` at 16.43ms. No unexpected CPU-heavy
  stage exceeded one second.

Commit 2 evidence was collected on 2026-09-02 on the same RTX 4090 host, with
Commit 1 (`282f89d`) as the pinned local baseline. The final full-profile
source-worktree digest was
`722aecb57029cb51cc8cadb48fb8049e236552e5b849f00d27fec42c58e368ce`.

- The exact multi-key top-k path now prepares the primary key over the active
  rowset and gathers later keys only for the boundary. An independent 2M-row,
  three-key shape prepared 2M primary-key rows, gathered 300 selected-key rows,
  and sorted 100 final rows. The corresponding allocation trace fell from
  222,148,043 bytes and 99 allocations at `282f89d` to 209,655,882 bytes and
  77 allocations. Current one-key and three-key volumes were 209,653,023 and
  209,655,882 bytes, demonstrating that later discriminative keys no longer add
  full-row allocation. The pandas semantic matrix passed for ascending and
  descending selection, nulls, NaNs, signed zero, multi-key ties, and
  `keep=first|last|all`.
- Q3 now uses public native timestamp expressions and a fixed-domain grouped
  count/sum reduction whose output remains device-backed across scan batches.
  Mixed timestamp units retain exact integer subsecond deltas before conversion,
  and fp64 grouped sums use stable segmented reduction rather than contended
  atomics. An adversarial 1.2M-row cancellation case returned the exact pandas
  result on three consecutive runs, and the declared admission bound covered
  the measured 1M-row, three-sum operation peak. On SF100, the final
  repeat-three median was 12.23s (12.35s, 12.23s, and 12.23s), with zero
  fallback. Relative to
  the `282f89d` same-plan baseline, materialization events fell from 100 to 24
  and tracked D2H events from 80 to 27. The 24 materialization events comprise
  20 device-backed public-frame assemblies with no host transfer and four
  bounded terminal exports. Total D2H was 264,183 bytes: one 256 KiB count
  vector, three 84-row sum vectors, and scalar domain-validation fences. Total
  allocation stayed flat at 210.580 GB versus 210.518 GB, with zero selected
  source-row export, host groupby, or timezone warning.
- Paired SF100 medians against `282f89d` were Q1 12.49s versus 12.94s, Q3
  12.23s versus 13.35s, and Q7 4.79s versus 4.78s. Q7's wider seven-run sample
  was 4.81s, 4.82s, 4.80s, 4.79s, 4.79s, 4.78s, and 4.77s. Every rail is within
  5%, while Q7 telemetry reduced cumulative allocation from 296.171 GB and
  6,261 allocations to 292.425 GB and 5,821 allocations. The final hardened
  Q1 and Q3 outputs separately passed the frozen oracle for all 100 and 84
  rows, respectively.
- The complete SF100 result set collected immediately before the reviewer
  correctness hardening passed the frozen 12/12 oracle comparison. Its Q1,
  Q3, and Q7 medians were 12.86s, 12.74s, and 4.79s. The remaining
  strict-native medians were Q2 7.59s, Q4 7.88s, Q5 17.22s, Q6 13.47s, Q8
  16.91s, Q9 0.14s, Q10 58.50s, Q11 79.46s, and Q12 25.17s. That packet's
  assembled twelve-query total was 256.73s. All schemas, row order, exact
  integer/text cells, and configured fp64 tolerances passed. The post-hardening
  Q1/Q3 reruns above replace those two timings without claiming a second
  complete twelve-query run.
- The final full 1M pipeline profile completed with 22 active cases and two
  expected raster deferrals. All 102 stages were reviewed, with zero fallback
  and no stage above one second. The largest 1M stages were
  `predicate-heavy.read_geojson` at 69.29ms,
  `grouped-disjoint-constructive-reduce.build_device_disjoint_groups` at
  67.21ms, `grouped-capacity-partitions.mixed_strip_exact_union` at 64.39ms,
  and `small-grouped-constructive-reduce.native_grouped_union` at 45.74ms.
- The final changed-surface suite passed 892 tests. The repository-wide run
  immediately before the final review fixes passed 7,988 tests; its 11
  failures were reproduced unchanged at `282f89d`: one upstream Arrow test
  whose obsolete expectation requires a `TypeError`, and ten upstream CRS
  cases for a projection-grid method not supported by the installed
  `vibeproj`. The final fixes are covered by the changed-surface suite and do
  not touch those failures.
