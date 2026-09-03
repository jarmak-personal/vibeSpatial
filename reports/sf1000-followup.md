## Summary

Commit `23ca4c6195bbfa315a912c0e3180ba8ecb408db8` was tested from a clean,
unmodified source export using its committed SpatialBench implementation. It
contains real improvements, but the challenging SF1000-derived workload found
one missed interaction and one scale regression:

- **Q3 is improved and should be retained.** Full SF1000 falls from 244.31s to
  192.49s with exact output.
- **Q6 remains on the slow path upstream.** Its cumulative reuse test passes,
  but the selected compact derivative is independently rejected by the prepared
  index's one-shot 1-million-coordinate minimum. A minimal generalized proof
  resolves this and completes full SF1000 Q6 in 397.66s instead of the prior
  5,941.23s regression.
- **Q7's new streaming accumulator is slower at scale.** Restoring the prior
  committed public plan on the same new engine reduces full runtime from 97.44s
  to 84.09s and allocation events from 136,771 to 51,918.
- **Q1 improves over the immediately preceding revision but remains above its
  earlier controlled baseline.** Reusing the prior Q1 plan does not improve full
  runtime, so Q1 needs scan/distance profiling rather than another top-k guess.

This issue includes the working Q6 patch and the exact Q7 reference plan. The
request is to integrate/generalize these proven physical behaviors within
vibeSpatial's ownership, admission, telemetry, and testing contracts—not to
infer another implementation from smaller-scale results.

Production code must remain shape- and resource-driven. No query number,
SpatialBench column, scale factor, data path, or device model belongs in
dispatch logic.

## Evidence boundaries

The upstream measurements below use:

- exact clean commit `23ca4c6195bbfa315a912c0e3180ba8ecb408db8`;
- the benchmark modules committed at that revision;
- unchanged native-GeoArrow GeoParquet input;
- isolated query processes;
- one measured execution with no query warm-up at full SF1000;
- bounded warm medians of three runs after one query warm-up; and
- correctness comparison against previously verified results.

The Q6 proof and Q1/Q7 plan A/B are explicitly diagnostic variants. They are
not reported as current upstream performance. Each changes only the code
described in its section, uses the same `23ca4c6` engine/data/environment, and
exists to provide an executable implementation reference.

## Clean upstream results

### Full SF1000

| Workload | `c656c4a` | Clean `23ca4c6` | Change | Earlier controlled | Current vs controlled | Correctness |
|---|---:|---:|---:|---:|---:|:---:|
| Q1 | 227.46s | 212.61s | -6.5% | 173.69s | +22.4% | Pass |
| Q3 | 244.31s | 192.49s | **-21.2%** | 205.90s | **-6.5%** | Pass |
| Q7 | 88.26s | 97.44s | **+10.4%** | 86.94s | +12.1% | Pass |
| **Q1+Q3+Q7** | **560.03s** | **502.54s** | **-10.3%** | **466.53s** | **+7.7%** | **3/3** |

Q6 was stopped at a bounded gate because its clean upstream rate still projects
to a multi-hour full run.

### Bounded 100-file gate

The fixture contains the first 100 prepared fact files and complete dimension
tables. It is a source-file-count rail, not a claimed scale factor.

| Workload | `c656c4a` no-warmup | `23ca4c6` no-warmup | `c656c4a` warm median | `23ca4c6` warm median |
|---|---:|---:|---:|---:|
| Q1 | 8.82s | 10.07s | 8.08s | 8.22s |
| Q3 | 11.95s | 11.72s | 8.34s | 8.39s |
| Q6 | 433.68s | 409.43s | Not run | Not run |
| Q7 | 4.04s | 7.14s | 3.29s | 3.37s |

Clean v0.5.3 completes the same Q6 fixture in 42.01s. Clean upstream
`23ca4c6` remains 9.75x slower.

### Full telemetry

| Workload | Time | Peak VRAM | Cumulative allocation | Allocations | Tracked D2H | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 212.61s | 4.81 GiB | 1.332 TB | 145,269 | 0.0046 MiB | 0 |
| Q3 | 192.49s | 7.60 GiB | 2.338 TB | 70,261 | 0.252 MiB | 0 |
| Q7 | 97.44s | 7.94 GiB | 2.891 TB | 136,771 | 0.0031 MiB | 0 |

The residency goals are met: transfer is bounded and every query remains
strict-native. Q1/Q7 show that zero-D2H state retention can still lose when its
per-batch frame construction, compaction, and allocation lifecycle costs more
than exporting a tiny bounded candidate set.

## P0 Q6: cumulative reuse admission conflicts with the one-shot minimum

### Exact failure

The selected query geometry has:

- 34 selected geometries;
- 6,158 selected polygon parts;
- 177,308 selected coordinate lanes; and
- 2.89 MB selected geometry storage.

Its ancestor has:

- 1,033,728 geometries;
- 1,369,861 parts;
- 345,692,829 coordinate lanes; and
- 5.59 GB geometry storage.

`SpatialIndex._query_aggregate_owned_input()` correctly records repeated use
and computes cumulative candidate work. On the second call:

```text
cumulative_candidate_work = 396,886,896
compact_preparation_work   =     177,308
uses_to_amortize           =           1
```

The caller therefore admits a compact prepared derivative. It calls
`prepare_point_region_y_indexes()`, but `prepare_polygon_part_y_index()` has a
separate early return:

```python
_MIN_PREPARED_COORDINATES = 1_000_000

if coordinate_count < _MIN_PREPARED_COORDINATES and not force_prepared:
    return None
```

The compact derivative is below that one-shot threshold. Preparation is
declined, the caller does not try the ancestor, and every fact batch continues
through exact compact refinement.

The clean three-file trace records:

```text
preparation requests: 3
builds:               0
cache hits:           0
declines:             3
prepared consumers:  0
```

This explains why smaller validation did not find the issue: the interaction
requires a very large ancestral owner, a sparse selected derivative below the
static preparation minimum, and enough repeated point work to amortize that
small derivative.

### Working reference patch

The proof makes one semantic change: after cumulative reuse has already admitted
preparation, let that decision override the one-shot minimum. Ordinary callers
and one-shot workloads retain the existing cutoff. All normal memory-envelope,
membership, width, and coverage admissions remain active.

The complete applyable reference is attached as
`vibespatial-23ca4c6-q6-reuse-admission-proof.patch`. Its core is:

```diff
 def prepare_polygon_part_y_index(
     owned,
     family,
     *,
     _target_bin_count=None,
+    _reuse_admitted=False,
 ):
     ...
-    if coordinate_count < _MIN_PREPARED_COORDINATES and not force_prepared:
+    if (
+        coordinate_count < _MIN_PREPARED_COORDINATES
+        and not force_prepared
+        and not _reuse_admitted
+    ):
         return None

 # Only after cumulative candidate work admits preparation:
 prepare_point_region_y_indexes(
     preparation_owner,
     tree_owned,
+    _reuse_admitted=True,
 )
```

The production implementation may rename or internalize this flag, but it must
preserve the contract: the generic one-shot heuristic cannot veto a stronger
caller decision based on observed repeated work.

### Proof results

| Scope | Clean upstream | Proof | Speedup | Correctness |
|---|---:|---:|---:|:---:|
| Three fact files | 15.13s | 11.17s | 1.35x | Pass |
| 100 files, no warm-up | 409.43s | 32.49s | **12.60x** | Pass |
| 100 files, warm median | Not run | 30.70s (30.68/30.70/30.74) | - | Pass |
| Full SF1000 | 5,941.23s prior regression | **397.66s** | **14.94x** | Pass |

The full proof is also 1.57x faster than the earlier successful 626.00s Q6
result.

Full SF1000 proof counters:

```text
query aggregate calls:       771
prepared index builds:         1
prepared cache hits:          770
prepared consumers:           770
initial compact consumers:      1
fallbacks:                      0
```

The admitted compact index uses 128 y bins, a 16x16 conservative coverage grid,
and only 17.9 MiB persistent storage. Peak VRAM, cumulative allocation, and D2H
are essentially unchanged from the slow path; the speedup comes from less exact
refinement work.

### Required generalization and tests

- Represent heuristic strength explicitly: one-shot size heuristic, observed
  reuse admission, and forced diagnostic preparation must not collapse into an
  ambiguous boolean.
- Add a regression test with an ancestral coordinate count above the preparation
  threshold and a selected derivative below it.
- Require two or more aggregate calls so cumulative reuse becomes observable.
- Assert one compact prepared build, later cache hits, and prepared exact
  consumers.
- Test both admitted and constrained memory envelopes.
- Preserve exact counts and established fp64 reduction tolerances.
- If compact preparation is declined by actual memory admission, try the
  ancestral prepared carrier only when it independently fits and amortizes;
  otherwise retain exact compact fallback.

## Q3: retain the persistent grouped accumulator

The new Q3 plan improves full SF1000 from 244.31s to 192.49s while preserving:

- exact 84-row output;
- zero fallback;
- 0.252 MiB terminal D2H;
- deterministic stable fp64 reduction; and
- bounded device-resident grouped state.

Its bounded warm result is flat because fixed costs dominate there. The
full-scale result confirms that persistent `dense_grouped_reduce(..., out=...)`
state is beneficial across the complete stream. Do not revert this part of
`23ca4c6` while addressing Q1/Q7.

Follow-up optimization can still target stable ordering and allocation count,
but it should retain the public native plan and use 192.49s as the accepted
full-scale baseline.

## Q7: restore the prior public plan until streaming state is cheaper

### Controlled same-engine A/B

The diagnostic changes only the Q7 benchmark plan. Both rows use the
`23ca4c6` engine and public APIs.

| Plan | 100-file warm median | Full SF1000 | Allocations | Cumulative allocation | D2H | Correctness |
|---|---:|---:|---:|---:|---:|:---:|
| New `_streaming_topk` accumulator | 3.37s | 97.44s | 136,771 | 2.891 TB | 0.003 MiB | Pass |
| Prior per-batch bounded public plan | **3.14s** | **84.09s** | **51,918** | 2.987 TB | 0.589 MiB | Pass |

The prior plan is 13.7% faster at full scale and uses 62.0% fewer allocation
events. Its additional transfer is less than 0.6 MiB across the entire query.
This is an allocation/lifecycle regression, not a bulk-transfer tradeoff.

### Working reference plan

Restore the `c656c4a` Q7 physical plan as the immediate benchmark implementation:

```python
def _q7_shard_topk(self, trips):
    pickup = trips.geometry
    dropoff = trips.set_geometry("t_dropoffloc").geometry
    line_distance = pickup.distance(dropoff, align=False) / 0.000009
    reported = trips["t_distance"].astype(float)
    detour_ratio = (reported / line_distance) * (line_distance / line_distance)
    metrics = trips.drop(columns=["t_dropoffloc"]).assign(
        reported_distance_m=reported,
        line_distance_m=line_distance,
        detour_ratio=detour_ratio,
        __vibespatial_topk_tie_2=0 - trips["t_tripkey"],
    )
    selected = metrics.nlargest(
        100,
        ["detour_ratio", "reported_distance_m", "__vibespatial_topk_tie_2"],
    )
    columns = [
        "t_tripkey",
        "reported_distance_m",
        "line_distance_m",
        "detour_ratio",
    ]
    return pd.DataFrame({column: selected[column].to_numpy() for column in columns})
```

The inherited public Q7 loop keeps only 100 rows per batch and performs one
final exact lexicographic top-k over those bounded candidates. This plan is
already committed history, public-API-only, correct, and faster on the actual
challenging workload.

### Generalize `_streaming_topk` before re-enabling it here

The native accumulator should eventually win, but it needs a state carrier that
does not reconstruct and physically compact a public native frame for every
batch. Suggested contract:

- retain a dedicated bounded `NativeStreamingTopKState`, not a public
  `GeoDataFrame` as mutable accumulator state;
- own only ordering keys, stable global row identity, and the selected payload;
- reuse capacity for at most the prior `k` plus the new batch's `k` candidates;
- merge/sort at most `2k` candidate rows after each batch-local top-k;
- avoid rebuilding `RangeIndex`, attribute tables, geometry carriers, and
  provenance on every update;
- gather/physicalize geometry only when a row first enters the winner state, or
  once at terminal export when source lifetime permits; and
- admit device retention only when predicted lifecycle cost is below a bounded
  host-candidate merge. A tiny terminal/candidate D2H path is a valid public
  execution choice when it is faster and explicitly reported.

Re-enable the streaming plan only after a same-engine full-scale or multi-tier
rail beats the 84.09s reference without weakening exact ordering.

## Q1: do not treat top-k plan selection as the current bottleneck

The same Q1 A/B is effectively flat:

| Plan | Full SF1000 | Allocations | Cumulative allocation | D2H | Correctness |
|---|---:|---:|---:|---:|:---:|
| New streaming accumulator | 212.61s | 145,269 | 1.332 TB | 0.0046 MiB | Pass |
| Prior per-batch public plan | 213.76s | 64,063 | 1.236 TB | 0.883 MiB | Pass |

The prior plan removes more than half the allocation events but does not improve
wall time. That is strong evidence that Q1 is dominated elsewhere—most likely
scan/decode, scalar distance, predicate construction, or synchronization.

Keep either correct plan based on broader API goals, but do not claim a Q1
performance fix from switching between them. Add non-overlapping device-event
timing for:

```text
scan/decode
  -> scalar distance
  -> threshold predicate
  -> batch top-k
  -> winner merge/compaction
  -> x/y and attribute take
  -> terminal export
```

Only optimize the measured dominant stage. The next full-scale target remains
the 173.69s controlled result, with the 158.55s pinned v0.5.3 same-plan result
as a secondary reference.

## Test issue found during validation

Changed-surface tests produced 668 passes and one failure:

```text
test_point_region_profile_observes_public_pair_aggregate_boundedly
expected coverage_grid_width == 8
observed coverage_grid_width == 16
```

The implementation intentionally chooses coverage width from the device-memory
envelope, so a test that does not inject a fixed budget must not assert one
hardware-dependent width. Fix the test by injecting the intended memory envelope
and deriving the expected tier, or assert the invariant being tested: a supported
admitted width, exact result, bounded telemetry, and zero fallback.

## Acceptance criteria

### Q6

- The reuse-admitted compact derivative bypasses only the one-shot size cutoff;
  all actual device-memory admissions remain authoritative.
- The 100-file warm median is no slower than 1.20x the 30.70s proof baseline.
- Full SF1000 is no slower than 1.20x the 397.66s proof result.
- One prepared build and repeated cache hits/consumers are visible.
- Counts match exactly and fp64 outputs pass the established tolerance.
- Constrained-memory execution declines safely to an exact bounded fallback.

### Q3

- Retain full SF1000 at or below 1.05x the 192.49s accepted result.
- Retain deterministic native reduction, zero fallback, and bounded terminal D2H.

### Q7

- Immediately restore or match the prior public plan's 84.09s full-scale result.
- A future streaming accumulator must beat that result while remaining bounded
  by `O(batch + k)` device memory and exact pandas ordering semantics.
- Allocation events must not grow from ~52k to ~137k merely to avoid 0.6 MiB D2H.

### Q1

- Add stage timing before another implementation change.
- Demonstrate a paired improvement in the dominant stage and full-query median.
- Preserve exact 100-row ordering and zero fallback.

### Cross-query

- SF1 and SF100 remain 12/12 correct.
- Q5, Q10, Q11, and Q12 regress by no more than 5% in paired medians.
- Multi-tier rails include the large ancestral/small selected Q6 shape rather
  than scaling fact rows alone.
- One clean upstream SF1000 confirmation remains 12/12 correct.
- No production branch inspects benchmark or hardware identity.

## Suggested delivery order

1. Apply and generalize the attached Q6 reuse-admission proof; add its specific
   large-ancestor/small-derivative regression test.
2. Restore the prior committed Q7 public plan.
3. Fix the memory-envelope-sensitive coverage-width test.
4. Retain Q3 unchanged and run targeted/cross-query tests.
5. Add Q1 stage timing and collect a multi-tier profile before changing Q1.
6. Run bounded proof-shaped rails, SF100 correctness/performance, and then one
   clean SF1000 confirmation.
