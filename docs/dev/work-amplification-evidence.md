# Work-Amplification Evidence

<!-- DOC_HEADER:START
Scope: Current measured evidence, reliability decisions, ranked hypotheses, and experiment outcomes for the work-amplification research program.
Read If: You are choosing the next physical-shape performance investigation or validating a work-amplification claim.
STOP IF: You need the research methodology or an operation-specific implementation contract rather than current results.
Source Of Truth: Ranked work-amplification evidence and the disposition of measured hypotheses.
Body Budget: 221/240 lines
Document: docs/dev/work-amplification-evidence.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-17 | Request Signals |
| 18-25 | Open First |
| 26-32 | Verify |
| 33-39 | Risks |
| 40-53 | Evidence Contract |
| 54-76 | R0 Capture |
| 77-103 | R1 Capture And Observer Control |
| 104-138 | Ranked R1 Map |
| 139-176 | R2 Counterfactuals |
| 177-203 | Component-First Follow-Up |
| 204-221 | Decisions |
DOC_HEADER:END -->

## Intent

Retain the measured amplification map, observer-effect decisions,
counterfactual outcomes, and current research disposition separately from the
methodology in `docs/dev/work-amplification-research-plan.md`.

## Request Signals

- work-amplification evidence
- ranked physical-shape findings
- counterfactual results
- observer effect
- Q10 or Q11 amplification
- component-first point region

## Open First

- `docs/dev/work-amplification-research-plan.md`
- `benchmark_results/work_amplification/2026-08-25-r1/README.md`
- `benchmark_results/work_amplification/2026-08-25-r2/RESULTS.md`
- `docs/dev/point-region-execution-evidence.md`
- `docs/dev/grouped-constructive-distributive-execution-plan.md`

## Verify

- `uv run python scripts/analyze_work_amplification.py <artifact...>`
- `uv run pytest tests/test_work_amplification_analysis.py -q`
- `uv run python scripts/check_docs.py --check`
- validate each evidence directory with its `SHA256SUMS`

## Risks

- Counter and full-profile walls are not interchangeable with lean timing.
- A high ratio can describe cheap rejected work rather than recoverable wall.
- Historical A/B evidence cannot replace a current candidate rerun.
- Current experimental wins cannot become selectors before graduation gates.

## Evidence Contract

This is the living result ledger for
`docs/dev/work-amplification-research-plan.md`. Raw artifacts are retained under
`benchmark_results/work_amplification/`. A ratio is never ranked without an
absolute wall-time or memory consequence. Public rows are not substituted for
physical pairs, fragments, edges, or capacity.

Shootout timed medians and isolated post-timing profiles are distinct
executions. Profile timings are accepted for attribution only when their total
is consistent with the lean execution. SF100 cold-one timings identify the
queue but are not distribution estimates. Every derived number below is
arithmetic over fields in the retained artifacts.

## R0 Capture

The uninstrumented capture is revision `e8e7f22`, on `picard-4090` with an
Intel i9-13900K, RTX 4090 24 GiB, driver 580.173.02, and local NVMe storage.

| Rail | Protocol | Result |
|---|---|---|
| 10K shootout | repeat 3, warmup, static GeoPandas refreshed once | 14/14 exact; VS 2.732s subtotal; GPD 3.524s subtotal |
| 1M shootout | repeat 1, no warmup, strict native, VS-only diagnostic | 13/14 native successes; 161.776s subtotal |
| Full pipeline | repeat 1, strict native, GPU sparkline | 22 successful, 2 deferred |
| SF100 | 12 isolated queries, cold one, strict native | 12/12; 468.17s total |

The 1M corridor-flood-priority workflow took an explicit observable off-ramp
because mixed/null/empty buffer input was unsupported in strict-native mode. It
is not counted as a native success.

The conservative offline R0 analyzer emits 1,424 schema-valid records and zero
automatic findings. This is intentional: existing artifacts establish wall
cost and some structural context, but do not yet pair attributable physical
amplification with that cost. Five provisional memory findings were withdrawn
after review showed that materialization-event `bytes` described source
geometry carrier size, not transfer, allocation, or peak-live memory.

## R1 Capture And Observer Control

R1 adds versioned Level-0 packets to existing hotpaths and one opt-in Level-1
point-region reduction. Counter mode records bounded host-known values without
NVTX, per-stage timers, or CUDA synchronization. Full mode retains the prior
synchronized diagnostic behavior.

| Rail | R1 captured result |
|---|---|
| 10K shootout | 14/14 exact; VS 2.714s; validated static GeoPandas 3.524s |
| 1M VS-only diagnostic | 13/14 native; 159.608s subtotal; same explicit corridor off-ramp; superseded by the current-source R2 rerun |
| Full pipeline | 22 successful, 2 deferred; zero compute D2H/materialization |
| SF100 lean | 12/12; 465.39s |
| SF100 counters | 12/12; 464.55s, 0.998x lean |

Counter-only vegetation replay is 52.51s versus 52.04s lean, closing the prior
8.378x full-profile observer effect. Transit initially failed because the
post-run ownership audit called `DeviceGeometryArray.to_owned()` and attempted
to concatenate a 30,553,577-row partitioned result into another 1.142 GiB
carrier. The audit now observes only cached owned arrays or existing
composition parts. Transit counter replay is 30.70s versus 31.05s lean, remains
partitioned, and records zero fallback.

The offline analyzer consumes deep-copied JSON artifacts and emits 3,295
schema-valid records, 62 ranked findings, and 27 observer-effect records. It
does not affect timed execution or production dispatch.

## Ranked R1 Map

The dominant exact point-region signal is now physical rather than inferred.
Fresh Q11 Level-1 evidence records 7.981B candidates, 1.611T parts considered,
6.028B active parts, and 4.564T selected-bin edge visits. Exact kernels consume
about 185.09s. Only five prepared indexes are built and each is reused 461
times, rejecting index build as the main cost.

Across the five region groups:

- candidate parts per lane range from 24.1 to 623;
- edge visits per lane range from 324 to 913;
- edge visits per survivor range from 1,800 to 3,960;
- 3.058B candidates have no active part.

Q10 and Q11 together remain 341.47s, 73.4% of the current SF100 run, and use
the same prepared point-region refinement family. Q10 executes 770 point-region
calls over 3.741B Level-0 candidate lanes; Q11 executes 2,310 calls over
11.266B Level-0 lanes before classification-once exclusion. Preparation has
five builds and thousands of cache hits in both cases.

Constructive group-compression packets rank vegetation seam stitching at
51.48s with about 65,500 input rows per output group. Habitat tile clipping is
38.37s with about 1,080 rows per output group. These are queue signals, not
proof that the terminal geometry can legally discard fragments: output
coordinates and exclusive stage attribution remain unavailable.

Capacity-only SF100 findings provide memory context, not automatically
recoverable work. Q11 peaks near 9.25 GB in the telemetry artifact; cumulative
allocation traffic is churn and is not labeled peak-live memory.

The pipeline negative control remains important: nearly one million examined
pairs can finish in tens of milliseconds. Ratios rank only when paired with
material attributable wall or memory.

## R2 Counterfactuals

Three independent public-workflow shapes were adjudicated. Frozen historical
controls are retained when their source identity, exact result, and mechanism
remain valid; the current vibeSpatial candidate is rerun. The final R2 capsule
also refreshes the public regression rails on source identity `708b2b41...`:
10K is 14/14 exact at 2.583s, 1M has 13/14 native completions at 158.427s with
the same explicit corridor off-ramp, and SF100 is 464.88s lean versus 464.32s
with counters. The current full pipeline has 22 successes, 2 deferred, and
zero compute D2H, materialization, or fallback.

| Counterfactual | Baseline | Alternative | Current decision |
|---|---:|---:|---|
| Q11 independent endpoint classification vs classification once | 311.46s | 238.32s | 23.5% faster, exact; already implemented |
| 1M paged grouped construction vs reduce before construct | 64.095s | 11.116s | 5.77x faster, exact; already implemented |
| Q12 indexed nearest vs dense bbox certificate | 23.357s | 24.205s | dense is 3.63% slower; archive |

Classification once removes 29.2% of exact candidates and 27.3% of exact
kernel time. It is also supported on H200, where public Q11 improves 8.0%.
Fresh R1 Q11 remains exact at 226.59s and retains the 7.981B-candidate shape.

The current paged constructive control and reduced workflow produce the same
canonical 1M fingerprint:

```text
rows=4 bounds=(599.05, 309.85, 850.0, 738.62) convex_hull_area=279078.10
```

The 52.979s current-revision saving is credited only to the fair 64.095s paged
control after provenance continuity. The older 311.996s result also contained
a separate lost-provenance defect and is not used for this A/B.

Current Q12 reverses the earlier dense win. Indexed nearest is 3.998s inside a
23.357s query; the dense certificate and exact refinement total 4.848s inside
a 24.205s query. Ordered keys match and distances pass `rtol=1e-6, atol=1e-9`.
The dense shape is archived instead of being tuned against a now-faster
candidate.

## Component-First Follow-Up

The next Q11 falsifier explodes parent MultiPolygons into Polygon components
outside timing, queries identical public point indexes against component
bounds, and measures the margin available for a future stable parent reducer.
One 3,896,103-row trip batch against all five zone partitions records. The
aggregate wall is from the lean decision arms; every other row is from the
separate instrumented profile arms and is attribution-only:

| Metric | Parent | Components | Change |
|---|---:|---:|---:|
| lean aggregate wall before parent regroup | 2.187s | 2.100s | -3.98% |
| profile candidate lanes | 52.008M | 50.161M | -3.55% |
| profile exact-classification GPU span | 1.241s | 1.093s | -11.91% |
| profile parts considered | 10.496B | 50.161M | -99.52% |
| profile edge visits | 29.670B | 26.262B | -11.49% |

Left and right membership arrays are byte-identical. Component shared counts
are lower by 1,062 because endpoints may occupy different components of the
same parent. Production semantics require tri-state component-to-parent
reduction, stable lineage, hole preservation, and parent-level deduplication.

The 99.5% part-loop reduction becoming only 4.0% pre-reducer wall is the key
result: most removed parts were cheap rejects, while edge visits fall only
11.5%. This is promising enough for one parent-aware reducer experiment but
does not justify a production carrier or selector.

## Decisions

- Instrumentation remains profiling-only and cannot influence runtime
  selection. Counter mode is the observer-safe default for broad replays.
- Ownership auditing must never physicalize a partitioned native composition.
- Existing classification-once and reduce-before-construct work are the first
  graduated general mechanisms; this program validates rather than reclaims
  their already-landed gains.
- No new production workstream graduates from R2. Q12 dense filtering is
  archived, and component-first remains an experiment until parent regroup is
  measured end to end.
- The final 10K floor is 14/14 exact with a 2.583s vibeSpatial subtotal versus
  the reused 3.524s comparator. The final full profile is 22 successful and 2
  deferred, with zero compute D2H, materialization, or fallback.
- Q10/Q11 point-region refinement remains the highest current recoverable-wall
  queue. The next admissible experiment is parent-aware component lowering,
  not another point-index provider, warp scheduler, generic planner, or
  device-name selector.
