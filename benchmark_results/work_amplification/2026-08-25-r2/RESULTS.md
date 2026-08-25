# R2 Work-Amplification Results

Status: complete

## Decision table

| Hypothesis | Baseline | Alternative | Complete result | Decision |
|---|---:|---:|---:|---|
| Q11 classify both endpoint relations | 311.46s | classification once 238.32s | 23.5% faster, exact; H200 8.0% faster | graduated and already implemented |
| 1M grouped constructive pages | 64.095s | reduce before construct 11.116s | 5.766x faster, identical fingerprint | graduated and already implemented |
| Q12 indexed nearest | 23.357s | dense bbox certificate 24.205s | dense arm 3.63% slower; numeric oracle passes | archive dense arm for current shape |
| Q11 parent MultiPolygon refinement | 2.187s | component-first 2.100s | 3.98% faster before parent regroup | continue as research only |

The first two controls are general physical-shape wins already present in the
current library. The current research therefore does not claim them as new
speedups. They validate that the amplification method ranks mechanisms that
have produced large public-workflow gains.

## Q11 classification once

The retained forced A/B classified aligned pickup and dropoff memberships
independently in the baseline, then reused exact dropoff classifications from
the pickup conservative superset and classified only unseen candidates.

- public Q11: 311.46s to 238.32s, 23.5% faster;
- exact candidates: 11.266B to 7.981B, 29.2% fewer;
- exact-kernel time: 290.165s to 210.904s, 27.3% lower;
- normalized result: byte-identical;
- H200 public Q11: 116.43s to 107.16s, 8.0% faster.

Fresh R1 current execution remains exact at 226.59s lean and records 7.981B
classification-once candidates. Preparation builds only five region indexes,
so repeated rebuild was correctly rejected as the cause.

## Grouped constructive reduction

The current-revision control replays the pre-rewrite public workflow: construct
parcel-zone fragments in pages, dissolve every page, persist/reload pages, then
dissolve again. It completes in 64.095s at 1M. The current public workflow
reduces certified source and grouped zone coverages before the few-right
intersection and completes in 11.116s in the final 1M shootout.

Both arms produce:

```text
rows=4 bounds=(599.05, 309.85, 850.0, 738.62) convex_hull_area=279078.10
```

The current comparison is a 5.766x speedup and saves 52.979s. Both arms carry
source identity `708b2b41` and emit the same fingerprint. The paged control's
isolated process requires
`PYTHONPATH=/home/picard/repos/vibeSpatial/benchmarks/shootout` to import its
shared `_data` fixture module; `README.md` retains the canonical command. The older
311.996-second workflow is not used for this A/B because it also lost operation
provenance before the dense existential query.

## Q12 dense regular work

The current identified indexed output completes in 23.357s, including 3.998s
in the nearest stage. The identified dense output performs 6.115B tiled bbox
lower-bound tests,
320,000 seed exact pairs, 100,000 final exact pairs, and public exact distance
refinement. It completes in 24.205s; its nearest work totals 4.848s.

Ordered trip keys are identical. Maximum absolute and relative distance deltas
are `6.52e-6` and `7.90e-8`, passing `rtol=1e-6, atol=1e-9`. The dense arm's
historical 7.42% win does not survive the improved current indexed path: dense
is 3.63% slower end to end and is archived without production work.

## Component-first point-region prototype

One 3,896,103-row Q11 trip batch was tested against all five zone partitions.
Zone preparation exploded 1,033,509 parent MultiPolygons into 1,300,118 Polygon
components outside timing. Both timed arms used the public paired aggregate.
The first four rows below come from the lean decision arms. The final three
come from separate instrumented profile arms and are attribution-only.

| Metric | Parent | Component | Change |
|---|---:|---:|---:|
| lean complete aggregate wall | 2.187s | 2.100s | -3.98% |
| lean exact-classification work units | 73.150M | 69.168M | -5.44% |
| lean exact-classification GPU span | 1.156s | 1.071s | -7.37% |
| lean candidate-generation GPU span | 0.122s | 0.151s | +23.60% |
| profiled parts considered | 10.496B | 50.161M | -99.52% |
| profiled active parts | 39.168M | 29.077M | -25.76% |
| profiled edge visits | 29.670B | 26.262B | -11.49% |

Left and right membership arrays are byte-identical. Component shared counts
are lower by 1,062 because aligned endpoints can occupy different components
of one parent; a production lowering must perform stable tri-state
component-to-parent reduction before consumer semantics.

This is not yet a graduated workstream. The 99.5% drop in cheap part-loop work
becomes only an 11.5% edge reduction and a 4.0% pre-reducer wall gain. A parent
reducer, component lineage carrier, readiness contract, and protected simple
Polygon cases could consume that margin. The next experiment must measure the
complete parent-aware reducer before any production plan or selector exists.

## R3 disposition

The research program successfully rediscovers two already-landed general wins
and falsifies the formerly winning Q12 dense alternative against the current
candidate. Component-first does not satisfy the graduation gates yet.

The next queued hypothesis is parent-aware component-first point-region
lowering across Q10 and Q11. It remains an experiment until it proves exact
tri-state parent semantics, bounded memory, zero fallback/materialization, a
complete public-stage win after reduction, and protected small/simple shapes.
No generic planner or automatic selector is justified by the present data.

## Regression floor

The final public 10K rerun reuses the validated static GeoPandas comparator,
reruns every vibeSpatial candidate three times after warmup, and passes 14/14
exact fingerprints. The vibeSpatial subtotal is 2.583s versus 3.524s for the
retained comparator and 2.714s in R1.

The current 1M candidate-only diagnostic completes 13 strict-native workflows
in a 158.427s subtotal. `corridor_flood_priority` retains its explicit off-ramp
because mixed-family buffer is unsupported in strict-native mode. The invalid
comparator provides no fingerprint, GeoPandas timing, or speedup claim for this
run.

The final mandatory full profile has 22 successful and 2 deferred pipelines.
Every active pipeline reports zero compute D2H, materialization, and fallback.
The slowest complete 1M pipeline is `grouped-capacity-partitions` at 0.176s;
its largest compute stage is the 67.2ms `mixed_strip_exact_union` canary. No
stage crosses the one-second CPU-heavy
investigation threshold.

The final current-source SF100 lean run succeeds 12/12 in 464.88s. The
separate Level-0 counter/telemetry run also succeeds 12/12 in 464.32s, a 0.12%
difference. `SF100_RUN_IDENTITY.json` binds both cold one-run artifacts and
their result directories to source identity `708b2b41`, the same dataset and
machine environment, and their canonical commands.
