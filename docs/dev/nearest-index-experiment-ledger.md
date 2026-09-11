# Nearest Index Experiment Ledger

<!-- DOC_HEADER:START
Scope: Exact GPU nearest index experiments, hypotheses, timings, parity, and decisions.
Read If: You are optimizing nearest search, GPU STRtree alternatives, or road snapping at 100K and 1M scale.
STOP IF: You need the original public API comparator contract rather than current experiments.
Source Of Truth: Ongoing nearest index experiment ledger and production graduation criteria.
Body Budget: 233/240 lines
Document: docs/dev/nearest-index-experiment-ledger.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-10 | Intent |
| 11-17 | Request Signals |
| 18-28 | Open First |
| 29-34 | Verify |
| 35-53 | Acceptance Contract |
| 54-60 | Risks |
| 61-79 | 2026-09-10 / E0 / Starting Evidence |
| 80-107 | 2026-09-10 / E1 / Competing Experiments |
| 108-143 | 2026-09-10 / E2 / Packed Binary Hierarchy, FP64 |
| 144-184 | 2026-09-10 / E3 / Matched Repetitions and Host Output |
| 185-233 | 2026-09-10 / E4 / Generalization and Graduation |
DOC_HEADER:END -->

## Intent

Find an exact, reusable GPU nearest index that reaches Shapely parity at
100,000 queries and beats it at 1,000,000 queries through a simple public API.
Start with bounded experiments before changing production dispatch. This work
completes native nearest functionality; no approximate-distance contract is
being introduced. Entries record unsuccessful experiments as well as wins.

## Request Signals

- nearest index experiments
- GPU STRtree
- road snapping performance
- nearest experiment ledger

## Open First

- docs/dev/nearest-index-strategy-ledger.md
- scripts/experiment_nearest_hierarchy.py
- docs/testing/nearest-index-experiment-evidence.json
- docs/testing/osm-nearest-performance.md
- docs/testing/osm-nearest-evidence.json
- src/vibespatial/spatial/nearest.py
- src/vibespatial/spatial/spatial_index_knn_device.py
- src/vibespatial/api/sindex.py

## Verify

- `uv run pytest tests/test_experimental_nearest_hierarchy.py tests/test_benchmark_osm_nearest.py --run-gpu -q`
- `uv run python scripts/experiment_nearest_hierarchy.py --smoke`
- `uv run python scripts/check_docs.py --check`

## Acceptance Contract

The target is `tree = roads.sindex` followed by
`tree.nearest(points, return_all=True, return_distance=True)`, as easy as
constructing and querying Shapely STRtree. Callers must not tune radii, manage
GPU buffers, or split long roads by hand. Index reuse must work for k=1.

Initially, 100K and 1M mean building query rows against the same full
Massachusetts road set: 991,441 roads / 8,728,450 segments. Query points are
nested prefixes of the existing seed-20260910 statewide shuffle, EPSG:26986.
Also test independent spatial distributions before choosing a general design.
10K is diagnostic, not the optimization target.

Require exact query/target tie-pair multisets; distance tolerance is 1e-6 m
absolute plus 1e-10 relative. A tolerance must never manufacture extra ties.
Compare completed build + first query, repeated query, and ingress + build +
query separately, including public result export. Report startup/JIT separately.
Prototype timings do not establish public API acceptance.

## Risks

- Segment tie identity differs from parent-road tie identity.
- Floating-point parity on sampled fixtures is not a pruning-error proof.
- Cached-compilation timings exclude clean-install startup.
- Overlapping bounds and intrinsically large tie outputs can amplify work.

## 2026-09-10 / E0 / Starting Evidence

Source HEAD `6c09268`, production source unchanged from `cd89cde`.
RTX 4090 24 GiB, i9-13900K. The immutable comparator and prepared WKB fixtures
are in `/tmp/vibespatial-osm/nearest-v2-baseline/`; hashes, packages, preparation,
and timing contracts are in the linked checked-in baseline evidence.

At 100K queries, Shapely repeated queries take 1.779 s against whole roads or
0.800 s against segments. The current public GPU road query takes 5.511 s and
fails tie parity. Segment queries completed in about 61 s before the worker
timed out. The statewide road query attempts 2.93 billion candidate pairs and
exceeds the device relation budget. There is no existing 1M timing; measure it.

The k > 1 gate only controls passing a retained NativeSpatialIndex to the
nearest engine. The device fixed-k engine itself supports k=1. However,
return_all=True takes a separate progressive-radius candidate path; changing
that gate alone cannot repair all-match search. NativeSpatialIndex currently
retains a flat Morton index, not an explicit packed node hierarchy.

## 2026-09-10 / E1 / Competing Experiments

1. Reuse the existing bounded, tiled fixed-k engine with k=1. This tests whether
   caching and its existing pruning suffice. Exactly-one output is only a
   diagnostic: a subsequent exact tie pass would still be necessary.
2. Build a packed GPU bounding hierarchy over spatially ordered segments.
   Traverse by point-to-box lower bounds, refining only visited segments.
   Keep index buffers resident and avoid a global query-by-road relation.
   First test explicit FP64 arithmetic to separate search shape from precision
   tuning. Compare Morton packing with spatial tiling only if shape warrants it.

Physical contract for experiment 2: segment-shaped construction, query/node
visits plus segment refinements as work units, retained device bounds/order,
and variable-size tie output. Nearest/count then scan/scatter bounds output
memory by actual matches. Host work is limited to tree-level scheduling,
allocation counts, fixture preparation, and terminal result export. Production
graduation requires NativeSpatialIndex ownership/invalidation and NativeRelation
output, PrecisionPlan policy, warmup, dispatch coverage, and end-to-end profiling.

Existing fixed-k diagnostic outcome: completed cached index build in 39 ms,
but its first 1K query against all 8.73M segments did not finish before a
90-second whole-process timeout (exit 124). WKB ingress was 534 ms. Bounds
preparation is excluded from this diagnostic's query timer. No query duration,
parity result, or speedup is available; cold compilation may contribute.
Evidence: `/tmp/vibespatial-osm/nearest-experiments/e1-fixed-k-1k.json` and its
source snapshot; wrapper log `/tmp/vibespatial-osm/e1-fixed-k-1k.log`.
Decision: do not spend the initial experiment budget tuning this path.

## 2026-09-10 / E2 / Packed Binary Hierarchy, FP64

Prototype: `scripts/experiment_nearest_hierarchy.py`. Morton-sort segment
centres, pack eight segments per leaf, reduce parent bounds bottom-up. Query
threads visit nearer children first. First pass finds the minimum and counts
exact ties; scan plus second traversal emits the relation. No global candidate
relation is allocated. CUDA FMA contraction is disabled in this reference
variant to preserve the oracle's binary64 metric operation order. A conservative
traversal pad broadens pruning only; match equality has no distance tolerance.

Six adversarial smoke cases passed: leaf widths 1/8/32 with original and +1e7
coordinates, including duplicate/degenerate segments, near ties, endpoints,
long crossing edges and far queries. All 1,010 pairs and distances matched in
each case. This is evidence, not a general floating-point pruning proof.

Initial 100K MA run: one first-process trial then one fresh-input warm trial.
Warm WKB ingress 431 ms, completed build 13.9 ms, first query 65.0 ms, repeated
query 65.2 ms. All four query outputs match all 105,748 Shapely pairs exactly;
maximum distance difference is zero. First pass averages 685 node visits and
324 segment evaluations per query. Retained index buffers occupy 448,441,928
bytes (427.7 MiB). Query timing includes both traversals and host output.
Raw evidence and exact source snapshot:
`/tmp/vibespatial-osm/nearest-experiments/e2-hierarchy-100k.{json,source.py}`.

Decision: pursue the hierarchy experiment at 1M and repeat measurements.
The search shape is credible even before mixed-precision tuning. Public API
acceptance remains pending: this restricted experiment consumes pre-segmented
finite inputs and does not yet own a NativeSpatialIndex/NativeRelation.

At 1M, one first-process plus three warm-process trials all passed exact
parity: 1,057,555 pairs, zero distance difference, eight checked outputs.
Warm medians: ingress 458 ms, build 13.9 ms, first query 507 ms, repeated
query 509 ms. First-pass node visits average 686 per query. Evidence:
`/tmp/vibespatial-osm/nearest-experiments/e2-hierarchy-1m.{json,source.py}`.
This measurement includes both traversals, result allocation and host export.

## 2026-09-10 / E3 / Matched Repetitions and Host Output

Final prototype exports int64 pairs and float64 distances, matching the public
boundary. E2 exported int32 pairs; reran the candidate after that correction.
Fresh Shapely measurements establish the previously absent 1M comparator and
three-warm-trial timing contract. Its code is unchanged by the candidate export
correction. Each backend/scale runs sequentially in a separate process. Every
first/repeated output passes parity. First-process timing is separate from the
three warm-process medians; compilation caches can already be warm.
The [evidence JSON](../testing/nearest-index-experiment-evidence.json) preserves
all final trials, fixture/oracle hashes, source hashes, environment and combined
medians. Exact script snapshots remain beside the local raw JSON artifacts.

Seconds against 8,728,450 segments; host pair/distance export included:

| Queries | Backend | Build | First query | Repeated query | Build + first | Ingress + build + first |
|---|---|---:|---:|---:|---:|---:|
| 100K | Shapely | 1.375 | 0.805 | 0.801 | 2.179 | 4.117 |
| 100K | prototype | 0.014 | 0.065 | 0.065 | 0.079 | 0.509 |
| 1M | Shapely | 1.377 | 7.959 | 7.987 | 9.336 | 11.367 |
| 1M | prototype | 0.014 | 0.508 | 0.510 | 0.522 | 0.982 |

Repeated-query speedups: 12.24x at 100K and 15.66x at 1M. Combined columns are
medians of each trial's sum, not sums of independent medians. All 105,748 / 
1,057,555 pairs match exactly, with zero observed distance difference. This
establishes the search strategy's potential, not the public API target.

Reproduce each cell with a new output path (existing results are never replaced):

```bash
uv run python scripts/experiment_nearest_hierarchy.py \
  --backend hierarchy --rows 1000000 --repeat 3 \
  --output /tmp/vibespatial-osm/nearest-experiments/new-hierarchy-1m.json
```

Use `--backend shapely` for the comparator and `--rows 100000` for 100K.
Both use the existing validated fixture cache by default. Do not rerun the
unchanged comparator after candidate-only edits when its identity still holds.
Raw final artifacts: `nearest-experiments/e3-{shapely,hierarchy}-{100k,1m}.json`
under `/tmp/vibespatial-osm/`. Original public baseline remains immutable.

## 2026-09-10 / E4 / Generalization and Graduation

Ten targeted tests passed, including independent 100K-segment / 100K-query
uniform and clustered fixtures, a single leaf, and 263,425 exact ties across
leaf widths 1/8/32, plus leaf-width overflow rejection. These are correctness
canaries, not timed performance
claims. The six original translated-coordinate smoke cases also passed.
CUDA Compute Sanitizer memcheck reports zero errors for those six cases.
Local logs are `nearest-experiments/{pytest-final,memcheck}.log` under
`/tmp/vibespatial-osm/`.
The required full pipeline profile completed: 22 executed cases passed, with
two existing raster-to-vector cases explicitly deferred by the suite. Reviewed
all 102 stages (51 at 1M); none exceeded 1 s. At 1M, the largest stage was
`grouped-capacity-partitions/mixed_strip_exact_union`, 75.34 ms, followed by
`predicate-heavy/read_geojson`, 70.03 ms. Stage names/times are preserved in the
evidence JSON; full sparkline log: `nearest-experiments/full-profile.log`.
This suite checks existing pipelines; the isolated nearest measurements above
exercise the new experiment itself.

Decision: integrate a reusable bounding hierarchy beneath NativeSpatialIndex;
there is enough evidence to proceed without tuning FP32 arithmetic first.
Literal CPU STR packing is not a prerequisite: the measured prototype uses
Morton ordering and a binary bounding hierarchy. Preserve that as one candidate,
then compare packing/traversal alternatives on measured failure distributions.

Next work, in dependency order:

1. Extract segments on device from full road buffers, retain original row IDs,
   and deduplicate parent-road ties. Prove whole-road Shapely parity; the current
   segment oracle alone does not establish this public contract.
2. Cache the hierarchy in NativeSpatialIndex for k=1, with input ownership,
   lineage validation, readiness and invalidation. Produce NativeRelation;
   keep host conversion at the existing public `.sindex.nearest` boundary.
3. Establish numeric pruning error bounds and ranking/tie refinement under the
   normal PrecisionPlan. Exercise nulls/empties, views, mixed line lengths,
   near/far distances, large offsets, degenerate edges, exclusion and limits.
   The reference's finite-coordinate admission does not prove overflow safety
   for extreme binary64 magnitudes, or general geometric robustness.
4. Add capacity admission, warmup/registry coverage and observability. The
   prototype retains 428 MiB of index arrays plus input, scratch and output;
   it does not implement production memory-budget enforcement. Traversal has
   bounded stack memory, but heavily overlapping bounds can still require
   many visits, and all-tie output can be intrinsically large.
5. Rerun unchanged comparators against the final public path only when needed
   by identity changes. Measure auto/GPU mode, full road inputs, resident reuse,
   first use and end-to-end pipeline profiles before claiming completion.

Status: initial experiments complete; production implementation and public API
acceptance are pending. Work remains local; no commit or push was requested.
