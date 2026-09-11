# Nearest Index Strategy Ledger

<!-- DOC_HEADER:START
Scope: Comparative nearest acceleration experiments: Morton, STR, wide trees, grid seeds and whole-road provenance.
Read If: You are identifying the best GPU nearest strategy or comparing index packing and traversal designs.
STOP IF: You need the initial public API gap measurement or isolated reference hierarchy only.
Source Of Truth: Continuation ledger for nearest algorithm comparison and empirical acceleration recommendation.
Body Budget: 177/260 lines
Document: docs/dev/nearest-index-strategy-ledger.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-10 | Intent |
| 11-17 | Request Signals |
| 18-26 | Open First |
| 27-31 | Verify |
| 32-39 | Risks |
| 40-95 | 2026-09-10 / E5 / Search Matrix |
| 96-129 | 2026-09-10 / E6 / Screening Results |
| 130-158 | 2026-09-10 / E7 / Direct Grid, No Hierarchy |
| 159-177 | 2026-09-10 / E8 / Whole Roads and Full State |
DOC_HEADER:END -->

## Intent

Identify the best measured acceleration path for building-to-road nearest
queries, comparing established and exploratory approaches. Continue the
[initial experiment ledger](nearest-index-experiment-ledger.md). Keep evidence
and decisions local. An empirical recommendation is not proof of a globally
optimal algorithm, nor a claim that experimental code is production ready.

## Request Signals

- nearest acceleration strategy
- GPU STRtree alternatives
- nearest index experiments
- whole-road nearest

## Open First

- docs/dev/nearest-index-refinement-ledger.md
- scripts/experiment_nearest_strategies.py
- scripts/nearest_strategy_kernels.py
- tests/test_experimental_nearest_hierarchy.py
- docs/dev/nearest-index-experiment-ledger.md
- docs/testing/nearest-index-experiment-evidence.json

## Verify

- `uv run pytest tests/test_experimental_nearest_hierarchy.py --run-gpu -q`
- `uv run python scripts/check_docs.py --check`

## Risks

- Leaf tiling alone does not implement recursive STR packing.
- An approximate seed is safe only when a complete traversal verifies it.
- Parent-road ties require deduplication on original road IDs.
- A packing winner can depend on query order, geometry overlap, and reuse.
- Screening with one warm trial is weaker than the final three-trial evidence.

## 2026-09-10 / E5 / Search Matrix

The previous phase made concrete progress: a packed Morton binary hierarchy
passed same-data parity and beat Shapely at 100K and 1M query scale. Its tests
and artifacts were re-inspected before continuing. Current production source
is still `6c09268`; the GPU is idle between sequential experiment workers.

Remaining research questions:

1. Does coherent query ordering dominate layout improvements?
2. Does a 4/8-way hierarchy reduce depth enough to offset child sorting cost?
3. Does STR packing beat Morton construction after accounting for build cost?
4. Can a cheap grid seed improve pruning without weakening exactness?
5. Does the path still win when it accepts whole roads and returns road IDs?

Physical contract: device scan expands variable-length road buffers to segments
and provenance; packed bounding nodes retain device coordinates and IDs;
query/node traversal yields a counted relation; sort/unique deduplicates
parent-road pairs before terminal NumPy export. Work is measured in segment
construction, node visits, segment evaluations, tie pairs, bytes and wall time.
No global query-by-road candidate matrix is materialized.

Variants preserve the reference FP64 metric and tie test. Tree fanout is a
compile-time 2/4/8 parameter, with host admission proving the stack bound.
Optional Morton query sorting includes its cost and restores original query
IDs. Optional 1024x1024 seeding stores one segment per occupied cell and tests
the 3x3 neighborhood. Seeds only establish a candidate upper bound: the full
hierarchy still verifies the minimum and all ties, including outside-grid queries.

STR-style leaf tiling sorts segment centres into equal-population x slices,
then by y within each slice. This preliminary variant keeps contiguous upper
nodes; a result for it cannot establish a conclusion about full recursive STR.

The initial implementation exposed an unsupported CuPy operation: device
repeat counts cannot be passed to `cp.repeat`. Replaced it with segment-start
flags and a device inclusive scan, using the known coordinate/row counts for
allocation. No host geometry expansion is performed.

Twelve whole-road canaries passed across fanout, packing, query ordering and
seeding, including varied vertex counts, duplicate roads, degenerate edges,
large offsets and far queries. Log:
`/tmp/vibespatial-osm/nearest-experiments/e5-tests-v2.log`.

Initial 1M screening uses all 8,728,450 segments and the same shuffled query
prefix. One first-process trial plus one fresh-input warm trial per worker;
both first/repeated query outputs are validated. Seven workers run sequentially,
with a 120-second whole-worker timeout. Raw results, exit status, and exact
dependency snapshots are under `nearest-experiments/e5-*-1m.*` in
`/tmp/vibespatial-osm/`. Finalists will receive three warm trials at both target
scales before formal comparison to the validated E3 Shapely comparator.

Background references: [Shapely STRtree contract](https://shapely.readthedocs.io/en/stable/strtree.html),
[parallel GPU hierarchy construction](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees),
and [CCCL compute primitives](https://nvidia.github.io/cccl/unstable/python/compute_api.html).
The experiment uses repository primitive wrappers and runtime compilation.

## 2026-09-10 / E6 / Screening Results

All cases below pass all four checked outputs at 1M queries. Times are one warm
trial, include query sorting and host export, and are not final medians.

| Structure | Query order | Seed | Build ms | Repeated ms | Mean nodes | Mean segment evaluations |
|---|---|---|---:|---:|---:|---:|
| Morton binary, child pruning | input | no | 16.9 | 331.8 | 356.0 | 247.7 |
| Morton binary | Morton | no | 16.9 | 182.1 | 356.0 | 247.7 |
| Morton binary | Morton | grid | 18.2 | 70.2 | 82.8 | 51.7 |
| Morton 4-way | Morton | no | 17.1 | 126.0 | 231.3 | 194.6 |
| Morton 8-way | Morton | no | 16.9 | 112.6 | 214.4 | 152.2 |
| STR leaves, 8-way | Morton | no | 18.4 | 61.5 | 95.2 | 45.3 |
| Recursive STR, 8-way | input | no | 19.2 | 84.6 | 84.1 | 38.5 |
| Recursive STR, 8-way | Morton | no | 19.5 | 50.4 | 84.1 | 38.5 |
| Recursive STR, 8-way | Morton | grid | 20.6 | 34.2 | 18.9 | 30.7 |
| Recursive STR, 4-way | Morton | no | 20.6 | 54.7 | 78.4 | 44.6 |
| STR leaves, 8-way | Morton | grid | 19.1 | 46.0 | 25.7 | 31.8 |
| Morton 8-way | Morton | grid | 18.1 | 50.3 | 31.2 | 48.8 |

Recursive STR re-sorts child bounding-box centres at every level and stores
explicit child references, with no power-of-fanout padding. This is distinct
from the earlier leaf-only tiling. Its seeded eight-way version retains 363.3
MB of index buffers versus 452.6 MB for seeded binary Morton. E6 adds 18
passing whole-road canaries across recursive/nonrecursive packing and traversal
options. Source and result snapshots: `nearest-experiments/e6-*-1m.*`.

Interpretation: query locality, tighter upper bounds, and hierarchy quality
each matter independently. Spatial sorting cannot reduce visit counts, but
improves execution coherence. Grid seeding reduces first-pass work; it never
replaces the exact verification traversal. Recursive STR is currently the
strongest measured structure. A direct grid query remains an untested
alternative, and whole-road end-to-end timings are still required.

## 2026-09-10 / E7 / Direct Grid, No Hierarchy

Implemented a separate exact search family: replicate segment bounds into a
uniform grid, search cell rings until the unvisited boundary certifies the
minimum, then count/emit ties once using a canonical cell. All three grid
canaries pass, including boundary ownership, long crossing roads and queries
outside the grid. A nearest/count/emit sequence is necessary because replicated
segments must not create duplicate ties.

The first 4096-resolution build failed inside CuPy's CUB histogram with a
48 GiB temporary allocation. Reproduced the failure in a separate diagnostic;
it was a primitive workspace problem, not evidence against the grid algorithm.
Replaced the redundant histogram with run lengths from already sorted cell IDs
and the repository's cached exclusive scan. The repaired variant completes.

Three-warm-trial medians at 1M, all outputs passing exact parity:

| Grid resolution | Build ms | Repeated query ms |
|---|---:|---:|
| 1024 | 21.0 | 136.5 |
| 2048 | 22.1 | 71.3 |
| 4096 | 23.0 | 41.8 |

The best direct grid so far remains slower than seeded recursive STR (34.7 ms
in E8's three-trial run). Higher resolutions trade replication and dense grid
memory for fewer segment evaluations; long bounds can inflate replication.
Keep the grid as a comparator, and prefer it as a cheap seed table in the
leading hierarchy. Raw repaired evidence: `nearest-experiments/e7b-grid*-1m.*`.

## 2026-09-10 / E8 / Whole Roads and Full State

The experiment now accepts full road lines, expands segments on device,
preserves road IDs, and deduplicates tied parent pairs before host export.
Three warm trials pass exact whole-road oracle parity at both target scales.
At 100K, ingress/build/repeated query are 158.9/14.0/8.8 ms; at 1M they are
190.5/14.0/37.6 ms. This establishes that caller-side segmentation is unnecessary
for the proposed path. These are still experimental constructors, not public
`.sindex.nearest` dispatch.

The complete 2,680,339-building batch also passes the full-state road oracle.
One warm trial: ingress 253.2 ms, build 14.1 ms, repeated query 80.7 ms. This
case previously exceeded the public candidate-relation memory budget. The
hierarchy avoids that relation allocation. Raw evidence:
`nearest-experiments/e8-str8-seed-{roads,segments}-*.*`.

Next: screen fanout/seed resolution/leaf width at 1M, confirm finalists with
three trials at both target scales, measure independent spatial distributions,
and document the measured recommendation and production integration contract.
