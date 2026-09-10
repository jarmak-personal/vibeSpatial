# OSM Building-to-Road Nearest Comparison

<!-- DOC_HEADER:START
Scope: Same-data Massachusetts building-to-road and segment nearest benchmark, public API timing boundaries, and STRtree comparison.
Read If: You are comparing road snapping, query_nearest, all-match ties, or NativeSpatialIndex k=1 performance.
STOP IF: You only need the OSM reader or a nearest kernel implementation detail.
Source Of Truth: Reproducible public nearest benchmark contract and Massachusetts evidence.
Body Budget: 200/220 lines
Document: docs/testing/osm-nearest-performance.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-16 | Request Signals |
| 17-25 | Open First |
| 26-30 | Verify |
| 31-52 | Reproduce |
| 53-60 | Risks |
| 61-94 | Workload Contract |
| 95-120 | Measurement and Correctness |
| 121-170 | Measured Results |
| 171-200 | Interpretation |
DOC_HEADER:END -->

## Intent

Reproduce the public nearest-query gap on real Massachusetts OSM data before
changing the index or search implementation. This is a Native* coverage
measurement, not an implementation of road snapping or a new GPU index.

## Request Signals

- road snapping benchmark
- building nearest road
- Massachusetts query_nearest
- Shapely STRtree comparison
- NativeSpatialIndex k=1

## Open First

- scripts/benchmark_osm_nearest.py
- docs/testing/osm-nearest-evidence.json
- tests/test_benchmark_osm_nearest.py
- docs/architecture/spatial-joins.md
- src/vibespatial/api/sindex.py
- src/vibespatial/spatial/nearest.py

## Verify

- `uv run pytest tests/test_benchmark_osm_nearest.py -q`
- `uv run python scripts/check_docs.py --check`

## Reproduce

```bash
uv run python scripts/benchmark_osm_nearest.py \
  /tmp/vibespatial-osm/massachusetts.osm.pbf \
  --cache /tmp/vibespatial-osm/nearest-baseline \
  --output /tmp/vibespatial-osm/nearest-gpu/report.json \
  --repeat 1 --timeout 240 --mode gpu
```

Repeat with the same cache and `--mode auto`, using a different output directory.
The immutable Shapely baseline is reused only after fixture, oracle, script,
package/lock, Python, CPU host, storage, and measurement identities match.
Candidate timings are always measured again. Select a new cache after changing
any comparator identity field. `--scales 1000 --repeat 3 --profile` is useful
for a smaller repeated measurement and an additional untimed cProfile query.
The default is three warm-process trials after one first-process trial.
For that many statewide repetitions, use `--timeout 600` or higher: the
timeout covers the entire worker, including every query, imports, file reads,
and validation. The command above uses one warm trial to bound comparator
cost. A timeout does not imply that no individual query completed.

## Risks

- All-match ties and exactly-one-neighbor selection are different contracts.
- Host query batches can trigger conversion even when the road input is native.
- A lazy sindex accessor does not measure completed index construction.
- Failed parity and timed-out cases are evidence, not validated speedups.
- Segmentation trades tighter bounds for a larger index and more endpoint ties.

## Workload Contract

The source is the 310,023,499-byte Geofabrik extract used in the
[ingestion benchmark](osm-pbf-performance.md), SHA256
`d7f2220cf2d0ed6b6eeb432f1986c27799a7248bd5ec6f24746fdf1d53c17263`.
OSM data is copyright OpenStreetMap contributors, available under ODbL.

GDAL/pyogrio preparation selects `highway IS NOT NULL` line ways, including
paths, and `building IS NOT NULL AND building <> 'no'` polygon features.
Both are projected from EPSG:4326 into EPSG:26986 (metres). Buildings become
Shapely point-on-surface representatives after projection; no topology repair
is applied. Selection, projection, point generation, segmentation and file
writes are shared CPU fixture preparation, measured separately from querying.
This isolates nearest performance from differences in readers or projection.

The fixture contains 991,441 roads with 9,719,891 coordinates, 8,728,450
two-vertex segments, and 2,680,339 building points. Query scales are nested
prefixes of a seeded random permutation across the whole state. Every scale
uses the full road/segment index. Consecutive vertices become segments;
zero-length edges are retained, and no edge joins separate road features.
Each segment has a parent row in the filtered road fixture.

Both backends decode identical WKB values. The candidate uses only public
`vibespatial.GeoSeries.from_wkb`, `.sindex`, and
`.nearest(return_all=True, return_distance=True)` calls. Shapely uses
`from_wkb`, `STRtree`, and
`query_nearest(all_matches=True, return_distance=True)`.
There is no `max_distance`, exclusion, or approximate search.

`k=1` with all matches is a variable-size tie relation, not exactly one output
per building. Segment results remain segment IDs: endpoint ties can produce
multiple segments from one parent road. Deduplicating parent IDs or selecting
a snap location is downstream work excluded from both query measurements.

## Measurement and Correctness

Each backend/scale/layout runs in a separate process. Imports and reading the
prepared WKB files are excluded. A trial measures WKB ingress, index access,
first query, then a repeated query on the same index. CUDA is synchronized
before and after each candidate stage. The first process trial is reported
separately; later fresh-input trials supply warm-process medians. Public
NumPy pair and distance exports are included. The on-disk compilation cache
may be warm; this is not a clean-machine startup benchmark.

Shapely constructs its tree in the index stage. The candidate can defer work
until the query, so compare index access plus first query as well as repeated
queries. The combined median is computed from per-trial sums. WKB ingress
plus index plus first query is also reported. Shared PBF preparation is a
separate cost and is never included selectively in one backend's denominator.

Every first and repeated result is sorted by query/target IDs and checked
against the same-data Shapely oracle. Match-pair multisets must agree exactly,
including ties; distances use `atol=1e-6` metres and `rtol=1e-10`. Failed
parity is retained as evidence and receives no speedup ratio. Timeout/error
cases preserve logs and checkpoints and are not presented as completed times.
Dispatch, fallback, materialization, and runtime D2H events are captured per
stage; detailed queues retain at most 512 events. Runtime D2H counters cover
instrumented transfers, not an exhaustive hardware trace. Continuous event
file logging is disabled during timing to avoid file-IO contamination.

## Measured Results

Measured 2026-09-10 on RTX 4090 24 GiB / i9-13900K, source `cd89cde`.
The [evidence JSON](osm-nearest-evidence.json) records dataset, script, package,
lock, host and oracle hashes, first-process timings, transfers, and parity.
Shapely was freshly measured for this workload. One warm-process trial follows
one first-process trial, with a 240-second limit for each complete worker.
The main candidate uses explicit GPU mode, admitting both road and point WKB
through public constructors. These are observed times, **not validated
speedups**: every completed candidate result failed exact all-match pair parity.

Seconds, excluding shared preparation and WKB ingress:

| Index | Buildings | Shapely build + first | vS build + first | Shapely repeated | vS repeated |
|---|---:|---:|---:|---:|---:|
| roads | 1,000 | 0.173 | 3.769 | 0.0166 | 3.771 |
| roads | 10,000 | 0.334 | 3.784 | 0.178 | 3.821 |
| roads | 100,000 | 1.954 | 5.541 | 1.779 | 5.511 |
| roads | 2,680,339 | 48.338 | capacity error | 48.112 | no result |
| segments | 1,000 | 1.381 | 16.997 | 0.00862 | 17.004 |
| segments | 10,000 | 1.489 | 16.975 | 0.0801 | 16.978 |
| segments | 100,000 | 2.159 | worker timeout | 0.800 | worker timeout |
| segments | 2,680,339 | 22.613 | worker timeout | 21.464 | no result |

The 100K segment worker completed three queries (61.223, 60.827, 60.816 s)
before its fourth query exceeded the total worker budget. There is no
completed warm-trial aggregate for that case. Full-state road nearest requested
2,925,663,094 candidate pairs: at least 21.80 GiB for two index columns alone,
exceeding the admitted 20.24 GiB. The full-state segment worker hit its
240-second limit in the first query, after completing ingress and index
access. These failures are not complete query timings.

The first 1,000-query road result has 1,004 pairs versus Shapely's 1,003.
Query row 619 gets an extra road at 44.79108061 m when the nearest road is
44.79093003 m away: 0.15058 mm farther, not an equidistant match. Segment
endpoint ties are real and remain part of the oracle; near-but-unequal
candidates must not be accepted as additional ties.

At 1,000 queries, repeated-query runtime telemetry records 34.3 MB D2H for
roads and 281.2 MB for segments, including cached row bounds and intermediate
candidate indices. The respective public outputs are only about 24/25 KB.
No explicit fallback events were recorded; GPU dispatch labels alone do not
establish a device-resident path. Follow-up production work is tracked in
[issue #18](https://github.com/jarmak-personal/vibeSpatial/issues/18).

The full pipeline profile passed. All 51 stages at 1M were reviewed and their
wall times are in the evidence JSON. The largest was `read_geojson` at
71.33 ms, with no unexpected CPU-heavy stage above 1 s. That synthetic suite
does not cover the OSM nearest failures exposed here.

## Interpretation

In the current public dispatch, `nearest_relation` passes a cached
`NativeSpatialIndex` only for `k > 1`. That does not mean GPU nearest is
unavailable at `k=1`: specialized point nearest and generic GPU paths exist.
With `return_all=True`, the generic path differs from fixed-k selection.

In `auto` mode, WKB construction can retain small query batches on the host
while large road inputs become owned device geometry. Therefore explicit
`gpu` and default `auto` are separate candidate measurements against the same
CPU comparator. Device residency and index reuse need to be established
before attributing the entire gap to R-tree packing.

A GPU hierarchy is a plausible general solution to test after this baseline:
compare a packed R-tree with the existing hierarchy or a segment BVH, using
exact point-to-segment distance and distance-bound traversal. Preserve all
minimum-distance ties and parent-road provenance. Smaller segment boxes can
improve pruning but increase index size and build cost; measure the amortized
benefit across repeated query batches.

This is a design hypothesis, not a measured implementation result. GPU BVH
construction has established parallel algorithms; see
[Karras, HPG 2012](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees).
The comparison does not establish that STR packing beats other GPU layouts.

The Shapely [STRtree contract](https://shapely.readthedocs.io/en/stable/strtree.html)
describes two-dimensional Cartesian nearest distance, all-match tie semantics,
and the benefit of tighter geometry bounds. A nearest road index and distance
are only part of snapping: selecting a point on that road and applying any
access/network constraints are separate operations.
