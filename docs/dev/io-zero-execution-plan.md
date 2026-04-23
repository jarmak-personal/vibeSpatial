# IO-Zero Execution Plan

<!-- DOC_HEADER:START
Scope: Execution plan for removing avoidable host-side churn from public read paths and making GPU-first IO the default shape.
Read If: You are planning or executing the next IO read push, public read_file acceleration work, or GPU-first ingest cleanup.
STOP IF: You already have the active IO surface open and only need local implementation detail.
Source Of Truth: Program plan for making public read surfaces GPU-first and reducing hybrid IO to explicit compatibility boundaries.
Body Budget: 635/700 lines
Document: docs/dev/io-zero-execution-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-15 | Intent |
| 16-32 | Request Signals |
| 33-47 | Open First |
| 48-59 | Verify |
| 60-78 | Risks |
| 79-104 | Mission |
| 105-144 | What IO-Zero Means |
| 145-187 | Baseline Snapshot |
| 188-379 | Current Debt By Surface |
| 380-393 | Non-Goals |
| 394-407 | Working Principles |
| 408-428 | Program Structure |
| 429-455 | Milestone M0: Metrics And Read Taxonomy |
| ... | (6 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Turn the current IO improvement push into an execution plan for "IO-zero".

This plan is centered on public read paths, because `io_read` is now the
largest remaining weighted GPU-coverage lever and still the biggest source of
avoidable read-then-promote churn. The objective is not to make every file
format perfect at once. The objective is to make the public read path choose
the right GPU-first execution shape by default, reduce the number of formats
that depend on host container parsing plus compatibility bridges, and keep the
end-to-end performance story strong on both the `10k` shootouts and real large
benchmarks.

## Request Signals

- io-zero
- gpu-first io
- read_file plan
- public read path
- io read coverage
- read-then-promote churn
- geojson ingest
- geopackage ingest
- flatgeobuf ingest
- osm pbf ingest
- csv ingest
- kml ingest
- io execution plan
- milestone plan

## Open First

- `docs/dev/io-zero-execution-plan.md`
- `docs/architecture/io-files.md`
- `docs/architecture/io-arrow.md`
- `docs/architecture/io-support-matrix.md`
- `docs/architecture/runtime.md`
- `docs/architecture/residency.md`
- `src/vibespatial/io/file.py`
- `src/vibespatial/io/support.py`
- `src/vibespatial/bench/io_benchmark_rails.py`
- `scripts/gpu_acceleration_coverage.py`
- `tests/test_io_file.py`
- `tests/test_io_support.py`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run pytest tests/test_io_file.py -q`
- `uv run pytest tests/test_io_arrow.py -q`
- `uv run pytest tests/test_io_support.py -q`
- `uv run python scripts/benchmark_io_file.py --suite smoke`
- `uv run python scripts/health.py --tier gpu --check`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`
- `uv run vsbench shootout benchmarks/shootout --scale 10000 --repeat 3 --json --output <artifact>`
- `uv run python examples/benchmark_nearby.py`

## Risks

- Raising GPU coverage by moving tiny helper reads to device without improving
  the public read path can inflate the metric while leaving product IO mostly
  unchanged.
- Re-centering compatibility formats instead of GeoArrow and GeoParquet can
  turn the WKB bridge into the permanent layout rather than a temporary seam.
- Making `read_file` more GPU-shaped while preserving broad host-side property
  decode can improve dispatch logging without materially reducing wall time.
- Removing compatibility routing too aggressively can regress GeoPandas
  semantics on nulls, datetime fields, append-mode behavior, or exotic GDAL
  driver details.
- Adding per-format exceptions faster than the shared boundary evolves can
  create a planner that is "smart" but incoherent.
- Reading a source more than once for planning, sniffing, or fallback probes
  can erase gains on small and medium public workflows.
- Improving isolated parser throughput without preserving `10k` shootout or
  Florida-scale results would be the wrong win.

## Mission

Make supported public reads GPU-first by default and reduce hybrid IO to
explicit, narrow compatibility boundaries.

This push is successful only if all of the following become true:

- supported public reads stop choosing CPU by default when the next meaningful
  stage is likely GPU
- `read_file(...)` no longer pays avoidable read-then-promote churn on the
  public path
- GeoArrow and GeoParquet stay the canonical GPU-native IO seams
- WKB becomes a shrinking compatibility bridge rather than the default generic
  interchange layout
- compatibility-sensitive host routing remains explicit and observable instead
  of masquerading as native IO
- `10k --repeat 3` and `examples/benchmark_nearby.py` stay healthy while
  coverage rises

The program is not "done" when a few more formats say `selected=gpu`. It is
done when the repo's default read shape is correct:

- one repo-owned native boundary
- one explicit public materialization boundary
- no avoidable host churn before the first real GPU consumer

## What IO-Zero Means

"IO-zero" does not mean "no CPU exists anywhere in IO".

It means:

- zero avoidable host materialization before the first meaningful GPU stage
- zero silent compatibility bridges
- zero default-path CPU reads for supported public formats when a GPU runtime
  is available
- zero duplicated source passes for planning when one staged parse can answer
  both planning and ingest needs
- zero public `GeoDataFrame` assembly before the explicit compatibility/export
  boundary

It does **not** mean:

- every legacy GDAL format gets a bespoke GPU parser immediately
- public compatibility sinks like Fiona or append-mode pyogrio magically become
  GPU-native
- every request shape can skip host work even when the semantics are still
  host-defined

The correct contract is:

- canonical formats: GPU-native
- promoted compatibility formats: GPU-first hybrid with explicit boundaries
- untargeted legacy formats: explicit fallback

For public reads, the planning objective is:

`read + first meaningful downstream GPU stage`

not:

`best isolated parser throughput`

That is why the `10k` public pipeline rails matter more than naked parse
microbenchmarks for routing decisions.

## Baseline Snapshot

As of the current landed state:

- GPU tier is green and honest enough to use for ratchets.
- value-weighted GPU acceleration is about `75.50%`
- raw dispatch breadth is about `54.45%`
- `io_read` remains the biggest weighted headroom source:
  - about `400 / 965` GPU by dispatch
  - about `41.45%` GPU by dispatch
- `io_write` improved materially. Public device-backed GeoJSON, Shapefile,
  GeoPackage, and FlatGeobuf writes now use the shared native Arrow/WKB sink
  instead of rebuilding a host GeoDataFrame, and force device WKB encode below
  the generic small-row threshold.

The public read story is mixed but improving:

- GeoJSON public pipeline now clears the `10k` rail and the Florida benchmark
  is back in the intended band:
  - `examples/benchmark_nearby.py`:
    - `Read GeoJSON`: about `7.1s` on vibeSpatial GPU
    - `Read GeoJSON`: about `55.2s` on GeoPandas
- Shapefile public pipeline clears parity-or-better at `10k`
- FlatGeobuf public pipeline clears parity-or-better at `10k`
- GeoPackage public `engine="pyogrio"` native boundary clears parity-or-better
  at `10k`

The current large gaps are structural, not accidental:

- GeoJSON still has a host-side property seam, but it is now narrowed to
  staged property-object decode instead of full-feature reparsing
- generic container formats still rely on host `pyogrio.read_arrow(...)`
  container parse plus the WKB bridge
- supported pyogrio-backed vector containers now keep `mask` and explicit
  `layer` filters on the native Arrow/WKB read boundary
- CSV and KML no longer demote eligible local unfiltered reads solely because
  of the old coarse `10 MB` GPU gate
- OSM tags are still host-shaped at the low-level parser boundary
- remote URLs, untargeted legacy formats, invalid row windows, and unsupported
  geometry families still force explicit compatibility routing

This is enough progress to push for IO-zero. It is not yet IO-zero.

## Current Debt By Surface

### 1. GeoJSON

Current strengths:

- all eligible public GeoJSON reads now route GPU-first again
- mixed point/line/polygon files are back on the intended GPU path
- point-only fast paths exist
- Florida-scale public read performance is recovered

Remaining debt:

- property extraction is still host-side for string/object columns even though
  the path now stages property-object spans instead of reparsing full features
- the pipeline is still text-heavy and RFC 7946-specific rather than fully
  columnar after structural discovery
- public routing still treats filesystem existence as the proxy for
  pipeline-oriented GPU preference instead of a more principled crossover model

IO-zero implication:

- GeoJSON becomes IO-zero only when the remaining host bottleneck is reduced to
  an explicit compatibility export seam or a truly narrow property boundary,
  not when geometry decode alone is fast

### 2. Shapefile

Current strengths:

- eligible public reads use the repo-owned native plan
- explicit `engine="pyogrio"` reads now stay on the Arrow/WKB native bridge for
  supported 2D request shapes
- direct SHP GPU decode exists
- DBF GPU parsing exists
- public `10k` rails are healthy

Remaining debt:

- polygon direct decode still cannot preserve multipart polygon semantics in
  the direct SHP path, so those cases drop to Arrow/WKB
- container/sidecar semantics are still fundamentally hybrid

IO-zero implication:

- Shapefile will stay hybrid by design, but it should become "zero avoidable
  host churn" on the default public read path, including polygon-heavy inputs

### 3. FlatGeobuf

Current strengths:

- eligible local unfiltered public reads already use the direct FlatBuffer GPU
  decoder
- public `10k` rail is green

Remaining debt:

- explicit `engine="pyogrio"` and container-shaped requests still go through
  Arrow/WKB
- direct decode is not yet the broad answer for all request shapes

IO-zero implication:

- FlatGeobuf is the closest promoted container to IO-zero after GeoJSON and
  should be used as the design model for other container formats

### 4. GeoPackage, FileGDB, GML, GPX, TopoJSON, GeoJSON-Seq

Current strengths:

- public or native promoted reads can stay on the shared Arrow/WKB native
  boundary
- GeoPackage public `engine="pyogrio"` now stays native for supported 2D
  families and clears `10k` strongly (`11.89x` on the latest smoke rail)
- GeoPackage `mask`, `bbox+layer`, and `mask+layer` request shapes now stay on
  that same native bridge
- uniform WKB fast paths already improved the bridge materially

Remaining debt:

- host `pyogrio.read_arrow(...)` container parse is still the dominant read
  seam
- WKB is still the practical interchange layer instead of a shrinking
  compatibility bridge
- unsupported GeoPackage public geometry families such as `Point Z` and
  `Unknown` still need the explicit compatibility path

IO-zero implication:

- this entire family needs a de-bridging program, not format-by-format
  patchwork

### 5. CSV And KML

Current strengths:

- both have repo-owned GPU read paths
- CSV already uses `pylibcudf` table parse for large geometry-column inputs
- WKT/WKB geometry columns can already land on the native geometry boundary

Remaining debt:

- the old coarse `10 MB` public gate is removed; CSV still uses a large-file
  crossover internally to choose libcudf table parse before GPU geometry decode
- small and medium public files can still choose CPU for the wrong reason
- CSV layout sniffing and geometry-format sniffing are still lightweight host
  passes

IO-zero implication:

- these need crossover-driven public planning, not static size gates

### 6. WKT

Current strengths:

- raw WKT public reads are already GPU-first with no CPU fallback

Remaining debt:

- no GPU write path
- still a compatibility format rather than a canonical storage or interchange
  target

IO-zero implication:

- WKT read is not a blocker; it should remain correct but is not the main
  weighted coverage lever

### 7. OSM PBF

Current strengths:

- native parser exists
- default public layered reads already use supported pyogrio layers in
  parallel
- public tag projection is bounded instead of exploding into thousands of
  eager columns

Remaining debt:

- low-level tags are still host-resident dicts
- `other_relations` still needs explicit compatibility bridging because real
  data contains `GeometryCollection`
- standard public layers still rely on a compatibility container reader rather
  than a device-native parser end-to-end

IO-zero implication:

- OSM needs a targeted de-hosting plan for tags and public layer materialization
  rather than one more round of routing tweaks

### 8. GeoParquet And Arrow

Current strengths:

- these are already the canonical read boundary
- GeoParquet is GPU-native for reads by design
- GeoArrow is the canonical interchange target

Remaining debt:

- the repo still lets the WKB bridge dominate promoted compatibility formats
- some chunking, pushdown, and compatibility edges remain hybrid

IO-zero implication:

- GeoParquet and GeoArrow are not the emergency; they are the target shape the
  rest of read IO should converge toward

### 9. Request-Shape Debt Across Formats

Current strengths:

- `bbox`, `columns`, and `rows` already stay on the promoted native path for
  the pyogrio-backed vector containers

Remaining debt:

- supported `mask` and safe `layer` filters now stay on promoted native read
  paths
- remaining explicit engine choices that force compatibility are remote,
  invalid, legacy, or unsupported-geometry request shapes
- planning is still per-format and partly heuristic rather than one shared
  crossover model

IO-zero implication:

- public request shapes need to become first-class GPU-planned surfaces, not
  exceptions that immediately demote to CPU

## Non-Goals

This plan is not about:

- making every legacy GDAL driver GPU-native
- hiding CPU semantics behind fake native logging
- squeezing isolated parse microbenchmarks while public workflows regress
- building a bespoke direct parser for every format before shared-native
  boundaries improve
- pretending write-side gaps are read-side failures

The read push is now clean enough to keep moving write-side de-hybridization
in parallel, as long as the same `10k` and Florida-scale rails stay healthy.

## Working Principles

Apply these principles throughout the push:

- optimize for public pipeline shape, not parser vanity metrics
- prefer one shared native read boundary over many format-local public builders
- prefer canonical GeoArrow-family layout over deeper WKB dependence
- do not add new silent compatibility routes
- do not accept read plans that intentionally choose CPU when the next real
  stage is likely GPU
- preserve GeoPandas parity by making compatibility boundaries explicit and
  narrow, not by widening CPU routing
- prove every milestone on both synthetic rails and end-to-end workflows

## Program Structure

The IO-zero push should run in six ordered milestones:

| Milestone | Name | Primary Surfaces | Why First |
|---|---|---|---|
| M0 | Metrics And Read Taxonomy | health, rails, router, docs | Prevents the push from drifting into ad hoc format work |
| M1 | GeoJSON Completion | `geojson.py`, `geojson_gpu.py`, public routing | Biggest single public read story and still hybrid at properties |
| M2 | Container De-Bridging | `file.py`, `wkb.py`, container readers | Biggest remaining weighted read debt after GeoJSON |
| M3 | Request-Shape Parity | `read_file` planner, mask/bbox/engine paths | Removes avoidable demotion on real public calls |
| M4 | Text, Tabular, And OSM | CSV, KML, WKT, OSM PBF | Cleans up the remaining hybrid special cases |
| M5 | Canonicalization And Proof | rails, coverage, shootouts, docs | Locks IO-zero in as the repo default |

Do not start by spreading effort evenly across every format. The correct order
is:

1. finish GeoJSON
2. de-bridge the generic containers
3. fix request-shape parity
4. clean up the remaining special formats

## Milestone M0: Metrics And Read Taxonomy

### Goal

Make the repo's IO-zero target measurable and reduce ambiguity about which
formats are already close versus structurally wrong.

### Checklist

- [ ] Add an explicit `io_read` breakdown view by format family and request
  shape in the GPU coverage output or supporting diagnostics.
- [ ] Split read-family status into:
  - canonical GPU-native
  - GPU-first hybrid
  - explicit compatibility
  - fallback-only
- [ ] Add or refresh docs so explicit engine and request-shape demotions are
  visible.
- [ ] Expand rails so the missing high-value public read surfaces have enforced
  bars, not only informal benchmark notes.

### Exit Criteria

- we can identify remaining `io_read` debt by format family instead of only by
  one aggregated percentage
- the repo has a written list of which request shapes still force CPU and why

## Milestone M1: GeoJSON Completion

### Goal

Finish the GeoJSON read path so the public route is GPU-first across geometry
families and the remaining CPU seam is no longer the main wall-time cost.

### Primary Surfaces

- `src/vibespatial/io/geojson.py`
- `src/vibespatial/io/geojson_gpu.py`
- `src/vibespatial/io/geojson_gpu_kernels.py`
- `src/vibespatial/io/file.py`

### Checklist

- [ ] Keep all eligible public GeoJSON geometry families on the GPU path by
  default.
- [x] Reduce or replace the current CPU property extraction bottleneck with a
  more columnar or staged design.
- [ ] Eliminate any remaining multi-pass planning or full-file reread behavior
  that is not required for correctness.
- [ ] Add separate public rails for point-heavy and polygon-heavy GeoJSON so we
  do not regress one while optimizing the other.
- [ ] Keep the Florida benchmark in the recovered band while making the small
  public path better, not worse.

### Acceptance Bars

- Florida `benchmark_nearby.py` `Read GeoJSON` stays at or below the recovered
  `~11-12s` band on the target machine; current staged property-span path is
  about `6.5s` on the target machine
- GeoJSON public pipeline rail stays `>= 1.0x` at `10k`
- point-heavy and polygon-heavy public rails both pass

## Milestone M2: Container De-Bridging

### Goal

Shrink the pyogrio Arrow + WKB bridge from the default generic read shape into
an explicit compatibility seam.

### Primary Surfaces

- `src/vibespatial/io/file.py`
- `src/vibespatial/io/wkb.py`
- container-specific readers and planners

### Checklist

- [ ] Identify which promoted container formats can move from generic Arrow/WKB
  to direct or partially direct GPU-native decode.
- [ ] Treat FlatGeobuf as the reference model for promoted container work.
- [x] Prioritize GeoPackage first because it is already on the public `10k`
  rails and is the clearest remaining generic-container user surface.
- [ ] Reduce WKB dependence where native GeoArrow-family decode is feasible.
- [x] Keep explicit `engine="pyogrio"` native-compatible where semantics allow,
  and narrow the compatibility-only exceptions.

### Acceptance Bars

- GeoPackage public `engine="pyogrio"` rail stays `>= 1.0x` at `10k`
- FlatGeobuf public auto rail stays `>= 1.0x` at `10k`
- WKB bridge usage shrinks by design, not only by better fast paths

## Milestone M3: Request-Shape Parity

### Goal

Stop common public request shapes from demoting otherwise GPU-ready reads to
CPU.

### Primary Surfaces

- `src/vibespatial/io/file.py`
- public planner/routing tests

### Checklist

- [ ] Audit every reason the public read path drops to CPU:
  - `mask`
  - explicit engine selection
  - small-file gates
  - unsupported bbox/rows/columns combinations
- [x] Replace static size gates for CSV and KML with crossover-driven planning.
- [ ] Decide which `mask` shapes can be supported as read + immediate GPU
  spatial filter instead of disabling the GPU-native read entirely.
- [ ] Preserve explicit compatibility exceptions, but narrow them to proven
  semantic gaps.
- [ ] Add tests that fail if a supported public GPU-resident read demotes to
  CPU only because of a planner default rather than a real compatibility
  boundary.

### Acceptance Bars

- `read_file(..., engine="pyogrio")` remains native for every promoted format
  that can satisfy the request through the repo-owned boundary
- CSV and KML no longer rely on the fixed `10 MB` gate
- the number of public read CPU selections drops materially in GPU coverage

## Milestone M4: Text, Tabular, And OSM

### Goal

Clean up the remaining special-format read debt after the main public route is
correct.

### Primary Surfaces

- `src/vibespatial/io/file.py`
- CSV/KML readers
- OSM PBF readers and bundle/projectors

### Checklist

- [ ] Make CSV public planning depend on layout and pipeline shape, not only
  file size.
- [ ] Do the same for KML.
- [ ] Keep WKT simple and correct; do not over-invest there.
- [ ] De-host OSM tags and public layer projection where that work still forces
  a host-shaped seam.
- [ ] Reduce explicit OSM compatibility bridges to only the unsupported family
  cases that really need them.

### Acceptance Bars

- CSV and KML public reads stay GPU-first whenever the next meaningful stage is
  GPU
- OSM standard public layers no longer look like a permanent pyogrio
  compatibility product

## Milestone M5: Canonicalization And Proof

### Goal

Lock IO-zero in as the repo's default read shape and prove it on the right
surfaces.

### Checklist

- [ ] Update docs so GeoArrow and GeoParquet are the clear destination shape
  for read IO.
- [ ] Publish the final support taxonomy for:
  - canonical GPU-native
  - GPU-first hybrid
  - explicit compatibility
  - fallback-only
- [ ] Add or refresh all enforced public rails needed for the remaining
  promoted read formats.
- [ ] Re-run the full `10k --repeat 3` shootout.
- [ ] Re-run `examples/benchmark_nearby.py`.
- [ ] Re-run GPU health and capture the new `io_read` rate.

### Acceptance Bars

- value-weighted GPU acceleration improves materially from the current landed
  baseline
- `io_read` rises far above the current `~41%` dispatch rate
- `10k --repeat 3` remains healthy or improves
- Florida `benchmark_nearby.py` remains in the recovered fast band

## Exit Criteria

The IO-zero read push is complete when all of the following are true:

- supported public reads default to the GPU-oriented path whenever the next
  meaningful stage is likely GPU
- the remaining CPU read routes are explicit compatibility or fallback
  decisions, not default planner drift
- GeoJSON no longer has a dominant host-side read bottleneck on the public path
- promoted container formats depend less on the generic WKB bridge
- `io_read` is no longer the dominant weighted GPU-coverage debt
- `10k` shootouts and Florida-scale public benchmarks remain strong

At that point, the repo can move from "fix the public read shape" to the next
IO program:

- write-side de-hybridization
- canonical sink completion
- format-specific compatibility edge cleanup
