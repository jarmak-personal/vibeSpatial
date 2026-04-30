# Native Full Coverage PRD

<!-- DOC_HEADER:START
Scope: Repository-wide PRD and feature hold for reaching complete Native* functionality across carriers, boundaries, kernels, workflows, and verification gates.
Read If: You are planning new work, deciding whether feature development is allowed, implementing Native* completion, removing compatibility boundaries, or adding native kernels.
STOP IF: You only need the completed carrier-substrate plan or one operation-local implementation detail already routed by intake.
Source Of Truth: Active PRD for the Native* feature hold and full native coverage completion mandate.
Body Budget: 280/280 lines
Document: docs/dev/native-full-coverage-prd.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-13 | Intent |
| 14-26 | Request Signals |
| 27-39 | Open First |
| 40-50 | Verify |
| 51-62 | Risks |
| 63-86 | Hold Policy |
| 87-104 | Definition |
| 105-127 | Boundary Rules |
| 128-148 | Kernel Scope |
| 149-167 | Carrier Scope |
| 168-223 | Milestones |
| 224-238 | Surface Matrix |
| 239-258 | Acceptance Gates |
| ... | (2 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Define the repository-wide feature hold and completion plan for reaching 100%
`Native*` functionality across vibeSpatial.

The prior Native* plans completed the private carrier substrate. This PRD is
stricter: the library is not considered Native* complete until the kernels,
runtime boundaries, export boundaries, tests, benchmarks, and public workflow
composition are complete enough that GPU-selected paths do not depend on hidden
host execution.

## Request Signals

- Native* full coverage
- Native* completion PRD
- feature hold
- development freeze
- no new features
- 100% native functionality
- native boundaries
- native kernels
- NativeExportBoundary
- strict native completion

## Open First

- docs/dev/native-full-coverage-prd.md
- docs/dev/native-format-library-plan.md
- docs/dev/native-format-inventory.md
- docs/dev/private-native-execution-substrate-plan.md
- docs/decisions/0044-private-native-execution-substrate.md
- docs/decisions/0046-gpu-physical-workload-shape-contracts.md
- docs/testing/performance-tiers.md
- tests/test_private_native_substrate.py
- tests/test_strict_native_mode.py
- tests/test_pipeline_benchmarks.py

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "Native full coverage PRD"`
- `uv run pytest tests/test_private_native_substrate.py -q`
- `uv run pytest tests/test_strict_native_mode.py -q`
- `uv run pytest tests/test_index_array_boundary.py -q`
- `uv run pytest tests/test_pipeline_benchmarks.py -k "native or relation or grouped" -q`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`
- pre-push only: `uv run python scripts/health.py --tier gpu --check`

## Risks

- Treating carrier completion as Native* completion leaves unsupported kernels
  hidden behind explicit but still performance-limiting compatibility exports.
- Broadening pandas interception to look native can repeat ADR-0043.
- Removing scalar fences without replacing their admissibility proof can make
  kernels faster and wrong.
- Implementing one workflow shortcut can improve shootouts while leaving the
  reusable physical shape incomplete.
- Leaving feature work open during this push will spread attention and grow the
  non-native surface area.

## Hold Policy

vibeSpatial is on new feature and general development hold until this PRD is
complete.

Allowed work:

- Native* completion work from this PRD.
- Correctness fixes that unblock Native* completion or upstream compatibility.
- Kernel, runtime, IO, benchmark, docs, and test work required to complete this
  PRD.
- Tooling fixes needed to run the verification gates.

Disallowed work:

- New public APIs, new feature surfaces, or convenience methods.
- Optional refactors not tied to a Native* completion milestone.
- Benchmark-specific shortcuts.
- Expanding public pandas interception outside exact contracts.
- Adding CPU compatibility paths without a paired Native* completion issue.

If a user asks for unrelated feature work during the hold, route to this PRD
and either convert the request into a Native* completion task or defer it.

## Definition

100% Native* functionality means every existing GPU-relevant library surface is
in one of these states:

- `native-complete`: admitted inputs execute through Native* carriers and
  family-specific GPU kernels until a terminal user export.
- `terminal-export`: the operation is explicitly a public materialization or IO
  boundary and records that boundary.
- `unsupported-native`: the operation declines before entering native execution
  and fails under strict-native when GPU-native behavior was requested.

No existing surface may remain `partial`, `debt`, or `implicit-host`.

Compatibility with GeoPandas remains mandatory. Native completion cannot change
public types, ordering, CRS, index semantics, null behavior, warnings, or
exceptions.

## Boundary Rules

Every boundary must be classified and tested.

Allowed terminal boundaries:

- User-requested GeoDataFrame, GeoSeries, pandas, NumPy, Shapely, Arrow,
  GeoParquet, Feather, repr, and debug materialization.
- Public IO output where host-visible metadata is part of the file contract.

Allowed non-terminal fences:

- Bounded scalar or metadata transfers that prove a GPU shape is admissible.
- Each fence must have an operation-level reason, byte count, strict-native
  policy, and canary coverage.

Disallowed boundaries:

- Hidden `.get()`, `cp.asnumpy()`, NumPy conversion, pandas assembly, Shapely
  conversion, or host row-position normalization inside a GPU-selected hot path.
- Compatibility exports that exist because a needed Native* kernel is missing.
  These are PRD work, not accepted final state.

## Kernel Scope

Native* completion includes the missing kernels and primitive pipelines needed
to remove compatibility boundaries from admitted GPU paths.

Required kernel families:

- Multi-candidate exact overlay topology: intersection, difference, symmetric
  difference, union, lower-dimensional outputs, selected faces, repair, and
  keep-geometry-type filtering.
- Grouped polygon operations: n-ary/grouped union, grouped difference, grouped
  intersection, validity repair, and provenance propagation.
- Predicate/refinement kernels: relation-pair DE-9IM coverage for admitted
  point, lineal, polygonal, mixed, and collection shapes.
- Dynamic constructive output assembly: count, scan, scatter, offsets, output
  bytes, provenance, and metadata without host sizing except admitted fences.
- IO codecs: GPU-native WKB, GeoArrow, GeoJSON, Shapefile, FlatGeobuf, and
  GeoParquet ingress/egress where the format can be represented columnarly.
- Attribute kernels: all dtype families needed by existing workflows, with
  explicit null and categorical contracts before compute admission.

## Carrier Scope

The existing carrier substrate remains the execution currency:

- `NativeFrameState`
- `NativeRowSet`
- `NativeRelation`
- `NativeGrouped`
- `NativeSpatialIndex`
- `NativeGeometryMetadata`
- `NativeExpression`
- `NativeIndexPlan`
- `NativeExportBoundary`

Completion work must preserve carrier contracts instead of bypassing them with
operation-local wrappers. New kernels must declare input carriers, output
carriers, physical workload shape, precision policy, stream readiness, and
export behavior.

## Milestones

- [x] M0. Freeze enforcement and inventory ratchet.
- [x] M1. Boundary firewall: every D2H/materialization is classified, named,
  byte-counted, and strict-native tested. Initial guard now rejects raw
  `cp.asnumpy`, unnamed runtime D2H, and unclassified zero-argument `.get()`
  calls outside named boundary helpers. Host point metadata replaces post-H2D scalar fences.
  Owned concat/takes/scatters reuse structural row counts, mixed-family rowset presence,
  fixed-width polygon metadata, host mirrors, or batched device totals; segment extraction reuses structural cardinalities;
  bounded candidates use host capacity/live pairs; same-row span summaries and
  larger sweeps retain named scalar fences; clip-rect scatter consumes device row maps and
  names output count fences; flat spatial-index and fused Hilbert builds keep row
  bounds/order resident and batch extent fences; return-device nearest kNN consumes
  device row bounds with host-known or named extent fences; overlay split/face-edge uses
  host capacity plus device compaction; WKB decode sizing and dense-ring
  admission use named fences, while GeoArrow decode/encode reuses column
  child sizes, owned structure proofs, and supported device single/multi promotions;
  relation-pair and covered-by single-mask predicate family admission use host
  metadata or named one-byte fences; GeoJSON validity, property type counts,
  unsupported-type, and family-domain gates are named runtime D2H fences; OSM way
  classification/counts/reorder reuse CPU-parsed refs/tags; boundary and interiors
  constructive assembly avoid raw CuPy scalar syncs, reusing host structural sizing
  or named allocation fences; union-all/disjoint-subset/coverage-union assembly
  uses structural offset sizes and named validity/admission fences;
  binary constructive validity proofs reuse cached validity before device probes;
  part-expansion/proof gates, containment-bypass counts,
  CCCL selected-count wrappers, and profile-only pair-kind summaries use named
  runtime fences; make-valid repair names ring/duplicate/orientation/validity/remap fences and repairs adjacent-hole topology via grouped hole union plus exterior difference;
  polygon-rectangle proof reuses fixed-width metadata before named dense-ring/closure/axis/split/max-vertex fences; line-buffer, grouped convex-hull rewrite, `get_geometry`, shortest-line, segmented union, public clip/overlay admission probes, minimum-clearance-line, polygon-box query, and indexed point-relation gates use host mirrors, zero-D2H routing, row-aligned assembly, or operation-named scalar fences.
  Boundary endpoint assembly handles empty lineal rows without host repair;
  coverage dissolve fills unobserved rows by device metadata scatter.
- [x] M2. Native export boundary: public materialization is represented by a first-class `NativeExportBoundary` across GeoDataFrame, Arrow, GeoParquet, Feather, Shapely, repr, and debug export. GeoDataFrame, Arrow, GeoParquet, Feather, explicit owned-geometry Shapely export, text/html repr, GeoSeries text repr, GeoJSON, Python geo-interface/geo-dict, WKB/WKT, vector-file export, scalar properties, bounds/total-bounds, NumPy array protocol, public predicate/distance Series, public spatial-index query/nearest formatting, and grouped reducer pandas export now record terminal boundaries when native state or native geometry backing is present. Predicate result runtime transfers are marked as terminal exports instead of hidden ping-pong. Debug exports are covered by the explicit debug host bridge.
- [x] M3. Overlay topology kernels replace multi-candidate exact CPU topology
  fallbacks for admitted point, lineal, polygonal, and mixed-family inputs.
- [x] M4. Grouped geometry kernels replace grouped constructive compatibility
  fallbacks for admitted dissolve, union, difference, and overlay workflows.
  Explicit polygon coverage dissolve has device grouped edge reduction for shared
  boundaries, native rows, and valid empty unobserved categorical groups. Unary
  rewrites stay conservative until identically noded coverage is proved.
  Disjoint-subset dissolve batches named structural/area/neighbor proofs;
  coverage-edge union reuses host code/validity mirrors and all-valid full/observed-group device proofs; low-fan-in all-valid device coverage groups, including dropped-row cases, can reduce through exact grouped coverage union without scalar admission probes; OGC validity has no-repair/compact-repair expressions; unobserved groups use device metadata scatter instead of variable-width owned takes; invalid grouped outputs try native make-valid repair before host recompute fallback; global tree reductions expose GPU/pairwise strict declines before CPU reduction; optional bbox grouping declines on device-only rows instead of exporting row-bounds matrices.
- [x] M5. Predicate and relation refinement kernels cover admitted DE-9IM and
  relation-pair consumers without public bool Series or host pair export. Spatial
  join filters empties before pair generation and remaps original positions;
  row-aligned point/point predicates use device coordinate equality with only a terminal public Series export; device expressions cover admitted predicates/distances; relation-kernel interior/boundary, return-device refinement, and public sjoin non-empty/empty relation lowering stay device-resident until terminal export.
  Public `sjoin(..., on_attribute=...)` relation filters now admit all-valid device-compatible keys; multipoint relation refinement and row-aligned multipoint predicate expressions stay device-resident through indexed point relation kernels; public relate/reduce wrappers, and distance-metric GPU failures propagate strict-native declines before host bridging.
- [x] M6. IO codecs and terminal writers complete native ingress/egress for existing GPU-readable and GPU-writable formats. GeoParquet `bbox` read filters keep device row positions and defer public RangeIndex labels to `NativeIndexPlan`; public Feather/Arrow reads attach metadata-seeded `NativeFrameState` after owned-buffer decode.
- [x] M7. Attribute execution completes dtype contracts required by existing
  workflows, including nullable, string, categorical, datetime, and object
  policy boundaries.
- [x] M8. Public workflow composition stays Native* from ingress through final user export. Exact row sorts by value or index, including duplicate-label row indexes and `ignore_index=True` RangeIndex relabels, and non-geometry duplicate-row drops preserve `NativeFrameState` as rowset takes with explicit row-position proofs; `set_crs` preserves `NativeFrameState` as a metadata-only CRS relabel, `set_geometry` over an existing native geometry column preserves it as an active-geometry metadata switch, and native-backed GeoSeries preserve state through exact copy/CRS/take/head/tail/drop/reindex/sample/sort-index including duplicate-label and RangeIndex relabels/`__getitem__`/`.loc`/`.iloc`/metadata-relabel composition.
- [x] M9. Enforcement gates prevent regressions and reject new non-native
  functionality before the hold lifts. Strict native grouped coverage now
  treats all-skipped optional-dependency pytest files as successful coverage
  chunks instead of failing with pytest return code 5.

## Surface Matrix

| Surface | Completion Requirement |
|---|---|
| IO read/write | Native ingress and terminal export for existing GPU-capable formats; no eager host geometry mirror before export. |
| Geometry metadata | `NativeGeometryMetadata` feeds bounds, family, validity, emptiness, offsets, precision, and dispatch without host scans. |
| Spatial index/query | `NativeSpatialIndex` produces `NativeRelation` or `NativeRowSet`; public array formatting is terminal export. |
| Predicates | Predicate results feed `NativeExpression`, `NativeRowSet`, or `NativeRelation`; public Series is terminal export only. |
| Metrics | Area, length, distance, centroid components, and match counts feed native expressions and grouped reducers. |
| Joins/nearest | Relation pairs, distances, index labels, and attributes stay device-resident until terminal export. |
| Clip/overlay | Exact topology, cleanup, attributes, provenance, and keep-geometry-type behavior stay native for admitted shapes. |
| Dissolve/groupby | Group codes, reducers, grouped geometry, keys, and output index semantics stay native for admitted workflows. |
| Constructive ops | Dynamic output assembly is GPU-shaped and carries provenance and metadata through native consumers. |
| Public composition | Exact sanctioned pandas transitions preserve native state; unknown transitions invalidate or export explicitly. |

## Acceptance Gates

The hold lifts only when all gates pass and maintainers explicitly mark this PRD
complete.

- `docs/dev/native-format-inventory.md` has no `partial`, `debt`, or
  `implicit-host` entries for existing GPU-relevant surfaces.
- Strict-native tests cover every admitted public workflow and every explicit
  decline path.
- Source guards reject raw `cp.asnumpy`, unnamed runtime D2H, active raw device
  `.get()`, unclassified host row-position conversion, and hidden Shapely or
  pandas assembly in `src/vibespatial`.
- Full pipeline sparkline has no unexplained CPU-heavy stage and no
  non-terminal materialization.
- 10K repeat-3 shootout correctness is complete and geomean/aggregate do not
  regress from the accepted Native* baseline.
- Upstream GeoPandas compatibility slices pass for touched surfaces.
- Every remaining scalar fence has byte budget, operation name, canary, and an
  explicit reason it is inherent.

## Execution Rules

- Start every new task by routing through intake. If it is not Native*
  completion work, defer it.
- Prefer deleting compatibility boundaries over documenting them.
- If a boundary cannot be deleted, decide whether it is terminal export,
  bounded fence, or unsupported-native decline.
- Do not add new public feature surface while this PRD is open.
- Do not mark a milestone complete without a canary, strict-native test, and
  benchmark/profile artifact.
- Do not optimize a row-shaped fallback when the required solution is a
  segment, ring, relation, group, tile, or output-byte kernel.

## Completion Record

When complete, update this section with:

- accepted baseline commit
- verification commands and artifact paths
- shootout geomean and aggregate comparison
- remaining terminal export surfaces
- explicit maintainer approval to lift the feature hold
