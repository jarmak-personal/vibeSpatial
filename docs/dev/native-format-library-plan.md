# Native Format Library Plan

<!-- DOC_HEADER:START
Scope: Library-wide execution plan for making Native* carriers the complete private internal format across vibeSpatial.
Read If: You are planning or implementing repo-wide Native* carrier adoption, native format invariants, or public workflow composition through native carriers.
STOP IF: You only need the ADR decision context or one operation-local native implementation detail.
Source Of Truth: Program plan for completing the Native* internal execution format across IO, predicates, joins, expressions, grouped reducers, constructive operations, and export.
Body Budget: 520/520 lines
Document: docs/dev/native-format-library-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-14 | Intent |
| 15-34 | Request Signals |
| 35-51 | Open First |
| 52-62 | Verify |
| 63-78 | Risks |
| 79-98 | Mission |
| 99-118 | Definition |
| 119-136 | Carrier Contract |
| 137-206 | Carrier Map |
| 207-287 | Library Surfaces |
| 288-421 | Adoption Phases |
| 422-437 | Acceptance Matrix |
| 438-462 | Measurement Gates |
| ... | (3 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Define the library-wide plan for making `Native*` carriers the complete
internal execution format across vibeSpatial.

This is an implementation plan, not a new public API and not a replacement for
ADR-0044 or ADR-0046. ADR-0044 defines the private native substrate. ADR-0046
defines physical workload shape contracts. This document turns those decisions
into a complete migration plan across IO, geometry storage, metadata, indexes,
predicates, metrics, joins, constructive operations, grouped operations,
attributes, export, tests, benchmarks, and agent workflows.

## Request Signals

- Native format
- Native* format
- complete native substrate
- library-wide native execution
- native carrier adoption
- `NativeFrameState`
- `NativeRowSet`
- `NativeRelation`
- `NativeGrouped`
- `NativeSpatialIndex`
- `NativeGeometryMetadata`
- `NativeExpression`
- `NativeIndexPlan`
- `NativeExportBoundary`
- materialization firewall
- public composition overhead
- native workflow plan

## Open First

- docs/dev/native-format-library-plan.md
- docs/decisions/0044-private-native-execution-substrate.md
- docs/decisions/0046-gpu-physical-workload-shape-contracts.md
- docs/dev/private-native-execution-substrate-plan.md
- docs/dev/constructive-result-unification-execution-plan.md
- docs/testing/performance-tiers.md
- docs/testing/native-coverage.md
- src/vibespatial/api/_native_result_core.py
- src/vibespatial/api/_native_state.py
- src/vibespatial/api/_native_rowset.py
- src/vibespatial/api/_native_relation.py
- src/vibespatial/api/_native_grouped.py
- src/vibespatial/api/_native_expression.py
- src/vibespatial/bench/pipeline.py

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "Native* format library-wide adoption"`
- `uv run pytest tests/test_private_native_substrate.py -q`
- `uv run pytest tests/test_pipeline_benchmarks.py -k "native or relation or grouped" -q`
- `uv run pytest tests/test_strict_native_mode.py -q`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`
- `uv run python scripts/health.py --tier contract --check`
- `uv run python scripts/health.py --tier gpu --check`

## Risks

- Treating `Native*` as a naming convention instead of an execution format
  would preserve pandas-shaped hot paths under new class names.
- Building a broad lazy public dataframe would repeat the ADR-0043 failure mode.
- Making carriers too generic can erase physical workload shape and hide
  performance cliffs.
- Exporting to pandas, Shapely, NumPy, or Arrow mid-pipeline can look correct
  while destroying the GPU performance shape.
- Preserving native state through unknown pandas operations can create stale
  sidecars and wrong answers.
- Adding native carriers without shape canaries can improve local tests while
  failing public workflows.
- Strict-native checks can become noisy if export boundaries are not explicitly
  labeled and tested.

## Mission

Make `Native*` the complete private execution format for GPU-selected
vibeSpatial paths.

Complete means:

- every native-capable public operation has a declared native input carrier,
  physical workload shape, native output carrier, and export boundary
- sanctioned operations consume native carriers directly instead of rebuilding
  public GeoDataFrame, GeoSeries, pandas Series, pandas Index, NumPy arrays, or
  Shapely objects
- unsupported transitions either decline before entering native execution,
  materialize at an explicit export boundary, or fail under strict-native mode
- benchmarks prove shape-level improvements, not only increased GPU dispatch
  counts

This plan is successful when public workflows compose through native carriers
until a real user export is requested.

## Definition

`Native*` is the private internal execution format family.

It is not:

- a public dataframe API
- a lazy pandas replacement
- a serialization format
- an Arrow wrapper
- a blanket promise that every operation stays on GPU

It is:

- a set of typed private carriers with row-count, lineage, residency,
  readiness, null, precision, and index semantics
- a way to preserve physical workload shape across operation boundaries
- the currency sanctioned native producers and consumers use before public
  export

## Carrier Contract

Every persistent native carrier must provide these fields or an explicit reason
why they do not apply:

- `row_count`, `pair_count`, `group_count`, or another cardinality contract
- source lineage token and invalidation semantics
- residency label and device readiness or stream/event readiness
- null, empty, and validity semantics
- index or row-position semantics where rows are involved
- precision policy for coordinate-derived values
- materialization/export behavior
- shape-level work estimates when the carrier feeds dispatch

Every native carrier must be consumed only by sanctioned native operations. A
carrier is not public evidence that an arbitrary pandas operation can preserve
native state.

## Carrier Map

`NativeFrameState` is the logical frame carrier.

- Owns active geometry, attributes, column order, CRS, attrs, provenance,
  index plan, row count, lineage, residency, and readiness.
- Produces row takes, projections, renames, scalar and exact multi-column assignments, expression
  vectors, and terminal native tabular results.
- Must never preserve through unknown pandas mutation.

`NativeRowSet` is row flow.

- Uses row positions as the canonical representation.
- Boolean masks are producer or consumer artifacts, not the primary currency.
- Feeds `NativeFrameState.take`, owned geometry take, relation semijoin,
  anti-join, filters, and native export.

`NativeRelation` is pair flow.

- Carries left/right row-position arrays, optional distances, predicate
  metadata, source lineage, sortedness, duplicate policy, and grouped offsets.
- Feeds semijoin, anti-join, relation count, relation attribute reduction,
  relation-pair attribute filters, nearest join, and relation consumers.

`NativeGrouped` is segmented group flow.

- Carries dense group codes, sorted order, group offsets, observed group ids,
  null-key policy, output index plan, and reducer metadata.
- Feeds numeric reducers, boolean reducers, expression reducers, grouped
  geometry reducers, and dissolve/export assembly.

`NativeSpatialIndex` is reusable spatial-index execution state.

- Carries index parameters, device bounds, sorted or partitioned row order,
  grid/bin/tree metadata, lineage, admissibility limits, total bounds, and
  readiness.
- Feeds candidate generation and relation production without rebuilding from
  public objects.

`NativeGeometryMetadata` is reusable geometry metadata.

- Carries device bounds, family flags, dimensional flags, validity masks,
  emptiness masks, row-to-part/ring/segment offsets, coordinate stats, and
  compact shape summaries.
- Feeds dispatch, precision planning, predicate coarse filters, spatial index
  construction, constructive output assembly, and metric kernels.

`NativeExpression` is private scalar/vector expression state.

- Carries device scalar vectors, dtype, precision, source lineage, row count,
  null policy, and readiness.
- Feeds comparisons to `NativeRowSet`, grouped reducers, relation reducers, and
  native attribute assembly.
- Must not become a public Series proxy.

`NativeIndexPlan` is index semantics.

- Maps row positions to public labels and declares duplicate, MultiIndex,
  name, level, ordering, and materialization policy.
- Range labels may remain device-resident until export.
- `.loc` and label-preserving operations are admitted only after this plan can
  prove exact public semantics.

`NativeExportBoundary` is every public materialization surface.

- Covers GeoDataFrame, GeoSeries, pandas, NumPy, Shapely, Arrow, GeoParquet,
  Feather, repr, debug output, and oracle checks.
- Every hidden `.get()`, `cp.asnumpy()`, row-position host normalization, or
  Shapely conversion must be one of these boundaries or a test failure.

## Library Surfaces

IO ingress:

- Public readers should produce `NativeTabularResult` or `NativeFrameState`
  when the format supports GPU-native ingestion.
- Legacy formats that require CPU parsing may still enter native execution only
  after an explicit ingress boundary.
- GeoParquet and GeoArrow are the first complete native ingress surfaces.

Geometry storage:

- `OwnedGeometryArray` remains canonical owned geometry storage.
- `NativeFrameState` decides how owned geometry composes with attributes,
  index semantics, lineage, and readiness.
- Device-resident owned arrays must expose metadata through
  `NativeGeometryMetadata` instead of repeated host probes.

Attributes:

- Numeric, boolean, and all-valid device columns are admitted first.
- Nullable, categorical, string, datetime, and generic device columns are
  movement-only unless `NativeAttributeTable` marks them compute-admitted.
- Arrow is schema and export metadata unless it is backed by a real device
  column view.

Spatial indexes:

- Spatial indexes are reusable native execution state.
- Index building must return a `NativeSpatialIndex` or a documented transitional
  equivalent with the same invalidation and readiness contract.
- Candidate generation must prefer relation output over public row assembly.

Predicates:

- Predicate kernels should consume owned geometry, metadata, indexes, rowsets,
  or relations.
- Predicate results that feed filters should produce `NativeRowSet`.
- Predicate results that feed joins should produce `NativeRelation`.
- Public boolean Series export is a terminal compatibility path.

Metrics and expressions:

- Metric kernels may still return public arrays for public `.area`, `.length`,
  `.distance`, and related APIs.
- When the next consumer is native, metrics should produce `NativeExpression`.
- Threshold comparisons, boolean combinations, and grouped reducers must stay
  private until an explicit export boundary.

Spatial joins:

- Join query stages should produce `NativeRelation`.
- Joined public row assembly is an export boundary, not the join execution
  shape.
- Semijoin, anti-join, count, nearest-distance, and relation attribute reducers
  should consume the same relation arrays.

Constructive operations:

- Constructive outputs should carry geometry plus provenance, family,
  dimensional, validity, and source-row metadata.
- Overlay, clip, buffer, make-valid, dissolve, and union outputs should not
  route through public GeoDataFrame assembly before sanctioned downstream
  native consumers.
- Dynamic-output assembly must declare count, scan, scatter, offsets, and
  output-byte work units.

Grouped operations:

- Grouped reducers must use `NativeGrouped` as the execution shape.
- Numeric and expression reducers are first-class.
- Grouped geometry reducers need family-specific contracts and shape canaries
  before replacing public dissolve semantics.

Export:

- Public export is allowed, explicit, observable, and tested.
- Public export must preserve GeoPandas semantics exactly.
- Export cost should be measured separately from native execution cost in
  benchmark stages.

## Adoption Phases

### Phase 0: Inventory And Transitional Debt

- List every public operation that can currently dispatch to GPU.
- For each, record current input carrier, output carrier, hidden host probes,
  materialization events, and shape canary coverage.
- Label transitional helpers that are native-like but do not yet implement a
  full `Native*` contract.

Exit criteria:

- every GPU-capable operation has an inventory row
- every hidden D2H probe is classified as export, admissibility fence, or debt
- every current native canary is mapped to a carrier family

### Phase 1: Carrier Base Contract

- Normalize cardinality, lineage, residency, readiness, null, precision, and
  export fields across carriers.
- Add helper assertions for source token, row count, device residency, and
  strict-native materialization boundaries.
- Make carrier construction cheap and validation explicit.

Exit criteria:

- each carrier has focused unit tests for invariants
- strict-native catches unadmitted host conversion
- intake routes carrier work to this plan and the owning implementation files

### Phase 2: Metadata And Index Completion

- Promote reusable device bounds and geometry classification data into
  `NativeGeometryMetadata`.
- Promote `FlatSpatialIndex` or its successor into `NativeSpatialIndex`.
- Separate required scalar admissibility fences from avoidable host probes.

Exit criteria:

- spatial query can reuse native index and metadata across at least two
  consumers
- regular-grid, flat-index, and bounds paths have explicit native carriers
- remaining D2H fences are documented with byte budgets and removal plans

### Phase 3: Row Flow And Relation Flow

- Make row filters, semijoins, antijoins, and exact takes consume
  `NativeRowSet`.
- Make join and predicate pair outputs produce `NativeRelation`.
- Preserve index semantics through `NativeIndexPlan` only where exact.

Exit criteria:

- public join-heavy canary avoids joined row assembly before native consumers
- rowset take has zero runtime D2H for fixed-width admitted device paths, and
  variable-width geometry output-size fences are documented with byte budgets
- relation consumers share pair arrays instead of recomputing or exporting

### Phase 4: Expression Flow

- Expand `NativeExpression` from area to length, distance, centroid components,
  relation distances, and predicate-derived scalar scores.
- Lower only expressions with an immediate sanctioned native consumer.
- Add exactness guards for threshold comparisons where precision can change row
  flow.

Exit criteria:

- area, length, and centroid-component filters can feed `NativeRowSet`
- guarded metric thresholds expose ambiguous rowsets without public Series
- area, length, and centroid-component grouped summaries can feed `NativeGrouped`
- public Series materialization is absent from admitted expression workflows

### Phase 5: Grouped And Attribute Flow

- Extend `NativeGrouped` reducers to the numeric and boolean attribute surface
  needed by public workflows.
- Admit device attributes by dtype contract rather than by backend accident.
- Keep object-like and nullable cases exact through export or explicit decline.

Exit criteria:

- relation attribute reducers and expression reducers share grouped machinery
- device numeric/bool shared attributes can filter `NativeRelation` pairs
- grouped public workflows reduce without pandas groupby in admitted cases
- unsupported dtype use fails or exports observably under strict-native mode

### Phase 6: Constructive Native Flow

- Make constructive outputs carry enough provenance and geometry metadata for
  downstream native consumers.
- Replace public intermediate assembly in overlay, clip, buffer, and dissolve
  when an admissible native consumer follows.
- Keep geometry-producing kernels family-specific and shape-specific.

Exit criteria:

- overlay/clip area workflows avoid intermediate public GeoDataFrame assembly
- grouped constructive reducers declare group, output-byte, and geometry-family
  work units
- public export remains exactly GeoPandas-compatible

### Phase 7: Public Composition Wiring

- Wire sanctioned public operations to discover attached native state through
  registry handles.
- Preserve native state only for exact transitions: copy, projection, admitted
  take/filter, safe rename, safe scalar or multi-column assignment, and exact reset-index
  patterns.
- Drop native state on unknown mutation, arbitrary pandas operations, and
  unsupported index changes.

Exit criteria:

- public workflows compose through native carriers without user-visible API
  changes
- stale state tests cover mutation, reindex, sort, concat, apply, and in-place
  writes
- strict-native distinguishes unsupported public behavior from hidden fallback

### Phase 8: Enforcement And Cleanup

- Add canaries for each carrier family and each high-value composition pattern.
- Teach skills/checklists to require native input/output carrier declarations.
- Delete or quarantine transitional helpers once the real carrier exists.

Exit criteria:

- shape canaries cover rowset, relation, grouped, expression, index, metadata,
  constructive output, and terminal export
- pre-land review can reject new GPU paths that export mid-pipeline without a
  documented boundary
- obsolete host-shaped helper paths are removed from admitted GPU execution

## Acceptance Matrix

| Surface | Native input | Native output | First complete gate |
|---|---|---|---|
| GeoParquet read | file/device columns | `NativeFrameState` | zero-transfer pipeline |
| Bounds/metadata | owned geometry | `NativeGeometryMetadata` | metadata reuse canary |
| Spatial index | geometry metadata | `NativeSpatialIndex` | relation query canary |
| Predicate filter | geometry/index | `NativeRowSet` | zero-transfer predicate filter |
| Spatial join | index/query pairs | `NativeRelation` | relation-semijoin pipeline |
| Metric expression | geometry metadata | `NativeExpression` | native-area-expression pipeline |
| Grouped reducer | dense codes/expression | `NativeGroupedReduction` | grouped-reducer pipeline |
| Relation reducer | relation/attributes | `NativeGrouped` result | relation-attribute-reducer pipeline |
| Relation attribute filter | relation/device attributes | `NativeRelation` | nearest-relation-producer pipeline |
| Constructive output | owned geometry/relation | native geometry result plus row-aligned provenance and metadata | constructive result canary |
| Dissolve | grouped geometry | native tabular result | dissolve pipeline |
| Public export | any native carrier | public object/file | explicit export tests |
## Measurement Gates

Every new or expanded native path needs:

- a focused unit test proving carrier invariants
- a strict-native test proving hidden materialization fails
- a benchmark or pipeline stage proving the native consumer avoids public
  intermediate assembly
- D2H count and byte reporting for every admitted stage
- a public oracle check at an explicit export boundary

Required canary classes:

- rowset take with zero runtime D2H
- relation semijoin and antijoin
- relation attribute reducer
- nearest relation producer, including right-join remap without public pair export
- grouped numeric/bool reducer
- expression-to-rowset filter
- expression-to-grouped reducer
- reusable spatial index query
- geometry metadata reuse
- constructive output to expression or grouped consumer
- terminal native export

## Implementation Rules

- Declare physical shape before primitive choice.
- Name native input and output carriers in the implementation docstring or
  adjacent comment for new native paths.
- Prefer direct carrier consumption over public object reconstruction.
- Keep public APIs exact and boring; all novelty belongs under the private
  native substrate.
- Add native lowering only when the next consumer is known and sanctioned.
- Treat unknown pandas operations as invalidation or export boundaries.
- Do not add a public proxy for `NativeExpression`, `NativeGrouped`, or
  `NativeRelation`.
- Do not optimize an operation by hiding a host conversion in a helper.
- Do not preserve native state unless row, index, attribute, geometry, and
  lineage semantics are all exact.

## Open Decisions

- `NativeSpatialIndex` and `NativeGeometryMetadata` now live in
  `src/vibespatial/api/_native_metadata.py`; operation modules should depend
  on those carrier contracts instead of owning local carrier variants.
- How to represent stream readiness when CuPy, cuda-python, and pylibcudf
  objects are mixed in the same carrier.
- Which dtype families enter `NativeAttributeTable` after numeric and boolean.
- Which public filters are narrow enough for guarded expression lowering.
- When a constructive output should carry full provenance versus compact source
  row lineage.
- Whether benchmark suites should include all carrier canaries by default or
  keep some as targeted rails to control runtime.

## Current Signal

- metric expressions, guarded threshold ambiguity rowsets, relation distances,
  row-aligned point-in-polygon predicate expressions, counts, attributes, and
  `on_attribute` filters compose through Native* carriers and device
  numeric/bool attribute tables before public export
- `NativeGeometryMetadata` and `NativeSpatialIndex` preserve cached
  decode/index state through tabular/frame carriers, rowset takes, and
  terminal export metadata reconciliation
- Arrow WKB/GeoArrow ingress attaches metadata-seeded `NativeFrameState`, and
  device GeoArrow family gates avoid host metadata scans
- `sjoin`, `sindex.query`, and admitted `sindex.nearest` format
  `NativeRelation` pairs only at compatibility boundaries; private consumers
  use relation queries, reducers, device shared-attribute filters, and deferred
  joined-row native tabular lowering directly
- constructive outputs lower through `NativeTabularResult -> NativeFrameState`
  with metadata, provenance, and device-metadata scalar admission fences instead of full host
  row-metadata copies
- `NativeTabularResult` carries `NativeIndexPlan` for private device labels, so relation-join device tabular lowering avoids pair/index host export until user export
- homogeneous device `NativeAttributeTable` fragments concatenate with `pylibcudf.concatenate` and keep numeric/bool attributes on device
- all-valid device `NativeAttributeTable` fragments also support admitted append, projected joined-row export, distance-column append, numeric/bool reset-index group keys, and grouped `first`/`last` take reducers for all-valid string/datetime/categorical columns without falling back to loader-backed pandas assembly; nullable non-numeric columns remain movement-only attributes
- public dissolve admits categorical and ordinary nullable string/object/numeric single-key group contracts plus scalar object, categorical, and typed nullable multi-key contracts, including unobserved categorical-null products, into `NativeGrouped` without pandas group assembly; custom/unhashable object keys are a pandas compatibility policy; integer/bool/categorical device-backed single or multi-keys, including nullable keys, can encode dense row codes on device when the next grouped-geometry consumer can consume device codes
- arbitrary pandas expression/mutator surfaces invalidate native state; exact filters/takes/relabels/concat/explode preserve only admitted contracts
- row-position, pairwise/relation index, host-array, grouped export, cleanup-mask, spatial index/query/nearest/KNN/distance/segment exports, selected-face fallback, lower-level topology export, expression assignment, device-label, and owned-geometry materialization are explicit runtime-counted boundaries
- admitted overlay allocations, grouped pair expansion, non-empty row filters, grouped exact-difference source rows, aligned single-pair grouped difference, grouped-union exact fallback, broadcast-right chunk row restoration, and admitted single-batch difference scatter stay device-shaped through native cardinalities/device CSR offsets; public non-empty cleanup and pandas-facing row export remain explicit terminal boundaries
- source/canary guards reject generic D2H, raw `cp.asnumpy`, unnamed runtime D2H, and active raw device `.get()`; pylibcudf WKB/GeoArrow/GeoJSON scalar fences are explicit runtime events

Native* carrier scope is complete. Multi-candidate exact overlay topology remains future family-specific kernel work and an observable compatibility boundary when grouped union cannot replace it. Broader non-numeric predicates stay out of scope until canaries prove they dominate.
