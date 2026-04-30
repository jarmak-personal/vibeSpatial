# Private Native Execution Substrate Plan

<!-- DOC_HEADER:START
Scope: Execution plan for building the private GPU-native substrate beneath exact GeoPandas-compatible public APIs.
Read If: You are planning or implementing native frame state, rowsets, relations, grouped reducers, native export boundaries, or performance generalization work.
STOP IF: You only need operation-local kernel detail already routed by intake.
Source Of Truth: Program plan for implementing ADR-0044 without reviving a broad public lazy planner.
Body Budget: 858/860 lines
Document: docs/dev/private-native-execution-substrate-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-12 | Intent |
| 13-35 | Request Signals |
| 36-53 | Open First |
| 54-65 | Verify |
| 66-82 | Risks |
| 83-98 | Mission |
| 99-128 | Target Shape |
| 129-141 | Non-Goals |
| 142-164 | Working Principles |
| 165-177 | Tracking |
| 178-187 | P0 Transient Native Work |
| 188-320 | Current Branch Slice |
| 321-339 | Implementation Packages |
| ... | (16 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Turn ADR-0044 into an implementation sequence. The goal is to generalize public
workflow performance by making private native state the execution substrate
under exact GeoPandas-compatible APIs.

This plan is deliberately narrower than a public lazy dataframe system. It
codifies local admissible lowering, private native state, explicit export
boundaries, and strict invalidation rules.

## Request Signals

- private native execution
- native substrate
- public performance generalization
- native carriers (`NativeFrameState`, `NativeRowSet`, `NativeRelation`,
  `NativeGrouped`, `NativeSpatialIndex`, `NativeGeometryMetadata`,
  `NativeIndexPlan`, `NativeExpression`, `NativeExportBoundary`)
- materialization firewall
- strict native
- stale sidecar
- native state handle
- implementation package
- admissibility contract
- early signal gate
- kill criteria
- transient native work
- latency budget
- scalar fence budget
- semijoin
- anti-join
- grouped reduce

## Open First

- `docs/dev/private-native-execution-substrate-plan.md`
- `docs/decisions/0044-private-native-execution-substrate.md`
- `docs/decisions/0042-device-native-result-boundary.md`
- `docs/decisions/0043-public-api-physical-plan-coverage.md`
- `docs/architecture/api-dispatch.md`
- `docs/architecture/runtime.md`
- `docs/testing/public-api-performance-roadmap.md`
- `docs/testing/gpu-performance-remediation-plan.md`
- `src/vibespatial/api/_native_result_core.py`
- `src/vibespatial/api/_native_results.py`
- `src/vibespatial/api/geodataframe.py`
- `src/vibespatial/api/tools/sjoin.py`
- `src/vibespatial/overlay/dissolve.py`
- `tests/test_strict_native_mode.py`
- `tests/test_index_array_boundary.py`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run pytest tests/test_decision_log.py`
- `uv run pytest tests/test_strict_native_mode.py -q`
- `uv run pytest tests/test_index_array_boundary.py -q`
- `uv run pytest tests/test_geopandas_dispatch.py -q`
- `uv run pytest tests/upstream/geopandas/tests/test_pandas_methods.py -k groupby_metadata`
- `uv run python scripts/health.py --tier contract --check`
- `uv run python scripts/health.py --tier gpu --check`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Treating public GeoDataFrame objects as the execution model preserves the
  existing composition bottleneck.
- Building public proxy Series or DataFrame objects can break pandas identity,
  alignment, groupby, repr, mutation, and upstream GeoPandas tests.
- Preserving native state through unknown pandas operations can create stale
  sidecars and silent wrong results.
- Host row-position conversion can hide synchronization and transfer costs.
- Arrow-backed attributes can keep joins and projections CPU-shaped if device
  attribute gather is postponed past the point where profiling shows it
  dominates.
- Lowering `.loc` before index semantics are explicit can repeat the ADR-0043
  correctness regression.
- Raising GPU dispatch counts without reducing composition overhead is not
  evidence of generalized performance.

## Mission

Build a private GPU-native substrate that lets public workflows keep executing
on device across operation boundaries while preserving exact GeoPandas public
behavior.

The plan is successful only if all of the following are true:

- public users keep normal GeoPandas APIs and return types
- native state is private, validated, and aggressively invalidated
- sanctioned vibeSpatial methods can consume native carriers without rebuilding
  pandas intermediates
- strict-native mode catches hidden materialization and stale native state
- public shootout canaries improve through reusable physical shapes rather than
  workflow-specific shortcuts

## Target Shape

The target substrate has nine private execution carriers:

- `NativeFrameState`: geometry columns, attributes, active geometry, CRS,
  column order, row count, attrs, provenance, lineage, residency, and readiness.
- `NativeRowSet`: device row positions as canonical row flow, including
  source-validated set algebra for composed filters; masks are cached producer
  or consumer artifacts.
- `NativeRelation`: left/right row-position arrays, optional distances,
  predicate metadata, source lineage, sortedness, duplicate policy, and grouped
  offsets.
- `NativeGrouped`: dense group codes, sorted order, offsets/spans, null-key
  policy, output index plan, and reducer metadata.
- `NativeSpatialIndex`: reusable device index state with parameters, sorted or
  partitioned row order, node/bin metadata, bounds, lineage, and readiness.
- `NativeGeometryMetadata`: compact bounds, geometry-family and dimensional
  flags, validity/emptiness masks, and row-to-part/ring/segment offsets.
- `NativeIndexPlan`: public index semantics for mapping device row positions to
  labels, including duplicate, name, level, ordering, and materialization policy.
- `NativeExpression`: private device scalar vectors, threshold lowerings,
  guarded ambiguity rowsets, and metric ranges consumed by sanctioned native
  operations, not a public `pd.Series` replacement.
- `NativeExportBoundary`: compatibility export to GeoDataFrame, pandas, Arrow,
  GeoParquet, Feather, Shapely, repr, and debug materialization surfaces.

The public layer may attach native state only through sanctioned routes. Unknown
pandas operations clear native state or force explicit export. Full native
state must not be blindly stored in pandas `_metadata`.

## Non-Goals

This plan is not about:

- changing public APIs
- exposing a new public native dataframe object
- building a broad lazy GeoDataFrame planner
- making public `pd.Series` look lazy
- lowering arbitrary pandas expressions
- preserving native state optimistically through pandas internals
- routing around correctness failures to win benchmark scripts
- increasing GPU dispatch counts without reducing end-to-end wall time

## Working Principles

- Public GeoPandas is the compatibility language, not the hot execution model.
- Workflow shootouts are canaries and regression signals, not the design
  center. Optimize reusable physical shapes and native contracts, then use
  workflows to check whether the acceleration generalized.
- Native state preservation is opt-in and must be revalidated before use.
- Unknown pandas operations drop native state by default.
- Local admissibility beats broad planning.
- Row positions are the native row-flow currency; labels belong to
  `NativeIndexPlan`.
- Relation consumers should share pair arrays and grouped offsets instead of
  repeatedly sorting or exporting.
- Reusable spatial indexes and geometry metadata are execution carriers, and
  public row alignment does not imply row-shaped execution.
- CCCL fits compaction, scans, sort-by-key, run-length offsets, and segmented
  numeric reductions; grouped geometry union needs family-specific kernels.
- Every persistent device object carries stream or event readiness metadata.
- Hidden `.get()`, `cp.asnumpy()`, host row-position conversion, and accidental
  `.to_geodataframe()` are materialization boundaries, not helper details.
- Device attribute storage should expand only behind explicit dtype and index
  contracts.

## Tracking

- [x] P0. Transient native work substrate
- [x] M0. Guardrails and evidence
- [x] M1. Private native carriers
- [x] M2. Native-aware terminal exports
- [x] M3. Relation-first spatial join shapes
- [x] M4. Explicit native rowset selection
- [x] M5. Native grouped reducers and dissolve
- [x] M6. Private expression lowering
- [x] M7. Device attribute backend expansion
- [x] M8. Performance gates and cleanup

## P0 Transient Native Work

ADR-0045 is the P0 lens for ADR-0044 work. Native carriers are not enough if
workflows still pay one public-object rebuild, scalar fence, launch, or
allocation per tiny intermediate.

Immediate P0 packages: scalar-fence budgets, small-op batching, native scratch,
small grouped geometry canaries, and 10K transient-shape gates for D2H events,
materialization count, launch count, and device-resident outputs.

## Current Branch Slice

As of 2026-04-27, the implementation on this branch is a broad vertical
slice, not full milestone completion.

Completed in the slice:

- WP0a guardrail plumbing: materialization events are distinct from fallback events.
- WP1a/WP1b carrier skeletons: `NativeFrameState`, `NativeRowSet`,
  `NativeRelation`, `NativeGrouped`, `NativeExpression`, and
  `NativeIndexPlan` exist privately with basic readiness metadata.
- WP2a/WP2b attachment: registry-backed native state preserves exact
  copy/projection/mutation/index/filter/selector/reorder contracts.
- Public GeoDataFrame objects from `NativeTabularResult` attach validated
  `NativeFrameState`; compatibility export remains explicit and observable.
- WP3a/WP3b terminal exports: native-backed `GeoDataFrame.to_arrow`,
  `to_parquet`, and `to_feather` consume private native state.
- Native GeoParquet reads no longer eagerly mirror device geometry buffers to
  host authoritative arrays before the public materialization boundary.
- WP4 probe: existing sjoin relation pairs can be wrapped as
  `NativeRelation`, and private semijoin, anti-join, unique-row, and match-count
  row flows can be derived without joined pandas row assembly.
- WP4 probe: `NativeRelation` semijoin rowsets support sorted unique and
  first-seen order; public `.loc` admits exact row-position label-list takes.
- WP4 probe: `RelationJoinExportResult` consumes attached frame state into
  semijoin/antijoin native frames without assembling the joined GeoDataFrame.
- WP4 canary: `relation-bridge-consumer` preserves device pairs and RangeIndex
  device labels across chained semijoin takes until export.
- WP4 guardrail/admissibility: private semijoin is row-position based;
  unique-label lowering admits only unique single-level indexes.
- WP5 probe: explicit private `NativeRowSet` selection can take a
  `NativeFrameState` after source token and row-count validation.
- WP5 hardening: `NativeAttributeTable.take(..., preserve_index=False)` now
  consistently rebases host, Arrow, loader, and device gathers to a RangeIndex.
- WP5 probe: the point-tree/box predicate now has a private device-rowset
  helper that returns device row positions without CUDA-runtime D2H copies.
- WP5/zero-transfer canary consumes private native reads, hidden
  `NativeFrameState`, and private point-box `NativeRowSet` directly, with fixed
  400x400 selectivity instead of public mask rebuilds or identity takes.
- Native GeoParquet payload writes now try the device-native writer before
  constructing an authoritative host geometry view.
- Terminal export helpers preserve cached `NativeGeometryMetadata` across
  public metadata reconciliation.
- Device GeoArrow family gates use owned family keys instead of host metadata scans; WKB/GeoArrow decode scalar fences are explicit runtime D2H events.
- WP8 probe: index-free private takes, exact column projections, and exact
  scalar or multi-column numeric/bool public assignments preserve pylibcudf-backed
  `NativeAttributeTable` payloads; null-bearing and unsupported dtypes decline.
- WP8 hardening: device `NativeAttributeTable.with_column()` appends admitted numeric/bool columns without runtime D2H, unsupported appended dtypes export explicitly, and relation joined-row native exports can request all-valid device attribute storage including distance columns.
- WP8 hardening: all-valid string/datetime/categorical device columns add grouped `first`/`last`; nullable non-numeric stays movement-only and broader predicates/expressions decline.
- Grouped geometry probe: owned unary and rectangle-coverage reducers consume
  `NativeGrouped` rows/codes without host-normalizing group-code arrays.
- Dense all-valid POINT `OwnedGeometryArray.device_take` uses a direct
  `x[rows]`/`y[rows]` gather and synthetic offsets instead of the generic
  offset-slice path, avoiding scalar output-size probes for this common
  rowset-take case.
- P0 transient predicate probe: `point_in_polygon(..., _return_device=True)`
  keeps lazy device metadata through fused PIP without host null/coarse masks.
- WP4 early signal rail: `relation-semijoin` now measures the first
  relation-first vertical slice: private native reads, right-side spatial
  index, device relation pairs, `NativeRelation.left_semijoin_rowset()`,
  `NativeFrameState.take()`, and native GeoParquet export.
- Device `FlatSpatialIndex` keeps regular-grid rectangle bounds device-lazy
  with only admitted scalar certification fences.
- `RelationJoinExportResult.to_native_relation()` now uses attached
  `NativeFrameState` lineage tokens when they are available, so relation-derived
  rowsets can validate against the original native frames instead of only
  carrying `gdf:id(...)` fallback tokens.
- Join-heavy uses the generic `NativeRelation.right_semijoin_rowset()` plus
  `OwnedGeometryArray.take()` relation row-flow shape instead of host pair export.
- Device dense group codes materialize only with an explicit event on host-only
  exact fallback; bulk grouped disjoint polygon assembly admits all-device
  groups and regular-grid rectangles use a device neighbor certificate.
- WP6 early signal: grouped, relation-side, export-bridge, and public dissolve numeric/bool `sum`/`count`/`mean`/`min`/`max`/`first`/`last`/`any`/`all` reducers plus host metadata take-reducers with pandas-compatible skip-null semantics avoid pandas group assembly. Public dissolve admits categorical and ordinary nullable string/object/numeric single-key group contracts plus scalar object, categorical, and typed nullable multi-key contracts, including `dropna=False` null groups and unobserved categorical-null products where admitted; custom/unhashable object keys explicitly stay on pandas compatibility policy. Integer, boolean, and categorical device-backed single or multi-keys, including nullable keys, can encode dense row codes on device when the grouped-geometry consumer can consume device codes directly.
- Grouped constructive results now defer `as_index=False` reset-index assembly through `NativeAttributeTable` loader metadata instead of forcing grouped reducers through pandas before terminal export; numeric/bool device group indexes stay in a device table.
- Device `NativeGrouped` unary polygon reducers feed grouped overlay union without host group-code assembly for admitted owned dissolve shapes.
- Grouped constructive native results carry grouped-union provenance, including repaired-operation markers when boundary repair runs.
- Constructive output canaries lower pairwise intersections and polygonal parts to `NativeTabularResult` and feed native consumers before public export.
- Row-position, host-array, pairwise/relation index, grouped export, public expression assignment, clip/overlay cleanup mask, spatial index/query/nearest/distance/segment exports, selected-face fallback, lower-level topology CPU export boundaries, device index-label, and owned-geometry host bridges carry strictness/runtime D2H accounting.
- WP9 source cleanup: `src/vibespatial` has no raw `cp.asnumpy` outside the CUDA runtime, no unnamed runtime D2H copies, and zero active zero-copy lint violations for raw device `.get()`.
- Homogeneous device `NativeAttributeTable.concat()` uses `pylibcudf.concatenate` for native concat consumers.
- Homogeneous public `concat(...)` preserves exact same-schema appends with RangeIndex or appended public indexes, including duplicates and MultiIndex.
- Exact boolean Series, row drops, repeated reindex labels, non-geometry set/reset-index relabels, and metadata-only relabels preserve native state.
- Point/lineal/polygonal part expansion, mixed-family explode, and public GeometryCollection explode ingress carry source rows plus family-rowset tags into native consumers.
- Overlay assembly allocations and grouped pair-position expansion use native cardinalities, compacted sizes, host-known pair counts, or host-known capacity; the grouped canary rejects overlay D2H.
- WP8 policy cleanup: device attribute compute/movement/export policies are inspectable.
- WP9 cleanup: public overlay uses the shared D2H bridge; selected-face assembly is device-primary, non-empty filtering uses device row views, and contraction microcell rectangles reduce through grouped device union instead of host boundary walking.
- WP9 cleanup: rectangle/few-right/many-vs-one box admission certifies dense boxes on device; grouped exact difference keeps device bbox pairs, CSR offsets, right-geometry source rows, aligned single-pair rowwise differences, grouped-union fallback, and admitted single-batch scatter rowsets through native paths, and broadcast-right chunk restoration keeps many-vs-one positions device-resident until public scatter.
- Clip and pairwise constructive native-tabular assembly reuse attached `NativeFrameState` Arrow/device attributes for admissible row projections until explicit export.
- Native* substrate scope is complete: device-label `NativeIndexPlan` survives `NativeFrameState -> NativeTabularResult`, device relation-join tabular lowering avoids relation-pair host export, and broader overlay topology remains future family-specific kernel work with explicit boundaries.

Current verification artifacts:

- Artifacts under `benchmark_results/working/` include full profile, relation
  bridge, grouped reducer, and small grouped constructive JSON outputs.
- Zero-transfer passes at 1M with `materialization_count=0`, zero runtime D2H,
  and clean `read_input`, `predicate_filter`, `subset_rows`, and `write_output`
  stages; before native read/write/fixed-selectivity updates it reported 10
  runtime D2H transfers totaling 31,000,012 bytes.
- Shootout post-timing profiles report materialization deltas per statement so
  workflow canaries classify hidden exports without becoming the design target.
- Current 1M zero-transfer timings: `read_input=13.84ms`,
  `predicate_filter=395us`, `subset_rows=1.66ms`, `write_output=4.81ms`.
- The 1M relation-semijoin rail reports `status=ok`, 160,000 selected rows,
  zero materialization, and zero runtime D2H in semijoin/subset/export stages.
- The `relation-bridge-consumer` canary preserves RangeIndex `device-labels`
  through native takes with zero consumer-stage materialization or D2H.
- The relation-semijoin rail now reports 3 fixed-size runtime D2H transfers
  totaling 88 bytes; the 2-transfer/8B device-bounds probe slowed `sjoin_relation`.
- WP6 canaries report `status=ok` at 1M: grouped `native_sum=609us` and
  relation `native_attribute_reduce=5.65ms`; both native stages have zero
  D2H/materialization and decline unsupported reducer states.
- `small-grouped-constructive-reduce` is exact with no materialization, 5 named D2H fences / 56 bytes, and no overlay assembly D2H.
- Materialization events now report payload size; the selective 1M zero-transfer
  path no longer reports `_host_row_positions` in `subset_rows`.
- After the P0 PIP metadata and fused grid-index slices, 1M predicate-heavy
  reports zero runtime D2H/materialization; PIP moved from 6 D2H / 12MB /
  ~3.3ms to 272us wall, and join-heavy grid indexing is 1 D2H / 64B.
- After the relation right-rowset slice, 1M join-heavy `sjoin_query` moved from
  16.76ms / 7,954,976 D2H bytes to 770us / 8 D2H bytes; `assemble_join_rows` moved from 17.74ms to 828us with zero D2H.
- Bulk grouped-disjoint assembly moved 1M join-heavy `dissolve_groups` from
  46.97ms / 8 D2H / 9,500,008 bytes / 1 materialization to about 6.51ms / zero D2H/materialization; overall join-heavy is about 31.6ms / named scalar fences only.
- A per-group GPU disjoint loop was killed: it avoided geometry materialization
  but measured ~0.42s and 1,280 tiny D2H reads at the 100K grouped-polygon scale.
- Constructive output-size fences are named and byte-sized; line merge batches two output sizes into one 16B runtime D2H, and segmentize admits one <=8B allocation fence.
Current next signal: constructive-output, expression-column, and nearest-producer composition now prove known-consumer native handoffs into `NativeFrameState`, `NativeExpression`, `NativeTabularResult`, rowset/grouped consumers, direct assignment, row-aligned repair provenance, exact native concat, and `NativeRelation` right-join remapping. The point-nearest producer now uses the indexed point path for device relations, leaving only named dynamic-output count fences and explicit public host exports visible in canaries; these are bridges, not broad pandas interception.
- Do not broaden public pandas interception beyond exact row-position contracts
  with strictness/export canaries.
- Do not expand device attributes beyond all-valid numeric/bool plus all-valid
  non-numeric grouped `first`/`last` until a canary proves broader predicates
  dominate.
- Bulk grouped assembly is useful but not true topology; broader coverage still needs grouped kernels. Coverage-edge/exact probes cut D2H but slowed 62ms->2.88s and ~50ms->21s+, so early signal killed that path.
- The remaining regular-grid build-index 64B read is an admissibility fence, not just a metadata leak; removing it cleanly needs a deferred dual-plan index contract, not a local hostless-candidate patch.

## Implementation Packages

Use this package split for PRs. Each package lands with tests and avoids public
behavior changes unless explicitly stated.

| Package | Purpose | Primary Output |
|---|---|---|
| P0 | Transient native work budget | scalar-fence budgets, small-op batching, shape canaries |
| WP0 | Guardrails and profile baselines | materialization events, strict-native traps, before-state artifacts |
| WP1 | Native carrier skeletons | private carrier dataclasses and stream/readiness contract |
| WP2 | Attachment and invalidation | native state handles, registry, sanctioned preservation paths |
| WP3 | Terminal exports | native-aware Arrow, GeoParquet, and Feather export |
| WP4 | Relation row flow | `NativeRelation`, semijoin, anti-join, unique-left, group counts |
| WP5 | Rowset selection | explicit `NativeRowSet` row-take and projection paths |
| WP6 | Grouped reducers | `NativeGrouped`, numeric reducers, dissolve integration |
| WP7 | Private expressions | admitted metric filters consumed by native rowsets |
| WP8 | Device attributes | typed device gather and projection where profiles require it |
| WP9 | Cleanup and gates | remove host-shaped admitted-path helpers and publish artifacts |

## Preferred Module Layout

Keep the new substrate private and modular. Avoid growing
`_native_result_core.py` into another all-purpose compatibility file.

- `src/vibespatial/runtime/materialization.py`: materialization events,
  strict-native checks, transfer metadata, and user-visible export reasons.
- `src/vibespatial/api/_native_state.py`: `NativeFrameState`,
  `NativeStateHandle`, registry helpers, lineage/fingerprint validation, and
  state attach/get/drop helpers.
- `src/vibespatial/api/_native_rowset.py`: `NativeRowSet` and
  `NativeIndexPlan`.
- `src/vibespatial/api/_native_relation.py`: `NativeRelation` and grouped
  relation views.
- `src/vibespatial/api/_native_grouped.py`: `NativeGrouped` and grouped
  reducer descriptors.
- `src/vibespatial/api/_native_expression.py`: private expression vectors and
  admitted metric-filter builders.
- `src/vibespatial/api/_native_result_core.py`: keep `NativeTabularResult`,
  `NativeAttributeTable`, and `GeometryNativeResult`; delegate new substrate
  behavior to the private modules above.
- `src/vibespatial/api/_native_results.py`: keep public operation export
  assembly and gradually replace host-shaped relation helpers with
  `NativeRelation`.

## Native State Attachment

Do not put full native state in pandas `_metadata`.

Use one of two safe attachment patterns:

- A private handle stored outside `_metadata`, preserved only by sanctioned
  vibeSpatial methods.
- A lightweight token that revalidates against a private registry before use.

Every handle must validate:

- object identity or sanctioned descendant identity
- row count
- active geometry name
- column order or projected column subset
- index plan compatibility
- geometry backing identity or lineage
- state generation
- stream/event readiness

If validation fails, drop the state and run the exact public path. In
strict-native mode, failing validation inside an admitted GPU-selected hot path
should raise instead of silently materializing.

## Invalidation Matrix

Default rule: clear native state unless preservation is explicitly listed here.

| Public Operation | Native State Behavior |
|---|---|
| `copy(deep=False)` | preserve handle only if immutable buffers and fingerprint validate |
| `copy(deep=True)` | preserve only with immutable/COW device buffers; otherwise clear |
| column projection / label filter | produce projected/taken state when active geometry and index plan remain valid |
| explicit `NativeRowSet` filter | produce `state.take(rowset)` |
| ordinary pandas boolean Series filter | preserve when the mask index exactly matches the source RangeIndex or host-label index; otherwise clear |
| `.loc` with ordinary labels | preserve only when pandas-admitted labels map exactly to source row positions; otherwise clear |
| `.iloc` | preserve only for admitted exact position selectors; otherwise clear |
| `assign` non-geometry column | rebuild attribute state for pandas/Arrow/loader attrs; device attrs clear |
| `__setitem__` non-geometry column | scalar column only; rebuild attribute state for pandas/Arrow/loader attrs |
| `insert`, `pop`, `del` non-geometry column | rebuild attribute state for pandas/Arrow/loader attrs; device attrs clear |
| `__setitem__` geometry column | clear or build a new geometry lineage |
| `set_geometry` | rebuild only if the target geometry has native backing; otherwise clear |
| `rename` | rebuild metadata only for unique columns with a valid active geometry |
| `drop` | rebuild only for unique-index row drops or sanctioned column drops; otherwise clear |
| `reset_index`, `set_index`, `reindex` | preserve exact non-geometry index relabels/repeated labels; missing rows/columns clear |
| `sort_values`, `sort_index` | exact unique-label reorders only; duplicate-label value sorts clear because labels do not identify row order |
| metadata-only `rename_axis`/`set_axis` | preserve exact index relabels and column-label relabels that NativeFrameState can represent; column-axis-name relabels clear |
| exact homogeneous row-wise `concat` | preserve exact same-schema appends with RangeIndex output or appended public indexes |
| active geometry `explode` | preserve admitted point/lineal/polygonal, mixed-family, and GeometryCollection part shapes with source-row and family-tag provenance |
| `merge`, `join`, other concat | clear until a native relation or broader concat contract exists |
| `groupby`, `apply`, arbitrary pandas methods | exact pandas behavior; clear state |
| `to_arrow`, `to_parquet`, `to_feather` | consume native state through export boundary |
| `repr`, iteration, `np.asarray` | explicit materialization boundary |

## Admissibility Contract Template

Before adding a native fast path, write the contract in the implementation PR.
The contract can live in a test docstring, module docstring, or focused helper
comment, but it must be reviewable.

Each contract must state:

- source operation and public API surface
- accepted input geometry families and null/empty behavior
- accepted index plan and duplicate-label behavior
- row-order and stable-order guarantees
- CRS and active-geometry behavior
- attribute dtype and projection behavior
- whether secondary geometry columns are supported
- device residency and stream/event readiness requirements
- exact export behavior
- strict-native behavior when the shape is not admitted
- narrow tests and benchmark artifact required before expansion

No operation may lower to native state because "the data happens to be on GPU".
It lowers only when the contract says row flow, index flow, and export flow are
safe.

## First PR Stack

This is the recommended landing order.

1. WP0a: add materialization events and strict-native tests for explicit export.
2. WP0b: add profile baselines for emergency, retail, insurance, and habitat
   canaries with composition-overhead ratios recorded in artifacts.
3. WP1a: add carrier skeletons with no public call sites.
4. WP1b: add stream/readiness metadata and tests that carrier methods do not
   host-normalize row positions by default.
5. WP2a: add private native-state handle and registry helpers.
6. WP2b: wire handle preservation only through `copy(deep=False)` and simple
   sanctioned projection; all other tested pandas mutations clear state.
7. WP3a: route `to_arrow` through native state for a simple RangeIndex,
   single-geometry shape.
8. WP3b: route `to_parquet` and `to_feather` through the same export boundary.
9. WP4a: wrap existing sjoin relation pairs in `NativeRelation`.
10. WP4b: add semijoin and anti-join rowsets from cached relation grouping.
11. WP4c: add unique-left and grouped-count consumers; measure canaries.
12. WP5a: add explicit `NativeRowSet` filtering for sanctioned rowsets only.
13. WP6a: add `NativeGrouped` numeric reducers.
14. WP6b: integrate grouped dissolve behind existing dissolve contracts.
15. WP7a: add private area-threshold expression lowering for one overlay
   area-filter shape.
16. WP8a: add device numeric/bool attribute gather only if canary profiles
   still show host attribute projection as dominant.
17. WP9: delete or demote helpers that host-normalize admitted GPU paths.

## Verification Matrix

Each package has a minimum verification set before expansion.

| Package | Required Checks |
|---|---|
| WP0 | `tests/test_strict_native_mode.py`, materialization event unit tests, canary baseline artifact |
| WP1 | carrier unit tests, `tests/test_index_array_boundary.py`, no public behavior delta |
| WP2 | invalidation tests for copy, projection, mutation, rename, drop, reset_index, concat |
| WP3 | IO export tests, GeoParquet metadata tests, exact pandas materialization comparison |
| WP4 | sjoin contract tests, duplicate/named index tests, semijoin/anti-join canaries |
| WP5 | explicit rowset selection tests, ordinary pandas selector regression tests |
| WP6 | grouped reducer tests, dissolve upstream slice, groupby metadata upstream slice |
| WP7 | metric-filter contract tests, precision-boundary tests, overlay canary |
| WP8 | dtype-specific attribute tests and canary proof that attributes dominate |
| WP9 | full contract and GPU health, 10K shootout canaries, 1M pipeline sparkline |

Do not promote a package from private diagnostics to public execution until its
required checks pass and a profile artifact shows the expected bottleneck moved.

## Early Signal Gates

This plan must prove shape early. Do not spend another long implementation push
before learning whether the architecture moves the real bottleneck.

Each implementation package must define an early signal gate before broadening
scope. The gate must be cheap enough to run during development and specific
enough to kill or redirect the package.

### Global Stop Rules

Stop and reassess before continuing if any of these happen:

- a vertical slice needs broad pandas method interception to show benefit
- public correctness drops on the canary matrix
- native state preservation requires optimistic `_metadata` propagation
- a fast path cannot explain index, duplicate-label, or row-order behavior
- composition-overhead ratio does not move on the target canary
- GPU trace improves only by adding tiny helper dispatches while wall time does
  not improve
- the first useful speedup depends on workflow-specific script detection
- strict-native cannot distinguish user-visible export from hidden hot-path
  materialization

### Probe Before Build

Before implementing each package broadly, write the smallest probe that proves
the expected bottleneck can move:

| Package | Early Signal Probe | Continue Only If |
|---|---|---|
| WP0 | instrument one existing hidden host conversion in a canary path | the event appears in profile output without changing results |
| WP1 | construct carriers from one existing `NativeTabularResult` and one relation result | carrier creation does not call `.get()` or host-normalize row positions |
| WP2 | preserve then invalidate one private handle across `copy`, projection, and mutation | stale handles are dropped or rejected deterministically |
| WP3 | export one simple native-backed frame to Arrow without `to_geodataframe()` | output matches public export and transfer counts are lower |
| WP4 | compute semijoin rowset from existing sjoin pairs without joined pandas rows | target canary stage time or composition overhead moves measurably |
| WP5 | apply one explicit `NativeRowSet` take to geometry and attributes | row order and index output match exact materialized reference |
| WP6 | reduce one numeric column by dense group codes without pandas groupby | reducer output and index labels match GeoPandas reference |
| WP7 | lower one area-threshold filter only inside an admitted overlay consumer | rowset matches materialized `area > threshold` including boundary cases |
| WP8 | gather one numeric attribute column on device after relation row flow | attribute projection time is a measured remaining bottleneck |

### Vertical Slice Order

The first real vertical slice should be relation semijoin, not expression
lowering. It has the best signal-to-risk ratio because it can reuse existing
sjoin pair outputs and avoids pretending public `pd.Series` is lazy.

The slice is:

1. existing public `sjoin` produces relation pairs
2. wrap pairs in `NativeRelation`
3. derive a left-row `NativeRowSet` from cached grouped offsets
4. apply rowset to a left `NativeFrameState`
5. export only at the terminal boundary
6. compare against exact public materialization
7. measure composition-overhead ratio and wall time on the relevant canary

If this slice does not reduce composition overhead without public correctness
loss, do not proceed to expression lowering or grouped dissolve. Revisit the
substrate shape first.

### Canary Matrix

Keep the early canary matrix small and stable:

- emergency response: many-few overlay plus composition overhead
- retail trade-area: relation/grouped composition with current above-parity
  guard against regression
- insurance flood screening: clip/overlay/filter/export composition
- habitat compliance: mixed workflow with known CPU/offramp sensitivity

These workflows must not become the implementation contract. A canary failure
should classify the reusable shape that failed, such as relation semijoin,
native rowset selection, grouped reduce, area-filter export, or terminal IO.
Continue only when the fix improves the shape contract without relying on
script-specific structure.

For each package, record:

- correctness fingerprint
- wall time and speedup versus GeoPandas
- hotpath total
- composition-overhead seconds and ratio
- fallback and materialization event counts
- transfer counts
- top five stage tags by time

### Decision Points

At the end of WP3, decide whether native terminal export is enough to justify
continuing relation work. If terminal export does not reduce transfer or
composition metrics, fix export before expanding row flow.

At the end of WP4, decide whether relation-first execution is the correct first
generalization shape. If semijoin and anti-join do not improve canaries,
reconsider whether the dominant cost is relation row flow, attributes, or
geometry reconstruction.

At the end of WP6, decide whether grouped dissolve needs new kernels before
more Python-side substrate work. If numeric grouped reducers improve but
geometry grouping remains slow, stop Python composition work and prioritize
family-specific grouped union kernels.

At the end of WP8, decide whether device attributes should become a default
backend. Do not make cuDF or pylibcudf storage default without dtype contracts
and canary proof.

## Milestone M0: Guardrails And Evidence

### Goal

Prevent another planner-style regression by making hidden materialization and
stale native state visible before new fast paths land.

### Primary Surfaces

- `tests/test_strict_native_mode.py`
- `src/vibespatial/runtime/fallbacks.py`
- `src/vibespatial/runtime/residency.py`
- `src/vibespatial/api/_native_result_core.py`
- `src/vibespatial/api/_native_results.py`
- profiling artifacts under `benchmark_results/`

### Checklist

- [x] Add a materialization event type distinct from CPU fallback when export is
  user-visible but still performance-relevant.
- [x] Make strict-native fail on unexpected `.to_geodataframe()`, `.get()`,
  `cp.asnumpy()`, host row-position conversion, and repr-triggered export in
  GPU-selected hot paths.
- [x] Add stale-state assertions for any native state token or registry entry.
- [x] Add transfer counters to relation, rowset, grouped, and export paths.
- [x] Capture current M6 canary profiles and composition-overhead ratios.
- [x] Define the first admissible shape contracts before code paths use them.

### Exit Criteria

- hidden materialization is observable or fatal in strict-native mode
- stale native sidecars cannot be consumed silently
- before-state performance artifacts exist for the first canary workflows

## Milestone M1: Private Native Carriers

### Goal

Introduce the substrate types privately without changing public behavior.

### Primary Surfaces

- `src/vibespatial/api/_native_result_core.py`
- `src/vibespatial/api/_native_results.py`
- new private modules under `src/vibespatial/api/` or `src/vibespatial/runtime/`

### Checklist

- [x] Add `NativeFrameState` as a private logical frame carrier.
- [x] Add `NativeRowSet` with device positions as canonical representation.
- [x] Add `NativeRelation` over existing relation-pair outputs.
- [x] Add `NativeGrouped` as a grouped execution carrier.
- [x] Add `NativeIndexPlan` with RangeIndex and unchanged-index support.
- [x] Add stream/event readiness metadata to all persistent device carriers.
- [x] Keep existing public return values unchanged.

### Exit Criteria

- existing `NativeTabularResult` and relation results can produce private
  carrier state
- public tests see no behavior change
- no carrier method requires host row positions unless it is an export method

## Milestone M2: Native-Aware Terminal Exports

### Goal

Reduce composition overhead where public semantics are least risky: terminal
exports.

### Primary Surfaces

- `src/vibespatial/api/geodataframe.py`
- `src/vibespatial/io/arrow.py`
- `src/vibespatial/io/geoparquet.py`
- `src/vibespatial/api/_native_result_core.py`

### Checklist

- [x] Route `GeoDataFrame.to_arrow` through native state when present.
- [x] Route `GeoDataFrame.to_parquet` through native state when present.
- [x] Route `GeoDataFrame.to_feather` through native state when present.
- [x] Preserve exact metadata, index, CRS, and geometry encoding behavior.
- [x] Record materialization or transfer events at the export boundary.

### Exit Criteria

- terminal export avoids `GeoDataFrame -> Arrow` detours when native state is
  already present
- upstream IO compatibility remains unchanged
- export paths explain every D2H transfer in event or profile output

## Milestone M3: Relation-First Spatial Join Shapes

### Goal

Make spatial join composition use `NativeRelation` instead of immediately
assembling joined pandas rows.

### Primary Surfaces

- `src/vibespatial/api/tools/sjoin.py`
- `src/vibespatial/api/_native_results.py`
- `tests/test_index_array_boundary.py`
- public shootout canaries using semijoin and anti-join patterns

### Checklist

- [x] Preserve relation pairs and grouped offsets behind public `sjoin` results.
- [x] Implement semijoin and anti-join rowsets from relation state.
- [x] Implement unique-left and grouped-count consumers from cached offsets.
- [x] Keep joined GeoDataFrame materialization exact when requested.
- [x] Decline native lowering outside declared join/index admissibility.

### Exit Criteria

- semijoin, anti-join, `index.unique`, and grouped-count shapes avoid full
  joined pandas assembly
- duplicate and named-index contracts are covered by tests before `.loc`
  lowering expands
- M6 join-heavy canaries show reduced composition overhead

## Milestone M4: Explicit Native Rowset Selection

### Goal

Allow sanctioned native selectors to drive row selection without lowering
general pandas indexing.

### Primary Surfaces

- `src/vibespatial/api/geodataframe.py`
- `src/vibespatial/api/geoseries.py`
- `src/vibespatial/api/_native_result_core.py`

### Checklist

- [x] Recognize explicit private `NativeRowSet` selectors in `__getitem__`.
- [x] Preserve native state across exact row-take and column-project operations.
- [x] Clear native state for unknown selectors and unsupported pandas paths.
- [x] Admit `.loc` only where labels map exactly to source row positions and
  `NativeIndexPlan` preserves the resulting public index.

### Exit Criteria

- native row filtering works only for explicit private selectors
- ordinary pandas boolean Series and label selectors keep exact behavior
- stale sidecars are cleared on mutation, reindex, sort, merge, concat, apply,
  and unsupported in-place operations

## Milestone M5: Native Grouped Reducers And Dissolve

### Goal

Move grouped public workflows from pandas group assembly to `NativeGrouped`.

### Primary Surfaces

- `src/vibespatial/overlay/dissolve.py`
- `src/vibespatial/api/_native_results.py`
- grouped reducer kernels and CCCL primitive wrappers

### Checklist

- [x] Build dense group codes and offsets without host group iteration for admitted categorical, ordinary nullable single-key, scalar object multi-key, categorical multi-key, and typed nullable multi-key dissolve groups.
- [x] Move remaining categorical device-key encoding to native device carriers instead of host factorization.
- [x] Implement numeric and boolean segmented reducers first.
- [x] Keep all-device grouped numeric outputs as `NativeAttributeTable` when
  the next consumer is native.
- [x] Preserve output index and group key semantics through `NativeIndexPlan`.
- [x] Add grouped geometry union only behind family-specific contracts.
- [x] Keep exact GeoPandas export for public dissolve outputs.

### Exit Criteria

- grouped reducers avoid pandas groupby in admitted native paths
- grouped geometry union does not use Python tree reduction on the hot path
- dissolve canaries improve without upstream groupby metadata regressions

## Milestone M6: Private Expression Lowering

### Goal

Lower only narrow expression shapes that are immediately consumed by sanctioned
native operations.

### Primary Surfaces

- geometry metric accessors
- rowset selection paths
- overlay area-filter workflows

### Checklist

- [x] Keep public `geometry.area`, `geometry.length`, and `geometry.geom_type`
  returning exact pandas-compatible objects by default.
- [x] Add private expression vectors for relation distance filters and grouped reducers.
- [x] Produce `NativeRowSet` from comparisons and rowset composition only when
  the next consumer is a sanctioned native operation.
- [x] Add exactness guards around threshold comparisons where precision can
  affect row flow.

### Exit Criteria

- no public Series proxy is exposed
- overlay area-filter canaries avoid materializing intermediate Series in
  admitted paths
- unsupported expression use materializes exactly or fails in strict-native mode

## Milestone M7: Device Attribute Backend Expansion

### Goal

Replace host-shaped attribute gather only where profiling proves it remains the
dominant bottleneck after relation, rowset, grouped, and export work.

### Primary Surfaces

- `NativeAttributeTable`
- IO adapters that can produce device columns
- relation projection and grouped reducer paths

### Checklist

- [x] Define dtype contracts for numeric and boolean attributes first.
- [x] Add device gather/projection for admitted numeric and boolean columns.
- [x] Add nullable/categorical/string/datetime/object policies only after explicit contracts exist; string/datetime movement is admitted.
- [x] Keep Arrow as schema/export metadata when it is not the hot execution
  substrate.

### Exit Criteria

- attribute movement no longer dominates surviving canaries
- unsupported dtypes decline to exact export or observable fallback
- public pandas dtype behavior remains exact after materialization

## Milestone M8: Performance Gates And Cleanup

### Goal

Prove the substrate generalizes performance and remove transitional host-shaped
helpers from admitted paths.

### Checklist

- [x] Re-run 10K repeat-3 public shootout canaries.
- [x] Re-run full 1M pipeline sparkline.
- [x] Compare composition-overhead ratios before and after each milestone.
- [x] Delete or demote host-shaped helper methods that are no longer admissible
  in GPU-selected paths.
- [x] Update ADR or plan docs if implementation reveals a different contract.

### Exit Criteria

- public canaries retain correctness
- emergency, insurance, and habitat reach parity or have documented external
  bounds
- retail does not regress
- GPU-only traces show materially lower composition overhead
- full pipeline sparklines have no unexplained CPU-heavy stages
