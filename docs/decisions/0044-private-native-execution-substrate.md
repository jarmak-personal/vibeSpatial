---
id: ADR-0044
status: accepted
date: 2026-04-23
deciders:
  - vibeSpatial maintainers
tags:
  - architecture
  - gpu
  - performance
  - public-api
  - dataframe
---

# Private Native Execution Substrate For Public GeoPandas APIs

## Context

vibeSpatial can now make individual GPU operations fast when a benchmark focuses
on one metric. The performance failure mode appears when a new public workflow
composes several otherwise GPU-capable operations. Recent public shootouts show
that correctness generalizes better than performance: workflows can produce the
right GeoPandas-compatible output while still losing to GeoPandas because the
hot path repeatedly rebuilds pandas-shaped intermediate objects.

ADR-0042 moved the architectural boundary to device-native results and explicit
compatibility export. ADR-0043 rejected a broad staged-planner or eager-chain
interceptor after an experiment moved timing only slightly while dropping public
shootout correctness from 14/14 to 9/14.

ADR-0045 adds the latency budget for the small transient native work items that
make this substrate useful in real public workflows.

The remaining gap is not the public API. Users should continue to write normal
GeoPandas code. The gap is the internal execution model: too many GPU-selected
paths still use GeoDataFrame, GeoSeries, pandas Index, and pandas Series objects
as intermediate execution currency instead of as ingress and export surfaces.

## Decision

Adopt a private GPU-native execution substrate underneath the exact public
GeoPandas-compatible API.

The public API contract remains:

- public objects are real `GeoDataFrame`, `GeoSeries`, and `pd.Series` objects
- public method names, arguments, return types, CRS behavior, index behavior,
  repr behavior, mutation behavior, and materialized outputs match GeoPandas
- unsupported native acceleration materializes or falls back observably
- strict-native mode fails on hidden host execution instead of silently
  exporting to pandas or Shapely

The internal execution contract changes:

- vibeSpatial-owned public operations may create private native execution state
  for sanctioned downstream vibeSpatial operations
- native state is not a general pandas proxy, not a public lazy dataframe, and
  not an unbounded planner
- sanctioned methods may lower locally into private native state when an
  explicit admissibility contract is satisfied
- unknown pandas operations clear or materialize native state instead of
  preserving stale sidecars

## Native Substrate

The canonical internal carriers are:

- `NativeFrameState`: immutable logical frame state containing geometry buffers,
  attribute storage, CRS, column order, index plan, provenance, row count,
  lineage, residency, and stream/event readiness
- `NativeRowSet`: device row positions as the canonical row-flow currency, with
  boolean masks only as cached derivatives
- `NativeRelation`: device left/right pair arrays plus optional distances,
  predicate metadata, source lineage, sortedness, duplicate policy, and cached
  grouped offsets
- `NativeGrouped`: dense group codes, sorted order, offsets/spans, null-key
  policy, output-index plan, and segmented reducer metadata
- `NativeIndexPlan`: explicit index semantics for row-position to public-label
  mapping. RangeIndex labels may remain device-resident as private label
  vectors until explicit public export; duplicate-label and MultiIndex
  semantics must be admitted separately before public `.loc` lowering.
- `NativeExpression`: private device scalar vectors and predicates consumed only
  by sanctioned native operations, not a public replacement for `pd.Series`
- `NativeExportBoundary`: explicit materialization to GeoDataFrame, pandas,
  Arrow, GeoParquet, Feather, Shapely, repr, and debug surfaces

Every persistent device-native carrier must have an explicit stream-readiness
contract. Host conversions such as `.get()`, `cp.asnumpy()`, and host
row-position normalization are export boundaries or fallback boundaries, not
hidden synchronization mechanisms.

## Public Object Attachment

Native state must remain private and conservative.

Full native state must not be blindly stored in pandas `_metadata`. Pandas
propagates metadata through copies and many structural operations, which can
make stale sidecars look valid after row, column, or index mutation.

If native state is attached to a public object, it must be either:

- stored outside pandas metadata and preserved only by sanctioned vibeSpatial
  methods, or
- represented by a lightweight token that revalidates against a private
  registry before use

Unknown pandas operations should drop native state. Preserving native state is
an optimization that requires proof, not the default behavior.

Sanctioned public composition may preserve native state only when the row and
index transition is exact, such as copy, column projection, and admitted
RangeIndex boolean filters, `.iloc`/`take` position takes, unique-index drops,
unique-index row reorders, or metadata-only column renames with valid active
geometry, `reset_index(drop=True)` index rewrites, and exact unique-label
`reindex`, label-filter, single-level unique `set_index`, or non-geometry
attribute `assign`/`insert`/scalar `__setitem__` transitions and non-geometry
`pop`/`del` projections. Public methods that delegate to `take` inherit this
only when the
result remains a geometry frame and registry validation succeeds. Public indexer
writes and in-place shape changes must invalidate attached state before
mutation; `.loc` lowering remains a separate admissibility decision because
duplicate and MultiIndex semantics are easy to get wrong.

## Admissibility

Every native fast path must define:

- row-flow semantics
- index semantics
- attribute projection semantics
- geometry family and dimensional behavior
- CRS and active-geometry behavior
- ordering and duplicate behavior
- fallback and export behavior
- materialization and transfer visibility
- strict-native behavior

If a public operation is outside the declared admissible shape, the native path
must decline locally and either run the existing exact public path with an
observable event or fail in strict-native mode.

## Workflow Shapes

Workflow shootouts are evidence, not the optimization target. They should
measure whether native carriers and reusable physical shapes generalize, but
they must not define private paths, script-specific shortcuts, or benchmark
conditionals. A workflow can justify a new contract only when the underlying
shape is reusable outside that script.

Dedicated shape canaries are allowed when they isolate a reusable substrate
contract, such as relation export consumption, rowset selection, grouped
reduction, or terminal native IO. They measure the contract directly; they do
not license workflow-specific branches in production code.

Spatial joins should center on `NativeRelation`, not immediate joined pandas
rows. Semijoin, anti-join, unique left rows, grouped counts, and relation-based
projection should consume the same pair arrays and grouped offsets without
repeated host export.

Overlay and clip should produce `NativeFrameState` with geometry and provenance
that can feed native row selection, area filtering, and terminal export before a
GeoDataFrame is materialized.

Dissolve should consume `NativeGrouped`. Numeric reducers can use segmented
primitive reductions. Grouped geometry union should flow through
`NativeGrouped` rows and offsets; production acceleration still requires
family-specialized n-ary/grouped kernels rather than Python tree reduction.

Terminal IO should prefer native export. `to_arrow`, `to_parquet`, and
`to_feather` should consume native frame state directly when present instead of
first materializing a full GeoDataFrame.

## Consequences

- Public compatibility remains the external contract while private native state
  becomes the execution substrate.
- Performance work can target reusable physical shapes without reviving the
  broad planner rejected by ADR-0043.
- Native execution becomes stricter: hidden host transfers, stale sidecars, and
  accidental pandas assembly are correctness and performance bugs.
- Some current helper APIs that accept host row positions or pandas attribute
  frames will need to be split into native and export variants.
- Attribute execution remains a known risk. Arrow-backed attributes are a useful
  compatibility/export boundary, and all-valid numeric/bool device attributes
  are admissible for private reducers. Nullable, string, categorical, datetime,
  object, and default device-backed public-frame behavior require explicit dtype
  contracts and canary proof before expansion.
- Debugging becomes more explicit because intermediate GPU state is no longer
  automatically visible as pandas objects.

## Implementation Stance

The migration should start with guardrails and terminal exports before broader
row selection or expression lowering.

1. Apply ADR-0045 transient native work budgets to every admitted shape.
2. Add materialization firewalls and stale-state detection.
3. Introduce private native carriers behind existing native result objects.
4. Make terminal exports native-aware.
5. Rework relation-preserving spatial join shapes.
6. Add native rowset selection only for explicit native selectors.
7. Rework grouped dissolve around `NativeGrouped`.
8. Add private expression fusion only after row/index semantics are stable.
9. Expand device attribute storage only after profiling proves it is the next
   bottleneck.

## Alternatives Considered

- Build a broad lazy GeoDataFrame planner.
  Rejected. ADR-0043 documents that this shape regressed public correctness.
- Expose new public native dataframe objects.
  Rejected. Users expect exact GeoPandas-compatible public APIs.
- Store full native state in pandas metadata and preserve it optimistically.
  Rejected. Pandas metadata propagation makes stale native state likely.
- Keep optimizing individual public methods one at a time.
  Rejected. This does not generalize performance to new workflows.
- Move all tabular state to cuDF immediately.
  Rejected for now. Device attributes are important, but general pandas dtype
  and index semantics need explicit contracts before cuDF becomes the default
  public-frame backing.
