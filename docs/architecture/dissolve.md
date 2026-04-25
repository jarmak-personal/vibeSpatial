# Dissolve Pipeline

<!-- DOC_HEADER:START
Scope: Grouped dissolve pipeline staging, segmented union, and attribute aggregation policy.
Read If: You are changing dissolve, grouped union, or segmented attribute aggregation.
STOP IF: Your task already has the dissolve pipeline open and only needs local implementation detail.
Source Of Truth: Dissolve pipeline architecture for grouped constructive work.
Body Budget: 86/220 lines
Document: docs/architecture/dissolve.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-16 | Request Signals |
| 17-23 | Open First |
| 24-28 | Verify |
| 29-34 | Risks |
| 35-44 | Decision |
| 45-53 | Pipeline |
| 54-86 | Performance Notes |
DOC_HEADER:END -->

## Intent

Define the repo-owned dissolve pipeline so grouped constructive work can later map
onto GPU sorting and segmented-union primitives instead of Python group
iteration.

## Request Signals

- dissolve
- grouped union
- segmented union
- GeoDataFrame.dissolve
- attribute aggregation

## Open First

- docs/architecture/dissolve.md
- src/vibespatial/overlay/dissolve.py
- tests/test_dissolve_pipeline.py
- tests/test_gpu_dissolve.py

## Verify

- `uv run pytest tests/test_dissolve_pipeline.py tests/test_gpu_dissolve.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Python group iteration dominates wall time if the group-span stage is not bulk.
- Stable in-group row order is required for deterministic output.
- Replacing the full public method instead of just the grouped union stage overcomplicates the GPU transition.

## Decision

- Encode dissolve groups once and preserve stable per-group row order.
- Keep attribute aggregation and geometry union as separate stages.
- Use grouped union as the canonical geometry stage for `GeoDataFrame.dissolve`.
- Build a native grouped constructive result first, then export to GeoPandas at
  the explicit public boundary.
- Favor CCCL-style building blocks: stable sort, run-length encode, reduce-by-key,
  compaction, and scatter/gather.

## Pipeline

1. Encode group keys into dense integer codes.
2. Stable-sort rows by group code.
3. Run-length encode sorted codes into group spans.
4. Reduce non-geometry columns per group.
5. Union each group independently.
6. Reassemble grouped geometries with aggregated attributes.

## Performance Notes

- Sorting and group-span discovery are reusable across attribute and geometry
  work, which keeps the eventual GPU path coherent.
- Numeric and bool `sum`/`count`/`mean`/`min`/`max`/`first`/`last` reducers may consume
  `NativeGrouped` directly for admitted single-key shapes, including categorical keys with
  explicit null-group handling.
- Host metadata columns may use `NativeGrouped` `first`/`last` take-reducers
  with pandas-compatible skip-null semantics.
- `as_index=False` assembly should stay a native export-boundary concern:
  reset-index columns may be represented by deferred `NativeAttributeTable`
  loader metadata until public materialization or terminal IO requires pandas.
- Grouped union should be per-group work dispatch, not one global union followed
  by regrouping.
- Many small polygon groups should still batch when enough groups need real
  reduction. The reusable shape is `OwnedGeometryArray + dense group offsets ->
  grouped constructive result`; public `dissolve` is only the first consumer.
- `o18.x` is allowed to route polygon coverage dissolve groups into a shared-edge
  elimination fast path: cancel duplicate undirected edges inside each group,
  reconstruct grouped boundary linework in bulk, and build the final coverage
  areas without reopening generic overlay topology.
- `o18.x` is also allowed to expose a lazy dissolve surface for predicate-heavy
  workflows: keep grouped members and per-group bounds, answer exact scalar
  `intersects` without materializing the dissolved geometry, answer exact point
  `contains` the same way, and only materialize the true grouped union when a
  geometry-producing surface is actually requested.
- Stable in-group row order matters for deterministic output and debugability.
- Host performance is acceptable enough to route `GeoDataFrame.dissolve` through
  the grouped pipeline today; future GPU work should replace only the grouped
  union stage, not the full public method.
- `o17.9.6.5` is allowed to route axis-aligned rectangle coverages into a
  dedicated grouped GPU union fast path when that workload can be reduced to
  per-group bounds aggregation without reopening generic union topology work.
