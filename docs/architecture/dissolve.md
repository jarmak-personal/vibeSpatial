<!-- DOC_HEADER:START
Scope: Grouped dissolve pipeline staging, segmented union, and attribute aggregation policy.
Read If: You are changing dissolve, grouped union, or segmented attribute aggregation.
STOP IF: Your task already has the dissolve pipeline open and only needs local implementation detail.
Source Of Truth: Dissolve pipeline architecture for grouped constructive work.
Body Budget: 63/220 lines
Document: docs/architecture/dissolve.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Intent |
| 7-14 | Request Signals |
| 15-21 | Open First |
| 22-26 | Verify |
| 27-32 | Risks |
| 33-41 | Decision |
| 42-50 | Pipeline |
| 51-63 | Performance Notes |
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
- src/vibespatial/dissolve_pipeline.py
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
- Treat pandas aggregation semantics as the public contract boundary.
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
- Grouped union should be per-group work dispatch, not one global union followed
  by regrouping.
- Stable in-group row order matters for deterministic output and debugability.
- Host performance is acceptable enough to route `GeoDataFrame.dissolve` through
  the grouped pipeline today; future GPU work should replace only the grouped
  union stage, not the full public method.
- `o17.9.6.5` is allowed to route axis-aligned rectangle coverages into a
  dedicated grouped GPU union fast path when that workload can be reduced to
  per-group bounds aggregation without reopening generic union topology work.
