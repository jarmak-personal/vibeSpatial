# Make Valid Pipeline

<!-- DOC_HEADER:START
Scope: Compact-invalid-row make_valid pipeline staging and repair-only-invalids policy.
Read If: You are changing make_valid, validity checking, or topology repair pipelines.
STOP IF: Your task already has the make_valid pipeline open and only needs local implementation detail.
Source Of Truth: Make-valid pipeline architecture for compact-and-repair staging.
Body Budget: 68/220 lines
Document: docs/architecture/make-valid.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-16 | Request Signals |
| 17-22 | Open First |
| 23-27 | Verify |
| 28-32 | Risks |
| 33-50 | Decision |
| 51-60 | Dispatch |
| 61-68 | Performance Notes |
DOC_HEADER:END -->

## Intent

Define the repo-owned `make_valid` pipeline so topology repair work only runs on
invalid rows and can later map onto GPU compaction plus constructive repair
stages.

## Request Signals

- make_valid
- validity
- topology repair
- compaction
- invalid rows

## Open First

- docs/architecture/make-valid.md
- src/vibespatial/constructive/make_valid_pipeline.py
- tests/test_make_valid_pipeline.py

## Verify

- `uv run pytest tests/test_make_valid_pipeline.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Running repair on all rows instead of compacted invalids wastes compute on already-valid geometries.
- Validity checking and repair becoming coupled prevents staging them as separate GPU stages.

## Decision

- Compute validity first.
- Compact invalid rows into a dense repair batch.
- Leave valid rows untouched.
- Repair only the compacted invalid subset.
- Scatter repaired rows back into original order.
- When constructing a replacement ``GeoSeries`` from repaired geometry,
  always pass ``index=df.index`` (or ``index=gs.index``) to preserve
  non-contiguous index alignment from upstream operations like ``clip()``
  or ``iloc`` slicing.  Omitting the index creates a default
  ``RangeIndex(0..N-1)`` which silently drops rows during pandas column
  assignment when the DataFrame index is non-contiguous.
- When all rows pass validation and an ``OwnedGeometryArray`` was provided,
  ``MakeValidResult.owned`` carries the original device-resident array so
  downstream stages (e.g., dissolve) can stay on device without re-uploading
  (ADR-0005 zero-transfer chain).

## Dispatch

- ``make_valid_owned()`` owns runtime dispatch via ``plan_dispatch_selection()``
  and records dispatch events internally; the API layer (``GeometryArray``,
  ``DeviceGeometryArray``) does not record its own events.
- Two kernel variants are registered (ADR-0033):
  ``make_valid/gpu-nvrtc`` (polygon/multipolygon GPU repair) and
  ``make_valid/cpu`` (all families, Shapely fallback).
- The ``dispatch_mode`` parameter controls GPU/CPU/AUTO selection.

## Performance Notes

- Validity checking is much cheaper than topology repair, so compacting invalid
  rows is the right default for valid-heavy datasets.
- This staging is directly compatible with CCCL-style `DeviceSelect` and scatter
  primitives.
- The current host implementation already benefits from skipping repair work on
  valid rows.
