<!-- DOC_HEADER:START
Scope: Compact-invalid-row make_valid pipeline staging and repair-only-invalids policy.
Read If: You are changing make_valid, validity checking, or topology repair pipelines.
STOP IF: Your task already has the make_valid pipeline open and only needs local implementation detail.
Source Of Truth: Make-valid pipeline architecture for compact-and-repair staging.
Body Budget: 46/220 lines
Document: docs/architecture/make-valid.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Intent |
| 7-14 | Request Signals |
| 15-20 | Open First |
| 21-25 | Verify |
| 26-30 | Risks |
| 31-38 | Decision |
| 39-46 | Performance Notes |
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
- src/vibespatial/make_valid_pipeline.py
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

## Performance Notes

- Validity checking is much cheaper than topology repair, so compacting invalid
  rows is the right default for valid-heavy datasets.
- This staging is directly compatible with CCCL-style `DeviceSelect` and scatter
  primitives.
- The current host implementation already benefits from skipping repair work on
  valid rows.
