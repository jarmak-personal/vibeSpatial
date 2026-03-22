# Stroke Kernels

<!-- DOC_HEADER:START
Scope: Buffer and offset-curve kernel seam, prefix-sum emission strategy, and Shapely fallback policy.
Read If: You are changing buffer, offset_curve, or stroke-style constructive kernels.
STOP IF: Your task already has the stroke kernel implementation open and only needs local implementation detail.
Source Of Truth: Stroke kernel architecture for buffer and offset-curve constructive work.
Body Budget: 64/220 lines
Document: docs/architecture/stroke-kernels.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-16 | Request Signals |
| 17-22 | Open First |
| 23-27 | Verify |
| 28-32 | Risks |
| 33-43 | Decision |
| 44-52 | Current Scope |
| 53-64 | Performance Notes |
DOC_HEADER:END -->

## Intent

Define the repo-owned buffer and offset-curve kernel seam so stroke-style
constructive work can later move to GPU-friendly prefix-sum and scatter
pipelines.

## Request Signals

- buffer
- offset curve
- stroke kernel
- constructive
- point buffer

## Open First

- docs/architecture/stroke-kernels.md
- src/vibespatial/kernels/
- tests/test_stroke_kernels.py

## Verify

- `uv run pytest tests/test_stroke_kernels.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Current host prototypes are 4-5x slower than Shapely; public dispatch must stay on Shapely fallback until GPU variants land.
- Prefix-sum emission complexity grows with join and cap classification; land simple cases first.

## Decision

- Treat stroke construction as a bulk vertex-emission problem.
- Expand distances once, derive segment frames in parallel, classify joins and
  caps, prefix-sum output counts, and scatter final vertices.
- Land a real repo-owned point-buffer prototype now.
- Land a deterministic LineString offset-curve prototype for simple mitre and
  bevel joins.
- Keep the public GeoPandas surface on explicit Shapely fallback for now because
  current host benchmarks are still slower than direct Shapely execution.

## Current Scope

- `buffer`: owned prototype for positive-distance `Point` rows.
- `offset_curve`: owned prototype for simple `LineString` rows with non-round
  joins.
- GPU-dispatched buffer surfaces (point, linestring, polygon) return
  OwnedGeometryArray directly without Shapely materialization.
- CPU/host fallback paths still defer to Shapely.

## Performance Notes

- Prefix-sum plus scatter is the right output strategy for future GPU stroke
  kernels because per-row vertex counts vary.
- Point buffers are the first clean constructive case because they avoid segment
  topology and still exercise arc sampling.
- Offset curves need bulk segment-frame generation and join classification; the
  current host prototype exists to validate shape and semantics, not to claim
  host-wide speed leadership yet.
- Current host benchmarks at `1K` rows are about `4-5x` slower than Shapely for
  both prototypes; GPU variants have landed and bypass Shapely entirely, while
  the Shapely fallback applies only to CPU/host execution.
