# Degeneracy Corpus

<!-- DOC_HEADER:START
Scope: Deterministic degeneracy corpus for overlay, clip, and segment-regression verification.
Read If: You are adding or verifying degeneracy cases for overlay, clip, or constructive kernels.
STOP IF: You already have the corpus module open and only need local case detail.
Source Of Truth: Phase-5 degeneracy corpus policy for overlay and clip verification.
Body Budget: 73/220 lines
Document: docs/testing/degeneracy-corpus.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-14 | Request Signals |
| 15-21 | Open First |
| 22-29 | Verify |
| 30-35 | Risks |
| 36-40 | Intent |
| 41-61 | Decision |
| 62-68 | Performance Note |
| 69-73 | Consequences |
DOC_HEADER:END -->

Phase 5 needs a deterministic corpus for the cases most likely to break overlay,
clip, and later GPU constructive kernels.

## Request Signals

- degeneracy corpus
- invalid geometry
- overlay regression
- clip regression
- touching rings
- duplicate vertices

## Open First

- docs/testing/degeneracy-corpus.md
- src/vibespatial/testing/degeneracy.py
- tests/test_degeneracy_corpus.py
- src/vibespatial/spatial/segment_primitives.py

## Verify

- `uv run pytest tests/test_degeneracy_corpus.py tests/test_segment_primitives.py`
- `uv run python scripts/verify_degeneracy_corpus.py`
- `uv run pytest tests/upstream/geopandas/tests/test_overlay.py -k invalid_input`
- `uv run pytest tests/upstream/geopandas/tools/tests/test_clip.py -k test_clip_poly`
- `uv run python scripts/check_docs.py --check`

## Risks

- Random invalid-shape generators are useful for fuzzing but poor regression fixtures.
- Overlay and clip can diverge on invalid inputs if the corpus does not record both expectations.
- Missing hole, touching-ring, or null/empty cases leads to false confidence in later GPU kernels.

## Intent

Keep a small, named, deterministic corpus for the degeneracies Phase 5 must
handle before overlay assembly expands.

## Decision

The repo-local corpus lives in `src/vibespatial/testing/degeneracy.py` and is
verified in three ways:

- segment primitive classification
- `geopandas.overlay(..., how="intersection")`
- `geopandas.clip(...)`

Each case carries explicit expectations instead of only raw geometries.

Current coverage includes:

- shared-vertex line touch
- collinear line overlap
- polygon-with-hole clip/overlay window
- duplicate-vertex polygon
- self-intersecting bowtie polygon
- touching-hole invalid polygon
- null and empty polygon rows

## Performance Note

The corpus must stay deterministic and small. It is for correctness regressions,
not random fuzzing or throughput benchmarks. GPU performance work should consume
the named cases as a compact ambiguity set, not expand them into large host-only
fixture suites.

## Consequences

- overlay and clip verification now share the same repo-local edge cases
- later constructive kernels can reuse the same case names and expectations
- invalid and degenerate behavior is anchored in one discoverable place
