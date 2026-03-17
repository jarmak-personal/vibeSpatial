# Explicit Fallback Surfaces

<!-- DOC_HEADER:START
Scope: Observable CPU fallback policy for unsupported predicates, geometry mixes, and host-only query paths.
Read If: You are changing fallback visibility, host-only geospatial paths, or runtime observability.
STOP IF: You already have the fallback event API open and only need local implementation detail.
Source Of Truth: Phase-4 explicit fallback policy for predicate and spatial-query surfaces.
Body Budget: 76/220 lines
Document: docs/architecture/fallbacks.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-11 | Request Signals |
| 12-19 | Open First |
| 20-26 | Verify |
| 27-32 | Risks |
| 33-37 | Intent |
| 38-49 | Decision |
| 50-60 | Current Surfaces |
| 61-70 | Observability Contract |
| 71-76 | Consequences |
DOC_HEADER:END -->

## Request Signals

- fallback
- cpu fallback
- unsupported predicate
- unsupported geometry mix
- diagnostics
- runtime visibility

## Open First

- docs/architecture/fallbacks.md
- docs/architecture/runtime.md
- src/vibespatial/fallbacks.py
- src/vibespatial/binary_predicates.py
- src/vibespatial/spatial_query.py

## Verify

- `uv run pytest tests/test_geopandas_fallbacks.py`
- `uv run pytest tests/upstream/geopandas/tests/test_sindex.py -q`
- `uv run pytest tests/upstream/geopandas/tests/test_array.py -k "contains or within or intersects"`
- `uv run python scripts/check_docs.py --check`

## Risks

- Silent host execution makes it impossible to tell whether a GPU path exists.
- Warning-based fallback can pollute upstream behavior if used as the only visibility channel.
- Fallback logs are not useful if they omit the surface and the reason.

## Intent

Make host-only predicate and spatial-query execution observable without changing
GeoPandas-compatible behavior.

## Decision

Use a lightweight fallback-event log instead of warnings as the primary
observability channel.

Current policy:

- fallback surfaces record `surface`, `requested`, `selected`, `reason`, and `detail`
- the GeoPandas shim exposes `get_fallback_events()` and `clear_fallback_events()`
- explicit `gpu` requests in repo-owned kernels still fail loudly
- `auto` / GeoPandas-facing host paths record fallback events whenever they stay on CPU because the GPU path is unavailable or unsupported

## Current Surfaces

Fallback events are currently recorded for:

- GeoPandas-facing binary predicates routed through `array.GeometryArray._binary_method`
- `sindex.query`
- `sindex.nearest`

This covers the host-only predicate and spatial-query entry points landed in
Phase 4.

## Observability Contract

Fallback visibility must be:

- machine-readable in code
- narrow enough for repo-local tests
- quiet enough to avoid changing upstream warning behavior

That is why the event log is preferred over unconditional warnings.

## Consequences

- unsupported GPU paths are now observable in code without changing result semantics
- repo-local tests can assert fallback behavior directly
- later GPU rollouts can shrink fallback-event volume surface by surface instead
  of relying on manual inspection
