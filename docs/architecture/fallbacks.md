# Explicit Fallback Surfaces

<!-- DOC_HEADER:START
Scope: Observable CPU fallback policy for unsupported predicates, geometry mixes, and host-only query paths.
Read If: You are changing fallback visibility, host-only geospatial paths, or runtime observability.
STOP IF: You already have the fallback event API open and only need local implementation detail.
Source Of Truth: Phase-4 explicit fallback policy for predicate and spatial-query surfaces.
Body Budget: 93/220 lines
Document: docs/architecture/fallbacks.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-11 | Request Signals |
| 12-19 | Open First |
| 20-27 | Verify |
| 28-33 | Risks |
| 34-39 | Intent |
| 40-53 | Decision |
| 54-73 | Current Surfaces |
| 74-85 | Observability Contract |
| 86-93 | Consequences |
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
- src/vibespatial/runtime/fallbacks.py
- src/vibespatial/predicates/binary.py
- src/vibespatial/spatial/query.py

## Verify

- `uv run pytest tests/test_geopandas_fallbacks.py`
- `uv run pytest tests/test_clip_public_api.py tests/test_index_array_boundary.py tests/test_io_arrow.py tests/test_spatial_query.py -q`
- `uv run pytest tests/upstream/geopandas/tests/test_sindex.py -q`
- `uv run pytest tests/upstream/geopandas/tests/test_array.py -k "contains or within or intersects"`
- `uv run python scripts/check_docs.py --check`

## Risks

- Silent host execution makes it impossible to tell whether a GPU path exists.
- Warning-based fallback can pollute upstream behavior if used as the only visibility channel.
- Fallback logs are not useful if they omit the surface and the reason.

## Intent

Make host-only execution observable without changing GeoPandas-compatible
behavior, and require explicit fallback or compatibility events before any
post-selection host materialization.

## Decision

Use a lightweight fallback-event log instead of warnings as the primary
observability channel.

Current policy:

- fallback surfaces record `surface`, `requested`, `selected`, `reason`, and `detail`
- the GeoPandas shim exposes `get_fallback_events()` and `clear_fallback_events()`
- explicit `gpu` requests in repo-owned kernels still fail loudly
- `auto` / GeoPandas-facing host paths record fallback events whenever they stay on CPU because the GPU path is unavailable or unsupported
- strict-native tests assert that fallback recording happens before host semantic
  cleanup or Shapely materialization

## Current Surfaces

Fallback events are currently recorded for:

- GeoPandas-facing binary predicates routed through `array.GeometryArray._binary_method`
- `GeometryArray.sindex` and other owned-backed public adapters that must
  materialize a host-side spatial index
- `sindex.query`
- `sindex.nearest`
- `distance_owned` -- element-wise distance GPU-to-CPU fallback
- `geometry_array_dwithin` -- dwithin GPU-to-CPU fallback
- `clip` host semantic cleanup and native clip export when the public result
  cannot stay in the owned/native boundary
- overlay `keep_geom_type` semantic probes and exact remainder paths that must
  leave the owned boundary
- DeviceGeometryArray Shapely bridges for unsupported predicate, measurement,
  and constructive operations
- GeoArrow / WKB / GeoParquet compatibility bridges and staged GPU decode or
  device-writer misses

## Observability Contract

Fallback visibility must be:

- machine-readable in code
- narrow enough for repo-local tests
- quiet enough to avoid changing upstream warning behavior
- recorded before host semantic probes, object-array conversion, or Shapely
  materialization begin

That is why the event log is preferred over unconditional warnings.

## Consequences

- unsupported GPU paths are now observable in code without changing result semantics
- repo-local tests can assert fallback behavior directly
- strict-native mode can reject hidden mid-pipeline host exits instead of only
  warning after the fact
- later GPU rollouts can shrink fallback-event volume surface by surface instead
  of relying on manual inspection
