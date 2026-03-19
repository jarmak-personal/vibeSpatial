# API Dispatch

<!-- DOC_HEADER:START
Scope: GeoPandas method delegation, dispatch events, and object-construction avoidance.
Read If: You are changing public method delegation, dispatch events, or GeoPandas adapter wiring.
STOP IF: Your task already has the adapter implementation open and only needs local detail.
Source Of Truth: API dispatch policy for GeoPandas-facing method boundaries.
Body Budget: 63/220 lines
Document: docs/architecture/api-dispatch.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-16 | Request Signals |
| 17-23 | Open First |
| 24-28 | Verify |
| 29-34 | Risks |
| 35-46 | Decision |
| 47-56 | Performance Notes |
| 57-63 | Current Behavior |
DOC_HEADER:END -->

## Intent

Define how GeoPandas-facing methods should cross from pandas objects into
repo-owned execution surfaces without paying avoidable object-construction cost
or hiding which implementation actually ran.

## Request Signals

- api dispatch
- method delegation
- dispatch events
- GeoPandas adapter
- public method boundary

## Open First

- docs/architecture/api-dispatch.md
- src/vibespatial/device_geometry_array.py
- src/geopandas/__init__.py
- tests/test_geopandas_dispatch.py

## Verify

- `uv run pytest tests/test_geopandas_dispatch.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Reconstructing wrapper arrays inside delegate helpers adds avoidable overhead per call.
- Caching owned conversions without invalidation on mutation leads to stale state.
- Dispatch events that materialize host buffers defeat the purpose of lightweight metadata.

## Decision

- Reuse the existing `GeometryArray` already stored on `GeoSeries` /
  `GeoDataFrame` instead of rebuilding wrapper arrays inside delegate helpers.
- Allow `GeometryArray` to cache its owned-geometry conversion for repo-owned
  kernels that need it repeatedly.
- Record explicit dispatch events at the public method boundary for repo-owned
  method surfaces such as `buffer`, `offset_curve`, `clip_by_rect`,
  `make_valid`, and `dissolve`.
- Keep host-competitive Shapely paths in place when owned host prototypes are
  slower, but make that choice visible.

## Performance Notes

- Reconstructing `GeometryArray(...)` inside delegate helpers is pure overhead
  on every public method call.
- Caching owned conversions is important because owned-buffer construction is
  the real bridge cost between the extension-array surface and GPU-oriented
  kernels.
- Dispatch events should be lightweight metadata only; they must not materialize
  host buffers or alter row order.

## Current Behavior

- GeoSeries and GeoDataFrame delegate helpers now reuse the existing geometry
  extension array.
- `GeometryArray.to_owned()` caches the owned-buffer conversion until mutation.
- Public method dispatch is observable through `geopandas.get_dispatch_events()`.
- Host fallbacks remain explicit through the separate fallback-event stream.
