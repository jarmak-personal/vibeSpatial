---
id: ADR-0020
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - dispatch
  - geopandas
  - api
  - performance
---

# Public API Dispatch Boundary

## Context

Phase 6a needs two things at once:

- the GeoPandas-facing method surface must stop rebuilding avoidable wrapper
  arrays on every call
- public method routing must be visible when the repo chooses between
  repo-owned execution and host-side Shapely fallback

## Decision

Use one shared API-dispatch boundary:

- reuse the existing `GeometryArray` inside GeoSeries and GeoDataFrame delegate
  helpers
- cache owned-geometry conversions on `GeometryArray`
- emit explicit dispatch events from repo-owned GeoPandas-facing method surfaces

This keeps the hot path leaner while making public routing observable without
forcing slower host prototypes into default execution.

## Consequences

- Public methods can report whether they used a repo-owned path or Shapely host
  execution.
- Owned-buffer conversion can be cached across repeated calls until mutation.
- Delegate helpers stop paying needless `GeometryArray(...)` reconstruction
  overhead.
- Dispatch observability is now separate from fallback observability: both are
  available and neither hides the other.

## Alternatives Considered

- keep method routing implicit and only log fallback events
- force all public methods onto owned host implementations immediately
- leave delegate helpers untouched until a full GPU implementation exists

## Acceptance Notes

The landed implementation adds dispatch events, exposes them through the top-level
`geopandas` shim, reuses the existing geometry extension array in delegate
helpers, caches owned conversions, and keeps the current host-performance-based
fallback decisions explicit.
