---
id: ADR-0042
status: accepted
date: 2026-04-08
deciders:
  - vibeSpatial maintainers
tags:
  - architecture
  - gpu
  - overlay
  - dataframe
  - compatibility
  - performance
---

# Device-Native Result Boundary And Explicit Compatibility Export

## Context

The repo originally adopted an index-array boundary model: geometry work stayed
on device, attributes stayed in pandas on host, and public spatial operations
crossed the device/host boundary by returning index arrays and then assembling
attributes on the host.

That model was useful as a containment boundary while the library was proving
out correctness and compatibility. It kept the first generation of join,
overlay, dissolve, and clip paths understandable and testable.

It is no longer the right architectural center.

Recent profiling and shootout work shows that the boundary itself is now one of
the biggest recurring costs in the repo:

- overlay-class workloads repeatedly hit expensive public-boundary handling
  around attribute assembly, `keep_geom_type`, lower-dimensional remnants, and
  `GeometryCollection` filtering
- once GPU is selected, host-side re-entry for assembly or semantics handling
  breaks the intended execution family and destroys 10K-scale parity
- specialized kernels and GPU-native rewrites lose much of their value if their
  outputs immediately collapse into pandas/Shapely-shaped intermediate work
- the old rule incorrectly encouraged people to think "index arrays on GPU,
  pandas on host" was the desired steady state rather than a transitional seam

The core diagnosis is simple: the hot architectural boundary is in the wrong
place. vibeSpatial should not treat pandas assembly as the default result model
for GPU-selected workflows. GeoPandas compatibility is still required, but it
must become an explicit export surface rather than the execution model itself.

## Decision

Adopt a device-native result boundary.

### Core contract

When a workflow selects GPU execution, its internal result model stays
device-native until an explicit compatibility or materialization boundary is
requested.

The canonical GPU result shape is:

- device-resident geometry buffers (`OwnedGeometryArray`)
- device-resident provenance or relation arrays such as row indices, source
  tags, row-group ids, and type/dimension flags
- columnar attribute gather plans or device-native attribute tables when
  attribute work is part of the selected GPU workflow

Host pandas/Shapely materialization is no longer the default architectural
boundary for constructive or relational operations.

### Scope

This decision changes the architectural target for:

- overlay
- clip
- dissolve
- constructive operations that currently bounce through host assembly
- join and relation workflows when their natural output is more than a bare
  index pair contract

Low-level spatial-query kernels may still use typed integer index arrays as an
internal contract. What changes is that index arrays are no longer the required
public execution boundary for GPU-selected workflows.

### Compatibility boundary

GeoPandas and pandas remain supported, but as explicit compatibility/export
surfaces:

- `to_geopandas()`
- `to_pandas()`
- `to_shapely()`
- repr/debugging/materialization APIs
- explicit CPU fallback machinery

If a workflow has already selected GPU, it must not silently re-enter host-side
execution just to satisfy an internal convenience boundary.

### Semantics handling

Semantics that were previously handled by host inspection should move toward
typed device-side classification:

- `keep_geom_type`
- lower-dimensional filtering
- provenance tagging
- validity and emptiness filtering
- relation and family classification

If an operation cannot yet preserve semantics in the selected GPU execution
family, the behavior must be explicit:

- in `auto`, emit an observable fallback event at the public boundary
- in explicit GPU or strict-native mode, fail loudly instead of silently
  routing through host execution

### Migration stance

This ADR is the new architectural target even where the current codebase still
contains transitional host assembly.

Existing index-array boundary sites may remain temporarily during migration, but
they must be treated as debt to remove rather than the desired end state.

## Consequences

- Overlay, clip, dissolve, and related workflows can be optimized around
  device-native execution without paying an automatic pandas/Shapely tax.
- Specialized kernels and workflow-specific GPU plans become worth building
  because their outputs can stay in the selected execution family.
- GeoPandas compatibility becomes a clearer layer: export and semantics
  boundary, not the center of every hot path.
- Some existing helper APIs and tests that assume immediate host-side assembly
  will need to be rewritten around device-native result objects.
- The repo will need a stronger internal result model for provenance, row
  ownership, and attribute gather/merge plans.
- Debugging may initially feel less familiar because intermediate results will
  no longer default to pandas objects.

## Alternatives Considered

- Keep the old index-array boundary model as the architectural center and
  optimize around it.
  Rejected. That encourages the wrong performance shape and keeps reintroducing
  host boundaries into GPU-selected workflows.
- Move all public execution to cuDF immediately.
  Rejected. That still conflates execution model and compatibility model and
  does not solve the need for device-native geometry/provenance result objects.
- Limit the new boundary to overlay only.
  Rejected. Overlay exposed the problem most clearly, but the same issue
  appears in clip, dissolve, and other constructive workflows.
- Keep index arrays as the only canonical result and add more host-side
  optimization.
  Rejected. Index arrays remain useful internally, but they are too narrow to
  serve as the sole architectural result model for GPU-selected workflows.
