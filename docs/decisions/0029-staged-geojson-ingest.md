---
id: ADR-0029
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - io
  - geojson
  - performance
  - tokenizer
---

# Staged GeoJSON Ingest

## Context

GeoJSON is a first-class hybrid format, but the repo still only had a routed
`pyogrio` adapter for it. That kept public behavior stable, but it did not
create the staged geometry-ingest seam needed for GPU-native work. `o17.6.23`
needs a real tokenizer-plus-assembly pipeline and an honest comparison between
that approach, the existing host adapter, and possible `pylibcudf` use.

## Decision

Adopt a staged GeoJSON ingest design with three layers:

- streaming FeatureCollection tokenizer for container structure
- repo-owned geometry assembly directly into owned buffers
- optional `pylibcudf` exploration for per-feature JSON batches after feature
  boundaries are already isolated

The default implementation for this decision is the streaming tokenizer plus
native geometry assembly. After benchmarking, `read_geojson_owned(...,
prefer="auto")` now selects full `json.loads` plus native assembly on host
because it is faster than the current stream tokenizer. Public
`geopandas.read_file(..., driver="GeoJSON")` stays on `pyogrio` for now; the
new staged path is exposed as an owned-ingest API and benchmark surface until
it is semantically complete enough to replace the public host route.

Follow-on evaluation also added a structural feature-span tokenizer as a
separate strategy. It makes the future CCCL boundary more explicit, but in pure
Python it is slower than both full-json native ingest and the older stream path.
So it remains an opt-in strategy and a design seam for later GPU work, not the
host default.

A second follow-on added an explicit `pylibcudf` strategy. It keeps the same
host feature-span splitter for the outer `FeatureCollection`, then uses
`pylibcudf` for JSON-path extraction, family-local JSON parsing, and typed
coordinate column recovery. That gives the repo a real GPU-assisted tokenizer
path, but it is still not the host-default winner because the current owned
buffer contract remains host-materialized.

A third follow-on prototyped device-side rowization of the full parsed
`features` array via `pylibcudf` plus `interleave_columns`. It is kept only as
an explicit experimental strategy. Measured sweeps showed that it is much
slower than the current host-split GPU path even for tiny homogeneous point
inputs, and it still fails on heterogeneous feature schemas.

A fourth follow-on prototyped wildcard-array extraction from the full
`FeatureCollection` using `$.features[*].geometry.type` and
`$.features[*].geometry.coordinates`. This is the cleaner splitter-free design
seam because it avoids host feature splitting entirely and can assemble owned
buffers for homogeneous point, line, and polygon families from typed
concatenated columns. It also remains explicit-only because current JSONPath
wildcard extraction is still dramatically slower than the host-split GPU path.

That GPU path is now hybrid on purpose:

- point, multipoint, linestring, and multilinestring use coordinates-only
  parsing on device
- polygon and multipolygon keep full-geometry parsing on device because
  coordinates-only parsing collapses ring structure

## Consequences

- GeoJSON now has a real staged owned-ingest seam instead of only a routed host
  adapter
- geometry assembly no longer depends on Shapely objects for the staged path
- the remaining bottleneck was isolated to Python-side tokenization, which
  ADR-0038 resolved with GPU byte-classification (12 NVRTC kernels, 1.8s for
  2.16 GB / 7.2M polygons). CPU property extraction (9.2s) is now the
  remaining bottleneck
- the repo now has a real GPU-assisted GeoJSON tokenizer path instead of only a
  hypothetical seam, which means future work can optimize the device stages
  instead of starting from scratch
- the repo also has an experimental device-rowization prototype, but it is
  intentionally not on the default `pylibcudf` path because the current
  `interleave_columns` approach is dramatically slower than host-span planning
- the repo now also has an experimental wildcard-array GPU path that bypasses
  host feature splitting entirely for homogeneous families, but it is still not
  promoted because JSONPath wildcard extraction is the new bottleneck
- property materialization is now lazy on the owned batch, which avoids paying
  host-side property decode for geometry-only ingest paths
- `auto` now picks the measured host winner instead of forcing the stream path
  prematurely
- the public `read_file` behavior avoids regressions while the owned path
  matures

## Alternatives Considered

- keep GeoJSON entirely behind `pyogrio`
- use `json.loads` of the full FeatureCollection as the permanent design center
- treat `pylibcudf` as the mandatory container parser for standard
  FeatureCollection GeoJSON

## Acceptance Notes

Benchmarks on this machine show:

- point-heavy GeoJSON at `100K` rows:
  - `pyogrio`: about `300K` rows/s
  - full `json.loads` plus native assembly: about `651K` rows/s
  - staged stream-native: about `331K` rows/s
  - structural tokenizer-native: about `126K` rows/s
  - `pylibcudf` GPU tokenizer-native: about `147K` rows/s
- focused wildcard-array rejection sweep:
  - `100` point rows: about `199` rows/s wildcard-array vs about `405` rows/s
    for the current host-split GPU path
  - `1K` point rows: about `274` rows/s wildcard-array vs about `33.2K` rows/s
    for the current host-split GPU path
  - `5K` point rows: about `275` rows/s wildcard-array vs about `94.0K` rows/s
    for the current host-split GPU path
- focused rowization rejection sweep:
  - `100` point rows: about `92` rows/s rowized vs about `4.00K` rows/s for
    the current host-split GPU path
  - `1K` point rows: about `120` rows/s rowized vs about `33.1K` rows/s for the
    current host-split GPU path
- polygon-heavy GeoJSON at `20K` rows:
  - `pyogrio`: about `161K` rows/s
  - full `json.loads` plus native assembly: about `288K` rows/s
  - staged stream-native: about `147K` rows/s
  - structural tokenizer-native: about `52K` rows/s
  - `pylibcudf` GPU tokenizer-native: about `52K` rows/s

Additional sweeps at `10K` and `500K` point rows and at `5K` polygon rows kept
the same ranking. That is enough to validate the staged direction, keep
full-json native as the host winner, and justify focusing the next acceleration
step on CCCL-backed tokenization and direct device-to-owned buffer writeout
rather than on geometry assembly alone.
