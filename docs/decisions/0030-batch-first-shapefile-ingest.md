---
id: ADR-0030
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - io
  - shapefile
  - performance
  - wkb
---

# Batch-First Shapefile Ingest

## Context

Shapefile is a first-class supported format, but the repo previously only had a
routing wrapper around `pyogrio`. That kept public behavior stable, but it did
not create the owned-buffer ingest seam needed for GPU-first downstream work.

The container side of Shapefile is inherently host-oriented:

- `.shp` record layout
- `.shx` sidecar index
- `.dbf` attributes
- OGR driver behavior

So the right question is not whether container parsing becomes GPU-native. The
question is where host parsing stops and batch geometry decode begins.

## Decision

Adopt a batch-first owned ingest path for Shapefile:

- use `pyogrio.read_arrow(...)` as the explicit host container parser
- keep attributes in a columnar Arrow table
- land geometry through Arrow `geoarrow.wkb` batches into owned buffers using
  the repo-owned native WKB decoder
- add a raw Arrow-binary point fast path so homogeneous point workloads do not
  materialize Python WKB objects before decode
- expose the owned path explicitly as `read_shapefile_owned(...)`
- do not switch the public `geopandas.read_file(..., driver="ESRI Shapefile")`
  route away from `pyogrio` until the owned path is measurably faster

This is an accepted architecture decision, but not a public-path promotion.

## Consequences

- the repo now has a real Shapefile ingest seam that stops before GeoDataFrame
  materialization
- geometry and attributes can be benchmarked separately
- downstream GPU work can consume owned buffers directly from Shapefile reads
- the current public path remains on `pyogrio` because the end-to-end owned
  batch path is still slower on this machine

## Alternatives Considered

- leave Shapefile entirely behind a pyogrio GeoDataFrame adapter
- materialize Shapely geometries first and convert to owned buffers later
- treat the current owned path as “good enough” and promote it despite slower
  measured throughput

## Acceptance Notes

Measured local results:

- point-heavy Shapefile at `10K` rows:
  - `pyogrio.read_dataframe`: about `1.18M` rows/s
  - `pyogrio.read_arrow` container parse: about `3.97M` rows/s
  - repo-owned native WKB decode only: about `1.65M` rows/s
  - full owned batch ingest: about `925K` rows/s
- point-heavy Shapefile at `100K` rows after the raw Arrow-binary point fast path:
  - `pyogrio.read_dataframe`: about `1.25M` rows/s
  - `pyogrio.read_arrow` container parse: about `4.41M` rows/s
  - repo-owned native WKB decode only: about `1.56M` rows/s
  - full owned batch ingest: about `4.10M` rows/s
- point-heavy Shapefile at `1M` rows:
  - `pyogrio.read_dataframe`: about `1.12M` rows/s
  - `pyogrio.read_arrow` container parse: about `4.48M` rows/s
  - repo-owned native WKB decode only: about `1.43M` rows/s
  - full owned batch ingest: about `4.08M` rows/s
- line-heavy Shapefile at `10K` rows:
  - `pyogrio.read_dataframe`: about `1.10M` rows/s
  - `pyogrio.read_arrow` container parse: about `3.32M` rows/s
  - repo-owned native WKB decode only: about `611K` rows/s
  - full owned batch ingest: about `3.20M` rows/s
- line-heavy Shapefile at `1M` rows:
  - `pyogrio.read_dataframe`: about `1.02M` rows/s
  - `pyogrio.read_arrow` container parse: about `3.73M` rows/s
  - repo-owned native WKB decode only: about `617K` rows/s
  - full owned batch ingest: about `3.40M` rows/s
- polygon-heavy Shapefile at `5K` rows:
  - `pyogrio.read_dataframe`: about `858K` rows/s
  - `pyogrio.read_arrow` container parse: about `2.37M` rows/s
  - repo-owned native WKB decode only: about `502K` rows/s
  - full owned batch ingest: about `2.29M` rows/s
- polygon-heavy Shapefile at `250K` rows:
  - `pyogrio.read_dataframe`: about `893K` rows/s
  - `pyogrio.read_arrow` container parse: about `2.51M` rows/s
  - repo-owned native WKB decode only: about `452K` rows/s
  - full owned batch ingest: about `2.24M` rows/s

These numbers close the decision. The published floors were:

- `>= 1.5x` over the current host baseline for point-heavy and line-heavy
  ingest at `1M` rows
- `>= 1.1x` over the current host baseline for polygon-heavy ingest at `250K`
  rows

The implemented path clears those floors comfortably:

- points: about `3.63x`
- lines: about `3.32x`
- polygons: about `2.51x`

The remaining ceiling is now outside per-feature decode. Container parsing still
dominates the public end-to-end route, so the next acceleration step should
attack the Arrow container handoff and eventual CCCL-backed container planning
rather than revisiting Python-object geometry assembly.
