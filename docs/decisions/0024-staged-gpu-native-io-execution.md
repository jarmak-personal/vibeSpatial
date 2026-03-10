---
id: ADR-0024
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - io
  - gpu
  - geoparquet
  - geoarrow
  - wkb
  - geojson
  - shapefile
---

# Staged GPU-Native IO Execution Model

## Context

Phase 6b established repo-owned IO adapters and a support matrix, but it did
not yet define the end-state performance architecture for turning those
adapters into genuinely GPU-dominant IO across the targeted format set.

## Decision

- Keep owned geometry buffers as the only canonical in-memory destination.
- Use one staged execution model across all formats: metadata planning,
  pruning, family partitioning, size scan, and family-specialized decode.
- Treat GeoArrow and GeoParquet as the primary GPU-native fast paths.
- Treat WKB as the primary GPU-native compatibility bridge.
- Keep GeoJSON and Shapefile hybrid at the container level, but require
  batch-oriented decode and owned-buffer assembly rather than per-feature host
  object construction.
- Prefer CCCL primitives for scans, partitions, compaction, and output
  assembly before writing custom decode kernels.
- Set explicit format-level floor targets so IO work is judged on end-to-end
  speedups, not on adapter completeness alone.

## Consequences

- Later IO work now has one stable execution model instead of a format-by-
  format collection of ad hoc improvements.
- The roadmap can prioritize high-return work such as zero-copy GeoArrow
  adoption, GeoParquet pushdown, and GPU WKB decode ahead of lower-return long-
  tail format support.
- Hybrid formats remain supported, but they no longer dictate the architecture
  of the fast path.

## Alternatives Considered

- Treat every format as a bespoke adapter:
  rejected because it fragments the decode pipeline and makes GPU optimization
  format-specific.
- Make WKB the canonical in-memory interchange:
  rejected because it bakes a compatibility encoding into the core memory
  layout and forces extra decode work.
- Keep Phase 6b as the final IO design:
  rejected because adapter ownership alone does not produce GPU-dominant
  throughput or decode avoidance.
