---
id: ADR-0001
status: accepted
date: 2026-03-10
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - geometry
  - memory
  - performance
---

# Mixed Geometry Storage And Execution

## Context

`vibeSpatial` needs a canonical mixed-geometry layout before owned buffers can
land. The repo benchmarked dense tagged storage, family-split layouts, and
sort-partition execution for mixed inputs.

## Decision

Use a dense tagged mixed representation as the canonical storage model, then
sort-partition by coarse geometry family for truly mixed execution paths.
Near-homogeneous inputs can stay on the tagged path without permanent
family-split storage.

## Amendment (2026-04-26)

ADR-0046 amends this decision by separating canonical storage from physical
execution shape. Dense tagged mixed storage remains the stable owned geometry
contract, but GPU plans may create operation-local layouts such as sorted family
partitions, grouped offsets, pair lists, dense matrices, or scratch buffers when
those layouts better fit the selected workload shape.

Do not read this ADR as requiring kernels to execute directly over the canonical
mixed layout when a different temporary device layout improves coherence,
saturation, or output assembly.

## Consequences

- Owned geometry buffers can target one canonical mixed layout.
- Execution logic can recover warp coherence by partitioning when mixed inputs
  are noisy enough to justify it.
- Early kernels do not need permanent family-specific storage trees.

## Alternatives Considered

- Permanent split-by-family storage as the default layout.
- Common-type promotion for mixed inputs.
- Pure tagged execution without partitioning.

## Acceptance Notes

Benchmarks favored direct tagged execution for dominant-family inputs and
sort-partitioning once family mix falls below the chosen threshold.
