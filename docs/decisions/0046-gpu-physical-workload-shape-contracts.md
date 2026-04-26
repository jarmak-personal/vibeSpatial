---
id: ADR-0046
status: accepted
date: 2026-04-26
deciders:
  - vibeSpatial maintainers
tags:
  - architecture
  - gpu
  - performance
  - dispatch
  - physical-plans
---

# GPU Physical Workload Shape Contracts

## Context

Recent profiling of vibeSpatial workflows has shown the same architectural
lesson across multiple operation families: GPU performance generalizes when
work is shaped for the device, not when public API calls are copied onto
individual kernels.

vibeSpatial has accumulated the pieces of that lesson across several ADRs:

- ADR-0042 moved GPU-selected workflows to a device-native result boundary.
- ADR-0043 documented that public API coverage and performance coverage are
  different, and warned against broad public staged planning.
- ADR-0044 introduced a private native execution substrate under exact
  GeoPandas-compatible APIs.
- ADR-0045 made transient native work a first-class latency budget.

Those decisions are correct, but they leave a gap. ADR-0043 is primarily a
warning note: the failed broad planner was the wrong implementation of physical
planning because it rewrote public execution before reusable semantic contracts
existed. The repo still needs a positive contract for how GPU work is described
before kernels, primitive choices, dispatch thresholds, or public workflow
optimizations are accepted.

The missing unit is the physical workload shape.

## Decision

Adopt GPU physical workload shape contracts as the design unit for new or
expanded native GPU work.

A physical workload shape is a reusable device execution form that may sit
under one public method, several public methods, or one stage of a larger
workflow. It is not the public API contract, not a benchmark script path, and
not a broad lazy dataframe planner.

Every new GPU operation, native fast path, or substantial expansion of an
existing path must declare its physical workload shape before implementation.
The contract must cover:

- logical public contract and admissibility boundary
- physical shape family, such as aligned pairwise, broadcast one-to-many,
  all-pairs matrix, candidate-pair refine, many-few fragments, rowset take,
  relation consume, or segmented grouped reduction
- work-unit estimates used for dispatch, including coordinates, vertices,
  segments, candidate pairs, relation pairs, groups, output rows, output bytes,
  and expected temporary bytes
- device input carriers and any operation-local staging layout
- native output carrier, such as `NativeFrameState`, `NativeRowSet`,
  `NativeRelation`, `NativeGrouped`, `NativeSpatialIndex`,
  `NativeGeometryMetadata`, `NativeExpression`, or an owned geometry/scalar
  array
- saturation plan for large-single, many-small, sparse, dense, and skewed
  cases where those cases are relevant
- precision plan and coordinate normalization policy
- transfer, synchronization, and public export boundary
- transient native work budget from ADR-0045
- shape canary, benchmark floor, or profiling rail that proves the shape, not a
  one-off workflow script

Physical shape selection happens before primitive selection. The decision order
is:

1. Check public semantics and admissibility.
2. Check residency and native carrier availability.
3. Estimate physical work units and output size.
4. Select the physical shape and variant.
5. Choose implementation primitives: custom NVRTC, CCCL, CuPy, or a staged
   combination.
6. Decline or fall back observably when the shape contract is not satisfied.

Canonical storage is not execution storage. ADR-0001 and ADR-0008 still define
the stable owned geometry representation, but kernels may create temporary
device-local physical layouts such as sorted partitions, CSR/COO pair lists,
dense all-pairs matrices, grouped offsets, transposed coordinate views, centered
fp32 coordinate buffers, or scratch arenas when those layouts fit the selected
shape.

Dispatch policy must use shape-level work estimates. Fixed row-count thresholds
may remain as conservative bootstrap defaults, but they are not the steady-state
dispatch model for GPU work.

Reusable shape families include:

- aligned pairwise output, where each public output row may still require many
  primitive work units and a reduction back to that row
- broadcast or matrix output, where one side expands across another and the
  implementation must choose between dense tiles, compact relations, or staged
  reductions
- candidate-refine pipelines, where a coarse native structure creates a
  relation and later kernels consume that relation without public row assembly
- segmented grouped work, where sorted order and offsets are the execution
  state for reducers and grouped geometry construction
- constructive provenance pipelines, where generated geometry carries source
  lineage, family and dimension flags, validity or emptiness masks, and row flow
  until export
- dynamic-output assembly, where count, scan, scatter, compact, and offset
  construction are part of the operation contract rather than incidental helper
  code
- terminal native export, where device state is converted to a public
  compatibility format only at an explicit boundary

A public row-aligned result does not imply row-shaped execution. The physical
contract may be vertex, segment, ring, tile, candidate-pair, group, relation,
or output-byte shaped even when the public result is one value or one geometry
per row.

This ADR amends ADR-0001, ADR-0006, ADR-0009, ADR-0012, ADR-0020,
ADR-0033, and ADR-0043. It builds on ADR-0042, ADR-0044, and ADR-0045.

## Consequences

- Reviews can reject GPU code that mirrors public API shape but has no credible
  device execution shape.
- Kernel work becomes more explicit up front: a good implementation starts with
  work units, result carrier, residency, and saturation behavior.
- Older row-threshold and pandas-assembly decisions remain historical
  stepping stones, not the current design center.
- Shape canaries become more important than one-method microbenchmarks when
  validating performance generalization.
- Some current implementations are now clearly transitional debt because they
  have GPU kernels but no complete native physical shape contract.

## Alternatives Considered

- Continue optimizing public methods one by one.
  Rejected. That can improve API coverage while leaving workflows dominated by
  public composition, export, or host assembly.
- Use a broad staged planner or eager public-chain interceptor.
  Rejected. ADR-0043 documents that this implementation shape regressed
  correctness because semantic contracts were not explicit enough.
- Keep fixed row-count dispatch thresholds as the primary policy.
  Rejected. Row count misses the dominant GPU cost drivers for spatial work:
  coordinates, segments, candidate pairs, output size, residency, launches, and
  export.
- Adopt another project's public contracts.
  Rejected. The reusable idea is physical workload shape, not someone else's
  public surface. vibeSpatial still has to preserve exact GeoPandas-compatible
  public semantics.
