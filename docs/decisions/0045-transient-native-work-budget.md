---
id: ADR-0045
status: accepted
date: 2026-04-25
deciders:
  - vibeSpatial maintainers
tags:
  - architecture
  - gpu
  - performance
  - native-substrate
  - latency
---

# Transient Native Work Budget

## Context

ADR-0042 moved GPU-selected workflows toward device-native result boundaries.
ADR-0044 defines the private native execution substrate underneath exact
GeoPandas-compatible public APIs. Recent ADR-0044 profiling showed a second
performance center: many real workflows are composed from small transient work
items rather than one large throughput kernel.

Examples include rowset filters, relation consumers, many-few overlay fragments,
grouped reductions with small groups, bounds/type/empty metadata checks,
count-scatter allocation totals, and terminal export preparation. These items
may each do little arithmetic, but they can dominate a workflow when every step
rebuilds public objects, asks the host for scalar sizes, allocates temporary
buffers independently, or crosses a compatibility boundary.

Accessibility-style workflow canaries exposed this pattern, but the pattern is
not workflow-specific. The reusable lesson is that device-resident small
operations need a latency/control-plane budget just as large kernels need
throughput gates.

## Decision

Make transient native work a P0 performance target.

A transient native work item is a small, short-lived device-resident operation
created inside a larger public workflow. It may operate on a small rowset, a
small geometry batch, a tiny group, a scalar allocation count, or intermediate
native metadata. It is not user-visible by itself, but its composition cost is
user-visible.

Transient native work must use private native carriers as execution currency:

- `NativeFrameState` for frame and geometry state
- `NativeRowSet` for row flow
- `NativeRelation` for pair flow
- `NativeGrouped` for grouped work
- `NativeExpression` for admitted private scalar/vector predicates
- `NativeExportBoundary` for explicit public compatibility export

The design target is:

- zero hidden materialization inside admitted native transient stages
- device-resident metadata and row-flow by default
- batched scalar reads when a host allocation size is still unavoidable
- shape-level batching for many tiny operations instead of Python loops over
  groups, fragments, or relation slices
- reusable scratch or arena allocation for temporary device buffers
- shape canaries that measure reusable work-item classes, not workflow scripts

Throughput gates remain necessary for large kernels, but they are not sufficient
evidence of generalized performance. A GPU implementation that wins a large
kernel benchmark can still fail ADR-0045 if normal public composition pays too
many scalar fences, launches, allocations, or compatibility exports.

## Budget Rules

Every new or expanded native transient shape must declare a budget covering:

- runtime D2H transfer count and bytes
- materialization count
- scalar allocation fences
- launch or stage count when observable
- device residency of inputs and outputs
- public export boundary, if any
- shape canary command and expected scale

Default budget stance:

- Hidden materialization in admitted native stages is forbidden.
- Device row positions are preferred over host row positions.
- Scalar size reads should be batched across independent count-scatter totals.
- Public `GeoDataFrame`, `GeoSeries`, pandas, or Shapely construction is an
  export boundary, not transient execution currency.
- Workflow canaries may measure impact, but production code must target the
  reusable transient shape.

## Consequences

- ADR-0044 work is reprioritized around composition latency, not only carrier
  coverage.
- Small-operation canaries become first-class acceptance signals.
- Generic primitives such as packed scalar reads, batched count-scatter totals,
  device rowset takes, grouped-offset caches, and scratch allocation can produce
  broad gains without workflow-specific branches.
- Some current helpers that synchronously return Python sizes remain debt until
  they are replaced by device-planned allocation or batched scalar resolution.
- Launch overhead and allocation churn become reviewable performance bugs even
  when total D2H bytes are small.

## Alternatives Considered

- Keep transient work as an implementation detail of ADR-0044.
  Rejected. The repeated small-operation cost is a distinct design center and
  needs explicit budget rules.
- Optimize each workflow canary directly.
  Rejected. That would repeat the benchmark-specific path ADR-0043 warned
  against.
- Focus only on large-kernel throughput tiers.
  Rejected. Throughput gates do not catch public composition latency.
- Build a broad public lazy planner to fuse transient work.
  Rejected. ADR-0043 documents why broad public interception is not the right
  next step. ADR-0045 keeps the scope to admitted private native shapes.
