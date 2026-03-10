---
id: ADR-0009
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - fusion
  - dag
  - pipeline
---

# Staged Fusion Strategy

## Context

The repo now has owned geometry arrays, coarse kernels, residency diagnostics,
and a probe-first runtime planner. The remaining question is how to eliminate
intermediate buffers in multi-step pipelines without overcommitting to a full
graph runtime too early.

## Decision

Use a lightweight staged operator DAG as the default fusion mechanism.

- fuse ephemeral device-local chains
- persist reusable structures such as indexes and partition metadata
- treat explicit host materialization as a hard boundary
- keep user-facing APIs unchanged
- allow specialized fused kernels as an optimization inside the staged model, not as the only strategy

## Consequences

- The runtime planner has a clear place to choose stage shapes at dispatch time.
- Diagnostics remain visible because stage boundaries are explicit.
- Future kernels can register fusible chains without building a whole-program graph engine first.

## Alternatives Considered

- full lazy evaluation graph
- explicit fused kernels only
- no shared fusion contract until much later

## Acceptance Notes

The landed implementation is a policy module plus tests. It defines how to
classify ephemeral versus persisted intermediates and where fusion must stop.
Actual fused execution and memory accounting remain follow-up implementation work.
