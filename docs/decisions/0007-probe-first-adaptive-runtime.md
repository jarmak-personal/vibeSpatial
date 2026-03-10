---
id: ADR-0007
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - runtime
  - adaptation
  - nvml
  - variants
---

# Probe-First Adaptive Runtime

## Context

The repo now has explicit precision, residency, robustness, and crossover
policy, plus a kernel variant registry. The remaining question is how much
adaptive runtime machinery should land before real kernels, owned buffers, and
streaming pipelines exist.

## Decision

Adopt a probe-first planner as the first adaptive runtime.

- collect one device snapshot per operation or chunk boundary
- choose runtime target, precision plan, variant, and chunk-size hint from
  declared metadata plus the snapshot
- allow a re-plan after the first chunk for streaming workloads
- keep explicit user pins authoritative
- avoid continuous feedback control, mid-kernel switching, and CUPTI-style
  machinery in the first landing

## Consequences

- Kernel authors get a stable variant-registration contract immediately.
- The planner can evolve into a fuller controller later without changing public dispatch APIs.
- NVML remains an optional telemetry source instead of becoming a hard dependency for basic planning.

## Alternatives Considered

- a full live controller with continuous utilization feedback from day one
- static registry metadata with no runtime adaptation at all
- embedding variant logic directly inside each kernel module

## Acceptance Notes

The landed implementation provides typed registry metadata, optional NVML-style
device snapshots, plan objects, and chunk-boundary re-planning. Continuous
control, telemetry history, and richer diagnostics remain follow-up work.
