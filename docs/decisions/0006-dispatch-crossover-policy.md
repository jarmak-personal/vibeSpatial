---
id: ADR-0006
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - dispatch
  - crossover
  - auto-mode
---

# Per-Kernel Dispatch Crossover Policy

## Context

The runtime already distinguishes `auto`, `cpu`, and `gpu`, but it does not yet
define when `auto` should stop preferring CPU for small workloads. A global size
gate would be too coarse, because bounds, metrics, predicates, and constructive
kernels have materially different launch overhead and crossover behavior.

## Decision

Use fixed per-kernel-class crossover thresholds until adaptive runtime lands.

- explicit `cpu` always stays on host
- explicit `gpu` always attempts device execution
- `auto` dispatches CPU below the class threshold and GPU at or above it
- the initial thresholds are `1K`, `5K`, `10K`, and `50K` rows for coarse,
  metric, predicate, and constructive kernels respectively

## Consequences

- Kernel dispatch code can rely on one shared threshold policy instead of ad hoc size checks.
- Benchmark work now has concrete constants to validate and replace when measurements improve.
- Explicit overrides remain stable even if later runtime adaptation changes the `auto` path.

## Alternatives Considered

- one global crossover threshold for all kernels
- always preferring GPU whenever the runtime is available
- deciding thresholds independently inside each kernel module
- delaying all crossover policy until adaptive runtime exists

## Acceptance Notes

The landed policy encodes fixed thresholds only. `o17.2.10` may replace the
constants with adaptive inputs later, but should preserve the same override semantics.
