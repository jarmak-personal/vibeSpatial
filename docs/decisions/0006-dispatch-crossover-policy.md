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

## Amendment (2026-04-26)

ADR-0046 amends this decision. Fixed row-count thresholds are now only a
bootstrap fallback for paths that do not yet expose shape-level estimates.
They are not the steady-state dispatch abstraction.

GPU dispatch must move toward physical workload estimates: coordinate count,
vertex count, segment count, candidate-pair count, relation-pair count, group
count, expected output rows, expected output bytes, temporary bytes, device
residency, launch count, and export cost. Public row count may contribute to
that estimate, but it must not be treated as the primary signal once a shape
contract exists.

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
