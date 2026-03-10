---
id: ADR-0031
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - determinism
  - reproducibility
  - reductions
  - runtime
---

# Determinism And Reproducibility Policy

## Context

GPU-first geometry systems eventually hit nondeterministic arithmetic surfaces:

- floating-point reductions
- scan-like accumulation
- compaction and restore order
- floating atomics

Those are acceptable in the default fast path, but not acceptable for every
user. Scientific, debugging, compliance, and regression-triage workflows need a
clear determinism contract before more GPU reductions land.

The repo already had pieces of this implicitly:

- stable sorting in dissolve planning
- explicit row restoration in staged GPU-friendly designs
- precision and robustness policies that distinguish correctness from speed

What was missing was a single policy defining when reproducibility is promised
and what the guarantee actually means.

## Decision

Adopt a two-mode determinism policy:

- `default`
  - performance-first
  - allows faster reduction and scan implementations
  - does not promise bitwise-identical output for reduction-sensitive GPU work
- `deterministic`
  - explicit opt-in via `VIBESPATIAL_DETERMINISM=deterministic`
  - requires stable output order plus fixed reduction and scan order for
    affected kernels
  - forbids floating-point atomics as the final accumulation mechanism
  - guarantees bitwise-identical output only for same input, same device
    architecture, and same driver/runtime stack

Cross-device bitwise reproducibility is explicitly rejected as a contract.

## Consequences

- determinism becomes a named runtime policy instead of an implementation rumor
- future GPU reductions must declare whether they honor deterministic mode and
  what overhead they incur
- kernel authors now have an explicit preferred implementation shape:
  stable sort, fixed-order reduce, deterministic restore
- debugging and CI now have a single probe command for the baseline dissolve +
  area reproducibility path

## Alternatives Considered

- promise cross-device reproducibility
  - rejected as not technically defensible
- force deterministic order in all modes
  - rejected because it would waste throughput on unaffected operations
- leave determinism undocumented until more GPU reductions land
  - rejected because the performance-first default needs an explicit counter-mode

## Acceptance Notes

The decision adds:

- `src/vibespatial/determinism.py`
- `docs/architecture/determinism.md`
- `scripts/check_determinism.py`

The current proof surface is a dissolve + area aggregation probe repeated many
times on the same input. Today that path is CPU-hosted and already stable, so
the measured overhead is effectively the control baseline for future GPU work.

The important part is not the current overhead number. The important part is
that future GPU reductions now have:

- a named mode switch
- a same-device bitwise contract
- a repeatable verification command
