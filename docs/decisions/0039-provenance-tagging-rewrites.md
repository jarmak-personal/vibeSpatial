---
id: ADR-0039
status: accepted
date: 2026-03-20
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - provenance
  - rewrite
  - optimization
  - dispatch
---

# Provenance Tagging and Rewrite System

## Context

vibeSpatial evaluates eagerly: each spatial operation materializes its result
before the next one runs. This means common patterns like
`gdf.buffer(100).intersects(other)` perform expensive geometry construction
only to answer a yes/no question that `dwithin` could answer directly.

The staged fusion strategy (ADR-0009) handles intra-pipeline optimization on
device-local chains but cannot see across user-visible operation boundaries.
A lightweight metadata system that travels with intermediate results closes
this gap.

## Decision

Attach a frozen `ProvenanceTag` to `GeometryArray` results that records what
operation created them and with what parameters. A declarative registry of
`RewriteRule` definitions maps (producer, consumer) pairs to attempt functions
that check preconditions and substitute cheaper equivalents.

Key properties:

- tags are immutable frozen dataclasses, zero-cost when no rewrite matches
- rules are pure data; adding a new rule is a dataclass + attempt function
- rewrites are observable via `RewriteEvent` deque and JSONL event log
- precondition failures fall through silently to the original operation
- provenance propagates through `GeometryArray.copy()` and `__init__` so it
  survives pandas Series wrapping
- rewrites are globally toggleable via `VIBESPATIAL_PROVENANCE_REWRITES` env
  var (`0`/`false`/`no`/`off` to disable; default enabled) or the
  `set_provenance_rewrites(bool | None)` programmatic override for A/B
  benchmarking; `provenance_rewrites_enabled()` reads: explicit override >
  env var > `True` default
- each `RewriteEvent` carries `elapsed_seconds` wall-clock timing of the
  rewritten computation for performance analysis

Initial rewrite rules:

| Rule | Pattern | Rewrite | Constraint |
|---|---|---|---|
| R5 | `buffer(0)` | identity | always valid |
| R1 | `buffer(r).intersects(Y)` | `dwithin(Y, r)` | point-only, round cap/join |
| R6 | `buffer(a).buffer(b)` | `buffer(a+b)` | positive radii, same style, point-only |

## Consequences

- Users who write inefficient but obvious code get automatic speedups.
- Every rewrite is logged in the dispatch event stream, so profiling and
  debugging remain transparent.
- The tag carries a strong reference to the source GeometryArray; this is
  acceptable because the buffer result is typically larger than its source.
- New rewrite rules require only a rule definition and attempt function in
  `provenance.py`, not changes to dispatch logic.

## Alternatives Considered

- **Full lazy evaluation graph**: Maximum optimization power but fundamentally
  changes GeoPandas-compatible eager semantics. Deferred to future work.
- **Deferred execution context manager**: Opt-in lazy block where operations
  return plan nodes. Worth prototyping later but higher complexity.
- **Declarative pipeline API**: New surface (`gpd.compile([...])`) that is not
  GeoPandas-compatible. Outside the current scope.

## Acceptance Notes

The landed implementation covers the data model, registry, event logging, and
three rewrite rules (R5, R1, R6) with full test coverage. R2 (sjoin rewrite)
and future rules are follow-up work within the same framework.
