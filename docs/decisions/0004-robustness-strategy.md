---
id: ADR-0004
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - robustness
  - predicates
  - overlay
---

# Predicate And Overlay Robustness Strategy

## Context

The repo now has explicit precision dispatch and null/empty contracts, but
those do not by themselves guarantee correct predicate signs or topology-safe
overlay results. Nearly-collinear, nearly-cocircular, and ambiguous segment
intersection cases need a separate robustness strategy before predicate and
constructive kernels expand.

## Decision

Use staged exactness.

- coarse and metric kernels may remain bounded-error
- predicate kernels must provide an exact sign guarantee
- constructive kernels must preserve topology and use exact-style fallback for
  ambiguous intersection decisions
- ambiguous rows should be compacted and sent to exact-style fallback, rather
  than running exact arithmetic on every row

The chosen default exact-style fallback is:

- expansion-arithmetic style exact predicates for ambiguous predicate rows
- rational-reconstruction or equivalent exact-style intersection fallback for
  constructive kernels

## Consequences

- Kernel docs must state guarantee level, not just dtype or tolerance.
- Performance work has to measure ambiguity-rate and fallback cost, not only fast-path throughput.
- Later overlay work can build on one shared exactness policy instead of ad hoc epsilon fixes.

## Alternatives Considered

- fp64 alone as the final robustness strategy
- fixed epsilon thresholds as the final decision rule
- universal snap-rounding as the only topology policy
- exact arithmetic on every row

## Acceptance Notes

The first landed policy encodes guarantee levels and fallback classes, while
deferring the actual GPU predicate kernels and Shewchuk-style stress corpus to
later implementation work.
