---
id: ADR-0005
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - residency
  - zero-copy
  - fallback
---

# Device Residency And Transfer Visibility

## Context

The repo now has explicit precision, null/empty, and robustness policy, but it
still needs a clear answer for where owned geometry buffers live and when they
are allowed to move. Without that, Phase 2 buffer work risks baking in silent
host copies and interop-specific materialization paths.

## Decision

Use lazy device residency with explicit transfer boundaries.

- owned geometry buffers are lazy-resident
- after first device use, owned buffers are device-resident by default
- host materialization is explicit and belongs to APIs such as `to_pandas`,
  `to_numpy`, `values`, and `__repr__`
- non-user transfers must remain visible in diagnostics
- interop should prefer zero-copy views when ownership and layout permit

## Consequences

- Buffer APIs need to distinguish a view from a materialization.
- Runtime diagnostics will need a transfer audit surface in `o17.2.4`.
- Kernel fusion and pipeline work can assume residency stability once buffers are on device.

## Alternatives Considered

- host-resident buffers with eager upload on every GPU call
- unconditional mirroring of host and device copies
- silent host materialization inside unsupported GPU paths
- eager copies for every interop adapter

## Acceptance Notes

The landed policy defines residency and transfer contracts only. Actual owned
buffer storage, diagnostics, and zero-copy adapters land in later work.
