---
id: ADR-0016
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - architecture
  - overlay
  - cccl
---

# Shared Overlay Reconstruction Plan

## Context

After segment classification and rectangle clip landed, Phase 5 still needed the
reconstruction plan that turns split edges and labeled faces into union,
difference, and symmetric-difference outputs. Without that plan, each later
constructive surface would be tempted to invent its own topology assembly path.

## Decision

Adopt one shared reconstruction graph:

- classify segments
- emit nodes
- split segments into directed edges
- stable-sort half-edges
- walk rings and chains
- label faces by source coverage
- perform operation-specific face selection
- emit geometry buffers

Only the selection stage changes between union, difference, symmetric
difference, and identity.

## Consequences

- reconstruction logic is reusable across overlay operations
- stable-order and exact-sign requirements are recorded before CUDA work lands
- dissolve can build on the union branch instead of inventing a new graph

## Alternatives Considered

- Separate bespoke reconstruction per overlay operation.
  Rejected because it duplicates the most expensive topology work.
- Pure host-side Shapely reconstruction for now.
  Rejected because it leaves no owned assembly seam for GPU work.

## Acceptance Notes

This decision lands the reconstruction planner, CCCL-oriented stage mapping, and
tests that validate the shared prefix and operation-specific selection stages.
