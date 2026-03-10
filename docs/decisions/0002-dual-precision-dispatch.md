---
id: ADR-0002
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - precision
  - runtime
  - performance
---

# Dual Precision Dispatch

## Context

Consumer GPUs often have dramatically worse fp64 throughput than fp32, while
GeoPandas and Shapely semantics expect double-precision-like accuracy for real
world coordinates. The repo needs a precision strategy before kernel work
expands because the choice affects buffer layout, kernel signatures, and later
robustness contracts.

## Decision

Store authoritative coordinates in fp64, then choose compute precision at
dispatch time.

- `auto` chooses by device profile and kernel class
- `fp32` forces staged fp32 execution with centering and compensation
- `fp64` forces native fp64 execution

On consumer-style GPUs:

- coarse and metric kernels may use staged fp32
- predicate kernels may use staged fp32 only with selective fp64 refinement
- constructive kernels stay on fp64 until later robustness work proves a safe alternative

On datacenter-style GPUs with favorable fp64 throughput:

- default broadly to native fp64

## Consequences

- Owned buffers keep one authoritative coordinate representation.
- Kernel APIs need a shared precision plan contract instead of ad hoc precision booleans.
- Later kernel work can use consumer GPU throughput without forcing canonical fp32 storage.
- Robustness work remains a separate decision for exact predicates and overlay.

## Alternatives Considered

- fp64 everywhere
- fp32 everywhere
- canonical dual fp32/fp64 buffer storage
- a single global precision switch with no kernel-class distinctions

## Acceptance Notes

The first landed policy exposes `PrecisionPlan` selection and documents staged
fp32 versus native fp64 defaults. Numerical corpus validation and exact
predicate policy remain follow-up work for later work.
