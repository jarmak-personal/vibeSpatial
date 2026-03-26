---
id: ADR-0040
status: accepted
date: 2026-03-25
deciders:
  - vibeSpatial maintainers
tags:
  - runtime
  - memory
  - rmm
  - oom
---

# Tiered GPU Memory Pool (RMM)

## Context

vibeSpatial's overlay and spatial-join pipelines allocate 80-120 device
arrays per operation at 1M geometry scale.  CuPy's built-in `MemoryPool`
uses power-of-2 binning, which wastes up to 50% of VRAM on large
coordinate arrays.  More critically, there is no recovery path when an
allocation fails — the process crashes with `OutOfMemoryError`.

RAPIDS RMM (RAPIDS Memory Manager) provides composable memory resource
adaptors: coalescing pools, failure callbacks, managed memory, and
statistics tracking.  It integrates with CuPy via
`rmm.allocators.cupy.rmm_cupy_allocator`.

## Decision

Adopt a three-tier memory management architecture, with RMM as an optional
dependency and CuPy's pool as the fallback:

| Tier | Activation | Allocator Stack |
|------|------------|-----------------|
| A (default) | RMM installed | `PoolMemoryResource` → `CudaMemoryResource` |
| B (safe) | `VIBESPATIAL_GPU_OOM_SAFETY=1` | `FailureCallbackResourceAdaptor` → Pool → Cuda |
| C (oversubscription) | `VIBESPATIAL_GPU_MANAGED_MEMORY=1` | Bare `ManagedMemoryResource` |
| Fallback | RMM not installed | CuPy `MemoryPool` |

### Design choices

- **Tier A uses `initial_pool_size=0`** to avoid starving other processes on
  shared GPUs.  The pool grows on demand.
- **Tier B's OOM callback** calls `gc.collect()` and retries up to 3 times
  per allocation attempt, with a time-based reset (>1 s gap) so independent
  OOM events each get the full retry budget.
- **Tier C uses bare `ManagedMemoryResource`** without pool wrapping.  Pooling
  managed memory adds overhead without benefit because CUDA already handles
  demand paging.  `PrefetchResourceAdaptor` was evaluated and rejected:
  vibeSpatial's SoA coordinate layout means each segment access touches 4-8
  separate array pages, making prefetch ineffective under oversubscription.
- **Deferred initialization**: RMM resources require a CUDA context, but
  `CudaDriverRuntime.__init__` runs at module import time before any CUDA
  call.  RMM setup is deferred to `_ensure_context()`.  If it fails, the
  runtime falls back to the CuPy pool with a warning.
- **CCCL one-shot `cudaMallocAsync` bypasses RMM**: CCCL primitives without
  `make_*` precompilation allocate temp storage via the CUDA driver's
  internal async pool.  This is a known limitation affecting only cold-start
  paths.

## Consequences

- **Positive**: ~5-15% peak VRAM reduction from coalescing (Tier A); OOM
  resilience without overhead (Tier B); ability to process datasets exceeding
  VRAM (Tier C, with documented 2-10× slowdown).
- **Negative**: New optional dependency (rmm).  `memory_pool_stats()` returns
  different key sets per backend.  `free_pool_memory()` becomes a no-op for
  RMM backends (the pool retains its arena for reuse).
- **Risk**: The SoA coordinate layout is worst-case for managed memory page
  faults.  The face-walk kernel's pointer-chasing through `next_edge_ids`
  can degrade 50-100× under Tier C oversubscription.  This is documented
  but not mitigated — Tier C is opt-in for users who accept the tradeoff.
