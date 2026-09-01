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
| A | `VIBESPATIAL_GPU_OOM_SAFETY=0` | `LimitingResourceAdaptor` → `CudaAsyncMemoryResource` |
| B (default) | RMM installed and OOM safety not disabled | `FailureCallbackResourceAdaptor` → Limiting → CudaAsync |
| C (oversubscription) | `VIBESPATIAL_GPU_MANAGED_MEMORY=1` | Bare `ManagedMemoryResource` |
| Fallback | RMM not installed | CuPy `MemoryPool` |

### Design choices

- **Tiers A/B use an explicit 1 MiB initial async pool** (or the configured ceiling
  when smaller). Supported RMM releases interpret zero and sub-granularity
  seeds as unspecified and eagerly reserve half of `maximum_pool_size`; the
  explicit seed keeps idle orchestration processes small while retaining
  on-demand growth. A limiting adaptor preserves the larger of 1 GiB or 10%
  of device memory outside the allocator's live envelope;
  `VIBESPATIAL_GPU_POOL_LIMIT=0` explicitly opts into an unlimited pool. The
  same ceiling is the async release threshold, so cached memory above it is
  returned at a CUDA synchronization boundary.
- **The CUDA async resource replaces the original coalescing pool.** Repeated
  SF100 WKB/GeoParquet reads left only about 81 MiB live but stranded about
  19.9 GiB in non-contiguous pool blocks; a following 1.74 GiB coordinate
  allocation failed below the live-memory ceiling. CUDA's stream-ordered pool
  can reuse the freed pages without requiring one contiguous arena block while
  the limiting adaptor retains the fail-closed capacity contract.
- **Tier B's OOM callback** calls `gc.collect()` and retries up to 3 times
  per allocation attempt, with a time-based reset (>1 s gap) so independent
  OOM events each get the full retry budget.
- **Tier C uses bare `ManagedMemoryResource`** without pool wrapping.  Pooling
  managed memory adds overhead without benefit because CUDA already handles
  demand paging.  `PrefetchResourceAdaptor` was evaluated and rejected:
  vibeSpatial's SoA coordinate layout means each segment access touches 4-8
  separate array pages, making prefetch ineffective under oversubscription.
- **Deferred, fail-closed initialization**: RMM resources require a CUDA context, but
  `CudaDriverRuntime.__init__` runs at module import time before any CUDA
  call. RMM setup is deferred to `_ensure_context()`. If RMM is installed but
  setup fails, initialization raises instead of creating a split CuPy/libcudf
  allocation domain. CuPy's pool is used only when RMM is unavailable.
- **CCCL one-shot `cudaMallocAsync` bypasses RMM**: CCCL primitives without
  `make_*` precompilation allocate temp storage via the CUDA driver's
  internal async pool.  This is a known limitation affecting only cold-start
  paths.

## Consequences

- **Positive**: ~5-15% peak VRAM reduction from coalescing (Tiers A/B); OOM
  resilience without overhead (Tier B); ability to process datasets exceeding
  VRAM (Tier C, with documented 2-10× slowdown).
- **Negative**: New optional dependency (rmm).  `memory_pool_stats()` returns
  different key sets per backend. RMM 26.02 does not expose the private
  `cudaMemPool_t`, so async-pool `reserved_bytes` is the measurable live lower
  bound while the live ceiling is reported separately as
  `allocation_limit_bytes`.
- **Risk**: The SoA coordinate layout is worst-case for managed memory page
  faults.  The face-walk kernel's pointer-chasing through `next_edge_ids`
  can degrade 50-100× under Tier C oversubscription.  This is documented
  but not mitigated — Tier C is opt-in for users who accept the tradeoff.
