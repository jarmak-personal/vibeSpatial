# Runtime Model

<!-- DOC_HEADER:START
Scope: GPU-first runtime rules, fallback policy, and execution invariants.
Read If: You are changing runtime selection, GPU execution, fallback visibility, or kernels.
STOP IF: Your task is docs-only or limited to vendored test maintenance.
Source Of Truth: Runtime architecture policy for GPU-first execution.
Body Budget: 191/200 lines
Document: docs/architecture/runtime.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-9 | Intent |
| 10-20 | Request Signals |
| 21-27 | Open First |
| 28-31 | Verify |
| 32-37 | Risks |
| 38-70 | Core Rules |
| 71-80 | Fallback |
| 81-94 | Session Execution Mode Override |
| 95-109 | Provenance Rewrite Override |
| 110-132 | Device-Native Result Boundary (ADR-0042) |
| 133-185 | Memory Pool Tiers (ADR-0040) |
| 186-191 | Compatibility |
DOC_HEADER:END -->

`vibeSpatial` is GPU-first, not GPU-optional.

## Intent

Define runtime-selection rules, fallback visibility requirements, and the first
files to inspect when execution behavior changes.

## Request Signals

- runtime
- gpu
- cuda
- fallback
- execution mode
- kernel
- cccl
- diagnostics

## Open First

- docs/architecture/runtime.md
- src/vibespatial/runtime/_runtime.py
- src/geopandas/__init__.py
- src/vibespatial/api/__init__.py

## Verify

- `uv run pytest`

## Risks

- Silent CPU fallback hides unsupported GPU behavior.
- Runtime-selection changes can desync the GeoPandas shim from the runtime layer.
- Kernel-oriented changes can look correct locally while breaking the upstream contract.

## Core Rules

- Design APIs around bulk device execution and parallel kernels.
- Prefer `cuda-python` for runtime control and kernel launch plumbing.
- Prefer CCCL for reusable data-parallel building blocks.
- Runtime availability means a real CUDA device is present, not just that the
  Python package imports successfully.
- CPU execution exists to preserve correctness and debuggability, not to define
  the architecture.
- Canonical geometry storage stays `fp64`; metric precision statistics and
  centering reduce logical row-indirected bounds before compute dispatch.
- Null and empty geometries are distinct states and must stay distinct through
  buffer layout and kernel outputs.
- Predicate and constructive kernels must declare a robustness guarantee, not
  just a precision mode.
- Deterministic reproducibility is opt-in; default mode stays performance-first.
- `auto` dispatch must use per-kernel crossover thresholds, not one global size gate.
- Public API workflows should make one CPU/GPU dispatch decision at the
  boundary, not re-plan execution family at each internal step.
- `auto` crossover thresholds apply at promotion time while inputs are host-resident;
  once a workload is already device-resident, `auto` stays on GPU and only
  re-plans among GPU variants.
- Generic runtime probing must not claim GPU execution for `auto` by itself; the
  actual switch to GPU happens only inside kernel-specific dispatch planning.
- Adaptive planning may re-evaluate at chunk boundaries, but not mid-kernel.
- Repo-owned `GeoSeries` and `GeoDataFrame` methods must carry explicit
  dispatch registrations.
- Repo-owned kernel modules must register at least one kernel variant before
  they are allowed to land.
- Phase 9 bounds execution is the first live cuda-python kernel and keeps
  family-specialized CPU and GPU variants side by side so dispatch can stay
  performance-driven instead of one-size-fits-all.

## Fallback

- `auto` mode may fall back to CPU when GPU execution is unavailable.
- Explicit `gpu` mode must fail loudly if the required GPU path is unsupported.
- Fallback events should be observable. Silent host execution is not acceptable.
- New fallback surfaces should be paired with tests or diagnostics.
- Non-user host-to-device and device-to-host transfers must remain visible.
- Device-to-host transfers belong only in explicit materialization surfaces such
  as `to_pandas`, `to_numpy`, `values`, and `__repr__`.

## Session Execution Mode Override

The session-wide execution mode follows the `determinism.py` pattern:

- `VIBESPATIAL_EXECUTION_MODE` env var (`auto`, `cpu`, `gpu`).
- `set_execution_mode()` programmatic override (takes priority over env var).
- `get_requested_mode()` reads: explicit override > env var > `auto` default.
- CPU mode causes early returns in IO (`_try_gpu_read_file`, WKB decode/encode),
  `DeviceGeometryArray` operations (`to_crs`, `dwithin`, `_binary_predicate`,
  `clip_by_rect`), binary predicates, and `geoseries_from_owned`.
- Setting the mode invalidates the adaptive runtime snapshot cache.
- All entry points call `get_requested_mode()` to determine dispatch; internal
  GPU-only helpers are safe because their callers gate on mode first.

## Provenance Rewrite Override

The provenance rewrite system (ADR-0039) follows the same pattern:

- `VIBESPATIAL_PROVENANCE_REWRITES` env var (default: enabled; `0`/`false`/
  `no`/`off` to disable).
- `set_provenance_rewrites(bool | None)` programmatic override (takes priority
  over env var; `None` clears override back to default).
- `provenance_rewrites_enabled()` reads: explicit override > env var > `True`.
- Gated at five sites: `attempt_provenance_rewrite()` in `provenance.py`
  (covers R1 and all consumption-time binary predicate rules), the R5/R6
  branches in `geometry_array.py:buffer()`, the R7 branch in
  `geometry_array.py:simplify()`, and the R2 branch in
  `sjoin.py:_geom_predicate_query()`.

## Device-Native Result Boundary (ADR-0042)

GPU-selected workflows should remain device-native until an explicit
compatibility or materialization surface is requested.

- Low-level spatial query kernels may still return typed integer index arrays.
  `SpatialJoinIndices` and related dtype assertions remain useful for that
  narrow contract.
- The architectural target for overlay, clip, dissolve, and other
  constructive/relational workflows is broader: device-resident geometry,
  provenance, and relation data should stay off host until an explicit export
  boundary such as `to_geopandas()`, `to_pandas()`, or `to_shapely()`.
- `sjoin._frame_join` and similar pandas assembly seams remain transitional
  compatibility layers, not the desired steady-state execution model.
- Overlay's current attribute assembly and keep-geometry-type handling remain
  migration surfaces. New work should move semantics handling toward typed
  device-side classification instead of host inspection.
- I/O paths should keep Arrow or other columnar tables alive as long as
  possible and defer host conversion to explicit construction/materialization
  points.
- Once `auto` has selected GPU for a workflow, internal steps must not silently
  pivot back to host execution just because a host-shaped helper exists.

## Memory Pool Tiers (ADR-0040)

Device memory allocation uses a tiered strategy built on RAPIDS RMM when
available, with CuPy's built-in `MemoryPool` as the fallback.

| Tier | Env Var | Allocator Stack | Default? |
|------|---------|-----------------|----------|
| A | `VIBESPATIAL_GPU_OOM_SAFETY=0` | `PoolMemoryResource` → `CudaMemoryResource` | No |
| B | *(none)* | `FailureCallbackResourceAdaptor` → Pool → Cuda | Yes (when RMM installed) |
| C | `VIBESPATIAL_GPU_MANAGED_MEMORY=1` | `ManagedMemoryResource` (bare) | No |
| Fallback | *(RMM not installed)* | CuPy `MemoryPool` | Yes (without RMM) |

- **Tiers A/B** provide a coalescing pool with ~5-15% peak VRAM reduction over
  CuPy's power-of-2 binning. Tier B adds a GC-retry callback on OOM (bounded to
  3 retries per event) with zero overhead on the happy path.
- Tiers A/B start from an explicit 1 MiB seed (or a smaller configured ceiling).
  Zero and sub-granularity RMM seeds are interpreted as unspecified and can
  eagerly reserve half the ceiling, so they do not implement an empty pool.
- **Tier C** uses CUDA managed memory for datasets exceeding VRAM. Performance
  degrades 2-10× under oversubscription due to PCIe page migration; the SoA
  coordinate layout amplifies page faults.
- **Deferred initialization**: no device allocator is installed merely because
  CuPy or RMM imports. Pool selection runs inside `_ensure_context()` after the
  primary context is retained. If installed RMM setup fails, initialization
  fails closed rather than creating a split CuPy/libcudf allocation domain;
  the CuPy pool is used only when RMM is unavailable.
- `VIBESPATIAL_GPU_POOL_LIMIT` maps to `maximum_pool_size` (Tiers A/B) and is
  ignored for Tier C (managed memory uses OS overcommit semantics). An explicit
  value of `0` requests an unlimited pool; if the reserve-derived default leaves
  less than one 256-byte allocation unit, initialization fails instead of
  silently treating that zero ceiling as unlimited.
- `_memory_backend` discriminator values: `"cupy"`, `"rmm-pool"`, `"rmm-safe"`,
  `"rmm-managed"`, `"none"` (before context init).
- Explicit frees and every cached or one-shot CCCL launch use one completion-
  retirement service. Explicit streams coalesce short windows; PTDS submissions
  retain a lifetime-unique thread token and record their event in the submitting
  thread. Arrays, iterator owners, and scratch tokens release when the event
  completes without requiring a later vibeSpatial call or context sync.
  Explicit synchronization releases only retirements claimed before its
  boundary, so concurrent later submissions remain owned. Invocation failures
  transfer operands to the same service before re-raising. Event-record or
  event-query failures require a proven event, stream, or context boundary;
  batches retry with bounded backoff while available boundaries fail.
- Cached CCCL callables lease scratch per ordering domain. Their mutable
  iterator binding is serialized only through launch, same-stream scratch
  follows CUDA ordering, and explicit or per-thread concurrent streams receive
  independent scratch so they can overlap without corrupting callable state.
- A pylibcudf stream wrapper is attached to its CuPy stream object rather than
  retained in a process-global handle cache. Transient wrappers therefore die
  with their stream. Persistent values publish completion-scoped producer
  readiness; cross-stream pylibcudf consumers enqueue an event dependency while
  same-stream pipelines remain event-free.

## Compatibility

- GeoPandas behavior is measured with vendored upstream tests.
- Upstream parity matters more than mirroring GeoPandas internals.
- Rebuild abstractions only when the test contract or performance data demands
  them.
