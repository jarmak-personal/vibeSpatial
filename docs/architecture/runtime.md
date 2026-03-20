# Runtime Model

<!-- DOC_HEADER:START
Scope: GPU-first runtime rules, fallback policy, and execution invariants.
Read If: You are changing runtime selection, GPU execution, fallback visibility, or kernels.
STOP IF: Your task is docs-only or limited to vendored test maintenance.
Source Of Truth: Runtime architecture policy for GPU-first execution.
Body Budget: 127/200 lines
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
| 38-65 | Core Rules |
| 66-75 | Fallback |
| 76-89 | Session Execution Mode Override |
| 90-102 | Provenance Rewrite Override |
| 103-121 | Index-Array Boundary Model (ADR-0036) |
| 122-127 | Compatibility |
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
- Canonical geometry storage should stay `fp64`; compute precision may dispatch
  separately from storage precision.
- Null and empty geometries are distinct states and must stay distinct through
  buffer layout and kernel outputs.
- Predicate and constructive kernels must declare a robustness guarantee, not
  just a precision mode.
- Deterministic reproducibility is opt-in; default mode stays performance-first.
- `auto` dispatch must use per-kernel crossover thresholds, not one global size gate.
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
- Gated at three sites: `attempt_provenance_rewrite()` in `provenance.py`
  (covers R1 and all consumption-time rules), and the R5/R6 branches in
  `geometry_array.py:buffer()`.

## Index-Array Boundary Model (ADR-0036)

Spatial kernels produce only index arrays (`np.ndarray` with integer dtype).
Attribute assembly is always pandas on host.  GPU VRAM is reserved for geometry.

- The boundary is enforced by `SpatialJoinIndices` (frozen dataclass in
  `spatial_query_types.py`) and `__debug__`-gated dtype assertions at kernel
  return points in `spatial_query_utils.py` and `spatial_nearest.py`.
- `sjoin._frame_join` is structured into three delineated blocks: geometry
  extraction, attribute reindexing (geometry-free), and geometry reassembly.
  The outer-join geometry path is isolated in `_reassemble_outer_geometry`.
- Overlay's `_overlay_intersection` delegates attribute merging to
  `_assemble_intersection_attributes`, which receives only index arrays and
  attribute-only DataFrames.
- I/O paths (`io_geoparquet.py`) keep Arrow tables through geometry decode and
  defer `.to_pandas()` to the GeoDataFrame construction boundary.
- Contract tests in `tests/test_index_array_boundary.py` validate the boundary
  invariants across spatial query, sjoin, overlay, dissolve, and clip.

## Compatibility

- GeoPandas behavior is measured with vendored upstream tests.
- Upstream parity matters more than mirroring GeoPandas internals.
- Rebuild abstractions only when the test contract or performance data demands
  them.
