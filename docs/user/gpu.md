# GPU Acceleration

## How dispatch works

vibeSpatial uses an adaptive runtime that decides GPU vs CPU dispatch per
operation based on:

- **Input size** -- small inputs may be faster on CPU due to launch overhead
- **Device capability** -- fp64 throughput varies across GPU tiers
- **Kernel support** -- not all operations have GPU kernels yet

The dispatch decision is recorded as an observable event.

## GPU-accelerated operations

| Category | Operations |
|----------|------------|
| **Binary predicates** | contains, contains_properly, intersects, within, covers, covered_by, crosses, overlaps, touches, disjoint, dwithin |
| **Constructive** | buffer, offset_curve, centroid, make_valid, clip_by_rect |
| **Overlay** | intersection, union, difference, symmetric_difference, identity |
| **Aggregation** | dissolve (grouped union) |
| **Spatial join** | sjoin, sjoin_nearest |
| **Measurement** | area, length, distance, hausdorff_distance, frechet_distance, dwithin |
| **Spatial indexing** | flat R-tree, segment MBR extraction and sort-and-sweep pair generation |
| **I/O** | GeoJSON byte-classify parser (Point, LineString, Polygon), GeoParquet WKB decode/encode, GeoArrow native codec, kvikio parallel file-to-device transfer |

## Precision modes

vibeSpatial supports dual-precision dispatch via `PrecisionPlan`:

- **fp64** -- Full double precision. Default on datacenter GPUs (A100, H100).
- **fp32** -- Single precision. Faster on consumer GPUs (RTX 4090, etc.).

The precision planner automatically selects based on device capability and
kernel class. See ADR-0002 for the design rationale.

## Automatic query rewrites

vibeSpatial includes a provenance-based rewrite system that recognizes
common spatial patterns and substitutes cheaper equivalents automatically.
This is enabled by default and transparent -- rewrite events are logged
alongside dispatch events.

**Built-in rewrite rules:**

| Pattern | Rewrite | Condition |
|---------|---------|-----------|
| `buffer(r).intersects(Y)` | `dwithin(Y, r)` | Point geometries, positive radius, round cap/join |
| `sjoin(buffer(r, X), Y, "intersects")` | `sjoin(X, Y, "dwithin", r)` | Point geometries, positive radius |
| `buffer(a).buffer(b)` | `buffer(a + b)` | Positive radii, same style, point geometries |
| `buffer(0)` | identity (no-op) | Zero distance |
| `simplify(0)` | identity (no-op) | Zero tolerance |

To inspect which rewrites fired:

```python
from vibespatial.runtime.provenance import get_rewrite_events

for event in get_rewrite_events():
    print(f"{event.rule_name}: {event.original_operation} -> {event.rewritten_operation}")
```

Set `VIBESPATIAL_PROVENANCE_REWRITES=0` to disable rewrites (e.g. for
debugging or benchmarking the un-optimized path).

## Execution modes

```python
from vibespatial import ExecutionMode, set_execution_mode

set_execution_mode(ExecutionMode.AUTO)  # Default: GPU when available
set_execution_mode(ExecutionMode.GPU)   # Force GPU (raises if unavailable)
set_execution_mode(ExecutionMode.CPU)   # Force CPU
```

Or set via environment variable: `VIBESPATIAL_EXECUTION_MODE=gpu`.

## Strict native mode

In CI or testing you may want to guarantee that **no** operation silently
falls back to CPU. Strict native mode raises `StrictNativeFallbackError`
on any CPU fallback:

```bash
VIBESPATIAL_STRICT_NATIVE=1 uv run pytest tests/
```

Or programmatically:

```python
import os
os.environ["VIBESPATIAL_STRICT_NATIVE"] = "1"
```

## Deterministic mode

GPU floating-point operations are not bitwise reproducible by default (due
to non-deterministic reduction ordering). Enable deterministic mode when
you need reproducible results at the cost of some performance:

```bash
VIBESPATIAL_DETERMINISM=1 python my_script.py
```

## Sister projects

vibeSpatial is part of a GPU-first geospatial suite. Each project shares
core infrastructure and can exchange data on-device without host
round-trips.

| Project | Role | Integration |
|---------|------|-------------|
| {external+vibeproj:doc}`vibeProj <index>` | GPU coordinate projection | {external+vibeproj:doc}`transform_buffers() <user/vibespatial>` reads/writes `OwnedGeometryArray` coordinate buffers directly on device |
| {external+vibespatial-raster:doc}`vibeSpatial-Raster <index>` | GPU raster processing | Shares `residency`, `runtime`, and `cuda_runtime` core modules; {external+vibespatial-raster:doc}`OwnedRasterArray <user/vibespatial>` mirrors `OwnedGeometryArray` |

## Kernel caching and pre-compilation

vibeSpatial JIT-compiles GPU kernels on first use.  Compiled CUBINs are
cached on disk under `~/.cache/vibespatial/` so the cost is paid once per
install, not once per process.

**Automatic behaviour (no user action needed):** when a GPU operation runs,
its kernels compile in background threads and the result is persisted to
disk.  The next process start loads the cached CUBIN in ~1-100 ms instead
of recompiling (~1,300 ms per CCCL spec, ~100 ms per NVRTC unit).

**Pre-populate all caches** after install or in CI warm-up:

```python
from vibespatial.cuda.cccl_precompile import precompile_all
result = precompile_all()   # blocks until all specs/units are compiled
print(result)               # shows compiled/failed counts and timings
```

See [GPU Kernel Caching](../architecture/gpu-kernel-caching.md) for the
full architecture, cache key format, and ctypes replay design.

## Profiling

Use the dispatch event log to identify GPU vs CPU routing:

```python
import vibespatial

vibespatial.clear_dispatch_events()

# ... run operations ...

for event in vibespatial.get_dispatch_events():
    print(f"{event.surface}: {event.selected.value}")
```

Any CPU fallback is also recorded:

```python
for event in vibespatial.get_fallback_events():
    print(f"Fallback: {event.surface} -- {event.reason}")
```

For deeper tracing, use the execution trace context manager:

```python
from vibespatial import execution_trace

with execution_trace() as trace:
    gdf.geometry.buffer(1.0)

for warning in trace.warnings:
    print(warning)
```

Set `VIBESPATIAL_TRACE_WARNINGS=1` to emit trace warnings automatically.

Write a structured event log to a file for post-hoc analysis:

```bash
VIBESPATIAL_EVENT_LOG=events.jsonl python my_script.py
```

## Environment variables

All `VIBESPATIAL_*` environment variables in one place:

| Variable | Default | Effect |
|----------|---------|--------|
| `VIBESPATIAL_EXECUTION_MODE` | `auto` | Force execution mode (`auto`, `gpu`, `cpu`) |
| `VIBESPATIAL_STRICT_NATIVE` | disabled | Set `1` to error on any CPU fallback |
| `VIBESPATIAL_DETERMINISM` | disabled | Set `1` for deterministic (reproducible) results |
| `VIBESPATIAL_PROVENANCE_REWRITES` | enabled | Set `0` to disable automatic query rewrites |
| `VIBESPATIAL_TRACE_WARNINGS` | disabled | Set `1` to emit warnings from execution traces |
| `VIBESPATIAL_EVENT_LOG` | unset | Path to write structured dispatch/rewrite event log |
| `VIBESPATIAL_GPU_POOL_LIMIT` | unset | Limit GPU memory pool size (bytes) |
| `VIBESPATIAL_GPU_OOM_SAFETY` | disabled | Set `1` to enable Tier B: RMM pool with GC retry on OOM. Zero overhead on the happy path; ~5-50ms recovery on OOM. |
| `VIBESPATIAL_GPU_MANAGED_MEMORY` | disabled | Set `1` to enable Tier C: CUDA managed memory. Datasets exceeding VRAM will run to completion (slowly) instead of crashing. Expect 2-10x slowdown when oversubscribed. |
| `VIBESPATIAL_CCCL_CACHE` | enabled | Set `0` to disable the CCCL CUBIN disk cache |
| `VIBESPATIAL_NVRTC_CACHE` | enabled | Set `0` to disable the NVRTC CUBIN disk cache |
| `VIBESPATIAL_CCCL_CACHE_DIR` | `~/.cache/vibespatial/cccl` | Override CCCL cache directory |
| `VIBESPATIAL_NVRTC_CACHE_DIR` | `~/.cache/vibespatial/nvrtc` | Override NVRTC cache directory |
| `VIBESPATIAL_PRECOMPILE` | enabled | Set `0` to disable background kernel pre-compilation |
