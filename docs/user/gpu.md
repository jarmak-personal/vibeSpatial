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
| **Binary predicates** | contains, intersects, within, covers, crosses, overlaps, touches, disjoint, dwithin |
| **Constructive** | buffer, offset_curve, make_valid |
| **Overlay** | intersection, union, difference, symmetric_difference, identity |
| **Aggregation** | dissolve (grouped union) |
| **Clip** | clip_by_rect |
| **Spatial join** | sjoin, sjoin_nearest |
| **Measurement** | area, length, distance, dwithin |
| **I/O** | GeoJSON byte-classify parser, GeoParquet WKB decode, GeoArrow native codec |

## Precision modes

vibeSpatial supports dual-precision dispatch via `PrecisionPlan`:

- **fp64** -- Full double precision. Default on datacenter GPUs (A100, H100).
- **fp32** -- Single precision. Faster on consumer GPUs (RTX 4090, etc.).

The precision planner automatically selects based on device capability and
kernel class. See ADR-0002 for the design rationale.

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
from vibespatial.cccl_precompile import precompile_all
result = precompile_all()   # blocks until all specs/units are compiled
print(result)               # shows compiled/failed counts and timings
```

**Environment variables:**

| Variable | Default | Effect |
|----------|---------|--------|
| `VIBESPATIAL_CCCL_CACHE` | enabled | Set `0` to disable the CCCL CUBIN disk cache |
| `VIBESPATIAL_NVRTC_CACHE` | enabled | Set `0` to disable the NVRTC CUBIN disk cache |
| `VIBESPATIAL_CCCL_CACHE_DIR` | `~/.cache/vibespatial/cccl` | Override CCCL cache directory |
| `VIBESPATIAL_NVRTC_CACHE_DIR` | `~/.cache/vibespatial/nvrtc` | Override NVRTC cache directory |
| `VIBESPATIAL_PRECOMPILE` | enabled | Set `0` to disable background pre-compilation entirely |

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
