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
