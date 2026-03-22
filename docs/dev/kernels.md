# CUDA Kernels

## Kernel inventory

vibeSpatial's GPU kernels live in `src/vibespatial/kernels/`. Each kernel
is an NVRTC source compiled at runtime.

| Kernel | Module | Operations |
|--------|--------|------------|
| Bounds | `kernels/core.py` | Geometry bounds, total bounds, Morton keys |
| Predicates | `kernels/predicates.py` | Point-in-polygon, point-within-bounds |
| Segment intersection | `segment_primitives.py` | Extraction (count-scatter), candidate generation (sort-sweep + scatter), classification (Shewchuk adaptive) |
| Overlay | `overlay_gpu.py` | Half-edge graph, split events, face extraction |
| Stroke | `stroke_kernels.py` | Point buffer, offset curve |
| Make valid | `make_valid_gpu.py` | GPU polygon repair |

## Adding a new kernel

Use the scaffold script:

```bash
uv run python scripts/generate_kernel_scaffold.py my_kernel_name
```

This creates:
- `src/vibespatial/kernels/my_kernel_name.py` -- kernel source + compilation
- `tests/test_my_kernel_name.py` -- Shapely oracle test fixture
- Manifest entry for precompilation

## Precision compliance

Every kernel must support dual-precision dispatch per ADR-0002. The
`PrecisionPlan` selects fp32 or fp64 based on device capability. Kernels
receive a `precision_mode` parameter and must use the appropriate types.

## Precompilation

NVRTC kernels can be precompiled to reduce first-launch latency:

```bash
VIBESPATIAL_PRECOMPILE=1 uv run python -c "import vibespatial"
```
