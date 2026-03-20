# Architecture

## Layer diagram

```
vibespatial.GeoDataFrame / GeoSeries   (user-facing API)
        |
vibespatial.api.*                       (extracted GeoPandas-compatible surface)
        |
evaluate_geopandas_*()                  (GPU dispatch hooks)
        |
vibespatial.runtime.dispatch / runtime          (adaptive GPU/CPU routing)
        |
vibespatial.kernels.*                   (NVRTC CUDA kernels)
        |
OwnedGeometryArray                      (device-resident geometry storage)
```

## Key design decisions

- **GPU-first**: every operation is designed for GPU dispatch. CPU is an
  explicit fallback, never silent.
- **Observable dispatch**: every GPU/CPU routing decision is recorded as
  a `DispatchEvent`. Fallbacks produce `FallbackEvent`.
- **Device-resident storage**: `OwnedGeometryArray` keeps geometry on the
  GPU. `DeviceGeometryArray` is the pandas ExtensionArray wrapper that
  materializes Shapely objects lazily.
- **Dual precision**: `PrecisionPlan` selects fp32 or fp64 per kernel
  based on device capability (ADR-0002).
- **No vendoring**: vibeSpatial owns its full API surface. The
  GeoPandas-compatible classes live in `vibespatial.api.*`, extracted
  from upstream with import rewriting.

## ADRs

Architecture Decision Records are in `docs/decisions/`. Key ones:

| ADR | Topic |
|-----|-------|
| ADR-0002 | Dual-precision dispatch |
| ADR-0005 | Device-resident geometry |
| ADR-0007 | Adaptive runtime |
| ADR-0020 | Public API dispatch boundary |
| ADR-0032 | End-to-end profile gate |
| ADR-0033 | Kernel tier system |
