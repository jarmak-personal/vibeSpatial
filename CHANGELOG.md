# Changelog

## 0.1.0 (2026-03-16)

Initial release.

- GPU-accelerated GeoDataFrame and GeoSeries with GeoPandas-compatible API
- CUDA kernels for binary predicates, buffer, offset curve, clip, overlay, dissolve, make valid
- Adaptive GPU/CPU runtime with observable dispatch events
- Dual-precision dispatch (fp32/fp64) via PrecisionPlan
- Device-resident geometry storage (OwnedGeometryArray)
- GeoParquet, GeoArrow, Shapefile, GeoJSON I/O with GPU-accelerated WKB decode
- Spatial indexing and spatial joins (sjoin, sjoin_nearest)
- 98% upstream GeoPandas test suite pass rate under strict-native mode
- Sphinx docs site with NEON GRID theme
- PyPI publishing via GitHub Actions with OIDC trusted publishing
