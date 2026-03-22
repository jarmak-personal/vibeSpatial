# vibeSpatial

```{raw} html
<div class="cp-hero">
  <div class="cp-hero-content">
    <h1 class="cp-hero-title" data-glitch="vibeSpatial">vibe<span class="accent">Spatial</span></h1>
    <p class="cp-hero-subtitle">
      GPU-accelerated spatial analytics for Python. Drop-in GeoDataFrame with
      CUDA kernels for predicates, overlay, dissolve, buffer, and I/O.
    </p>
    <div class="cp-hero-actions">
      <a class="cp-btn cp-btn--primary" href="user/index.html">User Guide &rarr;</a>
      <a class="cp-btn cp-btn--secondary" href="dev/index.html">Developer Guide &rarr;</a>
    </div>
  </div>
</div>
```

```{raw} html
<div class="cp-features">
  <div class="cp-card cp-reveal">
    <h3>GPU-First Design</h3>
    <p>Every operation is designed for GPU dispatch first. NVRTC kernels for predicates, overlay, dissolve, buffer, and clip. CPU fallback is explicit and observable.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>GeoPandas Compatible</h3>
    <p>GeoDataFrame and GeoSeries with the same API you know. Import vibespatial, use the same methods. 98% of the upstream GeoPandas test suite passes natively.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>Device-Resident Geometry</h3>
    <p>OwnedGeometryArray keeps geometry on the GPU. No Shapely round-trips for GPU consumers. Lazy materialization when host access is needed.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>Precision Control</h3>
    <p>Dual-precision dispatch (fp32/fp64) via PrecisionPlan. Every kernel respects ADR-0002. Consumer GPUs get fp32 fast paths; datacenter GPUs get full fp64.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>Adaptive Runtime</h3>
    <p>Automatic GPU/CPU dispatch based on input size, device capability, and kernel support. Observable dispatch events for profiling and debugging.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>Native I/O</h3>
    <p>GeoParquet, GeoArrow, Shapefile, and GeoJSON with GPU-accelerated WKB decode. Zero-copy Arrow paths when available.</p>
  </div>
</div>
```

## Quick Example

```python
import vibespatial
from shapely.geometry import Point, box

# Build a GeoDataFrame
gdf = vibespatial.GeoDataFrame(
    {"name": ["a", "b", "c"]},
    geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
)

# GPU-accelerated operations
buffered = gdf.geometry.buffer(0.5)
result = vibespatial.sjoin(gdf, other_gdf, predicate="intersects")
dissolved = gdf.dissolve(by="name")
```

```{raw} html
<div class="cp-links" style="text-align: center; margin: 2rem 0;">
  <a href="https://github.com/jarmak-personal/vibeSpatial" style="margin: 0 1rem;">GitHub</a> &middot;
  <a href="https://pypi.org/project/vibespatial/" style="margin: 0 1rem;">PyPI</a> &middot;
  <a href="https://github.com/jarmak-personal/vibeSpatial/issues" style="margin: 0 1rem;">Issues</a>
</div>
```

```{toctree}
:hidden:
:maxdepth: 2

user/index
dev/index
architecture/index
decisions/adr-index
testing/index
apidocs/index
```
