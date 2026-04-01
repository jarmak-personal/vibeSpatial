---
id: ADR-0035
status: deferred
date: 2026-03-14
deciders:
  - claude-opus
  - vibeSpatial maintainers
tags:
  - distance
  - geodesic
  - vincenty
  - crs
  - kernel-strategy
  - gpu-primitives
---

# ADR-0035: Geodesic Distance Kernel and CRS-Aware Dispatch

## Status Note

This ADR is **deferred** and not targeted for implementation. The
current warn-and-compute-Euclidean behavior matches upstream GeoPandas
and is sufficient for the project's current scope. Users who need
geodesic accuracy should reproject to a projected CRS before running
distance operations. This ADR is preserved as a design reference if
geodesic GPU distance becomes a priority in the future.

## Context

All GPU distance kernels in vibeSpatial compute **planar Euclidean
distance** — `sqrt(dx*dx + dy*dy)` in coordinate space. This is
correct for projected CRS (UTM, State Plane, etc.) but produces
meaningless degree-based values for geographic CRS like EPSG:4326
(WGS 84), where one degree of longitude is ~111 km at the equator but
~0 km at the poles.

The current safeguard is `check_geographic_crs()` on
`DeviceGeometryArray` and the vendored `GeometryArray`, which emits a
`UserWarning` but does not block execution or switch distance
formulas. This matches upstream GeoPandas behavior: warn, then compute
planar distance anyway. The GPU query engine in `spatial_query.py` is
entirely CRS-unaware — no checks, no warnings.

### What exists today

| Layer | CRS awareness | Action on geographic CRS |
|---|---|---|
| `DeviceGeometryArray` | `self._crs` (pyproj CRS) | Warns on `area`, `length`, `centroid`, `buffer`, `distance` |
| Vendored `GeometryArray` | `self.crs` (pyproj CRS) | Warns on `area`, `length`, `centroid`, `buffer`, `distance`, `dwithin`, Hausdorff, Fréchet, `interpolate` |
| `OwnedGeometryArray` | **None** | No CRS metadata at all |
| `spatial_query.py` | **None** | No checks on `dwithin`, `nearest`, or any distance refinement |
| NVRTC distance kernels | **None** | Pure Euclidean always |
| `sjoin_nearest` (vendored) | Warns via `check_geographic_crs` | Warns, then uses Euclidean |

This means a user who calls `sjoin_nearest` on WGS 84 data gets a
warning from the vendored layer but then the GPU distance refinement
silently computes Euclidean distances in degree-space — which can
produce incorrect orderings (nearer in degrees ≠ nearer on the
ellipsoid) and incorrect `max_distance` filtering.

### Why this matters

1. **Incorrect nearest-neighbor results.** At mid-latitudes, 1° of
   longitude ≈ 79 km but 1° of latitude ≈ 111 km. Euclidean distance
   in degree-space distorts E-W vs N-S distances by up to 40%. This
   can change which geometry is "nearest."

2. **Incorrect dwithin filtering.** A `max_distance` of 1000 (meters)
   applied in degree-space is nonsensical. Users who forget to
   reproject get silently wrong results with no GPU-level guard.

3. **GeoPandas compatibility.** GeoPandas also only warns, so matching
   upstream behavior is technically correct. But vibeSpatial's mission
   is to be *better*, not just compatible.

### Vincenty vs Haversine vs reproject-first

Three approaches exist for geodesic distance:

| Formula | Accuracy | Cost | GPU-friendly? |
|---|---|---|---|
| **Haversine** | ~0.3% error (spherical assumption) | Cheap: 2 `sin`, 2 `cos`, 1 `atan2`, 1 `sqrt` | Yes — branchless, no iteration |
| **Vincenty** | ~0.01 mm on WGS 84 ellipsoid | Moderate: iterative (typically 3–6 iterations, max ~20) | Yes — convergent loop, all fp64 math available on GPU |
| **Karney (GeographicLib)** | sub-nanometer | Expensive: complex series expansion | Possible but high register pressure |
| **Reproject-first** | Depends on chosen projection | One-time cost + planar distance | Avoids kernel changes; shifts work to CRS transform |

### Relationship to ADR-0033

Per ADR-0033's decision tree:

> *Is the inner loop geometry-specific?* → **Yes** → Tier 1 (custom
> NVRTC kernel)

Geodesic distance is irreducibly geometry-specific: the formula
operates on coordinate pairs with trigonometric functions on an
ellipsoid model. There is no CCCL or CuPy equivalent. A Vincenty or
Haversine kernel is squarely **Tier 1: Custom NVRTC**.

The iterative nature of Vincenty (variable loop count per pair) raises
a warp divergence concern, but the convergence is fast and bounded
(max ~20 iterations, typically 3–6). This is acceptable warp
divergence — it's far less than the ring/part traversal loops already
present in existing Tier 1 polygon kernels.

The supporting pipeline stages (candidate generation, compaction,
nearest-reduce) remain at their current tiers per ADR-0033: CCCL for
scans/compaction, CuPy for element-wise, etc. Only the distance
*formula* changes.

### Precision implications

Per `docs/architecture/precision.md`, distance is a **metric** kernel
class. Vincenty requires fp64 — the iterative convergence and
trigonometric functions lose too much precision in fp32. This aligns
with the existing policy: metric kernels on consumer GPUs use staged
fp32 with compensation, but Vincenty should be **fp64-only** because:

- The WGS 84 ellipsoid parameters (a = 6378137.0, f = 1/298.257223563)
  require fp64 to represent without loss.
- The iterative loop's convergence criterion (~1e-12 radians) is below
  fp32 epsilon.
- Haversine could run in fp32 if the accuracy tradeoff (0.3% error) is
  acceptable for a coarse-distance fast path.

## Decision

### Option A: Reproject-first (recommended default, no new kernel)

**Policy:** For distance-dependent operations (`dwithin`, `nearest`,
`sjoin_nearest`, `.distance()`), if the CRS is geographic:

1. Auto-estimate a suitable projected CRS via `estimate_utm_crs()`
   (already available on `GeoSeries`).
2. Reproject both geometry columns to the estimated projection.
3. Compute planar Euclidean distance using existing GPU kernels.
4. Emit a diagnostic event recording the auto-reprojection.

**Pros:**
- Zero new NVRTC kernel code.
- Reuses the battle-tested Euclidean distance pipeline.
- Consistent with "reproject before operating" best practice.
- Works for all geometry types (points, lines, polygons) without
  per-type geodesic kernels.

**Cons:**
- Reprojection is currently CPU-only (`pyproj`). For large datasets
  this becomes a CPU bottleneck before the GPU distance kernel runs.
- UTM zone estimation can be wrong for datasets spanning multiple
  zones.
- Adds a silent auto-reproject step that changes coordinates — some
  users may not expect this.
- Memory: requires a temporary reprojected copy of both geometry
  arrays.

### Option B: Haversine GPU kernel (fast geodesic for points)

**Policy:** Add a Haversine great-circle distance NVRTC kernel for
point-to-point distance. Use it when:

- Both geometry columns contain only points.
- CRS is geographic (EPSG:4326 or similar).
- The user has not already reprojected.

Fall back to Euclidean for non-point geometries (lines, polygons)
with the existing warning.

**Kernel (Tier 1 per ADR-0033):**

```cuda
extern "C" __global__ void haversine_distance(
    const double* lat1, const double* lon1,
    const double* lat2, const double* lon2,
    double* out_distances,
    int n,
    double earth_radius  // 6371008.8 for mean radius
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double d_lat = (lat2[i] - lat1[i]) * M_PI / 180.0;
  const double d_lon = (lon2[i] - lon1[i]) * M_PI / 180.0;
  const double la1   = lat1[i] * M_PI / 180.0;
  const double la2   = lat2[i] * M_PI / 180.0;
  const double a     = sin(d_lat / 2.0) * sin(d_lat / 2.0)
                      + cos(la1) * cos(la2)
                      * sin(d_lon / 2.0) * sin(d_lon / 2.0);
  out_distances[i]   = 2.0 * earth_radius * asin(sqrt(a));
}
```

**Pros:**
- No host round-trip. Geodesic distance stays on device.
- Cheap: ~10 FP64 ops per pair. No iteration.
- Good enough for most spatial join / nearest-neighbor use cases.
- fp32-feasible for coarse-filter stages (0.3% error acceptable).

**Cons:**
- Points only. No point-to-line or point-to-polygon geodesic distance
  without significant additional work (project each segment to a great
  circle arc).
- 0.3% error near the poles and for very long distances.
- Need to handle x/y vs lon/lat axis ordering per CRS metadata.

### Option C: Vincenty GPU kernel (high-accuracy geodesic)

**Policy:** Add an iterative Vincenty inverse-formula NVRTC kernel.
Use it as the distance function when CRS is geographic and the user
requests meter-accurate results (or as the default geodesic path).

**Kernel (Tier 1 per ADR-0033):**

```cuda
extern "C" __global__ void vincenty_distance(
    const double* lat1, const double* lon1,
    const double* lat2, const double* lon2,
    double* out_distances,
    int n,
    double a,     // semi-major axis (6378137.0 for WGS 84)
    double f      // flattening (1/298.257223563 for WGS 84)
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double b = a * (1.0 - f);
  const double U1 = atan((1.0 - f) * tan(lat1[i] * M_PI / 180.0));
  const double U2 = atan((1.0 - f) * tan(lat2[i] * M_PI / 180.0));
  const double L  = (lon2[i] - lon1[i]) * M_PI / 180.0;
  const double sinU1 = sin(U1), cosU1 = cos(U1);
  const double sinU2 = sin(U2), cosU2 = cos(U2);

  double lambda = L, prev_lambda;
  double sin_sigma, cos_sigma, sigma, sin_alpha, cos2_alpha, cos_2sigma_m;

  for (int iter = 0; iter < 20; ++iter) {
    const double sinL = sin(lambda), cosL = cos(lambda);
    sin_sigma = sqrt(
        (cosU2 * sinL) * (cosU2 * sinL) +
        (cosU1 * sinU2 - sinU1 * cosU2 * cosL) *
        (cosU1 * sinU2 - sinU1 * cosU2 * cosL));
    if (sin_sigma < 1e-30) { out_distances[i] = 0.0; return; }
    cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cosL;
    sigma = atan2(sin_sigma, cos_sigma);
    sin_alpha = cosU1 * cosU2 * sinL / sin_sigma;
    cos2_alpha = 1.0 - sin_alpha * sin_alpha;
    cos_2sigma_m = (cos2_alpha > 1e-30)
        ? cos_sigma - 2.0 * sinU1 * sinU2 / cos2_alpha
        : 0.0;
    const double C = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha));
    prev_lambda = lambda;
    lambda = L + (1.0 - C) * f * sin_alpha *
        (sigma + C * sin_sigma *
         (cos_2sigma_m + C * cos_sigma *
          (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));
    if (fabs(lambda - prev_lambda) < 1e-12) break;
  }

  const double u_sq = cos2_alpha * (a * a - b * b) / (b * b);
  const double A_coeff = 1.0 + u_sq / 16384.0 *
      (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
  const double B_coeff = u_sq / 1024.0 *
      (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
  const double delta_sigma = B_coeff * sin_sigma *
      (cos_2sigma_m + B_coeff / 4.0 *
       (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
        - B_coeff / 6.0 * cos_2sigma_m *
          (-3.0 + 4.0 * sin_sigma * sin_sigma) *
          (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));
  out_distances[i] = b * A_coeff * (sigma - delta_sigma);
}
```

**Pros:**
- Sub-millimeter accuracy on WGS 84.
- No host round-trip. All on device.
- Convergence is fast (3–6 iterations typical). Warp divergence is
  bounded and modest — less than polygon ring traversal loops.
- fp64-only is acceptable: metric kernel class, and existing distance
  kernels already use fp64 storage.

**Cons:**
- Points only (same as Haversine). Extending to segment-to-segment
  geodesic distance is a research problem, not an engineering task.
- ~6× more FP64 ops per pair than Haversine (trig + iteration).
- Antipodal points can cause non-convergence (need fallback to
  Haversine or special-case handling).
- Register pressure: ~20+ fp64 registers per thread. Higher occupancy
  impact than Haversine.
- fp64 throughput is 1/32 of fp32 on consumer GPUs. For point-heavy
  workloads this may still be memory-bound, but for small batches the
  compute cost is real.

### Option D: Tiered geodesic (recommended — combine B + C)

**Policy:** Implement both Haversine (Option B) and Vincenty (Option C)
as Tier 1 NVRTC kernels, selectable by a distance-mode parameter:

| Mode | Formula | Accuracy | Use case |
|---|---|---|---|
| `euclidean` (default) | Existing kernels | Exact in coordinate space | Projected CRS |
| `haversine` | Spherical great-circle | ~0.3% | Fast nearest-neighbor screening, coarse dwithin |
| `vincenty` | Ellipsoidal iterative | ~0.01 mm | Accurate distance reporting, tight dwithin |
| `auto` | CRS-aware selection | Best available | Picks `euclidean` for projected, `vincenty` for geographic |

When `auto` mode detects a geographic CRS:
- Use Haversine for coarse-filter / candidate-generation stages where
  0.3% error doesn't change the result set.
- Use Vincenty for final distance refinement and `distance_col`
  output.
- Emit a diagnostic event showing which formula was used.

### Recommendation

**Option D** (tiered geodesic) is the recommended path. It aligns with
the project's performance-first philosophy:

- **Haversine for throughput-sensitive stages** (candidate MBR
  expansion, coarse nearest-neighbor filtering). Cheap enough that the
  GPU kernel isn't the bottleneck.
- **Vincenty for accuracy-sensitive stages** (final distance
  computation, distance column output, tight `max_distance`
  filtering). The iteration cost is acceptable because the final
  refinement stage operates on a much smaller candidate set.
- **CRS flows into the kernel layer** by threading CRS metadata
  through `OwnedGeometryArray` (or as a dispatch-time parameter), not
  by modifying the kernel signatures to accept CRS objects.

## Implementation Plan

### Phase 1: CRS metadata threading (prerequisite)

1. Add an optional `crs` field to `OwnedGeometryArray` (or pass CRS
   as a parameter at dispatch call sites in `spatial_query.py`).
2. Thread CRS from `DeviceGeometryArray._crs` into spatial query
   dispatch functions.
3. No kernel changes yet — this only enables the dispatch layer to
   *know* whether the CRS is geographic.

### Phase 2: Haversine point-to-point kernel

1. Add `haversine_distance.py` (or extend `point_distance.py`) with
   the Haversine NVRTC kernel.
2. Expose via the existing `compute_point_distance_gpu` interface with
   a `distance_mode` parameter.
3. Integration test: point-to-point distance on EPSG:4326 data
   compared against `pyproj.Geod.inv()` as the oracle.
4. Wire into `spatial_query.py` candidate generation for geographic
   CRS: expand query bounds in meters (via Haversine) rather than
   degrees.

### Phase 3: Vincenty point-to-point kernel

1. Add Vincenty kernel alongside Haversine in the same module.
2. Handle antipodal edge case (fall back to supplementary formula or
   Haversine for near-antipodal pairs).
3. Oracle test: compare against `pyproj.Geod.inv()` for a corpus of
   point pairs including equatorial, polar, antipodal, and coincident
   cases.
4. Wire into `spatial_query.py` final refinement for `dwithin` and
   `nearest` when CRS is geographic.

### Phase 4: Auto-mode dispatch integration

1. In `spatial_query.py`, detect geographic CRS at the top of
   `query_spatial_index` / `nearest_spatial_index`.
2. Use Haversine for coarse stages, Vincenty for refine stages.
3. Convert `max_distance` from meters to appropriate units for each
   stage (meters for geodesic kernels, coordinate units for bounds
   expansion heuristics).
4. Emit runtime diagnostic events showing which distance formula was
   selected and why.

### Phase 5: Non-point geodesic distance (future, deferred)

Geodesic distance for line-to-point or polygon-to-point on the
ellipsoid is substantially harder:
- "Distance to a geodesic arc" has no closed-form solution.
- Would require iterative closest-point-on-arc computation per
  segment.
- The performance shape is unclear — likely dominated by the iterative
  computation, not memory bandwidth.

**Decision:** Defer non-point geodesic distance. For non-point
geometries on geographic CRS, the existing behavior (warn + Euclidean)
or the Option A approach (auto-reproject to UTM, then Euclidean)
should be offered as the fallback. This is an explicit, non-silent
fallback per ADR-0013.

## Risks

- **Warp divergence in Vincenty.** Iteration counts vary per pair
  (3–20). Benchmarking is required to confirm the GPU kernel isn't
  slower than CPU `pyproj.Geod.inv()` for small batches (< 10K pairs).
  Mitigant: profile at ADR-0033 benchmark scales (10K, 100K, 1M).

- **Axis ordering ambiguity.** WGS 84 (EPSG:4326) has
  latitude-first axis order in the formal definition, but most spatial
  software stores coordinates as (longitude, latitude) = (x, y).
  The kernel must use the correct convention — read axis ordering from
  the CRS object or enforce (x=lon, y=lat) as a contract.

- **Antipodal non-convergence.** Vincenty fails for exactly antipodal
  points. The kernel must detect this (sin_sigma ≈ 0 after max
  iterations) and fall back to a supplementary formula.

- **fp64 throughput on consumer GPUs.** Vincenty is fp64-only. On
  RTX-class GPUs (1/32 fp64), the kernel may be compute-bound rather
  than memory-bound. If benchmarks show this, Haversine (which is
  fp32-feasible) becomes the default even for final refinement, with
  Vincenty available as an explicit opt-in.

- **CRS metadata plumbing scope creep.** Adding CRS to
  `OwnedGeometryArray` could trigger wider refactoring. Mitigant: pass
  CRS as a dispatch-time parameter in `spatial_query.py` rather than
  embedding it in the buffer schema.

- **MBR expansion in geographic coordinates.** The current MBR-based
  candidate generation expands bounds in coordinate units. For
  geographic CRS, expanding by `max_distance` meters requires a
  degree-to-meter conversion that varies with latitude. Haversine or a
  simpler approximation (meters / 111320) can handle this, but it adds
  complexity to the candidate generation stage.

## Verification

```bash
# Phase 1: CRS threading
uv run pytest tests/test_spatial_query.py -k "crs" -q

# Phase 2: Haversine kernel
uv run pytest tests/test_geodesic_distance.py -q
uv run vsbench run gpu-predicates --scale 100k  # replace with a geodesic-specific rail when landed

# Phase 3: Vincenty kernel
uv run pytest tests/test_geodesic_distance.py -k "vincenty" -q

# Phase 4: Auto-mode integration
env VIBESPATIAL_STRICT_NATIVE=1 uv run pytest tests/upstream/geopandas/tests/test_sindex.py -q
uv run pytest tests/upstream/geopandas/tools/tests/test_sjoin.py -k "nearest" -q

# End-to-end profile (mandatory per AGENTS.md)
uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline
```

## References

- ADR-0033: GPU Primitive Dispatch Rules (tier classification)
- ADR-0002: Dual Precision Dispatch (fp64 requirement for metric kernels)
- ADR-0013: Explicit CPU Fallback Events (non-silent fallback for non-point geodesic)
- `docs/architecture/precision.md`: metric kernel class policy
- `docs/implementation-order.md` Phase 6c: CRS policy (o17.6.3)
- GeoPandas `check_geographic_crs` pattern
- T. Vincenty, "Direct and Inverse Solutions of Geodesics on the Ellipsoid with
  Application of Nested Equations," Survey Review 23(176), 1975.
- C.F.F. Karney, "Algorithms for geodesics," J. Geodesy 87(1), 2013.
