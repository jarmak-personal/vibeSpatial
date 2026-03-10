---
id: ADR-0037
status: deferred
date: 2026-03-16
deciders:
  - claude-opus
  - vibeSpatial maintainers
tags:
  - kernel-strategy
  - gpu-primitives
  - constructive
  - voronoi
  - delaunay
  - computational-geometry
---

# ADR-0037: GPU Voronoi Diagram Kernel

## Status Note

This ADR is **deferred** and not targeted for 0.1.0.  The current
Shapely-delegated `voronoi_polygons` is sufficient for initial release
scope.  This ADR is preserved as a design reference if GPU-accelerated
Voronoi becomes a priority in the future.

## Context

`GeoSeries.voronoi_polygons()` computes Voronoi diagrams from point
geometries.  The current implementation delegates to
`shapely.voronoi_polygons()`, which runs single-threaded on the host
and requires full Shapely materialization of device-resident geometry.

Voronoi is a **CONSTRUCTIVE** operation per ADR-0002 (produces new
geometry) and requires **Tier 1 NVRTC** per ADR-0033 (the inner loop
is geometry-specific computational geometry, not reducible to CCCL
primitives or CuPy element-wise ops).

### Why Voronoi is harder than area/length

| | Area/Length | Voronoi |
|---|---|---|
| Output type | Scalar (float64 per row) | Geometry (variable-vertex polygon per row) |
| Precision class | METRIC (fp32+Kahan viable) | CONSTRUCTIVE (fp64 required) |
| Kernel count | 1 per family | 5+ in pipeline |
| CCCL usage | None | `radix_sort`, `exclusive_sum`, `segmented_sort` |
| Memory pattern | Fixed-size output | Count-scatter (variable output) |
| Algorithmic complexity | O(n) per thread | O(n log n) total, multi-pass |
| Core algorithm | Arithmetic (shoelace, sqrt) | Computational geometry (Delaunay) |

### GeoPandas API surface

```python
GeoSeries.voronoi_polygons(
    tolerance: float = 0.0,
    extend_to: Geometry | None = None,
    only_edges: bool = False,
) -> GeoSeries
```

- **tolerance**: snap input points within this distance before computing
- **extend_to**: clip Voronoi cells to this geometry (default: input envelope)
- **only_edges**: return edges (LineStrings) instead of cells (Polygons)

### Input constraints

- Input is Point or MultiPoint geometry (for non-point types, Shapely
  uses the vertices of the geometry as Voronoi sites)
- Output is one Polygon/MultiPolygon per input point (cell assignment)
  or a single GeometryCollection of all cells

## Decision

### Algorithm: GPU Delaunay triangulation then dual extraction

Fortune's sweep line (the classic O(n log n) Voronoi algorithm) is
inherently sequential and unsuitable for GPU execution.  The
GPU-native approach is:

1. **Delaunay triangulation** via parallel incremental insertion
2. **Dual extraction** to obtain Voronoi cells from the Delaunay mesh

This is the proven approach from the research literature (gDel2D by
Cao et al., GPU-DT by Qi et al.).

### Pipeline architecture

```
Phase 1: Spatial sort           (Tier 3a CCCL — radix_sort)
Phase 2: Delaunay triangulation (Tier 1 NVRTC — multi-kernel)
Phase 3: Circumcenter compute   (Tier 1 NVRTC — one thread per triangle)
Phase 4: Cell assembly           (Tier 3a CCCL + Tier 1 NVRTC — count-scatter)
Phase 5: Clip to envelope        (Tier 1 NVRTC — reuse clip_rect if rectangular)
```

#### Phase 1: Spatial sort

Spatially sort input points for locality-preserving insertion order.
Hilbert curve index computation is a Tier 1 NVRTC kernel (bit
interleaving of quantized coordinates).  Sort by Hilbert key uses
CCCL `radix_sort` (Tier 3a, benchmarked 1.4-3.1x faster than CuPy).

#### Phase 2: Delaunay triangulation

The core computational challenge.  Parallel incremental insertion
with star splaying (gDel2D approach):

1. Insert points in spatially-sorted batches
2. Each thread attempts to insert one point into the existing mesh
3. Conflicts (overlapping cavities) are resolved via star splaying
4. Flip pass corrects non-Delaunay edges

This requires multiple kernel launches per batch (insert, splay,
flip).  The triangle adjacency data structure lives entirely on
device as a flat array of (v0, v1, v2, neighbor0, neighbor1,
neighbor2) per triangle.

**Degeneracy handling** is critical:
- Four cocircular points: requires exact or robust in-circle predicate
- Collinear points: degenerate triangulation
- Duplicate points: must be detected and merged (tolerance parameter)
- Shewchuk-style adaptive exact arithmetic may be needed for the
  orientation and in-circle predicates to ensure robustness

#### Phase 3: Circumcenter computation

One thread per triangle.  Each triangle's circumcenter is a Voronoi
vertex.  Straightforward fp64 computation:

```
cx = (|B-A|^2 * (C-A) - |C-A|^2 * (B-A)) / (2 * cross(B-A, C-A))
```

No precision concerns at fp64.  Output is a flat (num_triangles, 2)
array of Voronoi vertex coordinates.

#### Phase 4: Cell assembly (count-scatter pattern)

Build one Voronoi cell (polygon) per input point:

1. **Count**: for each input point, count incident triangles.  This
   is a scatter-add of 1 per triangle vertex into a per-point counter.
   Uses `exclusive_sum` (CCCL) to build offsets.

2. **Gather**: for each input point, gather circumcenters of incident
   triangles.  These are the vertices of the Voronoi cell, but in
   arbitrary order.

3. **Angle sort**: sort gathered circumcenters by angle around the
   input point to form a valid polygon ring.  Uses CCCL
   `segmented_sort` (sort within offset-delimited segments).

4. **Build output geometry**: assemble x, y, ring_offsets,
   geometry_offsets into a Polygon-family `FamilyGeometryBuffer`.
   Convex hull boundary points produce unbounded cells that must be
   clipped (Phase 5).

#### Phase 5: Clip to envelope

Unbounded Voronoi cells at the convex hull boundary extend to
infinity.  Clip all cells to the bounding envelope:

- If `extend_to` is None: clip to the input point set's bounding box
  (potentially expanded)
- If `extend_to` is a rectangle: reuse existing `clip_rect` kernel
- If `extend_to` is an arbitrary geometry: requires full
  polygon-polygon clipping (reuse overlay infrastructure)

### Precision compliance (ADR-0002)

CONSTRUCTIVE class: fp64 on all devices.  Wire `PrecisionPlan`
through for observability but do not template kernels on `compute_t`.

The Delaunay predicates (orientation, in-circle) are the
precision-critical path.  These require either:
- Native fp64 (sufficient for most practical input), or
- Shewchuk-style adaptive exact arithmetic (for degenerate/near-
  degenerate configurations)

### Dispatch

```python
def voronoi_polygons_owned(
    owned: OwnedGeometryArray,
    *,
    tolerance: float = 0.0,
    extend_to: BaseGeometry | None = None,
    only_edges: bool = False,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = "auto",
) -> OwnedGeometryArray:
```

Adaptive dispatch via `plan_dispatch_selection()` with kernel name
`"voronoi_polygons"` and class `KernelClass.CONSTRUCTIVE`.  Crossover
threshold likely 2,000-5,000 rows due to multi-kernel pipeline
overhead (higher than the 500-row threshold for single-kernel
area/length).

### CPU fallback

NumPy/SciPy `Voronoi` from `scipy.spatial` (no Shapely dependency in
the CPU fallback path, matching the pattern established by
measurement_kernels.py).  Alternatively, delegate to Shapely's
`voronoi_polygons` for the CPU path since it handles all edge cases.

### Zero-copy device-resident path

Same pattern as measurement_kernels.py: when `owned.device_state` is
populated, read point coordinates directly from
`DeviceFamilyGeometryBuffer.x/.y` device pointers.  All intermediate
GPU buffers (triangle mesh, circumcenters, cell vertices) stay on
device.  Only the final `OwnedGeometryArray` output is constructed
with the result coordinates.  For the vibeFrame path, the output
could remain device-resident if the caller doesn't need host access.

## Consequences

### Positive

- Voronoi on large point sets (100K+) would be GPU-accelerated with
  no host round-trips during computation.
- Establishes the pattern for GPU computational geometry beyond
  simple per-row kernels (multi-pass pipelines with intermediate
  device-only data structures).
- CCCL `segmented_sort` gets a real use case (angle-sorting
  circumcenters per cell), validating the ADR-0033 tier system for
  complex pipelines.

### Negative

- Significant implementation effort: GPU Delaunay is research-grade
  code.  The gDel2D algorithm has ~2,000 lines of CUDA in the
  reference implementation.
- Degeneracy handling in GPU Delaunay is the hardest unsolved
  engineering problem.  Production-quality exact arithmetic on GPU is
  non-trivial (Shewchuk predicates require error-free transformations
  that are expensive in fp64).
- The multi-kernel pipeline makes profiling and debugging
  substantially harder than single-kernel ops like area/length.

## Alternatives Considered

### A: JFA (Jump Flooding Algorithm) on rasterized grid

Massively parallel and simple to implement.  Rasterize the point set
onto a grid, then iteratively flood to assign each pixel to its
nearest point.  Extract cell boundaries from the discrete grid.

**Rejected because:**
- Approximate — resolution-dependent accuracy
- Does not produce exact polygon geometry (produces raster cells)
- Memory-intensive for large domains (grid allocation)
- Not compatible with GeoPandas' expectation of exact polygon output

### B: Incremental CPU Delaunay then GPU dual extraction

Compute Delaunay on CPU (scipy.spatial.Delaunay), transfer triangle
mesh to GPU, compute circumcenters and cell assembly on GPU.

**Viable as a phased approach:**
- Phase 1: CPU Delaunay + GPU cell assembly (partial GPU acceleration)
- Phase 2: Replace CPU Delaunay with GPU Delaunay (full GPU)

This reduces risk by deferring the hardest part (GPU Delaunay) while
still getting GPU acceleration for the cell assembly and clipping
stages.

### C: CGAL or other library delegation

Use an existing computational geometry library for the Voronoi
computation.  CGAL has robust Voronoi with exact arithmetic.

**Rejected because:**
- CGAL is C++ and not GPU-accelerated
- Would require a host round-trip (defeating zero-copy goal)
- Adds a heavy external dependency
- Doesn't align with the project's build-from-cuda-python philosophy

## Implementation Phasing (recommended)

If this ADR is promoted from deferred:

1. **Phase 0**: CPU fallback via SciPy/Shapely (immediate, trivial)
2. **Phase 1**: CPU Delaunay + GPU cell assembly (medium effort,
   partial GPU benefit, validates the output pipeline)
3. **Phase 2**: GPU Delaunay replacement (high effort, full GPU,
   requires robust predicate infrastructure)

Phase 1 alone provides meaningful GPU acceleration for the
cell-assembly and clipping stages while keeping the well-tested
SciPy Delaunay for the hard computational geometry.

## References

- ADR-0033: GPU Primitive Dispatch Rules (tier classification)
- ADR-0002: Dual Precision Dispatch (CONSTRUCTIVE fp64 policy)
- ADR-0013: Explicit CPU Fallback Events
- Cao, T.T. et al., "A GPU accelerated algorithm for 3D Delaunay
  triangulation," I3D 2014.
- Qi, M. et al., "gpuDT: A GPU-based parallel Delaunay
  triangulation algorithm," CGF 2013.
- Shewchuk, J.R., "Adaptive Precision Floating-Point Arithmetic
  and Fast Robust Geometric Predicates," Discrete & Computational
  Geometry 18(3), 1997.
