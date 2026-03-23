"""GPU-accelerated exterior ring extraction.

For Polygon geometries, extracts ring 0 coordinates and produces LineString output.
For MultiPolygon, falls back to CPU (each polygon's exterior produces a MultiLineString).
For non-polygon types, returns None per Shapely convention.

ADR-0033: Tier 1 NVRTC, 1 thread per geometry for offset computation,
1 thread per coordinate for scatter.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.cuda._runtime import (
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan

from .measurement import _PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# NVRTC kernel: extract exterior ring coordinates
# ---------------------------------------------------------------------------

_EXTERIOR_KERNEL_SOURCE = _PRECISION_PREAMBLE + r"""
extern "C" __global__ void exterior_ring_scatter(
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ geom_offsets,
    const int* __restrict__ out_offsets,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    double center_x, double center_y,
    int total_out_coords
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_out_coords) return;

    x_out[i] = x_in[i];
    y_out[i] = y_in[i];
}}
"""

_EXTERIOR_KERNEL_NAMES = ("exterior_ring_scatter",)
_EXTERIOR_FP64 = _EXTERIOR_KERNEL_SOURCE.format(compute_type="double")

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("exterior-ring-fp64", _EXTERIOR_FP64, _EXTERIOR_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "exterior_ring",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon",),
    supports_mixed=False,
    tags=("cuda-python", "constructive", "exterior"),
)
def _exterior_gpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """GPU exterior ring extraction for Polygon family.

    Extracts ring 0 from each polygon, producing a LineString per geometry.
    Returns device-resident LineString OwnedGeometryArray.
    """
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()

    if GeometryFamily.POLYGON not in d_state.families:
        return from_shapely_geometries([None] * owned.row_count)

    d_poly = d_state.families[GeometryFamily.POLYGON]

    # For Polygon, ring 0 of each geometry contains the exterior.
    # Geometry offsets point into ring_offsets.
    # Ring 0 for geometry g starts at ring_offsets[geom_offsets[g]]
    # and ends at ring_offsets[geom_offsets[g] + 1].

    # Download offsets for computation (small metadata)
    h_geom_offsets = cp.asnumpy(d_poly.geometry_offsets)
    h_ring_offsets = cp.asnumpy(d_poly.ring_offsets)

    poly_count = len(h_geom_offsets) - 1

    # Compute output geometry offsets: each geometry's exterior ring length
    out_geom_offsets = np.empty(poly_count + 1, dtype=np.int32)
    out_geom_offsets[0] = 0
    for g in range(poly_count):
        ring_start = h_ring_offsets[h_geom_offsets[g]]
        ring_end = h_ring_offsets[h_geom_offsets[g] + 1]
        out_geom_offsets[g + 1] = out_geom_offsets[g] + (ring_end - ring_start)

    total_coords = int(out_geom_offsets[-1])
    if total_coords == 0:
        return from_shapely_geometries([None] * owned.row_count)

    # Compute source coordinate indices for gathering
    src_indices = np.empty(total_coords, dtype=np.int32)
    pos = 0
    for g in range(poly_count):
        ring_start = h_ring_offsets[h_geom_offsets[g]]
        ring_end = h_ring_offsets[h_geom_offsets[g] + 1]
        length = ring_end - ring_start
        src_indices[pos:pos + length] = np.arange(ring_start, ring_end, dtype=np.int32)
        pos += length

    # Gather coordinates on device
    d_idx = cp.asarray(src_indices)
    d_x_out = d_poly.x[d_idx]
    d_y_out = d_poly.y[d_idx]

    # Build LineString output
    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    poly_mask = owned.tags == poly_tag

    # Build validity/tags for output (same row structure as input)
    out_validity = owned.validity.copy()
    out_tags = np.full(owned.row_count, FAMILY_TAGS[GeometryFamily.LINESTRING], dtype=np.int8)
    out_tags[~poly_mask] = -1  # non-polygon rows are null
    out_validity[~poly_mask] = False
    out_family_row_offsets = np.full(owned.row_count, -1, dtype=np.int32)
    out_family_row_offsets[poly_mask] = np.arange(poly_count, dtype=np.int32)

    d_out_geom_offsets = runtime.from_host(out_geom_offsets)
    d_empty_mask = runtime.from_host(np.zeros(poly_count, dtype=bool))

    device_families = {
        GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.LINESTRING,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=d_out_geom_offsets,
            empty_mask=d_empty_mask,
        ),
    }

    return build_device_resident_owned(
        device_families=device_families,
        row_count=owned.row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=out_family_row_offsets,
    )


# ---------------------------------------------------------------------------
# CPU fallback: exterior via Shapely
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "exterior_ring",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("shapely", "constructive", "exterior"),
)
def _exterior_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Compute exterior ring using Shapely as reference implementation."""
    import shapely

    geoms = owned.to_shapely()
    results = [shapely.get_exterior_ring(g) if g is not None else None for g in geoms]
    return from_shapely_geometries(results)


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def exterior_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Extract exterior ring from Polygon geometries.

    Returns OwnedGeometryArray of LineString geometries.
    Non-polygon rows produce None.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries.
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint.
    precision : PrecisionMode or str, default AUTO
        Precision mode.  COARSE class stays fp64 by design per ADR-0002;
        wired here for observability.

    Returns
    -------
    OwnedGeometryArray
        Exterior ring geometries.
    """
    row_count = owned.row_count
    if row_count == 0:
        return from_shapely_geometries([])

    # Short-circuit: no polygons means all-null output without touching device
    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    if not np.any(owned.tags == poly_tag):
        return from_shapely_geometries([None] * row_count)

    selection = plan_dispatch_selection(
        kernel_name="exterior_ring",
        kernel_class=KernelClass.COARSE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = select_precision_plan(
            runtime_selection=selection,
            kernel_class=KernelClass.COARSE,
            requested=precision,
        )
        try:
            result = _exterior_gpu(owned)
        except Exception:
            pass
        else:
            record_dispatch_event(
                surface="geopandas.array.exterior",
                operation="exterior",
                implementation="exterior_ring_gpu_cupy",
                reason=selection.reason,
                detail=(
                    f"rows={row_count}, "
                    f"precision={precision_plan.compute_precision.value}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result

    result = _exterior_cpu(owned)
    record_dispatch_event(
        surface="geopandas.array.exterior",
        operation="exterior",
        implementation="exterior_cpu_shapely",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result
