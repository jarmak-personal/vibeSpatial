"""GPU-accelerated exterior ring extraction.

For Polygon geometries, extracts ring 0 coordinates and produces LineString output.
For MultiPolygon, falls back to CPU (each polygon's exterior produces a MultiLineString).
For non-polygon types, returns None per Shapely convention.

ADR-0033: Tier 2 CuPy vectorized offset computation and coordinate gather.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

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
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan

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

    All offset computation and coordinate gathering is done on device
    via CuPy vectorized operations (Tier 2 per ADR-0033).  No D2H/H2D
    transfers except one unavoidable scalar read (total_coords) for
    output allocation sizing.
    """
    d_state = owned._ensure_device_state()

    if GeometryFamily.POLYGON not in d_state.families:
        return from_shapely_geometries([None] * owned.row_count)

    d_poly = d_state.families[GeometryFamily.POLYGON]

    # For Polygon, ring 0 of each geometry contains the exterior.
    # Geometry offsets point into ring_offsets.
    # Ring 0 for geometry g starts at ring_offsets[geom_offsets[g]]
    # and ends at ring_offsets[geom_offsets[g] + 1].

    d_geom_offsets = d_poly.geometry_offsets
    d_ring_offsets = d_poly.ring_offsets

    poly_count = d_geom_offsets.shape[0] - 1

    # Gather ring-0 start/end for each geometry (device fancy indexing)
    d_ring0_start_idx = d_geom_offsets[:-1]
    d_ring_starts = d_ring_offsets[d_ring0_start_idx]
    d_ring_ends = d_ring_offsets[d_ring0_start_idx + 1]

    # Compute per-geometry exterior ring lengths and output offsets (all on device)
    d_ring_lengths = d_ring_ends - d_ring_starts
    d_out_geom_offsets = cp.zeros(poly_count + 1, dtype=cp.int32)
    cp.cumsum(d_ring_lengths, out=d_out_geom_offsets[1:])

    # Single scalar D2H to get total output size (unavoidable for allocation)
    total_coords = int(d_out_geom_offsets[-1])
    if total_coords == 0:
        return from_shapely_geometries([None] * owned.row_count)

    # Build gather index map on device via searchsorted:
    # Map each output position to its owning geometry, then compute the
    # source coordinate index.  This is robust to zero-length segments
    # (degenerate rings) since no output position maps to an empty segment.
    # TODO: Replace cp.arange + cp.searchsorted with CCCL lower_bound +
    # CountingIterator to avoid materializing the full d_positions array
    # (ADR-0033 Tier 3c opportunity).
    d_positions = cp.arange(total_coords, dtype=cp.int32)
    d_geom_ids = cp.searchsorted(d_out_geom_offsets[1:], d_positions, side="right")
    d_idx = d_ring_starts[d_geom_ids] + (d_positions - d_out_geom_offsets[d_geom_ids])

    # Gather coordinates on device (no host round-trip)
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

    d_empty_mask = cp.zeros(poly_count, dtype=cp.bool_)

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
        except Exception as exc:
            record_fallback_event(
                surface="geopandas.array.exterior",
                reason="GPU exterior kernel raised, fell back to CPU",
                detail=str(exc),
                d2h_transfer=True,
            )
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
