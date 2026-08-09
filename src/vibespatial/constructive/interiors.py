"""GPU-accelerated interior ring extraction.

For Polygon geometries, extracts interior rings (holes) by reading
ring_offsets and geometry_offsets on device.  Produces a MultiLineString
OGA where each row's parts are the interior rings of that polygon.

Non-Polygon families produce empty MultiLineString (zero parts).
Null rows propagate as null.

ADR-0033: Tier 2 — pure CuPy offset arithmetic, no custom NVRTC kernel.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.interiors_cpu import _interiors_cpu as _interiors_cpu
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    _device_gather_xy_offset_slices,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import estimate_physical_work_from_owned
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency

# ---------------------------------------------------------------------------
# Lightweight all-null OGA builder (no Shapely, no GPU required)
# ---------------------------------------------------------------------------


def _build_all_null_oga(row_count: int) -> OwnedGeometryArray:
    """Build a host-resident OGA where every row is null.

    No Shapely objects are created and no GPU allocation is needed.
    """
    return OwnedGeometryArray(
        validity=np.zeros(row_count, dtype=bool),
        tags=np.full(row_count, -1, dtype=np.int8),
        family_row_offsets=np.full(row_count, -1, dtype=np.int32),
        families={},
        residency=Residency.HOST,
    )


# ---------------------------------------------------------------------------
# GPU implementation — pure CuPy offset arithmetic (Tier 2)
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "interior_rings",
    "gpu-cupy",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon",),
    supports_mixed=True,
    tags=("cupy", "constructive", "interiors"),
)
def _interiors_gpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """GPU interior ring extraction for Polygon family.

    Extracts rings 1..N from each polygon (skipping exterior ring 0),
    producing a MultiLineString per geometry where each part is an
    interior ring.

    Returns device-resident MultiLineString OwnedGeometryArray.
    """
    d_state = owned._ensure_device_state()

    row_count = owned.row_count
    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    d_poly_valid = (d_state.tags == poly_tag) & d_state.validity

    has_polys = GeometryFamily.POLYGON in d_state.families
    if not has_polys:
        # No polygons at all — return all-empty MultiLineString
        return _build_all_empty_multilinestring(owned)

    d_poly = d_state.families[GeometryFamily.POLYGON]

    # -------------------------------------------------------------------
    # Step 1: Access device-resident offset arrays (no D2H transfer)
    # -------------------------------------------------------------------
    d_geom_offsets = d_poly.geometry_offsets
    d_ring_offsets = d_poly.ring_offsets

    # Number of polygon family rows
    poly_count = int(d_geom_offsets.size) - 1
    if poly_count == 0:
        return _build_all_empty_multilinestring(owned, poly_count=0)

    # -------------------------------------------------------------------
    # Step 2: Compute per-polygon ring counts on device
    # ring_count[i] = geom_offsets[i+1] - geom_offsets[i]
    # interior_count[i] = max(ring_count[i] - 1, 0)
    # -------------------------------------------------------------------
    d_ring_counts = d_geom_offsets[1:] - d_geom_offsets[:-1]
    d_interior_counts = cp.maximum(d_ring_counts - 1, 0)

    # -------------------------------------------------------------------
    # Step 3: Build output geometry_offsets for MultiLineString
    # geometry_offsets[i] = cumulative sum of interior_counts[0..i-1]
    # This gives the part index range for each polygon family row.
    # -------------------------------------------------------------------
    d_out_geom_offsets = cp.zeros(poly_count + 1, dtype=cp.int32)
    cp.cumsum(d_interior_counts, out=d_out_geom_offsets[1:])

    # -------------------------------------------------------------------
    # Step 4: Select interior ring indices at source-ring capacity.
    # For each polygon i, the interior rings are at ring indices:
    #   geom_offsets[i]+1, geom_offsets[i]+2, ..., geom_offsets[i+1]-1
    # We build a flat array of these source ring indices.
    # -------------------------------------------------------------------
    ring_capacity = max(int(d_ring_offsets.size) - 1, 0)
    d_ring_lanes = cp.arange(ring_capacity, dtype=cp.int64)
    d_is_interior = d_ring_lanes < d_geom_offsets[-1]

    # Empty polygons do not own an exterior ring. Route those writes to scratch
    # lanes so every source-ring lane remains a valid device address.
    d_polygon_lanes = cp.arange(poly_count, dtype=cp.int64)
    d_exterior_destinations = cp.where(
        d_ring_counts > 0,
        d_geom_offsets[:-1].astype(cp.int64, copy=False),
        np.int64(ring_capacity) + d_polygon_lanes,
    )
    d_not_exterior = cp.ones(ring_capacity + poly_count, dtype=cp.bool_)
    d_not_exterior[d_exterior_destinations] = False
    d_is_interior &= d_not_exterior[:ring_capacity]

    from vibespatial.api._native_rowset import NativeDeviceSelection

    interior_selection = NativeDeviceSelection.from_mask(d_is_interior)
    d_interior_ring_indices = interior_selection.safe_capacity_positions()

    # -------------------------------------------------------------------
    # Step 5: Gather coordinate spans into source-coordinate capacity. The
    # returned part offsets carry the active logical prefix.
    # -------------------------------------------------------------------
    d_x_out, d_y_out, d_out_part_offsets = _device_gather_xy_offset_slices(
        d_poly.x,
        d_poly.y,
        d_ring_offsets,
        d_interior_ring_indices,
        allocation_capacity=int(d_poly.x.size),
        active_row_count=interior_selection.logical_count,
    )

    # -------------------------------------------------------------------
    # Step 6: Build output OGA metadata
    # -------------------------------------------------------------------
    mls_tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
    out_validity = d_poly_valid.copy()
    out_tags = cp.full(row_count, -1, dtype=cp.int8)
    out_tags[d_poly_valid] = mls_tag
    out_family_row_offsets = cp.full(row_count, -1, dtype=cp.int32)
    out_family_row_offsets[d_poly_valid] = d_state.family_row_offsets[d_poly_valid]

    d_empty_mask = d_interior_counts == 0

    device_families = {
        GeometryFamily.MULTILINESTRING: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTILINESTRING,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=d_out_geom_offsets,
            empty_mask=d_empty_mask,
            part_offsets=d_out_part_offsets,
        ),
    }

    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )


def _build_all_empty_multilinestring(
    owned: OwnedGeometryArray,
    *,
    poly_count: int | None = None,
) -> OwnedGeometryArray:
    """Build a MultiLineString OGA where every row is empty or null.

    Polygon rows get an empty MultiLineString (zero parts).
    Non-polygon rows and null rows get null.
    """
    d_state = owned._ensure_device_state()

    row_count = owned.row_count
    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mls_tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
    d_poly_valid = (d_state.tags == poly_tag) & d_state.validity

    out_validity = d_poly_valid.copy()
    out_tags = cp.full(row_count, -1, dtype=cp.int8)
    out_tags[d_poly_valid] = mls_tag

    if poly_count is None:
        poly_buffer = d_state.families.get(GeometryFamily.POLYGON)
        poly_count = 0 if poly_buffer is None else int(poly_buffer.geometry_offsets.size) - 1

    out_family_row_offsets = cp.full(row_count, -1, dtype=cp.int32)
    out_family_row_offsets[d_poly_valid] = d_state.family_row_offsets[d_poly_valid]

    # All-zero offsets = all empty
    d_geom_offsets = cp.zeros(poly_count + 1, dtype=cp.int32)
    d_part_offsets = cp.zeros(1, dtype=cp.int32)
    d_empty = cp.ones(poly_count, dtype=cp.bool_)
    d_x = cp.empty(0, dtype=cp.float64)
    d_y = cp.empty(0, dtype=cp.float64)

    device_families = {
        GeometryFamily.MULTILINESTRING: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTILINESTRING,
            x=d_x,
            y=d_y,
            geometry_offsets=d_geom_offsets,
            empty_mask=d_empty,
            part_offsets=d_part_offsets,
        ),
    }

    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------


def interiors_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Extract interior rings from Polygon geometries.

    Returns OwnedGeometryArray of MultiLineString geometries where each
    row's parts are the interior rings (holes) of the input polygon.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries.
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint.
    precision : PrecisionMode or str, default AUTO
        Precision mode.  COARSE class; wired for observability.

    Returns
    -------
    OwnedGeometryArray
        Interior ring geometries as MultiLineString.
    """
    row_count = owned.row_count
    if row_count == 0:
        return _build_all_null_oga(0)

    # Short-circuit only when host metadata is already present.  Device-only
    # callers must route through the GPU implementation to avoid metadata D2H.
    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    if owned._tags is not None and not np.any(owned._tags == poly_tag):
        return _build_all_null_oga(row_count)

    selection = plan_dispatch_selection(
        kernel_name="interior_rings",
        kernel_class=KernelClass.COARSE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
        current_residency=owned.residency,
        work_estimate=estimate_physical_work_from_owned(
            owned,
            output_row_count=row_count,
            primary_unit_name="interior-ring-coordinate",
        ),
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        try:
            result = _interiors_gpu(owned)
        except Exception:
            record_fallback_event(
                kernel_name="interior_rings",
                reason="GPU interior ring extraction failed, falling back to CPU",
                d2h_transfer=True,
            )
        else:
            record_dispatch_event(
                surface="geopandas.array.interiors",
                operation="interiors",
                implementation="interior_rings_gpu_cupy",
                reason=selection.reason,
                detail=(f"rows={row_count}, precision={precision_plan.compute_precision.value}"),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result

    result = _interiors_cpu(owned)
    record_dispatch_event(
        surface="geopandas.array.interiors",
        operation="interiors",
        implementation="interior_rings_cpu_shapely",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result


def interiors_native_tabular_result(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
    attrs: dict[str, object] | None = None,
):
    from vibespatial.api._native_results import (
        _unary_constructive_owned_to_native_tabular_result,
    )

    result = interiors_owned(
        owned,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
    return _unary_constructive_owned_to_native_tabular_result(
        result,
        operation="interiors",
        crs=crs,
        geometry_name=geometry_name,
        source_rows=source_rows,
        source_tokens=source_tokens,
        attrs=attrs,
    )
