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

from vibespatial.cuda._runtime import get_cuda_runtime
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
    poly_mask = owned.tags == poly_tag

    has_polys = (
        np.any(poly_mask)
        and GeometryFamily.POLYGON in d_state.families
    )

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

    # -------------------------------------------------------------------
    # Step 2: Compute per-polygon ring counts on device
    # ring_count[i] = geom_offsets[i+1] - geom_offsets[i]
    # interior_count[i] = max(ring_count[i] - 1, 0)
    # -------------------------------------------------------------------
    d_ring_counts = d_geom_offsets[1:] - d_geom_offsets[:-1]
    d_interior_counts = cp.maximum(d_ring_counts - 1, 0)

    total_interior_rings = int(cp.sum(d_interior_counts).item())

    if total_interior_rings == 0:
        # All polygons have zero holes — return all-empty MultiLineString
        return _build_all_empty_multilinestring(owned)

    # -------------------------------------------------------------------
    # Step 3: Build output geometry_offsets for MultiLineString
    # geometry_offsets[i] = cumulative sum of interior_counts[0..i-1]
    # This gives the part index range for each polygon family row.
    # -------------------------------------------------------------------
    d_out_geom_offsets = cp.zeros(poly_count + 1, dtype=cp.int32)
    cp.cumsum(d_interior_counts, out=d_out_geom_offsets[1:])

    # -------------------------------------------------------------------
    # Step 4: Identify interior ring indices in the source ring_offsets
    # For each polygon i, the interior rings are at ring indices:
    #   geom_offsets[i]+1, geom_offsets[i]+2, ..., geom_offsets[i+1]-1
    # We build a flat array of these source ring indices.
    # -------------------------------------------------------------------
    # Build mask of which ring indices are interior (not exterior)
    # Total rings = ring_offsets.size - 1
    total_rings = int(d_ring_offsets.size) - 1

    # For each polygon, mark ring 0 (the exterior) for exclusion.
    # Exterior ring index for polygon i = geom_offsets[i]
    d_exterior_indices = d_geom_offsets[:poly_count]  # first ring of each polygon

    # Create a boolean mask: True = interior ring
    d_is_interior = cp.ones(total_rings, dtype=cp.bool_)
    d_is_interior[d_exterior_indices] = False

    # Gather the interior ring indices
    d_interior_ring_indices = cp.flatnonzero(d_is_interior)
    assert int(d_interior_ring_indices.size) == total_interior_rings

    # -------------------------------------------------------------------
    # Step 5: Build output part_offsets from interior ring coordinate spans
    # For interior ring j, coordinates span:
    #   ring_offsets[j] to ring_offsets[j+1]
    # part_offsets[k] = cumulative coordinate count for interior ring k
    # -------------------------------------------------------------------
    d_ring_starts = d_ring_offsets[d_interior_ring_indices]
    d_ring_ends = d_ring_offsets[d_interior_ring_indices + 1]
    d_ring_lengths = d_ring_ends - d_ring_starts

    d_out_part_offsets = cp.zeros(total_interior_rings + 1, dtype=cp.int32)
    cp.cumsum(d_ring_lengths, out=d_out_part_offsets[1:])

    total_coords = int(d_out_part_offsets[-1].item())

    # -------------------------------------------------------------------
    # Step 6: Gather interior ring coordinates via segmented arange
    #
    # Build a flat source-index array so that output position p maps to
    # source coordinate index src[p].  The pattern is:
    #   d_src = [1, 1, 1, ...]   (fill with 1 = +1 increment)
    #   d_src[part_start_0] = ring_starts[0]        (absolute start)
    #   d_src[part_start_k] = ring_starts[k] - ring_ends[k-1] + 1
    #                                                (jump between segments)
    #   cumsum(d_src) → [ring_starts[0], +1, +1, ...,
    #                     ring_starts[1], +1, +1, ...]
    # -------------------------------------------------------------------
    if total_coords == 0:
        # Edge case: interior rings exist but are all empty (degenerate)
        return _build_all_empty_multilinestring(owned)

    d_part_starts_out = d_out_part_offsets[:-1]
    d_nonempty = d_ring_lengths > 0
    d_nonempty_part_starts = d_part_starts_out[d_nonempty]
    d_nonempty_ring_starts = d_ring_starts[d_nonempty]

    d_src = cp.ones(total_coords, dtype=cp.int32)

    if int(d_nonempty.sum().item()) > 0:
        d_nonempty_ring_ends = d_ring_ends[d_nonempty]

        # First non-empty ring: absolute start value
        first_pos = int(d_nonempty_part_starts[0].item())
        d_src[first_pos] = d_nonempty_ring_starts[0]

        # Subsequent non-empty rings: jump from previous ring end
        if int(d_nonempty_part_starts.size) > 1:
            d_jumps = d_nonempty_ring_starts[1:] - d_nonempty_ring_ends[:-1] + 1
            d_src[d_nonempty_part_starts[1:]] = d_jumps

    d_src_indices = cp.cumsum(d_src)

    # Gather coordinates using fancy indexing (zero-copy on device)
    d_x_out = d_poly.x[d_src_indices]
    d_y_out = d_poly.y[d_src_indices]

    # -------------------------------------------------------------------
    # Step 7: Build output OGA metadata
    # -------------------------------------------------------------------
    out_validity = owned.validity.copy()
    mls_tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
    out_tags = np.full(row_count, mls_tag, dtype=np.int8)
    out_tags[~poly_mask] = -1  # non-polygon rows are null
    out_validity[~poly_mask] = False

    # Handle original null rows
    out_tags[~owned.validity] = -1
    out_validity[~owned.validity] = False

    out_family_row_offsets = np.full(row_count, -1, dtype=np.int32)
    out_family_row_offsets[poly_mask & owned.validity] = np.arange(
        poly_count, dtype=np.int32
    )

    d_empty_mask = cp.zeros(poly_count, dtype=cp.bool_)

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
    )


def _build_all_empty_multilinestring(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Build a MultiLineString OGA where every row is empty or null.

    Polygon rows get an empty MultiLineString (zero parts).
    Non-polygon rows and null rows get null.
    """
    runtime = get_cuda_runtime()

    row_count = owned.row_count
    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mls_tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
    poly_mask = owned.tags == poly_tag

    out_validity = owned.validity.copy()
    out_tags = np.full(row_count, mls_tag, dtype=np.int8)
    out_tags[~poly_mask] = -1
    out_validity[~poly_mask] = False
    out_tags[~owned.validity] = -1
    out_validity[~owned.validity] = False

    poly_count = int(np.sum(poly_mask & owned.validity))

    out_family_row_offsets = np.full(row_count, -1, dtype=np.int32)
    out_family_row_offsets[poly_mask & owned.validity] = np.arange(
        poly_count, dtype=np.int32
    )

    # All-zero offsets = all empty
    d_geom_offsets = runtime.from_host(np.zeros(poly_count + 1, dtype=np.int32))
    d_part_offsets = runtime.from_host(np.zeros(1, dtype=np.int32))
    d_empty = runtime.from_host(np.zeros(poly_count, dtype=bool))
    d_x = runtime.from_host(np.empty(0, dtype=np.float64))
    d_y = runtime.from_host(np.empty(0, dtype=np.float64))

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
    )


# ---------------------------------------------------------------------------
# CPU fallback: interiors via Shapely
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "interior_rings",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("shapely", "constructive", "interiors"),
)
def _interiors_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Compute interior rings using Shapely as reference implementation.

    Returns MultiLineString OGA where each row's parts are the interior rings.
    Non-polygon rows produce None.
    """
    from shapely.geometry import MultiLineString as ShapelyMultiLineString

    geoms = owned.to_shapely()
    results = []
    for g in geoms:
        if g is None:
            results.append(None)
            continue
        if hasattr(g, "interiors"):
            interior_rings = list(g.interiors)
            if len(interior_rings) == 0:
                results.append(ShapelyMultiLineString())
            else:
                results.append(ShapelyMultiLineString(interior_rings))
        else:
            results.append(None)
    return from_shapely_geometries(results)


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

    # Short-circuit: no polygons means all-null output
    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    if not np.any(owned.tags == poly_tag):
        return _build_all_null_oga(row_count)

    selection = plan_dispatch_selection(
        kernel_name="interior_rings",
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
                detail=(
                    f"rows={row_count}, "
                    f"precision={precision_plan.compute_precision.value}"
                ),
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
