"""GPU-accelerated boundary computation.

Computes the topological boundary of each geometry:
- Point / MultiPoint: boundary is empty (None per Shapely convention).
- LineString: boundary is the two endpoints as MultiPoint.
- Polygon: boundary is all rings (exterior + holes) as MultiLineString.
- MultiLineString: boundary is endpoints of all parts as MultiPoint.
- MultiPolygon: boundary is all rings from all polygons as MultiLineString.

GPU paths for Polygon and LineString families avoid the Shapely round-trip
entirely.  Polygon boundary is a pure offset-relabeling (zero-copy for
coordinates).  LineString boundary uses vectorized CuPy endpoint extraction.

ADR-0033 classification: Tier 2 (CuPy) for endpoint gather/interleave,
pure offset relabeling for polygon boundary.  No custom NVRTC kernel needed
because neither path involves geometry-specific inner loops.

ADR-0002: CONSTRUCTIVE class -- stays fp64 by design per ADR-0002.
PrecisionPlan wired at dispatch for observability.
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
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


# ---------------------------------------------------------------------------
# CPU fallback: boundary via Shapely
# ---------------------------------------------------------------------------

def _boundary_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Compute boundary of each geometry using Shapely as the reference
    implementation.

    Converts to Shapely geometries, calls ``shapely.boundary`` on each,
    and converts back to an OwnedGeometryArray.
    """
    import shapely

    geoms = owned.to_shapely()
    results = [shapely.boundary(g) if g is not None else None for g in geoms]
    return from_shapely_geometries(results)


# ---------------------------------------------------------------------------
# GPU helpers: per-family boundary implementations
# ---------------------------------------------------------------------------

def _boundary_polygon_gpu(device_buf, geom_count):
    """Polygon boundary: reinterpret ring offsets as MultiLineString parts.

    Zero-copy for coordinates -- only offset arrays are relabeled.
    Polygon's ring_offsets (ring -> coord range) becomes MultiLineString's
    part_offsets (part -> coord range).  geometry_offsets and coordinates
    are shared without copy.
    """
    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        x=device_buf.x,
        y=device_buf.y,
        geometry_offsets=device_buf.geometry_offsets,
        empty_mask=device_buf.empty_mask,
        part_offsets=device_buf.ring_offsets,  # rings become parts
        ring_offsets=None,
        bounds=None,
    )


def _boundary_linestring_gpu(device_buf, geom_count):
    """LineString boundary: extract 2 endpoints per geometry as MultiPoint.

    For each LineString, the boundary is a MultiPoint containing the first
    and last coordinate.  Uses vectorized CuPy fancy indexing to gather
    endpoints and interleave them into the output buffer.

    Empty LineStrings (0 coordinates) produce empty MultiPoint geometries
    and are handled via clamped indexing + validity masking.
    """
    d_geom_offsets = cp.asarray(device_buf.geometry_offsets)
    d_x = cp.asarray(device_buf.x)
    d_y = cp.asarray(device_buf.y)

    d_starts = d_geom_offsets[:-1]
    d_ends = d_geom_offsets[1:] - 1

    # Clamp ends to be >= starts to avoid negative indexing on empty geoms.
    # Empty geometries will index the same coord twice (harmless; masked out
    # by validity downstream).
    d_ends = cp.maximum(d_ends, d_starts)

    # Gather first and last coordinates
    d_first_x = d_x[d_starts]
    d_first_y = d_y[d_starts]
    d_last_x = d_x[d_ends]
    d_last_y = d_y[d_ends]

    # Interleave: [first0, last0, first1, last1, ...]
    total_pts = 2 * geom_count
    d_x_out = cp.empty(total_pts, dtype=cp.float64)
    d_y_out = cp.empty(total_pts, dtype=cp.float64)
    d_x_out[0::2] = d_first_x
    d_x_out[1::2] = d_last_x
    d_y_out[0::2] = d_first_y
    d_y_out[1::2] = d_last_y

    # geometry_offsets: [0, 2, 4, 6, ...] -- 2 points per geometry
    d_out_geom_offsets = cp.arange(
        0, total_pts + 2, 2, dtype=cp.int32,
    )[:geom_count + 1]

    # Detect empty LineStrings: length == 0 coords
    d_lens = d_geom_offsets[1:] - d_geom_offsets[:-1]
    d_empty = d_lens == 0

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOINT,
        x=d_x_out,
        y=d_y_out,
        geometry_offsets=d_out_geom_offsets,
        empty_mask=d_empty,
        part_offsets=None,
        ring_offsets=None,
        bounds=None,
    )


# ---------------------------------------------------------------------------
# GPU dispatch: registered kernel variant
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "boundary",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "linestring"),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "boundary"),
)
def _boundary_gpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """GPU boundary for Polygon and LineString families.

    - Polygon -> MultiLineString: zero-copy offset relabeling.
    - LineString -> MultiPoint: vectorized CuPy endpoint extraction.
    - Point / MultiPoint -> None: mark rows as invalid (empty boundary).
    - MultiLineString / MultiPolygon: not supported -- raises
      NotImplementedError so the caller falls through to CPU.
    """
    d_state = owned._ensure_device_state()

    # Gate: only handle families we have GPU paths for
    gpu_families = {
        GeometryFamily.POLYGON,
        GeometryFamily.LINESTRING,
        GeometryFamily.POINT,
        GeometryFamily.MULTIPOINT,
    }
    present = set(d_state.families.keys())
    if not present.issubset(gpu_families):
        raise NotImplementedError(
            "boundary GPU only supports Polygon/LineString/Point/MultiPoint; "
            f"found {present - gpu_families}"
        )

    new_device_families = {}
    out_tags = owned.tags.copy()
    out_validity = owned.validity.copy()

    for family, device_buf in d_state.families.items():
        geom_count = int(device_buf.geometry_offsets.shape[0]) - 1
        if geom_count == 0:
            continue

        if family is GeometryFamily.POLYGON:
            new_buf = _boundary_polygon_gpu(device_buf, geom_count)
            new_device_families[GeometryFamily.MULTILINESTRING] = new_buf
            # Remap tags: Polygon rows -> MultiLineString
            poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
            mls_tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
            out_tags[out_tags == poly_tag] = mls_tag

        elif family is GeometryFamily.LINESTRING:
            new_buf = _boundary_linestring_gpu(device_buf, geom_count)
            new_device_families[GeometryFamily.MULTIPOINT] = new_buf
            # Remap tags: LineString rows -> MultiPoint
            ls_tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
            mp_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
            out_tags[out_tags == ls_tag] = mp_tag

        elif family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
            # Boundary of Point / MultiPoint is empty -- mark rows invalid
            tag = FAMILY_TAGS[family]
            pt_mask = owned.tags == tag
            out_validity[pt_mask] = False

    # Recompute family_row_offsets for the new tag assignments.
    # Family types changed (Polygon -> MultiLineString, LineString -> MultiPoint),
    # so the mapping from global row -> family-local row must be rebuilt.
    new_family_row_offsets = np.full(owned.row_count, -1, dtype=np.int32)
    for fam in new_device_families:
        fam_tag = FAMILY_TAGS[fam]
        fam_global = np.flatnonzero(out_tags == fam_tag)
        new_family_row_offsets[fam_global] = np.arange(
            len(fam_global), dtype=np.int32,
        )

    return build_device_resident_owned(
        device_families=new_device_families,
        row_count=owned.row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=new_family_row_offsets,
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def boundary_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Compute the topological boundary of each geometry.

    Returns an OwnedGeometryArray whose geometry type depends on the input:
    - Point / MultiPoint rows produce None (empty boundary).
    - LineString rows produce MultiPoint (the two endpoints).
    - Polygon rows produce MultiLineString (ring boundaries).
    - MultiLineString rows produce MultiPoint (endpoints of all parts).
    - MultiPolygon rows produce MultiLineString (all rings as LineStrings).

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries.
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint.  ``GPU`` dispatches to the CuPy-based
        implementation for Polygon and LineString families; other
        families fall through to CPU/Shapely.

    Returns
    -------
    OwnedGeometryArray
        Boundary geometries.
    """
    row_count = owned.row_count
    if row_count == 0:
        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="boundary",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        try:
            return _boundary_gpu(owned)
        except Exception:
            pass

    return _boundary_cpu(owned)
