"""GPU-accelerated boundary computation.

Computes the topological boundary of each geometry:
- Point / MultiPoint: boundary is empty (None per Shapely convention).
- LineString: boundary is the two endpoints as MultiPoint.
- Polygon: single-ring boundaries are LineString; polygons with holes are
  MultiLineString.
- MultiLineString: boundary is endpoints of all parts as MultiPoint.
- MultiPolygon: boundary is all rings from all polygons as MultiLineString.

GPU paths for all six geometry families avoid the Shapely round-trip
entirely.  Polygon and MultiPolygon boundary use pure offset-relabeling
(zero-copy for coordinates).  LineString and MultiLineString boundary use
vectorized CuPy endpoint extraction.

ADR-0033 classification: Tier 2 (CuPy) for endpoint gather/interleave,
pure offset relabeling for polygon/multipolygon boundary.  No custom NVRTC
kernel needed because no path involves geometry-specific inner loops.

ADR-0002: CONSTRUCTIVE class -- stays fp64 by design per ADR-0002.
PrecisionPlan wired at dispatch for observability.
"""

from __future__ import annotations

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.boundary_cpu import _boundary_cpu as _boundary_cpu
from vibespatial.cuda._runtime import count_scatter_total, get_cuda_runtime
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
from vibespatial.runtime.precision import KernelClass, PrecisionMode

# ---------------------------------------------------------------------------
# GPU helpers: per-family boundary implementations
# ---------------------------------------------------------------------------


def _gather_ranged_coordinates(
    d_x,
    d_y,
    d_starts,
    d_ends,
    *,
    precomputed_total: int | None = None,
):
    d_lengths = d_ends - d_starts
    d_offsets = cp.empty(int(d_lengths.size) + 1, dtype=cp.int32)
    d_offsets[0] = 0
    cp.cumsum(d_lengths, out=d_offsets[1:])

    total_coords = (
        int(precomputed_total)
        if precomputed_total is not None
        else count_scatter_total(
            get_cuda_runtime(),
            d_lengths,
            d_offsets[:-1],
            reason="boundary coordinate allocation fence",
        )
    )
    if total_coords == 0:
        return (
            cp.empty(0, dtype=cp.float64),
            cp.empty(0, dtype=cp.float64),
            d_offsets,
        )

    d_out_idx = cp.arange(total_coords, dtype=cp.int32)
    d_range_idx = cp.searchsorted(d_offsets[1:], d_out_idx, side="right")
    d_src_idx = d_starts[d_range_idx] + (d_out_idx - d_offsets[d_range_idx])
    return d_x[d_src_idx], d_y[d_src_idx], d_offsets


def _polygon_rows_to_linestring_boundary_gpu(
    device_buf,
    d_polygon_rows,
    *,
    coord_total: int | None = None,
):
    d_geom_offsets = cp.asarray(device_buf.geometry_offsets)
    d_ring_offsets = cp.asarray(device_buf.ring_offsets)
    d_x = cp.asarray(device_buf.x)
    d_y = cp.asarray(device_buf.y)

    d_ring_idx = d_geom_offsets[d_polygon_rows]
    d_coord_starts = d_ring_offsets[d_ring_idx]
    d_coord_ends = d_ring_offsets[d_ring_idx + 1]
    d_x_out, d_y_out, d_geom_offsets_out = _gather_ranged_coordinates(
        d_x,
        d_y,
        d_coord_starts,
        d_coord_ends,
        precomputed_total=coord_total,
    )
    d_empty = (d_coord_ends - d_coord_starts) == 0
    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        x=d_x_out,
        y=d_y_out,
        geometry_offsets=d_geom_offsets_out,
        empty_mask=d_empty,
        part_offsets=None,
        ring_offsets=None,
        bounds=None,
    )


def _polygon_rows_to_multilinestring_boundary_gpu(
    device_buf,
    d_polygon_rows,
    *,
    ring_total: int | None = None,
    coord_total: int | None = None,
):
    d_geom_offsets = cp.asarray(device_buf.geometry_offsets)
    d_ring_offsets = cp.asarray(device_buf.ring_offsets)
    d_x = cp.asarray(device_buf.x)
    d_y = cp.asarray(device_buf.y)

    d_ring_starts = d_geom_offsets[d_polygon_rows]
    d_ring_ends = d_geom_offsets[d_polygon_rows + 1]
    d_ring_counts = d_ring_ends - d_ring_starts

    d_out_geom_offsets = cp.empty(int(d_polygon_rows.size) + 1, dtype=cp.int32)
    d_out_geom_offsets[0] = 0
    cp.cumsum(d_ring_counts, out=d_out_geom_offsets[1:])

    total_rings = (
        int(ring_total)
        if ring_total is not None
        else count_scatter_total(
            get_cuda_runtime(),
            d_ring_counts,
            d_out_geom_offsets[:-1],
            reason="boundary multiline ring allocation fence",
        )
    )
    if total_rings == 0:
        d_part_offsets = cp.empty(1, dtype=cp.int32)
        d_part_offsets[0] = 0
        d_x_out = cp.empty(0, dtype=cp.float64)
        d_y_out = cp.empty(0, dtype=cp.float64)
    else:
        d_part_idx = cp.arange(total_rings, dtype=cp.int32)
        d_geom_idx = cp.searchsorted(d_out_geom_offsets[1:], d_part_idx, side="right")
        d_source_ring = (
            d_ring_starts[d_geom_idx]
            + (d_part_idx - d_out_geom_offsets[d_geom_idx])
        )
        d_coord_starts = d_ring_offsets[d_source_ring]
        d_coord_ends = d_ring_offsets[d_source_ring + 1]
        d_x_out, d_y_out, d_part_offsets = _gather_ranged_coordinates(
            d_x,
            d_y,
            d_coord_starts,
            d_coord_ends,
            precomputed_total=coord_total,
        )

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        x=d_x_out,
        y=d_y_out,
        geometry_offsets=d_out_geom_offsets,
        empty_mask=d_ring_counts == 0,
        part_offsets=d_part_offsets,
        ring_offsets=None,
        bounds=None,
    )


def _host_polygon_boundary_size_plan(host_buf, geom_count):
    if (
        host_buf is None
        or not host_buf.host_materialized
        or host_buf.ring_offsets is None
        or int(host_buf.geometry_offsets.size) != geom_count + 1
    ):
        return None

    geom_offsets = host_buf.geometry_offsets
    ring_offsets = host_buf.ring_offsets
    ring_counts = geom_offsets[1:] - geom_offsets[:-1]

    single_rows_host = (ring_counts == 1).nonzero()[0]
    multi_rows_host = (ring_counts != 1).nonzero()[0]

    single_coord_total = 0
    if single_rows_host.size:
        single_ring_rows = geom_offsets[single_rows_host]
        single_coord_total = int(
            (
                ring_offsets[single_ring_rows + 1]
                - ring_offsets[single_ring_rows]
            ).sum()
        )

    multi_ring_total = int(ring_counts[multi_rows_host].sum())
    multi_coord_total = 0
    if multi_rows_host.size:
        multi_start = geom_offsets[multi_rows_host]
        multi_end = geom_offsets[multi_rows_host + 1]
        multi_coord_total = int((ring_offsets[multi_end] - ring_offsets[multi_start]).sum())

    return {
        "single_coord_total": single_coord_total,
        "multi_ring_total": multi_ring_total,
        "multi_coord_total": multi_coord_total,
    }


def _boundary_polygon_gpu(device_buf, geom_count, host_buf=None):
    """Polygon boundary: row-aware device offset relabeling.

    Single-ring polygons match Shapely's LineString boundary type.  Polygons
    with holes keep the old MultiLineString mapping where polygon ring offsets
    become line part offsets.  Homogeneous batches keep the zero-copy mapping;
    mixed batches compact each output family on device so per-row geometry
    types remain correct.
    """
    host_size_plan = _host_polygon_boundary_size_plan(host_buf, geom_count)
    d_geom_offsets = cp.asarray(device_buf.geometry_offsets)
    d_ring_counts = d_geom_offsets[1:] - d_geom_offsets[:-1]

    d_single_rows = cp.flatnonzero(d_ring_counts == 1)
    d_multi_rows = cp.flatnonzero(d_ring_counts != 1)

    boundary_buffers = {}
    if int(d_single_rows.size) == geom_count:
        boundary_buffers[GeometryFamily.LINESTRING] = (
            DeviceFamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                x=device_buf.x,
                y=device_buf.y,
                geometry_offsets=device_buf.ring_offsets,
                empty_mask=device_buf.empty_mask,
                part_offsets=None,
                ring_offsets=None,
                bounds=None,
            ),
            cp.arange(geom_count, dtype=cp.int32),
        )
    elif int(d_single_rows.size) > 0:
        boundary_buffers[GeometryFamily.LINESTRING] = (
            _polygon_rows_to_linestring_boundary_gpu(
                device_buf,
                d_single_rows,
                coord_total=(
                    None
                    if host_size_plan is None
                    else host_size_plan["single_coord_total"]
                ),
            ),
            d_single_rows,
        )

    if int(d_multi_rows.size) == geom_count:
        boundary_buffers[GeometryFamily.MULTILINESTRING] = (
            DeviceFamilyGeometryBuffer(
                family=GeometryFamily.MULTILINESTRING,
                x=device_buf.x,
                y=device_buf.y,
                geometry_offsets=device_buf.geometry_offsets,
                empty_mask=device_buf.empty_mask,
                part_offsets=device_buf.ring_offsets,  # rings become parts
                ring_offsets=None,
                bounds=None,
            ),
            cp.arange(geom_count, dtype=cp.int32),
        )
    elif int(d_multi_rows.size) > 0:
        boundary_buffers[GeometryFamily.MULTILINESTRING] = (
            _polygon_rows_to_multilinestring_boundary_gpu(
                device_buf,
                d_multi_rows,
                ring_total=(
                    None
                    if host_size_plan is None
                    else host_size_plan["multi_ring_total"]
                ),
                coord_total=(
                    None
                    if host_size_plan is None
                    else host_size_plan["multi_coord_total"]
                ),
            ),
            d_multi_rows,
        )

    return boundary_buffers


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

    d_lens = d_geom_offsets[1:] - d_geom_offsets[:-1]
    d_empty = d_lens == 0
    d_nonempty_rows = cp.flatnonzero(~d_empty).astype(cp.int32, copy=False)
    total_pts = 2 * int(d_nonempty_rows.size)

    d_pts_per_geom = cp.where(d_empty, 0, 2).astype(cp.int32, copy=False)
    d_out_geom_offsets = cp.empty(geom_count + 1, dtype=cp.int32)
    d_out_geom_offsets[0] = 0
    cp.cumsum(d_pts_per_geom, out=d_out_geom_offsets[1:])

    if total_pts == 0:
        d_x_out = cp.empty(0, dtype=cp.float64)
        d_y_out = cp.empty(0, dtype=cp.float64)
    else:
        d_starts = d_geom_offsets[d_nonempty_rows]
        d_ends = d_geom_offsets[d_nonempty_rows + 1] - 1
        d_first_x = d_x[d_starts]
        d_first_y = d_y[d_starts]
        d_last_x = d_x[d_ends]
        d_last_y = d_y[d_ends]

        # Interleave: [first0, last0, first1, last1, ...]
        d_x_out = cp.empty(total_pts, dtype=cp.float64)
        d_y_out = cp.empty(total_pts, dtype=cp.float64)
        d_x_out[0::2] = d_first_x
        d_x_out[1::2] = d_last_x
        d_y_out[0::2] = d_first_y
        d_y_out[1::2] = d_last_y

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


def _boundary_multilinestring_gpu(device_buf, geom_count):
    """MultiLineString boundary: extract 2 endpoints per part as MultiPoint.

    For each MultiLineString geometry, the boundary is a MultiPoint containing
    the first and last coordinate of every part (LineString).  Uses the same
    vectorized CuPy endpoint extraction as ``_boundary_linestring_gpu``, but
    iterates over *parts* within each geometry rather than geometries directly.

    Offset mapping:
        Input:  geometry_offsets -> part_offsets -> coords
        Output: geometry_offsets -> coords  (MultiPoint -- flat)

    Each part contributes 2 endpoints, so output geometry g has
    ``2 * (geometry_offsets[g+1] - geometry_offsets[g])`` points.
    """
    d_geom_offsets = cp.asarray(device_buf.geometry_offsets)
    d_part_offsets = cp.asarray(device_buf.part_offsets)
    d_x = cp.asarray(device_buf.x)
    d_y = cp.asarray(device_buf.y)

    total_parts = int(d_part_offsets.shape[0]) - 1
    if total_parts == 0:
        d_pts_per_geom = cp.zeros(geom_count, dtype=cp.int32)
        d_x_out = cp.empty(0, dtype=cp.float64)
        d_y_out = cp.empty(0, dtype=cp.float64)
    else:
        d_part_starts = d_part_offsets[:-1]
        d_part_ends = d_part_offsets[1:] - 1
        d_part_lens = d_part_offsets[1:] - d_part_offsets[:-1]
        d_nonempty_parts = cp.flatnonzero(d_part_lens > 0).astype(
            cp.int32,
            copy=False,
        )
        total_pts = 2 * int(d_nonempty_parts.size)
        d_part_idx = cp.arange(total_parts, dtype=cp.int32)
        d_geom_for_part = cp.searchsorted(
            d_geom_offsets[1:],
            d_part_idx,
            side="right",
        )
        d_nonempty_counts = cp.bincount(
            d_geom_for_part[d_nonempty_parts],
            minlength=geom_count,
        ).astype(cp.int32, copy=False)
        d_pts_per_geom = d_nonempty_counts * 2

        if total_pts == 0:
            d_x_out = cp.empty(0, dtype=cp.float64)
            d_y_out = cp.empty(0, dtype=cp.float64)
        else:
            d_starts = d_part_starts[d_nonempty_parts]
            d_ends = d_part_ends[d_nonempty_parts]
            d_first_x = d_x[d_starts]
            d_first_y = d_y[d_starts]
            d_last_x = d_x[d_ends]
            d_last_y = d_y[d_ends]

            # Interleave: [first_p0, last_p0, first_p1, last_p1, ...]
            d_x_out = cp.empty(total_pts, dtype=cp.float64)
            d_y_out = cp.empty(total_pts, dtype=cp.float64)
            d_x_out[0::2] = d_first_x
            d_x_out[1::2] = d_last_x
            d_y_out[0::2] = d_first_y
            d_y_out[1::2] = d_last_y

    d_out_geom_offsets = cp.empty(geom_count + 1, dtype=cp.int32)
    d_out_geom_offsets[0] = 0
    cp.cumsum(d_pts_per_geom, out=d_out_geom_offsets[1:])

    d_empty = d_pts_per_geom == 0

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


def _boundary_multipolygon_gpu(device_buf, geom_count):
    """MultiPolygon boundary: reinterpret ring offsets as MultiLineString parts.

    Zero-copy for coordinates -- only offset arrays are relabeled.
    The input MultiPolygon has a three-level offset hierarchy:
        geometry_offsets -> part_offsets -> ring_offsets -> coords
    where part_offsets maps polygon-parts to their first ring, and
    ring_offsets maps rings to their first coordinate.

    The boundary flattens the polygon level: every ring (from any polygon
    in the multi) becomes a LineString part in the output MultiLineString.
    Output offset hierarchy:
        geometry_offsets -> part_offsets -> coords
    where part_offsets = input ring_offsets, and geometry_offsets is
    recomputed so that geometry g maps to the total ring range contributed
    by all of its constituent polygons.

    Specifically, for geometry g the ring range is
    ``[part_offsets[geometry_offsets[g]], part_offsets[geometry_offsets[g+1]])``
    in the input.  The output geometry_offsets are obtained by gathering
    part_offsets at the input geometry_offsets positions.
    """
    d_geom_offsets = cp.asarray(device_buf.geometry_offsets)
    d_part_offsets = cp.asarray(device_buf.part_offsets)

    # Output geometry_offsets: compose geometry_offsets through part_offsets.
    # d_part_offsets[d_geom_offsets[g]] gives the ring index where geometry g
    # starts, which is exactly the output geometry_offsets value.
    d_out_geom_offsets = d_part_offsets[d_geom_offsets]

    # Detect empty MultiPolygons: 0 polygon-parts
    d_parts_per_geom = d_geom_offsets[1:] - d_geom_offsets[:-1]
    d_empty = d_parts_per_geom == 0

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        x=device_buf.x,
        y=device_buf.y,
        geometry_offsets=d_out_geom_offsets,
        empty_mask=d_empty,
        part_offsets=device_buf.ring_offsets,  # rings become LineString parts
        ring_offsets=None,
        bounds=None,
    )


def _merge_multipoint_buffers(buf_a, buf_b):
    """Merge two MultiPoint DeviceFamilyGeometryBuffers into one.

    Used when both LineString and MultiLineString families produce MultiPoint
    output.  Concatenates coordinate arrays and adjusts geometry_offsets for
    the second buffer.  Empty masks are concatenated.

    Both buffers must be MultiPoint (no part_offsets or ring_offsets).
    """
    # Coordinate offset: second buffer's coords shift by length of first
    n_coords_a = int(buf_a.x.shape[0])

    d_x = cp.concatenate([buf_a.x, buf_b.x])
    d_y = cp.concatenate([buf_a.y, buf_b.y])

    # Adjust second buffer's geometry_offsets (skip its leading 0)
    d_geom_offsets_b_shifted = cp.asarray(buf_b.geometry_offsets[1:]) + n_coords_a
    d_geom_offsets = cp.concatenate([
        cp.asarray(buf_a.geometry_offsets),
        d_geom_offsets_b_shifted,
    ])

    d_empty = cp.concatenate([
        cp.asarray(buf_a.empty_mask),
        cp.asarray(buf_b.empty_mask),
    ])

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOINT,
        x=d_x,
        y=d_y,
        geometry_offsets=d_geom_offsets,
        empty_mask=d_empty,
        part_offsets=None,
        ring_offsets=None,
        bounds=None,
    )


def _merge_multilinestring_buffers(buf_a, buf_b):
    """Merge two MultiLineString DeviceFamilyGeometryBuffers into one.

    Used when both Polygon and MultiPolygon families produce MultiLineString
    output.  Concatenates coordinate arrays and part_offsets, adjusting
    offsets for the second buffer.  Empty masks are concatenated.
    """
    n_coords_a = int(buf_a.x.shape[0])
    n_parts_a = int(buf_a.part_offsets.shape[0]) - 1  # number of parts in A

    d_x = cp.concatenate([buf_a.x, buf_b.x])
    d_y = cp.concatenate([buf_a.y, buf_b.y])

    # Merge part_offsets: shift B's coord references by A's coord count
    d_part_offsets_b_shifted = cp.asarray(buf_b.part_offsets[1:]) + n_coords_a
    d_part_offsets = cp.concatenate([
        cp.asarray(buf_a.part_offsets),
        d_part_offsets_b_shifted,
    ])

    # Merge geometry_offsets: shift B's part references by A's part count
    d_geom_offsets_b_shifted = cp.asarray(buf_b.geometry_offsets[1:]) + n_parts_a
    d_geom_offsets = cp.concatenate([
        cp.asarray(buf_a.geometry_offsets),
        d_geom_offsets_b_shifted,
    ])

    d_empty = cp.concatenate([
        cp.asarray(buf_a.empty_mask),
        cp.asarray(buf_b.empty_mask),
    ])

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        x=d_x,
        y=d_y,
        geometry_offsets=d_geom_offsets,
        empty_mask=d_empty,
        part_offsets=d_part_offsets,
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
    geometry_families=(
        "polygon", "linestring", "multilinestring", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "boundary"),
)
def _boundary_gpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """GPU boundary for all geometry families.

    - Polygon -> LineString for single-ring rows; MultiLineString otherwise.
    - MultiPolygon -> MultiLineString: offset composition (zero-copy coords).
    - LineString -> MultiPoint: vectorized CuPy endpoint extraction.
    - MultiLineString -> MultiPoint: vectorized CuPy part-endpoint extraction.
    - Point / MultiPoint -> None: mark rows as invalid (empty boundary).

    When both Polygon and MultiPolygon are present, the two MultiLineString
    outputs are merged via coordinate concatenation + offset adjustment.
    Similarly for LineString + MultiLineString producing two MultiPoint outputs.
    """
    d_state = owned._ensure_device_state()

    new_device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
    out_tags = cp.asarray(d_state.tags).copy()
    out_validity = cp.asarray(d_state.validity).copy()

    # Use original tags for source masks to avoid cross-contamination when
    # two input families produce the same output family.  For example,
    # MultiPolygon -> MultiLineString would pollute the MultiLineString ->
    # MultiPoint remap if we matched against the already-modified out_tags.
    src_tags = cp.asarray(d_state.tags)
    family_global_rows_ordered: dict[GeometryFamily, list] = {}

    for family, device_buf in d_state.families.items():
        geom_count = int(device_buf.geometry_offsets.shape[0]) - 1
        if geom_count == 0:
            continue

        if family is GeometryFamily.POLYGON:
            src_rows = cp.flatnonzero(src_tags == FAMILY_TAGS[GeometryFamily.POLYGON])
            boundary_buffers = _boundary_polygon_gpu(
                device_buf,
                geom_count,
                host_buf=owned.families.get(family),
            )
            for out_family, (new_buf, local_rows) in boundary_buffers.items():
                if int(local_rows.size) == 0:
                    continue
                if (
                    out_family is GeometryFamily.MULTILINESTRING
                    and GeometryFamily.MULTILINESTRING in new_device_families
                ):
                    new_device_families[GeometryFamily.MULTILINESTRING] = (
                        _merge_multilinestring_buffers(
                            new_device_families[GeometryFamily.MULTILINESTRING],
                            new_buf,
                        )
                    )
                else:
                    new_device_families[out_family] = new_buf
                output_rows = src_rows[local_rows]
                family_global_rows_ordered.setdefault(out_family, []).append(output_rows)
                out_tags[output_rows] = FAMILY_TAGS[out_family]

        elif family is GeometryFamily.MULTIPOLYGON:
            src_rows = cp.flatnonzero(src_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
            new_buf = _boundary_multipolygon_gpu(device_buf, geom_count)
            if GeometryFamily.MULTILINESTRING in new_device_families:
                new_device_families[GeometryFamily.MULTILINESTRING] = (
                    _merge_multilinestring_buffers(
                        new_device_families[GeometryFamily.MULTILINESTRING],
                        new_buf,
                    )
                )
            else:
                new_device_families[GeometryFamily.MULTILINESTRING] = new_buf
            family_global_rows_ordered.setdefault(GeometryFamily.MULTILINESTRING, []).append(
                src_rows,
            )
            # Remap tags: MultiPolygon rows -> MultiLineString
            mpoly_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
            mls_tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
            out_tags[src_tags == mpoly_tag] = mls_tag

        elif family is GeometryFamily.LINESTRING:
            src_rows = cp.flatnonzero(src_tags == FAMILY_TAGS[GeometryFamily.LINESTRING])
            new_buf = _boundary_linestring_gpu(device_buf, geom_count)
            if GeometryFamily.MULTIPOINT in new_device_families:
                new_device_families[GeometryFamily.MULTIPOINT] = (
                    _merge_multipoint_buffers(
                        new_device_families[GeometryFamily.MULTIPOINT],
                        new_buf,
                    )
                )
            else:
                new_device_families[GeometryFamily.MULTIPOINT] = new_buf
            family_global_rows_ordered.setdefault(GeometryFamily.MULTIPOINT, []).append(
                src_rows,
            )
            # Remap tags: LineString rows -> MultiPoint
            ls_tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
            mp_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
            out_tags[src_tags == ls_tag] = mp_tag

        elif family is GeometryFamily.MULTILINESTRING:
            src_rows = cp.flatnonzero(src_tags == FAMILY_TAGS[GeometryFamily.MULTILINESTRING])
            new_buf = _boundary_multilinestring_gpu(device_buf, geom_count)
            if GeometryFamily.MULTIPOINT in new_device_families:
                new_device_families[GeometryFamily.MULTIPOINT] = (
                    _merge_multipoint_buffers(
                        new_device_families[GeometryFamily.MULTIPOINT],
                        new_buf,
                    )
                )
            else:
                new_device_families[GeometryFamily.MULTIPOINT] = new_buf
            family_global_rows_ordered.setdefault(GeometryFamily.MULTIPOINT, []).append(
                src_rows,
            )
            # Remap tags: MultiLineString rows -> MultiPoint
            mls_tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
            mp_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
            out_tags[src_tags == mls_tag] = mp_tag

        elif family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
            # Boundary of Point / MultiPoint is empty -- mark rows invalid
            tag = FAMILY_TAGS[family]
            pt_mask = src_tags == tag
            out_validity[pt_mask] = False

    # Recompute family_row_offsets for the new tag assignments.
    # Family types changed (Polygon -> LineString/MultiLineString,
    # LineString -> MultiPoint, etc.), so the mapping from global row ->
    # family-local row must be rebuilt.
    new_family_row_offsets = cp.full(owned.row_count, -1, dtype=cp.int32)
    for row_chunks in family_global_rows_ordered.values():
        ordered_rows = cp.concatenate(row_chunks) if len(row_chunks) > 1 else row_chunks[0]
        new_family_row_offsets[ordered_rows] = cp.arange(
            int(ordered_rows.size), dtype=cp.int32,
        )

    return build_device_resident_owned(
        device_families=new_device_families,
        row_count=owned.row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=new_family_row_offsets,
        execution_mode="gpu",
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def boundary_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Compute the topological boundary of each geometry.

    Returns an OwnedGeometryArray whose geometry type depends on the input:
    - Point / MultiPoint rows produce None (empty boundary).
    - LineString rows produce MultiPoint (the two endpoints).
    - Polygon rows produce LineString for single-ring polygons, otherwise
      MultiLineString.
    - MultiLineString rows produce MultiPoint (endpoints of all parts).
    - MultiPolygon rows produce MultiLineString (all rings as LineStrings).

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries.
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint.  ``GPU`` dispatches to the device-native
        CuPy-based implementation for all geometry families.
    precision : PrecisionMode or str, default AUTO
        Precision mode.  CONSTRUCTIVE class stays fp64 by design per
        ADR-0002; wired here for observability.

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
        requested_precision=precision,
        current_residency=owned.residency,
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        result = _boundary_gpu(owned)
        record_dispatch_event(
            surface="geopandas.array.boundary",
            operation="boundary",
            implementation="boundary_gpu_cupy",
            reason=selection.reason,
            detail=(
                f"rows={row_count}, "
                f"precision={precision_plan.compute_precision.value} "
                f"(offset-only, not parameterized)"
            ),
            requested=selection.requested,
            selected=ExecutionMode.GPU,
        )
        return result

    result = _boundary_cpu(owned)
    record_dispatch_event(
        surface="geopandas.array.boundary",
        operation="boundary",
        implementation="boundary_cpu_shapely",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result


def boundary_native_tabular_result(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
):
    """Return boundary output as a private native constructive carrier."""
    from vibespatial.api._native_results import (
        _unary_constructive_owned_to_native_tabular_result,
    )

    result = boundary_owned(
        owned,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
    return _unary_constructive_owned_to_native_tabular_result(
        result,
        operation="boundary",
        crs=crs,
        geometry_name=geometry_name,
        source_rows=source_rows,
        source_tokens=source_tokens,
    )
