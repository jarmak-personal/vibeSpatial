"""GPU-accelerated disjoint_subset_union_all: geometry collection assembly.

Since geometries are disjoint by assumption, union is just geometry collection
assembly -- concatenate coordinate buffers and chain offset arrays.  No Boolean
geometry operations are performed.

For homogeneous input (all same family), the output is the corresponding Multi*
family:
  - Point -> MultiPoint
  - LineString -> MultiLineString
  - Polygon -> MultiPolygon
  - Multi* -> Multi* (merge all parts into a single row)

For mixed families, falls back to Shapely CPU path.

ADR-0002: CONSTRUCTIVE class -- fp64, no precision downgrade (coordinates are
          exact subsets, no new coordinates created).
ADR-0033: No NVRTC kernel needed -- pure CuPy buffer manipulation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

import shapely

from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan
from vibespatial.runtime.residency import Residency

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryDeviceState

logger = logging.getLogger(__name__)

# Merge groups: families that can be collapsed into a single multi-type.
_MERGE_TARGETS: dict[GeometryFamily, GeometryFamily] = {
    GeometryFamily.POINT: GeometryFamily.MULTIPOINT,
    GeometryFamily.MULTIPOINT: GeometryFamily.MULTIPOINT,
    GeometryFamily.LINESTRING: GeometryFamily.MULTILINESTRING,
    GeometryFamily.MULTILINESTRING: GeometryFamily.MULTILINESTRING,
    GeometryFamily.POLYGON: GeometryFamily.MULTIPOLYGON,
    GeometryFamily.MULTIPOLYGON: GeometryFamily.MULTIPOLYGON,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def disjoint_subset_union_all_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray | None:
    """Union all geometries by disjoint-subset assembly (no Boolean ops).

    Since geometries are disjoint by assumption, this concatenates all
    coordinate buffers and chains offset arrays to produce a single Multi*
    geometry.

    Returns ``None`` when the input contains mixed geometry families that
    cannot be merged into a single Multi* type (e.g. Point + Polygon).
    The caller should fall through to the Shapely CPU path in that case.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries (device- or host-resident).
    dispatch_mode : ExecutionMode or str
        Execution mode hint (AUTO, GPU, CPU).
    precision : PrecisionMode or str
        Precision mode.  Stays fp64 per ADR-0002 (exact coord subsets).

    Returns
    -------
    OwnedGeometryArray or None
        Single-row OGA containing all input geometries assembled into one,
        or ``None`` when the input has incompatible mixed families.
    """
    row_count = owned.row_count

    # Empty input -> empty geometry.
    if row_count == 0:
        return _empty_result()

    selection = plan_dispatch_selection(
        kernel_name="disjoint_subset_union_all",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU and cp is not None:
        precision_plan = select_precision_plan(  # noqa: F841 — called for ADR-0002 observability side-effects
            runtime_selection=selection,
            kernel_class=KernelClass.CONSTRUCTIVE,
            requested=precision,
        )
        try:
            result = _disjoint_subset_union_all_gpu(owned)
            if result is not None:
                record_dispatch_event(
                    surface="constructive.disjoint_subset_union_all",
                    operation="disjoint_subset_union_all",
                    implementation="disjoint_subset_union_all_gpu",
                    reason=selection.reason,
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                )
                return result
            # Mixed families -- GPU path returns None, signal caller to
            # fall back to Shapely.
            record_fallback_event(
                surface="constructive.disjoint_subset_union_all",
                reason="mixed families not supported on GPU",
                requested=dispatch_mode,
                selected=ExecutionMode.CPU,
                d2h_transfer=owned.residency is Residency.DEVICE,
            )
            return None
        except Exception:
            logger.debug(
                "disjoint_subset_union_all GPU failed, falling back to CPU",
                exc_info=True,
            )

    # CPU fallback -- try to assemble via host-side buffer concatenation.
    result = _disjoint_subset_union_all_cpu(owned)
    if result is not None:
        record_dispatch_event(
            surface="constructive.disjoint_subset_union_all",
            operation="disjoint_subset_union_all",
            implementation="disjoint_subset_union_all_cpu",
            reason="cpu fallback",
            requested=dispatch_mode,
            selected=ExecutionMode.CPU,
        )
        record_fallback_event(
            surface="constructive.disjoint_subset_union_all",
            reason="CPU path used",
            requested=dispatch_mode,
            selected=ExecutionMode.CPU,
            d2h_transfer=owned.residency is Residency.DEVICE,
        )
    return result


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "disjoint_subset_union_all",
    "gpu-cupy",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=False,
    tags=("cupy", "constructive", "union_all"),
)
def _disjoint_subset_union_all_gpu(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """Assemble all geometries into a single Multi* on GPU.

    Returns None if the input contains mixed families that cannot be merged
    into a single Multi* type, signalling the caller to fall back to CPU.
    """
    ds = owned._ensure_device_state()

    # Determine which families are present (skipping null rows).
    valid_families: set[GeometryFamily] = set()
    for family, d_buf in ds.families.items():
        # A family is present if it has at least one geometry row.
        n_rows = int(d_buf.geometry_offsets.size) - 1 if d_buf.geometry_offsets.size > 0 else 0
        if n_rows > 0:
            valid_families.add(family)

    if not valid_families:
        return _empty_result()

    # Check if all present families merge to the same multi-type.
    target_families = {_MERGE_TARGETS.get(f) for f in valid_families}
    if len(target_families) != 1 or None in target_families:
        # Mixed families that cannot be merged -> signal CPU fallback.
        return None

    target_family = target_families.pop()

    # Single row, no nulls -> identity (just return the input).
    if owned.row_count == 1:
        valid = ds.validity
        if bool(cp.all(valid)):
            return owned

    # Dispatch to family-specific assembly.
    if target_family is GeometryFamily.MULTIPOINT:
        return _assemble_multipoint_gpu(ds, valid_families)
    elif target_family is GeometryFamily.MULTILINESTRING:
        return _assemble_multilinestring_gpu(ds, valid_families)
    elif target_family is GeometryFamily.MULTIPOLYGON:
        return _assemble_multipolygon_gpu(ds, valid_families)
    else:
        return None


def _assemble_multipoint_gpu(
    ds: OwnedGeometryDeviceState,
    families: set[GeometryFamily],
) -> OwnedGeometryArray:
    """Assemble Point and/or MultiPoint rows into a single MultiPoint."""
    all_x_parts: list = []
    all_y_parts: list = []

    for family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
        if family not in families:
            continue
        d_buf = ds.families[family]
        if d_buf.x.size > 0:
            all_x_parts.append(d_buf.x)
            all_y_parts.append(d_buf.y)

    if not all_x_parts:
        return _empty_result()

    merged_x = cp.concatenate(all_x_parts) if len(all_x_parts) > 1 else all_x_parts[0]
    merged_y = cp.concatenate(all_y_parts) if len(all_y_parts) > 1 else all_y_parts[0]
    total_coords = int(merged_x.size)

    # Single-row MultiPoint: geometry_offsets = [0, total_coords]
    geom_offsets = cp.array([0, total_coords], dtype=cp.int32)
    empty_mask = cp.array([total_coords == 0], dtype=cp.bool_)

    out_family = GeometryFamily.MULTIPOINT
    d_buf = DeviceFamilyGeometryBuffer(
        family=out_family,
        x=merged_x,
        y=merged_y,
        geometry_offsets=geom_offsets,
        empty_mask=empty_mask,
    )

    return _build_single_row_oga(out_family, d_buf)


def _assemble_multilinestring_gpu(
    ds: OwnedGeometryDeviceState,
    families: set[GeometryFamily],
) -> OwnedGeometryArray:
    """Assemble LineString and/or MultiLineString rows into a single MultiLineString."""
    all_x_parts: list = []
    all_y_parts: list = []
    # For LineString: each geometry's coordinate span becomes one part.
    # For MultiLineString: each part is preserved as-is.
    part_offset_arrays: list = []
    part_counts: list[int] = []

    if GeometryFamily.LINESTRING in families:
        d_buf = ds.families[GeometryFamily.LINESTRING]
        if d_buf.x.size > 0:
            all_x_parts.append(d_buf.x)
            all_y_parts.append(d_buf.y)
            # LineString geometry_offsets define coordinate ranges per line;
            # these become part_offsets in the MultiLineString.
            part_offset_arrays.append(d_buf.geometry_offsets)
            part_counts.append(
                int(d_buf.geometry_offsets[-1]) if d_buf.geometry_offsets.size > 0 else 0
            )

    if GeometryFamily.MULTILINESTRING in families:
        d_buf = ds.families[GeometryFamily.MULTILINESTRING]
        if d_buf.x.size > 0:
            all_x_parts.append(d_buf.x)
            all_y_parts.append(d_buf.y)
            # MultiLineString already has part_offsets -> coords.
            part_offset_arrays.append(d_buf.part_offsets)
            part_counts.append(
                int(d_buf.part_offsets[-1]) if d_buf.part_offsets.size > 0 else 0
            )

    if not all_x_parts:
        return _empty_result()

    merged_x = cp.concatenate(all_x_parts) if len(all_x_parts) > 1 else all_x_parts[0]
    merged_y = cp.concatenate(all_y_parts) if len(all_y_parts) > 1 else all_y_parts[0]

    # Chain part_offsets from all sources.
    merged_part_offsets = _chain_device_offset_arrays(part_offset_arrays, part_counts)
    total_parts = int(merged_part_offsets.size) - 1

    # Single-row geometry_offsets = [0, total_parts]
    geom_offsets = cp.array([0, total_parts], dtype=cp.int32)
    empty_mask = cp.array([total_parts == 0], dtype=cp.bool_)

    out_family = GeometryFamily.MULTILINESTRING
    d_buf = DeviceFamilyGeometryBuffer(
        family=out_family,
        x=merged_x,
        y=merged_y,
        geometry_offsets=geom_offsets,
        empty_mask=empty_mask,
        part_offsets=merged_part_offsets,
    )

    return _build_single_row_oga(out_family, d_buf)


def _assemble_multipolygon_gpu(
    ds: OwnedGeometryDeviceState,
    families: set[GeometryFamily],
) -> OwnedGeometryArray:
    """Assemble Polygon and/or MultiPolygon rows into a single MultiPolygon."""
    all_x_parts: list = []
    all_y_parts: list = []
    # For Polygon: each geometry becomes one polygon part.
    #   geometry_offsets -> ring_offsets -> coords
    #   becomes: part_offsets (from geometry_offsets) -> ring_offsets -> coords
    # For MultiPolygon: preserve existing 3-level structure.
    ring_offset_arrays: list = []
    ring_counts: list[int] = []
    part_offset_arrays: list = []
    part_counts: list[int] = []

    if GeometryFamily.POLYGON in families:
        d_buf = ds.families[GeometryFamily.POLYGON]
        if d_buf.x.size > 0:
            all_x_parts.append(d_buf.x)
            all_y_parts.append(d_buf.y)
            # Polygon geometry_offsets index into ring_offsets (row -> ring).
            # In MultiPolygon, these become part_offsets (polygon -> ring).
            part_offset_arrays.append(d_buf.geometry_offsets)
            part_counts.append(
                int(d_buf.geometry_offsets[-1]) if d_buf.geometry_offsets.size > 0 else 0
            )
            ring_offset_arrays.append(d_buf.ring_offsets)
            ring_counts.append(
                int(d_buf.ring_offsets[-1]) if d_buf.ring_offsets.size > 0 else 0
            )

    if GeometryFamily.MULTIPOLYGON in families:
        d_buf = ds.families[GeometryFamily.MULTIPOLYGON]
        if d_buf.x.size > 0:
            all_x_parts.append(d_buf.x)
            all_y_parts.append(d_buf.y)
            # MultiPolygon has 3 levels: geom -> part -> ring -> coord.
            part_offset_arrays.append(d_buf.part_offsets)
            part_counts.append(
                int(d_buf.part_offsets[-1]) if d_buf.part_offsets.size > 0 else 0
            )
            ring_offset_arrays.append(d_buf.ring_offsets)
            ring_counts.append(
                int(d_buf.ring_offsets[-1]) if d_buf.ring_offsets.size > 0 else 0
            )

    if not all_x_parts:
        return _empty_result()

    merged_x = cp.concatenate(all_x_parts) if len(all_x_parts) > 1 else all_x_parts[0]
    merged_y = cp.concatenate(all_y_parts) if len(all_y_parts) > 1 else all_y_parts[0]

    # Chain ring_offsets.
    merged_ring_offsets = _chain_device_offset_arrays(ring_offset_arrays, ring_counts)

    # Chain part_offsets (polygon -> ring), shifting by cumulative ring counts.
    merged_part_ring_counts = [
        int(po[-1]) if po.size > 0 else 0 for po in part_offset_arrays
    ]
    merged_part_offsets = _chain_device_offset_arrays(
        part_offset_arrays, merged_part_ring_counts,
    )
    total_parts = int(merged_part_offsets.size) - 1

    # Single-row geometry_offsets = [0, total_parts]
    geom_offsets = cp.array([0, total_parts], dtype=cp.int32)
    empty_mask = cp.array([total_parts == 0], dtype=cp.bool_)

    out_family = GeometryFamily.MULTIPOLYGON
    d_buf = DeviceFamilyGeometryBuffer(
        family=out_family,
        x=merged_x,
        y=merged_y,
        geometry_offsets=geom_offsets,
        empty_mask=empty_mask,
        part_offsets=merged_part_offsets,
        ring_offsets=merged_ring_offsets,
    )

    return _build_single_row_oga(out_family, d_buf)


# ---------------------------------------------------------------------------
# Device offset chaining helper
# ---------------------------------------------------------------------------


def _chain_device_offset_arrays(
    offset_arrays: list,
    element_counts: list[int],
) -> cp.ndarray:
    """Chain multiple offset arrays into one, shifting by cumulative counts.

    Same logic as ``_concat_device_offset_arrays`` in ``geometry/owned.py``:
    drop the leading zero from all arrays after the first and shift values
    so they form a single contiguous offset array.  All work stays on device.
    """
    if len(offset_arrays) == 1:
        return offset_arrays[0]
    parts = [offset_arrays[0]]
    cumulative = element_counts[0]
    for offsets, count in zip(offset_arrays[1:], element_counts[1:], strict=True):
        parts.append(offsets[1:] + cumulative)
        cumulative += count
    return cp.concatenate(parts).astype(cp.int32)


# ---------------------------------------------------------------------------
# Build single-row OGA from a device family buffer
# ---------------------------------------------------------------------------


def _build_single_row_oga(
    family: GeometryFamily,
    d_buf: DeviceFamilyGeometryBuffer,
) -> OwnedGeometryArray:
    """Build a 1-row device-resident OwnedGeometryArray from a single device buffer."""
    tag_value = FAMILY_TAGS[family]
    tags = np.array([tag_value], dtype=np.int8)
    validity = np.array([True], dtype=np.bool_)
    family_row_offsets = np.array([0], dtype=np.int32)

    return build_device_resident_owned(
        device_families={family: d_buf},
        row_count=1,
        tags=tags,
        validity=validity,
        family_row_offsets=family_row_offsets,
    )


# ---------------------------------------------------------------------------
# Empty result helper
# ---------------------------------------------------------------------------


def _empty_result() -> OwnedGeometryArray:
    """Produce an empty geometry as a 1-row OGA.

    OwnedGeometryArray does not support GeometryCollection, so we use an
    empty Polygon (consistent with ``_get_empty_owned`` in
    ``segmented_union.py``).
    """
    from vibespatial.geometry.owned import from_shapely_geometries

    return from_shapely_geometries([shapely.Polygon()])


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "disjoint_subset_union_all",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("shapely", "constructive", "union_all"),
)
def _disjoint_subset_union_all_cpu(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """CPU fallback: assemble disjoint geometries via host-side buffer concatenation.

    Returns ``None`` when the result is a GeometryCollection (mixed families)
    since OwnedGeometryArray does not support that type.  The caller should
    fall through to the Shapely path in that case.
    """
    owned._ensure_host_state()

    # Determine which families have valid (non-empty) rows.
    valid_families: set[GeometryFamily] = set()
    for family, buf in owned.families.items():
        if buf.row_count > 0:
            valid_families.add(family)

    if not valid_families:
        return _empty_result()

    # Check if all families merge to the same multi-type.
    target_families = {_MERGE_TARGETS.get(f) for f in valid_families}
    if len(target_families) != 1 or None in target_families:
        # Mixed families -> return None, caller uses Shapely directly.
        return None

    target_family = target_families.pop()

    # Dispatch to family-specific host-side assembly.
    if target_family is GeometryFamily.MULTIPOINT:
        return _assemble_multipoint_host(owned, valid_families)
    elif target_family is GeometryFamily.MULTILINESTRING:
        return _assemble_multilinestring_host(owned, valid_families)
    elif target_family is GeometryFamily.MULTIPOLYGON:
        return _assemble_multipolygon_host(owned, valid_families)
    else:
        return None


# ---------------------------------------------------------------------------
# Host-side assembly helpers (numpy)
# ---------------------------------------------------------------------------


def _chain_host_offset_arrays(
    offset_arrays: list[np.ndarray],
    element_counts: list[int],
) -> np.ndarray:
    """Chain multiple offset arrays on host, same logic as ``_chain_device_offset_arrays``."""
    if len(offset_arrays) == 1:
        return offset_arrays[0]
    parts = [offset_arrays[0]]
    cumulative = element_counts[0]
    for offsets, count in zip(offset_arrays[1:], element_counts[1:], strict=True):
        parts.append(offsets[1:] + cumulative)
        cumulative += count
    return np.concatenate(parts).astype(np.int32)


def _build_single_row_oga_host(
    family: GeometryFamily,
    buf: FamilyGeometryBuffer,
) -> OwnedGeometryArray:
    """Build a 1-row host-resident OwnedGeometryArray from a single family buffer."""
    tag_value = FAMILY_TAGS[family]
    tags = np.array([tag_value], dtype=np.int8)
    validity = np.array([True], dtype=np.bool_)
    family_row_offsets = np.array([0], dtype=np.int32)

    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={family: buf},
        residency=Residency.HOST,
    )


def _assemble_multipoint_host(
    owned: OwnedGeometryArray,
    families: set[GeometryFamily],
) -> OwnedGeometryArray:
    """Assemble Point/MultiPoint rows into a single MultiPoint on host."""
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
        if family not in families:
            continue
        buf = owned.families[family]
        if buf.x.size > 0:
            all_x.append(buf.x)
            all_y.append(buf.y)

    if not all_x:
        return _empty_result()

    merged_x = np.concatenate(all_x) if len(all_x) > 1 else all_x[0]
    merged_y = np.concatenate(all_y) if len(all_y) > 1 else all_y[0]
    total_coords = int(merged_x.size)

    out_family = GeometryFamily.MULTIPOINT
    schema = get_geometry_buffer_schema(out_family)
    result_buf = FamilyGeometryBuffer(
        family=out_family,
        schema=schema,
        row_count=1,
        x=merged_x,
        y=merged_y,
        geometry_offsets=np.array([0, total_coords], dtype=np.int32),
        empty_mask=np.array([total_coords == 0], dtype=np.bool_),
    )
    return _build_single_row_oga_host(out_family, result_buf)


def _assemble_multilinestring_host(
    owned: OwnedGeometryArray,
    families: set[GeometryFamily],
) -> OwnedGeometryArray:
    """Assemble LineString/MultiLineString rows into a single MultiLineString on host."""
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    part_offset_arrays: list[np.ndarray] = []
    part_counts: list[int] = []

    if GeometryFamily.LINESTRING in families:
        buf = owned.families[GeometryFamily.LINESTRING]
        if buf.x.size > 0:
            all_x.append(buf.x)
            all_y.append(buf.y)
            part_offset_arrays.append(buf.geometry_offsets)
            part_counts.append(
                int(buf.geometry_offsets[-1]) if buf.geometry_offsets.size > 0 else 0
            )

    if GeometryFamily.MULTILINESTRING in families:
        buf = owned.families[GeometryFamily.MULTILINESTRING]
        if buf.x.size > 0:
            all_x.append(buf.x)
            all_y.append(buf.y)
            part_offset_arrays.append(buf.part_offsets)
            part_counts.append(
                int(buf.part_offsets[-1]) if buf.part_offsets.size > 0 else 0
            )

    if not all_x:
        return _empty_result()

    merged_x = np.concatenate(all_x) if len(all_x) > 1 else all_x[0]
    merged_y = np.concatenate(all_y) if len(all_y) > 1 else all_y[0]
    merged_part_offsets = _chain_host_offset_arrays(part_offset_arrays, part_counts)
    total_parts = int(merged_part_offsets.size) - 1

    out_family = GeometryFamily.MULTILINESTRING
    schema = get_geometry_buffer_schema(out_family)
    result_buf = FamilyGeometryBuffer(
        family=out_family,
        schema=schema,
        row_count=1,
        x=merged_x,
        y=merged_y,
        geometry_offsets=np.array([0, total_parts], dtype=np.int32),
        empty_mask=np.array([total_parts == 0], dtype=np.bool_),
        part_offsets=merged_part_offsets,
    )
    return _build_single_row_oga_host(out_family, result_buf)


def _assemble_multipolygon_host(
    owned: OwnedGeometryArray,
    families: set[GeometryFamily],
) -> OwnedGeometryArray:
    """Assemble Polygon/MultiPolygon rows into a single MultiPolygon on host."""
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    ring_offset_arrays: list[np.ndarray] = []
    ring_counts: list[int] = []
    part_offset_arrays: list[np.ndarray] = []
    part_counts: list[int] = []

    if GeometryFamily.POLYGON in families:
        buf = owned.families[GeometryFamily.POLYGON]
        if buf.x.size > 0:
            all_x.append(buf.x)
            all_y.append(buf.y)
            part_offset_arrays.append(buf.geometry_offsets)
            part_counts.append(
                int(buf.geometry_offsets[-1]) if buf.geometry_offsets.size > 0 else 0
            )
            ring_offset_arrays.append(buf.ring_offsets)
            ring_counts.append(
                int(buf.ring_offsets[-1]) if buf.ring_offsets.size > 0 else 0
            )

    if GeometryFamily.MULTIPOLYGON in families:
        buf = owned.families[GeometryFamily.MULTIPOLYGON]
        if buf.x.size > 0:
            all_x.append(buf.x)
            all_y.append(buf.y)
            part_offset_arrays.append(buf.part_offsets)
            part_counts.append(
                int(buf.part_offsets[-1]) if buf.part_offsets.size > 0 else 0
            )
            ring_offset_arrays.append(buf.ring_offsets)
            ring_counts.append(
                int(buf.ring_offsets[-1]) if buf.ring_offsets.size > 0 else 0
            )

    if not all_x:
        return _empty_result()

    merged_x = np.concatenate(all_x) if len(all_x) > 1 else all_x[0]
    merged_y = np.concatenate(all_y) if len(all_y) > 1 else all_y[0]
    merged_ring_offsets = _chain_host_offset_arrays(ring_offset_arrays, ring_counts)
    merged_part_ring_counts = [
        int(po[-1]) if po.size > 0 else 0 for po in part_offset_arrays
    ]
    merged_part_offsets = _chain_host_offset_arrays(
        part_offset_arrays, merged_part_ring_counts,
    )
    total_parts = int(merged_part_offsets.size) - 1

    out_family = GeometryFamily.MULTIPOLYGON
    schema = get_geometry_buffer_schema(out_family)
    result_buf = FamilyGeometryBuffer(
        family=out_family,
        schema=schema,
        row_count=1,
        x=merged_x,
        y=merged_y,
        geometry_offsets=np.array([0, total_parts], dtype=np.int32),
        empty_mask=np.array([total_parts == 0], dtype=np.bool_),
        part_offsets=merged_part_offsets,
        ring_offsets=merged_ring_offsets,
    )
    return _build_single_row_oga_host(out_family, result_buf)
