"""GPU-accelerated union/intersection reduction kernels.

Provides:
  - disjoint_subset_union_all: geometry collection assembly (no Boolean ops)
  - union_all_gpu: tree-reduction global union via pairwise overlay_union
  - coverage_union_all_gpu: coverage-optimized union (non-overlapping input)
  - intersection_all_gpu: tree-reduction global intersection with early termination
  - unary_union_gpu: thin wrapper -> union_all_gpu

For disjoint_subset, geometries are disjoint by assumption, so union is just
geometry collection assembly -- concatenate coordinate buffers and chain offset
arrays.  No Boolean geometry operations are performed.

For homogeneous input (all same family), the output is the corresponding Multi*
family:
  - Point -> MultiPoint
  - LineString -> MultiLineString
  - Polygon -> MultiPolygon
  - Multi* -> Multi* (merge all parts into a single row)

For mixed families, falls back to Shapely CPU path.

ADR-0002: CONSTRUCTIVE class -- fp64, no precision downgrade (coordinates are
          exact subsets, no new coordinates created).
ADR-0033: Disjoint subset: pure CuPy buffer manipulation (Tier 2).
          Tree-reduction: orchestrates Tier 1 overlay pipeline pairwise.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.union_all_cpu import (
    empty_owned,
    merge_pair_cpu,
    reduce_all_cpu,
)
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
from vibespatial.runtime.config import OVERLAY_GPU_FAILURE_THRESHOLD
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
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
        return empty_owned()

    selection = plan_dispatch_selection(
        kernel_name="disjoint_subset_union_all",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    if selection.selected is ExecutionMode.GPU and cp is not None:
        precision_plan = selection.precision_plan  # noqa: F841 — called for ADR-0002 observability side-effects
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
            # Mixed families -- GPU path returns None; fall through to
            # CPU path below.  Fallback event is recorded AFTER CPU
            # execution succeeds, not before.
            _gpu_fallback_reason = "mixed families not supported on GPU"
        except Exception:
            logger.debug(
                "disjoint_subset_union_all GPU failed, falling back to CPU",
                exc_info=True,
            )
            _gpu_fallback_reason = "GPU exception"
    else:
        _gpu_fallback_reason = "GPU not selected"

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
            reason=_gpu_fallback_reason,
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
        return empty_owned()

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
        return empty_owned()

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
        return empty_owned()

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
        return empty_owned()

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
    tags = cp.asarray([tag_value], dtype=cp.int8)
    validity = cp.asarray([True], dtype=cp.bool_)
    family_row_offsets = cp.asarray([0], dtype=cp.int32)

    return build_device_resident_owned(
        device_families={family: d_buf},
        row_count=1,
        tags=tags,
        validity=validity,
        family_row_offsets=family_row_offsets,
        execution_mode="gpu",
    )


# ---------------------------------------------------------------------------
# Empty result helper
# ---------------------------------------------------------------------------


def _empty_result() -> OwnedGeometryArray:
    """Produce an empty geometry as a 1-row OGA."""
    return empty_owned()


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
        return empty_owned()

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
        return empty_owned()

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
        return empty_owned()

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
        return empty_owned()

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


# ===========================================================================
# GPU tree-reduction global set operations
# ===========================================================================
#
# All four operations share the _tree_reduce_global helper which performs
# log2(N) rounds of pairwise binary constructive ops, keeping intermediate
# results on device.  The binary op itself is pluggable (union or intersection).
#
# ADR-0002: CONSTRUCTIVE class -- fp64 on all devices per policy.
# ADR-0033: Orchestrates Tier 1 overlay pipeline pairwise (no new NVRTC kernel).
# ===========================================================================


def _is_owned_empty(owned: OwnedGeometryArray) -> bool:
    """Check if a 1-row OGA represents an empty geometry.

    Fast check: if no family buffer has coordinates, the geometry is empty.
    Falls back to validity check if families are present.
    """
    if owned.row_count == 0:
        return True
    # Check validity first -- all-invalid means empty.
    if not np.any(owned.validity):
        return True
    # Check if all family buffers are coordinate-less.
    for buf in owned.families.values():
        if buf.x.size > 0:
            return False
    return True


def _tree_reduce_global(
    owned: OwnedGeometryArray,
    op: str,
    *,
    early_termination_on_empty: bool = False,
) -> OwnedGeometryArray:
    """Binary-tree reduction of all rows in *owned* via overlay pipeline.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Multi-row input.  Must have row_count >= 2.
    op : str
        "union" or "intersection".
    early_termination_on_empty : bool
        If True, stop reduction early when an intermediate result is empty.
        Used by intersection_all (since A inter empty = empty).

    Returns
    -------
    OwnedGeometryArray
        Single-row result.
    """
    from vibespatial.overlay.gpu import (
        overlay_intersection_owned,
        overlay_union_owned,
    )

    overlay_fn = overlay_union_owned if op == "union" else overlay_intersection_owned

    # Split into single-row OwnedGeometryArrays for pairwise reduction.
    current: list[OwnedGeometryArray] = [
        owned.take(np.array([i], dtype=np.intp))
        for i in range(owned.row_count)
    ]

    rounds = 0
    max_rounds = int(math.ceil(math.log2(max(len(current), 2)))) + 2
    consecutive_gpu_failures = 0
    while len(current) > 1 and rounds < max_rounds:
        next_round: list[OwnedGeometryArray] = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                gpu_ok = False
                if consecutive_gpu_failures < OVERLAY_GPU_FAILURE_THRESHOLD:
                    try:
                        merged = overlay_fn(
                            current[i],
                            current[i + 1],
                            dispatch_mode=ExecutionMode.GPU,
                        )
                        next_round.append(merged)
                        gpu_ok = True
                        consecutive_gpu_failures = 0
                    except Exception:
                        consecutive_gpu_failures += 1

                if not gpu_ok:
                    # CPU fallback for this pair.
                    next_round.append(merge_pair_cpu(current[i], current[i + 1], op=op))

                # Early termination: if result is empty and op is intersection,
                # the final result must be empty regardless of remaining elements.
                if early_termination_on_empty and _is_owned_empty(next_round[-1]):
                    return empty_owned()
            else:
                # Odd element passes through.
                next_round.append(current[i])

        # Explicit cleanup: release previous round's intermediates promptly
        # to avoid accumulating device memory across reduction rounds.
        del current
        current = next_round
        rounds += 1

    return current[0]


# ---------------------------------------------------------------------------
# union_all_gpu
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "union_all_gpu",
    "gpu-tree-reduce",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("constructive", "union_all", "gpu", "tree-reduce"),
)
def union_all_gpu_owned(
    owned: OwnedGeometryArray,
    *,
    grid_size: float | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """GPU tree-reduction global union of all rows into a single geometry.

    When *grid_size* is set, applies set_precision to the input first
    (snapping coordinates to the grid) before performing the union.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries (device- or host-resident).
    grid_size : float or None
        Optional snap grid size.  Applied before union via set_precision_owned.
    dispatch_mode : ExecutionMode or str
        Execution mode hint.
    precision : PrecisionMode or str
        Precision mode.  CONSTRUCTIVE stays fp64 per ADR-0002.

    Returns
    -------
    OwnedGeometryArray
        Single-row OGA containing the global union.
    """
    row_count = owned.row_count

    # Empty input -> empty geometry.
    if row_count == 0:
        return empty_owned()

    selection = plan_dispatch_selection(
        kernel_name="union_all_gpu",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    precision_plan = selection.precision_plan

    # Apply grid_size snapping if requested.
    if grid_size is not None and grid_size > 0:
        from vibespatial.constructive.set_precision import set_precision_owned

        owned = set_precision_owned(owned, grid_size, mode="valid_output")

    # Filter out null rows.
    keep = np.flatnonzero(owned.validity)
    if keep.size == 0:
        return empty_owned()
    if keep.size < owned.row_count:
        owned = owned.take(keep)

    # Single valid row -> identity.
    if owned.row_count == 1:
        record_dispatch_event(
            surface="constructive.union_all_gpu",
            operation="union_all",
            implementation="identity",
            reason="single row",
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
        )
        return owned

    if selection.selected is ExecutionMode.GPU and cp is not None:
        try:
            result = _tree_reduce_global(owned, "union")
            record_dispatch_event(
                surface="constructive.union_all_gpu",
                operation="union_all",
                implementation="gpu_tree_reduce_overlay",
                reason=selection.reason,
                detail=(
                    f"rows={row_count}, "
                    f"precision={precision_plan.compute_precision.value}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result
        except Exception:
            logger.debug(
                "union_all_gpu tree reduction failed, falling back to CPU",
                exc_info=True,
            )

    # CPU fallback: Shapely union_all.
    record_dispatch_event(
        surface="constructive.union_all_gpu",
        operation="union_all",
        implementation="shapely_cpu",
        reason="GPU fallback",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
    )
    return reduce_all_cpu(owned, op="union_all", grid_size=grid_size)


# ---------------------------------------------------------------------------
# coverage_union_all_gpu
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "coverage_union_all_gpu",
    "gpu-tree-reduce",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    tags=("constructive", "coverage_union_all", "gpu", "tree-reduce"),
)
def coverage_union_all_gpu_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """GPU coverage-optimized union for non-overlapping input.

    Since the input is assumed to be non-overlapping (coverage property),
    the binary union of any pair will not produce new intersection vertices.
    This uses the same tree-reduction as union_all_gpu; the coverage
    property simply means the overlay is cheaper (no self-intersections
    to resolve).

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input non-overlapping geometries (device- or host-resident).
    dispatch_mode : ExecutionMode or str
        Execution mode hint.
    precision : PrecisionMode or str
        Precision mode.  CONSTRUCTIVE stays fp64 per ADR-0002.

    Returns
    -------
    OwnedGeometryArray
        Single-row OGA containing the coverage union.
    """
    row_count = owned.row_count

    if row_count == 0:
        return empty_owned()

    selection = plan_dispatch_selection(
        kernel_name="coverage_union_all_gpu",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    precision_plan = selection.precision_plan

    # Filter nulls.
    keep = np.flatnonzero(owned.validity)
    if keep.size == 0:
        return empty_owned()
    if keep.size < owned.row_count:
        owned = owned.take(keep)

    if owned.row_count == 1:
        record_dispatch_event(
            surface="constructive.coverage_union_all_gpu",
            operation="coverage_union_all",
            implementation="identity",
            reason="single row",
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
        )
        return owned

    if selection.selected is ExecutionMode.GPU and cp is not None:
        try:
            # Coverage union uses the same tree reduction as regular union.
            # The coverage property (non-overlapping input) means the overlay
            # pipeline processes simpler topology, but the algorithm is identical.
            result = _tree_reduce_global(owned, "union")
            record_dispatch_event(
                surface="constructive.coverage_union_all_gpu",
                operation="coverage_union_all",
                implementation="gpu_tree_reduce_overlay",
                reason=selection.reason,
                detail=(
                    f"rows={row_count}, "
                    f"precision={precision_plan.compute_precision.value}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result
        except Exception:
            logger.debug(
                "coverage_union_all_gpu tree reduction failed, falling back to CPU",
                exc_info=True,
            )

    # CPU fallback: Shapely coverage_union_all.
    record_dispatch_event(
        surface="constructive.coverage_union_all_gpu",
        operation="coverage_union_all",
        implementation="shapely_cpu",
        reason="GPU fallback",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
    )
    return reduce_all_cpu(owned, op="coverage_union_all")


# ---------------------------------------------------------------------------
# intersection_all_gpu
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "intersection_all_gpu",
    "gpu-tree-reduce",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("constructive", "intersection_all", "gpu", "tree-reduce"),
)
def intersection_all_gpu_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """GPU tree-reduction global intersection of all rows.

    Early termination: if any intermediate result is empty, returns empty
    immediately (since A intersect empty = empty for all A).

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries (device- or host-resident).
    dispatch_mode : ExecutionMode or str
        Execution mode hint.
    precision : PrecisionMode or str
        Precision mode.  CONSTRUCTIVE stays fp64 per ADR-0002.

    Returns
    -------
    OwnedGeometryArray
        Single-row OGA containing the global intersection.
    """
    row_count = owned.row_count

    if row_count == 0:
        return empty_owned()

    selection = plan_dispatch_selection(
        kernel_name="intersection_all_gpu",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    precision_plan = selection.precision_plan

    # Filter nulls: null rows are skipped (intersection_all of [A, null, B]
    # should be intersection(A, B), matching Shapely semantics).
    keep = np.flatnonzero(owned.validity)
    if keep.size == 0:
        return empty_owned()
    if keep.size < owned.row_count:
        owned = owned.take(keep)

    if owned.row_count == 1:
        record_dispatch_event(
            surface="constructive.intersection_all_gpu",
            operation="intersection_all",
            implementation="identity",
            reason="single row",
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
        )
        return owned

    if selection.selected is ExecutionMode.GPU and cp is not None:
        try:
            result = _tree_reduce_global(
                owned, "intersection", early_termination_on_empty=True,
            )
            record_dispatch_event(
                surface="constructive.intersection_all_gpu",
                operation="intersection_all",
                implementation="gpu_tree_reduce_overlay",
                reason=selection.reason,
                detail=(
                    f"rows={row_count}, "
                    f"precision={precision_plan.compute_precision.value}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result
        except Exception:
            logger.debug(
                "intersection_all_gpu tree reduction failed, falling back to CPU",
                exc_info=True,
            )

    # CPU fallback: Shapely intersection_all.
    record_dispatch_event(
        surface="constructive.intersection_all_gpu",
        operation="intersection_all",
        implementation="shapely_cpu",
        reason="GPU fallback",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
    )
    return reduce_all_cpu(owned, op="intersection_all")


# ---------------------------------------------------------------------------
# unary_union_gpu (thin wrapper)
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "unary_union_gpu",
    "gpu-tree-reduce",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("constructive", "unary_union", "gpu", "tree-reduce"),
)
def unary_union_gpu_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """GPU unary_union -- thin wrapper around union_all_gpu.

    ``unary_union`` is deprecated in favour of ``union_all`` in Shapely 2.x
    but some codebases still use it.  This provides the GPU path.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries (device- or host-resident).
    dispatch_mode : ExecutionMode or str
        Execution mode hint.
    precision : PrecisionMode or str
        Precision mode.

    Returns
    -------
    OwnedGeometryArray
        Single-row OGA containing the global union.
    """
    return union_all_gpu_owned(
        owned,
        grid_size=None,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
