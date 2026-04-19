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

from vibespatial.constructive.binary_constructive_cpu import binary_constructive_cpu
from vibespatial.constructive.union_all_cpu import (
    empty_owned,
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
from vibespatial.runtime.residency import (
    Residency,
    TransferTrigger,
    combined_residency,
)

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


def _polygon_assembly_result_is_invalid(result: OwnedGeometryArray) -> bool:
    valid_tags = np.asarray(result.tags[result.validity], dtype=np.int8)
    polygon_tags = np.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=valid_tags.dtype if valid_tags.size else np.int8,
    )
    if not valid_tags.size or not np.all(np.isin(valid_tags, polygon_tags)):
        return False

    from vibespatial.constructive.validity import is_valid_owned

    validity = np.asarray(is_valid_owned(result), dtype=bool)
    if not bool(np.all(result.validity)):
        validity = validity.copy()
        validity[~result.validity] = True
    return not bool(np.all(validity))


def _polygon_inputs_have_bbox_interactions_requiring_exact_union(
    owned: OwnedGeometryArray,
) -> bool:
    valid_tags = np.asarray(owned.tags[owned.validity], dtype=np.int8)
    polygon_tags = np.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=valid_tags.dtype if valid_tags.size else np.int8,
    )
    if not valid_tags.size or not np.all(np.isin(valid_tags, polygon_tags)):
        return False
    if int(np.count_nonzero(owned.validity)) <= 1:
        return False

    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

    bounds = np.asarray(
        compute_geometry_bounds(
            owned,
            dispatch_mode=(
                ExecutionMode.GPU
                if cp is not None and owned.residency is Residency.DEVICE
                else ExecutionMode.CPU
            ),
        ),
        dtype=np.float64,
    )
    bounds = bounds[np.asarray(owned.validity, dtype=bool)]
    finite = np.isfinite(bounds).all(axis=1)
    bounds = bounds[finite]
    if bounds.shape[0] <= 1:
        return False

    order = np.argsort(bounds[:, 0], kind="stable")
    active: list[int] = []
    for current in order.astype(np.intp, copy=False):
        xmin = bounds[current, 0]
        ymin = bounds[current, 1]
        ymax = bounds[current, 3]
        active = [other for other in active if bounds[other, 2] >= xmin]
        for other in active:
            if bounds[other, 1] <= ymax and bounds[other, 3] >= ymin:
                return True
        active.append(int(current))
    return False


def _polygonal_family_only(owned: OwnedGeometryArray) -> bool:
    """Return True when all valid rows are polygon-family geometries."""
    valid_tags = np.asarray(owned.tags[owned.validity])
    if valid_tags.size == 0:
        return True
    polygon_tags = np.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=valid_tags.dtype,
    )
    return bool(np.all(np.isin(valid_tags, polygon_tags)))


def _compute_union_bounds_host(owned: OwnedGeometryArray) -> np.ndarray | None:
    try:
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

        return np.asarray(
            compute_geometry_bounds(
                owned,
                dispatch_mode=(
                    ExecutionMode.GPU
                    if cp is not None and owned.residency is Residency.DEVICE
                    else ExecutionMode.CPU
                ),
            ),
            dtype=np.float64,
        )
    except Exception:
        logger.debug("polygon union bounds computation failed", exc_info=True)
        return None


def _bbox_overlap_components(bounds: np.ndarray) -> list[np.ndarray]:
    row_count = int(bounds.shape[0])
    if row_count == 0:
        return []

    parent = np.arange(row_count, dtype=np.int32)

    def _find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    def _union(left: int, right: int) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    order = np.argsort(bounds[:, 0], kind="stable")
    active: list[int] = []
    for current in order.astype(np.intp, copy=False):
        xmin = bounds[current, 0]
        ymin = bounds[current, 1]
        ymax = bounds[current, 3]
        active = [other for other in active if bounds[other, 2] >= xmin]
        for other in active:
            if bounds[other, 1] <= ymax and bounds[other, 3] >= ymin:
                _union(int(current), int(other))
        active.append(int(current))

    groups: dict[int, list[int]] = {}
    for row in range(row_count):
        groups.setdefault(_find(row), []).append(row)
    return [np.asarray(rows, dtype=np.int64) for rows in groups.values()]


def _bbox_disjoint_color_groups(bounds: np.ndarray) -> list[np.ndarray]:
    row_count = int(bounds.shape[0])
    if row_count == 0:
        return []

    order = np.argsort(bounds[:, 0], kind="stable")
    color_bounds: list[list[int]] = []
    color_rows: list[list[int]] = []

    for row in order.astype(np.intp, copy=False):
        xmin, ymin, xmax, ymax = bounds[row]
        selected_color: int | None = None
        for color_index, active_bounds in enumerate(color_bounds):
            conflicts = False
            kept_bounds: list[int] = []
            for other in active_bounds:
                other_bounds = bounds[int(other)]
                if other_bounds[2] >= xmin:
                    kept_bounds.append(other)
                    if other_bounds[1] <= ymax and other_bounds[3] >= ymin:
                        conflicts = True
            color_bounds[color_index] = kept_bounds
            if not conflicts:
                selected_color = color_index
                break
        if selected_color is None:
            color_bounds.append([])
            color_rows.append([])
            selected_color = len(color_rows) - 1
        color_bounds[selected_color].append(int(row))
        color_rows[selected_color].append(int(row))

    return [np.asarray(rows, dtype=np.int64) for rows in color_rows if rows]


def _spatially_localize_polygon_union_inputs(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Reorder polygon rows so nearby inputs union in early tree-reduce rounds.

    Arbitrary input order can create large, fragmented intermediates very early
    in the exact-union tree reduction. Sorting by coarse spatial position keeps
    the first reduction rounds local, which substantially reduces downstream
    overlay complexity on corridor/network workloads.
    """
    if cp is None or owned.row_count <= 2 or not _polygonal_family_only(owned):
        return owned

    try:
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

        bounds = cp.asarray(compute_geometry_bounds_device(owned))
        if int(bounds.shape[0]) != owned.row_count:
            return owned
        order = cp.lexsort(cp.stack([bounds[:, 1], bounds[:, 0]])).astype(cp.int64)
        return owned.take(order)
    except Exception:
        logger.debug(
            "spatially local polygon union ordering failed; keeping input order",
            exc_info=True,
        )
        return owned


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
        current_residency=combined_residency(owned),
    )

    if _polygon_inputs_have_bbox_interactions_requiring_exact_union(owned):
        try:
            result = union_all_gpu_owned(
                owned,
                dispatch_mode=dispatch_mode,
                precision=precision,
            )
            record_dispatch_event(
                surface="constructive.disjoint_subset_union_all",
                operation="disjoint_subset_union_all",
                implementation="exact_union_for_interacting_polygon_subsets",
                reason=(
                    "polygon bounding boxes touch or overlap; routed to exact "
                    "GPU union instead of disjoint assembly"
                ),
                requested=dispatch_mode,
                selected=(
                    ExecutionMode.GPU
                    if result.residency is Residency.DEVICE
                    else ExecutionMode.CPU
                ),
            )
            return result
        except Exception:
            logger.debug(
                "exact disjoint-subset polygon union failed, falling back to CPU",
                exc_info=True,
            )
            record_fallback_event(
                surface="constructive.disjoint_subset_union_all",
                reason=(
                    "polygon bounding boxes touch or overlap; exact "
                    "disjoint-subset union required"
                ),
                requested=dispatch_mode,
                selected=ExecutionMode.CPU,
                d2h_transfer=owned.residency is Residency.DEVICE,
            )
            return None

    if selection.selected is ExecutionMode.GPU and cp is not None:
        precision_plan = selection.precision_plan  # noqa: F841 — called for ADR-0002 observability side-effects
        try:
            result = _disjoint_subset_union_all_gpu(owned)
            if result is not None:
                if _polygon_assembly_result_is_invalid(result):
                    from vibespatial.constructive.make_valid_gpu import (
                        gpu_repair_invalid_polygons,
                    )

                    repair = gpu_repair_invalid_polygons(
                        result,
                        np.arange(result.row_count, dtype=np.int32),
                    )
                    if (
                        repair is not None
                        and repair.repaired_owned is not None
                        and repair.still_invalid_rows.size == 0
                    ):
                        result = repair.repaired_owned
                if _polygon_assembly_result_is_invalid(result):
                    _gpu_fallback_reason = "GPU disjoint-subset assembly produced invalid polygon topology"
                    result = None
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
                # Invalid assembled polygon topology needs an exact GEOS path,
                # not the host-side assembly fallback below.
                record_fallback_event(
                    surface="constructive.disjoint_subset_union_all",
                    reason=_gpu_fallback_reason,
                    requested=dispatch_mode,
                    selected=ExecutionMode.CPU,
                    d2h_transfer=owned.residency is Residency.DEVICE,
                )
                return None
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
        if _polygon_assembly_result_is_invalid(result):
            record_fallback_event(
                surface="constructive.disjoint_subset_union_all",
                reason="host disjoint-subset assembly produced invalid polygon topology",
                requested=dispatch_mode,
                selected=ExecutionMode.CPU,
                d2h_transfer=owned.residency is Residency.DEVICE,
            )
            return None
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


def _owned_valid_nonempty_mask(
    owned: OwnedGeometryArray,
) -> np.ndarray | cp.ndarray:
    """Return a validity-and-nonempty mask without forcing host geometry materialization."""
    if cp is not None and owned.device_state is not None:
        ds = owned._ensure_device_state()
        validity = cp.asarray(ds.validity).astype(cp.bool_, copy=True)
        if int(validity.size) == 0:
            return validity

        tags = cp.asarray(ds.tags)
        row_offsets = cp.asarray(ds.family_row_offsets)
        keep_mask = validity.copy()

        for family, device_buf in ds.families.items():
            family_mask = validity & (tags == FAMILY_TAGS[family])
            if not bool(cp.any(family_mask)):
                continue
            family_rows = row_offsets[family_mask]
            keep_mask[family_mask] = ~device_buf.empty_mask[family_rows]

        return keep_mask

    validity = np.asarray(owned.validity, dtype=bool)
    if not validity.any():
        return validity

    tags = np.asarray(owned.tags)
    row_offsets = np.asarray(owned.family_row_offsets)
    keep_mask = validity.copy()

    for family in owned.families:
        owned._ensure_host_family_structure(family)
        family_mask = validity & (tags == FAMILY_TAGS[family])
        if not family_mask.any():
            continue
        family_rows = row_offsets[family_mask]
        keep_mask[family_mask] = ~np.asarray(
            owned.families[family].empty_mask[family_rows],
            dtype=bool,
        )

    return keep_mask


def _all_owned_rows_nonempty(owned: OwnedGeometryArray) -> bool:
    """Return True when every valid output row is non-empty."""
    keep_mask = _owned_valid_nonempty_mask(owned)
    if cp is not None and hasattr(keep_mask, "__cuda_array_interface__"):
        return bool(cp.all(keep_mask).item())
    return bool(np.all(np.asarray(keep_mask, dtype=bool)))


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
    from vibespatial.constructive.binary_constructive import binary_constructive_owned

    if cp is not None:
        owned = owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=f"{op}_all GPU tree reduction",
        )
        xp = cp
    else:
        xp = np

    current = owned

    rounds = 0
    max_rounds = int(math.ceil(math.log2(max(current.row_count, 2)))) + 2
    consecutive_gpu_failures = 0
    while current.row_count > 1 and rounds < max_rounds:
        pair_count = current.row_count // 2
        left_indices = (xp.arange(pair_count, dtype=xp.int64) * 2)
        right_indices = left_indices + 1

        carry_row: OwnedGeometryArray | None = None
        if current.row_count % 2:
            carry_index = xp.asarray([current.row_count - 1], dtype=xp.int64)
            carry_row = current.take(carry_index)

        left_batch = current.take(left_indices)
        right_batch = current.take(right_indices)

        gpu_ok = False
        if consecutive_gpu_failures < OVERLAY_GPU_FAILURE_THRESHOLD:
            try:
                next_round = binary_constructive_owned(
                    op,
                    left_batch,
                    right_batch,
                    dispatch_mode=ExecutionMode.GPU,
                )
                gpu_ok = True
                consecutive_gpu_failures = 0
            except Exception:
                consecutive_gpu_failures += 1

        if not gpu_ok:
            next_round = binary_constructive_cpu(op, left_batch, right_batch)

        if early_termination_on_empty and not _all_owned_rows_nonempty(next_round):
            return empty_owned()

        if carry_row is not None:
            reduced = OwnedGeometryArray.concat([next_round, carry_row])
        else:
            reduced = next_round

        del current, left_batch, right_batch, next_round, carry_row
        current = reduced
        rounds += 1

    return current


def _try_exact_union_disjoint_bbox_components(
    owned: OwnedGeometryArray,
    *,
    precision: PrecisionMode | str,
) -> OwnedGeometryArray | None:
    if owned.row_count <= 2 or not _polygonal_family_only(owned):
        return None

    bounds = _compute_union_bounds_host(owned)
    if bounds is None or int(bounds.shape[0]) != owned.row_count:
        return None
    finite = np.isfinite(bounds).all(axis=1)
    if not bool(np.all(finite)):
        return None

    components = _bbox_overlap_components(bounds)
    if len(components) <= 1:
        return None

    partials: list[OwnedGeometryArray] = []
    for rows in components:
        component = owned.take(rows.astype(np.int64, copy=False))
        if component.row_count == 1:
            partials.append(component)
            continue
        partials.append(_tree_reduce_global(component, "union"))

    reduced = disjoint_subset_union_all_owned(
        OwnedGeometryArray.concat(partials),
        dispatch_mode=ExecutionMode.GPU,
        precision=precision,
    )
    if reduced is not None:
        return reduced
    return _tree_reduce_global(OwnedGeometryArray.concat(partials), "union")


def _try_exact_union_bbox_disjoint_color_subsets(
    owned: OwnedGeometryArray,
    *,
    precision: PrecisionMode | str,
) -> OwnedGeometryArray | None:
    if owned.row_count < 16 or not _polygonal_family_only(owned):
        return None

    bounds = _compute_union_bounds_host(owned)
    if bounds is None or int(bounds.shape[0]) != owned.row_count:
        return None
    finite = np.isfinite(bounds).all(axis=1)
    if not bool(np.all(finite)):
        return None

    color_groups = _bbox_disjoint_color_groups(bounds)
    if len(color_groups) <= 1 or len(color_groups) >= owned.row_count:
        return None
    if len(color_groups) * 4 > owned.row_count:
        return None

    partials: list[OwnedGeometryArray] = []
    for rows in color_groups:
        subset = owned.take(rows.astype(np.int64, copy=False))
        partial = disjoint_subset_union_all_owned(
            subset,
            dispatch_mode=ExecutionMode.GPU,
            precision=precision,
        )
        if partial is None:
            return None
        partials.append(partial)

    localized = _spatially_localize_polygon_union_inputs(OwnedGeometryArray.concat(partials))
    return _tree_reduce_global(localized, "union")


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
        current_residency=combined_residency(owned),
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
            owned = _spatially_localize_polygon_union_inputs(owned)
            result = _try_exact_union_disjoint_bbox_components(
                owned,
                precision=precision,
            )
            if result is not None:
                record_dispatch_event(
                    surface="constructive.union_all_gpu",
                    operation="union_all",
                    implementation="gpu_bbox_component_decomposition",
                    reason=selection.reason,
                    detail=(
                        f"rows={row_count}, "
                        f"precision={precision_plan.compute_precision.value}"
                    ),
                    requested=selection.requested,
                    selected=ExecutionMode.GPU,
                )
                return result
            result = _try_exact_union_bbox_disjoint_color_subsets(
                owned,
                precision=precision,
            )
            if result is not None:
                record_dispatch_event(
                    surface="constructive.union_all_gpu",
                    operation="union_all",
                    implementation="gpu_bbox_disjoint_color_compression",
                    reason=selection.reason,
                    detail=(
                        f"rows={row_count}, "
                        f"precision={precision_plan.compute_precision.value}"
                    ),
                    requested=selection.requested,
                    selected=ExecutionMode.GPU,
                )
                return result
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
        current_residency=combined_residency(owned),
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
            owned = _spatially_localize_polygon_union_inputs(owned)
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
        current_residency=combined_residency(owned),
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
