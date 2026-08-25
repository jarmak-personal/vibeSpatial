"""GPU-accelerated binary constructive operations.

Operations: intersection(a,b), union(a,b), difference(a,b), symmetric_difference(a,b)

Element-wise binary constructive operations dispatched per family pair:
- Point-Point: coordinate comparison (Tier 2 CuPy)
- Point-Polygon: PIP kernel for intersection/difference
- Point-LineString: point-on-segment kernel (Tier 1 NVRTC)
- MultiPoint-Polygon: batch PIP + compact
- LineString-Polygon: segment clipping kernel (Tier 1 NVRTC)
- LineString-LineString: segment-segment intersection kernel (Tier 1 NVRTC)
- Polygon-Polygon: overlay pipeline (face selection)

All GPU paths return device-resident OwnedGeometryArray.  The function
``_binary_constructive_gpu`` never returns None: every family pair is
handled by a GPU kernel.

ADR-0033: Tier 3 — complex multi-stage pipeline orchestrating Tier 1/2 kernels.
ADR-0002: CONSTRUCTIVE class — stays fp64 on all devices per policy.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

from vibespatial.constructive.binary_constructive_cpu import binary_constructive_cpu
from vibespatial.cuda._runtime import DeviceArray, get_cuda_runtime

if TYPE_CHECKING:
    from vibespatial.spatial.segment_primitives import DeviceSegmentTable

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    NULL_TAG,
    TAG_FAMILIES,
    DeviceFamilyGeometryBuffer,
    DeviceFixedGeometrySizeMetadata,
    OwnedGeometryArray,
    _device_take_family_buffer,
    build_device_resident_owned,
    device_concat_owned_scatter,
    device_physicalize_owned_row_selections_exact,
    device_select_owned_capacity_partitions,
    device_take_owned_capacity_selection,
    device_take_owned_family_capacity_selection,
    from_shapely_geometries,
    seed_all_validity_cache,
    tile_single_row,
    unique_tag_pairs,
)
from vibespatial.runtime._runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import (
    PhysicalWorkEstimate,
    WorkloadShape,
    estimate_physical_work_from_owned,
)
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import (
    record_fallback_event,
    strict_native_mode_enabled,
)
from vibespatial.runtime.hotpath_trace import (
    attach_work_amplification,
    hotpath_stage,
    hotpath_timing_enabled,
)
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
)
from vibespatial.runtime.residency import Residency, TransferTrigger, combined_residency

logger = logging.getLogger(__name__)

# Constructive operations that this module handles
_CONSTRUCTIVE_OPS = frozenset({"intersection", "union", "difference", "symmetric_difference"})

# Polygon-family types supported by the GPU overlay pipeline
_POLYGONAL_FAMILIES = frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON})

# LineString-family types
_LINESTRING_FAMILIES = frozenset({GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING})

# Point-Polygon constructive operations supported by the PIP fast path
_POINT_POLYGON_OPS = frozenset({"intersection", "difference"})

# Direct multipart intersection packing is only exact when fragments from the
# same source row cannot overlap or touch.  Keep the proof cheap and bounded; if
# a group needs a larger pairwise disjointness probe, use the exact union plan.
_DIRECT_MULTIPART_PACK_MAX_PAIR_PROBE = 512

# Exact disjointness refinement is useful when bbox overlap is a tiny false
# positive set.  Dense overlap workloads should go straight to the grouped
# union carrier instead of spending another relation refine proving they need it.
_DIRECT_MULTIPART_PACK_MAX_EXACT_REFINE_PAIRS = 64

# A direct-pack admission is optional work ahead of collective topology. Bound
# its worst-case segment relation to one candidate page so proving the shortcut
# cannot cost more than the constructive plan it is intended to avoid.
_DIRECT_MULTIPART_PACK_MAX_EXACT_REFINE_SEGMENT_PAIRS = 8 * 1024 * 1024

# Aggregate root-coordinate capacity is a valid row bound for injective
# carriers, but only use that conservative proof when expanding it across
# logical rows remains within one bounded multipart work page.
_INDEXED_AGGREGATE_SEGMENT_BOUND_MAX_LANES = 8 * 1024 * 1024

# Above this physical shape, repeating a scalar mask's complete boundary in
# every logical topology row is categorically the wrong plan. Prepare the one
# physical boundary and prove pass-through/exterior rows before topology.
_BROADCAST_PREPARED_MASK_MIN_PHYSICAL_SEGMENTS = 2048
_BROADCAST_PREPARED_MASK_MIN_LOGICAL_SEGMENTS = 2 * 1024 * 1024

# Keep the fixed-capacity simple-polygon coordinate workspace bounded while
# chunk outputs and upstream native frames remain resident. The planner's row
# hint is additionally capped by this physical byte budget.
_POLYGON_INTERSECTION_CHUNK_WORKSPACE_BYTES = 192 * 1024 * 1024

# Overlay can emit sub-microscopic near-collinear polygons that are valid but
# have no material area contribution at projected-coordinate scale.  If exact
# partition union cannot produce polygon pieces for such a pair, preserve the
# dominant area operand instead of fragmenting the dissolve with a sliver part.


def _device_take_unique_rowset(
    owned: OwnedGeometryArray,
    indices: DeviceArray,
) -> OwnedGeometryArray:
    """Take a proven-unique device rowset without exact-size host fences."""
    return owned.device_take(
        cp.asarray(indices, dtype=cp.int64),
        allow_capacity_allocation=True,
        assume_unique_indices=True,
    )


def _polygon_constructive_chunk_rows(planned_rows: int) -> int:
    from vibespatial.kernels.constructive.polygon_simple_intersection import (
        polygon_simple_intersection_workspace_bytes_per_row,
    )

    workspace_rows = max(
        1,
        _POLYGON_INTERSECTION_CHUNK_WORKSPACE_BYTES
        // polygon_simple_intersection_workspace_bytes_per_row(),
    )
    return max(1, min(int(planned_rows), int(workspace_rows)))


# Full OGC validity for complex face-assembled multipolygons is an O(ring^2)
# guard.  Direct union already proves area preservation from the same overlay
# plan; keep the full scan for small outputs where it is cheap and use the
# overlay proof for larger outputs.

# Empty/lower-dimensional intersection rows are classified as a native rowset.
# The old row-count gate sent larger fallback rowsets directly into one exact
# overlay replan per row, which is the wrong physical shape for grouped and
# many/few difference repair.


def _sync_hotpath() -> None:
    if hotpath_timing_enabled():
        from vibespatial.cuda._runtime import get_cuda_runtime

        get_cuda_runtime().synchronize()


def _device_scalar_bool(value, *, reason: str) -> bool:
    if cp is None:  # pragma: no cover - CPU-only installs do not call this path
        return bool(value)
    runtime = get_cuda_runtime()
    d_value = cp.asarray(value, dtype=cp.bool_).reshape(1)
    host = runtime.copy_device_to_host(d_value, reason=reason)
    return bool(np.asarray(host).reshape(-1)[0])


def _device_scalar_int(value, *, reason: str) -> int:
    if cp is None:  # pragma: no cover - CPU-only installs do not call this path
        return int(value)
    runtime = get_cuda_runtime()
    d_value = cp.asarray(value).reshape(1)
    host = runtime.copy_device_to_host(d_value, reason=reason)
    return int(np.asarray(host).reshape(-1)[0])


def is_constructive_op(op: str) -> bool:
    """Check if an operation name is a binary constructive operation."""
    return op in _CONSTRUCTIVE_OPS


def _is_family_only(owned: OwnedGeometryArray, target_families: frozenset[GeometryFamily]) -> bool:
    """Return True if every family with rows belongs to *target_families*."""
    if cp is not None and getattr(owned, "device_state", None) is not None:
        state = owned._ensure_device_state(preserve_indexed_view=True)
        if state.trusted_family_domain is not None:
            domain = set(state.trusted_family_domain)
            return bool(owned.row_count > 0 and domain and domain.issubset(target_families))
        if target_families == _POLYGONAL_FAMILIES and state.trusted_polygonal_only is True:
            return owned.row_count > 0
        if state.trusted_polygonal_only is True and target_families.isdisjoint(_POLYGONAL_FAMILIES):
            return False
        if state.trusted_homogeneous_family is not None:
            return state.trusted_homogeneous_family in target_families and owned.row_count > 0
        families = set(state.families)
        if families.isdisjoint(target_families):
            return False
        if families.issubset(target_families):
            return owned.row_count > 0
        d_valid = cp.asarray(state.validity, dtype=cp.bool_)
        d_supported = cp.zeros(int(owned.row_count), dtype=cp.bool_)
        d_tags = cp.asarray(state.tags)
        for family in target_families:
            d_supported |= d_tags == cp.int8(FAMILY_TAGS[family])
        return _device_scalar_bool(
            cp.any(d_valid & d_supported) & ~cp.any(d_valid & ~d_supported),
            reason="binary constructive family-only unsupported-row scalar fence",
        )
    has_rows = False
    for family, buf in owned.families.items():
        if buf.row_count > 0:
            if family not in target_families:
                return False
            has_rows = True
    return has_rows


def _row_validity_array(owned: OwnedGeometryArray):
    if getattr(owned, "device_state", None) is not None:
        return owned._ensure_device_state(preserve_indexed_view=True).validity
    return owned.validity


def _row_tag_array(owned: OwnedGeometryArray):
    if getattr(owned, "device_state", None) is not None:
        return owned._ensure_device_state(preserve_indexed_view=True).tags
    return owned.tags


def _row_is_valid(owned: OwnedGeometryArray, row_index: int) -> bool:
    cached = owned._current_cached_validity_mask()
    if cached is not None:
        return bool(cached[int(row_index)])
    if getattr(owned, "_validity", None) is not None:
        return bool(owned._validity[int(row_index)])
    if cp is not None and getattr(owned, "device_state", None) is not None:
        state = owned._ensure_device_state(preserve_indexed_view=True)
        if state.trusted_all_valid is True:
            return True
        return _device_scalar_bool(
            cp.asarray(state.validity)[int(row_index)],
            reason="binary constructive row-validity scalar fence",
        )
    return bool(owned.validity[int(row_index)])


def _all_rows_valid(owned: OwnedGeometryArray) -> bool:
    cached = owned._current_cached_validity_mask()
    if cached is not None:
        return bool(np.asarray(cached, dtype=bool).all())
    if getattr(owned, "_validity", None) is not None:
        return bool(np.asarray(owned._validity, dtype=bool).all())
    if cp is not None and getattr(owned, "device_state", None) is not None:
        state = owned._ensure_device_state(preserve_indexed_view=True)
        if state.trusted_all_valid is True:
            return True
        return _device_scalar_bool(
            cp.all(cp.asarray(state.validity)),
            reason="binary constructive all-validity scalar fence",
        )
    return bool(np.asarray(owned.validity, dtype=bool).all())


def _device_single_family_covering_all_rows(
    owned: OwnedGeometryArray,
) -> GeometryFamily | None:
    """Return the sole device family when its buffer accounts for every row."""
    if cp is None or getattr(owned, "device_state", None) is None:
        return None
    state = owned._ensure_device_state(preserve_indexed_view=True)
    if state.trusted_homogeneous_family is not None:
        return state.trusted_homogeneous_family
    if len(state.families) != 1:
        return None
    family, buffer = next(iter(state.families.items()))
    if int(buffer.geometry_offsets.size) - 1 != int(owned.row_count):
        return None
    return family


def _device_family_domain_tag_pairs(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> list[tuple[int, int]] | None:
    """Return a conservative device family-domain pair carrier.

    The downstream constructive dispatcher always rebuilds a per-pair rowset on
    device and skips empty rowsets before launching a family kernel.  That makes
    a small Cartesian superset over the logical family domains correct, and it
    avoids reading a device-side ``unique`` summary back to the host just to
    drive Python dispatch.
    """
    if (
        cp is None
        or getattr(left, "device_state", None) is None
        or getattr(right, "device_state", None) is None
    ):
        return None
    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)

    def _logical_families(
        owned: OwnedGeometryArray,
        state,
    ) -> tuple[GeometryFamily, ...] | None:
        if state.trusted_family_domain is not None:
            return tuple(state.trusted_family_domain)
        if state.trusted_homogeneous_family is not None:
            return (state.trusted_homogeneous_family,)
        if state.trusted_polygonal_only is True:
            polygon_families = tuple(
                family for family in state.families if family in _POLYGONAL_FAMILIES
            )
            return polygon_families or None
        if not getattr(owned, "is_indexed_view", False):
            return tuple(state.families)
        if (
            getattr(owned, "_validity", None) is not None
            and getattr(owned, "_tags", None) is not None
            and int(owned._validity.size) == int(owned.row_count)
            and int(owned._tags.size) == int(owned.row_count)
        ):
            tags = np.asarray(owned._tags, dtype=np.int8)
            validity = np.asarray(owned._validity, dtype=bool)
            return tuple(
                TAG_FAMILIES[int(tag)]
                for tag in np.unique(tags[validity])
                if int(tag) in TAG_FAMILIES
            )
        return None

    left_families = _logical_families(left, left_state)
    right_families = _logical_families(right, right_state)
    if left_families is None or right_families is None:
        return None
    if not left_families or not right_families:
        return []
    return [
        (FAMILY_TAGS[left_family], FAMILY_TAGS[right_family])
        for left_family in left_families
        for right_family in right_families
    ]


def _valid_family_tag_pairs(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> list[tuple[int, int]]:
    left_single_family = _device_single_family_covering_all_rows(left)
    right_single_family = _device_single_family_covering_all_rows(right)
    if left_single_family is not None and right_single_family is not None:
        return [(FAMILY_TAGS[left_single_family], FAMILY_TAGS[right_single_family])]

    if (
        getattr(left, "_validity", None) is not None
        and getattr(left, "_tags", None) is not None
        and getattr(right, "_validity", None) is not None
        and getattr(right, "_tags", None) is not None
        and int(left._validity.size) == int(left.row_count)
        and int(left._tags.size) == int(left.row_count)
        and int(right._validity.size) == int(right.row_count)
        and int(right._tags.size) == int(right.row_count)
    ):
        valid_mask = np.asarray(left._validity, dtype=np.bool_) & np.asarray(
            right._validity,
            dtype=np.bool_,
        )
        return unique_tag_pairs(
            np.asarray(left._tags, dtype=np.int8)[valid_mask],
            np.asarray(right._tags, dtype=np.int8)[valid_mask],
        )

    domain_pairs = _device_family_domain_tag_pairs(left, right)
    if domain_pairs is not None:
        return domain_pairs

    left_validity = _row_validity_array(left)
    right_validity = _row_validity_array(right)
    left_tags = _row_tag_array(left)
    right_tags = _row_tag_array(right)
    if cp is not None and (
        hasattr(left_validity, "__cuda_array_interface__")
        or hasattr(right_validity, "__cuda_array_interface__")
        or hasattr(left_tags, "__cuda_array_interface__")
        or hasattr(right_tags, "__cuda_array_interface__")
    ):
        left_validity = cp.asarray(left_validity)
        right_validity = cp.asarray(right_validity)
        left_tags = cp.asarray(left_tags)
        right_tags = cp.asarray(right_tags)
    valid_mask = left_validity & right_validity
    return unique_tag_pairs(left_tags[valid_mask], right_tags[valid_mask])


def _is_polygon_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is Polygon or MultiPolygon."""
    return _is_family_only(owned, _POLYGONAL_FAMILIES)


def _is_polygon_only_native(owned: OwnedGeometryArray) -> bool:
    """Return polygonal-family proof from device metadata when available."""
    if (
        getattr(owned, "_validity", None) is not None
        and getattr(owned, "_tags", None) is not None
        and int(owned._validity.size) == int(owned.row_count)
        and int(owned._tags.size) == int(owned.row_count)
    ):
        validity = np.asarray(owned._validity, dtype=np.bool_)
        if not bool(np.any(validity)):
            return False
        tags = np.asarray(owned._tags, dtype=np.int8)
        polygon_tags = np.asarray(
            [FAMILY_TAGS[GeometryFamily.POLYGON], FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]],
            dtype=np.int8,
        )
        return bool(np.all(np.isin(tags[validity], polygon_tags)))
    state = getattr(owned, "device_state", None)
    if state is None:
        return _is_polygon_only(owned)
    if state.trusted_polygonal_only is True:
        return owned.row_count > 0
    if state.trusted_homogeneous_family in _POLYGONAL_FAMILIES and state.trusted_all_valid is True:
        return owned.row_count > 0
    tags = cp.asarray(state.tags) if cp is not None else np.asarray(state.tags)
    validity = (
        cp.asarray(state.validity, dtype=cp.bool_)
        if cp is not None
        else np.asarray(state.validity, dtype=np.bool_)
    )
    polygon_tags = (
        cp.asarray(
            [FAMILY_TAGS[GeometryFamily.POLYGON], FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]],
            dtype=cp.int8,
        )
        if cp is not None
        else np.asarray(
            [FAMILY_TAGS[GeometryFamily.POLYGON], FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]],
            dtype=np.int8,
        )
    )
    supported = cp.isin(tags, polygon_tags) if cp is not None else np.isin(tags, polygon_tags)
    admitted = (
        (cp.any(validity & supported) & ~cp.any(validity & ~supported))
        if cp is not None
        else (np.any(validity & supported) and not np.any(validity & ~supported))
    )
    return _device_scalar_bool(
        admitted,
        reason="binary constructive polygon-family logical-domain scalar fence",
    )


def _device_polygonal_valid_mask(owned: OwnedGeometryArray):
    """Return a device mask for logical rows that are valid polygonal geometry."""
    if cp is None or getattr(owned, "device_state", None) is None:
        return None
    state = owned._ensure_device_state(preserve_indexed_view=True)
    d_valid = cp.asarray(state.validity, dtype=cp.bool_)
    if state.trusted_polygonal_only is True:
        return d_valid
    if state.trusted_homogeneous_family is not None:
        if state.trusted_homogeneous_family in _POLYGONAL_FAMILIES:
            return d_valid
        return cp.zeros(int(owned.row_count), dtype=cp.bool_)
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    polygon_tag = cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    multipolygon_tag = cp.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
    return d_valid & ((d_tags == polygon_tag) | (d_tags == multipolygon_tag))


def _device_polygonal_capacity_view(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Narrow ancestral mixed storage to a row-aligned polygon carrier."""
    if cp is None:
        raise RuntimeError("CuPy is required for a polygon capacity view")
    from vibespatial.geometry.owned import build_device_resident_owned

    state = owned._ensure_device_state(preserve_indexed_view=True)
    d_polygonal = _device_polygonal_valid_mask(owned)
    if d_polygonal is None:
        raise RuntimeError("polygon capacity view requires device metadata")
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int32)
    result = build_device_resident_owned(
        device_families={
            family: state.families[family]
            for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
            if family in state.families
        },
        row_count=owned.row_count,
        tags=cp.where(d_polygonal, d_tags, cp.int8(NULL_TAG)),
        validity=d_polygonal,
        family_row_offsets=cp.where(
            d_polygonal,
            d_family_rows,
            cp.int32(-1),
        ),
        execution_mode="gpu",
    )
    result_state = result._ensure_device_state(preserve_indexed_view=True)
    if state.row_bounds is not None:
        result_state.row_bounds = cp.where(
            d_polygonal[:, None],
            cp.asarray(state.row_bounds, dtype=cp.float64).reshape(
                owned.row_count,
                4,
            ),
            cp.asarray(cp.nan, dtype=cp.float64),
        )
    result_state.trusted_all_valid = (
        True if state.trusted_all_valid is True and state.trusted_polygonal_only is True else False
    )
    result_state.trusted_all_ogc_valid = state.trusted_all_ogc_valid
    result_state.trusted_homogeneous_family = (
        state.trusted_homogeneous_family
        if state.trusted_homogeneous_family in _POLYGONAL_FAMILIES
        else None
    )
    result_state.trusted_all_non_empty = None
    result_state.trusted_nonempty_polygonal_positive_area = (
        state.trusted_nonempty_polygonal_positive_area
    )
    result_state.trusted_polygonal_only = True
    result_state.trusted_unique_family_rows = state.trusted_unique_family_rows
    result_state.trusted_family_domain = tuple(result_state.families)
    result._device_family_projection_implementation = "row_aligned_polygon_family_domain_view"
    return result


def _resolve_indexed_polygon_fast_path_candidate(
    owned: OwnedGeometryArray,
    *,
    allow_capacity_allocation: bool = False,
) -> OwnedGeometryArray:
    """Materialize indexed polygon batches before physical-family kernels.

    Gathered pair batches often arrive as device-side indexed views whose
    family buffers still carry compact row counts from the source arrays.
    Rectangle and SH kernels expect physically materialized family rows.
    Row-isolated exact overlay tries the row-indirected carrier first and only
    reaches this helper after that native path declines.
    """
    if not owned.is_indexed_view:
        return owned
    if owned.residency is Residency.DEVICE or owned.device_state is not None:
        return owned.physicalize_device_rows(
            allow_capacity_allocation=allow_capacity_allocation,
        )
    return owned._resolve()


def _ensure_constructive_device_state(
    owned: OwnedGeometryArray,
    *,
    reason: str,
):
    if owned.is_indexed_view and owned.residency is Residency.DEVICE:
        return owned._ensure_device_state(preserve_indexed_view=True)
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=reason,
    )
    return owned._ensure_device_state(preserve_indexed_view=True)


def _device_take_preserving_indexed_rows(
    owned: OwnedGeometryArray,
    d_rows,
    *,
    assume_unique_indices: bool = False,
) -> OwnedGeometryArray:
    """Take logical rows without flattening an existing device indexed view."""
    if (
        cp is not None
        and getattr(owned, "is_indexed_view", False)
        and hasattr(d_rows, "__cuda_array_interface__")
    ):
        return owned._device_indexed_take(
            cp.asarray(d_rows, dtype=cp.int64),
            assume_unique_indices=assume_unique_indices,
        )
    return owned.device_take(
        d_rows,
        assume_unique_indices=assume_unique_indices,
    )


def _dispatch_partitioned_polygon_intersection_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _broadcast_right_source: OwnedGeometryArray | None = None,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Partition aligned rectangle, SH, and exact work at public-row capacity."""
    if cp is None or left.row_count != right.row_count or left.row_count == 0:
        return None

    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.envelope import _build_device_boxes_from_bounds
    from vibespatial.geometry.owned import device_scatter_owned_capacity_selections_many
    from vibespatial.kernels.constructive.polygon_intersection import (
        polygon_intersection,
        polygon_intersection_sh_eligible_mask,
    )
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        device_polygon_shape_mask_bounds,
        polygon_rect_intersection_from_bounds,
        polygon_rect_split_boundary_component_replacements_from_bounds,
    )

    row_count = int(left.row_count)
    left_shape_info = device_polygon_shape_mask_bounds(left)
    right_shape_info = device_polygon_shape_mask_bounds(right)
    if left_shape_info is None or right_shape_info is None:
        return None
    _d_left_simple, d_left_rect, d_left_bounds = left_shape_info
    _d_right_simple, d_right_rect, d_right_bounds = right_shape_info

    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    left_polygon = left_state.families.get(GeometryFamily.POLYGON)
    right_polygon = right_state.families.get(GeometryFamily.POLYGON)
    all_rectangle_pairs = (
        set(left_state.families) == {GeometryFamily.POLYGON}
        and set(right_state.families) == {GeometryFamily.POLYGON}
        and left_polygon is not None
        and right_polygon is not None
        and int(getattr(left_polygon, "dense_single_ring_width", 0) or 0) == 5
        and int(getattr(right_polygon, "dense_single_ring_width", 0) or 0) == 5
        and bool(getattr(left_polygon, "axis_aligned_rectangles", False))
        and bool(getattr(right_polygon, "axis_aligned_rectangles", False))
    )
    if all_rectangle_pairs:
        d_intersection_bounds = cp.empty_like(d_left_bounds)
        d_intersection_bounds[:, 0] = cp.maximum(d_left_bounds[:, 0], d_right_bounds[:, 0])
        d_intersection_bounds[:, 1] = cp.maximum(d_left_bounds[:, 1], d_right_bounds[:, 1])
        d_intersection_bounds[:, 2] = cp.minimum(d_left_bounds[:, 2], d_right_bounds[:, 2])
        d_intersection_bounds[:, 3] = cp.minimum(d_left_bounds[:, 3], d_right_bounds[:, 3])
        d_valid = cp.asarray(left_state.validity, dtype=cp.bool_) & cp.asarray(
            right_state.validity,
            dtype=cp.bool_,
        )
        d_positive = (
            d_valid
            & (d_intersection_bounds[:, 2] > d_intersection_bounds[:, 0])
            & (d_intersection_bounds[:, 3] > d_intersection_bounds[:, 1])
        )
        result = _build_device_boxes_from_bounds(
            cp.ascontiguousarray(d_intersection_bounds),
            row_count=row_count,
        )._apply_row_activity(d_positive)
        result._polygon_rect_boundary_overlap = cp.zeros(
            row_count,
            dtype=cp.bool_,
        )
        result._polygon_rect_exact_polygon_only = d_positive
        result._polygon_intersection_lower_dimensional_remnant = (
            d_valid
            & (d_intersection_bounds[:, 2] >= d_intersection_bounds[:, 0])
            & (d_intersection_bounds[:, 3] >= d_intersection_bounds[:, 1])
            & ~d_positive
        )
        result._polygon_rect_boundary_repair_complete = True
        seed_all_validity_cache(result)
        _apply_polygonal_device_row_segment_bound(result, 4)
        record_dispatch_event(
            surface="vibespatial.constructive.binary",
            operation="intersection",
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
            implementation="rectangle_pair_bounds_intersection_gpu",
            reason=(
                "metadata-proven rectangle pairs constructed canonical "
                "fixed-width intersections directly from device bounds"
            ),
            detail=(f"rows={row_count}; physical_shape=aligned_rectangle_bounds; polygon_width=5"),
        )
        return result

    d_sh_eligible = polygon_intersection_sh_eligible_mask(left, right)
    if d_sh_eligible is None:
        d_sh_eligible = cp.zeros(row_count, dtype=cp.bool_)
    else:
        d_sh_eligible = cp.asarray(d_sh_eligible, dtype=cp.bool_)
        if d_sh_eligible.ndim != 1 or int(d_sh_eligible.size) != row_count:
            return None
    d_swapped_sh_eligible = polygon_intersection_sh_eligible_mask(right, left)
    if d_swapped_sh_eligible is None:
        d_swapped_sh_eligible = cp.zeros(row_count, dtype=cp.bool_)
    else:
        d_swapped_sh_eligible = cp.asarray(
            d_swapped_sh_eligible,
            dtype=cp.bool_,
        )
        if d_swapped_sh_eligible.ndim != 1 or int(d_swapped_sh_eligible.size) != row_count:
            return None
    d_right_rect_mask = d_right_rect & d_sh_eligible
    d_left_rect_mask = d_left_rect & d_swapped_sh_eligible & ~d_right_rect_mask
    d_remaining = ~(d_right_rect_mask | d_left_rect_mask)
    d_sh_mask = d_sh_eligible & d_remaining
    d_swapped_sh_mask = d_swapped_sh_eligible & d_remaining & ~d_sh_mask
    d_exact_mask = d_remaining & ~d_sh_mask & ~d_swapped_sh_mask

    right_rect_selection = NativeDeviceSelection.from_mask(
        d_right_rect_mask,
        source_row_count=row_count,
    )
    left_rect_selection = NativeDeviceSelection.from_mask(
        d_left_rect_mask,
        source_row_count=row_count,
    )
    sh_selection = NativeDeviceSelection.from_mask(
        d_sh_mask,
        source_row_count=row_count,
    )
    swapped_sh_selection = NativeDeviceSelection.from_mask(
        d_swapped_sh_mask,
        source_row_count=row_count,
    )

    def _mark_boundary_repair_complete(owned: OwnedGeometryArray) -> OwnedGeometryArray:
        owned._polygon_rect_boundary_repair_complete = True
        return owned

    def _boundary_repair_complete(owned: OwnedGeometryArray) -> bool:
        return bool(getattr(owned, "_polygon_rect_boundary_repair_complete", False))

    def _mark_convex_rect_clip_exact_polygon_only(
        owned: OwnedGeometryArray,
        subject: OwnedGeometryArray,
        selection: NativeDeviceSelection,
        bounds_capacity,
    ) -> OwnedGeometryArray:
        d_active = selection.active_capacity_mask()
        rect_owned = _build_device_boxes_from_bounds(
            cp.asarray(bounds_capacity, dtype=cp.float64),
            row_count=row_count,
        )._apply_row_activity(d_active)
        subject_view = _device_take_preserving_indexed_rows(
            subject,
            selection.partition_capacity_positions(),
        )._apply_row_activity(d_active)
        d_eligible = polygon_intersection_sh_eligible_mask(
            subject_view,
            rect_owned,
        )
        if d_eligible is None or int(d_eligible.size) != row_count:
            return owned
        state = owned._ensure_device_state(preserve_indexed_view=True)
        d_exact = getattr(owned, "_polygon_rect_exact_polygon_only", None)
        if d_exact is None or int(cp.asarray(d_exact).size) != row_count:
            d_exact = cp.zeros(row_count, dtype=cp.bool_)
        else:
            d_exact = cp.asarray(d_exact, dtype=cp.bool_).copy()
        d_exact |= (
            d_active
            & cp.asarray(d_eligible, dtype=cp.bool_)
            & cp.asarray(state.validity, dtype=cp.bool_)
        )
        owned._polygon_rect_exact_polygon_only = d_exact
        return owned

    def _repair_boundary_split_rows(
        subset: OwnedGeometryArray,
        bounds_capacity,
        subject: OwnedGeometryArray,
        selection: NativeDeviceSelection,
    ) -> OwnedGeometryArray | None:
        boundary_overlap = getattr(subset, "_polygon_rect_boundary_overlap", None)
        exact_polygon_only = getattr(subset, "_polygon_rect_exact_polygon_only", None)
        lower_dimensional_remnant = getattr(
            subset,
            "_polygon_intersection_lower_dimensional_remnant",
            None,
        )
        if boundary_overlap is None:
            return _mark_convex_rect_clip_exact_polygon_only(
                subset,
                subject,
                selection,
                bounds_capacity,
            )
        d_overlap = cp.asarray(boundary_overlap, dtype=cp.bool_)
        if int(d_overlap.size) != row_count:
            return None
        replacements = polygon_rect_split_boundary_component_replacements_from_bounds(
            subset,
            bounds_capacity,
            d_overlap,
        )
        if replacements is None:
            return None
        split_owned, d_split_mask = replacements
        if split_owned.row_count != row_count:
            return None
        repaired = device_select_owned_capacity_partitions(
            subset,
            [(split_owned, d_split_mask)],
        )
        repaired._polygon_rect_boundary_overlap = cp.zeros(
            row_count,
            dtype=cp.bool_,
        )
        if exact_polygon_only is not None:
            repaired._polygon_rect_exact_polygon_only = cp.asarray(
                exact_polygon_only,
                dtype=cp.bool_,
            )
        if lower_dimensional_remnant is not None:
            repaired._polygon_intersection_lower_dimensional_remnant = cp.asarray(
                lower_dimensional_remnant,
                dtype=cp.bool_,
            )
        repaired = _mark_convex_rect_clip_exact_polygon_only(
            repaired,
            subject,
            selection,
            bounds_capacity,
        )
        return _mark_boundary_repair_complete(repaired)

    def _clip_partition(
        selection: NativeDeviceSelection,
        subject: OwnedGeometryArray,
        bounds,
    ) -> OwnedGeometryArray | None:
        bounds_capacity = selection.gather_capacity(bounds, fill_value=cp.nan)
        subset = polygon_rect_intersection_from_bounds(
            subject,
            bounds_capacity,
            source_rows=selection.partition_capacity_positions(),
            dispatch_mode=ExecutionMode.GPU,
        )
        if subset.row_count != row_count:
            return None
        return _repair_boundary_split_rows(
            subset,
            bounds_capacity,
            subject,
            selection,
        )

    right_rect_result = _clip_partition(
        right_rect_selection,
        left,
        d_right_bounds,
    )
    left_rect_result = _clip_partition(
        left_rect_selection,
        right,
        d_left_bounds,
    )
    if right_rect_result is None or left_rect_result is None:
        return None
    if getattr(right_rect_result, "_polygon_rect_exact_polygon_only", None) is None:
        right_rect_result._polygon_rect_exact_polygon_only = (
            right_rect_selection.active_capacity_mask()
            & right_rect_selection.gather_capacity(d_left_rect)
        )

    def _sh_partition(
        selection: NativeDeviceSelection,
        subject: OwnedGeometryArray,
        clip: OwnedGeometryArray,
    ) -> OwnedGeometryArray | None:
        d_active = selection.active_capacity_mask()
        subject_capacity = _device_take_preserving_indexed_rows(
            subject,
            selection.partition_capacity_positions(),
        )._apply_row_activity(d_active)
        clip_capacity = _device_take_preserving_indexed_rows(
            clip,
            selection.partition_capacity_positions(),
        )._apply_row_activity(d_active)
        partition = polygon_intersection(
            subject_capacity,
            clip_capacity,
            dispatch_mode=ExecutionMode.GPU,
        )
        if partition is None or partition.row_count != row_count:
            return None
        state = partition._ensure_device_state(preserve_indexed_view=True)
        partition._polygon_rect_exact_polygon_only = d_active & cp.asarray(
            state.validity, dtype=cp.bool_
        )
        partition._polygon_rect_boundary_overlap = cp.zeros(
            row_count,
            dtype=cp.bool_,
        )
        partition._polygon_intersection_lower_dimensional_remnant = cp.zeros(
            row_count,
            dtype=cp.bool_,
        )
        return partition

    sh_result = _sh_partition(sh_selection, left, right)
    swapped_sh_result = _sh_partition(swapped_sh_selection, right, left)
    if sh_result is None or swapped_sh_result is None:
        return None

    def _unsupported_sh_source_mask(
        result: OwnedGeometryArray,
        selection: NativeDeviceSelection,
    ):
        d_supported = getattr(result, "_polygon_intersection_sh_supported", None)
        if d_supported is None or int(cp.asarray(d_supported).size) != row_count:
            return selection.source_mask()
        return selection.source_mask(
            active_mask=(
                selection.active_capacity_mask() & ~cp.asarray(d_supported, dtype=cp.bool_)
            )
        )

    d_sh_retry = _unsupported_sh_source_mask(sh_result, sh_selection)
    d_swapped_sh_retry = _unsupported_sh_source_mask(
        swapped_sh_result,
        swapped_sh_selection,
    )
    d_exact_mask = d_exact_mask | d_sh_retry | d_swapped_sh_retry
    exact_selection = NativeDeviceSelection.from_mask(
        d_exact_mask,
        source_row_count=row_count,
    )

    d_exact_active = exact_selection.active_capacity_mask()
    exact_left = _device_take_preserving_indexed_rows(
        left,
        exact_selection.partition_capacity_positions(),
    )._apply_row_activity(d_exact_active)
    if _broadcast_right_source is not None:
        exact_result = _dispatch_polygon_intersection_overlay_broadcast_right_gpu(
            exact_left,
            _broadcast_right_source,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=_cached_right_segments,
        )
    else:
        exact_right = _device_take_preserving_indexed_rows(
            right,
            exact_selection.partition_capacity_positions(),
        )._apply_row_activity(d_exact_active)
        exact_result = _dispatch_polygon_intersection_overlay_exact_batch_gpu(
            exact_left,
            exact_right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=None,
        )
    if exact_result is None or exact_result.row_count != row_count:
        return None

    result = device_scatter_owned_capacity_selections_many(
        _empty_device_constructive_output(row_count),
        [
            (right_rect_result, right_rect_selection, None),
            (left_rect_result, left_rect_selection, None),
            (sh_result, sh_selection, None),
            (swapped_sh_result, swapped_sh_selection, None),
            (exact_result, exact_selection, None),
        ],
    )

    def _scatter_metadata(base, values, selection: NativeDeviceSelection):
        d_base = cp.asarray(base, dtype=cp.bool_)
        d_values = cp.asarray(values, dtype=cp.bool_)
        d_active = selection.active_capacity_mask()
        d_lanes = cp.arange(row_count, dtype=cp.int64)
        d_destinations = cp.where(
            d_active,
            selection.partition_capacity_positions(),
            cp.int64(row_count) + d_lanes,
        )
        d_extended = cp.concatenate(
            [d_base, cp.zeros(row_count, dtype=cp.bool_)],
        )
        d_extended[d_destinations] = cp.where(
            d_active,
            d_values,
            cp.bool_(False),
        )
        return d_extended[:row_count]

    boundary_overlap = cp.zeros(row_count, dtype=cp.bool_)
    exact_polygon_only = cp.zeros(row_count, dtype=cp.bool_)
    exact_area = cp.zeros(row_count, dtype=cp.bool_)
    lower_dimensional_remnant = cp.zeros(row_count, dtype=cp.bool_)
    for replacement, selection in (
        (right_rect_result, right_rect_selection),
        (left_rect_result, left_rect_selection),
        (sh_result, sh_selection),
        (swapped_sh_result, swapped_sh_selection),
        (exact_result, exact_selection),
    ):
        overlap = getattr(replacement, "_polygon_rect_boundary_overlap", None)
        if overlap is not None:
            boundary_overlap = _scatter_metadata(
                boundary_overlap,
                overlap,
                selection,
            )
        exact = getattr(replacement, "_polygon_rect_exact_polygon_only", None)
        if exact is not None:
            exact_polygon_only = _scatter_metadata(
                exact_polygon_only,
                exact,
                selection,
            )
        area_proof = getattr(replacement, "_polygon_intersection_exact_area", None)
        if area_proof is not None:
            exact_area = _scatter_metadata(
                exact_area,
                area_proof,
                selection,
            )
        remnant = getattr(
            replacement,
            "_polygon_intersection_lower_dimensional_remnant",
            None,
        )
        if remnant is not None:
            lower_dimensional_remnant = _scatter_metadata(
                lower_dimensional_remnant,
                remnant,
                selection,
            )

    result._polygon_rect_boundary_overlap = boundary_overlap
    result._polygon_rect_exact_polygon_only = exact_polygon_only
    result._polygon_intersection_exact_area = exact_area
    result._polygon_intersection_lower_dimensional_remnant = lower_dimensional_remnant
    _apply_polygonal_device_row_segment_bound(
        result,
        _polygon_intersection_segment_span_bound(left, right),
    )
    if all(_boundary_repair_complete(part) for part in (right_rect_result, left_rect_result)):
        _mark_boundary_repair_complete(result)
        seed_all_validity_cache(result)
    record_dispatch_event(
        surface="vibespatial.constructive.binary",
        operation="intersection",
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
        implementation="polygon_intersection_partitioned_capacity_gpu",
        reason=(
            "polygon intersection retained rectangle, SH, and exact work in "
            "device-counted public-row capacity partitions"
        ),
        detail=(
            f"rows={row_count}; partition_counts=device-resident; "
            "workload_shape=aligned_pairwise_rect_sh_exact_capacity"
        ),
    )
    return result


def _dispatch_chunked_polygon_intersection_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    chunk_rows: int,
    dispatch_mode: ExecutionMode,
):
    """Execute aligned polygon intersections in bounded device workspaces."""
    if cp is None or left.row_count != right.row_count:
        return None
    row_count = int(left.row_count)
    chunk_rows = int(chunk_rows)
    if chunk_rows <= 0:
        raise ValueError("polygon intersection chunk_rows must be positive")

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeGeometryComposition,
    )

    chunks = []
    for start in range(0, row_count, chunk_rows):
        stop = min(start + chunk_rows, row_count)
        d_rows = cp.arange(start, stop, dtype=cp.int64)
        left_chunk = left._device_indexed_take(
            d_rows,
            assume_unique_indices=True,
        )
        right_chunk = right._device_indexed_take(
            d_rows,
            assume_unique_indices=True,
        )
        chunk = _dispatch_partitioned_polygon_intersection_gpu(
            left_chunk,
            right_chunk,
            dispatch_mode=dispatch_mode,
        )
        if chunk is None or int(chunk.row_count) != stop - start:
            return None
        # The partition dispatcher returns a row-indirected scatter view whose
        # expanded metadata already addresses its exact shared family buffers.
        # A completed chunk is the ownership boundary: detach that metadata
        # carrier so its inactive scatter root can be released immediately.
        chunk.detach_expanded_device_view()
        chunks.append(GeometryNativeResult.from_owned(chunk, crs=None))

    result = GeometryNativeResult.from_composition(
        NativeGeometryComposition.concat(chunks, crs=None),
        crs=None,
    )
    record_dispatch_event(
        surface="vibespatial.constructive.binary",
        operation="intersection",
        implementation="chunked_polygon_intersection_composition_gpu",
        reason=(
            "aligned polygon relation exceeded its bounded device workspace "
            "and retained chunk outputs in a contiguous native composition"
        ),
        detail=(
            f"rows={row_count}; chunk_rows={chunk_rows}; "
            f"chunks={len(chunks)}; workload_shape=bounded_relation_chunks"
        ),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return result


def broadcast_right_polygon_intersection_capacity_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    right_row: int = 0,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Intersect polygon rows with one row-indirected right geometry at capacity."""
    if cp is None or left.row_count == 0:
        return None
    right_row = int(right_row)
    if right_row < 0 or right_row >= int(right.row_count):
        raise IndexError("broadcast-right polygon intersection row is out of bounds")

    if left.residency is not Residency.DEVICE:
        left = left.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="broadcast-right polygon intersection promoted left capacity to device",
        )
    if right.residency is not Residency.DEVICE:
        right = right.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="broadcast-right polygon intersection promoted right source to device",
        )

    if right.row_count == 1 and right_row == 0:
        right_one = right
    else:
        right_one = right.device_take(cp.asarray([right_row], dtype=cp.int64))

    right_segment_span = _polygon_segment_span_bound(right_one)
    if (
        right_segment_span is not None
        and int(right_segment_span) >= _BROADCAST_PREPARED_MASK_MIN_PHYSICAL_SEGMENTS
        and int(right_segment_span) * int(left.row_count)
        >= _BROADCAST_PREPARED_MASK_MIN_LOGICAL_SEGMENTS
    ):
        prepared_result = _dispatch_prepared_polygon_intersection_broadcast_right_gpu(
            left,
            right_one,
            dispatch_mode=dispatch_mode,
        )
        if prepared_result is not None:
            prepared_result._aligned_left_pairs_owned = left
            prepared_result._aligned_right_pairs_owned = tile_single_row(
                right_one,
                int(left.row_count),
            )
            return prepared_result

    right_capacity = tile_single_row(right_one, int(left.row_count))
    result = _dispatch_partitioned_polygon_intersection_gpu(
        left,
        right_capacity,
        dispatch_mode=dispatch_mode,
        _broadcast_right_source=right_one,
    )
    if result is None or int(result.row_count) != int(left.row_count):
        return None
    result._aligned_left_pairs_owned = left
    result._aligned_right_pairs_owned = right_capacity
    return result


def _dispatch_prepared_polygon_intersection_broadcast_right_gpu(
    left: OwnedGeometryArray,
    right_one: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode,
) -> OwnedGeometryArray | None:
    """Partition one-mask intersections by sparse boundary evidence.

    Contained rows pass through unchanged and exterior rows become valid empty
    polygons. Rows whose bounds overlap a physical mask segment remain in a
    row-aligned device capacity while exact topology masks inactive lanes.
    """
    if cp is None or int(right_one.row_count) != 1:
        return None

    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.geometry.owned import build_empty_polygon_rows_device
    from vibespatial.spatial.prepared_polygon_mask import (
        PreparedPolygonMask,
        prepared_polygon_mask_fp64_plan,
    )

    prepared = PreparedPolygonMask.from_owned(
        right_one,
        precision_plan=prepared_polygon_mask_fp64_plan(),
    )
    if prepared is None:
        return None
    classification = prepared.classify_polygon_rows(left)
    if classification is None:
        return None

    row_count = int(left.row_count)
    d_valid = cp.asarray(classification.valid, dtype=cp.bool_)
    d_covered = cp.asarray(classification.covered_by, dtype=cp.bool_)
    d_unresolved = cp.asarray(
        classification.boundary_unresolved,
        dtype=cp.bool_,
    )
    unresolved_selection = NativeDeviceSelection.from_mask(
        d_unresolved,
        source_row_count=row_count,
    )
    result = build_empty_polygon_rows_device(row_count, validity=d_valid)
    result = device_select_owned_capacity_partitions(
        result,
        [(left, d_covered)],
    )

    exact_capacity = device_take_owned_capacity_selection(
        left,
        unresolved_selection,
    )
    (exact_left,) = device_physicalize_owned_row_selections_exact(
        [(exact_capacity, unresolved_selection.active_capacity_mask())],
        reason="prepared broadcast exact topology allocation packet",
        compact_concrete_prefix=True,
    )
    exact_result = None
    d_exact_rows = cp.empty(0, dtype=cp.int64)
    if exact_left is not None:
        exact_result = _dispatch_polygon_intersection_overlay_broadcast_right_gpu(
            exact_left,
            right_one,
            dispatch_mode=dispatch_mode,
        )
        if exact_result is None or int(exact_result.row_count) != int(exact_left.row_count):
            return None
        d_exact_rows = cp.asarray(
            unresolved_selection.positions[: exact_result.row_count],
            dtype=cp.int64,
        )
        result = device_concat_owned_scatter(
            result,
            exact_result,
            d_exact_rows,
        )

    def _scatter_exact_metadata(name: str, *, base):
        d_values = cp.asarray(base, dtype=cp.bool_).copy()
        if exact_result is None:
            return d_values
        exact_values = getattr(exact_result, name, None)
        if exact_values is not None:
            d_values[d_exact_rows] = cp.asarray(exact_values, dtype=cp.bool_)
        return d_values

    result._polygon_rect_exact_polygon_only = _scatter_exact_metadata(
        "_polygon_rect_exact_polygon_only",
        base=d_covered,
    )
    result._polygon_intersection_exact_area = _scatter_exact_metadata(
        "_polygon_intersection_exact_area",
        base=d_covered,
    )
    result._polygon_rect_boundary_overlap = _scatter_exact_metadata(
        "_polygon_rect_boundary_overlap",
        base=cp.zeros(row_count, dtype=cp.bool_),
    )
    result._polygon_intersection_lower_dimensional_remnant = _scatter_exact_metadata(
        "_polygon_intersection_lower_dimensional_remnant",
        base=cp.zeros(row_count, dtype=cp.bool_),
    )
    result._many_vs_one_containment_passthrough_mask = d_covered
    result._many_vs_one_left_containment_bypass_applied = True
    _apply_polygonal_device_row_segment_bound(
        result,
        _polygon_segment_span_bound(left),
    )
    record_dispatch_event(
        surface="vibespatial.constructive.binary",
        operation="intersection",
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
        implementation="prepared_broadcast_mask_capacity_partition_gpu",
        reason=(
            "one physical polygon mask classified contained, exterior, and "
            "boundary-unresolved rows before exact topology"
        ),
        detail=(
            f"rows={row_count}; exact_rows={int(d_exact_rows.size)}; "
            "workload_shape=sparse_boundary_relation_plus_exact_physicalization"
        ),
    )
    return result


def _is_point_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is Point."""
    return _is_family_only(owned, frozenset({GeometryFamily.POINT}))


def _is_linestring_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is LineString."""
    return _is_family_only(owned, frozenset({GeometryFamily.LINESTRING}))


def _is_lineal_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is LineString or MultiLineString."""
    return _is_family_only(owned, _LINESTRING_FAMILIES)


def _is_multipoint_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is MultiPoint."""
    return _is_family_only(owned, frozenset({GeometryFamily.MULTIPOINT}))


def _intersection_point_polygon_gpu(
    points: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """GPU Point-Polygon intersection with exact device membership assembly.

    For each element-wise pair (point_i, polygon_i):
    - point inside polygon  -> keep the point
    - point outside polygon -> empty Point
    - either input NULL     -> NULL

    ADR-0033: Tier 2 device masks over exact point-region classification.
    """
    points.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="binary_constructive point_polygon intersection GPU",
    )
    polygons.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="binary_constructive point_polygon intersection GPU",
    )

    from vibespatial.constructive.multipoint_polygon_constructive import (
        point_membership_rows_to_owned,
    )
    from vibespatial.predicates.binary import binary_predicate_expression

    intersects = binary_predicate_expression(
        "intersects",
        points,
        polygons,
        dispatch_mode=ExecutionMode.GPU,
        operation="constructive.point_polygon.intersection",
    )
    if intersects is None or not hasattr(
        intersects.values,
        "__cuda_array_interface__",
    ):
        raise RuntimeError("Point-Polygon intersection left native execution")
    d_output_validity = points.device_state.validity.astype(
        bool, copy=False
    ) & polygons.device_state.validity.astype(bool, copy=False)
    return point_membership_rows_to_owned(
        points,
        output_validity=d_output_validity,
        keep_rows=intersects.values,
    )


def _difference_point_polygon_gpu(
    points: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """GPU Point-Polygon difference with exact device membership assembly.

    For each element-wise pair (point_i, polygon_i):
    - point outside polygon -> keep the point
    - point inside polygon  -> empty Point
    - left (point) NULL     -> NULL
    - right (polygon) NULL  -> NULL

    ADR-0033: Tier 2 device masks over exact point-region classification.
    """
    points.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="binary_constructive point_polygon difference GPU",
    )
    polygons.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="binary_constructive point_polygon difference GPU",
    )

    from vibespatial.constructive.multipoint_polygon_constructive import (
        point_membership_rows_to_owned,
    )
    from vibespatial.predicates.binary import binary_predicate_expression

    intersects = binary_predicate_expression(
        "intersects",
        points,
        polygons,
        dispatch_mode=ExecutionMode.GPU,
        operation="constructive.point_polygon.difference",
    )
    if intersects is None or not hasattr(
        intersects.values,
        "__cuda_array_interface__",
    ):
        raise RuntimeError("Point-Polygon difference left native execution")
    d_output_validity = points.device_state.validity.astype(
        bool, copy=False
    ) & polygons.device_state.validity.astype(bool, copy=False)
    return point_membership_rows_to_owned(
        points,
        output_validity=d_output_validity,
        keep_rows=~cp.asarray(intersects.values, dtype=cp.bool_),
    )


def _dispatch_overlay_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
    _right_segment_broadcast=None,
    _row_isolated: bool = False,
    _left_geometry_source_rows=None,
    _right_geometry_source_rows=None,
    _right_segment_source_rows=None,
    _include_same_side_splits: bool = False,
) -> OwnedGeometryArray:
    """Dispatch to the GPU overlay pipeline for Polygon-Polygon pairs.

    Imports are lazy to avoid circular dependencies between constructive
    and overlay modules.
    """
    from vibespatial.overlay.gpu import _overlay_owned

    return _overlay_owned(
        left,
        right,
        dispatch_mode=dispatch_mode,
        operation=op,
        _cached_right_segments=_cached_right_segments,
        _right_segment_broadcast=_right_segment_broadcast,
        _row_isolated=_row_isolated,
        _left_geometry_source_rows=_left_geometry_source_rows,
        _right_geometry_source_rows=_right_geometry_source_rows,
        _right_segment_source_rows=_right_segment_source_rows,
        _include_same_side_splits=_include_same_side_splits,
    )


def _free_device_segment_table(device_segments: DeviceSegmentTable) -> None:
    """Release a cached device segment table allocated by _extract_segments_gpu."""
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    runtime.free(device_segments.x0)
    runtime.free(device_segments.y0)
    runtime.free(device_segments.x1)
    runtime.free(device_segments.y1)
    runtime.free(device_segments.row_indices)
    runtime.free(device_segments.segment_indices)
    if device_segments.part_indices is not None:
        runtime.free(device_segments.part_indices)
    if device_segments.ring_indices is not None:
        runtime.free(device_segments.ring_indices)


def _dispatch_polygon_overlay_row_isolated_batch_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Execute one row-isolated topology plan for aligned polygon pairs.

    Row preservation belongs to the topology plan and its native output
    carrier. Indexed inputs remain logical row-indirected sources; this helper
    never physicalizes them or falls back to one GPU launch per Python row.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    if left.row_count != right.row_count:
        return None
    if op == "union":
        return _dispatch_polygon_union_repair_gpu(
            left,
            right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=_cached_right_segments,
        )
    batch_result = _dispatch_overlay_gpu(
        op,
        left,
        right,
        dispatch_mode=dispatch_mode,
        _cached_right_segments=(
            None if bool(getattr(right, "is_indexed_view", False)) else _cached_right_segments
        ),
        _row_isolated=True,
    )
    if batch_result.row_count != left.row_count:
        logger.debug(
            "row-isolated polygon %s topology returned %d rows (expected %d)",
            op,
            batch_result.row_count,
            left.row_count,
        )
        return None

    if op == "intersection":
        state = batch_result._ensure_device_state(preserve_indexed_view=True)
        polygon_tags = cp.asarray(
            [
                FAMILY_TAGS[GeometryFamily.POLYGON],
                FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            ],
            dtype=cp.int8,
        )
        batch_result._polygon_intersection_exact_area = cp.asarray(
            state.validity,
            dtype=cp.bool_,
        ) & cp.isin(cp.asarray(state.tags, dtype=cp.int8), polygon_tags)

    if bool(getattr(left, "is_indexed_view", False)) or bool(
        getattr(right, "is_indexed_view", False)
    ):
        record_dispatch_event(
            surface="vibespatial.constructive.binary_constructive",
            operation=op,
            implementation="row_isolated_polygon_overlay_row_indirected_exact_gpu",
            reason=(
                "row-isolated exact polygon overlay consumed indexed owned "
                "views through the row-indirected topology carrier"
            ),
            detail=(
                f"rows={left.row_count}, "
                f"left_indexed={bool(getattr(left, 'is_indexed_view', False))}, "
                f"right_indexed={bool(getattr(right, 'is_indexed_view', False))}"
            ),
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
        )
    return batch_result


def _dispatch_polygon_intersection_overlay_broadcast_right_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Preserve row cardinality for polygon intersection against a scalar right polygon.

    This keeps the right operand truly broadcast-right: one right geometry,
    one extracted right-side segment table, many left rows. It avoids
    materializing a tiled right OwnedGeometryArray just to feed the exact
    overlay path.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if right.row_count != 1:
        return None
    if left.row_count == 0:
        return _empty_device_constructive_output(0)

    from vibespatial.spatial.segment_primitives import (
        DeviceBroadcastSegmentRelation,
        _extract_segments_gpu,
    )

    source_segments = _cached_right_segments
    owns_source_segments = source_segments is None
    if source_segments is None:
        source_segments = _extract_segments_gpu(right)
    relation = DeviceBroadcastSegmentRelation(
        physical_segments=source_segments,
        logical_row_count=int(left.row_count),
    )
    _polygon_segment_span_bound(left)
    try:
        batch_result = _dispatch_overlay_gpu(
            "intersection",
            left,
            right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=source_segments,
            _right_segment_broadcast=relation,
            _row_isolated=True,
        )
        if batch_result.row_count != left.row_count:
            return None
        record_dispatch_event(
            surface="vibespatial.constructive.binary",
            operation="intersection",
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
            implementation="broadcast_right_virtual_segment_topology_gpu",
            reason=(
                "scalar-right polygon topology retained one physical mask segment "
                "table and derived row-isolated source ids algebraically"
            ),
            detail=(
                f"rows={left.row_count}; "
                f"physical_right_segments={relation.physical_count}; "
                f"logical_right_segments={relation.logical_count}"
            ),
        )
        return batch_result
    finally:
        if owns_source_segments:
            _free_device_segment_table(source_segments)


def _dispatch_polygon_intersection_overlay_exact_batch_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    if left.row_count != right.row_count:
        return None
    segment_span_bound = _polygon_intersection_segment_span_bound(left, right)
    simple_result = _dispatch_polygon_simple_intersection_with_exact_remainder_gpu(
        left,
        right,
        dispatch_mode=dispatch_mode,
    )
    if simple_result is not None:
        _apply_polygonal_device_row_segment_bound(simple_result, segment_span_bound)
        return simple_result
    result = _dispatch_polygon_overlay_row_isolated_batch_gpu(
        "intersection",
        left,
        right,
        dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )
    if result is not None:
        _apply_polygonal_device_row_segment_bound(result, segment_span_bound)
    return result


def _dispatch_polygon_simple_intersection_with_exact_remainder_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Partition validated simple and exact rows at aligned row capacity."""
    if cp is None or left.row_count != right.row_count or left.row_count == 0:
        return None
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.geometry.owned import device_scatter_owned_capacity_selection
    from vibespatial.kernels.constructive.polygon_simple_intersection import (
        polygon_simple_intersection,
    )

    simple_intersection = polygon_simple_intersection(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    if simple_intersection is None:
        return None
    simple_result, d_simple_supported = simple_intersection
    row_count = int(left.row_count)
    d_simple_supported = cp.asarray(d_simple_supported, dtype=cp.bool_)
    if simple_result.row_count != row_count or int(d_simple_supported.size) != row_count:
        return None

    simple_state = simple_result._ensure_device_state(preserve_indexed_view=True)
    d_simple_exact = d_simple_supported & cp.asarray(
        simple_state.validity,
        dtype=cp.bool_,
    )
    exact_selection = NativeDeviceSelection.from_mask(
        ~d_simple_supported,
        source_row_count=row_count,
    )
    d_exact_positions = exact_selection.partition_capacity_positions()
    d_exact_active = exact_selection.active_capacity_mask()
    exact_left = _device_take_preserving_indexed_rows(
        left,
        d_exact_positions,
    )._apply_row_activity(d_exact_active)
    exact_right = _device_take_preserving_indexed_rows(
        right,
        d_exact_positions,
    )._apply_row_activity(d_exact_active)
    exact_remainder = _dispatch_polygon_overlay_row_isolated_batch_gpu(
        "intersection",
        exact_left,
        exact_right,
        dispatch_mode=dispatch_mode,
        _cached_right_segments=None,
    )
    if exact_remainder is None or exact_remainder.row_count != row_count:
        return None

    result = device_scatter_owned_capacity_selection(
        simple_result,
        exact_remainder,
        exact_selection,
    )

    def _scatter_exact_metadata(values, *, base):
        d_values = cp.asarray(values, dtype=cp.bool_)
        d_base = cp.asarray(base, dtype=cp.bool_)
        d_lanes = cp.arange(row_count, dtype=cp.int64)
        d_destinations = cp.where(
            d_exact_active,
            exact_selection.partition_capacity_positions(),
            cp.int64(row_count) + d_lanes,
        )
        d_extended = cp.concatenate(
            [d_base, cp.zeros(row_count, dtype=cp.bool_)],
        )
        d_extended[d_destinations] = cp.where(
            d_exact_active,
            d_values,
            cp.bool_(False),
        )
        return d_extended[:row_count]

    exact_polygon = getattr(exact_remainder, "_polygon_rect_exact_polygon_only", None)
    result._polygon_rect_exact_polygon_only = _scatter_exact_metadata(
        (cp.zeros(row_count, dtype=cp.bool_) if exact_polygon is None else exact_polygon),
        base=d_simple_exact,
    )
    exact_area = getattr(exact_remainder, "_polygon_intersection_exact_area", None)
    result._polygon_intersection_exact_area = _scatter_exact_metadata(
        (cp.zeros(row_count, dtype=cp.bool_) if exact_area is None else exact_area),
        base=d_simple_exact,
    )
    exact_overlap = getattr(exact_remainder, "_polygon_rect_boundary_overlap", None)
    if exact_overlap is not None:
        result._polygon_rect_boundary_overlap = _scatter_exact_metadata(
            exact_overlap,
            base=cp.zeros(row_count, dtype=cp.bool_),
        )
    exact_remnant = getattr(
        exact_remainder,
        "_polygon_intersection_lower_dimensional_remnant",
        None,
    )
    if exact_remnant is not None:
        result._polygon_intersection_lower_dimensional_remnant = _scatter_exact_metadata(
            exact_remnant,
            base=cp.zeros(row_count, dtype=cp.bool_),
        )
    record_dispatch_event(
        surface="vibespatial.constructive.binary",
        operation="intersection",
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
        implementation="row_isolated_polygon_capacity_partition_gpu",
        reason=(
            "row-isolated polygon intersection retained simple and exact rows in "
            "one device-counted aligned capacity partition"
        ),
        detail=(
            f"rows={row_count}; partition_counts=device-resident; "
            "workload_shape=aligned_pairwise_polygon_capacity"
        ),
    )
    return result


def point_parts_native_tabular_result(
    owned: OwnedGeometryArray,
    *,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
    attrs: dict | None = None,
):
    """Lower point part expansion to a non-row-aligned native result."""
    from vibespatial.api._native_results import (
        _point_parts_constructive_to_native_tabular_result,
    )

    return _point_parts_constructive_to_native_tabular_result(
        owned,
        operation="point_parts",
        crs=crs,
        geometry_name=geometry_name,
        source_rows=source_rows,
        source_tokens=source_tokens,
        attrs=attrs,
    )


def polygonal_parts_native_tabular_result(
    owned: OwnedGeometryArray,
    *,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
    attrs: dict | None = None,
):
    """Lower polygonal part expansion to a non-row-aligned native result."""
    from vibespatial.api._native_results import (
        _polygonal_parts_constructive_to_native_tabular_result,
    )

    return _polygonal_parts_constructive_to_native_tabular_result(
        owned,
        operation="polygonal_parts",
        crs=crs,
        geometry_name=geometry_name,
        source_rows=source_rows,
        source_tokens=source_tokens,
        attrs=attrs,
    )


def _dispatch_lineal_polygonal_constructive_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
):
    """Run aligned lineal/polygonal construction through collective topology."""
    if cp is None or left.row_count != right.row_count:
        return None
    if not (_is_lineal_only(left) and _is_polygon_only(right)):
        return None

    from vibespatial.constructive.line_polygon_difference import (
        lineal_polygonal_constructive_topology_gpu,
    )

    return lineal_polygonal_constructive_topology_gpu(
        left,
        right,
        operation=op,
        dispatch_mode=dispatch_mode,
    )


def _dispatch_lineal_polygonal_difference_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Compatibility entry point for grouped mixed line difference."""
    return _dispatch_lineal_polygonal_constructive_gpu(
        "difference",
        left,
        right,
        dispatch_mode=dispatch_mode,
    )


def _dispatch_lineal_lineal_difference_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Subtract aligned lineal rows through the collective split carrier."""
    if cp is None or left.row_count != right.row_count:
        return None
    if not (_is_lineal_only(left) and _is_lineal_only(right)):
        return None
    from vibespatial.constructive.line_polygon_difference import (
        lineal_lineal_difference_topology_gpu,
    )

    return lineal_lineal_difference_topology_gpu(
        left,
        right,
        dispatch_mode=dispatch_mode,
    )


def lineal_parts_native_tabular_result(
    owned: OwnedGeometryArray,
    *,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
    attrs: dict | None = None,
):
    """Lower lineal part expansion to a non-row-aligned native result."""
    from vibespatial.api._native_results import (
        _lineal_parts_constructive_to_native_tabular_result,
    )

    return _lineal_parts_constructive_to_native_tabular_result(
        owned,
        operation="lineal_parts",
        crs=crs,
        geometry_name=geometry_name,
        source_rows=source_rows,
        source_tokens=source_tokens,
        attrs=attrs,
    )


def _oriented_multipolygon_polygon_source_rows_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
):
    """Return identity rows for a structurally homogeneous oriented pair."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count or left.row_count == 0:
        return None
    if (
        _device_single_family_covering_all_rows(left) is not GeometryFamily.MULTIPOLYGON
        or _device_single_family_covering_all_rows(right) is not GeometryFamily.POLYGON
    ):
        return None
    return cp.arange(left.row_count, dtype=cp.int64)


def _dispatch_oriented_multipolygon_polygon_intersection_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    admitted_rows: DeviceArray | None = None,
) -> OwnedGeometryArray | None:
    """Exact GPU rescue for valid row-aligned MultiPolygon-Polygon intersection.

    Explodes multipolygon rows into polygon parts on device, intersects each
    part against its aligned polygon row using the existing exact polygon
    path, then unions the valid part intersections back per original row.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    if admitted_rows is None:
        admitted_rows = _oriented_multipolygon_polygon_source_rows_gpu(left, right)
    if admitted_rows is None:
        return None

    _sync_hotpath()
    with hotpath_stage(
        "constructive.intersection.multipart_explode",
        category="setup",
    ) as amplification_metadata:
        multipolygon_rows = _device_take_known_family_rows(
            left,
            cp.asarray(admitted_rows, dtype=cp.int64),
            GeometryFamily.MULTIPOLYGON,
            allow_capacity_allocation=True,
        )
        polygon_parts = _explode_polygonal_rows_to_polygon_capacity_gpu(
            multipolygon_rows,
        )
        if amplification_metadata is not None:
            explode_sums = {
                "input_rows": int(admitted_rows.size),
                "output_groups": int(left.row_count),
            }
            explode_unavailable = [
                "max_group_size",
                "input_segments",
                "input_coordinates",
                "output_parts",
                "output_coordinates",
            ]
            if polygon_parts is None:
                explode_unavailable.append("pre_reduction_fragments")
            else:
                explode_sums["pre_reduction_fragments"] = int(polygon_parts.capacity)
            attach_work_amplification(
                amplification_metadata,
                operation="constructive.intersection.multipart_explode",
                metric_family="group_compression",
                sums=explode_sums,
                maxima={"row_capacity": int(left.row_count)},
                unavailable=tuple(explode_unavailable),
            )
    _sync_hotpath()
    if polygon_parts is None or polygon_parts.capacity == 0:
        return _empty_device_constructive_output(left.row_count)

    d_source_rows = cp.asarray(admitted_rows, dtype=cp.int64)[
        cp.asarray(polygon_parts.source_rows, dtype=cp.int64)
    ].astype(cp.int64, copy=False)
    exploded_right = _device_take_known_family_rows(
        right,
        d_source_rows,
        GeometryFamily.POLYGON,
        allow_capacity_allocation=True,
    )
    exploded_left = polygon_parts.geometry
    if exploded_right.row_count != polygon_parts.capacity:
        return None
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
        polygon_rect_intersection_can_handle,
    )

    if polygon_rect_intersection_can_handle(exploded_left, exploded_right):
        part_result = polygon_rect_intersection(
            exploded_left,
            exploded_right,
            dispatch_mode=ExecutionMode.GPU,
        )
        if part_result is not None and part_result.row_count == exploded_left.row_count:
            d_valid_parts = polygon_parts.selection.active_capacity_mask() & cp.asarray(
                part_result._ensure_device_state().validity,
                dtype=cp.bool_,
            )
            return _pack_disjoint_multipart_intersection_parts_gpu(
                part_result,
                d_source_rows,
                output_row_count=left.row_count,
                assume_disjoint=True,
                d_valid_rows_mask=d_valid_parts,
            )

    if polygon_rect_intersection_can_handle(exploded_right, exploded_left):
        part_result = polygon_rect_intersection(
            exploded_right,
            exploded_left,
            dispatch_mode=ExecutionMode.GPU,
        )
        if part_result is not None and part_result.row_count == exploded_left.row_count:
            d_valid_parts = polygon_parts.selection.active_capacity_mask() & cp.asarray(
                part_result._ensure_device_state().validity,
                dtype=cp.bool_,
            )
            return _pack_disjoint_multipart_intersection_parts_gpu(
                part_result,
                d_source_rows,
                output_row_count=left.row_count,
                assume_disjoint=True,
                d_valid_rows_mask=d_valid_parts,
            )

    part_result = _dispatch_partitioned_polygon_intersection_gpu(
        exploded_left,
        exploded_right,
        dispatch_mode=dispatch_mode,
    )
    if part_result is None:
        part_result = _dispatch_polygon_intersection_overlay_exact_batch_gpu(
            exploded_left,
            exploded_right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=None,
        )
    if part_result is None or part_result.row_count != exploded_left.row_count:
        return None

    d_valid_parts = polygon_parts.selection.active_capacity_mask() & cp.asarray(
        part_result._ensure_device_state().validity,
        dtype=cp.bool_,
    )
    return _pack_disjoint_multipart_intersection_parts_gpu(
        part_result,
        d_source_rows,
        output_row_count=left.row_count,
        assume_disjoint=True,
        d_valid_rows_mask=d_valid_parts,
    )


def _dispatch_multipolygon_polygon_intersection_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Exact GPU rescue for homogeneous MultiPolygon-Polygon intersection."""
    left_rows = _oriented_multipolygon_polygon_source_rows_gpu(left, right)
    if left_rows is not None:
        return _dispatch_oriented_multipolygon_polygon_intersection_gpu(
            left,
            right,
            dispatch_mode=dispatch_mode,
            admitted_rows=left_rows,
        )
    right_rows = _oriented_multipolygon_polygon_source_rows_gpu(right, left)
    if right_rows is None:
        return None
    return _dispatch_oriented_multipolygon_polygon_intersection_gpu(
        right,
        left,
        dispatch_mode=dispatch_mode,
        admitted_rows=right_rows,
    )


def _regroup_native_grouped_parts_with_grouped_union_gpu(
    valid_parts: OwnedGeometryArray,
    d_sorted_order: DeviceArray,
    d_group_offsets: DeviceArray,
    d_group_ids: DeviceArray,
    *,
    output_row_count: int,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    allow_direct_disjoint_pack: bool = False,
    use_same_row_fast_path: bool | None = None,
    same_row_span_summary: tuple[int, int, int] | None = None,
    empty_output: OwnedGeometryArray | None = None,
    group_size_max: int | None = None,
) -> OwnedGeometryArray | None:
    """Regroup `NativeGrouped` polygon rows without a full sorted geometry take.

    `NativeGrouped` already carries the physical grouped shape: rows sorted by
    group plus compact observed-group offsets.  The constructive reducer only
    needs one seed row per observed group and the remaining rows as right-side
    fragments, so materializing a complete sorted `OwnedGeometryArray` is wasted
    dynamic-output work.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    d_sorted_order = cp.asarray(d_sorted_order, dtype=cp.int64)
    d_group_offsets = cp.asarray(d_group_offsets, dtype=cp.int64)
    d_group_ids = cp.asarray(d_group_ids, dtype=cp.int64)
    if int(d_group_offsets.size) != int(d_group_ids.size) + 1:
        return None
    if int(d_group_ids.size) == 0:
        return empty_output or _empty_device_constructive_output(output_row_count)
    if allow_direct_disjoint_pack:
        direct_grouped = _pack_native_grouped_disjoint_polygon_parts_gpu(
            valid_parts,
            d_sorted_order,
            d_group_offsets,
            d_group_ids,
            output_row_count=output_row_count,
            group_size_max=group_size_max,
            empty_output=empty_output,
            assume_all_valid=True,
        )
        if direct_grouped is not None:
            return direct_grouped

    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    _sync_hotpath()
    with hotpath_stage(
        "constructive.intersection.multipart_union.group_rows",
        category="setup",
    ) as amplification_metadata:
        d_group_counts = (d_group_offsets[1:] - d_group_offsets[:-1]).astype(
            cp.int64,
            copy=False,
        )
        from vibespatial.geometry.owned import device_valid_nonempty_mask

        d_group_starts = d_group_offsets[:-1].astype(cp.int64, copy=False)
        d_sorted_positions = cp.arange(d_sorted_order.size, dtype=cp.int64)
        d_group_local = cp.searchsorted(
            d_group_offsets[1:],
            d_sorted_positions,
            side="right",
        ).astype(cp.int64, copy=False)
        d_sorted_nonempty = cp.asarray(
            device_valid_nonempty_mask(valid_parts),
            dtype=cp.bool_,
        )[d_sorted_order]
        position_sentinel = np.uint64(d_sorted_order.size)
        d_seed_sorted_positions = cp.full(
            d_group_ids.size,
            position_sentinel,
            dtype=cp.uint64,
        )
        cp.minimum.at(
            d_seed_sorted_positions,
            d_group_local,
            cp.where(
                d_sorted_nonempty,
                d_sorted_positions.astype(cp.uint64, copy=False),
                position_sentinel,
            ),
        )
        d_seed_sorted_positions = cp.where(
            d_seed_sorted_positions < position_sentinel,
            d_seed_sorted_positions,
            d_group_starts.astype(cp.uint64, copy=False),
        ).astype(cp.int64, copy=False)
        d_effective_order = d_sorted_order.copy()
        d_start_rows = d_effective_order[d_group_starts].copy()
        d_seed_rows = d_effective_order[d_seed_sorted_positions].copy()
        d_effective_order[d_group_starts] = d_seed_rows
        d_effective_order[d_seed_sorted_positions] = d_start_rows
        d_seed_positions = d_effective_order[d_group_starts].astype(cp.int64, copy=False)
        seed_parts = valid_parts._device_indexed_take(
            d_seed_positions,
            assume_unique_indices=True,
        )
        d_group_has_nonempty_rest = None

        if int(seed_parts.row_count) == int(valid_parts.row_count):
            compact = seed_parts
        else:
            d_rest_counts = d_group_counts - 1
            d_rest_offsets = cp.empty(d_rest_counts.size + 1, dtype=cp.int64)
            d_rest_offsets[0] = 0
            cp.cumsum(d_rest_counts, out=d_rest_offsets[1:])
            rest_total = int(d_sorted_order.size) - int(d_group_ids.size)
            d_rest_positions = cp.arange(rest_total, dtype=cp.int64)
            d_rest_group_ids = (
                cp.searchsorted(d_rest_offsets, d_rest_positions, side="right").astype(
                    cp.int64, copy=False
                )
                - 1
            )
            d_rest_sorted_positions = (
                d_rest_positions
                - d_rest_offsets[d_rest_group_ids].astype(cp.int64, copy=False)
                + d_group_offsets[d_rest_group_ids].astype(cp.int64, copy=False)
                + 1
            )
            rest_parts = valid_parts._device_indexed_take(
                d_effective_order[d_rest_sorted_positions].astype(cp.int64, copy=False),
                assume_unique_indices=True,
            )
            d_right_group_rows = d_rest_group_ids.astype(cp.int32, copy=False)
            d_rest_nonempty = cp.asarray(
                device_valid_nonempty_mask(rest_parts),
                dtype=cp.bool_,
            )
            d_rest_counts_by_group = cp.zeros(d_group_ids.size, dtype=cp.int32)
            cp.add.at(
                d_rest_counts_by_group,
                d_right_group_rows,
                d_rest_nonempty.astype(cp.int32, copy=False),
            )
            d_group_has_nonempty_rest = d_rest_counts_by_group > 0
            compact = None
        if amplification_metadata is not None:
            group_sums = {
                "input_rows": int(d_sorted_order.size),
                "pre_reduction_fragments": int(valid_parts.row_count),
                "output_groups": int(d_group_ids.size),
            }
            group_maxima = {"group_capacity": int(output_row_count)}
            group_unavailable = [
                "input_segments",
                "input_coordinates",
                "output_parts",
                "output_coordinates",
            ]
            if group_size_max is None:
                group_unavailable.append("max_group_size")
            else:
                group_maxima["max_group_size"] = int(group_size_max)
            attach_work_amplification(
                amplification_metadata,
                operation="constructive.intersection.multipart_union.group_rows",
                metric_family="group_compression",
                sums=group_sums,
                maxima=group_maxima,
                unavailable=tuple(group_unavailable),
            )
    _sync_hotpath()

    if compact is None:
        _sync_hotpath()
        with hotpath_stage(
            "constructive.intersection.multipart_union.plan.build",
            category="setup",
        ) as amplification_metadata:
            local_same_row_span_summary = same_row_span_summary
            if local_same_row_span_summary is None and bool(use_same_row_fast_path):
                segment_span = _polygon_segment_span_bound(valid_parts)
                if segment_span is not None and group_size_max is not None:
                    from vibespatial.spatial.segment_primitives import (
                        _SAME_ROW_WARP_MAX_RIGHT_SEGMENTS_PER_ROW,
                    )

                    rest_span_bound = int(segment_span) * max(
                        int(group_size_max) - 1,
                        0,
                    )
                    if rest_span_bound <= _SAME_ROW_WARP_MAX_RIGHT_SEGMENTS_PER_ROW:
                        local_same_row_span_summary = (
                            int(segment_span),
                            rest_span_bound,
                            max(int(d_group_ids.size) - 1, 0),
                        )
            use_fast_same_row = (
                True if local_same_row_span_summary is not None else use_same_row_fast_path
            )
            plan = _build_overlay_execution_plan(
                seed_parts,
                rest_parts,
                dispatch_mode=dispatch_mode,
                _cached_right_segments=None,
                _row_isolated=True,
                _include_same_side_splits=True,
                _use_same_row_fast_path=use_fast_same_row,
                _same_row_single_group=(bool(use_fast_same_row) and output_row_count == 1),
                _same_row_span_summary=local_same_row_span_summary,
                _right_geometry_source_rows=d_right_group_rows,
                _right_segment_source_rows=d_right_group_rows,
            )
            if amplification_metadata is not None:
                plan_maxima = {"group_capacity": int(output_row_count)}
                if group_size_max is not None:
                    plan_maxima["max_group_size"] = int(group_size_max)
                plan_unavailable = [
                    "input_segments",
                    "input_coordinates",
                    "output_parts",
                    "output_coordinates",
                ]
                if group_size_max is None:
                    plan_unavailable.append("max_group_size")
                attach_work_amplification(
                    amplification_metadata,
                    operation="constructive.intersection.multipart_union.plan.build",
                    metric_family="group_compression",
                    sums={
                        "input_rows": int(rest_parts.row_count),
                        "pre_reduction_fragments": int(valid_parts.row_count),
                        "output_groups": int(seed_parts.row_count),
                    },
                    maxima=plan_maxima,
                    unavailable=tuple(plan_unavailable),
                )
        _sync_hotpath()
        with hotpath_stage(
            "constructive.intersection.multipart_union.plan.materialize",
            category="refine",
        ) as amplification_metadata:
            compact, _selected = _materialize_overlay_execution_plan(
                plan,
                operation="union",
                requested=ExecutionMode.GPU,
                preserve_row_count=seed_parts.row_count,
            )
            if (
                compact is not None
                and compact.row_count == seed_parts.row_count
                and d_group_has_nonempty_rest is not None
            ):
                compact = device_select_owned_capacity_partitions(
                    compact,
                    [(seed_parts, ~d_group_has_nonempty_rest)],
                )
            if amplification_metadata is not None:
                materialize_sums = {
                    "input_rows": int(rest_parts.row_count),
                    "pre_reduction_fragments": int(valid_parts.row_count),
                }
                materialize_unavailable = [
                    "input_segments",
                    "input_coordinates",
                    "output_parts",
                    "output_coordinates",
                ]
                if compact is None:
                    materialize_unavailable.append("output_groups")
                else:
                    materialize_sums["output_groups"] = int(compact.row_count)
                attach_work_amplification(
                    amplification_metadata,
                    operation="constructive.intersection.multipart_union.plan.materialize",
                    metric_family="group_compression",
                    sums=materialize_sums,
                    maxima={"group_capacity": int(output_row_count)},
                    unavailable=tuple(materialize_unavailable),
                )
        _sync_hotpath()

    if compact is None or compact.row_count != int(d_group_ids.size):
        return None
    if compact.row_count == output_row_count and int(d_group_ids.size) == output_row_count:
        return compact

    return device_concat_owned_scatter(
        empty_output or _empty_device_constructive_output(output_row_count),
        compact,
        d_group_ids.astype(cp.int64, copy=False),
    )


_NATIVE_GROUPED_DISJOINT_PACK_MAX_GROUP_SIZE = 32


def _native_grouped_strict_disjoint_mask_gpu(
    valid_parts: OwnedGeometryArray,
    d_sorted_order: DeviceArray,
    d_group_offsets: DeviceArray,
    *,
    group_size_max: int | None,
) -> DeviceArray | None:
    """Return per-group topological disjointness for a bounded grouped carrier."""
    if cp is None or group_size_max is None:
        return None
    if not _indexed_polygonal_part_capacity_is_bounded(valid_parts):
        return None
    max_group_size = int(group_size_max)
    if max_group_size <= 0 or max_group_size > _NATIVE_GROUPED_DISJOINT_PACK_MAX_GROUP_SIZE:
        return None

    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    d_sorted_order = cp.asarray(d_sorted_order, dtype=cp.int64)
    d_group_offsets = cp.asarray(d_group_offsets, dtype=cp.int64)
    group_count = max(int(d_group_offsets.size) - 1, 0)
    if group_count == 0 or int(d_sorted_order.size) != valid_parts.row_count:
        return None
    d_counts = d_group_offsets[1:] - d_group_offsets[:-1]
    d_slot_ids = cp.arange(max_group_size, dtype=cp.int64)
    d_structural_slot = d_slot_ids[None, :] < d_counts[:, None]
    d_safe_slot_positions = cp.minimum(
        d_group_offsets[:-1, None] + d_slot_ids[None, :],
        np.int64(max(int(d_sorted_order.size) - 1, 0)),
    )
    d_sorted_validity = cp.asarray(
        valid_parts._ensure_device_state(preserve_indexed_view=True).validity,
        dtype=cp.bool_,
    )[d_sorted_order]
    d_all_rows_valid = cp.all(
        ~d_structural_slot | d_sorted_validity[d_safe_slot_positions],
        axis=1,
    )
    pair_capacity = max_group_size * (max_group_size - 1) // 2
    if pair_capacity == 0:
        return (d_counts > 0) & (d_counts <= max_group_size) & d_all_rows_valid

    source_segment_span = _polygon_segment_span_bound(valid_parts)
    if source_segment_span is None:
        return None
    segment_pair_bound = (
        group_count
        * pair_capacity
        * int(source_segment_span)
        * int(source_segment_span)
    )
    if segment_pair_bound > _DIRECT_MULTIPART_PACK_MAX_EXACT_REFINE_SEGMENT_PAIRS:
        return None

    d_left_slots, d_right_slots = cp.triu_indices(max_group_size, k=1)
    d_pair_valid = (d_left_slots[None, :] < d_counts[:, None]) & (
        d_right_slots[None, :] < d_counts[:, None]
    )
    d_safe_left_positions = cp.minimum(
        d_group_offsets[:-1, None] + d_left_slots[None, :],
        np.int64(max(int(d_sorted_order.size) - 1, 0)),
    )
    d_safe_right_positions = cp.minimum(
        d_group_offsets[:-1, None] + d_right_slots[None, :],
        np.int64(max(int(d_sorted_order.size) - 1, 0)),
    )
    d_left_rows = d_sorted_order[d_safe_left_positions]
    d_right_rows = d_sorted_order[d_safe_right_positions]
    d_bounds = cp.asarray(
        compute_geometry_bounds_device(valid_parts, preserve_indexed_view=True),
        dtype=cp.float64,
    ).reshape(valid_parts.row_count, 4)
    d_left_bounds = d_bounds[d_left_rows]
    d_right_bounds = d_bounds[d_right_rows]
    d_bbox_separated = (
        (d_left_bounds[..., 2] < d_right_bounds[..., 0])
        | (d_right_bounds[..., 2] < d_left_bounds[..., 0])
        | (d_left_bounds[..., 3] < d_right_bounds[..., 1])
        | (d_right_bounds[..., 3] < d_left_bounds[..., 1])
    )
    d_exact_pair = d_pair_valid & ~d_bbox_separated

    from vibespatial.predicates.binary import binary_predicate_expression

    pair_rows = group_count * pair_capacity
    left_pairs = _device_take_preserving_indexed_rows(
        valid_parts,
        d_left_rows.reshape(pair_rows),
    )._apply_row_activity(d_exact_pair.reshape(pair_rows))
    right_pairs = _device_take_preserving_indexed_rows(
        valid_parts,
        d_right_rows.reshape(pair_rows),
    )._apply_row_activity(d_exact_pair.reshape(pair_rows))
    disjoint = binary_predicate_expression(
        "disjoint",
        left_pairs,
        right_pairs,
        dispatch_mode=ExecutionMode.GPU,
        operation="constructive.grouped_union.disjoint_admission",
    )
    if disjoint is None:
        return None
    d_topologically_disjoint = cp.asarray(
        disjoint.values,
        dtype=cp.bool_,
    ).reshape(group_count, pair_capacity)
    d_pair_disjoint = d_bbox_separated | (d_exact_pair & d_topologically_disjoint)
    return (
        (d_counts > 0)
        & (d_counts <= max_group_size)
        & d_all_rows_valid
        & cp.all(
            (~d_pair_valid) | d_pair_disjoint,
            axis=1,
        )
    )


def _pack_native_grouped_disjoint_polygon_parts_gpu(
    valid_parts: OwnedGeometryArray,
    d_sorted_order: DeviceArray,
    d_group_offsets: DeviceArray,
    d_group_ids: DeviceArray,
    *,
    output_row_count: int,
    group_size_max: int | None,
    empty_output: OwnedGeometryArray | None = None,
    assume_all_valid: bool = False,
    active_group_mask: DeviceArray | None = None,
    assume_active_groups_disjoint: bool = False,
) -> OwnedGeometryArray | None:
    """Assemble provably disjoint `NativeGrouped` polygon rows without overlay.

    Physical shape: sorted `NativeGrouped` offsets plus polygon family buffers.
    The admission proof is per group, not all-pairs over the whole batch, so
    many small groups can bypass grouped overlay union even when the total row
    count is large.
    """
    if cp is None:
        return None
    if group_size_max is None:
        return None
    max_group_size = int(group_size_max)
    if max_group_size <= 0 or max_group_size > _NATIVE_GROUPED_DISJOINT_PACK_MAX_GROUP_SIZE:
        return None

    d_sorted_order = cp.asarray(d_sorted_order, dtype=cp.int64)
    d_group_offsets = cp.asarray(d_group_offsets, dtype=cp.int64)
    d_group_ids = cp.asarray(d_group_ids, dtype=cp.int64)
    compact_group_count = int(d_group_ids.size)
    if int(d_group_offsets.size) != compact_group_count + 1:
        return None
    if compact_group_count == 0:
        return empty_output or _empty_device_constructive_output(output_row_count)
    if int(d_sorted_order.size) != int(valid_parts.row_count):
        return None

    if not _is_family_only(valid_parts, _POLYGONAL_FAMILIES):
        return None

    d_active_groups = (
        cp.ones(compact_group_count, dtype=cp.bool_)
        if active_group_mask is None
        else cp.asarray(active_group_mask, dtype=cp.bool_)
    )
    if d_active_groups.ndim != 1 or int(d_active_groups.size) != compact_group_count:
        return None

    from vibespatial.api._native_grouped import NativeGroupedSelection
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs

    source_state = valid_parts._ensure_device_state(preserve_indexed_view=True)
    polygon_parts = _explode_polygonal_rows_to_polygon_capacity_gpu(valid_parts)
    if polygon_parts is None or polygon_parts.capacity == 0:
        return empty_output or _empty_device_constructive_output(output_row_count)

    d_sorted_positions = cp.arange(int(d_sorted_order.size), dtype=cp.int64)
    d_sorted_group_local = cp.searchsorted(
        d_group_offsets[1:],
        d_sorted_positions,
        side="right",
    ).astype(cp.int64, copy=False)
    d_source_group_local = cp.empty(valid_parts.row_count, dtype=cp.int64)
    d_source_group_local[d_sorted_order] = d_sorted_group_local
    d_part_group_local = d_source_group_local[
        cp.asarray(polygon_parts.source_rows, dtype=cp.int64)
    ].astype(cp.int64, copy=False)
    d_part_active = polygon_parts.selection.active_capacity_mask()
    d_part_active &= d_active_groups[d_part_group_local]
    grouped_parts = NativeGroupedSelection(
        selection=NativeDeviceSelection.from_mask(d_part_active),
        group_codes=d_part_group_local.astype(cp.int32, copy=False),
        group_count=compact_group_count,
    )
    part_capacity = polygon_parts.capacity
    d_part_counts = grouped_parts.reduce_numeric(
        cp.ones(part_capacity, dtype=cp.int32),
        "count",
    ).values.astype(cp.int32, copy=False)
    d_polygon_group_offsets = cp.empty(compact_group_count + 1, dtype=cp.int64)
    d_polygon_group_offsets[0] = 0
    cp.cumsum(d_part_counts, out=d_polygon_group_offsets[1:])

    if part_capacity > np.iinfo(np.uint32).max:
        return None
    d_sort_groups = cp.where(
        d_part_active,
        d_part_group_local,
        cp.int64(compact_group_count),
    ).astype(cp.uint64, copy=False)
    d_sort_keys = (d_sort_groups << cp.uint64(32)) | cp.arange(part_capacity, dtype=cp.uint64)
    d_polygon_sorted_order = sort_pairs(
        d_sort_keys,
        cp.arange(part_capacity, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    ).values.astype(cp.int64, copy=False)

    sorted_parts = polygon_parts.geometry._device_indexed_take(
        d_polygon_sorted_order,
    )

    polygon_only = set(source_state.families) == {GeometryFamily.POLYGON}
    part_group_size_max = max_group_size
    if not polygon_only and not assume_active_groups_disjoint:
        part_group_size_max = _device_scalar_int(
            cp.max(d_part_counts),
            reason=(
                "binary constructive native grouped disjoint-pack multipart "
                "proof-width admission scalar fence"
            ),
        )
    if (
        part_group_size_max <= 0
        or part_group_size_max > _NATIVE_GROUPED_DISJOINT_PACK_MAX_GROUP_SIZE
    ):
        return None

    if not assume_active_groups_disjoint:
        slot_ids = cp.arange(part_group_size_max, dtype=cp.int32)
        slot_valid = slot_ids[None, :] < d_part_counts[:, None]

        from vibespatial.kernels.core.geometry_analysis import (
            compute_geometry_bounds_device,
        )

        d_bounds = cp.asarray(
            compute_geometry_bounds_device(sorted_parts, preserve_indexed_view=True),
            dtype=cp.float64,
        ).reshape(sorted_parts.row_count, 4)
        slot_positions = d_polygon_group_offsets[:-1, None].astype(
            cp.int64,
            copy=False,
        ) + slot_ids[None, :].astype(cp.int64, copy=False)
        slot_positions = cp.minimum(
            slot_positions,
            cp.asarray(max(int(sorted_parts.row_count) - 1, 0), dtype=cp.int64),
        )
        slot_bounds = d_bounds[slot_positions]
        left_bounds = slot_bounds[:, :, None, :]
        right_bounds = slot_bounds[:, None, :, :]
        pair_valid = (
            slot_valid[:, :, None]
            & slot_valid[:, None, :]
            & (slot_ids[None, :, None] < slot_ids[None, None, :])
        )
        separated = (
            (left_bounds[..., 2] < right_bounds[..., 0])
            | (right_bounds[..., 2] < left_bounds[..., 0])
            | (left_bounds[..., 3] < right_bounds[..., 1])
            | (right_bounds[..., 3] < left_bounds[..., 1])
        )
        d_sorted_group_active = d_active_groups[d_sorted_group_local]
        d_input_valid = cp.asarray(True, dtype=cp.bool_)
        if not assume_all_valid:
            d_sorted_valid = cp.asarray(source_state.validity, dtype=cp.bool_)[d_sorted_order]
            d_input_valid = cp.all(~d_sorted_group_active | d_sorted_valid)
        if not _device_scalar_bool(
            d_input_valid
            & cp.all(~d_active_groups | (d_part_counts > 0))
            & cp.all(~d_active_groups | (d_part_counts <= part_group_size_max))
            & cp.all((~pair_valid) | separated),
            reason=(
                "binary constructive native grouped disjoint-pack bounded "
                "separation admission scalar fence"
            ),
        ):
            return None

    result = _assemble_sorted_polygon_part_capacity_gpu(
        sorted_parts,
        grouped_parts.selection.logical_count,
        d_part_counts,
        d_group_ids.astype(cp.int32, copy=False),
        output_row_count=output_row_count,
        runtime_reason="native grouped disjoint polygon-part capacity assembly",
        ring_capacity=polygon_parts.ring_capacity,
        coord_capacity=polygon_parts.coord_capacity,
    )
    if result is None:
        return None
    if empty_output is not None:
        result = device_concat_owned_scatter(
            empty_output,
            result.device_take(d_group_ids),
            d_group_ids,
        )
    result._native_grouped_union_implementation = "native_grouped_disjoint_pack_union"
    record_dispatch_event(
        surface="vibespatial.constructive.binary_constructive",
        operation="grouped_union",
        implementation="native_grouped_disjoint_pack_union",
        reason=(
            "NativeGrouped polygon rows with per-group separated bounds were "
            "assembled directly as Polygon/MultiPolygon rows without overlay "
            "union topology"
        ),
        detail=(
            f"groups={output_row_count}, observed_groups={compact_group_count}, "
            f"rows={valid_parts.row_count}, max_group_size={part_group_size_max}"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return result


def _polygon_segment_span_bound(owned: OwnedGeometryArray) -> int | None:
    """Return a host-known per-row segment bound without D2H scans."""
    carried_bound = getattr(
        owned,
        "_active_family_row_segment_capacity_bound",
        None,
    )
    if carried_bound is not None:
        return int(carried_bound)
    if owned.is_indexed_view and owned._base is not None:
        base_bound = _polygon_segment_span_bound(owned._base)
        if base_bound is not None:
            owned._active_family_row_segment_capacity_bound = int(base_bound)
            return int(base_bound)
    device_state = getattr(owned, "device_state", None)
    if device_state is not None:
        segment_families = {
            GeometryFamily.LINESTRING,
            GeometryFamily.POLYGON,
            GeometryFamily.MULTILINESTRING,
            GeometryFamily.MULTIPOLYGON,
        }
        family_bounds: list[int] = []
        for family, device_buffer in device_state.families.items():
            if family not in segment_families:
                continue
            width = getattr(device_buffer, "dense_single_ring_width", None)
            if family is GeometryFamily.POLYGON and width is not None and int(width) > 1:
                family_bounds.append(int(width) - 1)
                continue
            fixed_size = getattr(device_buffer, "fixed_size", None)
            coord_bound = (
                None
                if fixed_size is None
                else getattr(fixed_size, "max_coord_count_per_row", None)
            )
            if coord_bound is None:
                family_bounds = []
                break
            family_bounds.append(max(int(coord_bound), 0))
        if family_bounds:
            bound = max(family_bounds)
            owned._active_family_row_segment_capacity_bound = bound
            return bound
        if device_state.trusted_unique_family_rows is True:
            aggregate_coord_capacity = sum(
                int(device_buffer.x.size)
                for family, device_buffer in device_state.families.items()
                if family in segment_families
            )
            if (
                int(owned.row_count) * aggregate_coord_capacity
                <= _INDEXED_AGGREGATE_SEGMENT_BOUND_MAX_LANES
            ):
                owned._active_family_row_segment_capacity_bound = (
                    aggregate_coord_capacity
                )
                return aggregate_coord_capacity
        from vibespatial.geometry.owned import ensure_device_geometry_size_bounds

        return ensure_device_geometry_size_bounds(
            owned,
            reason="constructive device geometry row-size planning packet",
        )
    if set(owned.families) != {GeometryFamily.POLYGON}:
        return None
    buffer = owned.families.get(GeometryFamily.POLYGON)
    if buffer is None:
        return None
    width = getattr(buffer, "dense_single_ring_width", None)
    if width is not None and int(width) > 1:
        bound = int(width) - 1
        owned._active_family_row_segment_capacity_bound = bound
        return bound

    geometry_offsets = getattr(buffer, "geometry_offsets", None)
    ring_offsets = getattr(buffer, "ring_offsets", None)
    empty_mask = getattr(buffer, "empty_mask", None)
    if geometry_offsets is None or ring_offsets is None or empty_mask is None:
        return None
    if (
        hasattr(geometry_offsets, "__cuda_array_interface__")
        or hasattr(ring_offsets, "__cuda_array_interface__")
        or hasattr(empty_mask, "__cuda_array_interface__")
    ):
        return None

    geom = np.asarray(geometry_offsets, dtype=np.int64)
    rings = np.asarray(ring_offsets, dtype=np.int64)
    empty = np.asarray(empty_mask, dtype=bool)
    if geom.ndim != 1 or rings.ndim != 1 or empty.ndim != 1:
        return None
    if int(geom.size) != int(empty.size) + 1 or int(empty.size) != owned.row_count:
        return None
    if int(geom.size) == 0 or int(rings.size) == 0:
        return None
    if int(geom.min(initial=0)) < 0 or int(geom.max(initial=0)) >= int(rings.size):
        return None

    ring_segments = np.maximum(np.diff(rings).astype(np.int64, copy=False) - 1, 0)
    prefix = np.empty(int(ring_segments.size) + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(ring_segments, out=prefix[1:])
    counts = prefix[geom[1:]] - prefix[geom[:-1]]
    counts[empty] = 0
    bound = int(counts.max(initial=0))
    owned._active_family_row_segment_capacity_bound = bound
    return bound


def _apply_polygonal_device_row_segment_bound(
    owned: OwnedGeometryArray,
    segment_bound: int | None,
) -> None:
    """Publish a conservative nested-width proof for constructive polygon rows."""
    if segment_bound is None:
        return
    bound = max(int(segment_bound), 0)
    owned._active_family_row_segment_capacity_bound = bound
    state = owned._ensure_device_state(preserve_indexed_view=True)

    def _tighter(current: int | None, candidate: int) -> int:
        return candidate if current is None else min(int(current), candidate)

    for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        buffer = state.families.get(family)
        if buffer is None:
            continue
        existing = buffer.fixed_size
        first_bound = _tighter(
            None if existing is None else existing.max_first_level_count_per_row,
            bound,
        )
        coord_bound = _tighter(
            None if existing is None else existing.max_coord_count_per_row,
            2 * bound,
        )
        if family is GeometryFamily.POLYGON:
            buffer.fixed_size = DeviceFixedGeometrySizeMetadata(
                first_level_count_per_row=(
                    None if existing is None else existing.first_level_count_per_row
                ),
                coord_count_per_row=(None if existing is None else existing.coord_count_per_row),
                max_first_level_count_per_row=first_bound,
                max_coord_count_per_row=coord_bound,
            )
        else:
            second_bound = _tighter(
                None if existing is None else existing.max_second_level_count_per_row,
                bound,
            )
            buffer.fixed_size = DeviceFixedGeometrySizeMetadata(
                first_level_count_per_row=(
                    None if existing is None else existing.first_level_count_per_row
                ),
                second_level_count_per_row=(
                    None if existing is None else existing.second_level_count_per_row
                ),
                coord_count_per_row=(None if existing is None else existing.coord_count_per_row),
                max_first_level_count_per_row=first_bound,
                max_second_level_count_per_row=second_bound,
                max_coord_count_per_row=coord_bound,
            )


def _polygon_intersection_segment_span_bound(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> int | None:
    """Bound output boundary fragments from aligned source segment spans.

    A source segment can be split by an opposite non-collinear segment once,
    or by both endpoints of a collinear overlap.  Counting both directions
    therefore gives a conservative ``4 * L * R + L + R`` fragment bound for
    every polygonal, lineal, or point remnant emitted by intersection.
    """
    left_bound = _polygon_segment_span_bound(left)
    right_bound = _polygon_segment_span_bound(right)
    if left_bound is None or right_bound is None:
        return None
    return 4 * int(left_bound) * int(right_bound) + int(left_bound) + int(right_bound)


def _assemble_sorted_polygon_part_capacity_gpu(
    sorted_parts: OwnedGeometryArray,
    d_active_part_count: DeviceArray,
    d_part_counts: DeviceArray,
    d_output_source_rows: DeviceArray,
    *,
    output_row_count: int,
    runtime_reason: str,
    ring_capacity: int,
    coord_capacity: int,
    d_valid_empty_rows: DeviceArray | None = None,
) -> OwnedGeometryArray | None:
    """Assemble sorted Polygon part capacity into public output rows."""
    from vibespatial.overlay.assemble import _build_device_resident_polygon_output
    from vibespatial.runtime import RuntimeSelection

    sorted_state = sorted_parts._ensure_device_state(preserve_indexed_view=True)
    polygon_buffer = sorted_state.families.get(GeometryFamily.POLYGON)
    if polygon_buffer is None or polygon_buffer.ring_offsets is None:
        return None

    part_capacity = sorted_parts.row_count
    d_geometry_offsets = cp.asarray(polygon_buffer.geometry_offsets, dtype=cp.int64)
    d_ring_offsets = cp.asarray(polygon_buffer.ring_offsets, dtype=cp.int64)
    d_part_slots = cp.arange(part_capacity, dtype=cp.int64)
    d_active_sorted_parts = (
        d_part_slots
        < cp.asarray(
            d_active_part_count,
            dtype=cp.int64,
        )[0]
    )
    d_part_validity = cp.asarray(sorted_state.validity, dtype=cp.bool_)
    d_part_family_rows = cp.where(
        d_part_validity,
        cp.asarray(sorted_state.family_row_offsets, dtype=cp.int64),
        cp.int64(0),
    )
    d_part_ring_starts = d_geometry_offsets[d_part_family_rows]
    d_rings_per_part = cp.where(
        d_active_sorted_parts & d_part_validity,
        d_geometry_offsets[d_part_family_rows + 1] - d_part_ring_starts,
        cp.int64(0),
    ).astype(cp.int32, copy=False)
    ring_capacity = int(ring_capacity)
    coord_capacity = int(coord_capacity)
    if ring_capacity < 0 or coord_capacity < 0:
        return None
    d_part_ring_offsets = cp.zeros(part_capacity + 1, dtype=cp.int64)
    cp.cumsum(d_rings_per_part, out=d_part_ring_offsets[1:])
    d_ring_slots = cp.arange(ring_capacity, dtype=cp.int64)
    d_ring_logical_count = d_part_ring_offsets[-1]
    d_ring_active = d_ring_slots < d_ring_logical_count
    d_safe_ring_slots = cp.minimum(
        d_ring_slots,
        cp.maximum(d_ring_logical_count - 1, 0),
    )
    d_ring_part_rows = cp.searchsorted(
        d_part_ring_offsets[1:],
        d_safe_ring_slots,
        side="right",
    ).astype(cp.int64, copy=False)
    d_ring_part_rows = cp.minimum(
        d_ring_part_rows,
        max(part_capacity - 1, 0),
    )
    d_ring_local = d_safe_ring_slots - d_part_ring_offsets[d_ring_part_rows]
    d_sorted_output_ids = cp.where(
        d_ring_active,
        d_part_ring_starts[d_ring_part_rows] + d_ring_local,
        cp.int64(0),
    ).astype(cp.int32, copy=False)
    d_sorted_output_edge_counts = cp.where(
        d_ring_active,
        cp.maximum(
            d_ring_offsets[d_sorted_output_ids.astype(cp.int64, copy=False) + 1]
            - d_ring_offsets[d_sorted_output_ids.astype(cp.int64, copy=False)]
            - 1,
            0,
        ),
        cp.int64(0),
    ).astype(cp.int32, copy=False)
    return _build_device_resident_polygon_output(
        d_all_x=cp.asarray(polygon_buffer.x, dtype=cp.float64),
        d_all_y=cp.asarray(polygon_buffer.y, dtype=cp.float64),
        d_all_coord_offsets=d_ring_offsets,
        d_all_edge_counts=None,
        d_sorted_output_ids=d_sorted_output_ids,
        d_rings_per_poly=d_rings_per_part,
        d_polys_per_row=cp.asarray(d_part_counts, dtype=cp.int32),
        d_poly_starts=d_part_ring_offsets[:-1],
        d_output_source_rows=cp.asarray(d_output_source_rows, dtype=cp.int32),
        n_output_rows=int(cp.asarray(d_part_counts).size),
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason=runtime_reason,
        ),
        preserve_row_count=output_row_count,
        d_valid_empty_rows=d_valid_empty_rows,
        coord_capacity=coord_capacity,
        d_sorted_output_edge_counts=d_sorted_output_edge_counts,
    )


def _pack_disjoint_multipart_intersection_capacity_gpu(
    valid_parts: OwnedGeometryArray,
    d_valid_source_rows: DeviceArray,
    *,
    output_row_count: int,
    d_valid_rows_mask: DeviceArray | None = None,
    require_disjoint_proof: bool = False,
) -> OwnedGeometryArray | None:
    """Pack proven-disjoint polygon parts from physical capacity."""
    from vibespatial.api._native_grouped import NativeGroupedSelection
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs

    d_valid_source_rows = cp.asarray(d_valid_source_rows, dtype=cp.int64)
    if int(d_valid_source_rows.size) != valid_parts.row_count:
        return None
    source_state = valid_parts._ensure_device_state(preserve_indexed_view=True)
    d_source_row_valid = cp.asarray(source_state.validity, dtype=cp.bool_)
    d_source_in_range = (d_valid_source_rows >= 0) & (d_valid_source_rows < output_row_count)
    valid_row_selection = NativeDeviceSelection.from_mask(
        d_source_row_valid & d_source_in_range,
    )
    valid_output_groups = NativeGroupedSelection(
        selection=valid_row_selection,
        group_codes=cp.where(
            d_source_in_range,
            d_valid_source_rows,
            cp.int64(0),
        ).astype(cp.int32, copy=False),
        group_count=output_row_count,
    )
    d_output_validity = (
        valid_output_groups.reduce_numeric(
            cp.ones(valid_parts.row_count, dtype=cp.int32),
            "count",
        ).values
        > 0
    )

    polygon_parts = _explode_polygonal_rows_to_polygon_capacity_gpu(valid_parts)
    if polygon_parts is None or polygon_parts.capacity == 0:
        from vibespatial.geometry.owned import build_empty_polygon_rows_device

        return build_empty_polygon_rows_device(
            output_row_count,
            validity=d_output_validity,
        )

    d_part_input_rows = cp.asarray(polygon_parts.source_rows, dtype=cp.int64)
    d_output_rows = d_valid_source_rows[d_part_input_rows]
    d_active = polygon_parts.selection.active_capacity_mask()
    if d_valid_rows_mask is not None:
        d_input_keep = cp.asarray(d_valid_rows_mask, dtype=cp.bool_)
        if int(d_input_keep.size) != valid_parts.row_count:
            return None
        d_active &= d_input_keep[d_part_input_rows]
    d_active &= (d_output_rows >= 0) & (d_output_rows < output_row_count)

    selected_parts = NativeDeviceSelection.from_mask(d_active)
    grouped_parts = NativeGroupedSelection(
        selection=selected_parts,
        group_codes=d_output_rows.astype(cp.int32, copy=False),
        group_count=output_row_count,
    )
    part_capacity = polygon_parts.capacity
    d_part_counts = grouped_parts.reduce_numeric(
        cp.ones(part_capacity, dtype=cp.int32),
        "count",
    ).values.astype(cp.int32, copy=False)

    d_sort_groups = cp.where(
        d_active,
        d_output_rows,
        cp.int64(output_row_count),
    ).astype(cp.uint64, copy=False)
    d_sort_keys = (d_sort_groups << cp.uint64(32)) | cp.arange(part_capacity, dtype=cp.uint64)
    d_order = sort_pairs(
        d_sort_keys,
        cp.arange(part_capacity, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    ).values.astype(cp.int64, copy=False)
    sorted_parts = polygon_parts.geometry._device_indexed_take(d_order)
    d_sorted_output_rows = cp.where(
        d_active[d_order],
        d_output_rows[d_order],
        cp.int64(output_row_count),
    )
    if require_disjoint_proof and not _sorted_polygon_parts_have_strictly_disjoint_group_bounds(
        sorted_parts,
        d_sorted_output_rows,
        d_active_part_count=selected_parts.logical_count,
    ):
        return None
    result = _assemble_sorted_polygon_part_capacity_gpu(
        sorted_parts,
        selected_parts.logical_count,
        d_part_counts,
        d_output_source_rows=cp.arange(output_row_count, dtype=cp.int32),
        output_row_count=output_row_count,
        runtime_reason="proven-disjoint multipart capacity assembly",
        ring_capacity=polygon_parts.ring_capacity,
        coord_capacity=polygon_parts.coord_capacity,
        d_valid_empty_rows=d_output_validity,
    )
    if result is None:
        return None
    record_dispatch_event(
        surface="vibespatial.constructive.binary_constructive",
        operation="intersection",
        implementation="direct_multipart_intersection_capacity_pack_gpu",
        reason=(
            "proven-disjoint multipart fragments were grouped and assembled "
            "from physical polygon-part capacity"
        ),
        detail=(
            f"part_capacity={part_capacity}, rows={output_row_count}; "
            f"exact_disjoint_proof={int(require_disjoint_proof)}; "
            "workload_shape=polygon_part_capacity_grouped_output"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return result


def _pack_disjoint_multipart_intersection_parts_gpu(
    valid_parts: OwnedGeometryArray,
    d_valid_source_rows: DeviceArray,
    *,
    output_row_count: int,
    assume_disjoint: bool = False,
    d_valid_rows_mask: DeviceArray | None = None,
) -> OwnedGeometryArray | None:
    """Regroup disjoint polygonal intersection fragments without overlay union.

    ``valid_parts`` comes from intersecting each exploded MultiPolygon part
    against the aligned mask polygon. Source MultiPolygon parts are disjoint by
    geometry semantics, and clipping them cannot introduce overlap, so regrouping
    only needs to pack fragments back into Polygon/MultiPolygon rows. Reopening
    the overlay union planner here adds topology work without changing area.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if valid_parts.row_count == 0:
        return _empty_device_constructive_output(output_row_count)
    if not _is_family_only(valid_parts, _POLYGONAL_FAMILIES):
        return None
    return _pack_disjoint_multipart_intersection_capacity_gpu(
        valid_parts,
        d_valid_source_rows,
        output_row_count=output_row_count,
        d_valid_rows_mask=d_valid_rows_mask,
        require_disjoint_proof=not assume_disjoint,
    )


def _sorted_polygon_parts_have_strictly_disjoint_group_bounds(
    sorted_parts: OwnedGeometryArray,
    d_sorted_source_rows: DeviceArray,
    *,
    d_active_part_count: DeviceArray | None = None,
) -> bool:
    """Return True when same-source polygon fragments are provably disjoint.

    Bounds are a cheap sufficient proof for most multipart regrouping, but
    curved or triangular fragments can have overlapping envelopes without
    intersecting.  Refine those envelope candidates with the native polygonal
    intersects kernel instead of reopening the grouped overlay union plan.
    """
    part_capacity = sorted_parts.row_count
    if part_capacity <= 1:
        return True
    if part_capacity > _DIRECT_MULTIPART_PACK_MAX_PAIR_PROBE:
        return False

    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    d_pair_i, d_pair_j = cp.triu_indices(part_capacity, k=1)
    d_active_count = (
        cp.asarray(part_capacity, dtype=cp.int64)
        if d_active_part_count is None
        else cp.asarray(d_active_part_count, dtype=cp.int64)[0]
    )
    d_same_source = (
        (d_pair_i < d_active_count)
        & (d_pair_j < d_active_count)
        & (d_sorted_source_rows[d_pair_i] == d_sorted_source_rows[d_pair_j])
    )

    d_bounds = cp.asarray(
        compute_geometry_bounds_device(
            sorted_parts,
            preserve_indexed_view=True,
        ),
        dtype=cp.float64,
    )
    d_left = d_bounds[d_pair_i]
    d_right = d_bounds[d_pair_j]
    d_strictly_separated = (
        (d_left[:, 2] < d_right[:, 0])
        | (d_right[:, 2] < d_left[:, 0])
        | (d_left[:, 3] < d_right[:, 1])
        | (d_right[:, 3] < d_left[:, 1])
    )
    d_candidate_pairs = d_same_source & ~d_strictly_separated
    from vibespatial.api._native_rowset import NativeDeviceSelection

    candidate_selection = NativeDeviceSelection.from_mask(d_candidate_pairs)
    refine_capacity = min(
        candidate_selection.capacity,
        _DIRECT_MULTIPART_PACK_MAX_EXACT_REFINE_PAIRS + 1,
    )
    d_candidate_i = candidate_selection.gather_capacity(
        d_pair_i,
        fill_value=0,
    )[:refine_capacity].astype(cp.int32, copy=False)
    d_candidate_j = candidate_selection.gather_capacity(
        d_pair_j,
        fill_value=0,
    )[:refine_capacity].astype(cp.int32, copy=False)
    d_refine_count = cp.minimum(
        cp.asarray(candidate_selection.logical_count, dtype=cp.int64),
        np.int64(refine_capacity),
    ).astype(cp.int32, copy=False)

    from vibespatial.predicates.polygon import compute_polygonal_intersects_gpu

    d_intersects = compute_polygonal_intersects_gpu(
        sorted_parts,
        sorted_parts,
        query_family=GeometryFamily.POLYGON,
        tree_family=GeometryFamily.POLYGON,
        d_left=d_candidate_i,
        d_right=d_candidate_j,
        d_pair_count=d_refine_count,
        pair_capacity=refine_capacity,
        return_device=True,
    )
    if d_intersects is None:
        return False
    d_refine_active = cp.arange(refine_capacity, dtype=cp.int64) < d_refine_count[0]
    admitted = _device_scalar_bool(
        (candidate_selection.logical_count[0] <= _DIRECT_MULTIPART_PACK_MAX_EXACT_REFINE_PAIRS)
        & ~cp.any(cp.asarray(d_intersects, dtype=cp.bool_) & d_refine_active),
        reason=("binary constructive multipart disjoint capacity-refine admission scalar fence"),
    )
    return admitted


def _assemble_disjoint_polygonal_pieces_gpu(
    pieces: list[OwnedGeometryArray],
) -> OwnedGeometryArray | None:
    """Assemble disjoint single-row polygonal pieces into one union geometry."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    live_pieces = [piece for piece in pieces if piece.row_count > 0]
    if not live_pieces:
        return _empty_device_constructive_output(1)
    merged = live_pieces[0] if len(live_pieces) == 1 else OwnedGeometryArray.concat(live_pieces)
    return _pack_disjoint_multipart_intersection_parts_gpu(
        merged,
        cp.zeros(merged.row_count, dtype=cp.int64),
        output_row_count=1,
        assume_disjoint=True,
    )


@dataclass(frozen=True)
class PolygonPartCapacitySelection:
    """Row-indirected Polygon parts with device-resident logical cardinality."""

    geometry: OwnedGeometryArray
    source_rows: DeviceArray
    selection: object
    ring_capacity: int
    coord_capacity: int

    def __post_init__(self) -> None:
        if int(self.geometry.row_count) != int(self.selection.capacity):
            raise ValueError("polygon-part geometry rows must match selection capacity")
        if int(self.source_rows.size) != int(self.selection.capacity):
            raise ValueError("polygon-part source rows must match selection capacity")
        if int(self.ring_capacity) < 0 or int(self.coord_capacity) < 0:
            raise ValueError("polygon-part structural capacities must be nonnegative")

    @property
    def capacity(self) -> int:
        return int(self.selection.capacity)

    @property
    def logical_count(self):
        return self.selection.logical_count


def _polygon_family_part_capacity(
    owned: OwnedGeometryArray,
    state,
) -> tuple[OwnedGeometryArray, DeviceArray, DeviceArray, int, int] | None:
    """Expose physical Polygon family rows as a validity-selected capacity."""
    buffer = state.families.get(GeometryFamily.POLYGON)
    if buffer is None or buffer.ring_offsets is None:
        return None
    capacity = max(int(buffer.geometry_offsets.size) - 1, 0)
    if capacity == 0:
        return None

    d_global_rows = cp.arange(owned.row_count, dtype=cp.int32)
    d_is_polygon = cp.asarray(state.validity, dtype=cp.bool_) & (
        cp.asarray(state.tags, dtype=cp.int8) == np.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    )
    d_family_rows = cp.where(
        d_is_polygon,
        cp.asarray(state.family_row_offsets, dtype=cp.int64),
        cp.int64(0),
    )
    d_nonempty_polygon = (
        d_is_polygon
        & ~cp.asarray(
            buffer.empty_mask,
            dtype=cp.bool_,
        )[d_family_rows]
    )
    d_encoded_sources = cp.zeros(capacity, dtype=cp.int32)
    cp.maximum.at(
        d_encoded_sources,
        d_family_rows,
        cp.where(d_nonempty_polygon, d_global_rows + 1, cp.int32(0)),
    )
    d_active = d_encoded_sources > 0
    d_source_rows = cp.maximum(d_encoded_sources - 1, 0).astype(
        cp.int32,
        copy=False,
    )
    geometry = build_device_resident_owned(
        device_families={GeometryFamily.POLYGON: buffer},
        row_count=capacity,
        tags=cp.full(
            capacity,
            FAMILY_TAGS[GeometryFamily.POLYGON],
            dtype=cp.int8,
        ),
        validity=d_active,
        family_row_offsets=cp.arange(capacity, dtype=cp.int32),
        execution_mode="gpu",
    )
    return (
        geometry,
        d_source_rows,
        d_active,
        max(int(buffer.ring_offsets.size) - 1, 0),
        int(buffer.x.size),
    )


def _multipolygon_part_capacity(
    owned: OwnedGeometryArray,
    state,
) -> tuple[OwnedGeometryArray, DeviceArray, DeviceArray, int, int] | None:
    """Reinterpret physical MultiPolygon parts as row-indirected Polygons."""
    buffer = state.families.get(GeometryFamily.MULTIPOLYGON)
    if buffer is None or buffer.part_offsets is None or buffer.ring_offsets is None:
        return None
    family_capacity = max(int(buffer.geometry_offsets.size) - 1, 0)
    part_capacity = max(int(buffer.part_offsets.size) - 1, 0)
    if family_capacity == 0 or part_capacity == 0:
        return None

    d_global_rows = cp.arange(owned.row_count, dtype=cp.int32)
    d_is_multipolygon = cp.asarray(state.validity, dtype=cp.bool_) & (
        cp.asarray(state.tags, dtype=cp.int8) == np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
    )
    d_family_rows = cp.where(
        d_is_multipolygon,
        cp.asarray(state.family_row_offsets, dtype=cp.int64),
        cp.int64(0),
    )
    d_encoded_sources = cp.zeros(family_capacity, dtype=cp.int32)
    cp.maximum.at(
        d_encoded_sources,
        d_family_rows,
        cp.where(d_is_multipolygon, d_global_rows + 1, cp.int32(0)),
    )
    d_family_active = d_encoded_sources > 0
    d_family_sources = cp.maximum(d_encoded_sources - 1, 0).astype(
        cp.int32,
        copy=False,
    )

    d_part_slots = cp.arange(part_capacity, dtype=cp.int64)
    d_logical_part_count = cp.asarray(buffer.geometry_offsets, dtype=cp.int64)[-1]
    d_active_part_slots = d_part_slots < d_logical_part_count
    d_safe_part_slots = cp.minimum(
        d_part_slots,
        cp.maximum(d_logical_part_count - 1, 0),
    )
    d_part_family_rows = cp.searchsorted(
        cp.asarray(buffer.geometry_offsets, dtype=cp.int64)[1:],
        d_safe_part_slots,
        side="right",
    ).astype(cp.int64, copy=False)
    d_part_family_rows = cp.minimum(d_part_family_rows, family_capacity - 1)
    d_active = d_active_part_slots & d_family_active[d_part_family_rows]
    d_source_rows = cp.where(
        d_active,
        d_family_sources[d_part_family_rows],
        cp.int32(0),
    )

    polygon_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=buffer.x,
        y=buffer.y,
        geometry_offsets=buffer.part_offsets,
        empty_mask=~d_active,
        ring_offsets=buffer.ring_offsets,
        bounds=None,
    )
    geometry = build_device_resident_owned(
        device_families={GeometryFamily.POLYGON: polygon_buffer},
        row_count=part_capacity,
        tags=cp.full(
            part_capacity,
            FAMILY_TAGS[GeometryFamily.POLYGON],
            dtype=cp.int8,
        ),
        validity=d_active,
        family_row_offsets=cp.arange(part_capacity, dtype=cp.int32),
        execution_mode="gpu",
    )
    return (
        geometry,
        d_source_rows,
        d_active,
        max(int(buffer.ring_offsets.size) - 1, 0),
        int(buffer.x.size),
    )


def _polygon_part_selection_from_capacities(
    partitions: list[tuple[OwnedGeometryArray, DeviceArray, DeviceArray, int, int]],
) -> PolygonPartCapacitySelection | None:
    """Compose polygon-family capacities into one selected-prefix carrier."""
    if not partitions:
        return None
    from vibespatial.api._native_rowset import NativeDeviceSelection

    raw_geometry = (
        partitions[0][0]
        if len(partitions) == 1
        else OwnedGeometryArray.concat([partition[0] for partition in partitions])
    )
    d_raw_source_rows = cp.concatenate(
        [cp.asarray(partition[1], dtype=cp.int32) for partition in partitions]
    )
    d_raw_active = cp.concatenate(
        [cp.asarray(partition[2], dtype=cp.bool_) for partition in partitions]
    )
    raw_selection = NativeDeviceSelection.from_mask(d_raw_active)
    geometry = raw_geometry._device_indexed_take(
        raw_selection.partition_capacity_positions(),
        assume_unique_indices=raw_selection.unique,
    )._apply_row_activity(
        raw_selection.active_capacity_mask(),
        assume_active_indices_unique=raw_selection.unique,
    )
    d_source_rows = raw_selection.gather_capacity(
        d_raw_source_rows,
        fill_value=0,
    ).astype(cp.int32, copy=False)
    return PolygonPartCapacitySelection(
        geometry=geometry,
        source_rows=d_source_rows,
        selection=raw_selection.as_capacity_prefix(),
        ring_capacity=sum(int(partition[3]) for partition in partitions),
        coord_capacity=sum(int(partition[4]) for partition in partitions),
    )


def _fixed_or_max_structural_count(
    fixed_size,
    field: str,
    *,
    structural_upper_bound: int | None = None,
) -> int | None:
    """Return the strongest host-visible per-row structural capacity proof."""
    if fixed_size is None:
        return None if structural_upper_bound is None else int(structural_upper_bound)
    fixed = getattr(fixed_size, field, None)
    if fixed is not None:
        return int(fixed)
    maximum = getattr(fixed_size, f"max_{field}", None)
    if maximum is not None:
        return int(maximum)
    return None if structural_upper_bound is None else int(structural_upper_bound)


def _indexed_polygonal_part_capacity_is_bounded(
    owned: OwnedGeometryArray,
) -> bool:
    """Return whether indexed polygon parts have a physical per-row bound.

    A root buffer total is a valid aggregate bound only when family rows are
    unique.  For duplicated or unresolved row indirection, capacity must come
    from fixed-width metadata; multiplying logical rows by the entire root
    buffer is not an admissible GPU execution shape.
    """
    if not owned.is_indexed_view:
        return True
    state = owned._ensure_device_state(preserve_indexed_view=True)
    if state.trusted_unique_family_rows is True:
        return True

    polygon = state.families.get(GeometryFamily.POLYGON)
    if polygon is not None:
        fixed_size = getattr(polygon, "fixed_size", None)
        dense_width = getattr(polygon, "dense_single_ring_width", None)
        if dense_width is None and (
            _fixed_or_max_structural_count(
                fixed_size,
                "first_level_count_per_row",
            )
            is None
            or _fixed_or_max_structural_count(
                fixed_size,
                "coord_count_per_row",
            )
            is None
        ):
            return False

    multipolygon = state.families.get(GeometryFamily.MULTIPOLYGON)
    if multipolygon is not None:
        fixed_size = getattr(multipolygon, "fixed_size", None)
        if any(
            _fixed_or_max_structural_count(fixed_size, field) is None
            for field in (
                "first_level_count_per_row",
                "second_level_count_per_row",
                "coord_count_per_row",
            )
        ):
            return False
    return True


def _indexed_polygonal_part_capacities(
    owned: OwnedGeometryArray,
    state,
    *,
    max_parts_per_row: int | None = None,
    max_rings_per_row: int | None = None,
    max_coords_per_row: int | None = None,
) -> list[tuple[OwnedGeometryArray, DeviceArray, DeviceArray, int, int]] | None:
    """Expose indexed Polygon parts through logical row/part-slot capacity."""
    from vibespatial.geometry.owned import ensure_device_geometry_size_bounds

    explicit_part_bound = max_parts_per_row is not None
    explicit_ring_bound = max_rings_per_row is not None
    explicit_coord_bound = max_coords_per_row is not None
    unique_family_rows = state.trusted_unique_family_rows is True
    carried_segment_bound = (
        None if unique_family_rows else _polygon_segment_span_bound(owned)
    )
    if not unique_family_rows and carried_segment_bound is None:
        carried_segment_bound = ensure_device_geometry_size_bounds(
            owned,
            reason="constructive indexed polygon-part size planning packet",
        )
    if carried_segment_bound is not None:
        if max_parts_per_row is None:
            max_parts_per_row = int(carried_segment_bound)
        if max_rings_per_row is None:
            max_rings_per_row = int(carried_segment_bound)
        if max_coords_per_row is None:
            max_coords_per_row = 2 * int(carried_segment_bound)
    for name, value in (
        ("max_parts_per_row", max_parts_per_row),
        ("max_rings_per_row", max_rings_per_row),
        ("max_coords_per_row", max_coords_per_row),
    ):
        if value is not None and int(value) < 0:
            raise ValueError(f"{name} must be nonnegative")

    d_validity = cp.asarray(state.validity, dtype=cp.bool_)
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)
    d_source_rows = cp.arange(owned.row_count, dtype=cp.int32)
    partitions: list[tuple[OwnedGeometryArray, DeviceArray, DeviceArray, int, int]] = []
    polygon = state.families.get(GeometryFamily.POLYGON)
    if polygon is not None:
        family_capacity = max(int(polygon.geometry_offsets.size) - 1, 0)
        if family_capacity > 0:
            fixed_size = getattr(polygon, "fixed_size", None)
            fixed_ring_count = _fixed_or_max_structural_count(
                fixed_size,
                "first_level_count_per_row",
                structural_upper_bound=max_rings_per_row,
            )
            fixed_coord_count = _fixed_or_max_structural_count(
                fixed_size,
                "coord_count_per_row",
                structural_upper_bound=max_coords_per_row,
            )
            if not unique_family_rows and max_rings_per_row is not None:
                fixed_ring_count = max(
                    0 if fixed_ring_count is None else int(fixed_ring_count),
                    int(max_rings_per_row),
                )
            if not unique_family_rows and max_coords_per_row is not None:
                fixed_coord_count = max(
                    0 if fixed_coord_count is None else int(fixed_coord_count),
                    int(max_coords_per_row),
                )
            if fixed_ring_count is None and polygon.dense_single_ring_width is not None:
                fixed_ring_count = 1
                fixed_coord_count = int(polygon.dense_single_ring_width)
            if (
                explicit_ring_bound
                and explicit_coord_bound
                and fixed_ring_count is not None
                and fixed_coord_count is not None
            ):
                ring_capacity = min(
                    max(int(polygon.ring_offsets.size) - 1, 0),
                    owned.row_count * int(fixed_ring_count),
                )
                coord_capacity = min(
                    int(polygon.x.size),
                    owned.row_count * int(fixed_coord_count),
                )
            elif unique_family_rows:
                ring_capacity = max(int(polygon.ring_offsets.size) - 1, 0)
                coord_capacity = int(polygon.x.size)
            elif fixed_ring_count is not None and fixed_coord_count is not None:
                ring_capacity = min(
                    max(int(polygon.ring_offsets.size) - 1, 0),
                    owned.row_count * int(fixed_ring_count),
                )
                coord_capacity = min(
                    int(polygon.x.size),
                    owned.row_count * int(fixed_coord_count),
                )
            else:
                return None
            d_polygon = d_validity & (d_tags == np.int8(FAMILY_TAGS[GeometryFamily.POLYGON]))
            d_in_bounds = d_polygon & (d_family_rows >= 0) & (d_family_rows < family_capacity)
            d_safe_rows = cp.where(d_in_bounds, d_family_rows, cp.int64(0))
            d_active = (
                d_in_bounds
                & ~cp.asarray(
                    polygon.empty_mask,
                    dtype=cp.bool_,
                )[d_safe_rows]
            )
            base = build_device_resident_owned(
                device_families={GeometryFamily.POLYGON: polygon},
                row_count=family_capacity,
                tags=cp.full(
                    family_capacity,
                    FAMILY_TAGS[GeometryFamily.POLYGON],
                    dtype=cp.int8,
                ),
                validity=cp.ones(family_capacity, dtype=cp.bool_),
                family_row_offsets=cp.arange(family_capacity, dtype=cp.int32),
                execution_mode="gpu",
            )
            geometry = base._device_indexed_take(d_safe_rows)._apply_row_activity(d_active)
            partitions.append(
                (
                    geometry,
                    d_source_rows,
                    d_active,
                    ring_capacity,
                    coord_capacity,
                )
            )

    multipolygon = state.families.get(GeometryFamily.MULTIPOLYGON)
    if multipolygon is not None:
        if multipolygon.part_offsets is None or multipolygon.ring_offsets is None:
            return None
        fixed_size = getattr(multipolygon, "fixed_size", None)
        family_capacity = max(int(multipolygon.geometry_offsets.size) - 1, 0)
        part_capacity = max(int(multipolygon.part_offsets.size) - 1, 0)
        part_width = (
            _fixed_or_max_structural_count(
                fixed_size,
                "first_level_count_per_row",
                structural_upper_bound=max_parts_per_row,
            )
            if explicit_part_bound or not unique_family_rows
            else None
        )
        if part_width is None and not unique_family_rows:
            return None
        if not unique_family_rows and max_parts_per_row is not None:
            part_width = max(
                0 if part_width is None else int(part_width),
                int(max_parts_per_row),
            )
        part_width = part_capacity if part_width is None else int(part_width)
        if unique_family_rows:
            logical_capacity = (
                min(part_capacity, owned.row_count * part_width)
                if explicit_part_bound
                else part_capacity
            )
        else:
            logical_capacity = owned.row_count * part_width
        if family_capacity > 0 and part_capacity > 0 and logical_capacity > 0:
            fixed_ring_count = _fixed_or_max_structural_count(
                fixed_size,
                "second_level_count_per_row",
                structural_upper_bound=max_rings_per_row,
            )
            fixed_coord_count = _fixed_or_max_structural_count(
                fixed_size,
                "coord_count_per_row",
                structural_upper_bound=max_coords_per_row,
            )
            if not unique_family_rows and max_rings_per_row is not None:
                fixed_ring_count = max(
                    0 if fixed_ring_count is None else int(fixed_ring_count),
                    int(max_rings_per_row),
                )
            if not unique_family_rows and max_coords_per_row is not None:
                fixed_coord_count = max(
                    0 if fixed_coord_count is None else int(fixed_coord_count),
                    int(max_coords_per_row),
                )
            if (
                explicit_ring_bound
                and explicit_coord_bound
                and fixed_ring_count is not None
                and fixed_coord_count is not None
            ):
                ring_capacity = min(
                    max(int(multipolygon.ring_offsets.size) - 1, 0),
                    owned.row_count * int(fixed_ring_count),
                )
                coord_capacity = min(
                    int(multipolygon.x.size),
                    owned.row_count * int(fixed_coord_count),
                )
            elif unique_family_rows:
                ring_capacity = max(int(multipolygon.ring_offsets.size) - 1, 0)
                coord_capacity = int(multipolygon.x.size)
            elif fixed_ring_count is not None and fixed_coord_count is not None:
                ring_capacity = min(
                    max(int(multipolygon.ring_offsets.size) - 1, 0),
                    owned.row_count * int(fixed_ring_count),
                )
                coord_capacity = min(
                    int(multipolygon.x.size),
                    owned.row_count * int(fixed_coord_count),
                )
            else:
                return None
            if unique_family_rows:
                d_multipolygon = d_validity & (
                    d_tags == np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
                )
                d_in_bounds = (
                    d_multipolygon & (d_family_rows >= 0) & (d_family_rows < family_capacity)
                )
                d_safe_family_rows = cp.where(
                    d_in_bounds,
                    d_family_rows,
                    cp.int64(0),
                )
                d_encoded_source_by_family = cp.zeros(
                    family_capacity,
                    dtype=cp.int32,
                )
                cp.maximum.at(
                    d_encoded_source_by_family,
                    d_safe_family_rows,
                    cp.where(d_in_bounds, d_source_rows + 1, cp.int32(0)),
                )
                d_part_rows = cp.arange(logical_capacity, dtype=cp.int64)
                d_logical_part_count = cp.asarray(
                    multipolygon.geometry_offsets,
                    dtype=cp.int64,
                )[-1]
                d_part_slot_active = d_part_rows < d_logical_part_count
                d_safe_logical_part_rows = cp.minimum(
                    d_part_rows,
                    cp.maximum(d_logical_part_count - 1, 0),
                )
                d_part_family_rows = cp.searchsorted(
                    cp.asarray(multipolygon.geometry_offsets, dtype=cp.int64)[1:],
                    d_safe_logical_part_rows,
                    side="right",
                ).astype(cp.int64, copy=False)
                d_part_family_rows = cp.minimum(
                    d_part_family_rows,
                    family_capacity - 1,
                )
                d_encoded_sources = d_encoded_source_by_family[d_part_family_rows]
                d_active = d_part_slot_active & (d_encoded_sources > 0)
                d_rows = cp.maximum(d_encoded_sources - 1, 0).astype(
                    cp.int32,
                    copy=False,
                )
                d_safe_part_rows = cp.where(
                    d_active,
                    d_part_rows,
                    cp.int64(0),
                )
            else:
                d_lanes = cp.arange(logical_capacity, dtype=cp.int64)
                d_rows = d_lanes // part_width
                d_local_parts = d_lanes - (d_rows * part_width)
                d_multipolygon = d_validity[d_rows] & (
                    d_tags[d_rows] == np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
                )
                d_logical_family_rows = d_family_rows[d_rows]
                d_in_bounds = (
                    d_multipolygon
                    & (d_logical_family_rows >= 0)
                    & (d_logical_family_rows < family_capacity)
                )
                d_safe_family_rows = cp.where(
                    d_in_bounds,
                    d_logical_family_rows,
                    cp.int64(0),
                )
                d_part_starts = cp.asarray(
                    multipolygon.geometry_offsets,
                    dtype=cp.int64,
                )[d_safe_family_rows]
                d_part_ends = cp.asarray(
                    multipolygon.geometry_offsets,
                    dtype=cp.int64,
                )[d_safe_family_rows + 1]
                d_part_rows = d_part_starts + d_local_parts
                d_active = d_in_bounds & (d_part_rows < d_part_ends)
                d_safe_part_rows = cp.where(d_active, d_part_rows, cp.int64(0))
            polygon_buffer = DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=multipolygon.x,
                y=multipolygon.y,
                geometry_offsets=multipolygon.part_offsets,
                empty_mask=cp.zeros(part_capacity, dtype=cp.bool_),
                ring_offsets=multipolygon.ring_offsets,
                bounds=None,
            )
            base = build_device_resident_owned(
                device_families={GeometryFamily.POLYGON: polygon_buffer},
                row_count=part_capacity,
                tags=cp.full(
                    part_capacity,
                    FAMILY_TAGS[GeometryFamily.POLYGON],
                    dtype=cp.int8,
                ),
                validity=cp.ones(part_capacity, dtype=cp.bool_),
                family_row_offsets=cp.arange(part_capacity, dtype=cp.int32),
                execution_mode="gpu",
            )
            geometry = base._device_indexed_take(d_safe_part_rows)._apply_row_activity(d_active)
            partitions.append(
                (
                    geometry,
                    d_rows.astype(cp.int32, copy=False),
                    d_active,
                    ring_capacity,
                    coord_capacity,
                )
            )
    return partitions


def _explode_polygonal_rows_to_polygon_capacity_gpu(
    owned: OwnedGeometryArray,
    *,
    max_parts_per_row: int | None = None,
    max_rings_per_row: int | None = None,
    max_coords_per_row: int | None = None,
) -> PolygonPartCapacitySelection | None:
    """Explode polygonal rows at physical part capacity without count fences.

    The returned geometry is an indexed view whose active Polygon parts occupy
    a compact prefix. Coordinate and ring buffers remain shared with the
    physical-capacity carrier; ``selection.logical_count`` is the only dynamic
    cardinality.
    """
    if cp is None:
        return None
    if owned.is_indexed_view:
        indexed_state = owned._ensure_device_state(preserve_indexed_view=True)
        indexed_partitions = _indexed_polygonal_part_capacities(
            owned,
            indexed_state,
            max_parts_per_row=max_parts_per_row,
            max_rings_per_row=max_rings_per_row,
            max_coords_per_row=max_coords_per_row,
        )
        if indexed_partitions is not None:
            return _polygon_part_selection_from_capacities(indexed_partitions)
        return None

    state = owned._ensure_device_state(preserve_indexed_view=False)
    partitions = [
        partition
        for partition in (
            _polygon_family_part_capacity(owned, state),
            _multipolygon_part_capacity(owned, state),
        )
        if partition is not None
    ]
    return _polygon_part_selection_from_capacities(partitions)


_NESTED_MULTIPOLYGON_REPAIR_MAX_PART_CAPACITY = 512
_NESTED_MULTIPOLYGON_REPAIR_MAX_RING_CAPACITY = 16 * 1024
_NESTED_MULTIPOLYGON_REPAIR_MAX_COORD_CAPACITY = 256 * 1024


def _drop_nested_multipolygon_parts_gpu(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """Remove same-row contained Polygon parts from MultiPolygon rows.

    Grouped overlay union can emit otherwise valid polygon parts with a nested
    exterior shell when one input component is wholly covered by another and
    their boundaries never cross.  OGC MultiPolygon validity forbids that
    nesting; union semantics keep only the containing component.  This sparse
    repair stays device-resident and bounds its dense same-row relation by the
    invalid output's physical part capacity.
    """
    if cp is None or owned.row_count == 0:
        return None
    state = owned._ensure_device_state(preserve_indexed_view=True)
    family_count = sum(
        family in state.families
        for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
    )
    if family_count == 0:
        return None
    family_row_slots = owned.row_count * family_count
    part_slot_bound = _NESTED_MULTIPOLYGON_REPAIR_MAX_PART_CAPACITY // family_row_slots
    ring_slot_bound = _NESTED_MULTIPOLYGON_REPAIR_MAX_RING_CAPACITY // family_row_slots
    coord_slot_bound = _NESTED_MULTIPOLYGON_REPAIR_MAX_COORD_CAPACITY // family_row_slots
    if min(part_slot_bound, ring_slot_bound, coord_slot_bound) <= 0:
        return None

    part_bounds: list[int] = []
    ring_bounds: list[int] = []
    coord_bounds: list[int] = []
    d_validity = cp.asarray(state.validity, dtype=cp.bool_)
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)
    polygonal_tags = cp.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=cp.int8,
    )
    # The assembled repair carrier contains polygonal rows only.  Rows from
    # every other family must stay selected from ``owned``; otherwise valid
    # points and lines are replaced by the repair carrier's null lanes.
    d_within_repair_capacity = d_validity & cp.isin(d_tags, polygonal_tags)
    polygon = state.families.get(GeometryFamily.POLYGON)
    if polygon is not None:
        fixed_size = getattr(polygon, "fixed_size", None)
        polygon_rings_bound = _fixed_or_max_structural_count(
            fixed_size,
            "first_level_count_per_row",
        )
        polygon_coords_bound = _fixed_or_max_structural_count(
            fixed_size,
            "coord_count_per_row",
        )
        polygon_rings = min(
            ring_slot_bound,
            ring_slot_bound if polygon_rings_bound is None else int(polygon_rings_bound),
        )
        polygon_coords = min(
            coord_slot_bound,
            coord_slot_bound if polygon_coords_bound is None else int(polygon_coords_bound),
        )
        part_bounds.append(1)
        ring_bounds.append(int(polygon_rings))
        coord_bounds.append(int(polygon_coords))

        family_capacity = max(int(polygon.geometry_offsets.size) - 1, 0)
        d_is_polygon = d_validity & (
            d_tags == np.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
        )
        d_polygon_in_bounds = (
            d_is_polygon
            & (d_family_rows >= 0)
            & (d_family_rows < family_capacity)
        )
        d_safe_polygon_rows = cp.where(
            d_polygon_in_bounds,
            d_family_rows,
            cp.int64(0),
        )
        d_polygon_offsets = cp.asarray(polygon.geometry_offsets, dtype=cp.int64)
        d_polygon_ring_count = (
            d_polygon_offsets[d_safe_polygon_rows + 1]
            - d_polygon_offsets[d_safe_polygon_rows]
        )
        from vibespatial.geometry.owned import device_family_coordinate_counts

        d_polygon_coord_count = device_family_coordinate_counts(
            polygon,
            d_safe_polygon_rows,
        )
        d_polygon_within = (
            d_polygon_in_bounds
            & (d_polygon_ring_count <= polygon_rings)
            & (d_polygon_coord_count <= polygon_coords)
        )
        d_within_repair_capacity &= ~d_is_polygon | d_polygon_within
    multipolygon = state.families.get(GeometryFamily.MULTIPOLYGON)
    if multipolygon is not None:
        fixed_size = getattr(multipolygon, "fixed_size", None)
        multipolygon_parts_bound = _fixed_or_max_structural_count(
            fixed_size,
            "first_level_count_per_row",
        )
        multipolygon_rings_bound = _fixed_or_max_structural_count(
            fixed_size,
            "second_level_count_per_row",
        )
        multipolygon_coords_bound = _fixed_or_max_structural_count(
            fixed_size,
            "coord_count_per_row",
        )
        multipolygon_parts = min(
            part_slot_bound,
            (
                part_slot_bound
                if multipolygon_parts_bound is None
                else int(multipolygon_parts_bound)
            ),
        )
        multipolygon_rings = min(
            ring_slot_bound,
            (
                ring_slot_bound
                if multipolygon_rings_bound is None
                else int(multipolygon_rings_bound)
            ),
        )
        multipolygon_coords = min(
            coord_slot_bound,
            (
                coord_slot_bound
                if multipolygon_coords_bound is None
                else int(multipolygon_coords_bound)
            ),
        )
        part_bounds.append(int(multipolygon_parts))
        ring_bounds.append(int(multipolygon_rings))
        coord_bounds.append(int(multipolygon_coords))

        if multipolygon.part_offsets is None or multipolygon.ring_offsets is None:
            return None
        family_capacity = max(int(multipolygon.geometry_offsets.size) - 1, 0)
        d_is_multipolygon = d_validity & (
            d_tags == np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
        )
        d_multipolygon_in_bounds = (
            d_is_multipolygon
            & (d_family_rows >= 0)
            & (d_family_rows < family_capacity)
        )
        d_safe_multipolygon_rows = cp.where(
            d_multipolygon_in_bounds,
            d_family_rows,
            cp.int64(0),
        )
        d_multipolygon_offsets = cp.asarray(
            multipolygon.geometry_offsets,
            dtype=cp.int64,
        )
        d_multipolygon_part_starts = d_multipolygon_offsets[
            d_safe_multipolygon_rows
        ]
        d_multipolygon_part_ends = d_multipolygon_offsets[
            d_safe_multipolygon_rows + 1
        ]
        d_multipolygon_ring_offsets = cp.asarray(
            multipolygon.part_offsets,
            dtype=cp.int64,
        )
        d_multipolygon_part_count = (
            d_multipolygon_part_ends - d_multipolygon_part_starts
        )
        d_multipolygon_ring_count = (
            d_multipolygon_ring_offsets[d_multipolygon_part_ends]
            - d_multipolygon_ring_offsets[d_multipolygon_part_starts]
        )
        from vibespatial.geometry.owned import device_family_coordinate_counts

        d_multipolygon_coord_count = device_family_coordinate_counts(
            multipolygon,
            d_safe_multipolygon_rows,
        )
        d_multipolygon_within = (
            d_multipolygon_in_bounds
            & (d_multipolygon_part_count <= multipolygon_parts)
            & (d_multipolygon_ring_count <= multipolygon_rings)
            & (d_multipolygon_coord_count <= multipolygon_coords)
        )
        d_within_repair_capacity &= ~d_is_multipolygon | d_multipolygon_within

    indexed = owned._device_indexed_take(
        cp.arange(owned.row_count, dtype=cp.int64),
        assume_unique_indices=True,
    )
    polygon_parts = _explode_polygonal_rows_to_polygon_capacity_gpu(
        indexed,
        max_parts_per_row=max(part_bounds),
        max_rings_per_row=max(ring_bounds),
        max_coords_per_row=max(coord_bounds),
    )
    if polygon_parts is None or polygon_parts.capacity == 0:
        return None
    part_capacity = polygon_parts.capacity
    if part_capacity > _NESTED_MULTIPOLYGON_REPAIR_MAX_PART_CAPACITY:
        return None

    from vibespatial.api._native_grouped import NativeGroupedSelection
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.measurement import _area_gpu_device_fp64
    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs
    from vibespatial.predicates.binary import binary_predicate_expression

    d_active = polygon_parts.selection.active_capacity_mask()
    d_source_rows = cp.asarray(polygon_parts.source_rows, dtype=cp.int32)
    d_rows = cp.arange(part_capacity, dtype=cp.int64)
    d_left_rows = cp.repeat(d_rows, part_capacity)
    d_right_rows = cp.tile(d_rows, part_capacity)
    d_pair_active = (
        d_active[d_left_rows]
        & d_active[d_right_rows]
        & (d_left_rows != d_right_rows)
        & (d_source_rows[d_left_rows] == d_source_rows[d_right_rows])
    )
    left_pairs = polygon_parts.geometry._device_indexed_take(
        d_left_rows,
    )._apply_row_activity(d_pair_active)
    right_pairs = polygon_parts.geometry._device_indexed_take(
        d_right_rows,
    )._apply_row_activity(d_pair_active)
    covered = binary_predicate_expression(
        "covered_by",
        left_pairs,
        right_pairs,
        dispatch_mode=ExecutionMode.GPU,
        operation="constructive.make_valid.nested_multipolygon_parts",
    )
    if covered is None:
        return None

    d_part_area = cp.abs(
        cp.asarray(_area_gpu_device_fp64(polygon_parts.geometry), dtype=cp.float64)
    )
    d_left_area = d_part_area[d_left_rows]
    d_right_area = d_part_area[d_right_rows]
    d_strictly_redundant = cp.asarray(covered.values, dtype=cp.bool_) & (
        (d_left_area < d_right_area)
        | ((d_left_area == d_right_area) & (d_left_rows > d_right_rows))
    )
    d_drop = cp.any(
        d_strictly_redundant.reshape(part_capacity, part_capacity),
        axis=1,
    )
    d_keep = d_active & ~d_drop

    grouped_parts = NativeGroupedSelection(
        selection=NativeDeviceSelection.from_mask(d_keep),
        group_codes=d_source_rows,
        group_count=owned.row_count,
    )
    d_part_counts = grouped_parts.reduce_numeric(
        cp.ones(part_capacity, dtype=cp.int32),
        "count",
    ).values.astype(cp.int32, copy=False)
    d_sort_groups = cp.where(
        d_keep,
        d_source_rows.astype(cp.int64, copy=False),
        cp.int64(owned.row_count),
    ).astype(cp.uint64, copy=False)
    d_sort_keys = (d_sort_groups << cp.uint64(32)) | cp.arange(
        part_capacity,
        dtype=cp.uint64,
    )
    d_order = sort_pairs(
        d_sort_keys,
        cp.arange(part_capacity, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    ).values.astype(cp.int64, copy=False)
    sorted_parts = polygon_parts.geometry._device_indexed_take(d_order)
    result = _assemble_sorted_polygon_part_capacity_gpu(
        sorted_parts,
        grouped_parts.selection.logical_count,
        d_part_counts,
        cp.arange(owned.row_count, dtype=cp.int32),
        output_row_count=owned.row_count,
        runtime_reason="nested multipolygon part canonicalization",
        ring_capacity=polygon_parts.ring_capacity,
        coord_capacity=polygon_parts.coord_capacity,
    )
    if result is not None:
        from vibespatial.geometry.owned import device_select_owned_capacity_partitions

        result = device_select_owned_capacity_partitions(
            owned,
            [(result, d_within_repair_capacity)],
        )
        result._native_nested_multipolygon_parts_canonicalized = True
    return result


@dataclass(frozen=True)
class PointPartCapacitySelection:
    """Row-indirected Point parts with device logical cardinality."""

    geometry: OwnedGeometryArray
    source_rows: DeviceArray
    selection: object

    def __post_init__(self) -> None:
        if int(self.geometry.row_count) != int(self.selection.capacity):
            raise ValueError("point-part geometry rows must match selection capacity")
        if int(self.source_rows.size) != int(self.selection.capacity):
            raise ValueError("point-part source rows must match selection capacity")

    @property
    def capacity(self) -> int:
        return int(self.selection.capacity)

    @property
    def logical_count(self):
        return self.selection.logical_count


def _point_family_part_capacity(
    owned: OwnedGeometryArray,
    state,
) -> tuple[OwnedGeometryArray, DeviceArray, DeviceArray] | None:
    buffer = state.families.get(GeometryFamily.POINT)
    if buffer is None:
        return None
    capacity = max(int(buffer.geometry_offsets.size) - 1, 0)
    if capacity == 0:
        return None

    d_global_rows = cp.arange(owned.row_count, dtype=cp.int32)
    d_is_point = cp.asarray(state.validity, dtype=cp.bool_) & (
        cp.asarray(state.tags, dtype=cp.int8) == np.int8(FAMILY_TAGS[GeometryFamily.POINT])
    )
    d_family_rows = cp.where(
        d_is_point,
        cp.asarray(state.family_row_offsets, dtype=cp.int64),
        cp.int64(0),
    )
    d_nonempty = d_is_point & ~cp.asarray(buffer.empty_mask, dtype=cp.bool_)[d_family_rows]
    d_encoded_sources = cp.zeros(capacity, dtype=cp.int32)
    cp.maximum.at(
        d_encoded_sources,
        d_family_rows,
        cp.where(d_nonempty, d_global_rows + 1, cp.int32(0)),
    )
    d_active = d_encoded_sources > 0
    d_source_rows = cp.maximum(d_encoded_sources - 1, 0).astype(
        cp.int32,
        copy=False,
    )
    geometry = build_device_resident_owned(
        device_families={GeometryFamily.POINT: buffer},
        row_count=capacity,
        tags=cp.full(capacity, FAMILY_TAGS[GeometryFamily.POINT], dtype=cp.int8),
        validity=d_active,
        family_row_offsets=cp.arange(capacity, dtype=cp.int32),
        execution_mode="gpu",
    )
    return geometry, d_source_rows, d_active


def _multipoint_part_capacity(
    owned: OwnedGeometryArray,
    state,
) -> tuple[OwnedGeometryArray, DeviceArray, DeviceArray] | None:
    buffer = state.families.get(GeometryFamily.MULTIPOINT)
    if buffer is None:
        return None
    family_capacity = max(int(buffer.geometry_offsets.size) - 1, 0)
    point_capacity = min(int(buffer.x.size), int(buffer.y.size))
    if family_capacity == 0 or point_capacity == 0:
        return None

    d_global_rows = cp.arange(owned.row_count, dtype=cp.int32)
    d_is_multipoint = cp.asarray(state.validity, dtype=cp.bool_) & (
        cp.asarray(state.tags, dtype=cp.int8) == np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOINT])
    )
    d_family_rows = cp.where(
        d_is_multipoint,
        cp.asarray(state.family_row_offsets, dtype=cp.int64),
        cp.int64(0),
    )
    d_encoded_sources = cp.zeros(family_capacity, dtype=cp.int32)
    cp.maximum.at(
        d_encoded_sources,
        d_family_rows,
        cp.where(d_is_multipoint, d_global_rows + 1, cp.int32(0)),
    )
    d_family_active = d_encoded_sources > 0
    d_family_sources = cp.maximum(d_encoded_sources - 1, 0).astype(
        cp.int32,
        copy=False,
    )

    d_point_slots = cp.arange(point_capacity, dtype=cp.int64)
    d_logical_point_count = cp.asarray(buffer.geometry_offsets, dtype=cp.int64)[-1]
    d_active_point_slots = d_point_slots < d_logical_point_count
    d_safe_point_slots = cp.minimum(
        d_point_slots,
        cp.maximum(d_logical_point_count - 1, 0),
    )
    d_point_family_rows = cp.searchsorted(
        cp.asarray(buffer.geometry_offsets, dtype=cp.int64)[1:],
        d_safe_point_slots,
        side="right",
    ).astype(cp.int64, copy=False)
    d_point_family_rows = cp.minimum(d_point_family_rows, family_capacity - 1)
    d_active = d_active_point_slots & d_family_active[d_point_family_rows]
    d_source_rows = cp.where(
        d_active,
        d_family_sources[d_point_family_rows],
        cp.int32(0),
    )
    point_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        x=buffer.x,
        y=buffer.y,
        geometry_offsets=cp.arange(point_capacity + 1, dtype=cp.int32),
        empty_mask=~d_active,
        bounds=None,
        fixed_size=DeviceFixedGeometrySizeMetadata(coord_count_per_row=1),
    )
    geometry = build_device_resident_owned(
        device_families={GeometryFamily.POINT: point_buffer},
        row_count=point_capacity,
        tags=cp.full(
            point_capacity,
            FAMILY_TAGS[GeometryFamily.POINT],
            dtype=cp.int8,
        ),
        validity=d_active,
        family_row_offsets=cp.arange(point_capacity, dtype=cp.int32),
        execution_mode="gpu",
    )
    return geometry, d_source_rows, d_active


def _point_part_selection_from_capacities(
    partitions: list[tuple[OwnedGeometryArray, DeviceArray, DeviceArray]],
) -> PointPartCapacitySelection | None:
    """Compose point-family capacities into one selected-prefix carrier."""
    if not partitions:
        return None
    from vibespatial.api._native_rowset import NativeDeviceSelection

    raw_geometry = (
        partitions[0][0]
        if len(partitions) == 1
        else OwnedGeometryArray.concat([partition[0] for partition in partitions])
    )
    d_raw_source_rows = cp.concatenate(
        [cp.asarray(partition[1], dtype=cp.int32) for partition in partitions]
    )
    d_raw_active = cp.concatenate(
        [cp.asarray(partition[2], dtype=cp.bool_) for partition in partitions]
    )
    raw_selection = NativeDeviceSelection.from_mask(d_raw_active)
    geometry = raw_geometry._device_indexed_take(
        raw_selection.partition_capacity_positions(),
        assume_unique_indices=raw_selection.unique,
    )._apply_row_activity(
        raw_selection.active_capacity_mask(),
        assume_active_indices_unique=raw_selection.unique,
    )
    d_source_rows = raw_selection.gather_capacity(
        d_raw_source_rows,
        fill_value=0,
    ).astype(cp.int32, copy=False)
    return PointPartCapacitySelection(
        geometry=geometry,
        source_rows=d_source_rows,
        selection=raw_selection.as_capacity_prefix(),
    )


def _indexed_point_part_capacities(
    owned: OwnedGeometryArray,
    state,
) -> list[tuple[OwnedGeometryArray, DeviceArray, DeviceArray]] | None:
    """Expose indexed point parts through logical row/point-slot capacity."""
    d_validity = cp.asarray(state.validity, dtype=cp.bool_)
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)
    d_source_rows = cp.arange(owned.row_count, dtype=cp.int32)
    partitions: list[tuple[OwnedGeometryArray, DeviceArray, DeviceArray]] = []
    unique_family_rows = state.trusted_unique_family_rows is True

    point = state.families.get(GeometryFamily.POINT)
    if point is not None:
        family_capacity = max(int(point.geometry_offsets.size) - 1, 0)
        if family_capacity > 0:
            d_is_point = d_validity & (d_tags == np.int8(FAMILY_TAGS[GeometryFamily.POINT]))
            d_in_bounds = d_is_point & (d_family_rows >= 0) & (d_family_rows < family_capacity)
            d_safe_rows = cp.where(d_in_bounds, d_family_rows, cp.int64(0))
            d_active = (
                d_in_bounds
                & ~cp.asarray(
                    point.empty_mask,
                    dtype=cp.bool_,
                )[d_safe_rows]
            )
            base = build_device_resident_owned(
                device_families={GeometryFamily.POINT: point},
                row_count=family_capacity,
                tags=cp.full(
                    family_capacity,
                    FAMILY_TAGS[GeometryFamily.POINT],
                    dtype=cp.int8,
                ),
                validity=cp.ones(family_capacity, dtype=cp.bool_),
                family_row_offsets=cp.arange(family_capacity, dtype=cp.int32),
                execution_mode="gpu",
            )
            geometry = base._device_indexed_take(
                d_safe_rows,
            )._apply_row_activity(d_active)
            partitions.append((geometry, d_source_rows, d_active))

    multipoint = state.families.get(GeometryFamily.MULTIPOINT)
    if multipoint is not None:
        family_capacity = max(int(multipoint.geometry_offsets.size) - 1, 0)
        point_capacity = min(int(multipoint.x.size), int(multipoint.y.size))
        if family_capacity > 0 and point_capacity > 0:
            if unique_family_rows:
                partition = _multipoint_part_capacity(owned, state)
                if partition is not None:
                    partitions.append(partition)
            else:
                fixed_size = getattr(multipoint, "fixed_size", None)
                point_width = _fixed_or_max_structural_count(
                    fixed_size,
                    "coord_count_per_row",
                    structural_upper_bound=point_capacity,
                )
                if point_width is None:
                    return None
                point_width = int(point_width)
                if point_width <= 0:
                    return None
                logical_capacity = owned.row_count * point_width
                d_lanes = cp.arange(logical_capacity, dtype=cp.int64)
                d_rows = d_lanes // point_width
                d_local_points = d_lanes - (d_rows * point_width)
                d_is_multipoint = d_validity[d_rows] & (
                    d_tags[d_rows] == np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOINT])
                )
                d_logical_family_rows = d_family_rows[d_rows]
                d_in_bounds = (
                    d_is_multipoint
                    & (d_logical_family_rows >= 0)
                    & (d_logical_family_rows < family_capacity)
                )
                d_safe_family_rows = cp.where(
                    d_in_bounds,
                    d_logical_family_rows,
                    cp.int64(0),
                )
                d_point_starts = cp.asarray(
                    multipoint.geometry_offsets,
                    dtype=cp.int64,
                )[d_safe_family_rows]
                d_point_ends = cp.asarray(
                    multipoint.geometry_offsets,
                    dtype=cp.int64,
                )[d_safe_family_rows + 1]
                d_point_rows = d_point_starts + d_local_points
                d_active = d_in_bounds & (d_point_rows < d_point_ends)
                d_safe_point_rows = cp.where(d_active, d_point_rows, cp.int64(0))
                point_buffer = DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.POINT,
                    x=multipoint.x,
                    y=multipoint.y,
                    geometry_offsets=cp.arange(point_capacity + 1, dtype=cp.int32),
                    empty_mask=cp.zeros(point_capacity, dtype=cp.bool_),
                    bounds=None,
                    fixed_size=DeviceFixedGeometrySizeMetadata(
                        coord_count_per_row=1,
                    ),
                )
                base = build_device_resident_owned(
                    device_families={GeometryFamily.POINT: point_buffer},
                    row_count=point_capacity,
                    tags=cp.full(
                        point_capacity,
                        FAMILY_TAGS[GeometryFamily.POINT],
                        dtype=cp.int8,
                    ),
                    validity=cp.ones(point_capacity, dtype=cp.bool_),
                    family_row_offsets=cp.arange(point_capacity, dtype=cp.int32),
                    execution_mode="gpu",
                )
                geometry = base._device_indexed_take(
                    d_safe_point_rows,
                )._apply_row_activity(d_active)
                partitions.append(
                    (
                        geometry,
                        d_rows.astype(cp.int32, copy=False),
                        d_active,
                    )
                )
    return partitions


def _explode_point_rows_to_point_capacity_gpu(
    owned: OwnedGeometryArray,
) -> PointPartCapacitySelection | None:
    """Explode pointlike rows at physical point capacity without count reads."""
    if cp is None:
        return None
    if owned.is_indexed_view:
        indexed_state = owned._ensure_device_state(preserve_indexed_view=True)
        indexed_partitions = _indexed_point_part_capacities(
            owned,
            indexed_state,
        )
        if indexed_partitions is not None:
            return _point_part_selection_from_capacities(indexed_partitions)
        return None

    state = owned._ensure_device_state(preserve_indexed_view=False)
    partitions = [
        partition
        for partition in (
            _point_family_part_capacity(owned, state),
            _multipoint_part_capacity(owned, state),
        )
        if partition is not None
    ]
    return _point_part_selection_from_capacities(partitions)


@dataclass(frozen=True)
class LinePartCapacitySelection:
    """Row-indirected LineString parts with device logical cardinality."""

    geometry: OwnedGeometryArray
    source_rows: DeviceArray
    selection: object
    coord_capacity: int

    def __post_init__(self) -> None:
        if int(self.geometry.row_count) != int(self.selection.capacity):
            raise ValueError("line-part geometry rows must match selection capacity")
        if int(self.source_rows.size) != int(self.selection.capacity):
            raise ValueError("line-part source rows must match selection capacity")
        if int(self.coord_capacity) < 0:
            raise ValueError("line-part coordinate capacity must be nonnegative")

    @property
    def capacity(self) -> int:
        return int(self.selection.capacity)

    @property
    def logical_count(self):
        return self.selection.logical_count


def _linestring_family_part_capacity(
    owned: OwnedGeometryArray,
    state,
) -> tuple[OwnedGeometryArray, DeviceArray, DeviceArray, int] | None:
    """Expose physical LineString family rows as selected part capacity."""
    buffer = state.families.get(GeometryFamily.LINESTRING)
    if buffer is None:
        return None
    capacity = max(int(buffer.geometry_offsets.size) - 1, 0)
    if capacity == 0:
        return None

    d_global_rows = cp.arange(owned.row_count, dtype=cp.int32)
    d_is_line = cp.asarray(state.validity, dtype=cp.bool_) & (
        cp.asarray(state.tags, dtype=cp.int8) == np.int8(FAMILY_TAGS[GeometryFamily.LINESTRING])
    )
    d_family_rows = cp.where(
        d_is_line,
        cp.asarray(state.family_row_offsets, dtype=cp.int64),
        cp.int64(0),
    )
    d_nonempty = d_is_line & ~cp.asarray(buffer.empty_mask, dtype=cp.bool_)[d_family_rows]
    d_encoded_sources = cp.zeros(capacity, dtype=cp.int32)
    cp.maximum.at(
        d_encoded_sources,
        d_family_rows,
        cp.where(d_nonempty, d_global_rows + 1, cp.int32(0)),
    )
    d_active = d_encoded_sources > 0
    d_source_rows = cp.maximum(d_encoded_sources - 1, 0).astype(
        cp.int32,
        copy=False,
    )
    geometry = build_device_resident_owned(
        device_families={GeometryFamily.LINESTRING: buffer},
        row_count=capacity,
        tags=cp.full(
            capacity,
            FAMILY_TAGS[GeometryFamily.LINESTRING],
            dtype=cp.int8,
        ),
        validity=d_active,
        family_row_offsets=cp.arange(capacity, dtype=cp.int32),
        execution_mode="gpu",
    )
    return geometry, d_source_rows, d_active, int(buffer.x.size)


def _multilinestring_part_capacity(
    owned: OwnedGeometryArray,
    state,
) -> tuple[OwnedGeometryArray, DeviceArray, DeviceArray, int] | None:
    """Reinterpret physical MultiLineString parts as LineString capacity."""
    buffer = state.families.get(GeometryFamily.MULTILINESTRING)
    if buffer is None or buffer.part_offsets is None:
        return None
    family_capacity = max(int(buffer.geometry_offsets.size) - 1, 0)
    part_capacity = max(int(buffer.part_offsets.size) - 1, 0)
    if family_capacity == 0 or part_capacity == 0:
        return None

    d_global_rows = cp.arange(owned.row_count, dtype=cp.int32)
    d_is_multiline = cp.asarray(state.validity, dtype=cp.bool_) & (
        cp.asarray(state.tags, dtype=cp.int8)
        == np.int8(FAMILY_TAGS[GeometryFamily.MULTILINESTRING])
    )
    d_family_rows = cp.where(
        d_is_multiline,
        cp.asarray(state.family_row_offsets, dtype=cp.int64),
        cp.int64(0),
    )
    d_encoded_sources = cp.zeros(family_capacity, dtype=cp.int32)
    cp.maximum.at(
        d_encoded_sources,
        d_family_rows,
        cp.where(d_is_multiline, d_global_rows + 1, cp.int32(0)),
    )
    d_family_active = d_encoded_sources > 0
    d_family_sources = cp.maximum(d_encoded_sources - 1, 0).astype(
        cp.int32,
        copy=False,
    )

    d_part_slots = cp.arange(part_capacity, dtype=cp.int64)
    d_logical_part_count = cp.asarray(buffer.geometry_offsets, dtype=cp.int64)[-1]
    d_active_part_slots = d_part_slots < d_logical_part_count
    d_safe_part_slots = cp.minimum(
        d_part_slots,
        cp.maximum(d_logical_part_count - 1, 0),
    )
    d_part_family_rows = cp.searchsorted(
        cp.asarray(buffer.geometry_offsets, dtype=cp.int64)[1:],
        d_safe_part_slots,
        side="right",
    ).astype(cp.int64, copy=False)
    d_part_family_rows = cp.minimum(d_part_family_rows, family_capacity - 1)
    d_active = d_active_part_slots & d_family_active[d_part_family_rows]
    d_source_rows = cp.where(
        d_active,
        d_family_sources[d_part_family_rows],
        cp.int32(0),
    )

    line_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        x=buffer.x,
        y=buffer.y,
        geometry_offsets=buffer.part_offsets,
        empty_mask=~d_active,
        bounds=None,
    )
    geometry = build_device_resident_owned(
        device_families={GeometryFamily.LINESTRING: line_buffer},
        row_count=part_capacity,
        tags=cp.full(
            part_capacity,
            FAMILY_TAGS[GeometryFamily.LINESTRING],
            dtype=cp.int8,
        ),
        validity=d_active,
        family_row_offsets=cp.arange(part_capacity, dtype=cp.int32),
        execution_mode="gpu",
    )
    return geometry, d_source_rows, d_active, int(buffer.x.size)


def _line_part_selection_from_capacities(
    partitions: list[tuple[OwnedGeometryArray, DeviceArray, DeviceArray, int]],
) -> LinePartCapacitySelection | None:
    """Compose line-family capacities into one selected-prefix carrier."""
    if not partitions:
        return None
    from vibespatial.api._native_rowset import NativeDeviceSelection

    raw_geometry = (
        partitions[0][0]
        if len(partitions) == 1
        else OwnedGeometryArray.concat([partition[0] for partition in partitions])
    )
    d_raw_source_rows = cp.concatenate(
        [cp.asarray(partition[1], dtype=cp.int32) for partition in partitions]
    )
    d_raw_active = cp.concatenate(
        [cp.asarray(partition[2], dtype=cp.bool_) for partition in partitions]
    )
    raw_selection = NativeDeviceSelection.from_mask(d_raw_active)
    geometry = raw_geometry._device_indexed_take(
        raw_selection.safe_capacity_positions(),
    )._apply_row_activity(
        raw_selection.active_capacity_mask(),
        assume_active_indices_unique=True,
    )
    d_source_rows = raw_selection.gather_capacity(
        d_raw_source_rows,
        fill_value=0,
    ).astype(cp.int32, copy=False)
    return LinePartCapacitySelection(
        geometry=geometry,
        source_rows=d_source_rows,
        selection=raw_selection.as_capacity_prefix(),
        coord_capacity=sum(int(partition[3]) for partition in partitions),
    )


def _indexed_lineal_part_capacities(
    owned: OwnedGeometryArray,
    state,
) -> list[tuple[OwnedGeometryArray, DeviceArray, DeviceArray, int]] | None:
    """Expose indexed line parts through proved logical part capacity."""
    d_validity = cp.asarray(state.validity, dtype=cp.bool_)
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)
    d_source_rows = cp.arange(owned.row_count, dtype=cp.int32)
    partitions: list[tuple[OwnedGeometryArray, DeviceArray, DeviceArray, int]] = []
    unique_family_rows = state.trusted_unique_family_rows is True

    linestring = state.families.get(GeometryFamily.LINESTRING)
    if linestring is not None:
        family_capacity = max(int(linestring.geometry_offsets.size) - 1, 0)
        if family_capacity > 0:
            if unique_family_rows:
                partition = _linestring_family_part_capacity(owned, state)
                if partition is not None:
                    partitions.append(partition)
            else:
                fixed_size = getattr(linestring, "fixed_size", None)
                fixed_coord_count = _fixed_or_max_structural_count(
                    fixed_size,
                    "coord_count_per_row",
                    structural_upper_bound=int(linestring.x.size),
                )
                if fixed_coord_count is None:
                    return None
                d_is_line = d_validity & (d_tags == np.int8(FAMILY_TAGS[GeometryFamily.LINESTRING]))
                d_in_bounds = d_is_line & (d_family_rows >= 0) & (d_family_rows < family_capacity)
                d_safe_rows = cp.where(d_in_bounds, d_family_rows, cp.int64(0))
                d_active = (
                    d_in_bounds
                    & ~cp.asarray(
                        linestring.empty_mask,
                        dtype=cp.bool_,
                    )[d_safe_rows]
                )
                base = build_device_resident_owned(
                    device_families={GeometryFamily.LINESTRING: linestring},
                    row_count=family_capacity,
                    tags=cp.full(
                        family_capacity,
                        FAMILY_TAGS[GeometryFamily.LINESTRING],
                        dtype=cp.int8,
                    ),
                    validity=cp.ones(family_capacity, dtype=cp.bool_),
                    family_row_offsets=cp.arange(family_capacity, dtype=cp.int32),
                    execution_mode="gpu",
                )
                geometry = base._device_indexed_take(
                    d_safe_rows,
                )._apply_row_activity(d_active)
                partitions.append(
                    (
                        geometry,
                        d_source_rows,
                        d_active,
                        owned.row_count * int(fixed_coord_count),
                    )
                )

    multiline = state.families.get(GeometryFamily.MULTILINESTRING)
    if multiline is not None and multiline.part_offsets is not None:
        family_capacity = max(int(multiline.geometry_offsets.size) - 1, 0)
        part_capacity = max(int(multiline.part_offsets.size) - 1, 0)
        if family_capacity > 0 and part_capacity > 0:
            if unique_family_rows:
                partition = _multilinestring_part_capacity(owned, state)
                if partition is not None:
                    partitions.append(partition)
            else:
                fixed_size = getattr(multiline, "fixed_size", None)
                part_width = _fixed_or_max_structural_count(
                    fixed_size,
                    "first_level_count_per_row",
                    structural_upper_bound=part_capacity,
                )
                fixed_coord_count = _fixed_or_max_structural_count(
                    fixed_size,
                    "coord_count_per_row",
                    structural_upper_bound=int(multiline.x.size),
                )
                if part_width is None or fixed_coord_count is None:
                    return None
                part_width = int(part_width)
                if part_width <= 0:
                    return None
                logical_capacity = owned.row_count * part_width
                d_lanes = cp.arange(logical_capacity, dtype=cp.int64)
                d_rows = d_lanes // part_width
                d_local_parts = d_lanes - (d_rows * part_width)
                d_is_multiline = d_validity[d_rows] & (
                    d_tags[d_rows] == np.int8(FAMILY_TAGS[GeometryFamily.MULTILINESTRING])
                )
                d_logical_family_rows = d_family_rows[d_rows]
                d_in_bounds = (
                    d_is_multiline
                    & (d_logical_family_rows >= 0)
                    & (d_logical_family_rows < family_capacity)
                )
                d_safe_family_rows = cp.where(
                    d_in_bounds,
                    d_logical_family_rows,
                    cp.int64(0),
                )
                d_part_starts = cp.asarray(
                    multiline.geometry_offsets,
                    dtype=cp.int64,
                )[d_safe_family_rows]
                d_part_ends = cp.asarray(
                    multiline.geometry_offsets,
                    dtype=cp.int64,
                )[d_safe_family_rows + 1]
                d_part_rows = d_part_starts + d_local_parts
                d_active = d_in_bounds & (d_part_rows < d_part_ends)
                d_safe_part_rows = cp.where(d_active, d_part_rows, cp.int64(0))
                line_buffer = DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.LINESTRING,
                    x=multiline.x,
                    y=multiline.y,
                    geometry_offsets=multiline.part_offsets,
                    empty_mask=cp.zeros(part_capacity, dtype=cp.bool_),
                    bounds=None,
                )
                base = build_device_resident_owned(
                    device_families={GeometryFamily.LINESTRING: line_buffer},
                    row_count=part_capacity,
                    tags=cp.full(
                        part_capacity,
                        FAMILY_TAGS[GeometryFamily.LINESTRING],
                        dtype=cp.int8,
                    ),
                    validity=cp.ones(part_capacity, dtype=cp.bool_),
                    family_row_offsets=cp.arange(part_capacity, dtype=cp.int32),
                    execution_mode="gpu",
                )
                geometry = base._device_indexed_take(
                    d_safe_part_rows,
                )._apply_row_activity(d_active)
                partitions.append(
                    (
                        geometry,
                        d_rows.astype(cp.int32, copy=False),
                        d_active,
                        owned.row_count * int(fixed_coord_count),
                    )
                )
    return partitions


def _explode_lineal_rows_to_line_capacity_gpu(
    owned: OwnedGeometryArray,
) -> LinePartCapacitySelection | None:
    """Explode lineal rows at physical part capacity without count reads."""
    if cp is None:
        return None
    if owned.is_indexed_view:
        indexed_state = owned._ensure_device_state(preserve_indexed_view=True)
        indexed_partitions = _indexed_lineal_part_capacities(
            owned,
            indexed_state,
        )
        if indexed_partitions is not None:
            return _line_part_selection_from_capacities(indexed_partitions)
        return None

    state = owned._ensure_device_state(preserve_indexed_view=False)
    partitions = [
        partition
        for partition in (
            _linestring_family_part_capacity(owned, state),
            _multilinestring_part_capacity(owned, state),
        )
        if partition is not None
    ]
    return _line_part_selection_from_capacities(partitions)


def _device_take_known_family_rows(
    owned: OwnedGeometryArray,
    rows: DeviceArray,
    family: GeometryFamily,
    *,
    allow_capacity_allocation: bool = False,
    assume_unique_indices: bool = False,
) -> OwnedGeometryArray:
    """Take device rows whose family tag has already been proven on device."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for device family-row takes")
    state = owned._ensure_device_state(preserve_indexed_view=True)
    if family not in state.families:
        return _empty_device_constructive_output(0)

    rows = cp.asarray(rows, dtype=cp.int64)
    row_count = int(rows.size)
    if row_count == 0:
        return _empty_device_constructive_output(0)

    device_family = state.families[family]
    source_family_rows = state.family_row_offsets[rows].astype(
        cp.int64,
        copy=False,
    )
    taken_family = _device_take_family_buffer(
        device_family,
        family,
        source_family_rows,
        owned.families.get(family),
        allow_capacity_allocation=allow_capacity_allocation,
        assume_unique_indices=assume_unique_indices,
    )
    result = build_device_resident_owned(
        device_families={family: taken_family},
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[family], dtype=cp.int8),
        validity=cp.ones(row_count, dtype=cp.bool_),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    if result.device_state is not None:
        result.device_state.trusted_all_valid = True
        result.device_state.trusted_homogeneous_family = family
        if family in _POLYGONAL_FAMILIES:
            result.device_state.trusted_polygonal_only = True
    return result


def _polygon_ring_segment_total_shape_hint(
    *,
    coord_count: int,
    ring_offset_count: int,
) -> int | None:
    """Return exact segment count when polygon rings are known non-empty."""
    ring_count = max(int(ring_offset_count) - 1, 0)
    coord_count = int(coord_count)
    if ring_count == 0 or coord_count < ring_count:
        return None
    return coord_count - ring_count


def _polygon_part_capacity_boundary_segments_gpu(
    polygon_parts: PolygonPartCapacitySelection,
    d_part_source_rows: DeviceArray,
    *,
    d_part_active_mask: DeviceArray | None = None,
) -> tuple[DeviceArray, DeviceArray, DeviceArray, DeviceArray, DeviceArray, DeviceArray] | None:
    """Expose polygon boundary segments at coordinate capacity with activity."""
    geometry = polygon_parts.geometry
    state = geometry._ensure_device_state(preserve_indexed_view=True)
    buffer = state.families.get(GeometryFamily.POLYGON)
    if buffer is None or buffer.ring_offsets is None:
        return None

    d_geometry_offsets = cp.asarray(buffer.geometry_offsets, dtype=cp.int64)
    d_ring_offsets = cp.asarray(buffer.ring_offsets, dtype=cp.int64)
    ring_capacity = int(polygon_parts.ring_capacity)
    coordinate_capacity = int(polygon_parts.coord_capacity)
    physical_coordinate_capacity = min(int(buffer.x.size), int(buffer.y.size))
    if (
        polygon_parts.capacity == 0
        or ring_capacity == 0
        or coordinate_capacity == 0
        or physical_coordinate_capacity == 0
    ):
        return None

    d_part_active = polygon_parts.selection.active_capacity_mask()
    if d_part_active_mask is not None:
        d_mask = cp.asarray(d_part_active_mask, dtype=cp.bool_)
        if int(d_mask.size) != polygon_parts.capacity:
            return None
        d_part_active &= d_mask
    d_part_source_rows = cp.asarray(d_part_source_rows, dtype=cp.int32)
    if int(d_part_source_rows.size) != polygon_parts.capacity:
        return None

    d_part_family_rows = cp.where(
        d_part_active,
        cp.asarray(state.family_row_offsets, dtype=cp.int64),
        cp.int64(0),
    )
    d_part_ring_starts = d_geometry_offsets[d_part_family_rows]
    d_part_ring_ends = d_geometry_offsets[d_part_family_rows + 1]
    d_part_ring_counts = cp.where(
        d_part_active,
        d_part_ring_ends - d_part_ring_starts,
        cp.int64(0),
    )
    d_part_ring_offsets = cp.zeros(polygon_parts.capacity + 1, dtype=cp.int64)
    cp.cumsum(d_part_ring_counts, out=d_part_ring_offsets[1:])

    d_ring_slots = cp.arange(ring_capacity, dtype=cp.int64)
    d_ring_in_range = d_ring_slots < d_part_ring_offsets[-1]
    d_safe_ring_slots = cp.minimum(
        d_ring_slots,
        cp.maximum(d_part_ring_offsets[-1] - 1, 0),
    )
    d_ring_part_rows = cp.searchsorted(
        d_part_ring_offsets[1:],
        d_safe_ring_slots,
        side="right",
    ).astype(cp.int64, copy=False)
    d_ring_part_rows = cp.minimum(
        d_ring_part_rows,
        max(polygon_parts.capacity - 1, 0),
    )
    d_ring_active = d_ring_in_range & d_part_active[d_ring_part_rows]
    d_local_ring_rows = d_safe_ring_slots - d_part_ring_offsets[d_ring_part_rows]
    d_physical_ring_rows = d_part_ring_starts[d_ring_part_rows] + d_local_ring_rows
    d_safe_physical_ring_rows = cp.where(
        d_ring_active,
        d_physical_ring_rows,
        cp.int64(0),
    )
    d_ring_segment_counts = cp.where(
        d_ring_active,
        cp.maximum(
            d_ring_offsets[d_safe_physical_ring_rows + 1]
            - d_ring_offsets[d_safe_physical_ring_rows]
            - 1,
            0,
        ),
        cp.int64(0),
    ).astype(cp.int64, copy=False)
    d_segment_offsets = cp.zeros(ring_capacity + 1, dtype=cp.int64)
    cp.cumsum(d_ring_segment_counts, out=d_segment_offsets[1:])

    d_segment_slots = cp.arange(coordinate_capacity, dtype=cp.int64)
    d_segment_in_range = d_segment_slots < d_segment_offsets[-1]
    d_safe_segment_slots = cp.minimum(
        d_segment_slots,
        cp.maximum(d_segment_offsets[-1] - 1, 0),
    )
    d_segment_ring_rows = cp.searchsorted(
        d_segment_offsets[1:],
        d_safe_segment_slots,
        side="right",
    ).astype(cp.int64, copy=False)
    d_segment_ring_rows = cp.minimum(d_segment_ring_rows, ring_capacity - 1)
    d_segment_active = d_segment_in_range & d_ring_active[d_segment_ring_rows]
    d_local_segments = d_safe_segment_slots - d_segment_offsets[d_segment_ring_rows]
    d_segment_physical_ring_rows = d_safe_physical_ring_rows[d_segment_ring_rows]
    d_vertex_rows = d_ring_offsets[d_segment_physical_ring_rows] + d_local_segments
    d_end_vertex_rows = cp.minimum(
        d_vertex_rows + 1,
        physical_coordinate_capacity - 1,
    )
    d_segment_part_rows = d_ring_part_rows[d_segment_ring_rows]
    d_segment_source_rows = d_part_source_rows[d_segment_part_rows]
    d_x = cp.asarray(buffer.x, dtype=cp.float64)
    d_y = cp.asarray(buffer.y, dtype=cp.float64)
    return (
        d_x[d_vertex_rows],
        d_y[d_vertex_rows],
        d_x[d_end_vertex_rows],
        d_y[d_end_vertex_rows],
        d_segment_source_rows,
        d_segment_active,
    )


def _assemble_noded_polygon_coverage_split_events_gpu(
    split_events,
    *,
    output_row_count: int,
    d_valid_empty_rows=None,
) -> OwnedGeometryArray | None:
    """Assemble noded coverage split events through parity and contour depth."""
    from vibespatial.overlay.boundary_graph import (
        build_polygon_output_from_boundary_segments_gpu,
        undirected_boundary_segment_orders_gpu,
    )
    from vibespatial.overlay.split import (
        _free_split_event_device_state,
        noded_boundary_segments_from_split_events_gpu,
    )

    try:
        start_x, start_y, end_x, end_y, d_segment_rows = (
            noded_boundary_segments_from_split_events_gpu(split_events)
        )
        boundary_orders = undirected_boundary_segment_orders_gpu(
            start_x,
            start_y,
            end_x,
            end_y,
            d_segment_rows,
        )
        result = build_polygon_output_from_boundary_segments_gpu(
            start_x[boundary_orders],
            start_y[boundary_orders],
            end_x[boundary_orders],
            end_y[boundary_orders],
            row_indices=d_segment_rows[boundary_orders],
            row_count=output_row_count,
            runtime_selection=split_events.runtime_selection,
            d_valid_empty_rows=d_valid_empty_rows,
        )
    finally:
        _free_split_event_device_state(split_events)
    if result.row_count != output_row_count:
        return None
    return result


def _dispatch_grouped_polygon_noded_coverage_union_gpu(
    owned: OwnedGeometryArray,
    source_rows,
    *,
    output_row_count: int,
    dispatch_mode: ExecutionMode,
    d_valid_empty_rows=None,
) -> OwnedGeometryArray | None:
    """Node and assemble a group-local polygon coverage on the device.

    Physical shape: polygon boundary segments -> group-local segment
    intersections -> noded boundary atoms -> odd-parity shared-edge removal ->
    nested contour assembly. This handles partial shared edges and disconnected
    hole contours; raw segment cancellation is only correct after noding.
    """
    if cp is None or output_row_count < 0 or owned.row_count == 0:
        return None
    d_source_rows = cp.asarray(source_rows, dtype=cp.int32)
    if int(d_source_rows.size) != owned.row_count:
        return None

    from vibespatial.geometry.owned import build_empty_polygon_rows_device
    from vibespatial.overlay.split import (
        build_gpu_split_events,
    )

    empty_left = build_empty_polygon_rows_device(output_row_count)
    split_events = build_gpu_split_events(
        empty_left,
        owned,
        dispatch_mode=dispatch_mode,
        require_same_row=True,
        use_same_row_fast_path=False,
        right_geometry_source_rows=d_source_rows,
        include_same_side_splits=True,
    )
    return _assemble_noded_polygon_coverage_split_events_gpu(
        split_events,
        output_row_count=output_row_count,
        d_valid_empty_rows=d_valid_empty_rows,
    )



def _dispatch_row_aligned_polygon_known_coverage_union_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    assume_all_valid: bool = False,
) -> OwnedGeometryArray | None:
    """Coverage-union aligned polygon rows through noded grouped boundaries."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count or left.row_count == 0:
        return None
    if not assume_all_valid and not (_all_rows_valid(left) and _all_rows_valid(right)):
        return None

    from vibespatial.overlay.split import build_gpu_split_events

    split_events = build_gpu_split_events(
        left,
        right,
        dispatch_mode=dispatch_mode,
        require_same_row=True,
        use_same_row_fast_path=True,
        include_same_side_splits=False,
    )
    result = _assemble_noded_polygon_coverage_split_events_gpu(
        split_events,
        output_row_count=left.row_count,
        d_valid_empty_rows=cp.ones(left.row_count, dtype=cp.bool_),
    )
    if result is None or result.row_count != left.row_count:
        return None
    seed_all_validity_cache(result)
    return result


def _dispatch_grouped_polygon_known_coverage_union_gpu(
    owned: OwnedGeometryArray,
    source_rows,
    *,
    output_row_count: int,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    assume_all_valid: bool = False,
    assume_source_rows_valid: bool = False,
    total_segments_hint: int | None = None,
    d_valid_empty_rows=None,
) -> OwnedGeometryArray | None:
    """Coverage-union polygon rows by device source group.

    Physical shape: group-local boundary noding, odd-parity shared-edge
    removal, and nested contour assembly. The public dissolve row is not the
    work unit.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if output_row_count < 0 or owned.row_count == 0:
        return None

    d_source_rows = cp.asarray(source_rows, dtype=cp.int32)
    if int(d_source_rows.size) != owned.row_count:
        return None

    state = owned._ensure_device_state(preserve_indexed_view=True)
    if not assume_all_valid and not _device_scalar_bool(
        cp.all(cp.asarray(state.validity, dtype=cp.bool_)),
        reason="grouped known-coverage input-validity admission scalar fence",
    ):
        return None

    if not assume_source_rows_valid:
        d_source_valid = (d_source_rows >= 0) & (d_source_rows < np.int32(output_row_count))
        owned = owned._apply_row_activity(d_source_valid)
        d_source_rows = cp.where(
            d_source_valid,
            d_source_rows,
            cp.int32(0),
        )
    result = _dispatch_grouped_polygon_noded_coverage_union_gpu(
        owned,
        d_source_rows,
        output_row_count=output_row_count,
        dispatch_mode=dispatch_mode,
        d_valid_empty_rows=d_valid_empty_rows,
    )
    if result is None or result.row_count != output_row_count:
        return None
    seed_all_validity_cache(result)
    record_dispatch_event(
        surface="constructive.grouped_polygon_coverage_union",
        operation="grouped_coverage_union",
        implementation="gpu_grouped_noded_boundary_coverage_assembly",
        reason=(
            "grouped coverage union noded partial shared boundaries before "
            "group-local odd-parity edge removal and contour assembly"
        ),
        detail=(
            f"rows={owned.row_count}, groups={output_row_count}, segment_hint={total_segments_hint}"
        ),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return result


def _dispatch_single_row_polygon_known_coverage_union_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Exact noded coverage union for one aligned polygon pair."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count or left.row_count != 1:
        return None
    if not (_row_is_valid(left, 0) and _row_is_valid(right, 0)):
        return None

    result = _dispatch_row_aligned_polygon_known_coverage_union_gpu(
        left,
        right,
        dispatch_mode=dispatch_mode,
    )
    if result is None or result.row_count != 1:
        return None
    return result


def _dispatch_single_row_polygon_partition_union_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _assume_valid: bool = False,
    _assume_not_strictly_disjoint: bool = False,
) -> OwnedGeometryArray | None:
    """Route a single aligned polygon pair through exact capacity topology."""
    if left.row_count != 1 or right.row_count != 1:
        return None
    return _dispatch_polygon_partition_union_gpu(
        left,
        right,
        dispatch_mode=dispatch_mode,
    )


def _dispatch_single_row_polygon_union_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _assume_valid: bool = False,
    _assume_not_strictly_disjoint: bool = False,
) -> OwnedGeometryArray | None:
    """Route a single aligned polygon union through exact capacity topology."""
    if left.row_count != 1 or right.row_count != 1:
        return None
    return _dispatch_polygon_partition_union_gpu(
        left,
        right,
        dispatch_mode=dispatch_mode,
    )


def _dispatch_polygon_union_repair_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Route aligned polygon union through the canonical capacity topology."""
    return _dispatch_polygon_partition_union_gpu(
        left,
        right,
        dispatch_mode=dispatch_mode,
    )


def _row_aligned_rectangle_partition_difference_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """Build clipped rectangle differences in row-capacity exact topology."""
    if cp is None or left.row_count != right.row_count or left.row_count == 0:
        return None

    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    if set(left_state.families) != {GeometryFamily.POLYGON}:
        return None
    if GeometryFamily.POLYGON not in right_state.families:
        return None

    left_polygon = left_state.families[GeometryFamily.POLYGON]
    right_polygon = right_state.families[GeometryFamily.POLYGON]
    right_is_dense_rectangle = int(
        getattr(right_polygon, "dense_single_ring_width", 0) or 0
    ) == 5 and bool(getattr(right_polygon, "axis_aligned_rectangles", False))
    if int(getattr(left_polygon, "dense_single_ring_width", 0) or 0) != 5 or not bool(
        getattr(left_polygon, "axis_aligned_rectangles", False)
    ):
        return None
    d_strip_rows = getattr(
        right,
        "_native_grouped_rectangle_strip_output_mask",
        None,
    )
    if not right_is_dense_rectangle and d_strip_rows is None:
        return None

    from vibespatial.constructive.envelope import _build_device_boxes_from_bounds
    from vibespatial.geometry.owned import (
        device_mask_owned_capacity,
        device_select_owned_capacity_partitions,
        device_valid_nonempty_mask,
    )
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    row_count = int(left.row_count)
    d_left_supported = device_valid_nonempty_mask(left)
    d_right_supported = device_valid_nonempty_mask(right)
    if d_strip_rows is not None:
        d_strip_rows = cp.asarray(d_strip_rows, dtype=cp.bool_)
        if d_strip_rows.ndim != 1 or int(d_strip_rows.size) != row_count:
            return None
        d_right_supported &= d_strip_rows
    d_supported = d_left_supported & d_right_supported
    left_bounds = cp.asarray(
        compute_geometry_bounds_device(left, preserve_indexed_view=True),
        dtype=cp.float64,
    ).reshape(row_count, 4)
    right_bounds = cp.asarray(
        compute_geometry_bounds_device(right, preserve_indexed_view=True),
        dtype=cp.float64,
    ).reshape(row_count, 4)

    intersection_bounds = cp.empty_like(left_bounds)
    intersection_bounds[:, 0] = cp.maximum(left_bounds[:, 0], right_bounds[:, 0])
    intersection_bounds[:, 1] = cp.maximum(left_bounds[:, 1], right_bounds[:, 1])
    intersection_bounds[:, 2] = cp.minimum(left_bounds[:, 2], right_bounds[:, 2])
    intersection_bounds[:, 3] = cp.minimum(left_bounds[:, 3], right_bounds[:, 3])
    d_overlap = (
        d_supported
        & (intersection_bounds[:, 2] > intersection_bounds[:, 0])
        & (intersection_bounds[:, 3] > intersection_bounds[:, 1])
    )

    clipped_right = _build_device_boxes_from_bounds(
        cp.ascontiguousarray(intersection_bounds),
        row_count=row_count,
    )
    exact_overlap = _dispatch_polygon_difference_overlay_exact_batch_gpu(
        device_mask_owned_capacity(left, d_overlap),
        device_mask_owned_capacity(clipped_right, d_overlap),
        dispatch_mode=ExecutionMode.GPU,
    )
    if exact_overlap is None or exact_overlap.row_count != row_count:
        return None

    result = device_select_owned_capacity_partitions(
        device_mask_owned_capacity(left, d_supported & ~d_overlap),
        [(exact_overlap, d_overlap)],
    )
    result._rectangle_partition_difference_support_mask = d_supported
    record_dispatch_event(
        surface="vibespatial.constructive.binary_constructive",
        operation="difference",
        implementation="rectangle_clipped_difference_capacity_gpu",
        reason=(
            "aligned rectangle difference clipped the right rows to overlap "
            "bounds and kept exact topology in row capacity"
        ),
        detail=(
            f"rows={row_count}; topology_capacity={row_count}; partition_counts=device-resident"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return result


def _row_aligned_rectangle_clipped_difference_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Compute rectangle-left polygon difference by clipping right to left first.

    Physical shape: row-aligned rectangle bounds plus single-ring polygon rows
    -> polygon/rectangle intersection carrier -> row-aligned constructive
    difference.  This removes the expensive exact rowwise overlay repair for
    parcel-grid cells whose subtracting polygon extends outside the cell.
    """
    if cp is None:
        return None
    if left.row_count != right.row_count or left.row_count == 0:
        return None

    from vibespatial.geometry.owned import (
        device_mask_owned_capacity,
        device_select_owned_capacity_partitions,
        device_valid_nonempty_mask,
    )
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
        polygon_rect_intersection_can_handle,
    )

    rect_left = _resolve_indexed_polygon_fast_path_candidate(left)
    polygon_right = _resolve_indexed_polygon_fast_path_candidate(right)
    if not polygon_rect_intersection_can_handle(polygon_right, rect_left):
        return None

    clipped_right = polygon_rect_intersection(
        polygon_right,
        rect_left,
        dispatch_mode=ExecutionMode.GPU,
    )
    if clipped_right.row_count != rect_left.row_count:
        return None

    rect_state = rect_left._ensure_device_state(preserve_indexed_view=True)
    right_state = polygon_right._ensure_device_state(preserve_indexed_view=True)
    d_input_valid = cp.asarray(rect_state.validity, dtype=cp.bool_) & cp.asarray(
        right_state.validity,
        dtype=cp.bool_,
    )
    d_overlap = d_input_valid & device_valid_nonempty_mask(clipped_right)
    overlap_result = _dispatch_polygon_difference_overlay_batched_gpu(
        device_mask_owned_capacity(rect_left, d_overlap),
        device_mask_owned_capacity(clipped_right, d_overlap),
        dispatch_mode=dispatch_mode,
    )
    if overlap_result is None or overlap_result.row_count != rect_left.row_count:
        return None
    result = device_select_owned_capacity_partitions(
        device_mask_owned_capacity(rect_left, d_input_valid & ~d_overlap),
        [(overlap_result, d_overlap)],
    )

    record_dispatch_event(
        surface="geopandas.array.difference",
        operation="difference",
        implementation="row_aligned_rectangle_clipped_difference_gpu",
        reason=(
            "rectangle-left polygon difference clipped the subtracting "
            "polygon to the rectangle rowset before native batched difference"
        ),
        detail=f"rows={left.row_count}",
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return result


def _dispatch_polygon_partition_union_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _partition_disjoint: bool = False,
    _active_rows=None,
) -> OwnedGeometryArray | None:
    """Partition aligned polygon union into disjoint pack and topology rows."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count:
        return None
    if left.row_count == 0:
        return _empty_device_constructive_output(0)

    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.geometry.owned import device_valid_nonempty_mask
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    row_count = int(left.row_count)
    for owned in (left, right):
        if owned.residency is not Residency.DEVICE:
            owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="row-aligned polygon union device partition",
            )
    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    d_both_valid = cp.asarray(left_state.validity, dtype=cp.bool_) & cp.asarray(
        right_state.validity,
        dtype=cp.bool_,
    )
    if _active_rows is not None:
        d_requested = cp.asarray(_active_rows, dtype=cp.bool_)
        if d_requested.ndim != 1 or int(d_requested.size) != row_count:
            raise ValueError("polygon union activity mask must be row-aligned")
        d_both_valid &= d_requested
    d_left_nonempty = device_valid_nonempty_mask(left)
    d_right_nonempty = device_valid_nonempty_mask(right)
    d_disjoint = cp.zeros(row_count, dtype=cp.bool_)
    direct_result = _empty_device_constructive_output(row_count)
    if _partition_disjoint:
        d_left_bounds = cp.asarray(
            compute_geometry_bounds_device(left, preserve_indexed_view=True),
            dtype=cp.float64,
        ).reshape(row_count, 4)
        d_right_bounds = cp.asarray(
            compute_geometry_bounds_device(right, preserve_indexed_view=True),
            dtype=cp.float64,
        ).reshape(row_count, 4)
        d_disjoint = (
            d_both_valid
            & d_left_nonempty
            & d_right_nonempty
            & (
                (d_left_bounds[:, 2] < d_right_bounds[:, 0])
                | (d_right_bounds[:, 2] < d_left_bounds[:, 0])
                | (d_left_bounds[:, 3] < d_right_bounds[:, 1])
                | (d_right_bounds[:, 3] < d_left_bounds[:, 1])
            )
        )

        pair_sources = OwnedGeometryArray.concat([left, right])
        d_rows = cp.arange(row_count, dtype=cp.int64)
        d_pair_order = cp.empty(row_count * 2, dtype=cp.int64)
        d_pair_order[0::2] = d_rows
        d_pair_order[1::2] = d_rows + np.int64(row_count)
        direct_result = _pack_native_grouped_disjoint_polygon_parts_gpu(
            pair_sources,
            d_pair_order,
            cp.arange(row_count + 1, dtype=cp.int64) * np.int64(2),
            d_rows,
            output_row_count=row_count,
            group_size_max=2,
            empty_output=_empty_device_constructive_output(row_count),
            active_group_mask=d_disjoint,
            assume_active_groups_disjoint=True,
        )
        if direct_result is None or direct_result.row_count != row_count:
            raise RuntimeError("disjoint polygon union partition assembly failed")

    d_topology = d_both_valid & d_left_nonempty & d_right_nonempty & ~d_disjoint
    selection = NativeDeviceSelection.from_mask(
        d_topology,
        source_row_count=row_count,
    )
    d_active = selection.active_capacity_mask()
    union_left = _device_take_preserving_indexed_rows(
        left,
        selection.partition_capacity_positions(),
        assume_unique_indices=True,
    )._apply_row_activity(d_active)
    union_right = _device_take_preserving_indexed_rows(
        right,
        selection.partition_capacity_positions(),
        assume_unique_indices=True,
    )._apply_row_activity(d_active)

    plan = _build_overlay_execution_plan(
        union_left,
        union_right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
        _include_same_side_splits=True,
    )
    union_result, selected = _materialize_overlay_execution_plan(
        plan,
        operation="union",
        requested=ExecutionMode.GPU,
        preserve_row_count=row_count,
    )
    if selected is not ExecutionMode.GPU or union_result.row_count != row_count:
        return None

    from vibespatial.geometry.owned import device_scatter_owned_capacity_selection

    result = device_scatter_owned_capacity_selection(
        direct_result,
        union_result,
        selection,
    )
    if left_state.trusted_all_ogc_valid is True and right_state.trusted_all_ogc_valid is True:
        seed_all_validity_cache(result)
    record_dispatch_event(
        surface="geopandas.array.union",
        operation="union",
        implementation="row_aligned_union_disjoint_topology_partition_gpu",
        reason=(
            "row-aligned polygon union packed separated rows directly and "
            "materialized overlapping rows through exact same-side-split topology"
        ),
        detail=(
            f"rows={row_count}; logical_rows=device-resident; "
            "workload_shape=aligned_polygon_union_capacity"
        ),
        requested=dispatch_mode,
        selected=selected,
    )
    return result


def _dispatch_polygon_difference_overlay_rowwise_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Preserve aligned difference rows through one native batch topology plan."""
    return _dispatch_polygon_difference_overlay_batched_gpu(
        left,
        right,
        dispatch_mode=dispatch_mode,
    )


def _dispatch_polygon_difference_overlay_exact_batch_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Run one row-isolated exact difference overlay for an aligned rowset."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count:
        return None
    if left.row_count == 0:
        return _empty_device_constructive_output(0)

    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    _sync_hotpath()
    with hotpath_stage("constructive.diff.exact_batch_plan.build", category="setup"):
        plan = _build_overlay_execution_plan(
            left,
            right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=None,
            _row_isolated=True,
        )
    _sync_hotpath()
    with hotpath_stage(
        "constructive.diff.exact_batch_plan.materialize",
        category="refine",
    ):
        result, selected = _materialize_overlay_execution_plan(
            plan,
            operation="difference",
            requested=ExecutionMode.GPU,
            preserve_row_count=left.row_count,
            valid_empty_rows=(
                cp.asarray(
                    left._ensure_device_state(
                        preserve_indexed_view=True,
                    ).validity,
                    dtype=cp.bool_,
                )
                & cp.asarray(
                    right._ensure_device_state(
                        preserve_indexed_view=True,
                    ).validity,
                    dtype=cp.bool_,
                )
            ),
        )
    _sync_hotpath()

    if selected is not ExecutionMode.GPU or result.row_count != left.row_count:
        return None
    record_dispatch_event(
        surface="geopandas.array.difference",
        operation="difference",
        implementation="row_aligned_difference_exact_batch_gpu",
        reason=(
            "row-aligned polygon difference consumed one null-masked capacity "
            "through a row-isolated exact overlay plan"
        ),
        detail=f"rows={left.row_count}",
        requested=dispatch_mode,
        selected=selected,
    )
    return result


def _dispatch_polygon_difference_overlay_batched_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Partition aligned polygon difference at public-row capacity."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count:
        return None
    if left.row_count == 0:
        return _empty_device_constructive_output(0)

    from vibespatial.api._native_grouped import NativeGrouped
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.api.tools.overlay import _grouped_polygon_hole_difference_owned
    from vibespatial.geometry.owned import (
        build_empty_polygon_rows_device,
        device_scatter_owned_capacity_selection,
        device_select_owned_capacity_partitions,
    )
    from vibespatial.predicates.binary import binary_predicate_expression

    row_count = int(left.row_count)
    left = _device_polygonal_capacity_view(left)
    right = _device_polygonal_capacity_view(right)
    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    d_left_valid = cp.asarray(left_state.validity, dtype=cp.bool_)
    d_right_valid = cp.asarray(right_state.validity, dtype=cp.bool_)
    d_both_valid = d_left_valid & d_right_valid

    predicate_values = []
    for predicate in ("intersects", "touches", "covered_by"):
        expression = binary_predicate_expression(
            predicate,
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
            operation=f"constructive.difference.capacity_{predicate}",
        )
        if expression is None:
            return None
        values = cp.asarray(expression.values, dtype=cp.bool_)
        if values.ndim != 1 or values.size != row_count:
            raise RuntimeError(
                "aligned polygon difference predicate violated row-capacity contract"
            )
        predicate_values.append(values)
    d_intersects, d_touches, d_covered_by_right = predicate_values

    d_preserve = d_both_valid & ((~d_intersects) | d_touches)
    d_exact = d_both_valid & d_intersects & ~d_touches & ~d_covered_by_right

    d_single_offsets = cp.arange(row_count + 1, dtype=cp.int64)
    single_grouped = NativeGrouped.from_sorted_offsets(
        d_single_offsets,
        row_count=row_count,
        all_groups_observed=True,
        group_size_min=1,
        group_size_max=1,
    )
    direct_holes = _grouped_polygon_hole_difference_owned(
        left,
        right,
        single_grouped,
        d_single_offsets,
        dispatch_mode=dispatch_mode,
        event_implementation="row_aligned_polygon_hole_difference_gpu",
        event_reason=(
            "row-aligned polygon difference emitted strictly contained "
            "single-ring right rows as interior rings from device buffers"
        ),
        event_pairs=row_count,
    )
    d_direct_holes = (
        cp.zeros(row_count, dtype=cp.bool_)
        if direct_holes is None
        else cp.asarray(direct_holes.support_mask, dtype=cp.bool_) & d_exact
    )
    d_exact &= ~d_direct_holes
    preserve_selection = NativeDeviceSelection.from_mask(
        d_preserve,
        source_row_count=row_count,
    )
    exact_selection = NativeDeviceSelection.from_mask(
        d_exact,
        source_row_count=row_count,
    )

    result = build_empty_polygon_rows_device(
        row_count,
        validity=d_both_valid,
    )
    d_preserve_active = preserve_selection.active_capacity_mask()
    preserve_left = _device_take_preserving_indexed_rows(
        left,
        preserve_selection.partition_capacity_positions(),
    )._apply_row_activity(d_preserve_active)
    result = device_scatter_owned_capacity_selection(
        result,
        preserve_left,
        preserve_selection,
    )
    if direct_holes is not None:
        result = device_select_owned_capacity_partitions(
            result,
            [(direct_holes.owned, d_direct_holes)],
        )

    d_exact_active = exact_selection.active_capacity_mask()
    exact_left = _device_take_preserving_indexed_rows(
        left,
        exact_selection.partition_capacity_positions(),
    )._apply_row_activity(d_exact_active)
    exact_right = _device_take_preserving_indexed_rows(
        right,
        exact_selection.partition_capacity_positions(),
    )._apply_row_activity(d_exact_active)
    exact_result = _dispatch_polygon_difference_overlay_exact_batch_gpu(
        exact_left,
        exact_right,
        dispatch_mode=dispatch_mode,
    )
    if exact_result is None or exact_result.row_count != row_count:
        return None
    result = device_scatter_owned_capacity_selection(
        result,
        exact_result,
        exact_selection,
    )

    record_dispatch_event(
        surface="geopandas.array.difference",
        operation="difference",
        implementation="row_aligned_difference_capacity_partition_gpu",
        reason=(
            "row-aligned polygon difference retained preserve, valid-empty, "
            "and exact topology work in device-counted row-capacity partitions"
        ),
        detail=(
            f"rows={row_count}; partition_counts=device-resident; "
            "workload_shape=aligned_polygon_difference_capacity"
        ),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return result


def _pair_supports_gpu_constructive(
    op: str,
    left_family: GeometryFamily,
    right_family: GeometryFamily,
) -> bool:
    """Return True when the current GPU dispatcher can handle this family pair."""
    if left_family is GeometryFamily.POINT and right_family is GeometryFamily.POINT:
        return op in _CONSTRUCTIVE_OPS
    if left_family is GeometryFamily.POINT and right_family in _POLYGONAL_FAMILIES:
        return op in _POINT_POLYGON_OPS
    if left_family in _POLYGONAL_FAMILIES and right_family is GeometryFamily.POINT:
        return op in {"intersection", "difference"}
    if left_family is GeometryFamily.POINT and right_family in _LINESTRING_FAMILIES:
        return op in {"intersection", "difference"}
    if left_family in _LINESTRING_FAMILIES and right_family is GeometryFamily.POINT:
        return op == "intersection"
    if left_family is GeometryFamily.MULTIPOINT and right_family in _POLYGONAL_FAMILIES:
        return op in _POINT_POLYGON_OPS
    if left_family in _POLYGONAL_FAMILIES and right_family is GeometryFamily.MULTIPOINT:
        return op in {"intersection", "difference"}
    if left_family in _LINESTRING_FAMILIES and right_family in _POLYGONAL_FAMILIES:
        return op in {"intersection", "difference"}
    if left_family in _POLYGONAL_FAMILIES and right_family in _LINESTRING_FAMILIES:
        return op == "intersection"
    if left_family in _LINESTRING_FAMILIES and right_family in _LINESTRING_FAMILIES:
        return op == "intersection"
    if left_family in _POLYGONAL_FAMILIES and right_family in _POLYGONAL_FAMILIES:
        return op in _CONSTRUCTIVE_OPS
    return False


def _supports_gpu_constructive(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> bool:
    """Return True when every valid family pair in the workload is GPU-supported."""
    for left_tag, right_tag in _valid_family_tag_pairs(left, right):
        left_family = TAG_FAMILIES.get(left_tag)
        right_family = TAG_FAMILIES.get(right_tag)
        if left_family is None or right_family is None:
            return False
        if not _pair_supports_gpu_constructive(op, left_family, right_family):
            return False
    return True


def _binary_constructive_work_estimate(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    workload: WorkloadShape,
) -> PhysicalWorkEstimate:
    """Estimate constructive work from coordinate/segment shape, not rows."""
    row_count = int(left.row_count)
    left_estimate = estimate_physical_work_from_owned(left)
    right_estimate = estimate_physical_work_from_owned(right)
    if workload in (WorkloadShape.BROADCAST_RIGHT, WorkloadShape.SCALAR_RIGHT):
        right_coordinate_work = int(right_estimate.coordinate_count) * max(row_count, 1)
        right_segment_work = int(right_estimate.segment_count) * max(row_count, 1)
    else:
        right_coordinate_work = int(right_estimate.coordinate_count)
        right_segment_work = int(right_estimate.segment_count)
    coordinate_count = int(left_estimate.coordinate_count) + right_coordinate_work
    segment_count = int(left_estimate.segment_count) + right_segment_work
    dispatch_units = max(row_count, coordinate_count, segment_count)
    return PhysicalWorkEstimate(
        row_count=row_count,
        coordinate_count=coordinate_count,
        segment_count=segment_count,
        output_row_count=row_count,
        primary_unit_count=dispatch_units,
        primary_unit_name="constructive-segment",
    )


def _constructive_segment_work_estimate(
    *,
    row_count: int,
    segment_count: int,
) -> PhysicalWorkEstimate:
    """Estimate graph/face assembly work from boundary segments."""
    row_count = int(row_count)
    segment_count = int(segment_count)
    return PhysicalWorkEstimate(
        row_count=row_count,
        segment_count=segment_count,
        output_row_count=row_count,
        primary_unit_count=max(row_count, segment_count),
        primary_unit_name="boundary-segment",
    )


def _needs_grouped_gpu_dispatch(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> bool:
    """Return True when the workload spans multiple valid family pairs."""
    if _is_polygon_only(left) and _is_polygon_only(right):
        return False
    return len(_valid_family_tag_pairs(left, right)) > 1


def _empty_device_constructive_output(row_count: int) -> OwnedGeometryArray:
    """Build an all-null device-resident constructive result."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for GPU constructive output assembly")
    result = build_device_resident_owned(
        device_families={},
        row_count=row_count,
        tags=cp.full(row_count, NULL_TAG, dtype=cp.int8),
        validity=cp.zeros(row_count, dtype=cp.bool_),
        family_row_offsets=cp.full(row_count, -1, dtype=cp.int32),
        execution_mode="gpu",
    )
    result._active_family_row_segment_capacity_bound = 0
    return result


def _compose_aligned_native_geometries(geometries, *, row_count: int, crs=None):
    """Flatten row-aligned native geometry carriers into one composition."""
    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeGeometryComposition,
        NativeGeometryCompositionPart,
    )

    parts = []
    for geometry in geometries:
        native = (
            geometry
            if isinstance(geometry, GeometryNativeResult)
            else GeometryNativeResult.from_owned(geometry, crs=crs)
        )
        if native.row_count != int(row_count):
            raise ValueError("aligned native geometry rows must match composition capacity")
        native = native.with_crs(crs)
        if native.composition is not None:
            parts.extend(native.composition.parts)
        else:
            parts.append(
                NativeGeometryCompositionPart(
                    geometry=native,
                    output_rows=cp.arange(row_count, dtype=cp.int64),
                )
            )
    return GeometryNativeResult.from_composition(
        NativeGeometryComposition(
            parts=tuple(parts),
            row_count=int(row_count),
            crs=crs,
        ),
        crs=crs,
    )


def _apply_binary_empty_row_semantics_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    result,
):
    """Apply valid-empty identities without converting empty rows to nulls."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return result
    for owned in (left, right):
        if owned.residency is not Residency.DEVICE:
            owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="binary constructive empty-row identity carrier",
            )
    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    if left_state.trusted_all_non_empty is True and right_state.trusted_all_non_empty is True:
        return result

    from vibespatial.geometry.owned import (
        device_physical_select_owned_capacity_partitions,
        device_valid_nonempty_mask,
    )

    d_left_valid = cp.asarray(left_state.validity, dtype=cp.bool_)
    d_right_valid = cp.asarray(right_state.validity, dtype=cp.bool_)
    d_left_nonempty = device_valid_nonempty_mask(left)
    d_right_nonempty = device_valid_nonempty_mask(right)
    d_both_valid = d_left_valid & d_right_valid
    d_core = d_both_valid & d_left_nonempty & d_right_nonempty

    from vibespatial.api._native_result_core import GeometryNativeResult
    from vibespatial.geometry.owned import device_mask_owned_capacity

    if isinstance(result, GeometryNativeResult):
        fallback_sources = []
        if op in {"union", "symmetric_difference"}:
            fallback_sources.extend(
                [
                    (left, d_both_valid & ~d_right_nonempty),
                    (right, d_both_valid & d_right_nonempty & ~d_left_nonempty),
                ]
            )
        elif op == "difference":
            fallback_sources.append(
                (left, d_both_valid & (~d_left_nonempty | ~d_right_nonempty))
            )
        elif op == "intersection":
            fallback_sources.extend(
                [
                    (left, d_both_valid & ~d_left_nonempty),
                    (right, d_both_valid & d_left_nonempty & ~d_right_nonempty),
                ]
            )
        else:  # pragma: no cover - guarded by the public operation contract
            return result

        core = result.mask_capacity(d_core)
        fallback_parts = [core]
        for source, d_mask in fallback_sources:
            source_part = device_mask_owned_capacity(
                source,
                d_mask,
            )
            fallback_parts.append(
                GeometryNativeResult.from_owned(source_part, crs=result.crs)
            )
        return _compose_aligned_native_geometries(
            fallback_parts,
            row_count=result.row_count,
            crs=result.crs,
        )

    replacements: list[tuple[OwnedGeometryArray, DeviceArray]] = [(result, d_core)]
    if op in {"union", "symmetric_difference"}:
        replacements.extend(
            [
                (left, d_both_valid & ~d_right_nonempty),
                (right, d_both_valid & d_right_nonempty & ~d_left_nonempty),
            ]
        )
    elif op == "difference":
        replacements.append((left, d_both_valid & (~d_left_nonempty | ~d_right_nonempty)))
    elif op == "intersection":
        replacements.extend(
            [
                (left, d_both_valid & ~d_left_nonempty),
                (right, d_both_valid & d_left_nonempty & ~d_right_nonempty),
            ]
        )
    else:  # pragma: no cover - guarded by the public operation contract
        return result

    def _has_bounded_family_widths(owned: OwnedGeometryArray) -> bool:
        state = owned._ensure_device_state(preserve_indexed_view=True)
        for family, buffer in state.families.items():
            fixed_size = getattr(buffer, "fixed_size", None)
            if family is GeometryFamily.POINT:
                continue
            if fixed_size is None:
                return False
            if fixed_size.max_coord_count_per_row is None:
                return False
            if (
                family
                in {
                    GeometryFamily.POLYGON,
                    GeometryFamily.MULTILINESTRING,
                    GeometryFamily.MULTIPOLYGON,
                }
                and fixed_size.max_first_level_count_per_row is None
            ):
                return False
            if (
                family is GeometryFamily.MULTIPOLYGON
                and fixed_size.max_second_level_count_per_row is None
            ):
                return False
        return True

    selector = (
        device_physical_select_owned_capacity_partitions
        if all(_has_bounded_family_widths(owned) for owned, _mask in replacements)
        else device_select_owned_capacity_partitions
    )
    selected_result = selector(
        _empty_device_constructive_output(result.row_count),
        replacements,
    )
    if op == "intersection":
        for metadata_name in (
            "_polygon_rect_boundary_overlap",
            "_polygon_rect_exact_polygon_only",
            "_polygon_intersection_exact_area",
            "_polygon_intersection_lower_dimensional_remnant",
        ):
            metadata = getattr(result, metadata_name, None)
            if metadata is None:
                continue
            d_metadata = cp.asarray(metadata, dtype=cp.bool_)
            if d_metadata.shape[0] == result.row_count:
                setattr(selected_result, metadata_name, d_metadata & d_core)
        remnant_parts = getattr(
            result,
            "_polygon_intersection_lower_dimensional_parts",
            None,
        )
        if remnant_parts is not None:
            selected_result._polygon_intersection_lower_dimensional_parts = tuple(
                device_mask_owned_capacity(part, d_core)
                for part in remnant_parts
            )
    return selected_result


def _dispatch_mixed_binary_constructive_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> object | None:
    """Dispatch mixed families through homogeneous row-capacity carriers."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    from vibespatial.api._native_rowset import NativeDeviceSelection

    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    d_left_valid = cp.asarray(left_state.validity, dtype=cp.bool_)
    d_right_valid = cp.asarray(right_state.validity, dtype=cp.bool_)
    d_left_tags = cp.asarray(left_state.tags, dtype=cp.int8)
    d_right_tags = cp.asarray(right_state.tags, dtype=cp.int8)
    row_count = int(left.row_count)
    base = _empty_device_constructive_output(row_count)
    replacements: list[tuple[OwnedGeometryArray, DeviceArray]] = []
    native_replacements = []
    pair_count = 0

    from vibespatial.api._native_result_core import GeometryNativeResult

    for left_tag, right_tag in _valid_family_tag_pairs(left, right):
        pair_count += 1
        left_family = TAG_FAMILIES.get(left_tag)
        right_family = TAG_FAMILIES.get(right_tag)
        if left_family is None or right_family is None:
            return None
        if not _pair_supports_gpu_constructive(op, left_family, right_family):
            return None

        d_pair_mask = (
            d_left_valid
            & d_right_valid
            & (d_left_tags == cp.int8(left_tag))
            & (d_right_tags == cp.int8(right_tag))
        )
        selection = NativeDeviceSelection.from_mask(
            d_pair_mask,
            source_row_count=row_count,
        )
        left_subset = device_take_owned_family_capacity_selection(
            left,
            selection,
            left_family,
        )
        right_subset = device_take_owned_family_capacity_selection(
            right,
            selection,
            right_family,
        )
        sub_result = _binary_constructive_gpu(
            op,
            left_subset,
            right_subset,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=_cached_right_segments,
        )
        if sub_result is None or sub_result.row_count != row_count:
            return None

        d_inverse = cp.empty(row_count, dtype=cp.int64)
        d_inverse[selection.partition_capacity_positions()] = cp.arange(
            row_count,
            dtype=cp.int64,
        )
        if isinstance(sub_result, GeometryNativeResult):
            native_replacements.append(sub_result.permute_capacity(d_inverse))
        else:
            row_aligned_result = sub_result._device_indexed_take(
                d_inverse,
                assume_unique_indices=True,
            )
            replacements.append((row_aligned_result, d_pair_mask))

    result = device_select_owned_capacity_partitions(base, replacements)
    if native_replacements:
        native_replacements.insert(0, result)
        result = _compose_aligned_native_geometries(
            native_replacements,
            row_count=row_count,
            crs=None,
        )
    record_dispatch_event(
        surface="vibespatial.constructive.binary",
        operation=op,
        implementation="mixed_family_capacity_dispatch_gpu",
        reason=(
            "mixed-family constructive retained family-pair work in "
            "device-counted row-capacity carriers"
        ),
        detail=(
            f"rows={row_count}; family_pairs={pair_count}; "
            "workload_shape=aligned_family_pair_capacity"
        ),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return result


# ---------------------------------------------------------------------------
# Registered kernel variants
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "binary_constructive",
    "gpu-overlay-pip",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point",
        "linestring",
        "polygon",
        "multipoint",
        "multilinestring",
        "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "overlay", "pip"),
)
def _binary_constructive_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
    _chunk_rows: int | None = None,
) -> object | None:
    """GPU binary constructive for all family combinations.

    Dispatches to specialized GPU kernels based on the geometry family
    combination. Structural declines return ``None``; unexpected native
    failures propagate atomically.

    Parameters
    ----------
    dispatch_mode : ExecutionMode
        Propagated to inner kernels (polygon_intersection, overlay).
        Default is GPU since this function is only called when the
        outer dispatch has already selected GPU execution.
    _cached_right_segments : DeviceSegmentTable, optional
        Pre-extracted right-side segments for reuse (lyy.15).
    """
    family_pairs = _valid_family_tag_pairs(left, right)
    if family_pairs == []:
        return _empty_device_constructive_output(left.row_count)
    if op == "intersection" and _is_lineal_only(left) and _is_lineal_only(right):
        return _dispatch_linestring_linestring_gpu(op, left, right)
    if op == "difference" and _is_lineal_only(left) and _is_lineal_only(right):
        return _dispatch_lineal_lineal_difference_gpu(
            left,
            right,
            dispatch_mode=dispatch_mode,
        )

    if _needs_grouped_gpu_dispatch(left, right):
        return _dispatch_mixed_binary_constructive_gpu(
            op,
            left,
            right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=_cached_right_segments,
        )

    # --- Point-Point ---
    if _is_point_only(left) and _is_point_only(right):
        return _dispatch_point_point_gpu(op, left, right)

    # --- Point-Polygon (existing PIP fast path) ---
    if _is_point_only(left) and _is_polygon_only(right):
        if op == "intersection":
            return _intersection_point_polygon_gpu(left, right)
        if op == "difference":
            return _difference_point_polygon_gpu(left, right)

    if _is_polygon_only(left) and _is_point_only(right):
        if op == "intersection":
            return _intersection_point_polygon_gpu(right, left)
        if op == "difference":
            return left

    # --- Point-LineString ---
    if _is_point_only(left) and _is_linestring_only(right):
        return _dispatch_point_linestring_gpu(op, left, right)

    if _is_linestring_only(left) and _is_point_only(right):
        if op == "intersection":
            return _dispatch_point_linestring_gpu("intersection", right, left)

    # --- MultiPoint-Polygon ---
    if _is_multipoint_only(left) and _is_polygon_only(right):
        return _dispatch_multipoint_polygon_gpu(op, left, right)

    if _is_polygon_only(left) and _is_multipoint_only(right):
        if op == "intersection":
            return _dispatch_multipoint_polygon_gpu("intersection", right, left)
        if op == "difference":
            return left

    # --- Lineal-Polygonal collective topology ---
    if op in {"intersection", "difference"} and _is_lineal_only(left) and _is_polygon_only(right):
        return _dispatch_lineal_polygonal_constructive_gpu(
            op,
            left,
            right,
            dispatch_mode=dispatch_mode,
        )

    if op == "intersection" and _is_polygon_only(left) and _is_lineal_only(right):
        return _dispatch_lineal_polygonal_constructive_gpu(
            "intersection",
            right,
            left,
            dispatch_mode=dispatch_mode,
        )

    # --- LineString-LineString ---
    if _is_linestring_only(left) and _is_linestring_only(right):
        return _dispatch_linestring_linestring_gpu(op, left, right)

    # --- Polygon-Polygon GPU kernel fast paths ---
    if _is_polygon_only(left) and _is_polygon_only(right):
        if op == "union":
            result = _dispatch_polygon_partition_union_gpu(
                left,
                right,
                dispatch_mode=dispatch_mode,
            )
            if result is None or result.row_count != left.row_count:
                return None
            return result

        if op == "intersection":
            if _chunk_rows is not None and left.row_count > int(_chunk_rows):
                result = _dispatch_chunked_polygon_intersection_gpu(
                    left,
                    right,
                    chunk_rows=int(_chunk_rows),
                    dispatch_mode=dispatch_mode,
                )
                if result is None or result.row_count != left.row_count:
                    return None
                return result
            result = _dispatch_partitioned_polygon_intersection_gpu(
                left,
                right,
                dispatch_mode=dispatch_mode,
            )
            if result is None or result.row_count != left.row_count:
                return None
            return result

        if op == "difference":
            result = _dispatch_polygon_difference_overlay_batched_gpu(
                left,
                right,
                dispatch_mode=dispatch_mode,
            )
            if result is None or result.row_count != left.row_count:
                return None
            return result

        result = _dispatch_overlay_gpu(
            op,
            left,
            right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=_cached_right_segments,
            _row_isolated=True,
            _include_same_side_splits=(op == "symmetric_difference"),
        )
        if result.row_count != left.row_count:
            raise RuntimeError("polygon overlay violated aligned constructive row-count contract")
        return result

    # For any remaining family pair not covered above, return None to
    # trigger CPU fallback.  This should only happen for exotic multi-type
    # combinations (e.g., MultiLineString-MultiPolygon).
    return None


# ---------------------------------------------------------------------------
# Non-polygon GPU dispatch helpers
# ---------------------------------------------------------------------------


def _dispatch_point_point_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Dispatch Point-Point GPU constructive to the appropriate kernel."""
    from vibespatial.kernels.constructive.nonpolygon_binary import (
        point_point_difference,
        point_point_intersection,
        point_point_symmetric_difference,
        point_point_union,
    )

    dispatch = {
        "intersection": point_point_intersection,
        "difference": point_point_difference,
        "union": point_point_union,
        "symmetric_difference": point_point_symmetric_difference,
    }
    return dispatch[op](left, right)


def _dispatch_point_linestring_gpu(
    op: str,
    points: OwnedGeometryArray,
    linestrings: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Dispatch Point-LineString GPU constructive."""
    from vibespatial.kernels.constructive.nonpolygon_binary import (
        point_linestring_difference,
        point_linestring_intersection,
    )

    if op == "intersection":
        return point_linestring_intersection(points, linestrings)
    elif op == "difference":
        return point_linestring_difference(points, linestrings)
    # union/symmetric_difference of Point-LineString produces mixed-type
    # results. Fall back to CPU for now.
    return None


def _dispatch_multipoint_polygon_gpu(
    op: str,
    multipoints: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Dispatch MultiPoint-Polygon GPU constructive."""
    from vibespatial.constructive.multipoint_polygon_constructive import (
        multipoint_polygon_difference,
        multipoint_polygon_intersection,
    )

    if op == "intersection":
        return multipoint_polygon_intersection(multipoints, polygons)
    elif op == "difference":
        return multipoint_polygon_difference(multipoints, polygons)
    # union/symmetric_difference produce mixed types. Fall back.
    return None


def _dispatch_linestring_linestring_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> object:
    """Dispatch LineString-LineString GPU constructive."""
    from vibespatial.kernels.constructive.nonpolygon_binary import (
        linestring_linestring_intersection_native,
    )

    if op == "intersection":
        return linestring_linestring_intersection_native(left, right)
    # difference/union/symmetric_difference of LineString-LineString are complex
    # mixed-type operations. Fall back to CPU.
    return None


def _binary_constructive_result(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    grid_size: float | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
    workload_shape: WorkloadShape | None = None,
):
    """Return an element-wise constructive result through the native carrier.

    Uses the standard dispatch framework: ``plan_dispatch_selection`` for
    GPU/CPU routing, ``select_precision_plan`` for precision, and
    ``record_dispatch_event`` for observability.

    GPU paths:
    - Polygon-Polygon pairs: overlay pipeline (face selection).
    - Point-Polygon intersection/difference: PIP kernel + validity masking.

    Parameters
    ----------
    op : str
        One of 'intersection', 'union', 'difference', 'symmetric_difference'.
    left, right : OwnedGeometryArray
        Input geometry arrays (must have same row count).
    grid_size : float or None
        Snap grid size for GEOS precision model.  When set, forces the
        CPU/Shapely path because the GPU pipeline does not support
        snapped precision.
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint.
    precision : PrecisionMode or str, default AUTO
        Precision mode for GPU path.
    _cached_right_segments : DeviceSegmentTable, optional
        Pre-extracted right-side device segments for reuse (lyy.15).
        Passed through to the overlay pipeline to avoid redundant
        segment extraction in N-vs-1 overlay loops.
    """
    if op not in _CONSTRUCTIVE_OPS:
        raise ValueError(f"unsupported constructive operation: {op}")

    from vibespatial.runtime.crossover import detect_workload_shape

    workload = workload_shape or detect_workload_shape(left.row_count, right.row_count)

    # Broadcast-right remains a logical row-indirected carrier. Downstream
    # family dispatchers consume its device index map directly.
    is_broadcast = workload in (WorkloadShape.BROADCAST_RIGHT, WorkloadShape.SCALAR_RIGHT)
    if is_broadcast:
        right = tile_single_row(right, left.row_count)

    if left.row_count == 0:
        empty = from_shapely_geometries([])
        from vibespatial.api._native_result_core import GeometryNativeResult

        return GeometryNativeResult.from_owned(empty, crs=None)

    # Force CPU when grid_size is set (GPU pipeline doesn't support snapped precision)
    effective_mode = dispatch_mode
    if grid_size is not None:
        effective_mode = ExecutionMode.CPU
    normalized_effective_mode = (
        effective_mode
        if isinstance(effective_mode, ExecutionMode)
        else ExecutionMode(effective_mode)
    )
    strict_gpu_promotion = (
        normalized_effective_mode is ExecutionMode.AUTO
        and strict_native_mode_enabled()
        and _supports_gpu_constructive(op, left, right)
    )
    if strict_gpu_promotion:
        effective_mode = ExecutionMode.GPU

    selection = plan_dispatch_selection(
        kernel_name="binary_constructive",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=left.row_count,
        requested_mode=effective_mode,
        requested_precision=precision,
        workload_shape=workload,
        work_estimate=_binary_constructive_work_estimate(
            left,
            right,
            workload=workload,
        ),
        current_residency=combined_residency(left, right),
    )
    selection_reason = selection.reason
    if strict_gpu_promotion:
        selection_reason = (
            f"{selection_reason}; strict-native promoted {workload.value} {op} to GPU"
        )

    gpu_attempted = False
    if selection.selected is ExecutionMode.GPU:
        # ADR-0002: CONSTRUCTIVE kernels stay fp64.  precision_plan is
        # computed for observability (dispatch event detail) only; the
        # overlay and PIP kernels manage their own precision internally.
        precision_plan = selection.precision_plan
        gpu_attempted = True
        result = _binary_constructive_gpu(
            op,
            left,
            right,
            dispatch_mode=selection.selected,
            _cached_right_segments=_cached_right_segments,
            _chunk_rows=_polygon_constructive_chunk_rows(
                int(selection.chunk_rows),
            ),
        )
        if result is not None:
            result = _apply_binary_empty_row_semantics_gpu(
                op,
                left,
                right,
                result,
            )
            from vibespatial.api._native_result_core import GeometryNativeResult

            if op == "intersection":
                segment_bound = _polygon_intersection_segment_span_bound(left, right)
                if isinstance(result, OwnedGeometryArray):
                    _apply_polygonal_device_row_segment_bound(result, segment_bound)
                elif isinstance(result, GeometryNativeResult):
                    concrete_parts = (
                        result.composition.parts if result.composition is not None else ()
                    )
                    if result.owned is not None:
                        _apply_polygonal_device_row_segment_bound(
                            result.owned,
                            segment_bound,
                        )
                    for part in concrete_parts:
                        if part.geometry.owned is not None:
                            _apply_polygonal_device_row_segment_bound(
                                part.geometry.owned,
                                segment_bound,
                            )

            native_result = (
                result
                if isinstance(result, GeometryNativeResult)
                else GeometryNativeResult.from_owned(result, crs=None)
            )
        if result is not None:
            record_dispatch_event(
                surface=f"geopandas.array.{op}",
                operation=op,
                implementation="binary_constructive_gpu",
                reason=selection_reason,
                detail=(
                    f"rows={left.row_count}, "
                    f"precision={precision_plan.compute_precision.value}, "
                    f"workload={workload.value}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return native_result

    # CPU fallback: Shapely element-wise
    if grid_size is not None:
        fallback_reason = "grid_size requires GEOS precision model"
    elif gpu_attempted:
        fallback_reason = "GPU kernel returned None (unsupported family pair)"
    else:
        fallback_reason = selection_reason

    # Phase 24: Guard CPU fallback when GPU was explicitly requested with
    # device-resident input.  This should not happen silently.
    if gpu_attempted and selection.requested is ExecutionMode.GPU:
        warnings.warn(
            f"[vibeSpatial] binary_constructive '{op}': GPU was explicitly "
            f"requested but the GPU kernel returned None for this family "
            f"pair. Falling back to CPU/Shapely with D2H transfer. "
            f"rows={left.row_count}",
            stacklevel=2,
        )

    record_fallback_event(
        surface=f"geopandas.array.{op}",
        reason=fallback_reason,
        detail=f"rows={left.row_count}, op={op}, workload={workload.value}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
        pipeline="binary_constructive_owned",
        d2h_transfer=gpu_attempted,  # D2H transfer occurs when GPU was attempted but fell back
    )

    result = binary_constructive_cpu(op, left, right, grid_size=grid_size)
    record_dispatch_event(
        surface=f"geopandas.array.{op}",
        operation=op,
        implementation="binary_constructive_cpu",
        reason=fallback_reason,
        detail=f"rows={left.row_count}, workload={workload.value}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    from vibespatial.api._native_result_core import GeometryNativeResult

    return GeometryNativeResult.from_owned(result, crs=None)


def _physicalize_binary_constructive_owned_boundary(native_result):
    """Cross the named legacy-owned boundary for a singular native result.

    Native constructive execution retains partitioned dynamic output.  Callers
    whose contract still requires one ``OwnedGeometryArray`` cross that shape
    boundary here, never through result inspection or an incidental property.
    """
    owned = native_result.cached_owned()
    if owned is not None:
        return owned
    composition = native_result.composition
    if composition is None:
        raise NotImplementedError(
            "binary constructive result has no device-owned geometry carrier"
        )
    owned = composition._singular_owned_device()
    if owned is None:
        raise NotImplementedError(
            "binary_constructive_owned requires at most one concrete geometry per row; "
            "use binary_constructive_native for heterogeneous compositions"
        )
    record_dispatch_event(
        surface="vibespatial.constructive.binary",
        operation="owned_physicalization_boundary",
        implementation="binary_constructive_owned_composition_physicalization",
        reason=(
            "legacy owned caller explicitly physicalized a singular native "
            "geometry composition"
        ),
        detail=(
            f"rows={native_result.row_count}; parts={len(composition.parts)}; "
            "workload_shape=terminal_owned_physicalization"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return owned


def binary_constructive_owned(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    grid_size: float | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
    workload_shape: WorkloadShape | None = None,
) -> OwnedGeometryArray:
    """Return a constructive result at the explicit legacy-owned boundary."""
    return _physicalize_binary_constructive_owned_boundary(
        _binary_constructive_result(
            op,
            left,
            right,
            grid_size=grid_size,
            dispatch_mode=dispatch_mode,
            precision=precision,
            _cached_right_segments=_cached_right_segments,
            workload_shape=workload_shape,
        )
    )


def binary_constructive_native(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    **kwargs,
):
    """Return binary constructive geometry through the native result boundary."""
    return _binary_constructive_result(
        op,
        left,
        right,
        **kwargs,
    )
