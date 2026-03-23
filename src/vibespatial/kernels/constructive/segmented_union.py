"""Segmented union_all kernel: per-group polygon union via GPU tree-reduce.

ADR-0002: CONSTRUCTIVE class -- fp64 by design on all devices.
ADR-0033: Tier classification -- delegates to overlay pipeline (Tier 1 NVRTC
          + Tier 3a CCCL + Tier 2 CuPy) via overlay_union_owned.
ADR-0034: Inherits overlay pipeline precompilation; no new NVRTC source.

Algorithm
---------
For each group defined by ``group_offsets`` (CSR-style):
  - size 0: produce empty polygon
  - size 1: pass through unchanged
  - size 2: single pairwise overlay union
  - size N>2: binary-tree reduction in ceil(log2(N)) rounds

Tree reduction intermediates stay device-resident throughout.  Final
group results are concatenated at the buffer level (no Shapely
round-trip) via ``_concat_owned_arrays``.

The overlay pipeline is strictly pairwise (left[i] vs right[i]), so each
pair union is a separate GPU dispatch.  Groups are iterated serially with
GPU parallelism *within* each overlay_union_owned call.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely.geometry import Polygon

from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    normalize_precision_mode,
    select_precision_plan,
)
from vibespatial.runtime.robustness import select_robustness_plan

if TYPE_CHECKING:
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FamilyGeometryBuffer, OwnedGeometryArray

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

_EMPTY_POLYGON = Polygon()
# Pre-built empty-polygon OwnedGeometryArray to avoid repeated from_shapely_geometries
# calls in the hot loop (one allocation per empty group otherwise).
_EMPTY_OWNED = None


def _get_empty_owned():
    """Lazily create the empty polygon OwnedGeometryArray constant."""
    global _EMPTY_OWNED
    if _EMPTY_OWNED is None:
        from vibespatial.geometry.owned import from_shapely_geometries as _fsgo

        _EMPTY_OWNED = _fsgo([Polygon()])
    return _EMPTY_OWNED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segmented_union_all(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Union all geometries within each group.  Returns one geometry per group.

    Parameters
    ----------
    geometries : OwnedGeometryArray
        Input polygons (device- or host-resident).
    group_offsets : array-like
        CSR-style int32/int64 offsets.  Group *i* contains
        ``geometries[group_offsets[i]:group_offsets[i+1]]``.
        Length is ``n_groups + 1``.
    dispatch_mode : ExecutionMode or str
        Execution mode hint (AUTO, GPU, CPU).
    precision : PrecisionMode or str
        Precision mode.  CONSTRUCTIVE kernels stay fp64 per ADR-0002.

    Returns
    -------
    OwnedGeometryArray
        One geometry per group.  May contain MultiPolygon when union
        produces disconnected regions.  Empty groups produce empty Polygon.
    """
    from vibespatial.geometry.owned import from_shapely_geometries

    requested = (
        dispatch_mode
        if isinstance(dispatch_mode, ExecutionMode)
        else ExecutionMode(dispatch_mode)
    )
    precision_mode = normalize_precision_mode(precision)

    # Ensure group_offsets is a host numpy array (small metadata).
    # np.asarray handles CuPy arrays via __array__ protocol.
    group_offsets = np.asarray(group_offsets, dtype=np.int64)
    n_groups = len(group_offsets) - 1
    if n_groups < 0:
        raise ValueError("group_offsets must have length >= 1")
    if n_groups == 0:
        return from_shapely_geometries([])

    total_geoms = int(group_offsets[-1])

    # Dispatch selection
    selection = plan_dispatch_selection(
        kernel_name="segmented_union_all",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=total_geoms,
        requested_mode=requested,
    )

    # ADR-0002: CONSTRUCTIVE kernels stay fp64.  Precision plan is computed
    # for observability (dispatch event detail) only.
    precision_plan = select_precision_plan(
        runtime_selection=selection,
        kernel_class=KernelClass.CONSTRUCTIVE,
        requested=precision_mode,
    )
    select_robustness_plan(
        kernel_class=KernelClass.CONSTRUCTIVE,
        precision_plan=precision_plan,
    )

    if selection.selected is ExecutionMode.GPU:
        result = _segmented_union_gpu(
            geometries,
            group_offsets,
            n_groups=n_groups,
            precision_plan=precision_plan,
        )
        if result is not None:
            record_dispatch_event(
                surface="segmented_union_all",
                operation="segmented_union_all",
                implementation="gpu_tree_reduce_overlay",
                reason=selection.reason,
                detail=(
                    f"groups={n_groups}, total_geoms={total_geoms}, "
                    f"precision={precision_plan.compute_precision.value}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            result.record_runtime_selection(selection)
            return result

    # CPU fallback
    result = _segmented_union_cpu(geometries, group_offsets, n_groups=n_groups)
    record_dispatch_event(
        surface="segmented_union_all",
        operation="segmented_union_all",
        implementation="shapely_union_all",
        reason=selection.reason if selection.selected is ExecutionMode.CPU else "GPU fallback to CPU",
        detail=f"groups={n_groups}, total_geoms={total_geoms}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    result.record_runtime_selection(selection)
    return result


# ---------------------------------------------------------------------------
# CPU variant: per-group shapely.union_all
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "segmented_union_all",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("polygon", "multipolygon"),
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=True,
    tags=("constructive", "segmented-union", "grouped"),
)
def _segmented_union_cpu_variant(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.CPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """CPU variant: iterate groups and call shapely.union_all per group."""
    group_offsets = np.asarray(group_offsets, dtype=np.int64)
    n_groups = len(group_offsets) - 1
    return _segmented_union_cpu(geometries, group_offsets, n_groups=n_groups)


def _segmented_union_cpu(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    n_groups: int,
) -> OwnedGeometryArray:
    """CPU implementation: per-group shapely.union_all."""
    from vibespatial.geometry.owned import from_shapely_geometries

    # Single bulk materialization to Shapely (1 D->H if device-resident).
    all_geoms = np.asarray(geometries.to_shapely(), dtype=object)

    results: list[object] = []
    for g in range(n_groups):
        start = int(group_offsets[g])
        end = int(group_offsets[g + 1])
        group_size = end - start

        if group_size == 0:
            results.append(_EMPTY_POLYGON)
        elif group_size == 1:
            geom = all_geoms[start]
            results.append(geom if geom is not None else _EMPTY_POLYGON)
        else:
            block = all_geoms[start:end]
            valid = block[block != np.array(None)]
            if len(valid) == 0:
                results.append(_EMPTY_POLYGON)
            elif len(valid) == 1:
                results.append(valid[0])
            else:
                merged = shapely.union_all(valid)
                if merged is not None and not shapely.is_valid(merged):
                    merged = shapely.make_valid(merged)
                results.append(merged if merged is not None else _EMPTY_POLYGON)

    return from_shapely_geometries(results)


# ---------------------------------------------------------------------------
# GPU variant: tree-reduce via overlay_union_owned per group
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "segmented_union_all",
    "gpu-overlay-tree-reduce",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("polygon", "multipolygon"),
    execution_modes=(ExecutionMode.GPU,),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP64),
    tags=("constructive", "segmented-union", "gpu", "tree-reduce"),
)
def _segmented_union_gpu_variant(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """GPU variant: tree-reduce pairwise overlay union per group."""
    group_offsets = np.asarray(group_offsets, dtype=np.int64)
    n_groups = len(group_offsets) - 1
    precision_mode = normalize_precision_mode(precision)
    selection = plan_dispatch_selection(
        kernel_name="segmented_union_all",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=int(group_offsets[-1]),
        requested_mode=dispatch_mode,
    )
    precision_plan = select_precision_plan(
        runtime_selection=selection,
        kernel_class=KernelClass.CONSTRUCTIVE,
        requested=precision_mode,
    )
    return _segmented_union_gpu(
        geometries, group_offsets, n_groups=n_groups, precision_plan=precision_plan,
    )


def _segmented_union_gpu(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    n_groups: int,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray | None:
    """GPU tree-reduce: union geometries within each group.

    For each group, performs binary-tree reduction using overlay_union_owned.
    Intermediate results stay device-resident (ADR-0005 zero-copy).
    Falls back to None on failure so the caller can retry on CPU.

    ADR-0002: CONSTRUCTIVE class, fp64 (segment intersection precision).
    ADR-0033: Inherits overlay pipeline tiers (NVRTC + CCCL + CuPy).
    """
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    # Validate: GPU overlay requires polygon-family geometries.
    polygon_tags = {FAMILY_TAGS[GeometryFamily.POLYGON], FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]}
    valid_tags = np.isin(geometries.tags[geometries.validity], list(polygon_tags))
    if not np.all(valid_tags):
        # Non-polygon geometry present: fall back to CPU.
        return None

    group_results: list[OwnedGeometryArray] = []

    for g in range(n_groups):
        start = int(group_offsets[g])
        end = int(group_offsets[g + 1])
        group_size = end - start

        if group_size == 0:
            group_results.append(_get_empty_owned())
            continue

        if group_size == 1:
            single = geometries.take(np.array([start], dtype=np.intp))
            if not single.validity[0]:
                group_results.append(_get_empty_owned())
            else:
                group_results.append(single)
            continue

        # Extract this group's geometries.
        indices = np.arange(start, end, dtype=np.intp)
        group_owned = geometries.take(indices)

        # Filter out invalid/empty rows (vectorized, not Python loop).
        keep = np.flatnonzero(group_owned.validity)
        if keep.size == 0:
            group_results.append(_get_empty_owned())
            continue
        if keep.size < group_owned.row_count:
            group_owned = group_owned.take(keep)

        if group_owned.row_count == 1:
            group_results.append(group_owned)
            continue

        # Tree-reduce: each round halves the geometry count.
        try:
            reduced = _tree_reduce_group(group_owned)
            group_results.append(reduced)
        except Exception:
            # Fall back to CPU for this group on any GPU overlay failure.
            return None

    # Buffer-level concatenation: no Shapely round-trip (zero-copy).
    return _concat_owned_arrays(group_results)


def _tree_reduce_group(group_owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Binary-tree reduction of group_owned via overlay_union_owned.

    Intermediate results stay device-resident.  O(log2(N)) rounds.
    """
    from vibespatial.overlay.gpu import overlay_union_owned

    # Split into single-row OwnedGeometryArrays for pairwise reduction.
    current: list[OwnedGeometryArray] = [
        group_owned.take(np.array([i], dtype=np.intp))
        for i in range(group_owned.row_count)
    ]

    rounds = 0
    max_rounds = int(math.ceil(math.log2(max(len(current), 2)))) + 1

    while len(current) > 1 and rounds < max_rounds:
        next_round: list[OwnedGeometryArray] = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                merged = overlay_union_owned(
                    current[i],
                    current[i + 1],
                    dispatch_mode=ExecutionMode.GPU,
                )
                next_round.append(merged)
            else:
                # Odd element passes through.
                next_round.append(current[i])
        # Explicit cleanup: release previous round's intermediates promptly
        # to avoid accumulating device memory across reduction rounds.
        del current
        current = next_round
        rounds += 1

    return current[0]


def _concat_owned_arrays(
    arrays: list[OwnedGeometryArray],
) -> OwnedGeometryArray:
    """Concatenate OwnedGeometryArrays at the buffer level (zero-copy).

    No Shapely round-trip: merges validity, tags, family_row_offsets, and
    per-family coordinate/offset buffers directly.  Preserves device
    residency when all inputs are host-resident (host buffers stay host);
    for device-resident inputs, host buffers are used if materialised,
    otherwise we force host materialisation once here (unavoidable for
    buffer-level concat of mixed-residency arrays, but no per-element
    to_shapely round-trip).
    """
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        NULL_TAG,
        DiagnosticKind,
    )
    from vibespatial.geometry.owned import OwnedGeometryArray as _OGA

    TAG_FAMILIES_LOCAL = {v: k for k, v in FAMILY_TAGS.items()}

    if not arrays:
        return _OGA(
            validity=np.array([], dtype=bool),
            tags=np.array([], dtype=np.int8),
            family_row_offsets=np.array([], dtype=np.int32),
            families={},
        )

    if len(arrays) == 1:
        return arrays[0]

    # Ensure host buffers are available for all inputs.
    for arr in arrays:
        arr._ensure_host_state()

    # Concatenate top-level arrays.
    new_validity = np.concatenate([o.validity for o in arrays])
    new_tags = np.concatenate([o.tags for o in arrays])

    # Collect per-family buffers across all arrays.
    all_families: dict[GeometryFamily, list[FamilyGeometryBuffer]] = {}
    for owned in arrays:
        for family, buf in owned.families.items():
            all_families.setdefault(family, []).append(buf)

    # Merge per-family buffers.
    new_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    for family, bufs in all_families.items():
        new_families[family] = _concat_family_buffers(family, bufs)

    # Rebuild family_row_offsets with correct cumulative offsets.
    new_family_row_offsets = np.full(new_validity.size, -1, dtype=np.int32)
    family_cursor: dict[GeometryFamily, int] = {f: 0 for f in all_families}
    global_offset = 0
    for owned in arrays:
        n = owned.row_count
        for i in range(n):
            global_idx = global_offset + i
            tag = new_tags[global_idx]
            if tag == NULL_TAG:
                continue
            family = TAG_FAMILIES_LOCAL[int(tag)]
            new_family_row_offsets[global_idx] = (
                family_cursor[family] + owned.family_row_offsets[i]
            )
        # Advance cursors.
        for family in all_families:
            if family in owned.families:
                family_cursor[family] += owned.families[family].row_count
        global_offset += n

    result = _OGA(
        validity=new_validity,
        tags=new_tags,
        family_row_offsets=new_family_row_offsets,
        families=new_families,
    )
    result._record(
        DiagnosticKind.CREATED,
        f"segmented_union: buffer-level concat of {len(arrays)} arrays",
        visible=False,
    )
    return result


def _concat_family_buffers(
    family: GeometryFamily,
    buffers: list[FamilyGeometryBuffer],
) -> FamilyGeometryBuffer:
    """Concatenate multiple FamilyGeometryBuffers for the same family.

    Mirrors the logic in ``DeviceGeometryArray._concat_same_type`` but is
    self-contained so this module does not import the device_array module.
    """
    from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
    from vibespatial.geometry.owned import FamilyGeometryBuffer as _FGB

    if len(buffers) == 1:
        return buffers[0]

    schema = get_geometry_buffer_schema(family)
    total_rows = sum(b.row_count for b in buffers)

    all_x = [b.x for b in buffers]
    all_y = [b.y for b in buffers]
    new_x = np.concatenate(all_x) if any(a.size for a in all_x) else np.empty(0, dtype=np.float64)
    new_y = np.concatenate(all_y) if any(a.size for a in all_y) else np.empty(0, dtype=np.float64)
    new_empty_mask = np.concatenate([b.empty_mask for b in buffers])

    # Concatenate bounds if all have them.
    if all(b.bounds is not None for b in buffers):
        new_bounds = np.concatenate([b.bounds for b in buffers])
    else:
        new_bounds = None

    # Concatenate geometry_offsets with cumulative shift.
    coord_cursor = 0
    geom_offset_parts: list[np.ndarray] = []
    for b in buffers:
        shifted = b.geometry_offsets[:-1] + coord_cursor
        geom_offset_parts.append(shifted)
        coord_cursor += int(b.geometry_offsets[-1])
    geom_offset_parts.append(np.array([coord_cursor], dtype=np.int32))
    new_geometry_offsets = np.concatenate(geom_offset_parts)

    new_part_offsets = None
    new_ring_offsets = None

    if family is GeometryFamily.POLYGON:
        ring_cursor = 0
        ring_parts: list[np.ndarray] = []
        for b in buffers:
            shifted = b.ring_offsets[:-1] + ring_cursor
            ring_parts.append(shifted)
            ring_cursor += int(b.ring_offsets[-1])
        ring_parts.append(np.array([ring_cursor], dtype=np.int32))
        new_ring_offsets = np.concatenate(ring_parts)

    elif family is GeometryFamily.MULTILINESTRING:
        part_cursor = 0
        part_parts: list[np.ndarray] = []
        for b in buffers:
            shifted = b.part_offsets[:-1] + part_cursor
            part_parts.append(shifted)
            part_cursor += int(b.part_offsets[-1])
        part_parts.append(np.array([part_cursor], dtype=np.int32))
        new_part_offsets = np.concatenate(part_parts)

    elif family is GeometryFamily.MULTIPOLYGON:
        part_cursor = 0
        part_parts_list: list[np.ndarray] = []
        for b in buffers:
            shifted = b.part_offsets[:-1] + part_cursor
            part_parts_list.append(shifted)
            part_cursor += int(b.part_offsets[-1])
        part_parts_list.append(np.array([part_cursor], dtype=np.int32))
        new_part_offsets = np.concatenate(part_parts_list)

        ring_cursor = 0
        ring_parts_list: list[np.ndarray] = []
        for b in buffers:
            shifted = b.ring_offsets[:-1] + ring_cursor
            ring_parts_list.append(shifted)
            ring_cursor += int(b.ring_offsets[-1])
        ring_parts_list.append(np.array([ring_cursor], dtype=np.int32))
        new_ring_offsets = np.concatenate(ring_parts_list)

    return _FGB(
        family=family,
        schema=schema,
        row_count=total_rows,
        x=new_x,
        y=new_y,
        geometry_offsets=new_geometry_offsets,
        empty_mask=new_empty_mask,
        part_offsets=new_part_offsets,
        ring_offsets=new_ring_offsets,
        bounds=new_bounds,
    )
