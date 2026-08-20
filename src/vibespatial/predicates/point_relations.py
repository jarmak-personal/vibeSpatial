from __future__ import annotations

from dataclasses import replace

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray
from vibespatial.predicates.point_relations_kernels import (
    _MULTIPOINT_BINARY_RELATIONS_KERNEL_SOURCE,
    _MULTIPOINT_KERNEL_NAMES,
    _POINT_BINARY_RELATIONS_KERNEL_NAMES,
    _POINT_BINARY_RELATIONS_KERNEL_SOURCE,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.precision import (
    CompensationMode,
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    RefinementMode,
    normalize_precision_mode,
    select_precision_plan,
)

POINT_LOCATION_OUTSIDE = np.uint8(0)
POINT_LOCATION_BOUNDARY = np.uint8(1)
POINT_LOCATION_INTERIOR = np.uint8(2)

request_nvrtc_warmup(
    [
        (
            "point-binary-relations-fp64",
            _POINT_BINARY_RELATIONS_KERNEL_SOURCE,
            _POINT_BINARY_RELATIONS_KERNEL_NAMES,
        ),
        (
            "multipoint-binary-relations-fp64",
            _MULTIPOINT_BINARY_RELATIONS_KERNEL_SOURCE,
            _MULTIPOINT_KERNEL_NAMES,
        ),
    ]
)


def _point_binary_relation_kernels():
    return compile_kernel_group(
        "point-binary-relations-fp64",
        _POINT_BINARY_RELATIONS_KERNEL_SOURCE,
        _POINT_BINARY_RELATIONS_KERNEL_NAMES,
    )


def _multipoint_relation_kernels():
    return compile_kernel_group(
        "multipoint-binary-relations-fp64",
        _MULTIPOINT_BINARY_RELATIONS_KERNEL_SOURCE,
        _MULTIPOINT_KERNEL_NAMES,
    )


def _is_device_array(value) -> bool:
    return hasattr(value, "__cuda_array_interface__")


def _identity_rows(count: int, *, device: bool):
    if device:
        import cupy as cp

        return cp.arange(count, dtype=cp.int32)
    return np.arange(count, dtype=np.int32)


def _false_like_bool(reference):
    if _is_device_array(reference):
        import cupy as cp

        return cp.zeros(reference.shape[0], dtype=cp.bool_)
    return np.zeros(reference.shape[0], dtype=bool)


def _device_scalar_bool(value, *, reason: str) -> bool:
    import cupy as cp

    runtime = get_cuda_runtime()
    host = runtime.copy_device_to_host(
        cp.asarray(value, dtype=cp.bool_).reshape(1),
        reason=reason,
    )
    return bool(np.asarray(host).reshape(-1)[0])


def _point_relation_to_predicate_array(
    predicate: str,
    relation,
    *,
    point_on_left: bool,
):
    outside = relation == POINT_LOCATION_OUTSIDE
    boundary = relation == POINT_LOCATION_BOUNDARY
    interior = relation == POINT_LOCATION_INTERIOR
    if predicate == "intersects":
        return ~outside
    if predicate == "disjoint":
        return outside
    if predicate == "touches":
        return boundary
    if predicate in {"crosses", "overlaps"}:
        return _false_like_bool(relation)
    if point_on_left:
        if predicate == "within":
            return interior
        if predicate == "covered_by":
            return ~outside
        return _false_like_bool(relation)
    if predicate == "contains":
        return interior
    if predicate == "covers":
        return ~outside
    if predicate == "contains_properly":
        return interior
    return _false_like_bool(relation)


def _point_equals_to_predicate_array(predicate: str, relation):
    equal = relation == POINT_LOCATION_INTERIOR
    if predicate in {
        "intersects",
        "contains",
        "within",
        "covers",
        "covered_by",
        "contains_properly",
        "equals",
    }:
        return equal
    if predicate == "disjoint":
        return ~equal
    return _false_like_bool(relation)


# ---------------------------------------------------------------------------
# Unified kernel launch -- replaces the three nearly-identical functions
# _launch_rows_kernel, _launch_indexed_kernel, _launch_indexed_mp_kernel.
# ---------------------------------------------------------------------------


class _PointRegionLaunchProfile:
    """Bind one prepared index to the active profiling session."""

    def __init__(self, profile, prepared) -> None:
        self._profile = profile
        self._prepared = prepared

    def begin_launch(self, *, logical_count, candidate_count: int) -> None:
        self._profile.begin_launch(
            self._prepared,
            logical_count=logical_count,
            candidate_count=candidate_count,
        )

    def end_launch(self) -> None:
        self._profile.end_launch(self._prepared)


def _launch_kernel(
    kernel_dict_fn,
    kernel_name: str,
    candidate_rows: np.ndarray,
    args: tuple[int, ...],
    arg_types: tuple[object, ...],
    *,
    extra_device_allocs: list | None = None,
    return_device: bool = False,
    logical_count=None,
    precision_plan: PrecisionPlan | None = None,
    candidate_rows_right=None,
    source_offset=None,
    launch_capacity: int | None = None,
    device_out=None,
    launch_profile=None,
) -> np.ndarray:
    """Launch a point or multipoint binary-relation kernel.

    Parameters
    ----------
    kernel_dict_fn : callable
        One of ``_point_binary_relation_kernels`` or ``_multipoint_relation_kernels``.
    kernel_name : str
        Name of the CUDA kernel to launch.
    candidate_rows : np.ndarray
        Row indices (int32) to pass as the first kernel argument.
    args : tuple
        Device pointer / scalar arguments between candidate_rows and (out, count).
    arg_types : tuple
        KERNEL_PARAM_* type tags matching *args*.
    extra_device_allocs : list or None
        Additional device allocations to free after launch (e.g. uploaded
        mapped FRO arrays).  With ``return_device=True``, ownership of
        ``device_out`` transfers to the caller as a CuPy array.
    """
    if precision_plan is not None:
        _require_indexed_point_precision_plan(precision_plan)
    n_items = int(candidate_rows.size)
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    returning_device = False
    device_rows_temp = None
    device_rows_right_temp = None
    if _is_device_array(candidate_rows):
        import cupy as cp

        device_rows = cp.asarray(candidate_rows)
        if device_rows.dtype != cp.int32:
            device_rows = device_rows.astype(cp.int32, copy=False)
            device_rows_temp = device_rows
    else:
        device_rows = runtime.from_host(candidate_rows.astype(np.int32, copy=False))
        device_rows_temp = device_rows
    if candidate_rows_right is None:
        device_rows_right = device_rows
    elif _is_device_array(candidate_rows_right):
        import cupy as cp

        device_rows_right = cp.asarray(candidate_rows_right)
        if device_rows_right.dtype != cp.int32:
            device_rows_right = device_rows_right.astype(cp.int32, copy=False)
            device_rows_right_temp = device_rows_right
    else:
        device_rows_right = runtime.from_host(
            candidate_rows_right.astype(np.int32, copy=False)
        )
        device_rows_right_temp = device_rows_right
    own_device_out = device_out is None
    if own_device_out:
        device_out = runtime.allocate((n_items,), np.uint8)
    try:
        kernel = kernel_dict_fn()[kernel_name]
        params = (
            (
                ptr(device_rows),
                ptr(device_rows_right),
                *args,
                ptr(device_out),
                ptr(source_offset),
                ptr(logical_count),
                n_items,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                *arg_types,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        launch_items = n_items if launch_capacity is None else int(launch_capacity)
        grid, block = runtime.launch_config(kernel, launch_items)
        if launch_profile is not None:
            launch_profile.begin_launch(
                logical_count=logical_count,
                candidate_count=n_items,
            )
        runtime.launch(kernel, grid=grid, block=block, params=params)
        if launch_profile is not None:
            launch_profile.end_launch()
        if return_device:
            returning_device = True
            return device_out
        runtime.synchronize()
        out = np.empty(n_items, dtype=np.uint8)
        runtime.copy_device_to_host(
            device_out,
            out,
            reason=f"point relation {kernel_name} result host export",
        )
        return out
    finally:
        runtime.free(device_rows_temp)
        runtime.free(device_rows_right_temp)
        if own_device_out and not returning_device:
            runtime.free(device_out)
        if extra_device_allocs:
            for alloc in extra_device_allocs:
                runtime.free(alloc)


# ---------------------------------------------------------------------------
# Non-indexed public API -- use candidate_rows and the owned array's
# device-side family_row_offsets directly.
# ---------------------------------------------------------------------------


def classify_point_equals_gpu(
    candidate_rows: np.ndarray,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    return_device: bool = False,
) -> np.ndarray:
    if candidate_rows.size == 0:
        if return_device:
            import cupy as cp

            return cp.empty(0, dtype=cp.uint8)
        return np.empty(0, dtype=np.uint8)
    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    left_buffer = left_state.families[GeometryFamily.POINT]
    right_buffer = right_state.families[GeometryFamily.POINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    return _launch_kernel(
        _point_binary_relation_kernels,
        "point_equals_compacted",
        candidate_rows,
        (
            ptr(left_state.family_row_offsets),
            ptr(left_buffer.geometry_offsets),
            ptr(left_buffer.empty_mask),
            ptr(left_buffer.x),
            ptr(left_buffer.y),
            ptr(right_state.family_row_offsets),
            ptr(right_buffer.geometry_offsets),
            ptr(right_buffer.empty_mask),
            ptr(right_buffer.x),
            ptr(right_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
        return_device=return_device,
    )


def classify_point_line_gpu(
    candidate_rows: np.ndarray,
    points: OwnedGeometryArray,
    lines: OwnedGeometryArray,
    *,
    line_family: GeometryFamily,
    return_device: bool = False,
) -> np.ndarray:
    if candidate_rows.size == 0:
        if return_device:
            import cupy as cp

            return cp.empty(0, dtype=cp.uint8)
        return np.empty(0, dtype=np.uint8)
    point_state = points._ensure_device_state(preserve_indexed_view=True)
    line_state = lines._ensure_device_state(preserve_indexed_view=True)
    point_buffer = point_state.families[GeometryFamily.POINT]
    line_buffer = line_state.families[line_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernel_name = (
        "point_on_linestring_compacted"
        if line_family is GeometryFamily.LINESTRING
        else "point_on_multilinestring_compacted"
    )
    args = [
        ptr(point_state.family_row_offsets),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(line_state.family_row_offsets),
        ptr(line_buffer.geometry_offsets),
    ]
    if line_family is not GeometryFamily.LINESTRING:
        args.append(ptr(line_buffer.part_offsets))
    args.extend(
        [
            ptr(line_buffer.empty_mask),
            ptr(line_buffer.x),
            ptr(line_buffer.y),
        ]
    )
    return _launch_kernel(
        _point_binary_relation_kernels,
        kernel_name,
        candidate_rows,
        tuple(args),
        (KERNEL_PARAM_PTR,) * len(args),
        return_device=return_device,
    )


def classify_point_region_gpu(
    candidate_rows: np.ndarray,
    points: OwnedGeometryArray,
    regions: OwnedGeometryArray,
    *,
    region_family: GeometryFamily,
    return_device: bool = False,
) -> np.ndarray:
    if candidate_rows.size == 0:
        if return_device:
            import cupy as cp

            return cp.empty(0, dtype=cp.uint8)
        return np.empty(0, dtype=np.uint8)
    point_state = points._ensure_device_state(preserve_indexed_view=True)
    region_state = regions._ensure_device_state(preserve_indexed_view=True)
    point_buffer = point_state.families[GeometryFamily.POINT]
    region_buffer = region_state.families[region_family]
    prepared = region_state.point_location_indexes.get(region_family)
    launch_profile = None
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernel_name = (
        "point_in_polygon_polygon_compacted_state"
        if region_family is GeometryFamily.POLYGON
        else "point_in_polygon_multipolygon_compacted_state"
    )
    kernel_dict_fn = _point_binary_relation_kernels
    args = [
        ptr(point_state.family_row_offsets),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(region_state.family_row_offsets),
        ptr(region_buffer.empty_mask),
        ptr(region_buffer.geometry_offsets),
    ]
    if region_family is not GeometryFamily.POLYGON:
        args.append(ptr(region_buffer.part_offsets))
    args.extend(
        [
            ptr(region_buffer.ring_offsets),
            ptr(region_buffer.x),
            ptr(region_buffer.y),
        ]
    )
    if prepared is not None:
        from .point_location_index import point_location_part_y_index_kernels
        from .point_region_profile import current_point_region_profile

        kernel_dict_fn = point_location_part_y_index_kernels
        kernel_name = (
            "point_in_polygon_prepared_part_y_index"
            if region_family is GeometryFamily.POLYGON
            else "point_in_multipolygon_prepared_part_y_index"
        )
        args = [
            ptr(point_state.family_row_offsets),
            ptr(point_buffer.geometry_offsets),
            ptr(point_buffer.empty_mask),
            ptr(point_buffer.x),
            ptr(point_buffer.y),
            ptr(region_state.family_row_offsets),
            ptr(region_buffer.empty_mask),
        ]
        if region_family is GeometryFamily.MULTIPOLYGON:
            args.append(ptr(region_buffer.geometry_offsets))
        args.extend(
            [
                ptr(region_buffer.ring_offsets),
                ptr(region_buffer.x),
                ptr(region_buffer.y),
                ptr(prepared.part_ymin),
                ptr(prepared.part_ymax),
                ptr(prepared.counts),
                ptr(prepared.offsets),
                ptr(prepared.entries),
            ]
        )
        profile = current_point_region_profile()
        if profile is not None:
            from .point_location_index import (
                point_location_part_y_index_profile_kernels,
            )

            summary, parts_histogram, edges_histogram, sample_plan = (
                profile.launch_arguments(prepared)
            )
            kernel_dict_fn = point_location_part_y_index_profile_kernels
            kernel_name += "_profiled"
            args.extend(
                [
                    ptr(summary),
                    ptr(parts_histogram),
                    ptr(edges_histogram),
                    ptr(sample_plan),
                ]
            )
            launch_profile = _PointRegionLaunchProfile(profile, prepared)
    return _launch_kernel(
        kernel_dict_fn,
        kernel_name,
        candidate_rows,
        tuple(args),
        (KERNEL_PARAM_PTR,) * len(args),
        return_device=return_device,
        launch_profile=launch_profile,
    )


# ---------------------------------------------------------------------------
# Indexed variants: separate left/right index arrays into original owned
# geometry arrays.  Avoids the expensive take() buffer copy by pre-gathering
# family_row_offsets on host and uploading the mapped arrays.
# ---------------------------------------------------------------------------

_POINT_TAG_INDEXED = FAMILY_TAGS[GeometryFamily.POINT]
_LINE_FAMILIES_INDEXED = (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
_REGION_FAMILIES_INDEXED = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
_LINE_TAGS_INDEXED = tuple(FAMILY_TAGS[f] for f in _LINE_FAMILIES_INDEXED)
_REGION_TAGS_INDEXED = tuple(FAMILY_TAGS[f] for f in _REGION_FAMILIES_INDEXED)


def _plan_indexed_point_precision(
    requested: PrecisionMode | str = PrecisionMode.AUTO,
    *,
    runtime_selection: RuntimeSelection | None = None,
) -> PrecisionPlan:
    """Resolve precision for adaptive exact indexed point predicates."""
    requested_mode = normalize_precision_mode(requested)
    selection = runtime_selection or RuntimeSelection(
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
        reason="indexed point-family predicates preserve authoritative fp64 results",
    )
    precision_plan = select_precision_plan(
        runtime_selection=selection,
        kernel_class=KernelClass.PREDICATE,
        requested=(
            PrecisionMode.FP64
            if requested_mode is PrecisionMode.AUTO
            else requested_mode
        ),
    )
    if requested_mode is PrecisionMode.AUTO:
        precision_plan = replace(
            precision_plan,
            reason=(
                "auto precision resolved to adaptive native fp64 after measured "
                "the exact point-in-polygon refinement kernel is implemented in fp64"
            ),
        )
    return _require_indexed_point_precision_plan(precision_plan)


def _default_indexed_point_precision_plan() -> PrecisionPlan:
    """Return the AUTO plan for the indexed point predicate shape."""
    return _plan_indexed_point_precision(PrecisionMode.AUTO)


def _require_indexed_point_precision_plan(precision_plan: PrecisionPlan) -> PrecisionPlan:
    """Validate an authoritative indexed point-predicate precision variant."""
    if not isinstance(precision_plan, PrecisionPlan):
        raise TypeError("indexed point predicates require a PrecisionPlan")
    if precision_plan.kernel_class is not KernelClass.PREDICATE:
        raise ValueError("indexed point predicates require a PREDICATE PrecisionPlan")
    if precision_plan.storage_precision is not PrecisionMode.FP64:
        raise NotImplementedError("indexed point predicates require fp64 storage")
    if precision_plan.compute_precision is PrecisionMode.FP64 and (
        precision_plan.compensation is not CompensationMode.NONE
        or precision_plan.refinement is not RefinementMode.NONE
        or precision_plan.center_coordinates
    ):
        raise ValueError(
            "indexed point predicate fp64 plans must be uncentered and require "
            "neither compensation nor refinement"
        )
    if precision_plan.compute_precision is not PrecisionMode.FP64:
        raise NotImplementedError(
            "indexed point predicates require adaptive authoritative fp64; "
            "the measured interval-fp32 variant is not admitted"
        )
    return precision_plan


def _resolve_indexed_point_precision_plan(
    precision_plan: PrecisionPlan | None,
) -> PrecisionPlan:
    if precision_plan is None:
        precision_plan = _default_indexed_point_precision_plan()
    return _require_indexed_point_precision_plan(precision_plan)


def _prepare_indexed_fro(owned, indices, runtime):
    """Map indices through family_row_offsets and return a device array."""
    if _is_device_array(indices):
        import cupy as cp

        state = owned._ensure_device_state(preserve_indexed_view=True)
        return state.family_row_offsets[indices].astype(cp.int32, copy=False)
    mapped = owned.family_row_offsets[indices].astype(np.int32, copy=False)
    return runtime.from_host(mapped)


def _prepare_indexed_pair_launch(
    left_owned,
    right_owned,
    left_indices,
    right_indices,
    runtime,
    *,
    source_offset,
):
    """Choose compacted-row or shared grouped-span kernel inputs."""
    if source_offset is None:
        left_fro = _prepare_indexed_fro(left_owned, left_indices, runtime)
        right_fro = _prepare_indexed_fro(right_owned, right_indices, runtime)
        identity_rows = _identity_rows(int(left_indices.size), device=_is_device_array(left_indices))
        return left_fro, right_fro, identity_rows, identity_rows, [left_fro, right_fro]
    left_state = left_owned._ensure_device_state(preserve_indexed_view=True)
    right_state = right_owned._ensure_device_state(preserve_indexed_view=True)
    return (
        left_state.family_row_offsets,
        right_state.family_row_offsets,
        left_indices,
        right_indices,
        [],
    )


def _classify_indexed_point_equals(
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    *,
    precision_plan: PrecisionPlan,
    return_device: bool = False,
    logical_count=None,
    source_offset=None,
    launch_capacity: int | None = None,
    relation_out=None,
) -> np.ndarray:
    n = int(left_indices.size)
    if n == 0:
        if return_device:
            import cupy as cp

            return cp.empty(0, dtype=cp.uint8)
        return np.empty(0, dtype=np.uint8)
    left_state = left_owned._ensure_device_state(preserve_indexed_view=True)
    right_state = right_owned._ensure_device_state(preserve_indexed_view=True)
    left_buffer = left_state.families[GeometryFamily.POINT]
    right_buffer = right_state.families[GeometryFamily.POINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_left_fro, device_right_fro, left_rows, right_rows, temporaries = (
        _prepare_indexed_pair_launch(
            left_owned,
            right_owned,
            left_indices,
            right_indices,
            runtime,
            source_offset=source_offset,
        )
    )
    return _launch_kernel(
        _point_binary_relation_kernels,
        "point_equals_compacted",
        left_rows,
        (
            ptr(device_left_fro),
            ptr(left_buffer.geometry_offsets),
            ptr(left_buffer.empty_mask),
            ptr(left_buffer.x),
            ptr(left_buffer.y),
            ptr(device_right_fro),
            ptr(right_buffer.geometry_offsets),
            ptr(right_buffer.empty_mask),
            ptr(right_buffer.x),
            ptr(right_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
        extra_device_allocs=temporaries,
        return_device=return_device,
        logical_count=logical_count,
        precision_plan=precision_plan,
        candidate_rows_right=right_rows,
        source_offset=source_offset,
        launch_capacity=launch_capacity,
        device_out=relation_out,
    )


def _classify_indexed_point_line(
    point_owned: OwnedGeometryArray,
    line_owned: OwnedGeometryArray,
    point_indices: np.ndarray,
    line_indices: np.ndarray,
    *,
    line_family: GeometryFamily,
    precision_plan: PrecisionPlan,
    return_device: bool = False,
    logical_count=None,
    source_offset=None,
    launch_capacity: int | None = None,
    relation_out=None,
) -> np.ndarray:
    n = int(point_indices.size)
    if n == 0:
        if return_device:
            import cupy as cp

            return cp.empty(0, dtype=cp.uint8)
        return np.empty(0, dtype=np.uint8)
    point_state = point_owned._ensure_device_state(preserve_indexed_view=True)
    line_state = line_owned._ensure_device_state(preserve_indexed_view=True)
    point_buffer = point_state.families[GeometryFamily.POINT]
    line_buffer = line_state.families[line_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_point_fro, device_line_fro, point_rows, line_rows, temporaries = (
        _prepare_indexed_pair_launch(
            point_owned,
            line_owned,
            point_indices,
            line_indices,
            runtime,
            source_offset=source_offset,
        )
    )
    kernel_name = (
        "point_on_linestring_compacted"
        if line_family is GeometryFamily.LINESTRING
        else "point_on_multilinestring_compacted"
    )
    args = [
        ptr(device_point_fro),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(device_line_fro),
        ptr(line_buffer.geometry_offsets),
    ]
    if line_family is not GeometryFamily.LINESTRING:
        args.append(ptr(line_buffer.part_offsets))
    args.extend(
        [
            ptr(line_buffer.empty_mask),
            ptr(line_buffer.x),
            ptr(line_buffer.y),
        ]
    )
    return _launch_kernel(
        _point_binary_relation_kernels,
        kernel_name,
        point_rows,
        tuple(args),
        (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=temporaries,
        return_device=return_device,
        logical_count=logical_count,
        precision_plan=precision_plan,
        candidate_rows_right=line_rows,
        source_offset=source_offset,
        launch_capacity=launch_capacity,
        device_out=relation_out,
    )


def _classify_indexed_point_region(
    point_owned: OwnedGeometryArray,
    region_owned: OwnedGeometryArray,
    point_indices: np.ndarray,
    region_indices: np.ndarray,
    *,
    region_family: GeometryFamily,
    precision_plan: PrecisionPlan,
    return_device: bool = False,
    logical_count=None,
    source_offset=None,
    launch_capacity: int | None = None,
    relation_out=None,
) -> np.ndarray:
    n = int(point_indices.size)
    if n == 0:
        if return_device:
            import cupy as cp

            return cp.empty(0, dtype=cp.uint8)
        return np.empty(0, dtype=np.uint8)
    point_state = point_owned._ensure_device_state(preserve_indexed_view=True)
    region_state = region_owned._ensure_device_state(preserve_indexed_view=True)
    point_buffer = point_state.families[GeometryFamily.POINT]
    region_buffer = region_state.families[region_family]
    prepared = region_state.point_location_indexes.get(region_family)
    launch_profile = None
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_point_fro, device_region_fro, point_rows, region_rows, temporaries = (
        _prepare_indexed_pair_launch(
            point_owned,
            region_owned,
            point_indices,
            region_indices,
            runtime,
            source_offset=source_offset,
        )
    )
    kernel_name = (
        "point_in_polygon_polygon_compacted_state"
        if region_family is GeometryFamily.POLYGON
        else "point_in_polygon_multipolygon_compacted_state"
    )
    kernel_dict_fn = _point_binary_relation_kernels
    args = [
        ptr(device_point_fro),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(device_region_fro),
        ptr(region_buffer.empty_mask),
        ptr(region_buffer.geometry_offsets),
    ]
    if region_family is not GeometryFamily.POLYGON:
        args.append(ptr(region_buffer.part_offsets))
    args.extend(
        [
            ptr(region_buffer.ring_offsets),
            ptr(region_buffer.x),
            ptr(region_buffer.y),
        ]
    )
    if prepared is not None:
        from .point_location_index import point_location_part_y_index_kernels
        from .point_region_profile import current_point_region_profile

        kernel_dict_fn = point_location_part_y_index_kernels
        kernel_name = (
            "point_in_polygon_prepared_part_y_index"
            if region_family is GeometryFamily.POLYGON
            else "point_in_multipolygon_prepared_part_y_index"
        )
        args = [
            ptr(device_point_fro),
            ptr(point_buffer.geometry_offsets),
            ptr(point_buffer.empty_mask),
            ptr(point_buffer.x),
            ptr(point_buffer.y),
            ptr(device_region_fro),
            ptr(region_buffer.empty_mask),
        ]
        if region_family is GeometryFamily.MULTIPOLYGON:
            args.append(ptr(region_buffer.geometry_offsets))
        args.extend(
            [
                ptr(region_buffer.ring_offsets),
                ptr(region_buffer.x),
                ptr(region_buffer.y),
                ptr(prepared.part_ymin),
                ptr(prepared.part_ymax),
                ptr(prepared.counts),
                ptr(prepared.offsets),
                ptr(prepared.entries),
            ]
        )
        profile = current_point_region_profile()
        if profile is not None:
            from .point_location_index import (
                point_location_part_y_index_profile_kernels,
            )

            summary, parts_histogram, edges_histogram, sample_plan = (
                profile.launch_arguments(prepared)
            )
            kernel_dict_fn = point_location_part_y_index_profile_kernels
            kernel_name += "_profiled"
            args.extend(
                [
                    ptr(summary),
                    ptr(parts_histogram),
                    ptr(edges_histogram),
                    ptr(sample_plan),
                ]
            )
            launch_profile = _PointRegionLaunchProfile(profile, prepared)
    return _launch_kernel(
        kernel_dict_fn,
        kernel_name,
        point_rows,
        tuple(args),
        (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=temporaries,
        return_device=return_device,
        logical_count=logical_count,
        precision_plan=precision_plan,
        candidate_rows_right=region_rows,
        source_offset=source_offset,
        launch_capacity=launch_capacity,
        device_out=relation_out,
        launch_profile=launch_profile,
    )


def classify_point_predicates_indexed(
    predicate: str,
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    *,
    precision_plan: PrecisionPlan | None = None,
) -> np.ndarray:
    """Evaluate point-family predicates using indexed access into original owned arrays.

    Avoids the expensive take() buffer copy by pre-gathering family_row_offsets
    on the host and passing them directly to existing GPU kernels.

    Returns a boolean array of length ``left_indices.size``.
    """
    from .binary import (
        _apply_relation_rows,
        _point_equals_to_predicate,
        _point_relation_to_predicate,
    )

    precision_plan = _resolve_indexed_point_precision_plan(precision_plan)
    n = left_indices.size
    if n == 0:
        return np.empty(0, dtype=bool)

    out = np.zeros(n, dtype=bool)
    left_tags = left_owned.tags[left_indices]
    right_tags = right_owned.tags[right_indices]

    # Point x point
    pp_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == _POINT_TAG_INDEXED)
    if pp_mask.any():
        idx = np.flatnonzero(pp_mask)
        relation = _classify_indexed_point_equals(
            left_owned,
            right_owned,
            left_indices[idx],
            right_indices[idx],
            precision_plan=precision_plan,
        )
        _apply_relation_rows(out, idx, _point_equals_to_predicate(predicate, relation))

    # Point x line and line x point
    for line_family, line_tag in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        pl_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == line_tag)
        if pl_mask.any():
            idx = np.flatnonzero(pl_mask)
            relation = _classify_indexed_point_line(
                left_owned,
                right_owned,
                left_indices[idx],
                right_indices[idx],
                line_family=line_family,
                precision_plan=precision_plan,
            )
            _apply_relation_rows(
                out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=True)
            )

        lp_mask = (left_tags == line_tag) & (right_tags == _POINT_TAG_INDEXED)
        if lp_mask.any():
            idx = np.flatnonzero(lp_mask)
            relation = _classify_indexed_point_line(
                right_owned,
                left_owned,
                right_indices[idx],
                left_indices[idx],
                line_family=line_family,
                precision_plan=precision_plan,
            )
            _apply_relation_rows(
                out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=False)
            )

    # Point x region and region x point
    for region_family, region_tag in zip(
        _REGION_FAMILIES_INDEXED, _REGION_TAGS_INDEXED, strict=True
    ):
        pr_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == region_tag)
        if pr_mask.any():
            idx = np.flatnonzero(pr_mask)
            relation = _classify_indexed_point_region(
                left_owned,
                right_owned,
                left_indices[idx],
                right_indices[idx],
                region_family=region_family,
                precision_plan=precision_plan,
            )
            _apply_relation_rows(
                out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=True)
            )

        rp_mask = (region_tag == left_tags) & (right_tags == _POINT_TAG_INDEXED)
        if rp_mask.any():
            idx = np.flatnonzero(rp_mask)
            relation = _classify_indexed_point_region(
                right_owned,
                left_owned,
                right_indices[idx],
                left_indices[idx],
                region_family=region_family,
                precision_plan=precision_plan,
            )
            _apply_relation_rows(
                out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=False)
            )

    # Multipoint x anything and anything x multipoint
    mp_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
    mp_left_mask = left_tags == mp_tag
    mp_right_mask = right_tags == mp_tag

    if mp_left_mask.any() or mp_right_mask.any():
        _dispatch_multipoint_pairs(
            predicate,
            out,
            left_owned,
            right_owned,
            left_indices,
            right_indices,
            left_tags,
            right_tags,
            mp_left_mask,
            mp_right_mask,
            _apply_relation_rows,
            precision_plan,
        )

    return out


def classify_point_predicates_indexed_device(
    predicate: str,
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices,
    right_indices,
    *,
    left_tags=None,
    right_tags=None,
    precision_plan: PrecisionPlan | None = None,
):
    """Evaluate point-family indexed predicates on device.

    This is the device-resident companion to
    :func:`classify_point_predicates_indexed`.  It covers point, multipoint,
    lineal, and polygonal relation-pair rows without exporting branch masks to
    host.
    """
    precision_plan = _resolve_indexed_point_precision_plan(precision_plan)

    import cupy as cp

    left_indices = cp.asarray(left_indices, dtype=cp.int32)
    right_indices = cp.asarray(right_indices, dtype=cp.int32)
    n = int(left_indices.size)
    if n == 0:
        return cp.empty(0, dtype=cp.bool_)

    left_state = left_owned._ensure_device_state(preserve_indexed_view=True)
    right_state = right_owned._ensure_device_state(preserve_indexed_view=True)
    left_tags = (
        cp.asarray(left_tags, dtype=cp.int8)
        if left_tags is not None
        else left_state.tags[left_indices]
    )
    right_tags = (
        cp.asarray(right_tags, dtype=cp.int8)
        if right_tags is not None
        else right_state.tags[right_indices]
    )

    out = cp.zeros(n, dtype=cp.bool_)

    if GeometryFamily.POINT in left_state.families and GeometryFamily.POINT in right_state.families:
        pp_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == _POINT_TAG_INDEXED)
        idx = cp.flatnonzero(pp_mask).astype(cp.int32, copy=False)
        relation = _classify_indexed_point_equals(
            left_owned,
            right_owned,
            left_indices[idx],
            right_indices[idx],
            precision_plan=precision_plan,
            return_device=True,
        )
        out[idx] = _point_equals_to_predicate_array(predicate, relation)

    for line_family, line_tag in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        if GeometryFamily.POINT in left_state.families and line_family in right_state.families:
            pl_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == line_tag)
            idx = cp.flatnonzero(pl_mask).astype(cp.int32, copy=False)
            relation = _classify_indexed_point_line(
                left_owned,
                right_owned,
                left_indices[idx],
                right_indices[idx],
                line_family=line_family,
                precision_plan=precision_plan,
                return_device=True,
            )
            out[idx] = _point_relation_to_predicate_array(
                predicate,
                relation,
                point_on_left=True,
            )

        if line_family in left_state.families and GeometryFamily.POINT in right_state.families:
            lp_mask = (left_tags == line_tag) & (right_tags == _POINT_TAG_INDEXED)
            idx = cp.flatnonzero(lp_mask).astype(cp.int32, copy=False)
            relation = _classify_indexed_point_line(
                right_owned,
                left_owned,
                right_indices[idx],
                left_indices[idx],
                line_family=line_family,
                precision_plan=precision_plan,
                return_device=True,
            )
            out[idx] = _point_relation_to_predicate_array(
                predicate,
                relation,
                point_on_left=False,
            )

    for region_family, region_tag in zip(
        _REGION_FAMILIES_INDEXED, _REGION_TAGS_INDEXED, strict=True
    ):
        if GeometryFamily.POINT in left_state.families and region_family in right_state.families:
            pr_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == region_tag)
            idx = cp.flatnonzero(pr_mask).astype(cp.int32, copy=False)
            relation = _classify_indexed_point_region(
                left_owned,
                right_owned,
                left_indices[idx],
                right_indices[idx],
                region_family=region_family,
                precision_plan=precision_plan,
                return_device=True,
            )
            out[idx] = _point_relation_to_predicate_array(
                predicate,
                relation,
                point_on_left=True,
            )

        if region_family in left_state.families and GeometryFamily.POINT in right_state.families:
            rp_mask = (left_tags == region_tag) & (right_tags == _POINT_TAG_INDEXED)
            idx = cp.flatnonzero(rp_mask).astype(cp.int32, copy=False)
            relation = _classify_indexed_point_region(
                right_owned,
                left_owned,
                right_indices[idx],
                left_indices[idx],
                region_family=region_family,
                precision_plan=precision_plan,
                return_device=True,
            )
            out[idx] = _point_relation_to_predicate_array(
                predicate,
                relation,
                point_on_left=False,
            )

    mp_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
    if (
        GeometryFamily.MULTIPOINT in left_state.families
        or GeometryFamily.MULTIPOINT in right_state.families
    ):
        _dispatch_multipoint_pairs_device(
            predicate,
            out,
            left_owned,
            right_owned,
            left_indices,
            right_indices,
            left_tags,
            right_tags,
            left_tags == mp_tag,
            right_tags == mp_tag,
            precision_plan,
        )

    return out


def classify_homogeneous_point_predicates_indexed_device(
    predicate: str,
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices,
    right_indices,
    *,
    left_family: GeometryFamily,
    right_family: GeometryFamily,
    precision_plan: PrecisionPlan,
    logical_count=None,
    source_offset=None,
    launch_capacity: int | None = None,
    predicate_out=None,
    relation_out=None,
):
    """Evaluate one point-family pair without relation-row compaction."""
    precision_plan = _require_indexed_point_precision_plan(precision_plan)

    import cupy as cp

    left_indices = cp.asarray(left_indices, dtype=cp.int32)
    right_indices = cp.asarray(right_indices, dtype=cp.int32)
    if left_indices.size != right_indices.size:
        raise ValueError("homogeneous point predicate indices must be aligned")
    grouped = source_offset is not None
    if grouped and (launch_capacity is None or predicate_out is None or relation_out is None):
        raise ValueError("grouped point classification requires launch and output storage")
    grouped_kernel_args = {
        "source_offset": source_offset,
        "launch_capacity": launch_capacity,
        "relation_out": relation_out,
    }

    def finish(relation, mode: int, *, target_family=None):
        if grouped:
            return _evaluate_point_relation_grouped(
                relation,
                predicate_out,
                source_offset=source_offset,
                logical_count=logical_count,
                launch_capacity=launch_capacity,
                predicate=predicate,
                relation_mode=mode,
                target_pointlike=target_family
                in {GeometryFamily.POINT, GeometryFamily.MULTIPOINT},
            )
        if mode == 0:
            return _point_equals_to_predicate_array(predicate, relation)
        if mode in {1, 2}:
            return _point_relation_to_predicate_array(
                predicate,
                relation,
                point_on_left=mode == 1,
            )
        return _multipoint_bits_to_predicate(
            predicate,
            relation,
            mp_on_left=mode == 3,
            target_family=target_family,
        )

    if left_family is GeometryFamily.POINT:
        if right_family is GeometryFamily.POINT:
            relation = _classify_indexed_point_equals(
                left_owned,
                right_owned,
                left_indices,
                right_indices,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
            return finish(relation, 0)
        if right_family in _LINE_FAMILIES_INDEXED:
            relation = _classify_indexed_point_line(
                left_owned,
                right_owned,
                left_indices,
                right_indices,
                line_family=right_family,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
            return finish(relation, 1)
        if right_family in _REGION_FAMILIES_INDEXED:
            relation = _classify_indexed_point_region(
                left_owned,
                right_owned,
                left_indices,
                right_indices,
                region_family=right_family,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
            return finish(relation, 1)
        if right_family is GeometryFamily.MULTIPOINT:
            bits = _classify_indexed_mp_point(
                right_owned,
                left_owned,
                right_indices,
                left_indices,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
            return finish(bits, 4, target_family=GeometryFamily.POINT)

    if right_family is GeometryFamily.POINT:
        if left_family in _LINE_FAMILIES_INDEXED:
            relation = _classify_indexed_point_line(
                right_owned,
                left_owned,
                right_indices,
                left_indices,
                line_family=left_family,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
            return finish(relation, 2)
        if left_family in _REGION_FAMILIES_INDEXED:
            relation = _classify_indexed_point_region(
                right_owned,
                left_owned,
                right_indices,
                left_indices,
                region_family=left_family,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
            return finish(relation, 2)
        if left_family is GeometryFamily.MULTIPOINT:
            bits = _classify_indexed_mp_point(
                left_owned,
                right_owned,
                left_indices,
                right_indices,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
            return finish(bits, 3, target_family=GeometryFamily.POINT)

    if left_family is GeometryFamily.MULTIPOINT:
        if right_family in _LINE_FAMILIES_INDEXED:
            bits = _classify_indexed_mp_line(
                left_owned,
                right_owned,
                left_indices,
                right_indices,
                line_family=right_family,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
        elif right_family in _REGION_FAMILIES_INDEXED:
            bits = _classify_indexed_mp_region(
                left_owned,
                right_owned,
                left_indices,
                right_indices,
                region_family=right_family,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
        elif right_family is GeometryFamily.MULTIPOINT:
            bits = _classify_indexed_mp_mp(
                left_owned,
                right_owned,
                left_indices,
                right_indices,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
            if predicate in {"contains", "covers", "contains_properly"}:
                bits = _classify_indexed_mp_mp(
                    right_owned,
                    left_owned,
                    right_indices,
                    left_indices,
                    precision_plan=precision_plan,
                    return_device=True,
                    logical_count=logical_count,
                    **grouped_kernel_args,
                )
                return finish(
                    bits,
                    4,
                    target_family=GeometryFamily.MULTIPOINT,
                )
        else:
            bits = None
        if bits is not None:
            return finish(bits, 3, target_family=right_family)

    if right_family is GeometryFamily.MULTIPOINT:
        if left_family in _LINE_FAMILIES_INDEXED:
            bits = _classify_indexed_mp_line(
                right_owned,
                left_owned,
                right_indices,
                left_indices,
                line_family=left_family,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
        elif left_family in _REGION_FAMILIES_INDEXED:
            bits = _classify_indexed_mp_region(
                right_owned,
                left_owned,
                right_indices,
                left_indices,
                region_family=left_family,
                precision_plan=precision_plan,
                return_device=True,
                logical_count=logical_count,
                **grouped_kernel_args,
            )
        else:
            bits = None
        if bits is not None:
            return finish(bits, 4, target_family=left_family)

    raise ValueError(
        "homogeneous point predicate requires at least one point-family input"
    )


# ---------------------------------------------------------------------------
# Multipoint support -- launch helpers, predicate conversion, and dispatch.
# Tier 1 per ADR-0033: geometry-specific inner loops (multipoint coord iteration).
# ---------------------------------------------------------------------------

# Bit flags in multipoint kernel output
_MP_ANY_OUTSIDE = np.uint8(1)
_MP_ANY_BOUNDARY = np.uint8(2)
_MP_ANY_INTERIOR = np.uint8(4)

_POINT_GROUPED_EVAL_KERNEL_SOURCE = r"""
extern "C" __global__ void evaluate_point_relation_grouped(
    const unsigned char* relation,
    unsigned char* out,
    const long long* source_offset,
    const int* logical_count,
    int predicate_code,
    int relation_mode,
    int target_pointlike
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = (int)source_offset[0];
    const int count = logical_count[0];
    for (int local_index = lane; local_index < count; local_index += stride) {
        const int index = offset + local_index;
        const unsigned char value = relation[index];
        const bool outside = value == 0u;
        const bool boundary = value == 1u;
        const bool interior = relation_mode < 3 ? value == 2u : (value & 4u) != 0u;
        const bool any_outside = relation_mode < 3 ? outside : (value & 1u) != 0u;
        const bool any_boundary = relation_mode < 3 ? boundary : (value & 2u) != 0u;
        const bool hit = any_boundary || interior;
        bool result = false;
        if (relation_mode == 0) {
            const bool equal = interior;
            result = predicate_code == 7 ? !equal :
                (predicate_code == 0 || predicate_code == 2 || predicate_code == 3 ||
                 predicate_code == 4 || predicate_code == 5 || predicate_code == 8 ||
                 predicate_code == 9) && equal;
        } else if (relation_mode == 1) {
            if (predicate_code == 0) result = !outside;
            else if (predicate_code == 7) result = outside;
            else if (predicate_code == 1) result = boundary;
            else if (predicate_code == 5) result = interior;
            else if (predicate_code == 3) result = !outside;
        } else if (relation_mode == 2) {
            if (predicate_code == 0) result = !outside;
            else if (predicate_code == 7) result = outside;
            else if (predicate_code == 1) result = boundary;
            else if (predicate_code == 4 || predicate_code == 8) result = interior;
            else if (predicate_code == 2) result = !outside;
        } else if (relation_mode == 3) {
            if (predicate_code == 0) result = hit;
            else if (predicate_code == 7) result = !hit;
            else if (predicate_code == 1) result = any_boundary && !interior;
            else if (predicate_code == 5) result = interior && !any_outside;
            else if (predicate_code == 3) result = hit && !any_outside;
            else if ((predicate_code == 4 || predicate_code == 2 || predicate_code == 8) && target_pointlike) result = interior;
        } else {
            if (predicate_code == 0) result = hit;
            else if (predicate_code == 7) result = !hit;
            else if (predicate_code == 1) result = any_boundary && !interior;
            else if (predicate_code == 4 || predicate_code == 8) result = interior && !any_outside;
            else if (predicate_code == 2) result = hit && !any_outside;
            else if ((predicate_code == 5 || predicate_code == 3) && target_pointlike) result = interior;
        }
        out[index] = result ? 1u : 0u;
    }
}
"""

_POINT_GROUPED_EVAL_KERNEL_NAMES = ("evaluate_point_relation_grouped",)

request_nvrtc_warmup(
    [
        (
            "point-relation-grouped-eval",
            _POINT_GROUPED_EVAL_KERNEL_SOURCE,
            _POINT_GROUPED_EVAL_KERNEL_NAMES,
        )
    ]
)


def _point_grouped_eval_kernels():
    return compile_kernel_group(
        "point-relation-grouped-eval",
        _POINT_GROUPED_EVAL_KERNEL_SOURCE,
        _POINT_GROUPED_EVAL_KERNEL_NAMES,
    )


_POINT_PREDICATE_CODES = {
    "intersects": 0,
    "touches": 1,
    "covers": 2,
    "covered_by": 3,
    "contains": 4,
    "within": 5,
    "overlaps": 6,
    "disjoint": 7,
    "contains_properly": 8,
    "equals": 9,
}


def _evaluate_point_relation_grouped(
    relation,
    out,
    *,
    source_offset,
    logical_count,
    launch_capacity: int,
    predicate: str,
    relation_mode: int,
    target_pointlike: bool = False,
):
    runtime = get_cuda_runtime()
    kernel = _point_grouped_eval_kernels()["evaluate_point_relation_grouped"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, int(launch_capacity))
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (
                ptr(relation),
                ptr(out),
                ptr(source_offset),
                ptr(logical_count),
                _POINT_PREDICATE_CODES.get(predicate, -1),
                int(relation_mode),
                int(target_pointlike),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    return out


def _multipoint_bits_to_predicate(
    predicate: str,
    bits: np.ndarray,
    *,
    mp_on_left: bool,
    target_family: GeometryFamily | None = None,
) -> np.ndarray:
    """Convert packed multipoint relation bits to boolean predicate results.

    Bits: 0x1 = any_outside, 0x2 = any_boundary, 0x4 = any_interior.
    Each bit records how the multipoint's coordinates relate to the target.

    **Key asymmetry:**  The bits tell us *"for each MP coord, its location in
    the target."*  This directly gives ``within`` / ``covered_by`` (all MP
    coords inside target) and the symmetric predicates (intersects / disjoint
    / touches).  For ``contains`` / ``covers`` / ``contains_properly`` we need
    the *reverse* -- whether the target fits inside the multipoint.

    * MP x Point ``contains``: at least one MP coord equals the point ->
      ``any_interior`` (the kernel records "equal" as interior).
    * MP x Line/Polygon ``contains``: always False (0-D cannot contain >=1-D).
    * MP x MP ``contains``: handled by the dispatch running the kernel in
      reverse and calling this function with a swapped predicate.
    """
    any_outside = (bits & _MP_ANY_OUTSIDE).astype(bool)
    any_boundary = (bits & _MP_ANY_BOUNDARY).astype(bool)
    any_interior = (bits & _MP_ANY_INTERIOR).astype(bool)
    any_hit = any_boundary | any_interior
    # --- Symmetric predicates ---
    if predicate == "intersects":
        return any_hit
    if predicate == "disjoint":
        return ~any_hit
    if predicate == "touches":
        return any_boundary & ~any_interior

    # --- within / covered_by: is the MP inside the target? ---
    # Condition: every MP coord must be inside (or on boundary of) the target.
    if mp_on_left:
        if predicate == "within":
            return any_interior & ~any_outside
        if predicate == "covered_by":
            return any_hit & ~any_outside

        # contains / covers / contains_properly: is the target inside the MP?
        if predicate in {"contains", "covers", "contains_properly"}:
            tf = target_family
            if tf is GeometryFamily.POINT or tf is GeometryFamily.MULTIPOINT:
                # MP contains point iff point matches at least one MP coord.
                # For MPxMP, the dispatch handles the reverse check.
                return any_interior
            # MP can't contain a line or polygon -- 0-D vs >=1-D.
            return _false_like_bool(bits)

    else:
        # MP is on the right (tree side), target on left.

        # contains / covers: does the target contain every MP coord?
        if predicate == "contains":
            return any_interior & ~any_outside
        if predicate == "covers":
            return any_hit & ~any_outside
        if predicate == "contains_properly":
            return any_interior & ~any_outside

        # within / covered_by: is the target within the MP?
        if predicate in {"within", "covered_by"}:
            tf = target_family
            if tf is GeometryFamily.POINT or tf is GeometryFamily.MULTIPOINT:
                return any_interior
            return _false_like_bool(bits)

    return _false_like_bool(bits)


# ---------------------------------------------------------------------------
# Indexed multipoint classify functions
# ---------------------------------------------------------------------------


def _classify_indexed_mp_point(
    mp_owned: OwnedGeometryArray,
    pt_owned: OwnedGeometryArray,
    mp_indices: np.ndarray,
    pt_indices: np.ndarray,
    *,
    precision_plan: PrecisionPlan,
    return_device: bool = False,
    logical_count=None,
    source_offset=None,
    launch_capacity: int | None = None,
    relation_out=None,
) -> np.ndarray:
    """MULTIPOINT x POINT relation bits."""
    n = int(mp_indices.size)
    if n == 0:
        if return_device:
            import cupy as cp

            return cp.empty(0, dtype=cp.uint8)
        return np.empty(0, dtype=np.uint8)
    mp_state = mp_owned._ensure_device_state(preserve_indexed_view=True)
    pt_state = pt_owned._ensure_device_state(preserve_indexed_view=True)
    mp_buffer = mp_state.families[GeometryFamily.MULTIPOINT]
    pt_buffer = pt_state.families[GeometryFamily.POINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_mp_fro, device_pt_fro, mp_rows, pt_rows, temporaries = (
        _prepare_indexed_pair_launch(
            mp_owned,
            pt_owned,
            mp_indices,
            pt_indices,
            runtime,
            source_offset=source_offset,
        )
    )
    return _launch_kernel(
        _multipoint_relation_kernels,
        "multipoint_point_relation_compacted",
        mp_rows,
        (
            ptr(device_mp_fro),
            ptr(mp_buffer.geometry_offsets),
            ptr(mp_buffer.empty_mask),
            ptr(mp_buffer.x),
            ptr(mp_buffer.y),
            ptr(device_pt_fro),
            ptr(pt_buffer.geometry_offsets),
            ptr(pt_buffer.empty_mask),
            ptr(pt_buffer.x),
            ptr(pt_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
        extra_device_allocs=temporaries,
        return_device=return_device,
        logical_count=logical_count,
        precision_plan=precision_plan,
        candidate_rows_right=pt_rows,
        source_offset=source_offset,
        launch_capacity=launch_capacity,
        device_out=relation_out,
    )


def _classify_indexed_mp_line(
    mp_owned: OwnedGeometryArray,
    line_owned: OwnedGeometryArray,
    mp_indices: np.ndarray,
    line_indices: np.ndarray,
    *,
    line_family: GeometryFamily,
    precision_plan: PrecisionPlan,
    return_device: bool = False,
    logical_count=None,
    source_offset=None,
    launch_capacity: int | None = None,
    relation_out=None,
) -> np.ndarray:
    """MULTIPOINT x LINESTRING/MULTILINESTRING relation bits."""
    n = int(mp_indices.size)
    if n == 0:
        if return_device:
            import cupy as cp

            return cp.empty(0, dtype=cp.uint8)
        return np.empty(0, dtype=np.uint8)
    mp_state = mp_owned._ensure_device_state(preserve_indexed_view=True)
    line_state = line_owned._ensure_device_state(preserve_indexed_view=True)
    mp_buffer = mp_state.families[GeometryFamily.MULTIPOINT]
    line_buffer = line_state.families[line_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_mp_fro, device_line_fro, mp_rows, line_rows, temporaries = (
        _prepare_indexed_pair_launch(
            mp_owned,
            line_owned,
            mp_indices,
            line_indices,
            runtime,
            source_offset=source_offset,
        )
    )
    kernel_name = (
        "multipoint_linestring_relation_compacted"
        if line_family is GeometryFamily.LINESTRING
        else "multipoint_multilinestring_relation_compacted"
    )
    args = [
        ptr(device_mp_fro),
        ptr(mp_buffer.geometry_offsets),
        ptr(mp_buffer.empty_mask),
        ptr(mp_buffer.x),
        ptr(mp_buffer.y),
        ptr(device_line_fro),
        ptr(line_buffer.geometry_offsets),
    ]
    if line_family is not GeometryFamily.LINESTRING:
        args.append(ptr(line_buffer.part_offsets))
    args.extend(
        [
            ptr(line_buffer.empty_mask),
            ptr(line_buffer.x),
            ptr(line_buffer.y),
        ]
    )
    return _launch_kernel(
        _multipoint_relation_kernels,
        kernel_name,
        mp_rows,
        tuple(args),
        (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=temporaries,
        return_device=return_device,
        logical_count=logical_count,
        precision_plan=precision_plan,
        candidate_rows_right=line_rows,
        source_offset=source_offset,
        launch_capacity=launch_capacity,
        device_out=relation_out,
    )


def _classify_indexed_mp_region(
    mp_owned: OwnedGeometryArray,
    region_owned: OwnedGeometryArray,
    mp_indices: np.ndarray,
    region_indices: np.ndarray,
    *,
    region_family: GeometryFamily,
    precision_plan: PrecisionPlan,
    return_device: bool = False,
    logical_count=None,
    source_offset=None,
    launch_capacity: int | None = None,
    relation_out=None,
) -> np.ndarray:
    """MULTIPOINT x POLYGON/MULTIPOLYGON relation bits."""
    n = int(mp_indices.size)
    if n == 0:
        if return_device:
            import cupy as cp

            return cp.empty(0, dtype=cp.uint8)
        return np.empty(0, dtype=np.uint8)
    mp_state = mp_owned._ensure_device_state(preserve_indexed_view=True)
    region_state = region_owned._ensure_device_state(preserve_indexed_view=True)
    mp_buffer = mp_state.families[GeometryFamily.MULTIPOINT]
    region_buffer = region_state.families[region_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_mp_fro, device_region_fro, mp_rows, region_rows, temporaries = (
        _prepare_indexed_pair_launch(
            mp_owned,
            region_owned,
            mp_indices,
            region_indices,
            runtime,
            source_offset=source_offset,
        )
    )
    kernel_name = (
        "multipoint_polygon_relation_compacted"
        if region_family is GeometryFamily.POLYGON
        else "multipoint_multipolygon_relation_compacted"
    )
    args = [
        ptr(device_mp_fro),
        ptr(mp_buffer.geometry_offsets),
        ptr(mp_buffer.empty_mask),
        ptr(mp_buffer.x),
        ptr(mp_buffer.y),
        ptr(device_region_fro),
        ptr(region_buffer.empty_mask),
        ptr(region_buffer.geometry_offsets),
    ]
    if region_family is not GeometryFamily.POLYGON:
        args.append(ptr(region_buffer.part_offsets))
    args.extend(
        [
            ptr(region_buffer.ring_offsets),
            ptr(region_buffer.x),
            ptr(region_buffer.y),
        ]
    )
    return _launch_kernel(
        _multipoint_relation_kernels,
        kernel_name,
        mp_rows,
        tuple(args),
        (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=temporaries,
        return_device=return_device,
        logical_count=logical_count,
        precision_plan=precision_plan,
        candidate_rows_right=region_rows,
        source_offset=source_offset,
        launch_capacity=launch_capacity,
        device_out=relation_out,
    )


def _classify_indexed_mp_mp(
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    *,
    precision_plan: PrecisionPlan,
    return_device: bool = False,
    logical_count=None,
    source_offset=None,
    launch_capacity: int | None = None,
    relation_out=None,
) -> np.ndarray:
    """MULTIPOINT x MULTIPOINT relation bits (left MP vs right MP)."""
    n = int(left_indices.size)
    if n == 0:
        if return_device:
            import cupy as cp

            return cp.empty(0, dtype=cp.uint8)
        return np.empty(0, dtype=np.uint8)
    left_state = left_owned._ensure_device_state(preserve_indexed_view=True)
    right_state = right_owned._ensure_device_state(preserve_indexed_view=True)
    left_buffer = left_state.families[GeometryFamily.MULTIPOINT]
    right_buffer = right_state.families[GeometryFamily.MULTIPOINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_left_fro, device_right_fro, left_rows, right_rows, temporaries = (
        _prepare_indexed_pair_launch(
            left_owned,
            right_owned,
            left_indices,
            right_indices,
            runtime,
            source_offset=source_offset,
        )
    )
    return _launch_kernel(
        _multipoint_relation_kernels,
        "multipoint_multipoint_relation_compacted",
        left_rows,
        (
            ptr(device_left_fro),
            ptr(left_buffer.geometry_offsets),
            ptr(left_buffer.empty_mask),
            ptr(left_buffer.x),
            ptr(left_buffer.y),
            ptr(device_right_fro),
            ptr(right_buffer.geometry_offsets),
            ptr(right_buffer.empty_mask),
            ptr(right_buffer.x),
            ptr(right_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
        extra_device_allocs=temporaries,
        return_device=return_device,
        logical_count=logical_count,
        precision_plan=precision_plan,
        candidate_rows_right=right_rows,
        source_offset=source_offset,
        launch_capacity=launch_capacity,
        device_out=relation_out,
    )


def _dispatch_multipoint_pairs(
    predicate: str,
    out: np.ndarray,
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    left_tags: np.ndarray,
    right_tags: np.ndarray,
    mp_left_mask: np.ndarray,
    mp_right_mask: np.ndarray,
    _apply_relation_rows,
    precision_plan: PrecisionPlan,
) -> None:
    """Dispatch multipoint pairs to the appropriate GPU kernel and convert results."""
    pt_tag = _POINT_TAG_INDEXED

    # --- MULTIPOINT on left ---

    # MP x point
    mask = mp_left_mask & (right_tags == pt_tag)
    if mask.any():
        idx = np.flatnonzero(mask)
        bits = _classify_indexed_mp_point(
            left_owned,
            right_owned,
            left_indices[idx],
            right_indices[idx],
            precision_plan=precision_plan,
        )
        _apply_relation_rows(
            out,
            idx,
            _multipoint_bits_to_predicate(
                predicate,
                bits,
                mp_on_left=True,
                target_family=GeometryFamily.POINT,
            ),
        )

    # MP x line families
    for lf, lt in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        mask = mp_left_mask & (right_tags == lt)
        if mask.any():
            idx = np.flatnonzero(mask)
            bits = _classify_indexed_mp_line(
                left_owned,
                right_owned,
                left_indices[idx],
                right_indices[idx],
                line_family=lf,
                precision_plan=precision_plan,
            )
            _apply_relation_rows(
                out,
                idx,
                _multipoint_bits_to_predicate(
                    predicate,
                    bits,
                    mp_on_left=True,
                    target_family=lf,
                ),
            )


def _dispatch_multipoint_pairs_device(
    predicate: str,
    out,
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices,
    right_indices,
    left_tags,
    right_tags,
    mp_left_mask,
    mp_right_mask,
    precision_plan: PrecisionPlan,
) -> None:
    """Dispatch multipoint relation-pair rows without exporting masks to host."""
    import cupy as cp

    left_state = left_owned._ensure_device_state(preserve_indexed_view=True)
    right_state = right_owned._ensure_device_state(preserve_indexed_view=True)
    pt_tag = _POINT_TAG_INDEXED

    if (
        GeometryFamily.MULTIPOINT in left_state.families
        and GeometryFamily.POINT in right_state.families
    ):
        idx = cp.flatnonzero(mp_left_mask & (right_tags == pt_tag)).astype(
            cp.int32,
            copy=False,
        )
        bits = _classify_indexed_mp_point(
            left_owned,
            right_owned,
            left_indices[idx],
            right_indices[idx],
            precision_plan=precision_plan,
            return_device=True,
        )
        out[idx] = _multipoint_bits_to_predicate(
            predicate,
            bits,
            mp_on_left=True,
            target_family=GeometryFamily.POINT,
        )

    for line_family, line_tag in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        if GeometryFamily.MULTIPOINT in left_state.families and line_family in right_state.families:
            idx = cp.flatnonzero(mp_left_mask & (right_tags == line_tag)).astype(
                cp.int32,
                copy=False,
            )
            bits = _classify_indexed_mp_line(
                left_owned,
                right_owned,
                left_indices[idx],
                right_indices[idx],
                line_family=line_family,
                precision_plan=precision_plan,
                return_device=True,
            )
            out[idx] = _multipoint_bits_to_predicate(
                predicate,
                bits,
                mp_on_left=True,
                target_family=line_family,
            )

    for region_family, region_tag in zip(
        _REGION_FAMILIES_INDEXED, _REGION_TAGS_INDEXED, strict=True
    ):
        if (
            GeometryFamily.MULTIPOINT in left_state.families
            and region_family in right_state.families
        ):
            idx = cp.flatnonzero(mp_left_mask & (right_tags == region_tag)).astype(
                cp.int32,
                copy=False,
            )
            bits = _classify_indexed_mp_region(
                left_owned,
                right_owned,
                left_indices[idx],
                right_indices[idx],
                region_family=region_family,
                precision_plan=precision_plan,
                return_device=True,
            )
            out[idx] = _multipoint_bits_to_predicate(
                predicate,
                bits,
                mp_on_left=True,
                target_family=region_family,
            )

    if (
        GeometryFamily.MULTIPOINT in left_state.families
        and GeometryFamily.MULTIPOINT in right_state.families
    ):
        idx = cp.flatnonzero(mp_left_mask & mp_right_mask).astype(cp.int32, copy=False)
        left_mp_indices = left_indices[idx]
        right_mp_indices = right_indices[idx]
        bits_forward = _classify_indexed_mp_mp(
            left_owned,
            right_owned,
            left_mp_indices,
            right_mp_indices,
            precision_plan=precision_plan,
            return_device=True,
        )
        if predicate in {"contains", "covers", "contains_properly"}:
            bits_reverse = _classify_indexed_mp_mp(
                right_owned,
                left_owned,
                right_mp_indices,
                left_mp_indices,
                precision_plan=precision_plan,
                return_device=True,
            )
            result = _multipoint_bits_to_predicate(
                predicate,
                bits_reverse,
                mp_on_left=False,
                target_family=GeometryFamily.MULTIPOINT,
            )
        elif predicate in {"within", "covered_by"}:
            result = _multipoint_bits_to_predicate(
                predicate,
                bits_forward,
                mp_on_left=True,
                target_family=GeometryFamily.MULTIPOINT,
            )
        else:
            result = _multipoint_bits_to_predicate(
                predicate,
                bits_forward,
                mp_on_left=True,
                target_family=GeometryFamily.MULTIPOINT,
            )
        out[idx] = result

    if (
        GeometryFamily.POINT in left_state.families
        and GeometryFamily.MULTIPOINT in right_state.families
    ):
        idx = cp.flatnonzero((left_tags == pt_tag) & mp_right_mask).astype(
            cp.int32,
            copy=False,
        )
        bits = _classify_indexed_mp_point(
            right_owned,
            left_owned,
            right_indices[idx],
            left_indices[idx],
            precision_plan=precision_plan,
            return_device=True,
        )
        out[idx] = _multipoint_bits_to_predicate(
            predicate,
            bits,
            mp_on_left=False,
            target_family=GeometryFamily.POINT,
        )

    for line_family, line_tag in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        if line_family in left_state.families and GeometryFamily.MULTIPOINT in right_state.families:
            idx = cp.flatnonzero((left_tags == line_tag) & mp_right_mask).astype(
                cp.int32,
                copy=False,
            )
            bits = _classify_indexed_mp_line(
                right_owned,
                left_owned,
                right_indices[idx],
                left_indices[idx],
                line_family=line_family,
                precision_plan=precision_plan,
                return_device=True,
            )
            out[idx] = _multipoint_bits_to_predicate(
                predicate,
                bits,
                mp_on_left=False,
                target_family=line_family,
            )

    for region_family, region_tag in zip(
        _REGION_FAMILIES_INDEXED, _REGION_TAGS_INDEXED, strict=True
    ):
        if (
            region_family in left_state.families
            and GeometryFamily.MULTIPOINT in right_state.families
        ):
            idx = cp.flatnonzero((left_tags == region_tag) & mp_right_mask).astype(
                cp.int32,
                copy=False,
            )
            bits = _classify_indexed_mp_region(
                right_owned,
                left_owned,
                right_indices[idx],
                left_indices[idx],
                region_family=region_family,
                precision_plan=precision_plan,
                return_device=True,
            )
            out[idx] = _multipoint_bits_to_predicate(
                predicate,
                bits,
                mp_on_left=False,
                target_family=region_family,
            )
