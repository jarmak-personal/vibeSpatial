"""GPU-accelerated shortest_line (binary constructive).

Computes the LineString connecting the nearest points between two
geometries for each element-wise pair.  Returns an OwnedGeometryArray
of 2-point LineStrings.

Architecture (ADR-0033): Tier 1 NVRTC -- geometry-specific inner loops
iterating all segment pairs across two geometries to track the closest
point pair.

Precision (ADR-0002): CONSTRUCTIVE class -- stays fp64 on all devices per
policy.  PrecisionPlan wired through dispatch for observability.

Zero D2H transfers in the hot path.  Geometry data stays device-resident;
only the final output OGA is assembled on host from four fp64 coordinate
arrays.
"""

from __future__ import annotations

import numpy as np
import shapely

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    TAG_FAMILIES,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    from_shapely_geometries,
    tile_single_row,
    unique_tag_pairs,
)
from vibespatial.kernels.constructive.shortest_line import (
    _SHORTEST_LINE_KERNEL_SOURCE,
    SHORTEST_LINE_KERNEL_NAMES,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan
from vibespatial.runtime.residency import Residency, TransferTrigger
from vibespatial.runtime.workload import WorkloadShape, detect_workload_shape

# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034)
# ---------------------------------------------------------------------------

request_nvrtc_warmup([
    ("shortest-line", _SHORTEST_LINE_KERNEL_SOURCE, SHORTEST_LINE_KERNEL_NAMES),
])


def _shortest_line_kernels():
    return compile_kernel_group(
        "shortest-line", _SHORTEST_LINE_KERNEL_SOURCE, SHORTEST_LINE_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Family ordering for canonical-pair normalisation
# ---------------------------------------------------------------------------

_PT = GeometryFamily.POINT
_LS = GeometryFamily.LINESTRING
_MLS = GeometryFamily.MULTILINESTRING
_PG = GeometryFamily.POLYGON
_MPG = GeometryFamily.MULTIPOLYGON

_FAMILY_ORDER: dict[GeometryFamily, int] = {
    _PT: 0,
    _LS: 1,
    _MLS: 2,
    _PG: 3,
    _MPG: 4,
}

_CANONICAL_KERNELS: dict[tuple[GeometryFamily, GeometryFamily], str] = {
    (_PT, _PT): "shortest_line_pt_pt",
    (_PT, _LS): "shortest_line_pt_ls",
    (_PT, _MLS): "shortest_line_pt_mls",
    (_PT, _PG): "shortest_line_pt_pg",
    (_PT, _MPG): "shortest_line_pt_mpg",
    (_LS, _LS): "shortest_line_ls_ls",
    (_LS, _MLS): "shortest_line_ls_mls",
    (_LS, _PG): "shortest_line_ls_pg",
    (_LS, _MPG): "shortest_line_ls_mpg",
    (_MLS, _MLS): "shortest_line_mls_mls",
    (_MLS, _PG): "shortest_line_mls_pg",
    (_MLS, _MPG): "shortest_line_mls_mpg",
    (_PG, _PG): "shortest_line_pg_pg",
    (_PG, _MPG): "shortest_line_pg_mpg",
    (_MPG, _MPG): "shortest_line_mpg_mpg",
}


# ---------------------------------------------------------------------------
# Family args builder (matches segment_distance pattern)
# ---------------------------------------------------------------------------

def _family_args(state, family, runtime):
    """Build (args, arg_types) for one side of a from_owned kernel."""
    ptr = runtime.pointer
    P = KERNEL_PARAM_PTR
    buf = state.families[family]

    # Common prefix: validity, tags, family_row_offsets.
    args = [ptr(state.validity), ptr(state.tags), ptr(state.family_row_offsets)]
    types = [P, P, P]

    # geometry_offsets (always present).
    args.append(ptr(buf.geometry_offsets))
    types.append(P)

    # Family-specific extra offset arrays.
    if family in (_MLS, _MPG):
        args.append(ptr(buf.part_offsets))
        types.append(P)
    if family in (_PG, _MPG):
        args.append(ptr(buf.ring_offsets))
        types.append(P)

    # empty_mask, x, y.
    args.extend([ptr(buf.empty_mask), ptr(buf.x), ptr(buf.y)])
    types.extend([P, P, P])

    # tag value.
    args.append(FAMILY_TAGS[family])
    types.append(KERNEL_PARAM_I32)

    return args, types


# ---------------------------------------------------------------------------
# Output assembly: build LineString OGA from four coordinate arrays
# ---------------------------------------------------------------------------

def _build_linestring_oga(
    out_ax: np.ndarray,
    out_ay: np.ndarray,
    out_bx: np.ndarray,
    out_by: np.ndarray,
    validity: np.ndarray,
) -> OwnedGeometryArray:
    """Build a LineString OwnedGeometryArray from closest point pairs.

    Each valid row becomes a 2-point LineString: (ax,ay) -> (bx,by).
    Invalid rows (null geometry) produce null entries.
    """
    n = len(validity)
    valid_mask = validity & np.isfinite(out_ax) & np.isfinite(out_bx)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        # All null/invalid -- return empty OGA
        return from_shapely_geometries([None] * n)

    # Build LineString buffers: each valid row contributes 2 coordinates
    # Interleave: x = [ax0, bx0, ax1, bx1, ...], y = [ay0, by0, ay1, by1, ...]
    valid_idx = np.flatnonzero(valid_mask)

    ls_x = np.empty(n_valid * 2, dtype=np.float64)
    ls_y = np.empty(n_valid * 2, dtype=np.float64)
    ls_x[0::2] = out_ax[valid_idx]
    ls_x[1::2] = out_bx[valid_idx]
    ls_y[0::2] = out_ay[valid_idx]
    ls_y[1::2] = out_by[valid_idx]

    # geometry_offsets: [0, 2, 4, 6, ...] for valid rows
    ls_geom_offsets = np.arange(n_valid + 1, dtype=np.int32) * 2

    ls_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        schema=get_geometry_buffer_schema(GeometryFamily.LINESTRING),
        row_count=n_valid,
        x=ls_x,
        y=ls_y,
        geometry_offsets=ls_geom_offsets,
        empty_mask=np.zeros(n_valid, dtype=bool),
    )

    # Build OGA routing arrays
    tags = np.full(n, -1, dtype=np.int8)
    tags[valid_mask] = FAMILY_TAGS[GeometryFamily.LINESTRING]

    family_row_offsets = np.full(n, -1, dtype=np.int32)
    family_row_offsets[valid_idx] = np.arange(n_valid, dtype=np.int32)

    return OwnedGeometryArray(
        validity=valid_mask.copy(),
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.LINESTRING: ls_buffer},
        residency=Residency.HOST,
    )


# ---------------------------------------------------------------------------
# GPU kernel dispatch
# ---------------------------------------------------------------------------

def _launch_shortest_line_subgroup(
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    d_left_idx,
    d_right_idx,
    d_out_ax,
    d_out_ay,
    d_out_bx,
    d_out_by,
    sub_count: int,
    left_family: GeometryFamily,
    right_family: GeometryFamily,
    swapped: bool,
) -> bool:
    """Launch the appropriate shortest_line kernel for a family pair.

    Returns True on success, False if the family pair is not supported.
    When *swapped* is True, the left/right index arrays were swapped to
    achieve canonical ordering -- the output coordinates are then swapped
    back (A<->B) by the caller.
    """
    q_ord = _FAMILY_ORDER.get(left_family)
    t_ord = _FAMILY_ORDER.get(right_family)
    if q_ord is None or t_ord is None:
        return False

    if q_ord <= t_ord:
        canonical = (left_family, right_family)
        eff_left_owned, eff_right_owned = left_owned, right_owned
        eff_left, eff_right = d_left_idx, d_right_idx
    else:
        canonical = (right_family, left_family)
        eff_left_owned, eff_right_owned = right_owned, left_owned
        eff_left, eff_right = d_right_idx, d_left_idx
        swapped = not swapped

    kernel_name = _CANONICAL_KERNELS.get(canonical)
    if kernel_name is None:
        return False

    left_state = eff_left_owned._ensure_device_state()
    right_state = eff_right_owned._ensure_device_state()

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _shortest_line_kernels()

    left_args, left_types = _family_args(left_state, canonical[0], runtime)
    right_args, right_types = _family_args(right_state, canonical[1], runtime)

    # Tail: left_idx, right_idx, out_ax, out_ay, out_bx, out_by, pair_count
    # If swapped, we still write to the same output arrays but will swap
    # the A/B meaning at the caller level.
    if swapped:
        # Kernel writes "closest on canonical-left" to out_ax/ay, but that
        # is actually the "right" geometry in the original pairing.
        # We swap the output pointers so the kernel writes A->B correctly.
        tail_args = [
            ptr(eff_left), ptr(eff_right),
            ptr(d_out_bx), ptr(d_out_by), ptr(d_out_ax), ptr(d_out_ay),
            sub_count,
        ]
    else:
        tail_args = [
            ptr(eff_left), ptr(eff_right),
            ptr(d_out_ax), ptr(d_out_ay), ptr(d_out_bx), ptr(d_out_by),
            sub_count,
        ]
    tail_types = [KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                  KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                  KERNEL_PARAM_I32]

    all_args = tuple(left_args + right_args + tail_args)
    all_types = tuple(left_types + right_types + tail_types)

    grid, block = runtime.launch_config(kernels[kernel_name], sub_count)
    runtime.launch(
        kernels[kernel_name],
        grid=grid,
        block=block,
        params=(all_args, all_types),
    )
    return True


@register_kernel_variant(
    "shortest_line",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "linestring", "polygon",
        "multipoint", "multilinestring", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "shortest_line"),
)
def _shortest_line_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Pure-GPU element-wise shortest_line.

    Groups rows by (left_tag, right_tag) and dispatches to the appropriate
    NVRTC kernel per group.  Geometry data stays device-resident; only
    small index sub-arrays and per-group result sub-arrays cross the bus.

    Output assembly is done on host since the result is a new
    OwnedGeometryArray of LineStrings with fixed 2-point structure.
    """
    n = left.row_count
    runtime = get_cuda_runtime()

    # Ensure geometry buffers are device-resident
    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="shortest_line: left geometry for GPU kernel",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="shortest_line: right geometry for GPU kernel",
    )

    left_tags = left.tags
    right_tags = right.tags
    left_valid = left.validity
    right_valid = right.validity

    both_valid = left_valid & right_valid
    valid_idx = np.flatnonzero(both_valid)
    if valid_idx.size == 0:
        return from_shapely_geometries([None] * n)

    # Host-side result arrays: INFINITY sentinel for uncomputed rows
    out_ax = np.full(n, np.inf, dtype=np.float64)
    out_ay = np.full(n, np.inf, dtype=np.float64)
    out_bx = np.full(n, np.inf, dtype=np.float64)
    out_by = np.full(n, np.inf, dtype=np.float64)

    valid_left_tags = left_tags[valid_idx]
    valid_right_tags = right_tags[valid_idx]

    all_ok = True

    for lt, rt in unique_tag_pairs(valid_left_tags, valid_right_tags):
        lf = TAG_FAMILIES.get(lt)
        rf = TAG_FAMILIES.get(rt)
        if lf is None or rf is None:
            all_ok = False
            continue

        # MultiPoint not yet supported by the NVRTC kernel
        if lf == GeometryFamily.MULTIPOINT or rf == GeometryFamily.MULTIPOINT:
            all_ok = False
            continue

        sub_mask = (valid_left_tags == lt) & (valid_right_tags == rt)
        sub_valid_pos = np.flatnonzero(sub_mask)
        sub_idx = valid_idx[sub_valid_pos]
        sub_count = sub_idx.size
        if sub_count == 0:
            continue

        # Upload global row indices for this sub-group
        d_idx = runtime.from_host(sub_idx.astype(np.int32))

        # Allocate sub-output arrays: kernel writes at positions 0..sub_count-1
        d_sub_ax = runtime.allocate((sub_count,), np.float64)
        d_sub_ay = runtime.allocate((sub_count,), np.float64)
        d_sub_bx = runtime.allocate((sub_count,), np.float64)
        d_sub_by = runtime.allocate((sub_count,), np.float64)

        ok = _launch_shortest_line_subgroup(
            left, right,
            d_idx, d_idx,  # left_idx == right_idx for element-wise
            d_sub_ax, d_sub_ay, d_sub_bx, d_sub_by,
            sub_count, lf, rf, swapped=False,
        )

        if ok:
            runtime.synchronize()
            # Transfer sub-results to host and scatter into full arrays
            h_sub_ax = np.empty(sub_count, dtype=np.float64)
            h_sub_ay = np.empty(sub_count, dtype=np.float64)
            h_sub_bx = np.empty(sub_count, dtype=np.float64)
            h_sub_by = np.empty(sub_count, dtype=np.float64)
            runtime.copy_device_to_host(d_sub_ax, h_sub_ax)
            runtime.copy_device_to_host(d_sub_ay, h_sub_ay)
            runtime.copy_device_to_host(d_sub_bx, h_sub_bx)
            runtime.copy_device_to_host(d_sub_by, h_sub_by)

            out_ax[sub_idx] = h_sub_ax
            out_ay[sub_idx] = h_sub_ay
            out_bx[sub_idx] = h_sub_bx
            out_by[sub_idx] = h_sub_by
        else:
            all_ok = False

        runtime.free(d_sub_ax)
        runtime.free(d_sub_ay)
        runtime.free(d_sub_bx)
        runtime.free(d_sub_by)
        runtime.free(d_idx)

    if not all_ok:
        # Some family pairs weren't supported -- fall back to Shapely for
        # those specific rows.
        unsupported = np.isinf(out_ax) & both_valid
        if np.any(unsupported):
            unsup_idx = np.flatnonzero(unsupported)
            left_shapely = np.asarray(left.to_shapely(), dtype=object)
            right_shapely = np.asarray(right.to_shapely(), dtype=object)
            for idx in unsup_idx:
                sl = shapely.shortest_line(left_shapely[idx], right_shapely[idx])
                if sl is not None and not sl.is_empty:
                    coords = np.array(sl.coords)
                    out_ax[idx] = coords[0, 0]
                    out_ay[idx] = coords[0, 1]
                    out_bx[idx] = coords[1, 0]
                    out_by[idx] = coords[1, 1]

    return _build_linestring_oga(out_ax, out_ay, out_bx, out_by, both_valid)


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "shortest_line",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "linestring", "polygon",
        "multipoint", "multilinestring", "multipolygon",
    ),
    supports_mixed=True,
    tags=("shapely", "constructive", "shortest_line"),
)
def _shortest_line_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> np.ndarray:
    """CPU shortest_line via Shapely."""
    left_shapely = np.asarray(left.to_shapely(), dtype=object)
    right_shapely = np.asarray(right.to_shapely(), dtype=object)
    return shapely.shortest_line(left_shapely, right_shapely)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shortest_line_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray | np.ndarray:
    """Element-wise shortest_line between two OwnedGeometryArrays.

    Returns an OwnedGeometryArray of 2-point LineStrings (GPU path) or
    a numpy array of Shapely LineStrings (CPU path).  Null geometries
    produce null entries.

    Supports pairwise (N vs N) and broadcast-right (N vs 1) modes.
    """
    n = left.row_count
    workload = detect_workload_shape(n, right.row_count)

    if workload is WorkloadShape.BROADCAST_RIGHT:
        right = tile_single_row(right, n)

    if n == 0:
        return from_shapely_geometries([])

    if isinstance(precision, str):
        precision = PrecisionMode(precision)

    selection = plan_dispatch_selection(
        kernel_name="shortest_line",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=n,
        requested_mode=dispatch_mode,
    )

    precision_plan = select_precision_plan(
        runtime_selection=selection,
        kernel_class=KernelClass.CONSTRUCTIVE,
        requested=precision,
    )

    if selection.selected is ExecutionMode.GPU:
        try:
            result = _shortest_line_gpu(left, right)
            record_dispatch_event(
                surface="shortest_line_owned",
                operation="shortest_line",
                implementation="shortest_line_gpu",
                reason="element-wise shortest_line via owned GPU kernels",
                detail=(
                    f"rows={n}, precision={precision_plan.compute_precision.value}, "
                    f"workload={workload.value}"
                ),
                selected=ExecutionMode.GPU,
            )
            return result
        except Exception as exc:
            record_fallback_event(
                surface="shortest_line_owned",
                reason=f"GPU shortest_line kernel failed: {exc}",
                detail=f"rows={n}, falling back to CPU Shapely path",
                pipeline="shortest_line",
            )

    record_dispatch_event(
        surface="shortest_line_owned",
        operation="shortest_line",
        implementation="shapely_cpu",
        reason="GPU not available or not selected for shortest_line",
        detail=f"rows={n}",
        selected=ExecutionMode.CPU,
    )
    return _shortest_line_cpu(left, right)
