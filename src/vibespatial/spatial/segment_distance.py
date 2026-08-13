from __future__ import annotations

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray

# ---------------------------------------------------------------------------
# Family ordering for canonical-pair normalisation (lower value = "left").
# ---------------------------------------------------------------------------
_FAMILY_ORDER: dict[GeometryFamily, int] = {
    GeometryFamily.LINESTRING: 0,
    GeometryFamily.MULTILINESTRING: 1,
    GeometryFamily.POLYGON: 2,
    GeometryFamily.MULTIPOLYGON: 3,
}


from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402
from vibespatial.spatial.segment_distance_kernels import (  # noqa: E402
    _SEGMENT_DISTANCE_KERNEL_NAMES,
    _SEGMENT_DISTANCE_KERNEL_SOURCE,
)

request_nvrtc_warmup(
    [
        ("segment-distance", _SEGMENT_DISTANCE_KERNEL_SOURCE, _SEGMENT_DISTANCE_KERNEL_NAMES),
    ]
)


def _segment_distance_kernels():
    return compile_kernel_group(
        "segment-distance", _SEGMENT_DISTANCE_KERNEL_SOURCE, _SEGMENT_DISTANCE_KERNEL_NAMES
    )


# ---------------------------------------------------------------------------
# Canonical pair table: (left_family, right_family) → kernel name
# ---------------------------------------------------------------------------
_LS = GeometryFamily.LINESTRING
_MLS = GeometryFamily.MULTILINESTRING
_PG = GeometryFamily.POLYGON
_MPG = GeometryFamily.MULTIPOLYGON

_FAMILY_KIND = {_LS: 0, _MLS: 1, _PG: 2, _MPG: 3}

_CANONICAL_KERNELS: dict[tuple[GeometryFamily, GeometryFamily], str] = {
    (_LS, _LS): "distance_ls_ls_from_owned",
    (_LS, _MLS): "distance_ls_mls_from_owned",
    (_LS, _PG): "distance_ls_pg_from_owned",
    (_LS, _MPG): "distance_ls_mpg_from_owned",
    (_MLS, _MLS): "distance_mls_mls_from_owned",
    (_MLS, _PG): "distance_mls_pg_from_owned",
    (_MLS, _MPG): "distance_mls_mpg_from_owned",
    (_PG, _PG): "distance_pg_pg_from_owned",
    (_PG, _MPG): "distance_pg_mpg_from_owned",
    (_MPG, _MPG): "distance_mpg_mpg_from_owned",
}


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


def compute_segment_distance_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    d_left,
    d_right,
    d_distances,
    pair_count: int,
    *,
    query_family: GeometryFamily,
    tree_family: GeometryFamily,
    exclusive: bool = False,
) -> bool:
    """Compute geometry-geometry distance on device.

    Covers all combinations of LINESTRING, MULTILINESTRING, POLYGON,
    MULTIPOLYGON on both sides.  Uses canonical-pair normalisation with
    index swapping for symmetric pairs.

    Writes results into *d_distances*.  Returns True on success, False
    if the family pair is not supported.
    """
    # Canonical ordering — lower _FAMILY_ORDER value goes on left.
    q_ord = _FAMILY_ORDER.get(query_family)
    t_ord = _FAMILY_ORDER.get(tree_family)
    if q_ord is None or t_ord is None:
        return False

    if q_ord <= t_ord:
        canonical = (query_family, tree_family)
        left_owned, right_owned = query_owned, tree_owned
        eff_left, eff_right = d_left, d_right
    else:
        canonical = (tree_family, query_family)
        left_owned, right_owned = tree_owned, query_owned
        eff_left, eff_right = d_right, d_left

    kernel_name = _CANONICAL_KERNELS.get(canonical)
    if kernel_name is None:
        return False

    from vibespatial.runtime.residency import Residency, TransferTrigger

    left_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"segment_distance GPU kernel: left {canonical[0].name}",
    )
    right_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"segment_distance GPU kernel: right {canonical[1].name}",
    )

    left_state = left_owned._ensure_device_state()
    right_state = right_owned._ensure_device_state()

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _segment_distance_kernels()

    left_args, left_types = _family_args(left_state, canonical[0], runtime)
    right_args, right_types = _family_args(right_state, canonical[1], runtime)

    # Tail: left_idx, right_idx, out, exclusive, pair_count.
    tail_args = [ptr(eff_left), ptr(eff_right), ptr(d_distances), 1 if exclusive else 0, pair_count]
    tail_types = [
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_I32,
    ]

    all_args = tuple(left_args + right_args + tail_args)
    all_types = tuple(left_types + right_types + tail_types)

    grid, block = runtime.launch_config(kernels[kernel_name], pair_count)
    runtime.launch(
        kernels[kernel_name],
        grid=grid,
        block=block,
        params=(all_args, all_types),
    )
    runtime.synchronize()
    return True


def supported_segment_distance_families() -> frozenset[GeometryFamily]:
    """Return the set of geometry families supported by segment-distance kernels."""
    return frozenset(_FAMILY_ORDER.keys())


def compute_segment_distance_partition_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    d_left,
    d_right,
    d_distances,
    launch_capacity: int,
    *,
    query_family: GeometryFamily,
    tree_family: GeometryFamily,
    source_offset,
    logical_count,
    source_positions,
    exclusive: bool = False,
) -> bool:
    """Run one non-point family span from a shared relation partition."""
    left_kind = _FAMILY_KIND.get(query_family)
    right_kind = _FAMILY_KIND.get(tree_family)
    if left_kind is None or right_kind is None:
        return False

    from vibespatial.runtime.residency import Residency, TransferTrigger

    query_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"partitioned segment distance: left {query_family.name}",
    )
    tree_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"partitioned segment distance: right {tree_family.name}",
    )
    left_state = query_owned._ensure_device_state()
    right_state = tree_owned._ensure_device_state()
    left_buffer = left_state.families[query_family]
    right_buffer = right_state.families[tree_family]

    def _offsets(buffer, family):
        part = (
            buffer.part_offsets
            if family in (_MLS, _MPG)
            else buffer.geometry_offsets
        )
        ring = (
            buffer.ring_offsets
            if family in (_PG, _MPG)
            else buffer.geometry_offsets
        )
        return part, ring

    left_part, left_ring = _offsets(left_buffer, query_family)
    right_part, right_ring = _offsets(right_buffer, tree_family)
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernel = _segment_distance_kernels()["distance_family_partition_from_owned"]
    args = (
        ptr(left_state.validity),
        ptr(left_state.tags),
        ptr(left_state.family_row_offsets),
        ptr(left_buffer.geometry_offsets),
        ptr(left_part),
        ptr(left_ring),
        ptr(left_buffer.empty_mask),
        ptr(left_buffer.x),
        ptr(left_buffer.y),
        FAMILY_TAGS[query_family],
        left_kind,
        ptr(right_state.validity),
        ptr(right_state.tags),
        ptr(right_state.family_row_offsets),
        ptr(right_buffer.geometry_offsets),
        ptr(right_part),
        ptr(right_ring),
        ptr(right_buffer.empty_mask),
        ptr(right_buffer.x),
        ptr(right_buffer.y),
        FAMILY_TAGS[tree_family],
        right_kind,
        ptr(d_left),
        ptr(d_right),
        ptr(source_positions),
        ptr(source_offset),
        ptr(logical_count),
        ptr(d_distances),
        1 if exclusive else 0,
        launch_capacity,
    )
    types = (
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_I32,
    )
    grid, block = runtime.launch_config(kernel, launch_capacity)
    runtime.launch(kernel, grid=grid, block=block, params=(args, types))
    return True
