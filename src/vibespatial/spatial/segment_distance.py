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

request_nvrtc_warmup([
    ("segment-distance", _SEGMENT_DISTANCE_KERNEL_SOURCE, _SEGMENT_DISTANCE_KERNEL_NAMES),
])


def _segment_distance_kernels():
    return compile_kernel_group("segment-distance", _SEGMENT_DISTANCE_KERNEL_SOURCE, _SEGMENT_DISTANCE_KERNEL_NAMES)


# ---------------------------------------------------------------------------
# Canonical pair table: (left_family, right_family) → kernel name
# ---------------------------------------------------------------------------
_LS = GeometryFamily.LINESTRING
_MLS = GeometryFamily.MULTILINESTRING
_PG = GeometryFamily.POLYGON
_MPG = GeometryFamily.MULTIPOLYGON

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
    tail_args = [ptr(eff_left), ptr(eff_right), ptr(d_distances),
                 1 if exclusive else 0, pair_count]
    tail_types = [KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                  KERNEL_PARAM_I32, KERNEL_PARAM_I32]

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
