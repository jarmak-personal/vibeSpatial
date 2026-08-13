from __future__ import annotations

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray
from vibespatial.runtime.precision import PrecisionMode
from vibespatial.spatial.point_distance_kernels import (
    _POINT_DISTANCE_KERNEL_NAMES,
    POINT_DISTANCE_KERNEL_SOURCE_FP32,
    POINT_DISTANCE_KERNEL_SOURCE_FP64,
    format_distance_kernel_source,
)

_POINT_DISTANCE_KERNEL_SOURCE = POINT_DISTANCE_KERNEL_SOURCE_FP64

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup(
    [
        (
            "point-distance-double",
            POINT_DISTANCE_KERNEL_SOURCE_FP64,
            _POINT_DISTANCE_KERNEL_NAMES,
        ),
        (
            "point-distance-float",
            POINT_DISTANCE_KERNEL_SOURCE_FP32,
            _POINT_DISTANCE_KERNEL_NAMES,
        ),
    ]
)


def _point_distance_kernels(compute_type: str = "double"):
    source = format_distance_kernel_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"point-distance-{compute_type}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_POINT_DISTANCE_KERNEL_NAMES,
    )


_FAMILY_KERNEL_MAP: dict[GeometryFamily, tuple[str, bool, bool]] = {
    GeometryFamily.LINESTRING: ("point_linestring_distance_from_owned", False, False),
    GeometryFamily.MULTILINESTRING: ("point_multilinestring_distance_from_owned", True, False),
    GeometryFamily.POLYGON: ("point_polygon_distance_from_owned", False, True),
    GeometryFamily.MULTIPOLYGON: ("point_multipolygon_distance_from_owned", True, True),
}


_POINTSET_TARGET_KIND = {
    GeometryFamily.POINT: 0,
    GeometryFamily.MULTIPOINT: 1,
    GeometryFamily.LINESTRING: 2,
    GeometryFamily.MULTILINESTRING: 3,
    GeometryFamily.POLYGON: 4,
    GeometryFamily.MULTIPOLYGON: 5,
}


def compute_distance_center_device(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
):
    """Reduce device row bounds to one device-resident centering vector."""
    import cupy as cp

    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
    from vibespatial.runtime.residency import Residency, TransferTrigger

    extrema = []
    for owned in (query_owned, tree_owned):
        owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="point distance centering consumes device bounds metadata",
        )
        bounds = cp.asarray(compute_geometry_bounds_device(owned), dtype=cp.float64).reshape(
            -1, 4
        )
        finite = cp.isfinite(bounds)
        extrema.append(
            cp.stack(
                (
                    cp.min(cp.where(finite[:, 0], bounds[:, 0], cp.inf)),
                    cp.min(cp.where(finite[:, 1], bounds[:, 1], cp.inf)),
                    cp.max(cp.where(finite[:, 2], bounds[:, 2], -cp.inf)),
                    cp.max(cp.where(finite[:, 3], bounds[:, 3], -cp.inf)),
                )
            )
        )
    combined = cp.stack(extrema)
    center = cp.stack(
        (
            (cp.min(combined[:, 0]) + cp.max(combined[:, 2])) * 0.5,
            (cp.min(combined[:, 1]) + cp.max(combined[:, 3])) * 0.5,
        )
    )
    return cp.where(cp.isfinite(center), center, 0.0).astype(cp.float64, copy=False)


def compute_point_distance_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    d_left,
    d_right,
    d_distances,
    pair_count: int,
    *,
    tree_family: GeometryFamily,
    exclusive: bool = False,
    compute_precision: PrecisionMode = PrecisionMode.AUTO,
    logical_count=None,
    center_device=None,
) -> bool:
    """Compute point -> geometry distance on device for a single tree family.

    Writes results into *d_distances* (device float64 array, shape pair_count).
    When *logical_count* is supplied, ``pair_count`` is physical capacity and
    the one-element device count guards the active compact prefix in-kernel.
    Returns True if the kernel was dispatched, False if the family is not
    supported (caller should fall back to Shapely).
    """
    spec = _FAMILY_KERNEL_MAP.get(tree_family)
    if spec is None:
        return False

    kernel_name, needs_part_offsets, needs_ring_offsets = spec

    # Determine compute type from precision plan.
    if compute_precision is PrecisionMode.AUTO:
        from vibespatial.runtime.adaptive import get_cached_snapshot

        snapshot = get_cached_snapshot()
        use_fp32 = not snapshot.device_profile.favors_native_fp64
    else:
        use_fp32 = compute_precision is PrecisionMode.FP32
    compute_type = "float" if use_fp32 else "double"

    from vibespatial.runtime.residency import Residency, TransferTrigger

    query_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="point_distance GPU kernel: query points",
    )
    tree_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"point_distance GPU kernel: tree {tree_family.name}",
    )

    if center_device is None:
        center_device = compute_distance_center_device(query_owned, tree_owned)

    query_state = query_owned._ensure_device_state()
    tree_state = tree_owned._ensure_device_state()
    query_points = query_state.families[GeometryFamily.POINT]
    tree_buffer = tree_state.families[tree_family]

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _point_distance_kernels(compute_type)

    # Build argument list following the from_owned convention.
    args = [
        # query point state
        ptr(query_state.validity),
        ptr(query_state.tags),
        ptr(query_state.family_row_offsets),
        ptr(query_points.geometry_offsets),
        ptr(query_points.empty_mask),
        ptr(query_points.x),
        ptr(query_points.y),
        FAMILY_TAGS[GeometryFamily.POINT],
        # tree state (common prefix)
        ptr(tree_state.validity),
        ptr(tree_state.tags),
        ptr(tree_state.family_row_offsets),
        ptr(tree_buffer.geometry_offsets),
    ]
    arg_types = [
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
    ]

    # Family-specific offset arrays.
    if needs_part_offsets:
        args.append(ptr(tree_buffer.part_offsets))
        arg_types.append(KERNEL_PARAM_PTR)
    if needs_ring_offsets:
        args.append(ptr(tree_buffer.ring_offsets))
        arg_types.append(KERNEL_PARAM_PTR)

    # Remaining tree buffer fields + pair / output + center coordinates.
    args.extend(
        [
            ptr(tree_buffer.empty_mask),
            ptr(tree_buffer.x),
            ptr(tree_buffer.y),
            FAMILY_TAGS[tree_family],
            ptr(d_left),
            ptr(d_right),
            ptr(d_distances),
            1 if exclusive else 0,
            ptr(logical_count),
            pair_count,
            ptr(center_device),
        ]
    )
    arg_types.extend(
        [
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
        ]
    )

    grid, block = runtime.launch_config(kernels[kernel_name], pair_count)
    runtime.launch(
        kernels[kernel_name],
        grid=grid,
        block=block,
        params=(tuple(args), tuple(arg_types)),
    )
    return True


def compute_pointset_distance_gpu(
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
    center_device,
    exclusive: bool = False,
    compute_precision: PrecisionMode = PrecisionMode.AUTO,
) -> bool:
    """Run one point/multipoint family span from a shared relation partition."""
    if query_family not in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
        return False
    target_kind = _POINTSET_TARGET_KIND.get(tree_family)
    if target_kind is None:
        return False

    if compute_precision is PrecisionMode.AUTO:
        from vibespatial.runtime.adaptive import get_cached_snapshot

        use_fp32 = not get_cached_snapshot().device_profile.favors_native_fp64
    else:
        use_fp32 = compute_precision is PrecisionMode.FP32

    from vibespatial.runtime.residency import Residency, TransferTrigger

    query_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"pointset distance GPU kernel: query {query_family.name}",
    )
    tree_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"pointset distance GPU kernel: tree {tree_family.name}",
    )
    query_state = query_owned._ensure_device_state()
    tree_state = tree_owned._ensure_device_state()
    query_buffer = query_state.families[query_family]
    tree_buffer = tree_state.families[tree_family]

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernel = _point_distance_kernels("float" if use_fp32 else "double")[
        "pointset_family_distance_from_owned"
    ]
    tree_part_offsets = (
        tree_buffer.part_offsets
        if tree_family in (GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON)
        else tree_buffer.geometry_offsets
    )
    tree_ring_offsets = (
        tree_buffer.ring_offsets
        if tree_family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
        else tree_buffer.geometry_offsets
    )
    args = (
        ptr(query_state.validity),
        ptr(query_state.tags),
        ptr(query_state.family_row_offsets),
        ptr(query_buffer.geometry_offsets),
        ptr(query_buffer.empty_mask),
        ptr(query_buffer.x),
        ptr(query_buffer.y),
        FAMILY_TAGS[query_family],
        ptr(tree_state.validity),
        ptr(tree_state.tags),
        ptr(tree_state.family_row_offsets),
        ptr(tree_buffer.geometry_offsets),
        ptr(tree_part_offsets),
        ptr(tree_ring_offsets),
        ptr(tree_buffer.empty_mask),
        ptr(tree_buffer.x),
        ptr(tree_buffer.y),
        FAMILY_TAGS[tree_family],
        target_kind,
        ptr(d_left),
        ptr(d_right),
        ptr(source_positions),
        ptr(source_offset),
        ptr(logical_count),
        ptr(d_distances),
        1 if exclusive else 0,
        launch_capacity,
        ptr(center_device),
    )
    types = (
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
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
        KERNEL_PARAM_PTR,
    )
    grid, block = runtime.launch_config(kernel, launch_capacity)
    runtime.launch(kernel, grid=grid, block=block, params=(args, types))
    return True


def supported_point_distance_families() -> frozenset[GeometryFamily]:
    """Return the set of tree families supported by GPU point-distance kernels."""
    return frozenset(_FAMILY_KERNEL_MAP.keys())


def supported_pointset_distance_families() -> frozenset[GeometryFamily]:
    """Return all owned target families supported by pointset distance."""
    return frozenset(_POINTSET_TARGET_KIND)
