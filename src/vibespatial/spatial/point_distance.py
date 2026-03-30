from __future__ import annotations

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
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
    POINT_DISTANCE_KERNEL_SOURCE_FP64,
    format_distance_kernel_source,
)

_POINT_DISTANCE_KERNEL_SOURCE = POINT_DISTANCE_KERNEL_SOURCE_FP64

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("point-distance", _POINT_DISTANCE_KERNEL_SOURCE, _POINT_DISTANCE_KERNEL_NAMES),
])



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


def _compute_center(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
) -> tuple[float, float]:
    """Compute the centroid of the combined coordinate extent for centering."""
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for owned in (query_owned, tree_owned):
        for buffer in owned.families.values():
            if buffer.x.size > 0:
                all_x.append(buffer.x)
                all_y.append(buffer.y)
    if not all_x:
        return 0.0, 0.0
    combined_x = np.concatenate(all_x)
    combined_y = np.concatenate(all_y)
    cx = (float(np.nanmin(combined_x)) + float(np.nanmax(combined_x))) * 0.5
    cy = (float(np.nanmin(combined_y)) + float(np.nanmax(combined_y))) * 0.5
    return cx, cy


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
) -> bool:
    """Compute point -> geometry distance on device for a single tree family.

    Writes results into *d_distances* (device float64 array, shape pair_count).
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

    # Compute center for coordinate centering (cheap host-side operation).
    center_x, center_y = _compute_center(query_owned, tree_owned)

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
        ptr(query_state.validity), ptr(query_state.tags), ptr(query_state.family_row_offsets),
        ptr(query_points.geometry_offsets), ptr(query_points.empty_mask),
        ptr(query_points.x), ptr(query_points.y),
        FAMILY_TAGS[GeometryFamily.POINT],
        # tree state (common prefix)
        ptr(tree_state.validity), ptr(tree_state.tags), ptr(tree_state.family_row_offsets),
        ptr(tree_buffer.geometry_offsets),
    ]
    arg_types = [
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
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
    args.extend([
        ptr(tree_buffer.empty_mask),
        ptr(tree_buffer.x), ptr(tree_buffer.y),
        FAMILY_TAGS[tree_family],
        ptr(d_left), ptr(d_right),
        ptr(d_distances),
        1 if exclusive else 0,
        pair_count,
        center_x,
        center_y,
    ])
    arg_types.extend([
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_F64,
        KERNEL_PARAM_F64,
    ])

    grid, block = runtime.launch_config(kernels[kernel_name], pair_count)
    runtime.launch(
        kernels[kernel_name],
        grid=grid,
        block=block,
        params=(tuple(args), tuple(arg_types)),
    )
    runtime.synchronize()
    return True


def supported_point_distance_families() -> frozenset[GeometryFamily]:
    """Return the set of tree families supported by GPU point-distance kernels."""
    return frozenset(_FAMILY_KERNEL_MAP.keys())
