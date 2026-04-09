from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.linestring_buffer_cpu import (
    build_linestring_buffers_cpu,
)
from vibespatial.constructive.linestring_kernels import (
    _LINESTRING_BUFFER_KERNEL_NAMES,
    _LINESTRING_BUFFER_KERNEL_SOURCE,
)
from vibespatial.constructive.polygon import _build_device_backed_polygon_output_variable
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency, TransferTrigger, combined_residency

request_nvrtc_warmup([
    ("linestring-buffer", _LINESTRING_BUFFER_KERNEL_SOURCE, _LINESTRING_BUFFER_KERNEL_NAMES),
])

from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402

request_warmup(["exclusive_scan_i32"])


def _linestring_buffer_kernels():
    return compile_kernel_group("linestring-buffer", _LINESTRING_BUFFER_KERNEL_SOURCE, _LINESTRING_BUFFER_KERNEL_NAMES)


_CAP_STYLE_MAP = {"round": 0, "flat": 1, "square": 2}
_JOIN_STYLE_MAP = {"round": 0, "mitre": 1, "bevel": 2}


def supports_two_point_linestring_buffer_fast_path(
    lines: OwnedGeometryArray,
    *,
    quad_segs: int,
    cap_style: str,
    join_style: str,
    single_sided: bool,
) -> bool:
    """Return True when every row is a simple two-point linestring.

    The general GPU linestring buffer path is not yet competitive on small AUTO
    workloads, but the exact two-point segment case used by the network/grid
    shootouts is. Recognizing that shape lets the public buffer surface stay on
    GPU without forcing broader multi-vertex linestring workloads down the same
    path.
    """

    if (
        single_sided
        or quad_segs < 1
        or cap_style != "round"
        or join_style != "round"
        or GeometryFamily.LINESTRING not in lines.families
        or len(lines.families) != 1
        or not np.all(lines.validity)
    ):
        return False

    line_buffer = lines.families[GeometryFamily.LINESTRING]
    if line_buffer.host_materialized:
        if np.any(line_buffer.empty_mask):
            return False
        offsets = np.asarray(line_buffer.geometry_offsets, dtype=np.int32)
        return bool(
            offsets.shape == (lines.row_count + 1,)
            and np.all((offsets[1:] - offsets[:-1]) == 2)
        )

    state = lines.device_state
    if cp is None or state is None or GeometryFamily.LINESTRING not in state.families:
        return False

    try:
        device_line_buffer = state.families[GeometryFamily.LINESTRING]
        if bool(cp.any(device_line_buffer.empty_mask).item()):
            return False
        offsets = cp.asarray(device_line_buffer.geometry_offsets)
        if int(offsets.size) != lines.row_count + 1:
            return False
        return bool(cp.all((offsets[1:] - offsets[:-1]) == 2).item())
    except Exception:
        return False


def linestring_buffer_owned_array(
    lines: OwnedGeometryArray,
    distance: float | np.ndarray,
    *,
    quad_segs: int = 8,
    cap_style: str = "round",
    join_style: str = "round",
    mitre_limit: float = 5.0,
    dispatch_mode: ExecutionMode = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    if GeometryFamily.LINESTRING not in lines.families or len(lines.families) != 1:
        raise ValueError("linestring_buffer_owned_array requires a linestring-only OwnedGeometryArray")
    if not np.all(lines.validity):
        raise ValueError("linestring_buffer_owned_array requires non-null rows only")
    if np.any(lines.tags != FAMILY_TAGS[GeometryFamily.LINESTRING]):
        raise ValueError("linestring_buffer_owned_array requires linestring-only rows")

    line_buffer = lines.families[GeometryFamily.LINESTRING]
    if np.any(line_buffer.empty_mask):
        raise ValueError("linestring_buffer_owned_array requires non-empty rows only")

    radii = (
        np.full(lines.row_count, float(distance), dtype=np.float64)
        if np.isscalar(distance)
        else np.asarray(distance, dtype=np.float64)
    )
    if radii.shape != (lines.row_count,):
        raise ValueError("distance must be a scalar or length-matched vector")

    selected_mode = plan_dispatch_selection(
        kernel_name="linestring_buffer",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=lines.row_count,
        requested_mode=dispatch_mode,
        current_residency=combined_residency(lines),
    ).selected

    cap_int = _CAP_STYLE_MAP.get(cap_style, 0)
    join_int = _JOIN_STYLE_MAP.get(join_style, 0)

    if selected_mode is not ExecutionMode.GPU:
        return build_linestring_buffers_cpu(
            lines, radii, quad_segs=quad_segs,
            cap_style=cap_style, join_style=join_style, mitre_limit=mitre_limit,
        )

    return _build_linestring_buffers_gpu(
        lines, radii, quad_segs=quad_segs,
        cap_style=cap_int, join_style=join_int, mitre_limit=mitre_limit,
    )
def _build_linestring_buffers_gpu(
    lines: OwnedGeometryArray,
    radii: np.ndarray,
    *,
    quad_segs: int,
    cap_style: int = 0,
    join_style: int = 0,
    mitre_limit: float = 5.0,
) -> OwnedGeometryArray:
    lines.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="linestring_buffer_owned_array selected GPU execution",
    )
    runtime = get_cuda_runtime()
    state = lines._ensure_device_state()
    line_buf = state.families[GeometryFamily.LINESTRING]

    device_radii = runtime.from_host(radii)
    device_counts = runtime.allocate((lines.row_count,), np.int32)
    device_offsets = None
    device_x = None
    device_y = None
    success = False

    try:
        kernels = _linestring_buffer_kernels()
        ptr = runtime.pointer

        # Pass 1: count output vertices per row
        count_params = (
            (
                ptr(state.family_row_offsets),
                ptr(line_buf.geometry_offsets),
                ptr(line_buf.empty_mask),
                ptr(line_buf.x),
                ptr(line_buf.y),
                ptr(device_radii),
                quad_segs,
                join_style,
                cap_style,
                mitre_limit,
                ptr(device_counts),
                lines.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        count_grid, count_block = runtime.launch_config(kernels["linestring_buffer_count"], lines.row_count)
        runtime.launch(kernels["linestring_buffer_count"],
                       grid=count_grid, block=count_block, params=count_params)

        # Compute exclusive prefix sum for scatter offsets
        device_offsets = exclusive_sum(device_counts)

        total_verts = count_scatter_total(runtime, device_counts, device_offsets)

        if total_verts == 0:
            device_x = runtime.allocate((0,), np.float64)
            device_y = runtime.allocate((0,), np.float64)
            d_geometry_offsets = cp.arange(lines.row_count + 1, dtype=cp.int32)
            d_ring_offsets = cp.zeros(lines.row_count + 1, dtype=cp.int32)
            success = True
            return _build_device_backed_polygon_output_variable(
                device_x, device_y,
                row_count=lines.row_count,
                geometry_offsets=d_geometry_offsets,
                ring_offsets=d_ring_offsets,
            )

        # Allocate output coordinate arrays
        device_x = runtime.allocate((total_verts,), np.float64)
        device_y = runtime.allocate((total_verts,), np.float64)

        # Pass 2: scatter vertices
        scatter_params = (
            (
                ptr(state.family_row_offsets),
                ptr(line_buf.geometry_offsets),
                ptr(line_buf.empty_mask),
                ptr(line_buf.x),
                ptr(line_buf.y),
                ptr(device_radii),
                quad_segs,
                join_style,
                cap_style,
                mitre_limit,
                ptr(device_offsets),
                ptr(device_x),
                ptr(device_y),
                lines.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(kernels["linestring_buffer_scatter"], lines.row_count)
        runtime.launch(kernels["linestring_buffer_scatter"],
                       grid=scatter_grid, block=scatter_block, params=scatter_params)

        d_geometry_offsets = cp.arange(lines.row_count + 1, dtype=cp.int32)
        d_ring_offsets = cp.empty(lines.row_count + 1, dtype=cp.int32)
        d_ring_offsets[0] = 0
        d_ring_offsets[1:] = (
            cp.asarray(device_offsets)[: lines.row_count]
            + cp.asarray(device_counts)[: lines.row_count]
        )

        success = True
        return _build_device_backed_polygon_output_variable(
            device_x, device_y,
            row_count=lines.row_count,
            geometry_offsets=d_geometry_offsets,
            ring_offsets=d_ring_offsets,
        )
    finally:
        runtime.free(device_radii)
        runtime.free(device_counts)
        if device_offsets is not None:
            runtime.free(device_offsets)
        if not success:
            if device_x is not None:
                runtime.free(device_x)
            if device_y is not None:
                runtime.free(device_y)
