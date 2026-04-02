"""GPU-accelerated segmentize: split long segments by interpolation.

Segmentize splits geometry segments longer than ``max_segment_length`` by
inserting linearly-interpolated intermediate points so that every output
segment is at most ``max_segment_length`` long.

Points/MultiPoints pass through unchanged.
For LineString/MultiLineString: segmentize coordinate sequences.
For Polygon/MultiPolygon: segmentize each ring, preserving closure.

ADR-0033: Tier 1 NVRTC kernel (count + scatter), Tier 3a CCCL (exclusive_sum).
ADR-0002: COARSE class -- segment length threshold is user-specified;
          constructive stays fp64 per ADR-0002.
"""

from __future__ import annotations

import math

import numpy as np

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    build_updated_device_family_buffer,
    build_updated_host_family_buffer,
    forward_result_metadata,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.segmentize_kernels import (
    _SEGMENTIZE_COUNT_FP64,
    _SEGMENTIZE_COUNT_KERNEL_NAMES,
    _SEGMENTIZE_SCATTER_FP64,
    _SEGMENTIZE_SCATTER_KERNEL_NAMES,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)

# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup

request_nvrtc_warmup([
    ("segmentize-count-fp64", _SEGMENTIZE_COUNT_FP64, _SEGMENTIZE_COUNT_KERNEL_NAMES),
    ("segmentize-scatter-fp64", _SEGMENTIZE_SCATTER_FP64, _SEGMENTIZE_SCATTER_KERNEL_NAMES),
])

from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402

request_warmup(["exclusive_scan_i32"])


# ---------------------------------------------------------------------------
# GPU implementation: 2-pass count -> scan -> scatter
# ---------------------------------------------------------------------------

def _segmentize_family_gpu(runtime, device_buf, family, max_segment_length):
    """GPU 2-pass segmentize for one family.

    Returns a new :class:`DeviceFamilyGeometryBuffer` with densified
    coordinates and updated span-level offsets.  Higher-level offsets
    (geometry_offsets for Polygon, etc.) are preserved from the input.
    """
    from vibespatial.cuda.cccl_primitives import exclusive_sum

    # 1. Determine span offsets for this family
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        d_span_offsets = device_buf.ring_offsets
    elif family is GeometryFamily.MULTILINESTRING:
        d_span_offsets = device_buf.part_offsets
    else:  # LineString
        d_span_offsets = device_buf.geometry_offsets

    if d_span_offsets is None:
        return device_buf

    span_count = int(d_span_offsets.shape[0]) - 1
    if span_count <= 0:
        return device_buf

    # 2. Pass 1: count kernel (1 thread per span)
    d_counts = runtime.allocate((span_count,), np.int32, zero=True)

    count_kernels = compile_kernel_group(
        "segmentize-count-fp64",
        _SEGMENTIZE_COUNT_FP64,
        _SEGMENTIZE_COUNT_KERNEL_NAMES,
    )
    count_kernel = count_kernels["segmentize_count"]

    ptr = runtime.pointer
    count_params = (
        (
            ptr(device_buf.x),
            ptr(device_buf.y),
            ptr(d_span_offsets),
            ptr(d_counts),
            float(max_segment_length),
            span_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_I32,
        ),
    )
    count_grid, count_block = runtime.launch_config(count_kernel, span_count)
    runtime.launch(
        count_kernel, grid=count_grid, block=count_block, params=count_params,
    )

    # 3. Scan: exclusive prefix sum (CCCL Tier 3a)
    d_offsets = exclusive_sum(d_counts, synchronize=False)

    # 4. Read total output size (single async pinned transfer)
    total_out = count_scatter_total(runtime, d_counts, d_offsets)

    if total_out == 0:
        # Degenerate: all spans empty.  Return a zero-coordinate buffer.
        d_x_out = runtime.allocate((0,), np.float64)
        d_y_out = runtime.allocate((0,), np.float64)
        d_out_offsets_full = runtime.allocate((span_count + 1,), np.int32, zero=True)
    else:
        # 5. Build full offsets array (length span_count + 1) on device
        d_out_offsets_full = runtime.allocate((span_count + 1,), np.int32)
        cp.copyto(
            cp.asarray(d_out_offsets_full[:span_count]),
            cp.asarray(d_offsets),
        )
        d_out_offsets_full[span_count] = total_out

        # Allocate output coordinate buffers
        d_x_out = runtime.allocate((total_out,), np.float64)
        d_y_out = runtime.allocate((total_out,), np.float64)

        # 6. Pass 2: scatter kernel (1 thread per output coordinate)
        scatter_kernels = compile_kernel_group(
            "segmentize-scatter-fp64",
            _SEGMENTIZE_SCATTER_FP64,
            _SEGMENTIZE_SCATTER_KERNEL_NAMES,
        )
        scatter_kernel = scatter_kernels["segmentize_scatter"]

        scatter_params = (
            (
                ptr(device_buf.x),
                ptr(device_buf.y),
                ptr(d_span_offsets),
                ptr(d_out_offsets_full),
                ptr(d_x_out),
                ptr(d_y_out),
                float(max_segment_length),
                span_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(
            scatter_kernel, total_out,
        )
        runtime.launch(
            scatter_kernel,
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )

    return build_updated_device_family_buffer(
        family,
        device_buf,
        d_x_out,
        d_y_out,
        d_out_offsets_full,
    )


@register_kernel_variant(
    "geometry_segmentize",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "linestring", "multilinestring", "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "segmentize"),
)
def _segmentize_gpu(owned, max_segment_length):
    """GPU segmentize -- returns device-resident OwnedGeometryArray."""
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()

    new_device_families = {}
    for family, device_buf in d_state.families.items():
        if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
            new_device_families[family] = device_buf
            continue
        new_device_families[family] = _segmentize_family_gpu(
            runtime, device_buf, family, max_segment_length,
        )

    tags, validity, family_row_offsets = forward_result_metadata(owned)

    return build_device_resident_owned(
        device_families=new_device_families,
        row_count=owned.row_count,
        tags=tags,
        validity=validity,
        family_row_offsets=family_row_offsets,
        execution_mode="gpu",
    )


# ---------------------------------------------------------------------------
# CPU segmentize helpers
# ---------------------------------------------------------------------------

def _segmentize_span(x, y, start, end, max_segment_length):
    """Segmentize one coordinate span on CPU.

    For each segment (pair of consecutive coordinates), if the segment length
    exceeds *max_segment_length*, interpolate intermediate points so that every
    output sub-segment is at most *max_segment_length* long.

    Returns (x_out, y_out) arrays of new coordinates.
    """
    n = end - start
    if n <= 1:
        return x[start:end].copy(), y[start:end].copy()

    out_x_parts: list[np.ndarray] = []
    out_y_parts: list[np.ndarray] = []

    for i in range(start, end - 1):
        x0, y0 = x[i], y[i]
        x1, y1 = x[i + 1], y[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.sqrt(dx * dx + dy * dy)

        splits = max(1, math.ceil(seg_len / max_segment_length))

        # Emit splits points for this segment (excludes the far endpoint
        # which will be the start of the next segment, or the final point).
        t = np.arange(splits, dtype=np.float64) / splits
        out_x_parts.append(x0 + t * dx)
        out_y_parts.append(y0 + t * dy)

    # Append the final endpoint
    out_x_parts.append(np.array([x[end - 1]], dtype=np.float64))
    out_y_parts.append(np.array([y[end - 1]], dtype=np.float64))

    return np.concatenate(out_x_parts), np.concatenate(out_y_parts)


# ---------------------------------------------------------------------------
# CPU fallback: segmentize per family
# ---------------------------------------------------------------------------

def _segmentize_family_cpu(buf, family, max_segment_length):
    """Segmentize one family's geometries on CPU.

    Points and MultiPoints pass through unchanged.  For LineString,
    MultiLineString, Polygon, and MultiPolygon families, segments longer
    than *max_segment_length* are subdivided by linear interpolation.
    """
    if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
        return buf  # Points don't segmentize

    # Determine which offsets define the spans to segmentize
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        span_offsets = buf.ring_offsets
    elif family is GeometryFamily.MULTILINESTRING:
        span_offsets = buf.part_offsets
    else:
        # LineString
        span_offsets = buf.geometry_offsets

    if span_offsets is None:
        return buf

    span_count = len(span_offsets) - 1
    new_x_parts: list[np.ndarray] = []
    new_y_parts: list[np.ndarray] = []
    new_span_offsets = [0]

    for s in range(span_count):
        start = int(span_offsets[s])
        end = int(span_offsets[s + 1])
        sx, sy = _segmentize_span(buf.x, buf.y, start, end, max_segment_length)

        new_x_parts.append(sx)
        new_y_parts.append(sy)
        new_span_offsets.append(new_span_offsets[-1] + len(sx))

    new_x = np.concatenate(new_x_parts) if new_x_parts else np.empty(0, dtype=np.float64)
    new_y = np.concatenate(new_y_parts) if new_y_parts else np.empty(0, dtype=np.float64)
    new_span_offsets_arr = np.asarray(new_span_offsets, dtype=np.int32)

    # Rebuild the family buffer with new coordinates and updated offsets
    return build_updated_host_family_buffer(
        family=family,
        host_buf=buf,
        x_out=new_x,
        y_out=new_y,
        new_offsets=new_span_offsets_arr,
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def segmentize_owned(
    owned: OwnedGeometryArray,
    max_segment_length: float,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Segmentize geometries: split segments exceeding *max_segment_length*.

    Inserts linearly-interpolated intermediate points so that no output
    segment exceeds the given length.

    Parameters
    ----------
    max_segment_length : float
        Maximum allowed segment length.  Segments longer than this are
        subdivided.  Must be positive.
    dispatch_mode : ExecutionMode
        Execution mode selection (AUTO / CPU / GPU).

    Returns
    -------
    OwnedGeometryArray
        New geometry array with densified coordinates.
    """
    row_count = owned.row_count
    if row_count == 0:
        return owned

    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive")

    selection = plan_dispatch_selection(
        kernel_name="geometry_segmentize",
        kernel_class=KernelClass.COARSE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        return _segmentize_gpu(owned, max_segment_length)

    new_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    for family, buf in owned.families.items():
        if buf.row_count == 0:
            new_families[family] = buf
            continue
        new_families[family] = _segmentize_family_cpu(buf, family, max_segment_length)

    return OwnedGeometryArray(
        validity=owned.validity.copy(),
        tags=owned.tags.copy(),
        family_row_offsets=owned.family_row_offsets.copy(),
        families=new_families,
        residency=Residency.HOST,
    )
