from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from vibespatial.constructive.clip_rect_cpu import (
    EMPTY,
    benchmark_clip_by_rect_baseline,
    clip_by_rect_array,
    reconstruct_polygon_result_from_rings,
)
from vibespatial.constructive.clip_rect_cpu import (
    build_linestring_result as _build_linestring_result,
)
from vibespatial.constructive.clip_rect_cpu import (
    clip_by_rect_cpu as _clip_by_rect_cpu,
)
from vibespatial.constructive.clip_rect_cpu import (
    merge_clipped_segments as _merge_clipped_segments,
)
from vibespatial.constructive.clip_rect_cpu import (
    normalize_values as _normalize_values,
)
from vibespatial.constructive.clip_rect_cpu import (
    point_clip_result_template as _point_clip_result_template,
)
from vibespatial.constructive.clip_rect_cpu import (
    polygon_ring_spans as _polygon_ring_spans,
)
from vibespatial.constructive.clip_rect_kernels import (
    _LB_KERNEL_NAMES,
    _LIANG_BARSKY_KERNEL_SOURCE,
    _LINE_ROW_KERNEL_NAMES,
    _LINE_ROW_KERNEL_SOURCE,
    _SEG_ARANGE_KERNEL_NAMES,
    _SEGMENTED_ARANGE_KERNEL_SOURCE,
    _SH_KERNEL_NAMES,
    _SUTHERLAND_HODGMAN_KERNEL_SOURCE,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    build_null_owned_array,
    concat_owned_scatter,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.config import SPATIAL_EPSILON
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
)
from vibespatial.runtime.residency import Residency, TransferTrigger, combined_residency
from vibespatial.runtime.robustness import RobustnessPlan, select_robustness_plan

from .point import (
    _build_device_backed_point_output,
    _clip_points_rect_gpu_arrays,
    _empty_point_output,
)

request_warmup(["exclusive_scan_i32", "exclusive_scan_i64", "segmented_reduce_sum_i32"])

_POINT_EPSILON = SPATIAL_EPSILON
_POINT_TYPE_ID = 0

# ---------------------------------------------------------------------------
# NVRTC kernel compilation helpers
# ---------------------------------------------------------------------------
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("sh-clip", _SUTHERLAND_HODGMAN_KERNEL_SOURCE, _SH_KERNEL_NAMES),
    ("lb-clip", _LIANG_BARSKY_KERNEL_SOURCE, _LB_KERNEL_NAMES),
    ("line-row-clip", _LINE_ROW_KERNEL_SOURCE, _LINE_ROW_KERNEL_NAMES),
    ("seg-arange", _SEGMENTED_ARANGE_KERNEL_SOURCE, _SEG_ARANGE_KERNEL_NAMES),
])


def _compile_sh_kernels():
    return compile_kernel_group("sh-clip", _SUTHERLAND_HODGMAN_KERNEL_SOURCE, _SH_KERNEL_NAMES)


def _compile_lb_kernels():
    return compile_kernel_group("lb-clip", _LIANG_BARSKY_KERNEL_SOURCE, _LB_KERNEL_NAMES)


def _compile_seg_arange_kernels():
    return compile_kernel_group(
        "seg-arange", _SEGMENTED_ARANGE_KERNEL_SOURCE, _SEG_ARANGE_KERNEL_NAMES,
    )


def _compile_line_row_kernels():
    return compile_kernel_group(
        "line-row-clip", _LINE_ROW_KERNEL_SOURCE, _LINE_ROW_KERNEL_NAMES,
    )


class RectClipResult:
    """Result of a rectangle clip operation.

    ``geometries`` is lazily materialized from ``owned_result`` when accessed
    for the first time on the GPU point path, avoiding D->H->Shapely overhead
    unless a caller actually needs Shapely objects.
    """

    __slots__ = (
        "_candidate_rows",
        "_candidate_rows_factory",
        "_fallback_rows",
        "_fallback_rows_factory",
        "_fast_rows",
        "_fast_rows_factory",
        "_geometries",
        "_geometries_factory",
        "_owned_result_rows",
        "_owned_result_rows_factory",
        "owned_result",
        "precision_plan",
        "robustness_plan",
        "row_count",
        "runtime_selection",
    )

    def __init__(
        self,
        *,
        geometries: np.ndarray | None = None,
        geometries_factory: object | None = None,
        row_count: int,
        candidate_rows: np.ndarray | None = None,
        candidate_rows_factory: object | None = None,
        fast_rows: np.ndarray | None = None,
        fast_rows_factory: object | None = None,
        fallback_rows: np.ndarray | None = None,
        fallback_rows_factory: object | None = None,
        runtime_selection: RuntimeSelection,
        precision_plan: PrecisionPlan,
        robustness_plan: RobustnessPlan,
        owned_result: OwnedGeometryArray | None = None,
        owned_result_rows: np.ndarray | None = None,
        owned_result_rows_factory: object | None = None,
    ):
        self._candidate_rows = candidate_rows
        self._candidate_rows_factory = candidate_rows_factory
        self._fast_rows = fast_rows
        self._fast_rows_factory = fast_rows_factory
        self._fallback_rows = fallback_rows
        self._fallback_rows_factory = fallback_rows_factory
        self._geometries = geometries
        self._geometries_factory = geometries_factory
        self._owned_result_rows = owned_result_rows
        self._owned_result_rows_factory = owned_result_rows_factory
        self.row_count = row_count
        self.runtime_selection = runtime_selection
        self.precision_plan = precision_plan
        self.robustness_plan = robustness_plan
        self.owned_result = owned_result

    @property
    def geometries(self) -> np.ndarray:
        if self._geometries is None and self._geometries_factory is not None:
            self._geometries = self._geometries_factory()
            self._geometries_factory = None
        if self._geometries is None:
            return np.empty(0, dtype=object)
        return self._geometries

    @property
    def candidate_rows(self) -> np.ndarray:
        if self._candidate_rows is None and self._candidate_rows_factory is not None:
            self._candidate_rows = self._candidate_rows_factory()
            self._candidate_rows_factory = None
        if self._candidate_rows is None:
            return np.empty(0, dtype=np.int32)
        return self._candidate_rows

    @property
    def fast_rows(self) -> np.ndarray:
        if self._fast_rows is None and self._fast_rows_factory is not None:
            self._fast_rows = self._fast_rows_factory()
            self._fast_rows_factory = None
        if self._fast_rows is None:
            return np.empty(0, dtype=np.int32)
        return self._fast_rows

    @property
    def fallback_rows(self) -> np.ndarray:
        if self._fallback_rows is None and self._fallback_rows_factory is not None:
            self._fallback_rows = self._fallback_rows_factory()
            self._fallback_rows_factory = None
        if self._fallback_rows is None:
            return np.empty(0, dtype=np.int32)
        return self._fallback_rows

    @property
    def owned_result_rows(self) -> np.ndarray | None:
        if self._owned_result_rows is None and self._owned_result_rows_factory is not None:
            self._owned_result_rows = self._owned_result_rows_factory()
            self._owned_result_rows_factory = None
        return self._owned_result_rows


@dataclass(frozen=True)
class RectClipBenchmark:
    dataset: str
    rows: int
    candidate_rows: int
    fast_rows: int
    fallback_rows: int
    owned_elapsed_seconds: float
    shapely_elapsed_seconds: float

    @property
    def speedup_vs_shapely(self) -> float:
        if self.owned_elapsed_seconds == 0.0:
            return float("inf")
        return self.shapely_elapsed_seconds / self.owned_elapsed_seconds


def _rect_intersects_bounds(bounds: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    xmin, ymin, xmax, ymax = rect
    return (
        (bounds[:, 0] <= xmax)
        & (bounds[:, 2] >= xmin)
        & (bounds[:, 1] <= ymax)
        & (bounds[:, 3] >= ymin)
    )


def _inside_left(point: tuple[float, float], xmin: float) -> bool:
    return point[0] >= xmin


def _inside_right(point: tuple[float, float], xmax: float) -> bool:
    return point[0] <= xmax


def _inside_bottom(point: tuple[float, float], ymin: float) -> bool:
    return point[1] >= ymin


def _inside_top(point: tuple[float, float], ymax: float) -> bool:
    return point[1] <= ymax


def _intersect_vertical(
    p0: tuple[float, float],
    p1: tuple[float, float],
    x: float,
) -> tuple[float, float]:
    x0, y0 = p0
    x1, y1 = p1
    if abs(x1 - x0) <= _POINT_EPSILON:
        return float(x), float(y0)
    t = (x - x0) / (x1 - x0)
    return float(x), float(y0 + t * (y1 - y0))


def _intersect_horizontal(
    p0: tuple[float, float],
    p1: tuple[float, float],
    y: float,
) -> tuple[float, float]:
    x0, y0 = p0
    x1, y1 = p1
    if abs(y1 - y0) <= _POINT_EPSILON:
        return float(x0), float(y)
    t = (y - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0)), float(y)


def _sutherland_hodgman_ring(
    coords: list[tuple[float, float]],
    rect: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    if len(coords) < 3:
        return []
    xmin, ymin, xmax, ymax = rect
    subject = coords[:-1] if coords[0] == coords[-1] else coords[:]
    if not subject:
        return []

    boundaries = (
        (lambda point: _inside_left(point, xmin), lambda a, b: _intersect_vertical(a, b, xmin)),
        (lambda point: _inside_right(point, xmax), lambda a, b: _intersect_vertical(a, b, xmax)),
        (lambda point: _inside_bottom(point, ymin), lambda a, b: _intersect_horizontal(a, b, ymin)),
        (lambda point: _inside_top(point, ymax), lambda a, b: _intersect_horizontal(a, b, ymax)),
    )

    output = subject
    for inside, intersect in boundaries:
        if not output:
            return []
        clipped: list[tuple[float, float]] = []
        previous = output[-1]
        previous_inside = inside(previous)
        for current in output:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    clipped.append(intersect(previous, current))
                clipped.append(current)
            elif previous_inside:
                clipped.append(intersect(previous, current))
            previous = current
            previous_inside = current_inside
        output = clipped

    if not output:
        return []
    deduped: list[tuple[float, float]] = []
    for point in output:
        if deduped and np.allclose(deduped[-1], point, atol=_POINT_EPSILON, rtol=0.0):
            continue
        deduped.append((float(point[0]), float(point[1])))
    if len(deduped) < 3:
        return []
    if not np.allclose(deduped[0], deduped[-1], atol=_POINT_EPSILON, rtol=0.0):
        deduped.append(deduped[0])
    unique = {
        (round(point[0], 12), round(point[1], 12))
        for point in deduped[:-1]
    }
    if len(unique) < 3:
        return []
    return deduped


# ---------------------------------------------------------------------------
# GPU polygon clip (Sutherland-Hodgman via NVRTC)
# ---------------------------------------------------------------------------

def _clip_polygon_rings_gpu(
    ring_x: np.ndarray,
    ring_y: np.ndarray,
    ring_offsets: np.ndarray,
    rect: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip polygon rings on GPU using Sutherland-Hodgman (host I/O).

    Returns (clipped_x, clipped_y, clipped_ring_offsets) as host numpy arrays.
    """
    import cupy as cp

    d_out_x, d_out_y, d_full_offsets = _clip_polygon_rings_gpu_device(
        ring_x, ring_y, ring_offsets, rect,
    )
    if d_out_x is None:
        ring_count = len(ring_offsets) - 1
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.zeros(ring_count + 1, dtype=np.int32),
        )
    return (
        cp.asnumpy(d_out_x),
        cp.asnumpy(d_out_y),
        cp.asnumpy(d_full_offsets),
    )


def _clip_polygon_rings_gpu_device(
    ring_x,
    ring_y,
    ring_offsets,
    rect: tuple[float, float, float, float],
):
    """Clip polygon rings on GPU using Sutherland-Hodgman (device I/O).

    Accepts numpy or CuPy arrays for ring_x, ring_y, ring_offsets.
    Returns (d_out_x, d_out_y, d_full_offsets) as CuPy device arrays,
    or (None, None, None) when all rings are clipped away.
    """
    import cupy as cp

    from vibespatial.cuda._runtime import KERNEL_PARAM_F64, KERNEL_PARAM_I32, KERNEL_PARAM_PTR
    from vibespatial.cuda.cccl_primitives import exclusive_sum

    runtime = get_cuda_runtime()
    xmin, ymin, xmax, ymax = rect

    # Accept both host and device arrays; upload only when needed.
    if isinstance(ring_x, np.ndarray):
        d_ring_x = cp.asarray(np.ascontiguousarray(ring_x, dtype=np.float64))
        d_ring_y = cp.asarray(np.ascontiguousarray(ring_y, dtype=np.float64))
        d_ring_offsets = cp.asarray(np.ascontiguousarray(ring_offsets, dtype=np.int32))
    else:
        d_ring_x = ring_x
        d_ring_y = ring_y
        d_ring_offsets = ring_offsets

    ring_count = int(d_ring_offsets.size) - 1
    if ring_count <= 0:
        return None, None, None

    d_vertex_counts = cp.empty(ring_count, dtype=cp.int32)

    kernels = _compile_sh_kernels()
    ptr = runtime.pointer

    # Pass 1: Count output vertices per ring
    count_params = (
        (
            ptr(d_ring_x),
            ptr(d_ring_y),
            ptr(d_ring_offsets),
            ptr(d_vertex_counts),
            float(xmin), float(ymin), float(xmax), float(ymax),
            ring_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
            KERNEL_PARAM_I32,
        ),
    )
    count_grid, count_block = runtime.launch_config(kernels["sh_count_vertices"], ring_count)
    runtime.launch(
        kernels["sh_count_vertices"],
        grid=count_grid,
        block=count_block,
        params=count_params,
    )

    # Compute output offsets via exclusive_scan
    d_out_offsets = exclusive_sum(d_vertex_counts)

    # Total output size via single-sync async transfer.
    total_verts = count_scatter_total(runtime, d_vertex_counts, d_out_offsets)

    if total_verts == 0:
        return None, None, None

    # Build full output offsets (ring_count + 1) entirely on device.
    d_full_offsets = cp.empty(ring_count + 1, dtype=cp.int32)
    d_full_offsets[:ring_count] = cp.asarray(d_out_offsets)
    d_full_offsets[ring_count] = total_verts

    # Pass 2: Write clipped vertices
    d_out_x = cp.empty(total_verts, dtype=cp.float64)
    d_out_y = cp.empty(total_verts, dtype=cp.float64)

    clip_params = (
        (
            ptr(d_ring_x),
            ptr(d_ring_y),
            ptr(d_ring_offsets),
            ptr(d_full_offsets),
            ptr(d_out_x),
            ptr(d_out_y),
            float(xmin), float(ymin), float(xmax), float(ymax),
            ring_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
            KERNEL_PARAM_I32,
        ),
    )
    clip_grid, clip_block = runtime.launch_config(kernels["sh_clip_rings"], ring_count)
    runtime.launch(
        kernels["sh_clip_rings"],
        grid=clip_grid,
        block=clip_block,
        params=clip_params,
    )

    return d_out_x, d_out_y, d_full_offsets


def _clip_polygon_family_gpu(
    buffer,
    family_row: int,
    rect: tuple[float, float, float, float],
) -> object:
    """Clip a polygon or multipolygon family row using GPU Sutherland-Hodgman."""
    all_rings = _polygon_ring_spans(buffer, family_row)
    if not all_rings:
        return EMPTY

    # Flatten all rings into coordinate arrays with ring offsets
    flat_x: list[float] = []
    flat_y: list[float] = []
    ring_offsets_list: list[int] = [0]
    ring_polygon_map: list[int] = []  # which polygon each ring belongs to
    ring_is_exterior: list[bool] = []  # whether ring is exterior or hole

    for poly_idx, rings in enumerate(all_rings):
        for ring_idx, ring in enumerate(rings):
            flat_x.extend(c[0] for c in ring)
            flat_y.extend(c[1] for c in ring)
            ring_offsets_list.append(len(flat_x))
            ring_polygon_map.append(poly_idx)
            ring_is_exterior.append(ring_idx == 0)

    if not flat_x:
        return EMPTY

    ring_x = np.asarray(flat_x, dtype=np.float64)
    ring_y = np.asarray(flat_y, dtype=np.float64)
    ring_offsets = np.asarray(ring_offsets_list, dtype=np.int32)

    out_x, out_y, out_ring_offsets = _clip_polygon_rings_gpu(ring_x, ring_y, ring_offsets, rect)

    if out_x.size == 0:
        return EMPTY

    return reconstruct_polygon_result_from_rings(
        ring_polygon_map, ring_is_exterior, out_x, out_y, out_ring_offsets,
    )


# ---------------------------------------------------------------------------
# GPU line clip (Liang-Barsky via NVRTC)
# ---------------------------------------------------------------------------

def _clip_line_segments_gpu(
    seg_x0: np.ndarray,
    seg_y0: np.ndarray,
    seg_x1: np.ndarray,
    seg_y1: np.ndarray,
    rect: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clip line segments on GPU using Liang-Barsky (host I/O path).

    Returns (out_x0, out_y0, out_x1, out_y1, valid_mask) as host numpy arrays.
    For the device-resident path, use ``_clip_line_segments_gpu_device`` instead.
    """
    import cupy as cp

    d_out_x0, d_out_y0, d_out_x1, d_out_y1, d_valid = _clip_line_segments_gpu_device(
        seg_x0, seg_y0, seg_x1, seg_y1, rect,
    )
    return (
        cp.asnumpy(d_out_x0),
        cp.asnumpy(d_out_y0),
        cp.asnumpy(d_out_x1),
        cp.asnumpy(d_out_y1),
        cp.asnumpy(d_valid),
    )


def _clip_line_segments_gpu_device(
    seg_x0: np.ndarray,
    seg_y0: np.ndarray,
    seg_x1: np.ndarray,
    seg_y1: np.ndarray,
    rect: tuple[float, float, float, float],
):
    """Clip line segments on GPU using Liang-Barsky (device I/O).

    Accepts host numpy arrays or device-backed arrays for input segments. Returns
    (d_out_x0, d_out_y0, d_out_x1, d_out_y1, d_valid) as CuPy device
    arrays.  No D2H transfer; kernel outputs stay on device.
    """
    import cupy as cp

    from vibespatial.cuda._runtime import KERNEL_PARAM_F64, KERNEL_PARAM_I32, KERNEL_PARAM_PTR

    runtime = get_cuda_runtime()
    xmin, ymin, xmax, ymax = rect
    segment_count = len(seg_x0)

    def _to_device_f64(values):
        if hasattr(values, "__cuda_array_interface__"):
            return cp.ascontiguousarray(values, dtype=cp.float64)
        return cp.asarray(np.ascontiguousarray(values, dtype=np.float64))

    # Upload host inputs or preserve existing device residency.
    d_x0 = _to_device_f64(seg_x0)
    d_y0 = _to_device_f64(seg_y0)
    d_x1 = _to_device_f64(seg_x1)
    d_y1 = _to_device_f64(seg_y1)

    # Allocate output buffers on device via CuPy
    d_out_x0 = cp.empty(segment_count, dtype=cp.float64)
    d_out_y0 = cp.empty(segment_count, dtype=cp.float64)
    d_out_x1 = cp.empty(segment_count, dtype=cp.float64)
    d_out_y1 = cp.empty(segment_count, dtype=cp.float64)
    d_valid = cp.empty(segment_count, dtype=cp.uint8)

    kernels = _compile_lb_kernels()
    ptr = runtime.pointer

    params = (
        (
            ptr(d_x0), ptr(d_y0), ptr(d_x1), ptr(d_y1),
            ptr(d_out_x0), ptr(d_out_y0), ptr(d_out_x1), ptr(d_out_y1),
            ptr(d_valid),
            float(xmin), float(ymin), float(xmax), float(ymax),
            segment_count,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernels["lb_clip_segments"], segment_count)
    runtime.launch(
        kernels["lb_clip_segments"],
        grid=grid,
        block=block,
        params=params,
    )
    # No explicit sync needed — downstream CuPy/CCCL ops on the null stream
    # will respect ordering automatically via stream-ordered dependencies.
    return d_out_x0, d_out_y0, d_out_x1, d_out_y1, d_valid


def _clip_line_family_gpu(
    buffer,
    family_row: int,
    rect: tuple[float, float, float, float],
) -> object:
    """Clip a linestring or multilinestring family row using GPU Liang-Barsky."""
    if buffer.family.value == "linestring":
        spans = [(int(buffer.geometry_offsets[family_row]), int(buffer.geometry_offsets[family_row + 1]))]
    else:
        part_start = int(buffer.geometry_offsets[family_row])
        part_end = int(buffer.geometry_offsets[family_row + 1])
        spans = [
            (int(buffer.part_offsets[index]), int(buffer.part_offsets[index + 1]))
            for index in range(part_start, part_end)
        ]

    # Collect all segments across all parts
    all_x0: list[float] = []
    all_y0: list[float] = []
    all_x1: list[float] = []
    all_y1: list[float] = []
    span_segment_counts: list[int] = []

    for start, end in spans:
        count = max(0, end - start - 1)
        span_segment_counts.append(count)
        for index in range(start, end - 1):
            all_x0.append(float(buffer.x[index]))
            all_y0.append(float(buffer.y[index]))
            all_x1.append(float(buffer.x[index + 1]))
            all_y1.append(float(buffer.y[index + 1]))

    total_segments = len(all_x0)
    if total_segments == 0:
        return EMPTY

    seg_x0 = np.asarray(all_x0, dtype=np.float64)
    seg_y0 = np.asarray(all_y0, dtype=np.float64)
    seg_x1 = np.asarray(all_x1, dtype=np.float64)
    seg_y1 = np.asarray(all_y1, dtype=np.float64)

    out_x0, out_y0, out_x1, out_y1, valid = _clip_line_segments_gpu(
        seg_x0, seg_y0, seg_x1, seg_y1, rect,
    )

    # Reassemble clipped segments into linestring parts
    parts: list[list[tuple[float, float]]] = []
    seg_idx = 0
    for span_count in span_segment_counts:
        part_segments: list[list[tuple[float, float]]] = []
        for _ in range(span_count):
            if valid[seg_idx]:
                segment = (
                    float(out_x0[seg_idx]),
                    float(out_y0[seg_idx]),
                    float(out_x1[seg_idx]),
                    float(out_y1[seg_idx]),
                )
                _merge_clipped_segments(part_segments, segment)
            seg_idx += 1
        parts.extend(part_segments)

    return _build_linestring_result(parts)


# ---------------------------------------------------------------------------
# Batched GPU line clip — vectorized segment extraction + reassembly
# ---------------------------------------------------------------------------


def _extract_segments_vectorized(
    owned: OwnedGeometryArray,
    line_families: list[GeometryFamily],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized extraction of segments from line-family buffers.

    Uses numpy offset arithmetic and fancy indexing on contiguous
    buffer.x / buffer.y arrays.  No per-row Python loops -- all span
    computation and coordinate gathering are bulk numpy operations.

    Returns
    -------
    seg_x0, seg_y0, seg_x1, seg_y1 : flat segment endpoint arrays (float64)
    row_segment_offsets : cumulative segment count per row (int32)
    segment_global_rows : global row id for each segment (int32)
    segment_part_break_after : bool mask where segment i terminates its source part
    """
    # Accumulate per-family vectorized results, then concatenate once.
    all_span_starts: list[np.ndarray] = []
    all_span_ends: list[np.ndarray] = []
    all_span_family_idx: list[np.ndarray] = []
    all_part_row_map: list[np.ndarray] = []
    global_row_indices_parts: list[np.ndarray] = []
    family_buffers: list[object] = []
    device_family_buffers: list[DeviceFamilyGeometryBuffer | None] = []
    row_base = 0  # running offset into global_row_indices

    for family in line_families:
        owned._ensure_host_family_structure(family)
        buffer = owned.families[family]
        tag = FAMILY_TAGS[family]
        family_rows = np.flatnonzero(owned.tags == tag)
        fam_idx = len(family_buffers)
        family_buffers.append(buffer)
        device_family_buffer = None
        if owned.device_state is not None and family in owned.device_state.families:
            device_family_buffer = owned.device_state.families[family]
        device_family_buffers.append(device_family_buffer)

        n_fam_rows = len(family_rows)
        if n_fam_rows == 0:
            continue

        if buffer.row_count == 0 or buffer.geometry_offsets is None or len(buffer.geometry_offsets) < 2:
            # Empty buffer -- register rows but produce no spans.
            global_row_indices_parts.append(family_rows)
            row_base += n_fam_rows
            continue

        # Register all family rows in global_row_indices.
        global_row_indices_parts.append(family_rows)

        # Map global rows -> local buffer rows (vectorized).
        local_rows = owned.family_row_offsets[family_rows].astype(np.intp, copy=False)

        # Filter out empty geometries (vectorized).
        if buffer.empty_mask.size > 0:
            safe_local = np.minimum(local_rows, buffer.empty_mask.size - 1)
            non_empty_mask = ~buffer.empty_mask[safe_local] | (local_rows >= buffer.empty_mask.size)
        else:
            non_empty_mask = np.ones(n_fam_rows, dtype=bool)

        if family == GeometryFamily.LINESTRING:
            # Each non-empty row -> one span from geometry_offsets.
            ne_idx = np.flatnonzero(non_empty_mask)
            if len(ne_idx) == 0:
                row_base += n_fam_rows
                continue
            ne_local = local_rows[ne_idx]
            s = buffer.geometry_offsets[ne_local]
            e = buffer.geometry_offsets[ne_local + 1]
            # Filter spans with >= 2 coordinates.
            valid_span = (e - s) >= 2
            keep = np.flatnonzero(valid_span)
            if len(keep) == 0:
                row_base += n_fam_rows
                continue
            all_span_starts.append(np.asarray(s[keep], dtype=np.int64))
            all_span_ends.append(np.asarray(e[keep], dtype=np.int64))
            all_span_family_idx.append(np.full(len(keep), fam_idx, dtype=np.int32))
            # Row indices are positions within the global_row_indices array.
            all_part_row_map.append((row_base + ne_idx[keep]).astype(np.int32))
        else:
            # MultiLineString: each non-empty row -> multiple spans via
            # geometry_offsets -> part_offsets indirection.
            ne_idx = np.flatnonzero(non_empty_mask)
            if len(ne_idx) == 0:
                row_base += n_fam_rows
                continue
            ne_local = local_rows[ne_idx]
            part_start = buffer.geometry_offsets[ne_local]
            part_end = buffer.geometry_offsets[ne_local + 1]
            part_counts = (part_end - part_start).astype(np.intp)
            total_parts = int(np.sum(part_counts))
            if total_parts == 0:
                row_base += n_fam_rows
                continue
            # Expand row indices: one entry per part within each row.
            row_idx_expanded = np.repeat(row_base + ne_idx, part_counts).astype(np.int32)
            # Build flat array of all part indices.
            # For each row, part indices are [part_start, ..., part_end - 1].
            part_offsets_cum = np.empty(len(ne_idx) + 1, dtype=np.int64)
            part_offsets_cum[0] = 0
            np.cumsum(part_counts, out=part_offsets_cum[1:])
            # Flat part indices via broadcasting with offset correction.
            flat_part_idx = np.repeat(part_start, part_counts) + (
                np.arange(total_parts, dtype=np.int64)
                - np.repeat(part_offsets_cum[:-1], part_counts)
            )
            s = buffer.part_offsets[flat_part_idx]
            e = buffer.part_offsets[flat_part_idx + 1]
            valid_span = (e - s) >= 2
            keep = np.flatnonzero(valid_span)
            if len(keep) == 0:
                row_base += n_fam_rows
                continue
            all_span_starts.append(np.asarray(s[keep], dtype=np.int64))
            all_span_ends.append(np.asarray(e[keep], dtype=np.int64))
            all_span_family_idx.append(np.full(len(keep), fam_idx, dtype=np.int32))
            all_part_row_map.append(row_idx_expanded[keep])

        row_base += n_fam_rows

    # Merge global_row_indices from all families.
    if global_row_indices_parts:
        global_row_indices_arr = np.concatenate(global_row_indices_parts)
    else:
        global_row_indices_arr = np.empty(0, dtype=np.intp)
    global_row_indices = global_row_indices_arr.astype(np.int32, copy=False)
    n_rows = global_row_indices.shape[0]

    if not all_span_starts:
        empty = np.empty(0, dtype=np.float64)
        return (
            empty, empty, empty, empty,
            np.zeros(n_rows + 1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=bool),
        )

    # Concatenate all per-family results into flat arrays.
    starts_arr = np.concatenate(all_span_starts)
    ends_arr = np.concatenate(all_span_ends)
    span_family_idx = np.concatenate(all_span_family_idx)
    part_row_map_arr = np.concatenate(all_part_row_map)

    # Compute segment counts per span: (end - start - 1).
    seg_counts = (ends_arr - starts_arr - 1).astype(np.int32)
    total_segments = int(np.sum(seg_counts))

    if total_segments == 0:
        empty = np.empty(0, dtype=np.float64)
        return (
            empty, empty, empty, empty,
            np.zeros(n_rows + 1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=bool),
        )

    # Build per-part segment offsets (cumulative).
    part_segment_offsets = np.empty(len(seg_counts) + 1, dtype=np.int32)
    part_segment_offsets[0] = 0
    np.cumsum(seg_counts, out=part_segment_offsets[1:])

    # Build flat gather indices for all spans at once.
    # For each span with k = end - start - 1 segments, we need indices
    # [start, start+1, ..., start+k-1] for the p0 endpoint and
    # [start+1, ..., start+k] for the p1 endpoint.
    # Use np.repeat(starts, seg_counts) + within-span offsets.
    within_offsets = np.arange(total_segments, dtype=np.int64)
    span_base = np.repeat(part_segment_offsets[:-1], seg_counts).astype(np.int64)
    local_offsets = within_offsets - span_base
    gather_p0 = np.repeat(starts_arr, seg_counts) + local_offsets
    gather_p1 = gather_p0 + 1

    unique_fam = np.unique(span_family_idx)
    use_device_gather = any(
        device_family_buffers[int(fi)] is not None
        for fi in unique_fam
    )

    if use_device_gather:
        import cupy as cp

        seg_x0 = cp.empty(total_segments, dtype=cp.float64)
        seg_y0 = cp.empty(total_segments, dtype=cp.float64)
        seg_x1 = cp.empty(total_segments, dtype=cp.float64)
        seg_y1 = cp.empty(total_segments, dtype=cp.float64)
    else:
        seg_x0 = np.empty(total_segments, dtype=np.float64)
        seg_y0 = np.empty(total_segments, dtype=np.float64)
        seg_x1 = np.empty(total_segments, dtype=np.float64)
        seg_y1 = np.empty(total_segments, dtype=np.float64)

    # Gather coordinates. When all spans come from a single family
    # buffer (common case), do a single fancy-index gather. Otherwise
    # gather per-family and scatter into the output.
    if len(unique_fam) == 1:
        # Fast path: single family buffer.
        fam_idx = int(unique_fam[0])
        buf = family_buffers[fam_idx]
        dev_buf = device_family_buffers[fam_idx]
        if dev_buf is not None:
            d_gather_p0 = cp.asarray(gather_p0.astype(np.int64, copy=False))
            d_gather_p1 = cp.asarray(gather_p1.astype(np.int64, copy=False))
            seg_x0 = cp.asarray(dev_buf.x)[d_gather_p0]
            seg_y0 = cp.asarray(dev_buf.y)[d_gather_p0]
            seg_x1 = cp.asarray(dev_buf.x)[d_gather_p1]
            seg_y1 = cp.asarray(dev_buf.y)[d_gather_p1]
        else:
            seg_x0[:] = np.asarray(buf.x, dtype=np.float64)[gather_p0]
            seg_y0[:] = np.asarray(buf.y, dtype=np.float64)[gather_p0]
            seg_x1[:] = np.asarray(buf.x, dtype=np.float64)[gather_p1]
            seg_y1[:] = np.asarray(buf.y, dtype=np.float64)[gather_p1]
    else:
        # Multi-family: gather per family using segment-level mask.
        seg_fam_idx = np.repeat(span_family_idx, seg_counts)
        for fi in unique_fam:
            fi_int = int(fi)
            buf = family_buffers[fi_int]
            dev_buf = device_family_buffers[fi_int]
            positions = np.flatnonzero(seg_fam_idx == fi_int)
            idx0 = gather_p0[positions]
            idx1 = gather_p1[positions]
            if use_device_gather:
                d_positions = cp.asarray(positions.astype(np.int64, copy=False))
                if dev_buf is not None and not buf.host_materialized:
                    d_idx0 = cp.asarray(idx0.astype(np.int64, copy=False))
                    d_idx1 = cp.asarray(idx1.astype(np.int64, copy=False))
                    seg_x0[d_positions] = cp.asarray(dev_buf.x)[d_idx0]
                    seg_y0[d_positions] = cp.asarray(dev_buf.y)[d_idx0]
                    seg_x1[d_positions] = cp.asarray(dev_buf.x)[d_idx1]
                    seg_y1[d_positions] = cp.asarray(dev_buf.y)[d_idx1]
                else:
                    seg_x0[d_positions] = cp.asarray(np.asarray(buf.x, dtype=np.float64)[idx0])
                    seg_y0[d_positions] = cp.asarray(np.asarray(buf.y, dtype=np.float64)[idx0])
                    seg_x1[d_positions] = cp.asarray(np.asarray(buf.x, dtype=np.float64)[idx1])
                    seg_y1[d_positions] = cp.asarray(np.asarray(buf.y, dtype=np.float64)[idx1])
            else:
                seg_x0[positions] = np.asarray(buf.x, dtype=np.float64)[idx0]
                seg_y0[positions] = np.asarray(buf.y, dtype=np.float64)[idx0]
                seg_x1[positions] = np.asarray(buf.x, dtype=np.float64)[idx1]
                seg_y1[positions] = np.asarray(buf.y, dtype=np.float64)[idx1]

    row_segment_offsets = np.zeros(n_rows + 1, dtype=np.int32)
    if seg_counts.size > 0:
        row_segment_offsets[1:] = np.bincount(
            part_row_map_arr,
            weights=seg_counts.astype(np.float64),
            minlength=n_rows,
        ).astype(np.int32)
    np.cumsum(row_segment_offsets, out=row_segment_offsets)

    # Build per-segment row ids and part-boundary flags once on host so the
    # GPU assembly path can avoid device-side searchsorted over part metadata.
    segment_global_rows = np.repeat(
        global_row_indices[part_row_map_arr],
        seg_counts,
    ).astype(np.int32, copy=False)
    segment_part_break_after = np.zeros(total_segments, dtype=bool)
    segment_part_break_after[part_segment_offsets[1:] - 1] = True

    return (
        seg_x0, seg_y0, seg_x1, seg_y1,
        row_segment_offsets,
        segment_global_rows,
        segment_part_break_after,
    )


def _build_segmented_gather_indices_device(
    d_offsets,
    d_selected_indices,
):
    """Build gather indices for selected offset ranges with segmented_arange."""
    import cupy as cp

    from vibespatial.cuda.cccl_primitives import exclusive_sum

    selected_count = int(d_selected_indices.shape[0])
    if selected_count == 0:
        return None

    runtime = get_cuda_runtime()
    d_selected_starts = d_offsets[d_selected_indices].astype(cp.int64, copy=False)
    d_selected_ends = d_offsets[d_selected_indices + 1].astype(cp.int64, copy=False)
    d_selected_lengths_i64 = (d_selected_ends - d_selected_starts).astype(cp.int64, copy=False)
    d_write_offsets = exclusive_sum(d_selected_lengths_i64, synchronize=False)
    total_selected_coords = count_scatter_total(runtime, d_selected_lengths_i64, d_write_offsets)

    d_new_offsets = cp.empty(selected_count + 1, dtype=cp.int32)
    d_new_offsets[:selected_count] = d_write_offsets.astype(cp.int32, copy=False)
    d_new_offsets[selected_count] = total_selected_coords

    if total_selected_coords == 0:
        return (
            cp.empty(0, dtype=cp.int64),
            d_new_offsets,
        )

    d_gather = cp.empty(total_selected_coords, dtype=cp.int64)
    kernels = _compile_seg_arange_kernels()
    ptr = runtime.pointer
    seg_params = (
        (
            ptr(d_selected_starts),
            ptr(d_selected_lengths_i64),
            ptr(d_write_offsets),
            ptr(d_gather),
            selected_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    seg_grid, seg_block = runtime.launch_config(kernels["segmented_arange"], selected_count)
    runtime.launch(
        kernels["segmented_arange"],
        grid=seg_grid,
        block=seg_block,
        params=seg_params,
    )
    return d_gather, d_new_offsets


def _gather_selected_runs_device(
    d_flat_x,
    d_flat_y,
    d_geom_offsets,
    d_selected_run_indices,
):
    """Gather selected line runs with the segmented-arange GPU helper."""
    segmented = _build_segmented_gather_indices_device(
        d_geom_offsets,
        d_selected_run_indices,
    )
    if segmented is None:
        return None
    d_gather, d_new_offsets = segmented
    return (
        d_flat_x[d_gather],
        d_flat_y[d_gather],
        d_new_offsets,
    )


def _build_line_clip_device_result(
    d_out_x0,
    d_out_y0,
    d_out_x1,
    d_out_y1,
    d_valid,
    segment_global_rows: np.ndarray,
    segment_part_break_after: np.ndarray,
) -> tuple[OwnedGeometryArray | None, object | None]:
    """Build a device-resident OwnedGeometryArray from GPU-clipped line segments.

    TRUE ZERO-COPY implementation: all coordinate assembly uses CuPy (Tier 2)
    and CCCL exclusive_sum (Tier 3a).  No numpy, no host round-trips in the
    hot path.  Break detection, coordinate gathering, and offset construction
    all happen on device.

    Parameters
    ----------
    d_out_x0, d_out_y0, d_out_x1, d_out_y1 : CuPy device arrays
        Clipped segment endpoints from the Liang-Barsky kernel.
    d_valid : CuPy device array (uint8)
        Per-segment validity mask from the kernel.
    segment_global_rows : host numpy array (int32)
        Global row id for each original segment before clip compaction.
    segment_part_break_after : host numpy array (bool)
        True where segment i is the last segment of its source part.

    Returns
    -------
    (OwnedGeometryArray, global_row_map) or (None, None).
    global_row_map is an int32 row map aligned to the compact output rows.
    It stays on device until a host consumer asks for scatter/materialization.
    """
    import cupy as cp

    from vibespatial.cuda.cccl_primitives import exclusive_sum

    n_total = int(d_valid.shape[0])
    if n_total == 0:
        return None, None

    # ---------------------------------------------------------------
    # 1. Filter valid segments (CuPy flatnonzero -- ADR-0033 Tier 2,
    #    compaction is CuPy-default per 2026-03-17 decision)
    # ---------------------------------------------------------------
    d_valid_indices = cp.flatnonzero(d_valid)
    n_valid = int(d_valid_indices.shape[0])
    if n_valid == 0:
        return None, None

    # Gather valid segment endpoints on device
    d_vx0 = d_out_x0[d_valid_indices]
    d_vy0 = d_out_y0[d_valid_indices]
    d_vx1 = d_out_x1[d_valid_indices]
    d_vy1 = d_out_y1[d_valid_indices]

    # ---------------------------------------------------------------
    # 2. Detect breaks between consecutive segments on device.
    #    A break occurs where end[i] != start[i+1] (within epsilon)
    #    OR where we cross a part boundary.
    #    Uses cp.abs -- CuPy Tier 2 element-wise ops, zero host trips.
    # ---------------------------------------------------------------

    # Upload direct per-segment metadata once. This avoids device-side
    # searchsorted over part offsets in the hot assembly path.
    d_segment_global_rows = cp.asarray(segment_global_rows)
    d_segment_part_break_after = cp.asarray(segment_part_break_after)
    d_valid_global_rows = d_segment_global_rows[d_valid_indices]

    # Build break mask: True at position i means a new linestring starts
    # at valid segment i.  Position 0 always starts a new line.
    if n_valid == 1:
        # Single valid segment: one linestring with 2 coords
        d_flat_x = cp.array([d_vx0[0], d_vx1[0]], dtype=cp.float64)
        d_flat_y = cp.array([d_vy0[0], d_vy1[0]], dtype=cp.float64)
        d_geom_offsets = cp.array([0, 2], dtype=cp.int32)
        line_count = 1
        d_run_starts = cp.zeros(1, dtype=cp.int64)
    else:
        # Continuity break: end[i] != start[i+1]
        d_dx = cp.abs(d_vx1[:-1] - d_vx0[1:])
        d_dy = cp.abs(d_vy1[:-1] - d_vy0[1:])
        d_coord_break = (d_dx > _POINT_EPSILON) | (d_dy > _POINT_EPSILON)

        # Part boundary break: the prior valid segment terminates its part.
        d_part_break = d_segment_part_break_after[d_valid_indices[:-1]]

        # Combined breaks (either discontinuity or part boundary)
        d_breaks = d_coord_break | d_part_break  # bool, length n_valid-1

        # ---------------------------------------------------------------
        # 3. Build coordinate arrays on device.
        #
        #    For a run of consecutive segments [s, s+1, ..., e-1], the
        #    linestring coords are: x0[s], (x0[s+1]..x0[e-1],) x1[e-1]
        #    i.e. all start x-coords in the run, plus the end x of the
        #    last segment.
        #
        #    Strategy: every valid segment contributes its start coord.
        #    At each break position (and at the end), the preceding
        #    segment also contributes its end coord.
        #
        #    This is done with CuPy scatter/gather (Tier 2) and
        #    exclusive_sum (CCCL Tier 3a) for offset computation.
        # ---------------------------------------------------------------

        # Compute per-segment coord counts:
        # - Each segment contributes 1 (its start coord)
        # - The last segment before each break contributes +1 (its end coord)
        # - The very last valid segment contributes +1 (its end coord)
        d_seg_coords = cp.ones(n_valid, dtype=cp.int32)
        # Break indices: where d_breaks is True => segment i is last
        # before a break, so it contributes an extra end coord.
        d_break_indices = cp.flatnonzero(d_breaks)
        if d_break_indices.shape[0] > 0:
            d_seg_coords[d_break_indices] += 1
        # Last valid segment always contributes end coord
        d_seg_coords[n_valid - 1] = d_seg_coords[n_valid - 1] + 1

        # Prefix sum to get scatter positions
        d_coord_offsets = exclusive_sum(d_seg_coords, synchronize=False)
        # Single device expression → single int() sync (not two separate reads)
        d_last_offset = d_coord_offsets[n_valid - 1]
        d_last_count = d_seg_coords[n_valid - 1]
        total_coords = int(d_last_offset + d_last_count)

        # Allocate flat coordinate arrays
        d_flat_x = cp.empty(total_coords, dtype=cp.float64)
        d_flat_y = cp.empty(total_coords, dtype=cp.float64)

        # Scatter start coords: every segment writes at coord_offsets[i]
        d_flat_x[d_coord_offsets] = d_vx0
        d_flat_y[d_coord_offsets] = d_vy0

        # Scatter end coords at break positions and at the tail.
        # A segment at index i writes its end coord at coord_offsets[i]+1
        # only when it is the last in its run.
        if d_break_indices.shape[0] > 0:
            d_end_positions = d_coord_offsets[d_break_indices] + 1
            d_flat_x[d_end_positions] = d_vx1[d_break_indices]
            d_flat_y[d_end_positions] = d_vy1[d_break_indices]
        # Last segment end coord — device indexing (no host sync needed)
        d_flat_x[d_last_offset + 1] = d_vx1[n_valid - 1]
        d_flat_y[d_last_offset + 1] = d_vy1[n_valid - 1]

        # ---------------------------------------------------------------
        # 4. Build geometry_offsets from break positions.
        #    Each run between breaks (inclusive of start) is one linestring.
        #    Line i starts at the coord offset of the first segment in
        #    run i.
        # ---------------------------------------------------------------

        # Run start indices in valid-segment space: [0, break+1, break+1, ...]
        if d_break_indices.shape[0] > 0:
            d_run_starts = cp.concatenate([
                cp.zeros(1, dtype=cp.int64),
                d_break_indices + 1,
            ])
        else:
            d_run_starts = cp.zeros(1, dtype=cp.int64)

        line_count = int(d_run_starts.shape[0])

        # Geometry offsets: coord position of each run start, plus total
        d_geom_start_coords = d_coord_offsets[d_run_starts]
        d_geom_offsets = cp.empty(line_count + 1, dtype=cp.int32)
        d_geom_offsets[:line_count] = d_geom_start_coords.astype(cp.int32)
        d_geom_offsets[line_count] = total_coords

    # ---------------------------------------------------------------
    # 5. Build global row map on device (CuPy Tier 2 gather).
    #    Each run inherits the global row id of its first valid segment.
    # ---------------------------------------------------------------
    d_global_row_map = d_valid_global_rows[d_run_starts]

    # ---------------------------------------------------------------
    # 6. Filter out degenerate linestrings (< 2 coords) on device
    # ---------------------------------------------------------------
    d_geom_lengths = cp.diff(d_geom_offsets)
    d_valid_geom_mask = d_geom_lengths >= 2
    # Single sync for the gate check — unavoidable since it controls branching
    valid_geom_count = int(cp.sum(d_valid_geom_mask))

    if valid_geom_count == 0:
        return None, None

    if valid_geom_count < line_count:
        # Compact: keep only valid geometries -- fully on device
        d_valid_geom_idx = cp.flatnonzero(d_valid_geom_mask)
        compacted = _gather_selected_runs_device(
            d_flat_x,
            d_flat_y,
            d_geom_offsets,
            d_valid_geom_idx.astype(cp.int64, copy=False),
        )
        if compacted is not None:
            d_flat_x, d_flat_y, d_geom_offsets = compacted
        else:
            d_flat_x = cp.empty(0, dtype=cp.float64)
            d_flat_y = cp.empty(0, dtype=cp.float64)
            d_geom_offsets = cp.zeros(1, dtype=cp.int32)
        # Filter the global row map to match compacted geometries
        d_global_row_map = d_global_row_map[d_valid_geom_idx]
        line_count = valid_geom_count

    def _compact_selected_runs(d_selected_run_indices):
        return _gather_selected_runs_device(
            d_flat_x,
            d_flat_y,
            d_geom_offsets,
            d_selected_run_indices.astype(cp.int64, copy=False),
        )

    d_row_breaks = cp.empty(line_count, dtype=cp.bool_)
    d_row_breaks[0] = True
    if line_count > 1:
        d_row_breaks[1:] = d_global_row_map[1:] != d_global_row_map[:-1]
    d_row_starts = cp.flatnonzero(d_row_breaks).astype(cp.int64, copy=False)
    compact_row_count = int(d_row_starts.shape[0])
    d_unique_rows = d_global_row_map[d_row_starts]

    if compact_row_count < line_count:
        d_row_offsets = cp.empty(compact_row_count + 1, dtype=cp.int64)
        d_row_offsets[:compact_row_count] = d_row_starts
        d_row_offsets[compact_row_count] = line_count
        d_row_run_counts = cp.diff(d_row_offsets).astype(cp.int32)
        d_single_row_mask = d_row_run_counts == 1
        d_multi_row_mask = d_row_run_counts > 1

        output_validity = cp.ones(compact_row_count, dtype=cp.bool_)
        d_single_prefix = (
            cp.cumsum(d_single_row_mask.astype(cp.int32)).astype(cp.int32)
            - d_single_row_mask.astype(cp.int32)
        )
        d_multi_prefix = (
            cp.cumsum(d_multi_row_mask.astype(cp.int32)).astype(cp.int32)
            - d_multi_row_mask.astype(cp.int32)
        )
        output_family_row_offsets = cp.where(
            d_single_row_mask,
            d_single_prefix,
            d_multi_prefix,
        ).astype(cp.int32)
        output_tags = cp.where(
            d_single_row_mask,
            FAMILY_TAGS[GeometryFamily.LINESTRING],
            FAMILY_TAGS[GeometryFamily.MULTILINESTRING],
        ).astype(cp.int8)

        device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}

        if int(cp.sum(d_single_row_mask)) > 0:
            d_single_run_indices = d_row_starts[d_single_row_mask]
            single_compacted = _compact_selected_runs(d_single_run_indices)
            if single_compacted is not None:
                d_single_x, d_single_y, d_single_geom_offsets = single_compacted
                device_families[GeometryFamily.LINESTRING] = DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.LINESTRING,
                    x=d_single_x,
                    y=d_single_y,
                    geometry_offsets=d_single_geom_offsets,
                    empty_mask=cp.zeros(d_single_run_indices.shape[0], dtype=cp.bool_),
                    bounds=None,
                )

        if int(cp.sum(d_multi_row_mask)) > 0:
            d_multi_row_starts = d_row_starts[d_multi_row_mask]
            d_multi_part_counts = d_row_run_counts[d_multi_row_mask].astype(cp.int32)
            multi_row_count = int(d_multi_row_starts.shape[0])
            total_multi_parts = int(cp.sum(d_multi_part_counts))
            d_multi_row_part_offsets = cp.empty(multi_row_count + 1, dtype=cp.int64)
            d_multi_row_part_offsets[0] = 0
            d_multi_row_part_offsets[1:] = cp.cumsum(d_multi_part_counts).astype(cp.int64)
            d_multi_intra = cp.arange(total_multi_parts, dtype=cp.int64)
            d_multi_row_ids = cp.searchsorted(
                d_multi_row_part_offsets[1:],
                d_multi_intra,
                side="right",
            )
            d_multi_run_indices = d_multi_row_starts[d_multi_row_ids] + (
                d_multi_intra - d_multi_row_part_offsets[d_multi_row_ids]
            )
            multi_compacted = _compact_selected_runs(d_multi_run_indices)
            if multi_compacted is not None:
                d_multi_x, d_multi_y, d_multi_part_offsets = multi_compacted
                d_multi_geom_offsets = cp.empty(multi_row_count + 1, dtype=cp.int32)
                d_multi_geom_offsets[0] = 0
                d_multi_geom_offsets[1:] = cp.cumsum(d_multi_part_counts).astype(cp.int32)
                device_families[GeometryFamily.MULTILINESTRING] = DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.MULTILINESTRING,
                    x=d_multi_x,
                    y=d_multi_y,
                    geometry_offsets=d_multi_geom_offsets,
                    empty_mask=cp.zeros(multi_row_count, dtype=cp.bool_),
                    part_offsets=d_multi_part_offsets,
                    bounds=None,
                )

        global_row_map = d_unique_rows.astype(cp.int32, copy=False)
        oga = build_device_resident_owned(
            device_families=device_families,
            row_count=compact_row_count,
            tags=output_tags,
            validity=output_validity,
            family_row_offsets=output_family_row_offsets,
            execution_mode="gpu",
        )
        return oga, global_row_map

    global_row_map = d_global_row_map.astype(cp.int32, copy=False)

    # ---------------------------------------------------------------
    # 7. Build device-resident OwnedGeometryArray
    # ---------------------------------------------------------------
    # Create metadata arrays directly on device to avoid H2D re-uploads.
    d_empty_mask = cp.zeros(line_count, dtype=cp.bool_)
    output_validity = cp.ones(line_count, dtype=cp.bool_)
    output_tags = cp.full(
        line_count,
        FAMILY_TAGS[GeometryFamily.LINESTRING],
        dtype=cp.int8,
    )
    output_family_row_offsets = cp.arange(line_count, dtype=cp.int32)

    device_families = {
        GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.LINESTRING,
            x=d_flat_x,
            y=d_flat_y,
            geometry_offsets=d_geom_offsets,
            empty_mask=d_empty_mask,
            bounds=None,
        )
    }
    oga = build_device_resident_owned(
        device_families=device_families,
        row_count=line_count,
        tags=output_tags,
        validity=output_validity,
        family_row_offsets=output_family_row_offsets,
        execution_mode="gpu",
    )
    return oga, global_row_map


def _clip_linestring_rows_gpu_fast_path(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
    seg_x0,
    seg_y0,
    seg_x1,
    seg_y1,
    row_segment_offsets: np.ndarray,
) -> tuple[OwnedGeometryArray, object] | None:
    """Specialized GPU path for dense LineString-only batches."""
    if (
        set(owned.families) != {GeometryFamily.LINESTRING}
        or owned.families[GeometryFamily.LINESTRING].row_count != owned.row_count
    ):
        return None
    if not np.all(owned.validity):
        return None

    import cupy as cp

    from vibespatial.cuda.cccl_primitives import exclusive_sum

    row_count = owned.row_count
    if row_count == 0:
        return None

    d_out_x0, d_out_y0, d_out_x1, d_out_y1, d_valid = _clip_line_segments_gpu_device(
        seg_x0, seg_y0, seg_x1, seg_y1, rect,
    )
    runtime = get_cuda_runtime()
    d_row_segment_offsets = cp.asarray(row_segment_offsets)
    d_run_counts = runtime.allocate((row_count,), cp.int32, zero=True)
    d_coord_counts = runtime.allocate((row_count,), cp.int32, zero=True)
    d_has_output = runtime.allocate((row_count,), cp.uint8, zero=True)

    kernels = _compile_line_row_kernels()
    ptr = runtime.pointer
    count_params = (
        (
            ptr(d_out_x0),
            ptr(d_out_y0),
            ptr(d_out_x1),
            ptr(d_out_y1),
            ptr(d_valid),
            ptr(d_row_segment_offsets),
            ptr(d_run_counts),
            ptr(d_coord_counts),
            ptr(d_has_output),
            row_count,
        ),
        (
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
        ),
    )
    count_grid, count_block = runtime.launch_config(kernels["line_single_run_count"], row_count)
    runtime.launch(
        kernels["line_single_run_count"],
        grid=count_grid,
        block=count_block,
        params=count_params,
    )

    d_output_rows = cp.flatnonzero(d_has_output != 0).astype(cp.int32, copy=False)
    output_row_count = int(d_output_rows.size)
    if output_row_count == 0:
        return None

    d_output_run_counts = d_run_counts[d_output_rows].astype(cp.int32, copy=False)
    d_output_coord_counts = d_coord_counts[d_output_rows].astype(cp.int32, copy=False)
    d_single_mask = d_output_run_counts == 1
    d_multi_mask = d_output_run_counts > 1
    d_single_rows = d_output_rows[d_single_mask]
    d_multi_rows = d_output_rows[d_multi_mask]

    d_single_prefix = (
        cp.cumsum(d_single_mask.astype(cp.int32)).astype(cp.int32) - d_single_mask.astype(cp.int32)
    )
    d_multi_prefix = (
        cp.cumsum(d_multi_mask.astype(cp.int32)).astype(cp.int32) - d_multi_mask.astype(cp.int32)
    )
    output_family_row_offsets = cp.where(
        d_single_mask,
        d_single_prefix,
        d_multi_prefix,
    ).astype(cp.int32)
    output_tags = cp.where(
        d_single_mask,
        FAMILY_TAGS[GeometryFamily.LINESTRING],
        FAMILY_TAGS[GeometryFamily.MULTILINESTRING],
    ).astype(cp.int8)
    output_validity = cp.ones(output_row_count, dtype=cp.bool_)

    device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}

    single_count = int(d_single_rows.size)
    if single_count > 0:
        d_single_counts_i64 = d_output_coord_counts[d_single_mask].astype(cp.int64, copy=False)
        d_single_offsets_i64 = exclusive_sum(d_single_counts_i64, synchronize=False)
        total_single_coords = count_scatter_total(runtime, d_single_counts_i64, d_single_offsets_i64)
        d_single_geom_offsets = cp.empty(single_count + 1, dtype=cp.int32)
        d_single_geom_offsets[:single_count] = d_single_offsets_i64.astype(cp.int32, copy=False)
        d_single_geom_offsets[single_count] = total_single_coords
        d_single_x = runtime.allocate((total_single_coords,), cp.float64)
        d_single_y = runtime.allocate((total_single_coords,), cp.float64)
        single_scatter_params = (
            (
                ptr(d_out_x0),
                ptr(d_out_y0),
                ptr(d_out_x1),
                ptr(d_out_y1),
                ptr(d_valid),
                ptr(d_row_segment_offsets),
                ptr(d_single_rows),
                ptr(d_single_geom_offsets),
                ptr(d_single_x),
                ptr(d_single_y),
                single_count,
            ),
            (
                KERNEL_PARAM_PTR,
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
            ),
        )
        single_grid, single_block = runtime.launch_config(
            kernels["line_single_run_scatter"],
            single_count,
        )
        runtime.launch(
            kernels["line_single_run_scatter"],
            grid=single_grid,
            block=single_block,
            params=single_scatter_params,
        )
        device_families[GeometryFamily.LINESTRING] = DeviceFamilyGeometryBuffer(
            family=GeometryFamily.LINESTRING,
            x=d_single_x,
            y=d_single_y,
            geometry_offsets=d_single_geom_offsets,
            empty_mask=cp.zeros(single_count, dtype=cp.bool_),
            bounds=None,
        )

    multi_count = int(d_multi_rows.size)
    if multi_count > 0:
        d_multi_part_counts_i64 = d_output_run_counts[d_multi_mask].astype(cp.int64, copy=False)
        d_multi_geom_offsets_i64 = exclusive_sum(d_multi_part_counts_i64, synchronize=False)
        total_multi_parts = count_scatter_total(runtime, d_multi_part_counts_i64, d_multi_geom_offsets_i64)
        d_multi_geom_offsets = cp.empty(multi_count + 1, dtype=cp.int32)
        d_multi_geom_offsets[:multi_count] = d_multi_geom_offsets_i64.astype(cp.int32, copy=False)
        d_multi_geom_offsets[multi_count] = total_multi_parts

        d_multi_coord_counts_i64 = d_output_coord_counts[d_multi_mask].astype(cp.int64, copy=False)
        d_multi_coord_offsets_i64 = exclusive_sum(d_multi_coord_counts_i64, synchronize=False)
        total_multi_coords = count_scatter_total(runtime, d_multi_coord_counts_i64, d_multi_coord_offsets_i64)
        d_multi_coord_offsets = cp.empty(multi_count + 1, dtype=cp.int32)
        d_multi_coord_offsets[:multi_count] = d_multi_coord_offsets_i64.astype(cp.int32, copy=False)
        d_multi_coord_offsets[multi_count] = total_multi_coords

        d_multi_part_offsets = cp.empty(total_multi_parts + 1, dtype=cp.int32)
        d_multi_x = runtime.allocate((total_multi_coords,), cp.float64)
        d_multi_y = runtime.allocate((total_multi_coords,), cp.float64)
        multi_scatter_params = (
            (
                ptr(d_out_x0),
                ptr(d_out_y0),
                ptr(d_out_x1),
                ptr(d_out_y1),
                ptr(d_valid),
                ptr(d_row_segment_offsets),
                ptr(d_multi_rows),
                ptr(d_multi_geom_offsets),
                ptr(d_multi_coord_offsets),
                ptr(d_multi_part_offsets),
                ptr(d_multi_x),
                ptr(d_multi_y),
                multi_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
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
            ),
        )
        multi_grid, multi_block = runtime.launch_config(
            kernels["line_multi_run_scatter"],
            multi_count,
        )
        runtime.launch(
            kernels["line_multi_run_scatter"],
            grid=multi_grid,
            block=multi_block,
            params=multi_scatter_params,
        )
        device_families[GeometryFamily.MULTILINESTRING] = DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTILINESTRING,
            x=d_multi_x,
            y=d_multi_y,
            geometry_offsets=d_multi_geom_offsets,
            empty_mask=cp.zeros(multi_count, dtype=cp.bool_),
            part_offsets=d_multi_part_offsets,
            bounds=None,
        )

    if not device_families:
        return None

    return (
        build_device_resident_owned(
            device_families=device_families,
            row_count=output_row_count,
            tags=output_tags,
            validity=output_validity,
            family_row_offsets=output_family_row_offsets,
            execution_mode="gpu",
        ),
        d_output_rows,
    )


def _clip_all_lines_gpu(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
) -> tuple[OwnedGeometryArray | None, object | None]:
    """Batch-clip ALL linestring/multilinestring rows on GPU via Liang-Barsky.

    TRUE ZERO-COPY implementation:
      1. Extract segments from buffers via vectorized slicing (host, fast metadata)
      2. Run Liang-Barsky GPU kernel -- outputs stay on device (CuPy)
      3. Assemble coordinate arrays on device (CuPy + CCCL exclusive_sum)
      4. Return device-resident OwnedGeometryArray -- no Shapely, no D2H transfers

    Returns (OwnedGeometryArray, global_row_map) or (None, None).
    global_row_map maps each compact output row back to the input row and
    stays device-resident until a host scatter/materialization consumer
    actually needs it.
    """
    line_families = [
        f for f in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
        if f in owned.families
    ]

    if not line_families:
        return None, None

    # Repeated viewport clips over the same device-resident line layer should
    # not rebuild static segment endpoints on every call.
    cached_segments = None
    cache_key = None
    if owned.device_state is not None and set(line_families) == {GeometryFamily.LINESTRING}:
        cache_key = ("clip_rect_linestring_segments", owned.row_count)
        cache_entry = getattr(owned, "_cached_clip_rect_line_segments", None)
        if cache_entry is not None and cache_entry[0] == cache_key:
            cached_segments = cache_entry[1]

    # -- Step 1: vectorized segment extraction (metadata-level) ------
    if cached_segments is None:
        cached_segments = _extract_segments_vectorized(owned, line_families)
        if cache_key is not None:
            owned._cached_clip_rect_line_segments = (cache_key, cached_segments)

    (
        seg_x0, seg_y0, seg_x1, seg_y1,
        row_segment_offsets,
        segment_global_rows,
        segment_part_break_after,
    ) = cached_segments

    total_segments = len(seg_x0)
    if total_segments == 0:
        return None, None

    fast_path_result = _clip_linestring_rows_gpu_fast_path(
        owned,
        rect,
        seg_x0,
        seg_y0,
        seg_x1,
        seg_y1,
        row_segment_offsets,
    )
    if fast_path_result is not None:
        return fast_path_result

    # -- Step 2: GPU kernel -- outputs stay on device as CuPy arrays ------
    d_out_x0, d_out_y0, d_out_x1, d_out_y1, d_valid = _clip_line_segments_gpu_device(
        seg_x0, seg_y0, seg_x1, seg_y1, rect,
    )

    # -- Step 3: device-resident assembly (CuPy + CCCL, zero numpy) -------
    return _build_line_clip_device_result(
        d_out_x0, d_out_y0, d_out_x1, d_out_y1, d_valid,
        segment_global_rows,
        segment_part_break_after,
    )


def _row_family_and_local_index(owned: OwnedGeometryArray, row_index: int):
    if not owned.validity[row_index]:
        return None, -1
    for family_name, tag in FAMILY_TAGS.items():
        if int(owned.tags[row_index]) == tag:
            return family_name, int(owned.family_row_offsets[row_index])
    return None, -1


# ---------------------------------------------------------------------------
# Batched GPU polygon clip — all polygons in a single kernel launch
# ---------------------------------------------------------------------------

def _extract_polygon_rings_single_family(
    buffer: FamilyGeometryBuffer,
    family_rows: np.ndarray,
    local_rows: np.ndarray,
    empty_flags: np.ndarray,
    device_buffer: DeviceFamilyGeometryBuffer | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Fully vectorized ring extraction for a single POLYGON family.

    Replaces the per-row Python loop with bulk numpy operations on the
    buffer's offset arrays.  At 100K rows this is ~200x faster than the
    per-row loop.

    When ``device_buffer`` is provided and host coordinates are empty
    (device-resident data), coordinate arrays are returned as CuPy device
    arrays.  The downstream GPU kernel (``_clip_polygon_rings_gpu_device``)
    accepts both numpy and CuPy arrays, so this avoids a pointless D->H->D
    round-trip.

    Returns the same 7-tuple as ``_extract_polygon_rings_vectorized``.
    Coordinate arrays (ring_x, ring_y) may be numpy or CuPy depending on
    data residency.
    """
    geom_offsets = buffer.geometry_offsets
    ring_off = buffer.ring_offsets

    # Per-geometry ring counts (vectorized gather on offset array).
    ring_starts = geom_offsets[local_rows]       # int32/int64
    ring_ends = geom_offsets[local_rows + 1]
    ring_counts = ring_ends - ring_starts        # per-row ring count

    # Zero out ring counts for empty rows.
    non_empty = ~empty_flags
    ring_counts = ring_counts * non_empty  # broadcast: empty -> 0

    # Global row indices -- all family rows participate.
    global_row_indices = family_rows.tolist()

    # geom_ring_offsets: cumulative ring counts per geometry.
    geom_ring_offsets = np.empty(len(ring_counts) + 1, dtype=np.int32)
    geom_ring_offsets[0] = 0
    np.cumsum(ring_counts, out=geom_ring_offsets[1:])
    total_rings = int(geom_ring_offsets[-1])

    if total_rings == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            geom_ring_offsets,
            global_row_indices,
        )

    # Identify non-empty rows to build a flat ring index array.
    has_rings = ring_counts > 0
    ne_ring_starts = ring_starts[has_rings]
    ne_ring_counts = ring_counts[has_rings]

    # ---- Fast path: contiguous identity mapping ----
    # When all rows are non-empty and local_rows is an identity mapping
    # (0, 1, ..., N-1), the buffer's coordinates and ring_offsets are
    # already in the exact layout needed.  No gather required.
    all_non_empty = not empty_flags.any()
    is_identity = (
        all_non_empty
        and len(local_rows) > 0
        and int(local_rows[0]) == 0
        and int(local_rows[-1]) == len(local_rows) - 1
    )

    if is_identity:
        # Ring offsets can be used directly from the buffer.
        ring_offsets_out = ring_off.astype(np.int32, copy=False)

        # Coordinates: use device arrays if host is empty (device-resident).
        if buffer.x.size == 0 and device_buffer is not None:
            import cupy as cp
            ring_x = cp.asarray(device_buffer.x)
            ring_y = cp.asarray(device_buffer.y)
            ring_offsets_out = cp.asarray(ring_offsets_out, dtype=cp.int32)
        else:
            ring_x = np.asarray(buffer.x, dtype=np.float64)
            ring_y = np.asarray(buffer.y, dtype=np.float64)
    else:
        # ---- General vectorized path: gather via flat_ring_idx ----
        # Build flat array of buffer ring indices for all rings of all
        # non-empty geometries using vectorized repeat+arange pattern.
        flat_ring_idx = np.repeat(ne_ring_starts, ne_ring_counts)
        offsets_within = np.arange(total_rings, dtype=flat_ring_idx.dtype)
        cum_base = np.repeat(
            np.concatenate(([np.int32(0)], np.cumsum(ne_ring_counts[:-1], dtype=np.int32))),
            ne_ring_counts,
        )
        flat_ring_idx = flat_ring_idx + (offsets_within - cum_base)

        # Gather ring boundary offsets from buffer.ring_offsets.
        ring_coord_starts = ring_off[flat_ring_idx].astype(np.int32, copy=False)
        ring_coord_ends = ring_off[flat_ring_idx + 1].astype(np.int32, copy=False)
        ring_lengths = ring_coord_ends - ring_coord_starts

        # Ring offset array (rebased to 0).
        ring_offsets_out = np.empty(total_rings + 1, dtype=np.int32)
        ring_offsets_out[0] = 0
        np.cumsum(ring_lengths, out=ring_offsets_out[1:])
        total_coords = int(ring_offsets_out[-1])

        # Build coordinate gather index.
        coord_gather = np.repeat(ring_coord_starts, ring_lengths)
        coord_offsets_within = np.arange(total_coords, dtype=np.int32)
        coord_cum_base = np.repeat(ring_offsets_out[:-1], ring_lengths)
        coord_gather = coord_gather + (coord_offsets_within - coord_cum_base)

        # Gather coordinates: use device arrays when host is empty.
        if buffer.x.size == 0 and device_buffer is not None:
            import cupy as cp
            d_gather = cp.asarray(coord_gather.astype(np.int64))
            ring_x = cp.asarray(device_buffer.x)[d_gather]
            ring_y = cp.asarray(device_buffer.y)[d_gather]
            ring_offsets_out = cp.asarray(ring_offsets_out, dtype=cp.int32)
        else:
            ring_x = np.asarray(buffer.x, dtype=np.float64)[coord_gather]
            ring_y = np.asarray(buffer.y, dtype=np.float64)[coord_gather]

    # ring_geom_map: which global row each ring belongs to.
    ne_global_rows = family_rows[has_rings].astype(np.int32, copy=False)
    ring_geom_map = np.repeat(ne_global_rows, ne_ring_counts.astype(np.int32, copy=False))

    # ring_is_exterior: 1 for the first ring of each geometry, 0 for holes.
    ring_is_exterior = np.zeros(total_rings, dtype=np.int32)
    # The first ring of each non-empty geometry is at cumulative position.
    ext_positions = geom_ring_offsets[:-1][has_rings]
    ring_is_exterior[ext_positions] = 1

    return (
        ring_x, ring_y, ring_offsets_out,
        ring_geom_map, ring_is_exterior,
        geom_ring_offsets, global_row_indices,
    )


def _extract_polygon_rings_vectorized(
    owned: OwnedGeometryArray,
    polygon_families: list[GeometryFamily],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Extract ALL polygon rings across ALL polygon-family rows into flat arrays.

    For single-family POLYGON input (the common hot path), delegates to
    ``_extract_polygon_rings_single_family`` which uses pure numpy
    vectorized operations with zero Python per-row loops.

    For mixed or MultiPolygon families, falls back to the per-row loop
    which handles the extra part_offsets indirection.

    Returns
    -------
    ring_x, ring_y : flat coordinate arrays (float64)
    ring_offsets : offsets into x/y for each ring (int32, length = total_rings + 1)
    ring_geom_map : which global row index each ring belongs to (int32)
    ring_is_exterior : 1 if exterior ring, 0 if hole (int32)
    geom_ring_offsets : offsets into ring arrays for each geometry row (int32)
    global_row_indices : ordered list of global row indices that have polygon data
    """
    # ---- Fast path: single POLYGON family (no MultiPolygon) ----
    # This is the common case for the gpu-constructive benchmark and
    # eliminates all per-row Python overhead.
    if (
        len(polygon_families) == 1
        and polygon_families[0] == GeometryFamily.POLYGON
    ):
        family = GeometryFamily.POLYGON
        buffer = owned.families[family]
        tag = FAMILY_TAGS[family]
        family_rows = np.flatnonzero(owned.tags == tag)

        if (
            buffer.row_count == 0
            or buffer.geometry_offsets is None
            or len(buffer.geometry_offsets) < 2
        ):
            # All rows are empty/degenerate.
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.zeros(1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.zeros(len(family_rows) + 1, dtype=np.int32),
                family_rows.tolist(),
            )

        local_rows = owned.family_row_offsets[family_rows].astype(np.int32, copy=False)

        if buffer.empty_mask.size > 0:
            empty_flags = buffer.empty_mask[local_rows]
        else:
            empty_flags = np.zeros(len(local_rows), dtype=bool)

        # Resolve device buffer for zero-copy device-resident coordinate
        # access (avoids D->H->D round-trip when host x/y are empty stubs).
        device_buffer = None
        if (
            not buffer.host_materialized
            and owned.device_state is not None
            and family in owned.device_state.families
        ):
            device_buffer = owned.device_state.families[family]

        return _extract_polygon_rings_single_family(
            buffer, family_rows, local_rows, empty_flags,
            device_buffer=device_buffer,
        )

    # ---- General path: mixed families or MultiPolygon ----
    # Collect per-family data in bulk, then concatenate once.
    all_x_chunks: list[np.ndarray] = []
    all_y_chunks: list[np.ndarray] = []
    all_ring_offsets_chunks: list[np.ndarray] = []
    all_ring_geom_map: list[np.ndarray] = []
    all_ring_is_exterior: list[np.ndarray] = []
    all_geom_ring_counts: list[int] = []
    global_row_indices: list[int] = []

    coord_cursor = 0  # running total of coordinates appended so far
    ring_cursor = 0   # running total of rings appended so far

    for family in polygon_families:
        buffer = owned.families[family]
        tag = FAMILY_TAGS[family]
        family_rows = np.flatnonzero(owned.tags == tag)

        if buffer.row_count == 0 or buffer.geometry_offsets is None or len(buffer.geometry_offsets) < 2:
            for global_row in family_rows:
                global_row_indices.append(int(global_row))
                all_geom_ring_counts.append(0)
            continue

        local_rows = owned.family_row_offsets[family_rows].astype(np.int32, copy=False)

        # Identify non-empty rows
        if buffer.empty_mask.size > 0:
            empty_flags = buffer.empty_mask[local_rows]
        else:
            empty_flags = np.zeros(len(local_rows), dtype=bool)

        if family == GeometryFamily.POLYGON:
            # Vectorized path for POLYGON within a mixed-family extraction.
            dev_buf = None
            if (
                not buffer.host_materialized
                and owned.device_state is not None
                and family in owned.device_state.families
            ):
                dev_buf = owned.device_state.families[family]
            (
                fam_x, fam_y, fam_ring_offsets,
                fam_geom_map, fam_ext, fam_geom_ring_offsets,
                fam_global_rows,
            ) = _extract_polygon_rings_single_family(
                buffer, family_rows, local_rows, empty_flags,
                device_buffer=dev_buf,
            )
            if fam_ring_offsets.size > 1:
                total_fam_rings = len(fam_ring_offsets) - 1
                total_fam_coords = int(fam_ring_offsets[-1])
                all_x_chunks.append(fam_x)
                all_y_chunks.append(fam_y)
                # Rebase ring offsets to coord_cursor.
                all_ring_offsets_chunks.append(fam_ring_offsets[:-1] + coord_cursor)
                all_ring_geom_map.append(fam_geom_map)
                all_ring_is_exterior.append(fam_ext)
                coord_cursor += total_fam_coords
                ring_cursor += total_fam_rings
            # Accumulate ring counts from geom_ring_offsets diff.
            fam_rc = np.diff(fam_geom_ring_offsets)
            all_geom_ring_counts.extend(fam_rc.tolist())
            global_row_indices.extend(fam_global_rows)
        else:
            # MultiPolygon: geometry_offsets -> part_offsets -> ring_offsets.
            # Keep per-row loop -- MultiPolygon has extra indirection and
            # is rarely the hot path.
            for idx, global_row in enumerate(family_rows):
                grow = int(global_row)
                lr = int(local_rows[idx])

                if empty_flags[idx]:
                    global_row_indices.append(grow)
                    all_geom_ring_counts.append(0)
                    continue

                part_start = int(buffer.geometry_offsets[lr])
                part_end = int(buffer.geometry_offsets[lr + 1])
                n_parts = part_end - part_start
                if n_parts == 0:
                    global_row_indices.append(grow)
                    all_geom_ring_counts.append(0)
                    continue

                total_rings_this_geom = 0
                for part_idx in range(part_start, part_end):
                    ring_start_idx = int(buffer.part_offsets[part_idx])
                    ring_end_idx = int(buffer.part_offsets[part_idx + 1])
                    n_rings = ring_end_idx - ring_start_idx
                    if n_rings == 0:
                        continue

                    local_ring_offsets = buffer.ring_offsets[ring_start_idx:ring_end_idx + 1]
                    coord_start = int(local_ring_offsets[0])
                    coord_end = int(local_ring_offsets[-1])
                    n_coords = coord_end - coord_start

                    all_x_chunks.append(np.asarray(buffer.x[coord_start:coord_end], dtype=np.float64))
                    all_y_chunks.append(np.asarray(buffer.y[coord_start:coord_end], dtype=np.float64))

                    rebased_offsets = np.asarray(local_ring_offsets[:-1], dtype=np.int32) - coord_start + coord_cursor
                    all_ring_offsets_chunks.append(rebased_offsets)

                    all_ring_geom_map.append(np.full(n_rings, grow, dtype=np.int32))

                    ext_flags = np.zeros(n_rings, dtype=np.int32)
                    ext_flags[0] = 1
                    all_ring_is_exterior.append(ext_flags)

                    coord_cursor += n_coords
                    ring_cursor += n_rings
                    total_rings_this_geom += n_rings

                global_row_indices.append(grow)
                all_geom_ring_counts.append(total_rings_this_geom)

    # Build final concatenated arrays
    if all_x_chunks:
        ring_x = np.concatenate(all_x_chunks)
        ring_y = np.concatenate(all_y_chunks)
        ring_offsets_body = np.concatenate(all_ring_offsets_chunks)
        ring_offsets = np.empty(ring_cursor + 1, dtype=np.int32)
        ring_offsets[:-1] = ring_offsets_body
        ring_offsets[-1] = coord_cursor
        ring_geom_map = np.concatenate(all_ring_geom_map)
        ring_is_exterior = np.concatenate(all_ring_is_exterior)
    else:
        ring_x = np.empty(0, dtype=np.float64)
        ring_y = np.empty(0, dtype=np.float64)
        ring_offsets = np.zeros(1, dtype=np.int32)
        ring_geom_map = np.empty(0, dtype=np.int32)
        ring_is_exterior = np.empty(0, dtype=np.int32)

    # Build geom_ring_offsets from per-geometry ring counts
    geom_ring_offsets = np.empty(len(all_geom_ring_counts) + 1, dtype=np.int32)
    geom_ring_offsets[0] = 0
    np.cumsum(all_geom_ring_counts, out=geom_ring_offsets[1:])

    return (
        ring_x, ring_y, ring_offsets,
        ring_geom_map, ring_is_exterior,
        geom_ring_offsets, global_row_indices,
    )


def _build_polygon_clip_owned_result(
    d_out_x,
    d_out_y,
    d_full_offsets,
    ring_is_exterior: np.ndarray,
    geom_ring_offsets: np.ndarray,
    global_row_indices: list[int],
    input_row_count: int,
) -> tuple[OwnedGeometryArray | None, np.ndarray]:
    """Build a device-resident OwnedGeometryArray from GPU clip output.

    Ring survival analysis and gather-index construction run entirely on
    device (CuPy).  Only small GeoArrow metadata offsets are brought to
    host for OwnedGeometryArray construction.  No D->H->D ping-pong.

    Returns (owned_result, validity_mask) where validity_mask is a bool array
    of length input_row_count indicating which global rows have valid output.
    owned_result contains only the valid output polygons (compact rows).
    """
    import cupy as cp

    from vibespatial.cuda.cccl_primitives import exclusive_sum, segmented_reduce_sum

    runtime = get_cuda_runtime()

    # ---- Vectorized ring survival analysis (device-resident) ----
    total_rings = len(ring_is_exterior)

    if total_rings == 0:
        validity_mask = np.zeros(input_row_count, dtype=bool)
        return None, validity_mask

    # Upload small host metadata arrays to device once.
    d_ring_is_exterior = cp.asarray(ring_is_exterior)
    d_geom_ring_offsets = cp.asarray(geom_ring_offsets)

    # Per-ring vertex counts from the GPU output offsets (stays on device).
    d_ring_verts = cp.diff(d_full_offsets[:total_rings + 1]).astype(cp.int32)

    # Exterior ring validity: an exterior ring survives if it has >= 4 verts.
    d_is_ext = d_ring_is_exterior.astype(bool)
    d_ext_positions = cp.flatnonzero(d_is_ext)
    d_ext_valid = d_ring_verts[d_ext_positions] >= 4  # bool per exterior ring

    # Forward-fill exterior validity to subsequent holes.  Each exterior
    # ring's validity applies to itself and all following holes until the
    # next exterior ring.
    #
    # Strategy: use exclusive_sum on a marker array to assign each ring its
    # exterior's index, then gather the exterior validity via that index.
    # This avoids cp.repeat with per-element device counts (which internally
    # calls .get() for output sizing — hidden D2H sync).
    n_ext = d_ext_positions.size
    if n_ext > 0:
        # Build a marker: 1 at each exterior position, 0 elsewhere.
        # exclusive_sum gives each ring the index of its owning exterior.
        d_ext_marker = cp.zeros(total_rings, dtype=cp.int32)
        d_ext_marker[d_ext_positions] = 1
        d_ext_owner_idx = exclusive_sum(d_ext_marker, synchronize=False)
        # Each ring's validity = its owning exterior's validity.
        d_ext_validity_filled = d_ext_valid[d_ext_owner_idx]
        # Any rings before the first exterior have owner index 0 from
        # the exclusive scan, but they should be invalid.  Guard by
        # zeroing rings before the first exterior position.
        first_ext_pos = int(d_ext_positions[0].item())  # zcopy:ok(single scalar for branch guard)
        if first_ext_pos > 0:
            d_ext_validity_filled[:first_ext_pos] = False
    else:
        d_ext_validity_filled = cp.zeros(total_rings, dtype=bool)

    # A ring survives if: (a) its exterior is valid, AND (b) it has >= 4 verts.
    d_ring_survives = d_ext_validity_filled & (d_ring_verts >= 4)

    # ---- Per-geometry aggregation (device-resident) ----
    # CCCL segmented_reduce_sum (Tier 3a) — replaces cumsum+indexing pattern.
    n_geoms = len(geom_ring_offsets) - 1
    d_ring_survives_i32 = d_ring_survives.astype(cp.int32)

    seg_result = segmented_reduce_sum(
        d_ring_survives_i32,
        d_geom_ring_offsets[:-1],
        d_geom_ring_offsets[1:],
        num_segments=n_geoms,
        synchronize=False,
    )
    d_surviving_per_geom = seg_result.values

    d_geom_has_output = d_surviving_per_geom > 0

    # Bring small per-geometry results to host for validity mask construction
    # and GeoArrow metadata.  These are O(n_geoms) -- small metadata.
    h_geom_has_output = cp.asnumpy(d_geom_has_output)  # zcopy:ok(small per-geom metadata for host mask)
    h_surviving_per_geom = cp.asnumpy(d_surviving_per_geom)  # zcopy:ok(small per-geom metadata for GeoArrow offsets)

    # Global row indices for geometries with output.
    global_rows_arr = np.asarray(global_row_indices, dtype=np.intp)
    output_valid_rows_arr = global_rows_arr[h_geom_has_output]
    output_row_count = int(output_valid_rows_arr.size)

    # Build validity mask for the caller (covers all input rows).
    validity_mask = np.zeros(input_row_count, dtype=bool)
    if output_row_count > 0:
        validity_mask[output_valid_rows_arr] = True

    if output_row_count == 0:
        return None, validity_mask

    # ---- Build GeoArrow offsets for surviving rings ----
    # Surviving ring indices (flat) -- stays on device for gather.
    d_surviving_idx = cp.flatnonzero(d_ring_survives)
    d_surviving_verts = d_ring_verts[d_surviving_idx]

    # Ring offsets: cumulative vertex counts of surviving rings.
    # Compute on device to avoid D->H->D round-trip, then D2H once for
    # GeoArrow metadata.
    d_ring_offsets_dev = cp.empty(d_surviving_verts.size + 1, dtype=cp.int32)
    d_ring_offsets_dev[0] = 0
    cp.cumsum(d_surviving_verts, out=d_ring_offsets_dev[1:])
    h_ring_offsets = cp.asnumpy(d_ring_offsets_dev)  # zcopy:ok(small ring offsets for GeoArrow metadata)

    # Geometry offsets: cumulative surviving-ring counts per output geometry.
    output_surviving_counts = h_surviving_per_geom[h_geom_has_output]
    h_geom_offsets = np.empty(output_row_count + 1, dtype=np.int32)
    h_geom_offsets[0] = 0
    np.cumsum(output_surviving_counts, out=h_geom_offsets[1:])

    # ---- Build gather indices on device via NVRTC segmented_arange ----
    # For each surviving ring, gather coordinates from
    # d_full_offsets[ring_idx] .. d_full_offsets[ring_idx+1].
    d_gather_starts = d_full_offsets[d_surviving_idx].astype(cp.int64)
    d_gather_lens = d_surviving_verts.astype(cp.int64)
    # total_gather is already available as the last element of h_ring_offsets
    # (cumsum of surviving vertex counts), avoiding an extra D2H transfer.
    total_gather = int(h_ring_offsets[-1])

    if total_gather > 0:
        # Compute per-ring write offsets via exclusive_sum (CCCL Tier 3a).
        d_write_offsets = exclusive_sum(d_gather_lens, synchronize=False)

        # Launch NVRTC segmented_arange kernel: one thread per surviving ring,
        # each writes its contiguous range of gather indices.
        from vibespatial.cuda._runtime import KERNEL_PARAM_I32, KERNEL_PARAM_PTR

        n_surviving_rings = int(d_surviving_idx.size)
        d_gather = cp.empty(total_gather, dtype=cp.int64)

        kernels = _compile_seg_arange_kernels()
        ptr = runtime.pointer
        seg_params = (
            (
                ptr(d_gather_starts),
                ptr(d_gather_lens),
                ptr(d_write_offsets),
                ptr(d_gather),
                n_surviving_rings,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        seg_grid, seg_block = runtime.launch_config(
            kernels["segmented_arange"], n_surviving_rings,
        )
        runtime.launch(
            kernels["segmented_arange"],
            grid=seg_grid,
            block=seg_block,
            params=seg_params,
        )
        gathered_x = d_out_x[d_gather]
        gathered_y = d_out_y[d_gather]
    else:
        gathered_x = cp.empty(0, dtype=cp.float64)
        gathered_y = cp.empty(0, dtype=cp.float64)

    output_empty_mask = cp.zeros(output_row_count, dtype=cp.bool_)

    output_validity = cp.ones(output_row_count, dtype=cp.bool_)
    output_tags = cp.full(
        output_row_count,
        FAMILY_TAGS[GeometryFamily.POLYGON],
        dtype=cp.int8,
    )
    output_family_row_offsets = cp.arange(output_row_count, dtype=cp.int32)

    device_families = {
        GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=gathered_x,
            y=gathered_y,
            geometry_offsets=runtime.from_host(h_geom_offsets),
            empty_mask=output_empty_mask,
            ring_offsets=runtime.from_host(h_ring_offsets),
            bounds=None,
        )
    }
    return (
        build_device_resident_owned(
            device_families=device_families,
            row_count=output_row_count,
            tags=output_tags,
            validity=output_validity,
            family_row_offsets=output_family_row_offsets,
            execution_mode="gpu",
        ),
        validity_mask,
    )


def _clip_all_polygons_gpu(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
) -> tuple[OwnedGeometryArray | None, np.ndarray | None]:
    """Batch-clip ALL polygon/multipolygon rows on GPU in a single kernel launch.

    TRUE ZERO-COPY implementation:
      1. Extract all rings from buffers via vectorized slicing (no per-element loops)
      2. Run Sutherland-Hodgman GPU kernel (count-scatter pattern)
      3. Build device-resident OwnedGeometryArray directly from GPU output arrays

    No Shapely construction. No from_shapely_geometries. No per-coordinate
    Python loops. When input is device-resident, the entire pipeline stays
    on device.

    Parameters
    ----------
    owned : OwnedGeometryArray with polygon/multipolygon families
    rect : (xmin, ymin, xmax, ymax) clip rectangle
    precision_plan : PrecisionPlan for observability (CONSTRUCTIVE class, stays fp64)

    Returns
    -------
    (owned_result, validity_mask) where owned_result is a device-resident
    OwnedGeometryArray built directly from clipped coordinates on the GPU
    (contains only the valid output polygons), and validity_mask is either a
    bool array of length owned.row_count indicating which global rows have
    output or None when the output stays row-aligned and validity can remain
    device-resident until host materialization is requested.
    """

    polygon_families = [
        f for f in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
        if f in owned.families
    ]
    empty_validity = np.zeros(owned.row_count, dtype=bool)
    if not polygon_families:
        return None, empty_validity

    fast_rect_result = _clip_all_polygons_gpu_rect_fast_path(
        owned,
        rect,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
    )
    if fast_rect_result is not None:
        return fast_rect_result

    # Step 1: Extract all rings -- vectorized, no per-coordinate Python loops
    (
        ring_x, ring_y, ring_offsets,
        ring_geom_map, ring_is_exterior,
        geom_ring_offsets, global_row_indices,
    ) = _extract_polygon_rings_vectorized(owned, polygon_families)

    total_rings = len(ring_offsets) - 1
    if total_rings == 0:
        return None, empty_validity

    # Step 2: Run GPU clip -- returns device arrays (CuPy)
    d_out_x, d_out_y, d_full_offsets = _clip_polygon_rings_gpu_device(
        ring_x, ring_y, ring_offsets, rect,
    )

    if d_out_x is None:
        return None, empty_validity

    # Step 3: Build device-resident OGA directly from GPU output -- no Shapely
    owned_result, validity_mask = _build_polygon_clip_owned_result(
        d_out_x, d_out_y, d_full_offsets,
        ring_is_exterior, geom_ring_offsets,
        global_row_indices, owned.row_count,
    )

    return owned_result, validity_mask


def _clip_all_polygons_gpu_rect_fast_path(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
    *,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
) -> tuple[OwnedGeometryArray | None, np.ndarray | None] | None:
    """Use the pair-owned rectangle kernel when the polygon batch fits it."""
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU dispatch
        return None

    from vibespatial.constructive.polygon_intersection_output import (
        build_device_backed_polygon_intersection_output,
        build_empty_device_backed_polygon_intersection_output,
    )
    from vibespatial.cuda._runtime import KERNEL_PARAM_I32, KERNEL_PARAM_PTR
    from vibespatial.cuda.cccl_primitives import exclusive_sum
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        _device_is_dense_single_ring_polygons,
        _extract_polygon_family_device_buffer,
        _polygon_rect_intersection_kernels,
    )

    row_count = owned.row_count
    if row_count == 0:
        return None

    if (
        set(owned.families) != {GeometryFamily.POLYGON}
        or owned.families[GeometryFamily.POLYGON].row_count != row_count
    ):
        return None

    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="clip_by_rect selected GPU rectangle polygon fast path",
    )
    left_dev, left_host = _extract_polygon_family_device_buffer(owned)
    if left_dev is None or left_host is None or left_host.row_count != row_count:
        return None
    if not _device_is_dense_single_ring_polygons(left_dev, row_count):
        return None

    left_state = owned.device_state
    if left_state is None:
        return None

    runtime = get_cuda_runtime()
    d_xmin = cp.full(row_count, rect[0], dtype=cp.float64)
    d_ymin = cp.full(row_count, rect[1], dtype=cp.float64)
    d_xmax = cp.full(row_count, rect[2], dtype=cp.float64)
    d_ymax = cp.full(row_count, rect[3], dtype=cp.float64)
    d_left_valid = (
        cp.asarray(left_state.validity).astype(cp.bool_, copy=False)
        & ~left_dev.empty_mask.astype(cp.bool_, copy=False)
    ).astype(cp.int32, copy=False)
    d_right_valid = cp.ones(row_count, dtype=cp.int32)
    d_counts = runtime.allocate((row_count,), cp.int32, zero=True)
    d_valid = runtime.allocate((row_count,), cp.int32, zero=True)
    d_boundary_overlap = runtime.allocate((row_count,), cp.int32, zero=True)

    kernels = _polygon_rect_intersection_kernels()
    ptr = runtime.pointer
    count_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(d_xmin),
            ptr(d_ymin),
            ptr(d_xmax),
            ptr(d_ymax),
            ptr(d_left_valid),
            ptr(d_right_valid),
            ptr(d_counts),
            ptr(d_valid),
            ptr(d_boundary_overlap),
            row_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
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
        ),
    )
    count_grid, count_block = runtime.launch_config(
        kernels["polygon_rect_intersection_count"],
        row_count,
    )
    runtime.launch(
        kernels["polygon_rect_intersection_count"],
        grid=count_grid,
        block=count_block,
        params=count_params,
    )

    d_offsets = exclusive_sum(d_counts, synchronize=False)
    total_verts = count_scatter_total(runtime, d_counts, d_offsets)
    detail = (
        f"rows={row_count}, "
        f"precision={precision_plan.compute_precision.value}, "
        "rect=scalar"
    )
    if total_verts == 0:
        empty_result = build_empty_device_backed_polygon_intersection_output(
            row_count=row_count,
            runtime_selection=runtime_selection,
        )
        empty_result._polygon_rect_boundary_overlap = d_boundary_overlap.astype(
            cp.bool_,
            copy=False,
        )
        record_dispatch_event(
            surface="vibespatial.kernels.constructive.polygon_rect_intersection",
            operation="polygon_rect_intersection",
            implementation="polygon_rect_intersection_gpu_scalar_rect",
            reason=runtime_selection.reason,
            detail=detail,
            requested=runtime_selection.requested,
            selected=ExecutionMode.GPU,
        )
        return empty_result, None

    d_out_x = runtime.allocate((total_verts,), cp.float64)
    d_out_y = runtime.allocate((total_verts,), cp.float64)
    scatter_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(d_xmin),
            ptr(d_ymin),
            ptr(d_xmax),
            ptr(d_ymax),
            ptr(d_left_valid),
            ptr(d_right_valid),
            ptr(d_offsets),
            ptr(d_counts),
            ptr(d_valid),
            ptr(d_out_x),
            ptr(d_out_y),
            row_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
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
        ),
    )
    scatter_grid, scatter_block = runtime.launch_config(
        kernels["polygon_rect_intersection_scatter"],
        row_count,
    )
    runtime.launch(
        kernels["polygon_rect_intersection_scatter"],
        grid=scatter_grid,
        block=scatter_block,
        params=scatter_params,
    )

    d_ring_offsets = cp.empty(row_count + 1, dtype=cp.int32)
    d_ring_offsets[:row_count] = cp.asarray(d_offsets)
    d_ring_offsets[row_count] = total_verts
    result = build_device_backed_polygon_intersection_output(
        d_out_x,
        d_out_y,
        row_count=row_count,
        validity=d_valid.astype(cp.bool_, copy=False),
        ring_offsets=d_ring_offsets,
        runtime_selection=runtime_selection,
    )
    result._polygon_rect_boundary_overlap = d_boundary_overlap.astype(cp.bool_, copy=False)
    record_dispatch_event(
        surface="vibespatial.kernels.constructive.polygon_rect_intersection",
        operation="polygon_rect_intersection",
        implementation="polygon_rect_intersection_gpu_scalar_rect",
        reason=runtime_selection.reason,
        detail=detail,
        requested=runtime_selection.requested,
        selected=ExecutionMode.GPU,
    )
    return result, None


def _clip_dispatch_residency(values: object) -> Residency:
    """Treat owned arrays with live device buffers as device-native clip inputs."""
    residency = combined_residency(values)
    if not isinstance(values, OwnedGeometryArray):
        return residency
    if values.device_state is None:
        return residency
    if any(
        family in values.families
        for family in (
            GeometryFamily.POINT,
            GeometryFamily.POLYGON,
            GeometryFamily.MULTIPOLYGON,
            GeometryFamily.LINESTRING,
            GeometryFamily.MULTILINESTRING,
        )
    ):
        return Residency.DEVICE
    return residency


def _supported_gpu_clip_row_families() -> tuple[GeometryFamily, ...]:
    return (
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTILINESTRING,
    )


def _materialize_gpu_clip_row_masks(
    owned: OwnedGeometryArray,
):
    import cupy as cp

    d_validity = cp.asarray(owned.device_state.validity).astype(cp.bool_, copy=False)
    d_tags = cp.asarray(owned.device_state.tags)
    d_fast_mask = cp.zeros(owned.row_count, dtype=cp.bool_)
    for family in _supported_gpu_clip_row_families():
        if family in owned.families:
            d_fast_mask |= d_tags == FAMILY_TAGS[family]
    d_fast_mask &= d_validity
    d_fallback_mask = d_validity & ~d_fast_mask
    return d_fast_mask, d_fallback_mask


def _build_gpu_clip_row_factories(
    owned: OwnedGeometryArray,
) -> tuple[bool, object, object, object]:
    """Build lazy row-classification factories for GPU clip results."""
    if owned.device_state is None:
        all_candidate_rows, fast_rows_arr, fallback_rows_arr = _classify_clip_rows_host(owned)

        def _candidate_rows_factory():
            return all_candidate_rows

        def _fast_rows_factory():
            return fast_rows_arr

        def _fallback_rows_factory():
            return fallback_rows_arr

        return bool(fallback_rows_arr.size), _candidate_rows_factory, _fast_rows_factory, _fallback_rows_factory

    import cupy as cp

    d_fast_mask, d_fallback_mask = _materialize_gpu_clip_row_masks(owned)
    has_fallback_rows = any(
        family not in _supported_gpu_clip_row_families() for family in owned.families
    )

    def _materialize_mask_rows(d_mask):
        return cp.asnumpy(
            cp.flatnonzero(d_mask).astype(cp.int32, copy=False)
        ).astype(np.int32, copy=False)  # zcopy:ok(lazy diagnostic row materialization on explicit property access)

    def _fast_rows_factory():
        return _materialize_mask_rows(d_fast_mask)

    def _fallback_rows_factory():
        return _materialize_mask_rows(d_fallback_mask)

    def _candidate_rows_factory():
        fast_rows_arr = _fast_rows_factory()
        if not has_fallback_rows:
            return fast_rows_arr
        fallback_rows_arr = _fallback_rows_factory()
        if fallback_rows_arr.size == 0:
            return fast_rows_arr
        return np.sort(
            np.concatenate([fast_rows_arr, fallback_rows_arr])
        ).astype(np.int32, copy=False)

    return has_fallback_rows, _candidate_rows_factory, _fast_rows_factory, _fallback_rows_factory


def _classify_clip_rows_host(
    owned: OwnedGeometryArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate/fast/fallback row indices on the host."""
    _gpu_family_tag_list = [
        FAMILY_TAGS.get(fam, -99)
        for fam in _supported_gpu_clip_row_families()
        if fam in owned.families
    ]
    gpu_family_mask = np.isin(owned.tags, _gpu_family_tag_list)
    valid_mask = owned.validity
    fast_rows_arr = np.flatnonzero(valid_mask & gpu_family_mask).astype(np.int32)
    fallback_rows_arr = np.flatnonzero(valid_mask & ~gpu_family_mask).astype(np.int32)
    all_candidate_rows = np.sort(
        np.concatenate([fast_rows_arr, fallback_rows_arr])
    ).astype(np.int32)
    return all_candidate_rows, fast_rows_arr, fallback_rows_arr


def _coerce_row_map_device(row_map):
    import cupy as cp

    if row_map is None:
        return None
    if hasattr(row_map, "__cuda_array_interface__"):
        return cp.asarray(row_map, dtype=cp.int32)
    return cp.asarray(np.asarray(row_map, dtype=np.int32), dtype=cp.int32)


def _combine_gpu_clip_owned_results(
    poly_owned: OwnedGeometryArray | None,
    poly_row_map,
    line_result: OwnedGeometryArray | None,
    line_row_map,
) -> tuple[OwnedGeometryArray | None, object | None]:
    """Combine compact polygon and line GPU clip outputs into one device OGA."""
    if poly_owned is None:
        return line_result, line_row_map
    if line_result is None:
        return poly_owned, poly_row_map

    import cupy as cp

    d_poly_row_map = _coerce_row_map_device(poly_row_map)
    d_line_row_map = _coerce_row_map_device(line_row_map)
    if d_poly_row_map is None:
        return line_result, line_row_map
    if d_line_row_map is None:
        return poly_owned, poly_row_map

    total_out = poly_owned.row_count + line_result.row_count
    d_all_row_map = cp.concatenate([d_poly_row_map, d_line_row_map]).astype(cp.int32, copy=False)
    d_order = cp.argsort(d_all_row_map.astype(cp.int64, copy=False))
    d_positions = cp.empty(total_out, dtype=cp.int64)
    d_positions[d_order] = cp.arange(total_out, dtype=cp.int64)

    result = build_null_owned_array(total_out, residency=Residency.DEVICE)
    result = concat_owned_scatter(result, poly_owned, d_positions[:poly_owned.row_count])
    result = concat_owned_scatter(result, line_result, d_positions[poly_owned.row_count:])
    return result, d_all_row_map[d_order].astype(cp.int32, copy=False)


def clip_by_rect_owned(
    values: Sequence[object | None] | np.ndarray | OwnedGeometryArray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> RectClipResult:
    rect = (float(xmin), float(ymin), float(xmax), float(ymax))
    _row_count = values.row_count if isinstance(values, OwnedGeometryArray) else (len(values) if hasattr(values, '__len__') else 0)
    runtime_selection = plan_dispatch_selection(
        kernel_name="clip_by_rect",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=_row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
        current_residency=_clip_dispatch_residency(values),
    )
    has_polygon_families = False
    has_line_families = False
    if isinstance(values, OwnedGeometryArray):
        has_polygon_families = any(
            f in values.families
            for f in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
        )
        has_line_families = any(
            f in values.families
            for f in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
        )
    point_only_owned_gpu = (
        runtime_selection.selected is ExecutionMode.GPU
        and isinstance(values, OwnedGeometryArray)
        and GeometryFamily.POINT in values.families
        and len(values.families) == 1
    )
    polygon_gpu_eligible = (
        runtime_selection.selected is ExecutionMode.GPU
        and isinstance(values, OwnedGeometryArray)
        and has_polygon_families
    )
    line_gpu_eligible = (
        runtime_selection.selected is ExecutionMode.GPU
        and isinstance(values, OwnedGeometryArray)
        and has_line_families
    )
    if point_only_owned_gpu:
        shapely_values = None
        owned = values
    elif polygon_gpu_eligible or line_gpu_eligible:
        shapely_values = None
        owned = values
    elif isinstance(values, OwnedGeometryArray):
        # OwnedGeometryArray on CPU path: defer shapely materialization
        # until after bounds filtering so we only pay the conversion cost
        # for candidate rows (not all rows).
        shapely_values = None
        owned = values
    else:
        shapely_values, owned = _normalize_values(values)
    precision_plan = runtime_selection.precision_plan
    robustness_plan = select_robustness_plan(
        kernel_class=KernelClass.CONSTRUCTIVE,
        precision_plan=precision_plan,
    )
    if runtime_selection.selected is ExecutionMode.GPU:
        if GeometryFamily.POINT in owned.families and len(owned.families) == 1:
            runtime = get_cuda_runtime()
            keep_rows = None
            clipped_x = None
            clipped_y = None
            try:
                keep_rows, clipped_x, clipped_y = _clip_points_rect_gpu_arrays(
                    owned,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    boundary_inclusive=False,
                )
                keep_rows_host = runtime.copy_device_to_host(keep_rows)
                kept_count = int(keep_rows_host.size)

                # Build owned_result directly from device arrays -- no Shapely
                # round-trip.  Ownership of clipped_x/clipped_y transfers to the
                # OwnedGeometryArray so they must NOT be freed in the finally block.
                if kept_count > 0:
                    gpu_owned_result = _build_device_backed_point_output(
                        clipped_x, clipped_y, row_count=kept_count,
                    )
                    # Ownership transferred -- prevent finally-block free.
                    clipped_x = None
                    clipped_y = None
                else:
                    gpu_owned_result = _empty_point_output()

                # Lazily build the Shapely geometries array only when a caller
                # actually reads .geometries (e.g. tests, GeoPandas adapter).
                # The hot path (pipeline_benchmarks) uses .owned_result instead.
                _owned_ref = owned
                _keep_rows_host_ref = keep_rows_host
                _owned_result_ref = gpu_owned_result
                _shapely_values_ref = shapely_values

                def _materialize_geometries():
                    result = (
                        _point_clip_result_template(_owned_ref)
                        if _shapely_values_ref is None
                        else np.asarray(
                            [None if value is None else EMPTY for value in _shapely_values_ref],
                            dtype=object,
                        )
                    )
                    if _keep_rows_host_ref.size:
                        if _shapely_values_ref is not None:
                            result[_keep_rows_host_ref] = _shapely_values_ref[_keep_rows_host_ref]
                        else:
                            # Materialize from owned_result (device -> host -> Shapely)
                            shapely_geoms = _owned_result_ref.to_shapely()
                            result[_keep_rows_host_ref] = np.asarray(shapely_geoms, dtype=object)
                    return result

                return RectClipResult(
                    geometries_factory=_materialize_geometries,
                    row_count=int(owned.row_count),
                    candidate_rows=keep_rows_host,
                    fast_rows=keep_rows_host,
                    fallback_rows=np.asarray([], dtype=np.int32),
                    runtime_selection=runtime_selection,
                    precision_plan=precision_plan,
                    robustness_plan=robustness_plan,
                    owned_result=gpu_owned_result,
                    owned_result_rows=keep_rows_host,
                )
            finally:
                runtime.free(keep_rows)
                runtime.free(clipped_x)
                runtime.free(clipped_y)
        # Batched GPU polygon + line clip path (ADR-0033 Tier 1)
        if has_polygon_families or has_line_families:
            poly_owned = None
            poly_validity_mask = None
            line_result = None

            if has_polygon_families:
                poly_owned, poly_validity_mask = _clip_all_polygons_gpu(
                    owned,
                    rect,
                    runtime_selection,
                    precision_plan,
                )

            if has_line_families:
                line_result, line_global_row_map = _clip_all_lines_gpu(owned, rect)
            else:
                line_result, line_global_row_map = None, None

            if owned.device_state is None:
                all_candidate_rows, fast_rows_arr, fallback_rows_arr = _classify_clip_rows_host(owned)
                has_fallback_rows = bool(fallback_rows_arr.size)
                candidate_rows_factory = None
                fast_rows_factory = None
                fallback_rows_factory = None
            else:
                has_fallback_rows, candidate_rows_factory, fast_rows_factory, fallback_rows_factory = (
                    _build_gpu_clip_row_factories(owned)
                )
                all_candidate_rows = None
                fast_rows_arr = None
                fallback_rows_arr = None

            poly_row_map = None
            if poly_owned is not None:
                if poly_owned.row_count == owned.row_count and poly_validity_mask is None:
                    try:
                        import cupy as cp
                    except ModuleNotFoundError:  # pragma: no cover - GPU path guarded above
                        cp = None
                    poly_row_map = (
                        cp.arange(owned.row_count, dtype=cp.int32)
                        if cp is not None
                        else np.arange(owned.row_count, dtype=np.int32)
                    )
                elif poly_validity_mask is not None:
                    poly_row_map = np.flatnonzero(poly_validity_mask).astype(np.int32, copy=False)

            owned_result, combined_row_map = _combine_gpu_clip_owned_results(
                poly_owned,
                poly_row_map,
                line_result,
                line_global_row_map,
            )
            owned_result_rows = None
            owned_result_rows_factory = None
            if (
                owned_result is not None
                and not has_fallback_rows
                and combined_row_map is not None
            ):
                def _materialize_combined_row_map():
                    try:
                        import cupy as cp
                    except ModuleNotFoundError:  # pragma: no cover - GPU path guarded above
                        cp = None
                    if cp is not None and hasattr(combined_row_map, "__cuda_array_interface__"):
                        return cp.asnumpy(combined_row_map).astype(np.int32, copy=False)  # zcopy:ok(explicit host row metadata for lazy public-result scatter/materialization)
                    return np.asarray(combined_row_map, dtype=np.int32)

                owned_result_rows_factory = _materialize_combined_row_map

            line_result_rows_factory = None
            if line_result is not None and line_result.row_count > 0 and line_global_row_map is not None:
                def _materialize_line_row_map():
                    try:
                        import cupy as cp
                    except ModuleNotFoundError:  # pragma: no cover - GPU path guarded above
                        cp = None
                    if cp is not None and hasattr(line_global_row_map, "__cuda_array_interface__"):
                        return cp.asnumpy(line_global_row_map).astype(  # zcopy:ok(explicit host row metadata for line-result scatter into public geometry array)
                            np.int32,
                            copy=False,
                        )
                    return np.asarray(line_global_row_map, dtype=np.int32)

                line_result_rows_factory = _materialize_line_row_map

            # Capture references for lazy factory
            _owned_ref = owned
            _poly_owned_ref = poly_owned
            _poly_validity_mask_ref = poly_validity_mask
            _line_result_ref = line_result
            _line_global_row_map_ref = line_global_row_map
            _line_result_rows_factory_ref = line_result_rows_factory
            _fallback_rows_arr_ref = fallback_rows_arr
            _fallback_rows_factory_ref = fallback_rows_factory
            _rect_ref = rect
            _line_global_row_map_host = None

            def _line_global_row_map_host_view():
                nonlocal _line_global_row_map_host
                if (
                    _line_global_row_map_host is None
                    and _line_global_row_map_ref is not None
                    and _line_result_rows_factory_ref is not None
                ):
                    _line_global_row_map_host = _line_result_rows_factory_ref()
                return _line_global_row_map_host

            def _materialize_poly_line_geometries():
                result = np.empty(_owned_ref.row_count, dtype=object)
                result[:] = None
                # Vectorized: set all valid rows to EMPTY in one shot
                result[_owned_ref.validity] = EMPTY

                # Materialize polygon results from device-resident OGA
                if _poly_owned_ref is not None:
                    try:
                        poly_shapely = np.asarray(_poly_owned_ref.to_shapely(), dtype=object)
                        if _poly_validity_mask_ref is None and len(poly_shapely) == _owned_ref.row_count:
                            valid_rows = np.flatnonzero(_poly_owned_ref.validity)
                        elif _poly_validity_mask_ref is not None:
                            valid_rows = np.flatnonzero(_poly_validity_mask_ref)
                        else:
                            valid_rows = np.empty(0, dtype=np.int64)
                        if len(poly_shapely) == _owned_ref.row_count:
                            if valid_rows.size > 0:
                                result[valid_rows] = poly_shapely[valid_rows]
                        else:
                            # Compact polygon output: scatter back to global positions.
                            n_poly = min(len(poly_shapely), len(valid_rows))
                            if n_poly > 0:
                                result[valid_rows[:n_poly]] = poly_shapely[:n_poly]
                    except Exception:
                        pass

                # Materialize line results from OwnedGeometryArray using
                # the global_row_map for correct scatter (preserves mapping
                # even when some rows are fully clipped away).
                if _line_result_ref is not None and _line_global_row_map_ref is not None:
                    try:
                        line_global_row_map_host = _line_global_row_map_host_view()
                        line_shapely = _line_result_ref.to_shapely()
                        n_lines = len(line_shapely)
                        if (
                            n_lines == _owned_ref.row_count
                            and _line_result_ref.row_count == _owned_ref.row_count
                        ):
                            valid_rows = np.flatnonzero(_line_result_ref.validity)
                            if valid_rows.size > 0:
                                line_arr = np.asarray(line_shapely, dtype=object)
                                result[valid_rows] = line_arr[valid_rows]
                        elif n_lines > 0 and line_global_row_map_host is not None:
                            result[line_global_row_map_host[:n_lines]] = np.asarray(
                                line_shapely[:n_lines], dtype=object,
                            )
                    except Exception:
                        pass

                # Handle fallback rows (non-polygon, non-line families)
                fallback_rows_arr = (
                    _fallback_rows_arr_ref
                    if _fallback_rows_arr_ref is not None
                    else (
                        _fallback_rows_factory_ref()
                        if _fallback_rows_factory_ref is not None
                        else np.empty(0, dtype=np.int32)
                    )
                )
                if fallback_rows_arr.size > 0:
                    shapely_geoms = np.asarray(_owned_ref.to_shapely(), dtype=object)
                    fallback_shapely = shapely_geoms[fallback_rows_arr]
                    clipped = clip_by_rect_array(
                        np.asarray(fallback_shapely, dtype=object), _rect_ref,
                    )
                    result[fallback_rows_arr] = clipped

                return result

            return RectClipResult(
                geometries_factory=_materialize_poly_line_geometries,
                row_count=int(owned.row_count),
                candidate_rows=all_candidate_rows,
                candidate_rows_factory=candidate_rows_factory,
                fast_rows=fast_rows_arr,
                fast_rows_factory=fast_rows_factory,
                fallback_rows=fallback_rows_arr,
                fallback_rows_factory=fallback_rows_factory,
                runtime_selection=runtime_selection,
                precision_plan=precision_plan,
                robustness_plan=robustness_plan,
                owned_result=owned_result,
                owned_result_rows=owned_result_rows,
                owned_result_rows_factory=owned_result_rows_factory,
            )
        raise NotImplementedError("clip_by_rect GPU variant currently supports point-only, polygon, and line owned arrays")

    # CPU path: delegate to registered CPU kernel variant.
    # Skip the expensive from_shapely_geometries round-trip on the CPU
    # path.  The owned_result field defaults to None and callers that need
    # it (e.g. pipeline_benchmarks) already handle the None case.  This
    # avoids ~19ms of overhead that dominates the CPU clip path.
    result, candidate_rows = _clip_by_rect_cpu(owned, _rect_intersects_bounds, rect, shapely_values)

    return RectClipResult(
        geometries=result,
        row_count=int(owned.row_count),
        candidate_rows=candidate_rows,
        fast_rows=candidate_rows,
        fallback_rows=np.asarray([], dtype=np.int32),
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
    )


def evaluate_geopandas_clip_by_rect(
    values: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    prebuilt_owned: OwnedGeometryArray | None = None,
) -> tuple[OwnedGeometryArray | np.ndarray | None, ExecutionMode]:
    from vibespatial.runtime.execution_trace import execution_trace

    with execution_trace("clip_by_rect"):
        geometries = None if prebuilt_owned is not None else np.asarray(values, dtype=object)
        clip_input = prebuilt_owned if prebuilt_owned is not None else geometries
        dispatch_mode = (
            ExecutionMode.GPU
            if (
                prebuilt_owned is not None
                and prebuilt_owned.residency is Residency.DEVICE
            )
            else ExecutionMode.AUTO
        )
        try:
            result = clip_by_rect_owned(
                clip_input,
                xmin,
                ymin,
                xmax,
                ymax,
                dispatch_mode=dispatch_mode,
            )
        except NotImplementedError:
            return None, ExecutionMode.CPU
        if result.owned_result is not None and result.owned_result_rows is not None:
            owned_result = result.owned_result
            row_map = np.asarray(result.owned_result_rows, dtype=np.int64)
            if (
                owned_result.row_count != result.row_count
                or row_map.size != result.row_count
                or not np.array_equal(row_map, np.arange(result.row_count, dtype=np.int64))
            ):
                base = build_null_owned_array(
                    result.row_count,
                    residency=owned_result.residency,
                )
                owned_result = concat_owned_scatter(
                    base,
                    owned_result,
                    row_map,
                )
            return owned_result, result.runtime_selection.selected
        return np.asarray(result.geometries, dtype=object), result.runtime_selection.selected


def benchmark_clip_by_rect(
    values: Sequence[object | None] | np.ndarray | OwnedGeometryArray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    dataset: str,
) -> RectClipBenchmark:
    # Build owned array once, pass it directly to avoid double conversion.
    if isinstance(values, OwnedGeometryArray):
        owned = values
    else:
        shapely_arr = np.asarray(values, dtype=object)
        owned = from_shapely_geometries(shapely_arr.tolist())

    started = perf_counter()
    result = clip_by_rect_owned(owned, xmin, ymin, xmax, ymax)
    owned_elapsed = perf_counter() - started

    # Shapely baseline: materialize shapely array for the comparison.
    shapely_values = np.asarray(owned.to_shapely(), dtype=object)
    shapely_elapsed = benchmark_clip_by_rect_baseline(
        shapely_values, xmin, ymin, xmax, ymax,
    )

    return RectClipBenchmark(
        dataset=dataset,
        rows=int(owned.row_count),
        candidate_rows=int(result.candidate_rows.size),
        fast_rows=int(result.fast_rows.size),
        fallback_rows=int(result.fallback_rows.size),
        owned_elapsed_seconds=owned_elapsed,
        shapely_elapsed_seconds=shapely_elapsed,
    )
