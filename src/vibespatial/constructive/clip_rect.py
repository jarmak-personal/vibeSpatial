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
    clip_by_rect_cpu as _clip_by_rect_cpu,
)
from vibespatial.constructive.clip_rect_cpu import (
    normalize_values as _normalize_values,
)
from vibespatial.constructive.clip_rect_cpu import (
    polygon_ring_spans as _polygon_ring_spans,
)
from vibespatial.constructive.clip_rect_kernels import (
    _LINE_ROW_KERNEL_NAMES,
    _LINE_ROW_KERNEL_SOURCE,
    _SH_KERNEL_NAMES,
    _SUTHERLAND_HODGMAN_KERNEL_SOURCE,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    build_empty_polygon_rows_device,
    build_null_owned_array,
    concat_owned_scatter,
    device_select_owned_capacity_partitions,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.config import SPATIAL_EPSILON
from vibespatial.runtime.crossover import (
    PhysicalWorkEstimate,
    estimate_physical_work_from_owned,
)
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
)
from vibespatial.runtime.residency import Residency, combined_residency
from vibespatial.runtime.robustness import RobustnessPlan, select_robustness_plan

request_warmup(["exclusive_scan_i32", "exclusive_scan_i64"])

_POINT_EPSILON = SPATIAL_EPSILON
_POINT_TYPE_ID = 0


def _clip_rect_int_scalar(value, *, reason: str) -> int:
    """Read a device scalar through the runtime so profiles see the fence."""
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
        return int(value)

    if hasattr(value, "__cuda_array_interface__") or type(value).__module__.startswith("cupy"):
        d_value = cp.asarray(value).reshape(1)
        host = get_cuda_runtime().copy_device_to_host(d_value, reason=reason)
        return int(np.asarray(host).reshape(-1)[0])
    return int(value)


# ---------------------------------------------------------------------------
# NVRTC kernel compilation helpers
# ---------------------------------------------------------------------------
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup(
    [
        ("sh-clip", _SUTHERLAND_HODGMAN_KERNEL_SOURCE, _SH_KERNEL_NAMES),
        ("line-row-clip", _LINE_ROW_KERNEL_SOURCE, _LINE_ROW_KERNEL_NAMES),
    ]
)


def _compile_sh_kernels():
    return compile_kernel_group("sh-clip", _SUTHERLAND_HODGMAN_KERNEL_SOURCE, _SH_KERNEL_NAMES)


def _compile_line_row_kernels():
    return compile_kernel_group(
        "line-row-clip",
        _LINE_ROW_KERNEL_SOURCE,
        _LINE_ROW_KERNEL_NAMES,
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
        "_owned_result_rows_device",
        "_owned_result_rows_factory",
        "owned_result",
        "owned_result_is_row_capacity",
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
        owned_result_rows_device: object | None = None,
        owned_result_rows_factory: object | None = None,
        owned_result_is_row_capacity: bool = False,
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
        self._owned_result_rows_device = owned_result_rows_device
        self._owned_result_rows_factory = owned_result_rows_factory
        self.owned_result_is_row_capacity = bool(owned_result_is_row_capacity)
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

    @property
    def owned_result_rows_device(self):
        return self._owned_result_rows_device


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


def _rect_intersects_bounds(
    bounds: np.ndarray, rect: tuple[float, float, float, float]
) -> np.ndarray:
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
    unique = {(round(point[0], 12), round(point[1], 12)) for point in deduped[:-1]}
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
    d_out_x, d_out_y, d_full_offsets = _clip_polygon_rings_gpu_device(
        ring_x,
        ring_y,
        ring_offsets,
        rect,
    )
    if d_out_x is None:
        ring_count = len(ring_offsets) - 1
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.zeros(ring_count + 1, dtype=np.int32),
        )
    runtime = get_cuda_runtime()
    out_ring_offsets = runtime.copy_device_to_host(
        d_full_offsets,
        reason="clip-rect polygon-ring full-offset host export",
    )
    logical_vertex_count = int(out_ring_offsets[-1])
    return (
        runtime.copy_device_to_host(
            d_out_x[:logical_vertex_count],
            reason="clip-rect polygon-ring x-coordinate host export",
        ),
        runtime.copy_device_to_host(
            d_out_y[:logical_vertex_count],
            reason="clip-rect polygon-ring y-coordinate host export",
        ),
        out_ring_offsets,
    )


def _clip_polygon_rings_gpu_device(
    ring_x,
    ring_y,
    ring_offsets,
    rect: tuple[float, float, float, float],
):
    """Clip polygon rings on GPU using Sutherland-Hodgman (device I/O).

    Accepts numpy or CuPy arrays for ring_x, ring_y, ring_offsets.
    Returns capacity-sized coordinate arrays plus logical ring offsets as CuPy
    device arrays.  Empty logical results retain the same physical carrier.
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
            float(xmin),
            float(ymin),
            float(xmax),
            float(ymax),
            ring_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
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

    # Build full output offsets (ring_count + 1) entirely on device.
    d_full_offsets = cp.empty(ring_count + 1, dtype=cp.int32)
    d_full_offsets[:ring_count] = cp.asarray(d_out_offsets)
    d_full_offsets[ring_count] = d_out_offsets[-1] + d_vertex_counts[-1]

    # Every output vertex is either a retained source vertex, one of at most
    # two rectangle crossings per source edge, or one of four rectangle
    # corners.  Include one closure slot per ring, then cap the allocation at
    # the kernel's 256 open vertices plus closure.
    source_bound = (2 * int(d_ring_x.size)) + (5 * ring_count)
    kernel_bound = 257 * ring_count
    vertex_capacity = min(source_bound, kernel_bound)
    d_out_x = cp.empty(vertex_capacity, dtype=cp.float64)
    d_out_y = cp.empty(vertex_capacity, dtype=cp.float64)

    clip_params = (
        (
            ptr(d_ring_x),
            ptr(d_ring_y),
            ptr(d_ring_offsets),
            ptr(d_full_offsets),
            ptr(d_out_x),
            ptr(d_out_y),
            float(xmin),
            float(ymin),
            float(xmax),
            float(ymax),
            ring_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
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
        ring_polygon_map,
        ring_is_exterior,
        out_x,
        out_y,
        out_ring_offsets,
    )


# ---------------------------------------------------------------------------
# GPU line clip (Liang-Barsky via NVRTC)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Batched GPU line clip — vectorized segment extraction + reassembly
# ---------------------------------------------------------------------------


def _clip_point_rows_device_capacity_path(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
) -> tuple[OwnedGeometryArray, object] | None:
    """Select point rows at source capacity using device coordinates."""
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - GPU path guarded by caller
        return None

    state = owned._ensure_device_state(preserve_indexed_view=True)
    point_buffer = state.families.get(GeometryFamily.POINT)
    if point_buffer is None:
        return None
    row_count = int(owned.row_count)
    d_point_source = cp.asarray(state.validity, dtype=cp.bool_) & (
        cp.asarray(state.tags, dtype=cp.int8) == cp.int8(FAMILY_TAGS[GeometryFamily.POINT])
    )
    d_family_rows = cp.where(
        d_point_source,
        cp.asarray(state.family_row_offsets, dtype=cp.int32),
        cp.int32(0),
    )
    if int(point_buffer.empty_mask.size) == 0:
        d_nonempty = cp.zeros(row_count, dtype=cp.bool_)
    else:
        d_nonempty = ~cp.asarray(
            point_buffer.empty_mask,
            dtype=cp.bool_,
        )[d_family_rows]
    if int(point_buffer.x.size) == 0:
        d_x = cp.zeros(row_count, dtype=cp.float64)
        d_y = cp.zeros(row_count, dtype=cp.float64)
    else:
        d_coord_rows = cp.asarray(
            point_buffer.geometry_offsets,
            dtype=cp.int32,
        )[d_family_rows]
        d_coord_rows = cp.minimum(
            d_coord_rows,
            cp.int32(int(point_buffer.x.size) - 1),
        )
        d_x = cp.asarray(point_buffer.x, dtype=cp.float64)[d_coord_rows]
        d_y = cp.asarray(point_buffer.y, dtype=cp.float64)[d_coord_rows]
    xmin, ymin, xmax, ymax = rect
    d_keep = d_point_source & d_nonempty & (d_x > xmin) & (d_x < xmax) & (d_y > ymin) & (d_y < ymax)
    output = build_device_resident_owned(
        device_families={GeometryFamily.POINT: point_buffer},
        row_count=row_count,
        tags=cp.full(
            row_count,
            FAMILY_TAGS[GeometryFamily.POINT],
            dtype=cp.int8,
        ),
        validity=d_keep,
        family_row_offsets=d_family_rows,
        execution_mode="gpu",
    )
    return output, cp.arange(row_count, dtype=cp.int32)


def _clip_line_rows_device_capacity_path(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
) -> tuple[OwnedGeometryArray, object] | None:
    """Fuse source-buffer segment traversal and rectangle clip assembly."""
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - GPU path guarded by caller
        return None

    state = owned._ensure_device_state(preserve_indexed_view=True)
    line_families = {
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTILINESTRING,
    }
    if state is None or not set(state.families).intersection(line_families):
        return None

    row_count = int(owned.row_count)
    if row_count == 0:
        return None

    line_buffer = state.families.get(GeometryFamily.LINESTRING)
    multi_buffer = state.families.get(GeometryFamily.MULTILINESTRING)
    dummy_f64 = cp.zeros(1, dtype=cp.float64)
    dummy_i32 = cp.zeros(2, dtype=cp.int32)
    dummy_u8 = cp.ones(1, dtype=cp.uint8)

    def _family_array(buffer, name, dummy):
        if buffer is None:
            return dummy
        value = getattr(buffer, name)
        return dummy if value is None else value

    cached_capacity = getattr(owned, "_clip_rect_line_segment_capacity", None)
    if cached_capacity is not None:
        segment_capacity = int(cached_capacity)
    elif owned.is_indexed_view:
        max_family_coords = max(
            (
                int(buffer.x.size)
                for family, buffer in state.families.items()
                if family in line_families
            ),
            default=0,
        )
        segment_capacity = row_count * max_family_coords
    else:
        segment_capacity = sum(
            int(buffer.x.size)
            for family, buffer in state.families.items()
            if family in line_families
        )
    if 2 * segment_capacity > np.iinfo(np.int32).max:
        return None

    runtime = get_cuda_runtime()
    kernels = _compile_line_row_kernels()
    ptr = runtime.pointer
    d_run_counts = runtime.allocate((row_count,), cp.int32, zero=True)
    d_coord_counts = runtime.allocate((row_count,), cp.int32, zero=True)
    d_has_output = runtime.allocate((row_count,), cp.uint8, zero=True)
    xmin, ymin, xmax, ymax = rect

    source_params = (
        ptr(state.validity),
        ptr(state.tags),
        ptr(state.family_row_offsets),
        ptr(_family_array(line_buffer, "x", dummy_f64)),
        ptr(_family_array(line_buffer, "y", dummy_f64)),
        ptr(_family_array(line_buffer, "geometry_offsets", dummy_i32)),
        ptr(_family_array(line_buffer, "empty_mask", dummy_u8)),
        ptr(_family_array(multi_buffer, "x", dummy_f64)),
        ptr(_family_array(multi_buffer, "y", dummy_f64)),
        ptr(_family_array(multi_buffer, "geometry_offsets", dummy_i32)),
        ptr(_family_array(multi_buffer, "part_offsets", dummy_i32)),
        ptr(_family_array(multi_buffer, "empty_mask", dummy_u8)),
    )
    count_params = (
        source_params
        + (
            ptr(d_run_counts),
            ptr(d_coord_counts),
            ptr(d_has_output),
            float(xmin),
            float(ymin),
            float(xmax),
            float(ymax),
            row_count,
        ),
        (KERNEL_PARAM_PTR,) * 15 + (KERNEL_PARAM_F64,) * 4 + (KERNEL_PARAM_I32,),
    )
    count_grid, count_block = runtime.launch_config(
        kernels["line_rect_capacity_count"],
        row_count,
    )
    runtime.launch(
        kernels["line_rect_capacity_count"],
        grid=count_grid,
        block=count_block,
        params=count_params,
    )

    from vibespatial.cuda.cccl_primitives import exclusive_sum

    d_single_mask = d_run_counts == 1
    d_multi_mask = d_run_counts > 1
    d_single_counts = cp.where(d_single_mask, d_coord_counts, 0).astype(
        cp.int32,
        copy=False,
    )
    d_multi_part_counts = cp.where(d_multi_mask, d_run_counts, 0).astype(
        cp.int32,
        copy=False,
    )
    d_multi_coord_counts = cp.where(d_multi_mask, d_coord_counts, 0).astype(
        cp.int32,
        copy=False,
    )

    def _offsets_with_terminal(d_counts):
        offsets = cp.empty(row_count + 1, dtype=cp.int32)
        offsets[:row_count] = exclusive_sum(d_counts, synchronize=False)
        offsets[row_count] = offsets[row_count - 1] + d_counts[row_count - 1]
        return offsets

    d_single_offsets = _offsets_with_terminal(d_single_counts)
    d_multi_geom_offsets = _offsets_with_terminal(d_multi_part_counts)
    d_multi_coord_offsets = _offsets_with_terminal(d_multi_coord_counts)
    coordinate_capacity = 2 * segment_capacity
    d_single_x = runtime.allocate((coordinate_capacity,), cp.float64)
    d_single_y = runtime.allocate((coordinate_capacity,), cp.float64)
    d_multi_x = runtime.allocate((coordinate_capacity,), cp.float64)
    d_multi_y = runtime.allocate((coordinate_capacity,), cp.float64)
    d_multi_part_offsets = cp.empty(segment_capacity + 1, dtype=cp.int32)
    d_multi_part_offsets[:] = d_multi_coord_offsets[-1]

    scatter_params = (
        source_params
        + (
            ptr(d_run_counts),
            ptr(d_single_offsets),
            ptr(d_multi_geom_offsets),
            ptr(d_multi_coord_offsets),
            ptr(d_multi_part_offsets),
            ptr(d_single_x),
            ptr(d_single_y),
            ptr(d_multi_x),
            ptr(d_multi_y),
            float(xmin),
            float(ymin),
            float(xmax),
            float(ymax),
            row_count,
        ),
        (KERNEL_PARAM_PTR,) * 21 + (KERNEL_PARAM_F64,) * 4 + (KERNEL_PARAM_I32,),
    )
    scatter_grid, scatter_block = runtime.launch_config(
        kernels["line_rect_capacity_scatter"],
        row_count,
    )
    runtime.launch(
        kernels["line_rect_capacity_scatter"],
        grid=scatter_grid,
        block=scatter_block,
        params=scatter_params,
    )

    output = build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                x=d_single_x,
                y=d_single_y,
                geometry_offsets=d_single_offsets,
                empty_mask=~d_single_mask,
                bounds=None,
            ),
            GeometryFamily.MULTILINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.MULTILINESTRING,
                x=d_multi_x,
                y=d_multi_y,
                geometry_offsets=d_multi_geom_offsets,
                empty_mask=~d_multi_mask,
                part_offsets=d_multi_part_offsets,
                bounds=None,
            ),
        },
        row_count=row_count,
        tags=cp.where(
            d_single_mask,
            FAMILY_TAGS[GeometryFamily.LINESTRING],
            FAMILY_TAGS[GeometryFamily.MULTILINESTRING],
        ).astype(cp.int8),
        validity=d_has_output.astype(cp.bool_, copy=False),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    output._clip_rect_line_segment_capacity = segment_capacity
    return output, cp.arange(row_count, dtype=cp.int32)


def _clip_all_lines_gpu(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
) -> tuple[OwnedGeometryArray | None, object | None, bool]:
    """Clip all line-family rows through fused source-row capacity assembly."""
    capacity_result = _clip_line_rows_device_capacity_path(owned, rect)
    if capacity_result is None:
        raise OverflowError("line rectangle clip exceeds the int32 owned-buffer capacity contract")
    return capacity_result[0], capacity_result[1], True


def _clip_all_polygons_gpu(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
) -> tuple[OwnedGeometryArray | None, object | None, bool]:
    """Clip polygon rows through a source-row-capacity topology carrier."""

    import cupy as cp

    polygon_families = {
        family
        for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
        if family in owned.families
    }
    if not polygon_families:
        return None, cp.empty(0, dtype=cp.int32), False

    from vibespatial.constructive.binary_constructive import _binary_constructive_gpu
    from vibespatial.constructive.envelope import _build_device_boxes_from_bounds

    state = owned._ensure_device_state(preserve_indexed_view=True)
    row_count = int(owned.row_count)
    d_polygon_tags = cp.asarray(
        [FAMILY_TAGS[family] for family in polygon_families],
        dtype=cp.int8,
    )
    d_polygon_mask = cp.asarray(state.validity, dtype=cp.bool_) & cp.isin(
        cp.asarray(state.tags, dtype=cp.int8),
        d_polygon_tags,
    )
    default_family = min(polygon_families, key=lambda family: FAMILY_TAGS[family])
    polygon_rows = build_device_resident_owned(
        device_families={family: state.families[family] for family in polygon_families},
        row_count=row_count,
        tags=cp.where(
            d_polygon_mask,
            cp.asarray(state.tags, dtype=cp.int8),
            cp.int8(FAMILY_TAGS[default_family]),
        ),
        validity=d_polygon_mask,
        family_row_offsets=cp.where(
            d_polygon_mask,
            cp.asarray(state.family_row_offsets, dtype=cp.int32),
            cp.int32(0),
        ),
        execution_mode="gpu",
    )
    d_rectangle_bounds = cp.broadcast_to(
        cp.asarray(rect, dtype=cp.float64),
        (row_count, 4),
    ).copy()
    rectangle_rows = _build_device_boxes_from_bounds(
        d_rectangle_bounds,
        row_count=row_count,
    )
    clipped = _binary_constructive_gpu(
        "intersection",
        polygon_rows,
        rectangle_rows,
        dispatch_mode=ExecutionMode.GPU,
    )
    if clipped is None or clipped.row_count != row_count:
        raise RuntimeError(
            "native polygon rectangle topology plan did not return a row-capacity result"
        )

    clipped_state = clipped._ensure_device_state(preserve_indexed_view=True)
    output_families = {
        family: buffer
        for family, buffer in clipped_state.families.items()
        if family in {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    }
    if not output_families:
        return (
            build_null_owned_array(row_count, residency=Residency.DEVICE),
            cp.arange(row_count, dtype=cp.int32),
            True,
        )

    d_output_tags = cp.asarray(clipped_state.tags, dtype=cp.int8)
    d_output_polygon = cp.asarray(clipped_state.validity, dtype=cp.bool_) & cp.isin(
        d_output_tags,
        cp.asarray(
            [FAMILY_TAGS[family] for family in output_families],
            dtype=cp.int8,
        ),
    )
    output_default_family = min(
        output_families,
        key=lambda family: FAMILY_TAGS[family],
    )
    output = build_device_resident_owned(
        device_families=output_families,
        row_count=row_count,
        tags=cp.where(
            d_output_polygon,
            d_output_tags,
            cp.int8(FAMILY_TAGS[output_default_family]),
        ),
        validity=d_output_polygon,
        family_row_offsets=cp.where(
            d_output_polygon,
            cp.asarray(clipped_state.family_row_offsets, dtype=cp.int32),
            cp.int32(0),
        ),
        execution_mode="gpu",
    )
    return output, cp.arange(row_count, dtype=cp.int32), True


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
        GeometryFamily.POINT,
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

        return (
            bool(fallback_rows_arr.size),
            _candidate_rows_factory,
            _fast_rows_factory,
            _fallback_rows_factory,
        )

    import cupy as cp

    d_fast_mask, d_fallback_mask = _materialize_gpu_clip_row_masks(owned)
    has_fallback_rows = any(
        family not in _supported_gpu_clip_row_families() for family in owned.families
    )

    def _materialize_mask_rows(d_mask):
        return (
            get_cuda_runtime()
            .copy_device_to_host(
                cp.flatnonzero(d_mask).astype(cp.int32, copy=False),
                reason="clip-rect lazy diagnostic row mask host export",
            )
            .astype(np.int32, copy=False)
        )  # zcopy:ok(lazy diagnostic row materialization on explicit property access)

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
        return np.sort(np.concatenate([fast_rows_arr, fallback_rows_arr])).astype(
            np.int32, copy=False
        )

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
    all_candidate_rows = np.sort(np.concatenate([fast_rows_arr, fallback_rows_arr])).astype(
        np.int32
    )
    return all_candidate_rows, fast_rows_arr, fallback_rows_arr


def _combine_gpu_clip_capacity_results(
    *parts: OwnedGeometryArray | None,
) -> tuple[OwnedGeometryArray | None, object | None, bool]:
    """Combine source-row-capacity family partitions without compaction."""
    active_parts = [part for part in parts if part is not None]
    if not active_parts:
        return None, None, False

    import cupy as cp

    row_count = int(active_parts[0].row_count)
    for part in active_parts[1:]:
        if part.row_count != row_count:
            raise ValueError("clip capacity partitions must have matching row counts")
    base = build_null_owned_array(row_count, residency=Residency.DEVICE)
    replacements = []
    for part in active_parts:
        state = part._ensure_device_state(preserve_indexed_view=True)
        replacements.append((part, state.validity))
    result = device_select_owned_capacity_partitions(base, replacements)
    return result, cp.arange(row_count, dtype=cp.int32), True


def _clip_rect_work_estimate(values: object, row_count: int) -> PhysicalWorkEstimate:
    """Describe rectangle clipping by source and bounded output coordinates."""
    if not isinstance(values, OwnedGeometryArray):
        return PhysicalWorkEstimate.from_rows(row_count)
    source = estimate_physical_work_from_owned(values)
    output_coordinate_capacity = max(
        source.coordinate_count * 2,
        source.coordinate_count + source.ring_count * 4,
    )
    return PhysicalWorkEstimate(
        row_count=row_count,
        coordinate_count=source.coordinate_count,
        segment_count=source.segment_count,
        ring_count=source.ring_count,
        output_row_count=row_count,
        output_byte_count=output_coordinate_capacity * 16,
        temporary_byte_count=(source.segment_count + source.ring_count) * 8,
        primary_unit_count=max(
            row_count,
            source.coordinate_count,
            source.segment_count,
            output_coordinate_capacity,
        ),
        primary_unit_name="clip-rect-output-coordinate",
    )


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
    _row_count = (
        values.row_count
        if isinstance(values, OwnedGeometryArray)
        else (len(values) if hasattr(values, "__len__") else 0)
    )
    runtime_selection = plan_dispatch_selection(
        kernel_name="clip_by_rect",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=_row_count,
        work_estimate=_clip_rect_work_estimate(values, _row_count),
        requested_mode=dispatch_mode,
        requested_precision=precision,
        current_residency=_clip_dispatch_residency(values),
    )
    has_point_families = False
    has_polygon_families = False
    has_line_families = False
    if isinstance(values, OwnedGeometryArray):
        has_point_families = GeometryFamily.POINT in values.families
        has_polygon_families = any(
            f in values.families for f in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
        )
        has_line_families = any(
            f in values.families
            for f in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
        )
    point_gpu_eligible = (
        runtime_selection.selected is ExecutionMode.GPU
        and isinstance(values, OwnedGeometryArray)
        and has_point_families
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
    if point_gpu_eligible or polygon_gpu_eligible or line_gpu_eligible:
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
        if has_point_families or has_polygon_families or has_line_families:
            point_owned = None
            poly_owned = None
            poly_row_map = None
            line_result = None

            if has_point_families:
                point_result = _clip_point_rows_device_capacity_path(owned, rect)
                if point_result is not None:
                    point_owned, _point_row_map = point_result

            if has_polygon_families:
                (
                    poly_owned,
                    poly_row_map,
                    _poly_result_is_row_capacity,
                ) = _clip_all_polygons_gpu(owned, rect)

            if has_line_families:
                (
                    line_result,
                    _line_global_row_map,
                    _line_result_is_row_capacity,
                ) = _clip_all_lines_gpu(owned, rect)

            if owned.device_state is None:
                all_candidate_rows, fast_rows_arr, fallback_rows_arr = _classify_clip_rows_host(
                    owned
                )
                has_fallback_rows = bool(fallback_rows_arr.size)
                candidate_rows_factory = None
                fast_rows_factory = None
                fallback_rows_factory = None
            else:
                (
                    has_fallback_rows,
                    candidate_rows_factory,
                    fast_rows_factory,
                    fallback_rows_factory,
                ) = _build_gpu_clip_row_factories(owned)
                all_candidate_rows = None
                fast_rows_arr = None
                fallback_rows_arr = None

            (
                owned_result,
                combined_row_map,
                owned_result_is_row_capacity,
            ) = _combine_gpu_clip_capacity_results(
                point_owned,
                poly_owned,
                line_result,
            )
            owned_result_rows = None
            owned_result_rows_factory = None
            if owned_result is not None and not has_fallback_rows and combined_row_map is not None:

                def _materialize_combined_row_map():
                    try:
                        import cupy as cp
                    except ModuleNotFoundError:  # pragma: no cover - GPU path guarded above
                        cp = None
                    if cp is not None and hasattr(combined_row_map, "__cuda_array_interface__"):
                        return (
                            get_cuda_runtime()
                            .copy_device_to_host(
                                combined_row_map,
                                reason="clip-rect combined row-map host export",
                            )
                            .astype(np.int32, copy=False)
                        )  # zcopy:ok(explicit host row metadata for lazy public-result scatter/materialization)
                    return np.asarray(combined_row_map, dtype=np.int32)

                owned_result_rows_factory = _materialize_combined_row_map

            _owned_ref = owned
            _capacity_owned_ref = owned_result
            _fallback_rows_arr_ref = fallback_rows_arr
            _fallback_rows_factory_ref = fallback_rows_factory
            _rect_ref = rect

            def _materialize_capacity_geometries():
                result = np.empty(_owned_ref.row_count, dtype=object)
                result[:] = None
                result[_owned_ref.validity] = EMPTY
                if _capacity_owned_ref is not None:
                    capacity_shapely = np.asarray(
                        _capacity_owned_ref.to_shapely(),
                        dtype=object,
                    )
                    valid_rows = np.flatnonzero(_capacity_owned_ref.validity)
                    if valid_rows.size:
                        result[valid_rows] = capacity_shapely[valid_rows]

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
                        np.asarray(fallback_shapely, dtype=object),
                        _rect_ref,
                    )
                    result[fallback_rows_arr] = clipped

                return result

            return RectClipResult(
                geometries_factory=_materialize_capacity_geometries,
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
                owned_result_rows_device=combined_row_map,
                owned_result_rows_factory=owned_result_rows_factory,
                owned_result_is_row_capacity=(
                    owned_result_is_row_capacity and not has_fallback_rows
                ),
            )
        raise NotImplementedError(
            "clip_by_rect GPU variant currently supports point-only, polygon, and line owned arrays"
        )

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
            if (prebuilt_owned is not None and prebuilt_owned.residency is Residency.DEVICE)
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
        row_map_native = result.owned_result_rows_device
        row_map_host = None
        if row_map_native is None:
            row_map_host = result.owned_result_rows
            row_map_native = row_map_host
        if result.owned_result is not None and row_map_native is not None:
            owned_result = result.owned_result
            row_map_is_device = hasattr(row_map_native, "__cuda_array_interface__")
            if row_map_is_device:
                row_map_size = int(row_map_native.size)
                scatter_rows = row_map_native
                needs_scatter = (
                    owned_result.row_count != result.row_count or row_map_size != result.row_count
                )
            else:
                row_map = np.asarray(row_map_native, dtype=np.int64)
                scatter_rows = row_map
                needs_scatter = (
                    owned_result.row_count != result.row_count
                    or row_map.size != result.row_count
                    or not np.array_equal(
                        row_map,
                        np.arange(result.row_count, dtype=np.int64),
                    )
                )
            empty_capacity = None
            if (
                prebuilt_owned is not None
                and prebuilt_owned.residency is Residency.DEVICE
            ):
                source_state = prebuilt_owned._ensure_device_state(
                    preserve_indexed_view=True,
                )
                empty_rows = build_empty_polygon_rows_device(result.row_count)
                empty_capacity = device_select_owned_capacity_partitions(
                    build_null_owned_array(
                        result.row_count,
                        residency=Residency.DEVICE,
                    ),
                    [(empty_rows, source_state.validity)],
                )
            if needs_scatter:
                base = build_null_owned_array(
                    result.row_count,
                    residency=owned_result.residency,
                )
                if empty_capacity is not None:
                    base = empty_capacity
                owned_result = concat_owned_scatter(
                    base,
                    owned_result,
                    scatter_rows,
                )
            elif empty_capacity is not None:
                result_state = owned_result._ensure_device_state(
                    preserve_indexed_view=True,
                )
                owned_result = device_select_owned_capacity_partitions(
                    empty_capacity,
                    [(owned_result, result_state.validity)],
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
        shapely_values,
        xmin,
        ymin,
        xmax,
        ymax,
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
