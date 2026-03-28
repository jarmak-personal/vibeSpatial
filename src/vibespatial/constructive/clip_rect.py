from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.runtime.adaptive import plan_dispatch_selection, plan_kernel_dispatch
from vibespatial.runtime.crossover import DispatchDecision

request_warmup(["exclusive_scan_i32"])
from vibespatial.cuda._runtime import (  # noqa: E402
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily  # noqa: E402
from vibespatial.geometry.owned import (  # noqa: E402
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    from_shapely_geometries,
)
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds  # noqa: E402
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.runtime.precision import (  # noqa: E402
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    select_precision_plan,
)
from vibespatial.runtime.robustness import RobustnessPlan, select_robustness_plan  # noqa: E402

from .point import (  # noqa: E402
    _build_device_backed_point_output,
    _clip_points_rect_gpu_arrays,
    _empty_point_output,
)

EMPTY = GeometryCollection()
_POINT_EPSILON = 1e-12
_POINT_TYPE_ID = 0

# ---------------------------------------------------------------------------
# GPU clip thresholds (ADR-0033 tier system)
# ---------------------------------------------------------------------------
_POLYGON_CLIP_GPU_THRESHOLD = 1_000
_LINE_CLIP_GPU_THRESHOLD = 1_000

# ---------------------------------------------------------------------------
# GPU Sutherland-Hodgman polygon clip kernel (NVRTC)
# ---------------------------------------------------------------------------
# Per-ring clip against a rectangle.  Each thread processes one ring through
# all four boundary edges sequentially.  The kernel writes clipped vertices
# into a pre-allocated output buffer using per-ring offsets computed via
# exclusive_scan on a vertex-count pass.

_SUTHERLAND_HODGMAN_KERNEL_SOURCE = r"""
#define EPSILON 1e-12

/* Clip a ring against one boundary edge.
   Returns the number of output vertices written to out_x/out_y. */
__device__ int clip_edge(
    const double* in_x, const double* in_y, int in_count,
    double* out_x, double* out_y, int max_out,
    int edge_type,  /* 0=left, 1=right, 2=bottom, 3=top */
    double edge_val
) {
  if (in_count == 0) return 0;
  int out_count = 0;

  double prev_x = in_x[in_count - 1];
  double prev_y = in_y[in_count - 1];

  int prev_inside;
  if (edge_type == 0)      prev_inside = (prev_x >= edge_val) ? 1 : 0;
  else if (edge_type == 1) prev_inside = (prev_x <= edge_val) ? 1 : 0;
  else if (edge_type == 2) prev_inside = (prev_y >= edge_val) ? 1 : 0;
  else                     prev_inside = (prev_y <= edge_val) ? 1 : 0;

  for (int i = 0; i < in_count; i++) {
    double cur_x = in_x[i];
    double cur_y = in_y[i];

    int cur_inside;
    if (edge_type == 0)      cur_inside = (cur_x >= edge_val) ? 1 : 0;
    else if (edge_type == 1) cur_inside = (cur_x <= edge_val) ? 1 : 0;
    else if (edge_type == 2) cur_inside = (cur_y >= edge_val) ? 1 : 0;
    else                     cur_inside = (cur_y <= edge_val) ? 1 : 0;

    if (cur_inside) {
      if (!prev_inside) {
        /* Compute intersection */
        double ix, iy;
        if (edge_type <= 1) {
          /* Vertical edge */
          double dx = cur_x - prev_x;
          if (fabs(dx) <= EPSILON) { ix = edge_val; iy = prev_y; }
          else { double t = (edge_val - prev_x) / dx; ix = edge_val; iy = prev_y + t * (cur_y - prev_y); }
        } else {
          /* Horizontal edge */
          double dy = cur_y - prev_y;
          if (fabs(dy) <= EPSILON) { ix = prev_x; iy = edge_val; }
          else { double t = (edge_val - prev_y) / dy; ix = prev_x + t * (cur_x - prev_x); iy = edge_val; }
        }
        if (out_count < max_out) { out_x[out_count] = ix; out_y[out_count] = iy; out_count++; }
      }
      if (out_count < max_out) { out_x[out_count] = cur_x; out_y[out_count] = cur_y; out_count++; }
    } else if (prev_inside) {
      double ix, iy;
      if (edge_type <= 1) {
        double dx = cur_x - prev_x;
        if (fabs(dx) <= EPSILON) { ix = edge_val; iy = prev_y; }
        else { double t = (edge_val - prev_x) / dx; ix = edge_val; iy = prev_y + t * (cur_y - prev_y); }
      } else {
        double dy = cur_y - prev_y;
        if (fabs(dy) <= EPSILON) { ix = prev_x; iy = edge_val; }
        else { double t = (edge_val - prev_y) / dy; ix = prev_x + t * (cur_x - prev_x); iy = edge_val; }
      }
      if (out_count < max_out) { out_x[out_count] = ix; out_y[out_count] = iy; out_count++; }
    }
    prev_x = cur_x;
    prev_y = cur_y;
    prev_inside = cur_inside;
  }
  return out_count;
}

/* Count output vertices for one ring after Sutherland-Hodgman clipping.
   Each thread handles one ring.  We use shared-memory scratch buffers. */
extern "C" __global__ void sh_count_vertices(
    const double* ring_x,
    const double* ring_y,
    const int* ring_offsets,
    int* out_vertex_counts,
    double xmin, double ymin, double xmax, double ymax,
    int ring_count
) {
  const int ring = blockIdx.x * blockDim.x + threadIdx.x;
  if (ring >= ring_count) { out_vertex_counts[ring >= ring_count ? 0 : ring] = 0; return; }
  if (ring >= ring_count) return;

  const int start = ring_offsets[ring];
  const int end = ring_offsets[ring + 1];
  int n = end - start;

  /* Strip closure vertex */
  if (n >= 2) {
    double dx = ring_x[start] - ring_x[end - 1];
    double dy = ring_y[start] - ring_y[end - 1];
    if (dx * dx + dy * dy < 1e-24) n--;
  }
  if (n < 3) { out_vertex_counts[ring] = 0; return; }

  /* Use local buffers (max reasonable ring size for GPU clip) */
  const int MAX_VERTS = 256;
  double buf_a_x[256], buf_a_y[256];
  double buf_b_x[256], buf_b_y[256];

  if (n > MAX_VERTS) { out_vertex_counts[ring] = 0; return; }

  for (int i = 0; i < n; i++) { buf_a_x[i] = ring_x[start + i]; buf_a_y[i] = ring_y[start + i]; }

  /* Clip against 4 edges: left, right, bottom, top */
  double edges[4] = {xmin, xmax, ymin, ymax};
  int count = n;
  double *src_x = buf_a_x, *src_y = buf_a_y;
  double *dst_x = buf_b_x, *dst_y = buf_b_y;

  for (int e = 0; e < 4; e++) {
    count = clip_edge(src_x, src_y, count, dst_x, dst_y, MAX_VERTS, e, edges[e]);
    if (count == 0) break;
    /* Swap buffers */
    double *tmp;
    tmp = src_x; src_x = dst_x; dst_x = tmp;
    tmp = src_y; src_y = dst_y; dst_y = tmp;
  }

  /* Add closure vertex if result is a valid ring */
  out_vertex_counts[ring] = (count >= 3) ? count + 1 : 0;
}

/* Write clipped vertices for one ring, using pre-computed offsets. */
extern "C" __global__ void sh_clip_rings(
    const double* ring_x,
    const double* ring_y,
    const int* ring_offsets,
    const int* out_offsets,
    double* out_x,
    double* out_y,
    double xmin, double ymin, double xmax, double ymax,
    int ring_count
) {
  const int ring = blockIdx.x * blockDim.x + threadIdx.x;
  if (ring >= ring_count) return;

  const int out_start = out_offsets[ring];
  const int out_end = out_offsets[ring + 1];
  const int expected = out_end - out_start;
  if (expected <= 0) return;

  const int start = ring_offsets[ring];
  const int end = ring_offsets[ring + 1];
  int n = end - start;

  if (n >= 2) {
    double dx = ring_x[start] - ring_x[end - 1];
    double dy = ring_y[start] - ring_y[end - 1];
    if (dx * dx + dy * dy < 1e-24) n--;
  }
  if (n < 3) return;

  const int MAX_VERTS = 256;
  double buf_a_x[256], buf_a_y[256];
  double buf_b_x[256], buf_b_y[256];

  if (n > MAX_VERTS) return;

  for (int i = 0; i < n; i++) { buf_a_x[i] = ring_x[start + i]; buf_a_y[i] = ring_y[start + i]; }

  double edges[4] = {xmin, xmax, ymin, ymax};
  int count = n;
  double *src_x = buf_a_x, *src_y = buf_a_y;
  double *dst_x = buf_b_x, *dst_y = buf_b_y;

  for (int e = 0; e < 4; e++) {
    count = clip_edge(src_x, src_y, count, dst_x, dst_y, MAX_VERTS, e, edges[e]);
    if (count == 0) break;
    double *tmp;
    tmp = src_x; src_x = dst_x; dst_x = tmp;
    tmp = src_y; src_y = dst_y; dst_y = tmp;
  }

  if (count < 3) return;

  /* Write output vertices + closure */
  for (int i = 0; i < count && i < expected - 1; i++) {
    out_x[out_start + i] = src_x[i];
    out_y[out_start + i] = src_y[i];
  }
  /* Closure vertex */
  out_x[out_start + count] = src_x[0];
  out_y[out_start + count] = src_y[0];
}
"""

_SH_KERNEL_NAMES = ("sh_count_vertices", "sh_clip_rings")


# ---------------------------------------------------------------------------
# GPU Liang-Barsky line clip kernel (NVRTC)
# ---------------------------------------------------------------------------
# Per-segment clip against a rectangle.  Each thread processes one segment.

_LIANG_BARSKY_KERNEL_SOURCE = r"""
#define LB_EPSILON 1e-12

extern "C" __global__ void lb_clip_segments(
    const double* seg_x0,
    const double* seg_y0,
    const double* seg_x1,
    const double* seg_y1,
    double* out_x0,
    double* out_y0,
    double* out_x1,
    double* out_y1,
    unsigned char* out_valid,
    double xmin, double ymin, double xmax, double ymax,
    int segment_count
) {
  const int seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= segment_count) return;

  double x0 = seg_x0[seg], y0 = seg_y0[seg];
  double x1 = seg_x1[seg], y1 = seg_y1[seg];
  double dx = x1 - x0, dy = y1 - y0;

  double p[4] = {-dx, dx, -dy, dy};
  double q[4] = {x0 - xmin, xmax - x0, y0 - ymin, ymax - y0};

  double u1 = 0.0, u2 = 1.0;
  for (int k = 0; k < 4; k++) {
    if (fabs(p[k]) <= LB_EPSILON) {
      if (q[k] < 0.0) { out_valid[seg] = 0; return; }
      continue;
    }
    double t = q[k] / p[k];
    if (p[k] < 0.0) { if (t > u1) u1 = t; }
    else             { if (t < u2) u2 = t; }
    if (u1 > u2) { out_valid[seg] = 0; return; }
  }

  double cx0 = x0 + u1 * dx, cy0 = y0 + u1 * dy;
  double cx1 = x0 + u2 * dx, cy1 = y0 + u2 * dy;

  /* Reject degenerate segments */
  double ddx = cx0 - cx1, ddy = cy0 - cy1;
  if (ddx * ddx + ddy * ddy < LB_EPSILON * LB_EPSILON) {
    out_valid[seg] = 0;
    return;
  }

  out_x0[seg] = cx0;
  out_y0[seg] = cy0;
  out_x1[seg] = cx1;
  out_y1[seg] = cy1;
  out_valid[seg] = 1;
}
"""

_LB_KERNEL_NAMES = ("lb_clip_segments",)


# ---------------------------------------------------------------------------
# NVRTC kernel compilation helpers
# ---------------------------------------------------------------------------

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("sh-clip", _SUTHERLAND_HODGMAN_KERNEL_SOURCE, _SH_KERNEL_NAMES),
    ("lb-clip", _LIANG_BARSKY_KERNEL_SOURCE, _LB_KERNEL_NAMES),
])


def _compile_sh_kernels():
    return compile_kernel_group("sh-clip", _SUTHERLAND_HODGMAN_KERNEL_SOURCE, _SH_KERNEL_NAMES)


def _compile_lb_kernels():
    return compile_kernel_group("lb-clip", _LIANG_BARSKY_KERNEL_SOURCE, _LB_KERNEL_NAMES)


class RectClipResult:
    """Result of a rectangle clip operation.

    ``geometries`` is lazily materialized from ``owned_result`` when accessed
    for the first time on the GPU point path, avoiding D->H->Shapely overhead
    unless a caller actually needs Shapely objects.
    """

    __slots__ = (
        "_geometries",
        "_geometries_factory",
        "candidate_rows",
        "fallback_rows",
        "fast_rows",
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
        candidate_rows: np.ndarray,
        fast_rows: np.ndarray,
        fallback_rows: np.ndarray,
        runtime_selection: RuntimeSelection,
        precision_plan: PrecisionPlan,
        robustness_plan: RobustnessPlan,
        owned_result: OwnedGeometryArray | None = None,
    ):
        self._geometries = geometries
        self._geometries_factory = geometries_factory
        self.row_count = row_count
        self.candidate_rows = candidate_rows
        self.fast_rows = fast_rows
        self.fallback_rows = fallback_rows
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


def _normalize_values(values: Sequence[object | None] | np.ndarray | OwnedGeometryArray) -> tuple[np.ndarray, OwnedGeometryArray]:
    if isinstance(values, OwnedGeometryArray):
        shapely_values = np.asarray(values.to_shapely(), dtype=object)
        return shapely_values, values
    shapely_values = np.asarray(values, dtype=object)
    return shapely_values, from_shapely_geometries(shapely_values.tolist())


def _point_clip_result_template(owned: OwnedGeometryArray) -> np.ndarray:
    result = np.empty(owned.row_count, dtype=object)
    result[:] = EMPTY
    result[~owned.validity] = None
    return result


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


def _liang_barsky_segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rect: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    xmin, ymin, xmax, ymax = rect
    dx = x1 - x0
    dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u1 = 0.0
    u2 = 1.0
    for pk, qk in zip(p, q, strict=True):
        if abs(pk) <= _POINT_EPSILON:
            if qk < 0.0:
                return None
            continue
        t = qk / pk
        if pk < 0.0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)
        if u1 > u2:
            return None
    cx0 = x0 + u1 * dx
    cy0 = y0 + u1 * dy
    cx1 = x0 + u2 * dx
    cy1 = y0 + u2 * dy
    if np.allclose((cx0, cy0), (cx1, cy1), atol=_POINT_EPSILON, rtol=0.0):
        return None
    return float(cx0), float(cy0), float(cx1), float(cy1)


def _merge_clipped_segments(parts: list[list[tuple[float, float]]], segment: tuple[float, float, float, float]) -> None:
    start = (segment[0], segment[1])
    end = (segment[2], segment[3])
    if not parts:
        parts.append([start, end])
        return
    current = parts[-1]
    if np.allclose(current[-1], start, atol=_POINT_EPSILON, rtol=0.0):
        if not np.allclose(current[-1], end, atol=_POINT_EPSILON, rtol=0.0):
            current.append(end)
        return
    parts.append([start, end])


def _build_linestring_result(parts: list[list[tuple[float, float]]]) -> object:
    normalized = [part for part in parts if len(part) >= 2]
    if not normalized:
        return EMPTY
    if len(normalized) == 1:
        return LineString(normalized[0])
    return MultiLineString(normalized)


def _build_polygon_result(polygons: list[Polygon]) -> object:
    non_empty = [polygon for polygon in polygons if not polygon.is_empty]
    if not non_empty:
        return EMPTY
    if len(non_empty) == 1:
        return non_empty[0]
    return shapely.multipolygons(non_empty)


def _clip_point_family(buffer, family_row: int, rect: tuple[float, float, float, float]) -> object:
    start = int(buffer.geometry_offsets[family_row])
    end = int(buffer.geometry_offsets[family_row + 1])
    if end <= start:
        return EMPTY
    xmin, ymin, xmax, ymax = rect
    if buffer.family.value == "point":
        x = float(buffer.x[start])
        y = float(buffer.y[start])
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return Point(x, y)
        return EMPTY

    points = [
        (float(buffer.x[index]), float(buffer.y[index]))
        for index in range(start, end)
        if xmin <= float(buffer.x[index]) <= xmax and ymin <= float(buffer.y[index]) <= ymax
    ]
    if not points:
        return EMPTY
    if len(points) == 1:
        return Point(points[0])
    return MultiPoint(points)


def _clip_line_family(buffer, family_row: int, rect: tuple[float, float, float, float]) -> object:
    parts: list[list[tuple[float, float]]] = []
    if buffer.family.value == "linestring":
        spans = [(int(buffer.geometry_offsets[family_row]), int(buffer.geometry_offsets[family_row + 1]))]
    else:
        part_start = int(buffer.geometry_offsets[family_row])
        part_end = int(buffer.geometry_offsets[family_row + 1])
        spans = [
            (int(buffer.part_offsets[index]), int(buffer.part_offsets[index + 1]))
            for index in range(part_start, part_end)
        ]
    for start, end in spans:
        part_segments: list[list[tuple[float, float]]] = []
        for index in range(start, end - 1):
            segment = _liang_barsky_segment(
                float(buffer.x[index]),
                float(buffer.y[index]),
                float(buffer.x[index + 1]),
                float(buffer.y[index + 1]),
                rect,
            )
            if segment is None:
                continue
            _merge_clipped_segments(part_segments, segment)
        parts.extend(part_segments)
    return _build_linestring_result(parts)


def _polygon_ring_spans(buffer, family_row: int) -> list[list[tuple[float, float]]]:
    def ring_lookup(ring_index: int) -> tuple[int, int]:
        return int(buffer.ring_offsets[ring_index]), int(buffer.ring_offsets[ring_index + 1])

    if buffer.family.value == "polygon":
        polygon_indices = [(int(buffer.geometry_offsets[family_row]), int(buffer.geometry_offsets[family_row + 1]))]
    else:
        polygon_start = int(buffer.geometry_offsets[family_row])
        polygon_end = int(buffer.geometry_offsets[family_row + 1])
        polygon_indices = [
            (int(buffer.part_offsets[polygon_index]), int(buffer.part_offsets[polygon_index + 1]))
            for polygon_index in range(polygon_start, polygon_end)
        ]

    polygons: list[list[tuple[float, float]]] = []
    for ring_start, ring_end in polygon_indices:
        rings = []
        for ring_index in range(ring_start, ring_end):
            coord_start, coord_end = ring_lookup(ring_index)
            rings.append(
                [
                    (float(buffer.x[index]), float(buffer.y[index]))
                    for index in range(coord_start, coord_end)
                ]
            )
        polygons.append(rings)
    return polygons


def _clip_polygon_family(buffer, family_row: int, rect: tuple[float, float, float, float]) -> object:
    polygons: list[Polygon] = []
    for rings in _polygon_ring_spans(buffer, family_row):
        if not rings:
            continue
        exterior = _sutherland_hodgman_ring(rings[0], rect)
        if not exterior:
            continue
        holes = []
        for ring in rings[1:]:
            clipped = _sutherland_hodgman_ring(ring, rect)
            if clipped:
                holes.append(clipped)
        polygon = Polygon(exterior, holes=holes)
        if not polygon.is_empty:
            polygons.append(polygon)
    return _build_polygon_result(polygons)


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
    runtime.synchronize()

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

    # Reconstruct polygons from clipped rings
    polygons: list[Polygon] = []
    current_exterior = None
    current_holes: list[list[tuple[float, float]]] = []

    for ring_idx in range(len(ring_polygon_map)):
        start = int(out_ring_offsets[ring_idx])
        end = int(out_ring_offsets[ring_idx + 1])
        verts = end - start
        if verts < 4:
            # Ring was clipped away
            if ring_is_exterior[ring_idx]:
                # Flush previous polygon if any
                if current_exterior is not None:
                    poly = Polygon(current_exterior, holes=current_holes)
                    if not poly.is_empty:
                        polygons.append(poly)
                current_exterior = None
                current_holes = []
            continue

        coords = [(float(out_x[i]), float(out_y[i])) for i in range(start, end)]

        if ring_is_exterior[ring_idx]:
            # Flush previous polygon
            if current_exterior is not None:
                poly = Polygon(current_exterior, holes=current_holes)
                if not poly.is_empty:
                    polygons.append(poly)
            current_exterior = coords
            current_holes = []
        else:
            current_holes.append(coords)

    # Flush last polygon
    if current_exterior is not None:
        poly = Polygon(current_exterior, holes=current_holes)
        if not poly.is_empty:
            polygons.append(poly)

    return _build_polygon_result(polygons)


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

    Accepts host numpy arrays for input segments.  Returns
    (d_out_x0, d_out_y0, d_out_x1, d_out_y1, d_valid) as CuPy device
    arrays.  No D2H transfer; kernel outputs stay on device.
    """
    import cupy as cp

    from vibespatial.cuda._runtime import KERNEL_PARAM_F64, KERNEL_PARAM_I32, KERNEL_PARAM_PTR

    runtime = get_cuda_runtime()
    xmin, ymin, xmax, ymax = rect
    segment_count = len(seg_x0)

    # Upload inputs to device via CuPy (managed by memory pool)
    d_x0 = cp.asarray(np.ascontiguousarray(seg_x0, dtype=np.float64))
    d_y0 = cp.asarray(np.ascontiguousarray(seg_y0, dtype=np.float64))
    d_x1 = cp.asarray(np.ascontiguousarray(seg_x1, dtype=np.float64))
    d_y1 = cp.asarray(np.ascontiguousarray(seg_y1, dtype=np.float64))

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Vectorized extraction of segments from line-family buffers.

    Uses numpy offset arithmetic and fancy indexing on contiguous
    buffer.x / buffer.y arrays.  No per-row Python loops -- all span
    computation and coordinate gathering are bulk numpy operations.

    Returns
    -------
    seg_x0, seg_y0, seg_x1, seg_y1 : flat segment endpoint arrays (float64)
    row_segment_offsets : cumulative segment count per row (int32, length = len(global_row_indices) + 1)
    part_segment_offsets : cumulative segment count per part (int32, length = total_parts + 1)
    part_row_map : which row index (into global_row_indices) each part belongs to (int32)
    global_row_indices : ordered list of global row indices that have line data
    """
    # Accumulate per-family vectorized results, then concatenate once.
    all_span_starts: list[np.ndarray] = []
    all_span_ends: list[np.ndarray] = []
    all_span_family_idx: list[np.ndarray] = []
    all_part_row_map: list[np.ndarray] = []
    global_row_indices_parts: list[np.ndarray] = []
    family_buffers: list[object] = []
    row_base = 0  # running offset into global_row_indices

    for family in line_families:
        buffer = owned.families[family]
        tag = FAMILY_TAGS[family]
        family_rows = np.flatnonzero(owned.tags == tag)
        fam_idx = len(family_buffers)
        family_buffers.append(buffer)

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
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            global_row_indices,
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
            np.zeros(len(seg_counts) + 1, dtype=np.int32),
            part_row_map_arr,
            global_row_indices,
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

    # Gather coordinates.  When all spans come from a single family
    # buffer (common case), do a single fancy-index gather.  Otherwise
    # gather per-family and scatter into the output.
    seg_x0 = np.empty(total_segments, dtype=np.float64)
    seg_y0 = np.empty(total_segments, dtype=np.float64)
    seg_x1 = np.empty(total_segments, dtype=np.float64)
    seg_y1 = np.empty(total_segments, dtype=np.float64)

    unique_fam = np.unique(span_family_idx)
    if len(unique_fam) == 1:
        # Fast path: single family buffer.
        buf = family_buffers[int(unique_fam[0])]
        seg_x0[:] = buf.x[gather_p0]
        seg_y0[:] = buf.y[gather_p0]
        seg_x1[:] = buf.x[gather_p1]
        seg_y1[:] = buf.y[gather_p1]
    else:
        # Multi-family: gather per family using segment-level mask.
        seg_fam_idx = np.repeat(span_family_idx, seg_counts)
        for fi in unique_fam:
            fi_int = int(fi)
            buf = family_buffers[fi_int]
            mask = seg_fam_idx == fi_int
            idx0 = gather_p0[mask]
            idx1 = gather_p1[mask]
            seg_x0[mask] = buf.x[idx0]
            seg_y0[mask] = buf.y[idx0]
            seg_x1[mask] = buf.x[idx1]
            seg_y1[mask] = buf.y[idx1]

    # Build per-row segment offsets from part data (np.bincount — truly vectorized C loop).
    row_segment_offsets = np.zeros(n_rows + 1, dtype=np.int32)
    if seg_counts.size > 0:
        row_segment_offsets[1:] = np.bincount(
            part_row_map_arr, weights=seg_counts.astype(np.float64), minlength=n_rows,
        ).astype(np.int32)
    np.cumsum(row_segment_offsets, out=row_segment_offsets)

    return (
        seg_x0, seg_y0, seg_x1, seg_y1,
        row_segment_offsets, part_segment_offsets,
        part_row_map_arr, global_row_indices,
    )


def _build_line_clip_device_result(
    d_out_x0,
    d_out_y0,
    d_out_x1,
    d_out_y1,
    d_valid,
    part_segment_offsets: np.ndarray,
    part_row_map: np.ndarray,
    global_row_indices: list[int],
) -> tuple[OwnedGeometryArray | None, np.ndarray | None]:
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
    part_segment_offsets : host numpy array (int32)
        Cumulative segment count per part (small metadata, stays on host).
    part_row_map : host numpy array (int32)
        Maps each part index to a row index into global_row_indices.
    global_row_indices : list of int
        Maps row indices to global row positions in the input OwnedGeometryArray.

    Returns
    -------
    (OwnedGeometryArray, global_row_map) or (None, None).
    global_row_map is a host numpy int32 array mapping each compact OGA row
    to its global row position in the input, preserving correct scatter
    targets even when some rows are fully clipped away.
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

    # Part boundaries: map each valid segment index to its part, then
    # detect where consecutive valid segments cross part boundaries.
    # part_segment_offsets is small metadata (n_parts+1 ints), upload once.
    d_part_offsets = cp.asarray(part_segment_offsets)

    # Upload part_row_map and global_row_indices for row mapping (small metadata).
    d_part_row_map = cp.asarray(part_row_map)
    d_global_row_indices = cp.asarray(
        np.asarray(global_row_indices, dtype=np.int32)
    )

    # For each valid segment, find which part it belongs to via searchsorted.
    # searchsorted(offsets[1:], idx, side='right') gives the part index.
    d_part_ids = cp.searchsorted(d_part_offsets[1:], d_valid_indices, side="right")

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

        # Part boundary break: different parts
        d_part_break = d_part_ids[:-1] != d_part_ids[1:]

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
    #    For each output linestring, find the part_id of its first
    #    segment, then map: part_id -> part_row_map -> global_row_indices.
    # ---------------------------------------------------------------
    d_run_part_ids = d_part_ids[d_run_starts]
    d_run_row_indices = d_part_row_map[d_run_part_ids]
    d_global_row_map = d_global_row_indices[d_run_row_indices]

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
        # Rebuild geometry offsets for valid geometries only
        d_valid_lengths = d_geom_lengths[d_valid_geom_idx]
        d_new_geom_offsets = cp.empty(valid_geom_count + 1, dtype=cp.int32)
        d_new_geom_offsets[0] = 0
        d_new_geom_offsets[1:] = cp.cumsum(d_valid_lengths).astype(cp.int32)

        # Build gather indices on device -- no Python loop, no D2H.
        # For each valid geometry, we need coord indices [start, ..., end-1].
        # Use CuPy repeat + arange pattern: repeat each geom's start offset
        # by its length, then add a per-element counter within each run.
        d_starts = d_geom_offsets[d_valid_geom_idx]
        # Derive total from cumsum result — no additional sync
        d_total_new = int(d_new_geom_offsets[valid_geom_count])
        # Per-coord offsets within each geometry run
        d_intra = cp.arange(d_total_new, dtype=cp.int64)
        d_run_offsets = cp.repeat(
            d_new_geom_offsets[:valid_geom_count],
            d_valid_lengths,
        ).astype(cp.int64)
        d_base = cp.repeat(d_starts, d_valid_lengths).astype(cp.int64)
        d_gather = d_base + (d_intra - d_run_offsets)
        d_flat_x = d_flat_x[d_gather]
        d_flat_y = d_flat_y[d_gather]
        d_geom_offsets = d_new_geom_offsets
        # Filter the global row map to match compacted geometries
        d_global_row_map = d_global_row_map[d_valid_geom_idx]
        line_count = valid_geom_count

    # Transfer global_row_map to host (small metadata -- single D2H).
    global_row_map = cp.asnumpy(d_global_row_map).astype(np.int32)

    # ---------------------------------------------------------------
    # 7. Build device-resident OwnedGeometryArray
    # ---------------------------------------------------------------
    output_empty_mask = np.zeros(line_count, dtype=np.bool_)
    output_validity = np.ones(line_count, dtype=np.bool_)
    output_tags = np.full(line_count, FAMILY_TAGS[GeometryFamily.LINESTRING], dtype=np.int8)
    output_family_row_offsets = np.arange(line_count, dtype=np.int32)

    # Upload small metadata arrays to device for DeviceFamilyGeometryBuffer
    d_empty_mask = cp.asarray(output_empty_mask)

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
    )
    return oga, global_row_map


def _clip_all_lines_gpu(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
) -> tuple[OwnedGeometryArray | None, np.ndarray | None]:
    """Batch-clip ALL linestring/multilinestring rows on GPU via Liang-Barsky.

    TRUE ZERO-COPY implementation:
      1. Extract segments from buffers via vectorized slicing (host, fast metadata)
      2. Run Liang-Barsky GPU kernel -- outputs stay on device (CuPy)
      3. Assemble coordinate arrays on device (CuPy + CCCL exclusive_sum)
      4. Return device-resident OwnedGeometryArray -- no Shapely, no D2H transfers

    Returns (OwnedGeometryArray, global_row_map) or (None, None).
    global_row_map is a host int32 array mapping each compact OGA row to
    its global row position in the input OwnedGeometryArray, preserving
    correct scatter targets when some rows are fully clipped away.
    """
    line_families = [
        f for f in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
        if f in owned.families
    ]

    if not line_families:
        return None, None

    # -- Step 1: vectorized segment extraction (host, metadata-level) ------
    (
        seg_x0, seg_y0, seg_x1, seg_y1,
        row_segment_offsets, part_segment_offsets,
        part_row_map, global_row_indices,
    ) = _extract_segments_vectorized(owned, line_families)

    total_segments = len(seg_x0)
    if total_segments == 0:
        return None, None

    # -- Step 2: GPU kernel -- outputs stay on device as CuPy arrays ------
    d_out_x0, d_out_y0, d_out_x1, d_out_y1, d_valid = _clip_line_segments_gpu_device(
        seg_x0, seg_y0, seg_x1, seg_y1, rect,
    )

    # -- Step 3: device-resident assembly (CuPy + CCCL, zero numpy) -------
    return _build_line_clip_device_result(
        d_out_x0, d_out_y0, d_out_x1, d_out_y1, d_valid,
        part_segment_offsets,
        part_row_map,
        global_row_indices,
    )


def _row_family_and_local_index(owned: OwnedGeometryArray, row_index: int):
    if not owned.validity[row_index]:
        return None, -1
    for family_name, tag in FAMILY_TAGS.items():
        if int(owned.tags[row_index]) == tag:
            return family_name, int(owned.family_row_offsets[row_index])
    return None, -1


def _supported_fast_row(
    family_name: str | None,
    local_index: int,
    owned: OwnedGeometryArray,
    shapely_value: object | None,
) -> bool:
    if family_name is None or local_index < 0 or shapely_value is None:
        return False
    if shapely_value.is_empty:
        return True
    if family_name in {"point", "multipoint", "linestring", "multilinestring"}:
        return True
    if family_name in {"polygon", "multipolygon"} and shapely.is_valid(shapely_value):
        return True
    return False


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

    Constructs GeoArrow offset arrays on host (they are small metadata) and
    keeps coordinates on device.  No Shapely construction, no per-element
    Python loops.

    Returns (owned_result, validity_mask) where validity_mask is a bool array
    of length input_row_count indicating which global rows have valid output.
    owned_result contains only the valid output polygons (compact rows).
    """
    import cupy as cp

    runtime = get_cuda_runtime()
    h_full_offsets = cp.asnumpy(d_full_offsets)

    # ---- Vectorized ring survival analysis ----
    # Replaces the O(total_rings) Python loop with bulk numpy operations.
    total_rings = len(ring_is_exterior)

    if total_rings == 0:
        validity_mask = np.zeros(input_row_count, dtype=bool)
        return None, validity_mask

    # Per-ring vertex counts from the GPU output offsets.
    ring_verts = np.diff(h_full_offsets[:total_rings + 1]).astype(np.int32)

    # Exterior ring validity: an exterior ring survives if it has >= 4 verts.
    is_ext = ring_is_exterior.astype(bool)
    ext_positions = np.flatnonzero(is_ext)
    ext_valid = ring_verts[ext_positions] >= 4  # bool per exterior ring

    # Forward-fill exterior validity to subsequent holes.  Each exterior
    # ring's validity applies to itself and all following holes until the
    # next exterior ring.  We compute segment lengths from the gaps between
    # consecutive exterior positions and use np.repeat to broadcast.
    if len(ext_positions) > 0:
        seg_lengths = np.empty(len(ext_positions), dtype=np.int32)
        seg_lengths[:-1] = np.diff(ext_positions)
        seg_lengths[-1] = total_rings - int(ext_positions[-1])
        ext_validity_filled = np.repeat(ext_valid, seg_lengths)
        # Handle any rings before the first exterior (shouldn't happen in
        # well-formed data, but guard: mark them invalid).
        if int(ext_positions[0]) > 0:
            prefix = np.zeros(int(ext_positions[0]), dtype=bool)
            ext_validity_filled = np.concatenate([prefix, ext_validity_filled])
    else:
        ext_validity_filled = np.zeros(total_rings, dtype=bool)

    # A ring survives if: (a) its exterior is valid, AND (b) it has >= 4 verts.
    ring_survives = ext_validity_filled & (ring_verts >= 4)

    # ---- Per-geometry aggregation ----
    # Compute per-geometry ring counts and survival using the offset array.
    n_geoms = len(geom_ring_offsets) - 1
    geom_ring_counts = np.diff(geom_ring_offsets)
    has_rings = geom_ring_counts > 0

    # For each geometry, count how many of its rings survived.
    # Use np.add.reduceat on ring_survives grouped by geom_ring_offsets.
    if total_rings > 0 and has_rings.any():
        # reduceat needs valid split points; filter to non-empty geometries.
        ne_starts = geom_ring_offsets[:-1][has_rings]
        surviving_per_ne_geom = np.add.reduceat(ring_survives.astype(np.int32), ne_starts)
        # Map back to all geometries.
        surviving_per_geom = np.zeros(n_geoms, dtype=np.int32)
        surviving_per_geom[has_rings] = surviving_per_ne_geom
    else:
        surviving_per_geom = np.zeros(n_geoms, dtype=np.int32)

    geom_has_output = surviving_per_geom > 0

    # Global row indices for geometries with output.
    global_rows_arr = np.asarray(global_row_indices, dtype=np.intp)
    output_valid_rows_arr = global_rows_arr[geom_has_output]
    output_row_count = int(output_valid_rows_arr.size)

    # Build validity mask for the caller (covers all input rows).
    validity_mask = np.zeros(input_row_count, dtype=bool)
    if output_row_count > 0:
        validity_mask[output_valid_rows_arr] = True

    if output_row_count == 0:
        return None, validity_mask

    # ---- Build GeoArrow offsets for surviving rings ----
    # Surviving ring indices (flat).
    surviving_idx = np.flatnonzero(ring_survives)
    surviving_verts = ring_verts[surviving_idx]

    # Ring offsets: cumulative vertex counts of surviving rings.
    h_ring_offsets = np.empty(len(surviving_verts) + 1, dtype=np.int32)
    h_ring_offsets[0] = 0
    np.cumsum(surviving_verts, out=h_ring_offsets[1:])

    # Geometry offsets: cumulative surviving-ring counts per output geometry.
    output_surviving_counts = surviving_per_geom[geom_has_output]
    h_geom_offsets = np.empty(output_row_count + 1, dtype=np.int32)
    h_geom_offsets[0] = 0
    np.cumsum(output_surviving_counts, out=h_geom_offsets[1:])

    # ---- Build gather indices vectorially ----
    # For each surviving ring, gather coordinates from
    # h_full_offsets[ring_idx] .. h_full_offsets[ring_idx+1].
    gather_starts = h_full_offsets[surviving_idx].astype(np.int64)
    gather_lens = surviving_verts.astype(np.int64)
    total_gather = int(gather_lens.sum())

    h_gather = np.repeat(gather_starts, gather_lens)
    within_offsets = np.arange(total_gather, dtype=np.int64)
    range_bases = np.repeat(
        np.concatenate(([np.int64(0)], np.cumsum(gather_lens[:-1]))),
        gather_lens,
    )
    h_gather += within_offsets - range_bases
    d_gather = cp.asarray(h_gather)
    gathered_x = d_out_x[d_gather]
    gathered_y = d_out_y[d_gather]

    output_empty_mask = np.zeros(output_row_count, dtype=bool)

    output_validity = np.ones(output_row_count, dtype=bool)
    output_tags = np.full(output_row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    output_family_row_offsets = np.arange(output_row_count, dtype=np.int32)

    device_families = {
        GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=gathered_x,
            y=gathered_y,
            geometry_offsets=runtime.from_host(h_geom_offsets),
            empty_mask=runtime.from_host(output_empty_mask),
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
        ),
        validity_mask,
    )


def _clip_all_polygons_gpu(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
    precision_plan: PrecisionPlan,
) -> tuple[OwnedGeometryArray | None, np.ndarray]:
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
    (contains only the valid output polygons), and validity_mask is a bool
    array of length owned.row_count indicating which global rows have output.
    """

    polygon_families = [
        f for f in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
        if f in owned.families
    ]
    empty_validity = np.zeros(owned.row_count, dtype=bool)
    if not polygon_families:
        return None, empty_validity

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


def _materialize_candidates_vectorized(
    owned: OwnedGeometryArray,
    candidate_rows: np.ndarray,
) -> np.ndarray:
    """Build a shapely object array for candidate rows using vectorized
    constructors (shapely.linestrings / shapely.polygons / shapely.points)
    when the candidates are homogeneous, falling back to per-row
    materialization otherwise.

    This is 3-4x faster than ``owned.to_shapely()`` + indexing because it
    avoids Python-level per-row dispatch and exploits shapely's bulk C paths.
    """
    from vibespatial.geometry.owned import TAG_FAMILIES, _materialize_family_row

    if candidate_rows.size == 0:
        return np.empty(0, dtype=object)

    # Determine if all candidates share a single family.
    tags = owned.tags[candidate_rows]
    unique_tags = np.unique(tags)

    if len(unique_tags) == 1:
        tag = int(unique_tags[0])
        family = TAG_FAMILIES[tag]
        buffer = owned.families[family]
        local_rows = owned.family_row_offsets[candidate_rows].astype(np.int32)

        if family in (GeometryFamily.LINESTRING,):
            # Vectorized shapely.linestrings from flat coordinate buffers.
            offsets = np.empty(len(local_rows) + 1, dtype=np.int32)
            offsets[0] = 0
            for idx, lr in enumerate(local_rows):
                s = int(buffer.geometry_offsets[lr])
                e = int(buffer.geometry_offsets[lr + 1])
                offsets[idx + 1] = offsets[idx] + (e - s)
            total_coords = int(offsets[-1])
            flat_xy = np.empty((total_coords, 2), dtype=np.float64)
            pos = 0
            for lr in local_rows:
                s = int(buffer.geometry_offsets[lr])
                e = int(buffer.geometry_offsets[lr + 1])
                n = e - s
                flat_xy[pos:pos + n, 0] = buffer.x[s:e]
                flat_xy[pos:pos + n, 1] = buffer.y[s:e]
                pos += n
            indices = np.repeat(np.arange(len(local_rows), dtype=np.int32), np.diff(offsets))
            return shapely.linestrings(flat_xy, indices=indices)

        if family in (GeometryFamily.POLYGON,):
            # Vectorized path: build linearrings via shapely.linearrings,
            # then group them into polygons via shapely.polygons(indices=).
            ring_coords: list[float] = []
            ring_coord_offsets = [0]
            ring_geom_indices: list[int] = []
            for geom_idx, lr in enumerate(local_rows):
                ring_start = int(buffer.geometry_offsets[lr])
                ring_end = int(buffer.geometry_offsets[lr + 1])
                for ring_idx in range(ring_start, ring_end):
                    cs = int(buffer.ring_offsets[ring_idx])
                    ce = int(buffer.ring_offsets[ring_idx + 1])
                    n = ce - cs
                    for ci in range(cs, ce):
                        ring_coords.append(float(buffer.x[ci]))
                        ring_coords.append(float(buffer.y[ci]))
                    ring_coord_offsets.append(ring_coord_offsets[-1] + n)
                    ring_geom_indices.append(geom_idx)
            flat_xy = np.asarray(ring_coords, dtype=np.float64).reshape(-1, 2)
            coord_offsets = np.asarray(ring_coord_offsets, dtype=np.int32)
            ring_indices_arr = np.repeat(
                np.arange(len(coord_offsets) - 1, dtype=np.int32),
                np.diff(coord_offsets),
            )
            rings = shapely.linearrings(flat_xy, indices=ring_indices_arr)
            geom_indices_arr = np.asarray(ring_geom_indices, dtype=np.int32)
            return shapely.polygons(rings, indices=geom_indices_arr)

        if family in (GeometryFamily.POINT,):
            xs = np.empty(len(local_rows), dtype=np.float64)
            ys = np.empty(len(local_rows), dtype=np.float64)
            for idx, lr in enumerate(local_rows):
                s = int(buffer.geometry_offsets[lr])
                xs[idx] = buffer.x[s]
                ys[idx] = buffer.y[s]
            return np.asarray(shapely.points(xs, ys), dtype=object)

    # Fallback: per-row materialization for mixed families or unsupported
    # types (multilinestring, multipolygon, multipoint).
    candidate_geoms = []
    for i in candidate_rows:
        family = TAG_FAMILIES[int(owned.tags[i])]
        buf = owned.families[family]
        local_row = int(owned.family_row_offsets[i])
        candidate_geoms.append(_materialize_family_row(buf, local_row))
    return np.asarray(candidate_geoms, dtype=object)


def _use_gpu_clip(owned: OwnedGeometryArray) -> bool:
    """Check if GPU clip kernels should be used based on ADR-0033 tier thresholds."""
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return False
    total_coords = 0
    for buffer in owned.families.values():
        if hasattr(buffer, 'x') and buffer.x is not None:
            total_coords += len(buffer.x)
    return total_coords >= _POLYGON_CLIP_GPU_THRESHOLD


# ---------------------------------------------------------------------------
# CPU clip path (called from clip_by_rect_owned; public variant registration
# lives in kernels/constructive/clip_rect.py)
# ---------------------------------------------------------------------------

def _clip_by_rect_cpu(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
    shapely_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU clip_by_rect using Shapely with bounds pre-filtering.

    Returns (result_geometries, candidate_rows).
    """
    bounds = compute_geometry_bounds(owned)
    candidate_rows = np.flatnonzero(
        _rect_intersects_bounds(bounds, rect)
    ).astype(np.int32, copy=False)

    result = np.empty(owned.row_count, dtype=object)
    result[:] = EMPTY
    invalid_mask = ~owned.validity
    if invalid_mask.any():
        result[invalid_mask] = None

    if candidate_rows.size > 0:
        if shapely_values is not None:
            candidate_shapely = shapely_values[candidate_rows]
        else:
            candidate_shapely = _materialize_candidates_vectorized(
                owned, candidate_rows
            )
        clipped = shapely.clip_by_rect(candidate_shapely, *rect)
        result[candidate_rows] = np.asarray(clipped, dtype=object)

    return result, candidate_rows


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
    precision_plan = select_precision_plan(
        runtime_selection=runtime_selection,
        kernel_class=KernelClass.CONSTRUCTIVE,
        requested=precision,
    )
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
                poly_owned, poly_validity_mask = _clip_all_polygons_gpu(owned, rect, precision_plan)

            if has_line_families:
                line_result, line_global_row_map = _clip_all_lines_gpu(owned, rect)
            else:
                line_result, line_global_row_map = None, None

            # Identify GPU-fast family tags -- vectorized classification
            _gpu_family_tag_list = [
                FAMILY_TAGS.get(fam, -99)
                for fam in (
                    GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON,
                    GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING,
                )
                if fam in owned.families
            ]
            gpu_family_mask = np.isin(owned.tags, _gpu_family_tag_list)
            valid_mask = owned.validity
            fast_rows_arr = np.flatnonzero(valid_mask & gpu_family_mask).astype(np.int32)
            fallback_rows_arr = np.flatnonzero(valid_mask & ~gpu_family_mask).astype(np.int32)
            all_candidate_rows = np.sort(
                np.concatenate([fast_rows_arr, fallback_rows_arr])
            ).astype(np.int32)

            # Prefer the polygon OGA as primary owned_result; when only
            # lines are present, carry the line OGA for zero-copy consumers.
            owned_result = poly_owned if poly_owned is not None else line_result

            # Capture references for lazy factory
            _owned_ref = owned
            _poly_owned_ref = poly_owned
            _poly_validity_mask_ref = poly_validity_mask
            _line_result_ref = line_result
            _line_global_row_map_ref = line_global_row_map
            _fallback_rows_arr_ref = fallback_rows_arr
            _rect_ref = rect

            def _materialize_poly_line_geometries():
                result = np.empty(_owned_ref.row_count, dtype=object)
                result[:] = None
                # Vectorized: set all valid rows to EMPTY in one shot
                result[_owned_ref.validity] = EMPTY

                # Materialize polygon results from device-resident OGA
                if _poly_owned_ref is not None and _poly_validity_mask_ref is not None:
                    try:
                        poly_shapely = _poly_owned_ref.to_shapely()
                        valid_rows = np.flatnonzero(_poly_validity_mask_ref)
                        # poly_owned has compact rows; scatter back to global positions
                        n_poly = min(len(poly_shapely), len(valid_rows))
                        if n_poly > 0:
                            result[valid_rows[:n_poly]] = np.asarray(
                                poly_shapely[:n_poly], dtype=object,
                            )
                    except Exception:
                        pass

                # Materialize line results from OwnedGeometryArray using
                # the global_row_map for correct scatter (preserves mapping
                # even when some rows are fully clipped away).
                if _line_result_ref is not None and _line_global_row_map_ref is not None:
                    try:
                        line_shapely = _line_result_ref.to_shapely()
                        n_lines = len(line_shapely)
                        if n_lines > 0:
                            result[_line_global_row_map_ref[:n_lines]] = np.asarray(
                                line_shapely[:n_lines], dtype=object,
                            )
                    except Exception:
                        pass

                # Handle fallback rows (non-polygon, non-line families)
                if _fallback_rows_arr_ref.size > 0:
                    shapely_geoms = _owned_ref.to_shapely()
                    fallback_shapely = shapely_geoms[_fallback_rows_arr_ref]
                    clipped = shapely.clip_by_rect(
                        np.asarray(fallback_shapely, dtype=object), *_rect_ref,
                    )
                    result[_fallback_rows_arr_ref] = np.asarray(clipped, dtype=object)

                return result

            return RectClipResult(
                geometries_factory=_materialize_poly_line_geometries,
                row_count=int(owned.row_count),
                candidate_rows=all_candidate_rows,
                fast_rows=fast_rows_arr,
                fallback_rows=fallback_rows_arr,
                runtime_selection=runtime_selection,
                precision_plan=precision_plan,
                robustness_plan=robustness_plan,
                owned_result=owned_result,
            )
        raise NotImplementedError("clip_by_rect GPU variant currently supports point-only, polygon, and line owned arrays")

    # CPU path: delegate to registered CPU kernel variant.
    # Skip the expensive from_shapely_geometries round-trip on the CPU
    # path.  The owned_result field defaults to None and callers that need
    # it (e.g. pipeline_benchmarks) already handle the None case.  This
    # avoids ~19ms of overhead that dominates the CPU clip path.
    result, candidate_rows = _clip_by_rect_cpu(owned, rect, shapely_values)

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
) -> tuple[np.ndarray | None, ExecutionMode]:
    from vibespatial.runtime.execution_trace import execution_trace

    with execution_trace("clip_by_rect"):
        geometries = np.asarray(values, dtype=object)
        non_null = np.fromiter((geometry is not None for geometry in geometries), dtype=bool, count=len(geometries))
        gpu_available = False
        if np.any(non_null):
            type_ids = np.asarray(shapely.get_type_id(geometries[non_null]), dtype=np.int32)
            gpu_available = bool(np.all(type_ids == _POINT_TYPE_ID))

        plan = plan_kernel_dispatch(
            kernel_name="clip_by_rect",
            kernel_class=KernelClass.CONSTRUCTIVE,
            row_count=len(geometries),
            gpu_available=gpu_available,
        )
        dispatch_mode = ExecutionMode.GPU if plan.dispatch_decision is DispatchDecision.GPU else ExecutionMode.CPU
        try:
            clip_input = prebuilt_owned if prebuilt_owned is not None else geometries
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
        return np.asarray(result.geometries, dtype=object), dispatch_mode


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
    started = perf_counter()
    shapely.clip_by_rect(shapely_values, xmin, ymin, xmax, ymax)
    shapely_elapsed = perf_counter() - started

    return RectClipBenchmark(
        dataset=dataset,
        rows=int(owned.row_count),
        candidate_rows=int(result.candidate_rows.size),
        fast_rows=int(result.fast_rows.size),
        fallback_rows=int(result.fallback_rows.size),
        owned_elapsed_seconds=owned_elapsed,
        shapely_elapsed_seconds=shapely_elapsed,
    )
