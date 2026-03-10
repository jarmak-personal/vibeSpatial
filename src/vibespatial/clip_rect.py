from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point, Polygon

from vibespatial.adaptive_runtime import plan_dispatch_selection, plan_kernel_dispatch
from vibespatial.cccl_precompile import request_warmup
from vibespatial.crossover import DispatchDecision

request_warmup(["exclusive_scan_i32"])
from vibespatial.geometry_buffers import GeometryFamily  # noqa: E402
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds  # noqa: E402
from vibespatial.owned_geometry import FAMILY_TAGS, OwnedGeometryArray, from_shapely_geometries  # noqa: E402
from vibespatial.point_constructive import _clip_points_rect_gpu_arrays  # noqa: E402
from vibespatial.precision import KernelClass, PrecisionMode, PrecisionPlan, select_precision_plan  # noqa: E402
from vibespatial.robustness import RobustnessPlan, select_robustness_plan  # noqa: E402
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.cuda_runtime import compile_kernel_group, count_scatter_total, get_cuda_runtime  # noqa: E402


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

from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402
request_nvrtc_warmup([
    ("sh-clip", _SUTHERLAND_HODGMAN_KERNEL_SOURCE, _SH_KERNEL_NAMES),
    ("lb-clip", _LIANG_BARSKY_KERNEL_SOURCE, _LB_KERNEL_NAMES),
])


def _compile_sh_kernels():
    return compile_kernel_group("sh-clip", _SUTHERLAND_HODGMAN_KERNEL_SOURCE, _SH_KERNEL_NAMES)


def _compile_lb_kernels():
    return compile_kernel_group("lb-clip", _LIANG_BARSKY_KERNEL_SOURCE, _LB_KERNEL_NAMES)


@dataclass(frozen=True)
class RectClipResult:
    geometries: np.ndarray
    row_count: int
    candidate_rows: np.ndarray
    fast_rows: np.ndarray
    fallback_rows: np.ndarray
    runtime_selection: RuntimeSelection
    precision_plan: PrecisionPlan
    robustness_plan: RobustnessPlan
    owned_result: OwnedGeometryArray | None = None


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
    """Clip polygon rings on GPU using Sutherland-Hodgman.

    Returns (clipped_x, clipped_y, clipped_ring_offsets).
    """
    from vibespatial.cccl_primitives import exclusive_sum
    from vibespatial.cuda_runtime import KERNEL_PARAM_F64, KERNEL_PARAM_I32, KERNEL_PARAM_PTR

    runtime = get_cuda_runtime()
    xmin, ymin, xmax, ymax = rect
    ring_count = len(ring_offsets) - 1

    d_ring_x = runtime.from_host(np.ascontiguousarray(ring_x, dtype=np.float64))
    d_ring_y = runtime.from_host(np.ascontiguousarray(ring_y, dtype=np.float64))
    d_ring_offsets = runtime.from_host(np.ascontiguousarray(ring_offsets, dtype=np.int32))
    d_vertex_counts = runtime.allocate((ring_count,), np.int32)

    kernels = _compile_sh_kernels()
    ptr = runtime.pointer

    try:
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
        if ring_count > 0:
            total_verts = count_scatter_total(runtime, d_vertex_counts, d_out_offsets)
        else:
            total_verts = 0

        if total_verts == 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.zeros(ring_count + 1, dtype=np.int32),
            )

        # Build full output offsets (ring_count + 1) entirely on device —
        # avoids the D2H + modify + H2D round trip.
        try:
            import cupy as _cp
        except ModuleNotFoundError:
            _cp = None
        d_full_offsets = runtime.allocate((ring_count + 1,), np.int32)
        if _cp is not None:
            _cp.copyto(_cp.asarray(d_full_offsets[:ring_count]), _cp.asarray(d_out_offsets))
            d_full_offsets[ring_count] = total_verts
        else:
            h_out_offsets = runtime.copy_device_to_host(d_out_offsets)
            full_offsets_h = np.empty(ring_count + 1, dtype=np.int32)
            full_offsets_h[:-1] = h_out_offsets
            full_offsets_h[-1] = total_verts
            runtime.copy_host_to_device(full_offsets_h, d_full_offsets)

        # Pass 2: Write clipped vertices
        d_out_x = runtime.allocate((total_verts,), np.float64)
        d_out_y = runtime.allocate((total_verts,), np.float64)

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

        out_x = runtime.copy_device_to_host(d_out_x)
        out_y = runtime.copy_device_to_host(d_out_y)
        full_offsets = runtime.copy_device_to_host(d_full_offsets)

        return out_x, out_y, full_offsets

    finally:
        runtime.free(d_ring_x)
        runtime.free(d_ring_y)
        runtime.free(d_ring_offsets)
        runtime.free(d_vertex_counts)
        try:
            runtime.free(d_out_offsets)
        except Exception:
            pass
        try:
            runtime.free(d_full_offsets)
        except Exception:
            pass
        try:
            runtime.free(d_out_x)
        except Exception:
            pass
        try:
            runtime.free(d_out_y)
        except Exception:
            pass


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
    """Clip line segments on GPU using Liang-Barsky.

    Returns (out_x0, out_y0, out_x1, out_y1, valid_mask) with valid_mask
    indicating which segments survived clipping.
    """
    from vibespatial.cuda_runtime import KERNEL_PARAM_F64, KERNEL_PARAM_I32, KERNEL_PARAM_PTR

    runtime = get_cuda_runtime()
    xmin, ymin, xmax, ymax = rect
    segment_count = len(seg_x0)

    d_x0 = runtime.from_host(np.ascontiguousarray(seg_x0, dtype=np.float64))
    d_y0 = runtime.from_host(np.ascontiguousarray(seg_y0, dtype=np.float64))
    d_x1 = runtime.from_host(np.ascontiguousarray(seg_x1, dtype=np.float64))
    d_y1 = runtime.from_host(np.ascontiguousarray(seg_y1, dtype=np.float64))
    d_out_x0 = runtime.allocate((segment_count,), np.float64)
    d_out_y0 = runtime.allocate((segment_count,), np.float64)
    d_out_x1 = runtime.allocate((segment_count,), np.float64)
    d_out_y1 = runtime.allocate((segment_count,), np.float64)
    d_valid = runtime.allocate((segment_count,), np.uint8)

    kernels = _compile_lb_kernels()
    ptr = runtime.pointer

    try:
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
        runtime.synchronize()

        h_out_x0 = runtime.copy_device_to_host(d_out_x0)
        h_out_y0 = runtime.copy_device_to_host(d_out_y0)
        h_out_x1 = runtime.copy_device_to_host(d_out_x1)
        h_out_y1 = runtime.copy_device_to_host(d_out_y1)
        h_valid = runtime.copy_device_to_host(d_valid)

        return h_out_x0, h_out_y0, h_out_x1, h_out_y1, h_valid

    finally:
        runtime.free(d_x0)
        runtime.free(d_y0)
        runtime.free(d_x1)
        runtime.free(d_y1)
        runtime.free(d_out_x0)
        runtime.free(d_out_y0)
        runtime.free(d_out_x1)
        runtime.free(d_out_y1)
        runtime.free(d_valid)


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

def _extract_all_polygon_rings(
    owned: OwnedGeometryArray,
    polygon_families: list[GeometryFamily],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Extract ALL polygon rings across ALL polygon-family rows into flat arrays.

    Returns
    -------
    ring_x, ring_y : flat coordinate arrays (float64)
    ring_offsets : offsets into x/y for each ring (int32, length = total_rings + 1)
    ring_geom_map : which global row index each ring belongs to (int32)
    ring_is_exterior : 1 if exterior ring, 0 if hole (int32)
    geom_ring_offsets : offsets into ring arrays for each geometry row (int32)
    global_row_indices : ordered list of global row indices that have polygon data
    """
    flat_x: list[float] = []
    flat_y: list[float] = []
    ring_offsets_list: list[int] = [0]
    ring_geom_map_list: list[int] = []
    ring_is_exterior_list: list[int] = []
    geom_ring_offsets_list: list[int] = [0]
    global_row_indices: list[int] = []

    for family in polygon_families:
        buffer = owned.families[family]
        tag = FAMILY_TAGS[family]
        # Find all global row indices that belong to this family
        family_rows = np.flatnonzero(owned.tags == tag)
        # Skip this family if buffer data arrays are empty
        if buffer.row_count == 0 or buffer.geometry_offsets is None or len(buffer.geometry_offsets) < 2:
            for global_row in family_rows:
                global_row_indices.append(int(global_row))
                geom_ring_offsets_list.append(len(ring_geom_map_list))
            continue

        for global_row in family_rows:
            local_row = int(owned.family_row_offsets[global_row])
            if buffer.empty_mask.size > local_row and buffer.empty_mask[local_row]:
                # Empty geometry — no rings to clip
                global_row_indices.append(int(global_row))
                geom_ring_offsets_list.append(len(ring_geom_map_list))
                continue

            # Get polygon parts for this row
            if family == GeometryFamily.POLYGON:
                # geometry_offsets points into ring_offsets
                ring_start_idx = int(buffer.geometry_offsets[local_row])
                ring_end_idx = int(buffer.geometry_offsets[local_row + 1])
                # All rings belong to one polygon; first ring is exterior
                for ring_idx in range(ring_start_idx, ring_end_idx):
                    coord_start = int(buffer.ring_offsets[ring_idx])
                    coord_end = int(buffer.ring_offsets[ring_idx + 1])
                    for ci in range(coord_start, coord_end):
                        flat_x.append(float(buffer.x[ci]))
                        flat_y.append(float(buffer.y[ci]))
                    ring_offsets_list.append(len(flat_x))
                    ring_geom_map_list.append(int(global_row))
                    ring_is_exterior_list.append(1 if ring_idx == ring_start_idx else 0)
            else:
                # MultiPolygon: geometry_offsets -> part_offsets -> ring_offsets
                part_start = int(buffer.geometry_offsets[local_row])
                part_end = int(buffer.geometry_offsets[local_row + 1])
                for part_idx in range(part_start, part_end):
                    ring_start_idx = int(buffer.part_offsets[part_idx])
                    ring_end_idx = int(buffer.part_offsets[part_idx + 1])
                    for ring_idx in range(ring_start_idx, ring_end_idx):
                        coord_start = int(buffer.ring_offsets[ring_idx])
                        coord_end = int(buffer.ring_offsets[ring_idx + 1])
                        for ci in range(coord_start, coord_end):
                            flat_x.append(float(buffer.x[ci]))
                            flat_y.append(float(buffer.y[ci]))
                        ring_offsets_list.append(len(flat_x))
                        ring_geom_map_list.append(int(global_row))
                        ring_is_exterior_list.append(1 if ring_idx == ring_start_idx else 0)

            global_row_indices.append(int(global_row))
            geom_ring_offsets_list.append(len(ring_geom_map_list))

    return (
        np.asarray(flat_x, dtype=np.float64),
        np.asarray(flat_y, dtype=np.float64),
        np.asarray(ring_offsets_list, dtype=np.int32),
        np.asarray(ring_geom_map_list, dtype=np.int32),
        np.asarray(ring_is_exterior_list, dtype=np.int32),
        np.asarray(geom_ring_offsets_list, dtype=np.int32),
        global_row_indices,
    )


def _clip_all_polygons_gpu(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
    precision_plan: PrecisionPlan,
) -> tuple[np.ndarray, OwnedGeometryArray | None]:
    """Batch-clip ALL polygon/multipolygon rows on GPU in a single kernel launch.

    Uses the count-scatter pattern (ADR-0033 Tier 1):
      1. Extract all rings from all polygon families
      2. Upload to device
      3. sh_count_vertices across ALL rings (one thread per ring)
      4. exclusive_scan for output offsets
      5. sh_clip_rings to scatter clipped coordinates
      6. Reassemble into OwnedGeometryArray directly from device arrays

    Parameters
    ----------
    owned : OwnedGeometryArray with polygon/multipolygon families
    rect : (xmin, ymin, xmax, ymax) clip rectangle
    precision_plan : PrecisionPlan for observability (CONSTRUCTIVE class, stays fp64)

    Returns
    -------
    (result_geometries, owned_result) where result_geometries is an object array
    of shapely geometries at the correct global row indices, and owned_result is
    the OwnedGeometryArray built from clipped coordinates.
    """

    polygon_families = [
        f for f in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
        if f in owned.families
    ]
    if not polygon_families:
        return np.empty(0, dtype=object), None

    # Step 1: Extract all rings
    (
        ring_x, ring_y, ring_offsets,
        ring_geom_map, ring_is_exterior,
        geom_ring_offsets, global_row_indices,
    ) = _extract_all_polygon_rings(owned, polygon_families)

    total_rings = len(ring_offsets) - 1
    if total_rings == 0:
        # All polygon rows are empty
        result = np.empty(owned.row_count, dtype=object)
        result[:] = None
        for row in global_row_indices:
            result[row] = EMPTY
        return result, None

    # Step 2: Upload to device and run batched clip
    out_x, out_y, out_ring_offsets = _clip_polygon_rings_gpu(
        ring_x, ring_y, ring_offsets, rect,
    )

    # Step 3: Reassemble clipped rings into shapely Polygons per geometry row
    result = np.empty(owned.row_count, dtype=object)
    result[:] = None

    # For building the output OwnedGeometryArray, collect all valid polygons
    all_clipped_polygons: list[object] = []

    for geom_idx, global_row in enumerate(global_row_indices):
        ring_start = int(geom_ring_offsets[geom_idx])
        ring_end = int(geom_ring_offsets[geom_idx + 1])

        if ring_start == ring_end:
            result[global_row] = EMPTY
            continue

        # Collect polygons for this geometry row
        polygons: list[Polygon] = []
        current_exterior = None
        current_holes: list[list[tuple[float, float]]] = []

        for ring_idx in range(ring_start, ring_end):
            out_start = int(out_ring_offsets[ring_idx])
            out_end = int(out_ring_offsets[ring_idx + 1])
            verts = out_end - out_start

            is_exterior = bool(ring_is_exterior[ring_idx])

            if verts < 4:
                # Ring was clipped away
                if is_exterior:
                    if current_exterior is not None:
                        poly = Polygon(current_exterior, holes=current_holes)
                        if not poly.is_empty:
                            polygons.append(poly)
                    current_exterior = None
                    current_holes = []
                continue

            coords = [
                (float(out_x[i]), float(out_y[i]))
                for i in range(out_start, out_end)
            ]

            if is_exterior:
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

        geom = _build_polygon_result(polygons)
        result[global_row] = geom
        if geom is not EMPTY and not (hasattr(geom, 'is_empty') and geom.is_empty):
            all_clipped_polygons.append(geom)

    # Build OwnedGeometryArray from clipped polygons (no shapely round-trip for
    # the kernel work — only for final OwnedGeometryArray construction which
    # needs the shapely-based builder)
    owned_result = None
    if all_clipped_polygons:
        owned_result = from_shapely_geometries(all_clipped_polygons)

    return result, owned_result


def _use_gpu_clip(owned: OwnedGeometryArray) -> bool:
    """Check if GPU clip kernels should be used based on ADR-0033 tier thresholds."""
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return False
    total_coords = 0
    for _name, buffer in owned.families.items():
        if hasattr(buffer, 'x') and buffer.x is not None:
            total_coords += len(buffer.x)
    return total_coords >= _POLYGON_CLIP_GPU_THRESHOLD


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
    if isinstance(values, OwnedGeometryArray):
        has_polygon_families = any(
            f in values.families
            for f in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
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
    if point_only_owned_gpu:
        shapely_values = None
        owned = values
    elif polygon_gpu_eligible:
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
                result = (
                    _point_clip_result_template(owned)
                    if shapely_values is None
                    else np.asarray([None if value is None else EMPTY for value in shapely_values], dtype=object)
                )
                if keep_rows_host.size:
                    if shapely_values is not None:
                        result[keep_rows_host] = shapely_values[keep_rows_host]
                    else:
                        clipped_points = shapely.points(
                            runtime.copy_device_to_host(clipped_x),
                            runtime.copy_device_to_host(clipped_y),
                        )
                        result[keep_rows_host] = np.asarray(clipped_points, dtype=object)
                # Build owned_result from kept points (ADR-0005: stay device-resident)
                kept_geoms = [g for g in result if g is not None and not (hasattr(g, "is_empty") and g.is_empty)]
                gpu_owned_result = from_shapely_geometries(kept_geoms) if kept_geoms else from_shapely_geometries([shapely.Point(0, 0)])
                return RectClipResult(
                    geometries=result,
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
        # Batched GPU polygon clip path (ADR-0033 Tier 1)
        if has_polygon_families:
            poly_result, poly_owned = _clip_all_polygons_gpu(owned, rect, precision_plan)

            # Start from None/EMPTY template
            result = np.empty(owned.row_count, dtype=object)
            result[:] = None
            for i in range(owned.row_count):
                if owned.validity[i]:
                    result[i] = EMPTY

            # Fill in polygon results from GPU clip
            for i in range(owned.row_count):
                if poly_result[i] is not None:
                    result[i] = poly_result[i]

            # Handle non-polygon families (points, lines) via shapely fallback
            non_polygon_rows = []
            for i in range(owned.row_count):
                if not owned.validity[i]:
                    continue
                tag = int(owned.tags[i])
                family = None
                for fam, ftag in FAMILY_TAGS.items():
                    if ftag == tag:
                        family = fam
                        break
                if family not in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
                    non_polygon_rows.append(i)

            if non_polygon_rows:
                # Materialize shapely for non-polygon rows only
                shapely_geoms = owned.to_shapely()
                non_poly_arr = np.asarray(non_polygon_rows, dtype=np.int32)
                non_poly_shapely = np.asarray(
                    [shapely_geoms[i] for i in non_polygon_rows], dtype=object
                )
                if non_poly_arr.size > 0:
                    clipped = shapely.clip_by_rect(non_poly_shapely, *rect)
                    for idx, row in enumerate(non_polygon_rows):
                        result[row] = clipped[idx]

            # Determine fast/fallback row arrays
            poly_family_rows = np.asarray(
                [i for i in range(owned.row_count)
                 if owned.validity[i] and int(owned.tags[i]) in
                 [FAMILY_TAGS.get(GeometryFamily.POLYGON, -99),
                  FAMILY_TAGS.get(GeometryFamily.MULTIPOLYGON, -99)]],
                dtype=np.int32,
            )
            fallback_rows = np.asarray(non_polygon_rows, dtype=np.int32)
            all_candidate_rows = np.sort(
                np.concatenate([poly_family_rows, fallback_rows])
            ).astype(np.int32)

            # Build owned_result
            kept = [
                g for g in result
                if g is not None and not (hasattr(g, "is_empty") and g.is_empty)
                   and (not hasattr(g, "geom_type") or g.geom_type != "GeometryCollection")
            ]
            for g in result:
                if g is not None and hasattr(g, "geom_type") and g.geom_type == "GeometryCollection":
                    for part in g.geoms:
                        if part.geom_type in ("Point", "LineString", "Polygon",
                                              "MultiPoint", "MultiLineString", "MultiPolygon"):
                            kept.append(part)
            owned_result = from_shapely_geometries(kept) if kept else (
                poly_owned if poly_owned is not None
                else from_shapely_geometries([shapely.Point(0, 0)])
            )

            return RectClipResult(
                geometries=result,
                row_count=int(owned.row_count),
                candidate_rows=all_candidate_rows,
                fast_rows=poly_family_rows,
                fallback_rows=fallback_rows,
                runtime_selection=runtime_selection,
                precision_plan=precision_plan,
                robustness_plan=robustness_plan,
                owned_result=owned_result,
            )
        raise NotImplementedError("clip_by_rect GPU variant currently supports point-only and polygon owned arrays")

    assert shapely_values is not None
    bounds = compute_geometry_bounds(owned)
    candidate_rows = np.flatnonzero(_rect_intersects_bounds(bounds, rect)).astype(np.int32, copy=False)

    result = np.asarray([None if value is None else EMPTY for value in shapely_values], dtype=object)

    # Use shapely's vectorized clip_by_rect on all candidates at once.
    # This is shapely's optimized C implementation and is faster than
    # per-row Python dispatch for large arrays (ADR-0033: prefer vectorized
    # C operations over Python loops).
    if candidate_rows.size > 0:
        clipped = shapely.clip_by_rect(shapely_values[candidate_rows], *rect)
        result[candidate_rows] = np.asarray(clipped, dtype=object)

    fast_rows_arr = candidate_rows
    fallback_index = np.asarray([], dtype=np.int32)

    # Build device-resident OwnedGeometryArray directly from the clipped
    # shapely geometries so callers can stay on the owned-array path without
    # a redundant shapely->OwnedGeometryArray round-trip (ADR-0005).
    kept = [
        g for g in result
        if g is not None and not (hasattr(g, "is_empty") and g.is_empty)
           and (not hasattr(g, "geom_type") or g.geom_type != "GeometryCollection")
    ]
    # Also extract supported parts from GeometryCollections
    for g in result:
        if g is not None and hasattr(g, "geom_type") and g.geom_type == "GeometryCollection":
            for part in g.geoms:
                if part.geom_type in ("Point", "LineString", "Polygon",
                                      "MultiPoint", "MultiLineString", "MultiPolygon"):
                    kept.append(part)
    if not kept:
        owned_result = from_shapely_geometries([shapely.Point(0, 0)])
    else:
        owned_result = from_shapely_geometries(kept)

    return RectClipResult(
        geometries=result,
        row_count=int(shapely_values.size),
        candidate_rows=candidate_rows,
        fast_rows=fast_rows_arr,
        fallback_rows=fallback_index,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
        owned_result=owned_result,
    )


def evaluate_geopandas_clip_by_rect(
    values: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[np.ndarray | None, ExecutionMode]:
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
        result = clip_by_rect_owned(
            geometries,
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
    shapely_values, _ = _normalize_values(values)
    started = perf_counter()
    result = clip_by_rect_owned(shapely_values, xmin, ymin, xmax, ymax)
    owned_elapsed = perf_counter() - started

    started = perf_counter()
    shapely.clip_by_rect(shapely_values, xmin, ymin, xmax, ymax)
    shapely_elapsed = perf_counter() - started

    return RectClipBenchmark(
        dataset=dataset,
        rows=int(shapely_values.size),
        candidate_rows=int(result.candidate_rows.size),
        fast_rows=int(result.fast_rows.size),
        fallback_rows=int(result.fallback_rows.size),
        owned_elapsed_seconds=owned_elapsed,
        shapely_elapsed_seconds=shapely_elapsed,
    )
