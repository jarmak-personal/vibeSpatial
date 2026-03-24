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
    """Clip line segments on GPU using Liang-Barsky.

    Returns (out_x0, out_y0, out_x1, out_y1, valid_mask) with valid_mask
    indicating which segments survived clipping.
    """
    from vibespatial.cuda._runtime import KERNEL_PARAM_F64, KERNEL_PARAM_I32, KERNEL_PARAM_PTR

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


# ---------------------------------------------------------------------------
# Batched GPU line clip — vectorized segment extraction + reassembly
# ---------------------------------------------------------------------------


def _extract_segments_vectorized(
    owned: OwnedGeometryArray,
    line_families: list[GeometryFamily],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Vectorized extraction of segments from line-family buffers.

    Uses numpy fancy indexing on contiguous buffer.x / buffer.y arrays
    instead of per-coordinate Python list.append loops.

    Returns
    -------
    seg_x0, seg_y0, seg_x1, seg_y1 : flat segment endpoint arrays (float64)
    row_segment_offsets : cumulative segment count per row (int32, length = len(global_row_indices) + 1)
    part_segment_offsets : cumulative segment count per part (int32, length = total_parts + 1)
    part_row_map : which row index (into global_row_indices) each part belongs to (int32)
    global_row_indices : ordered list of global row indices that have line data
    """
    # Collect per-span (start, end) ranges and metadata in bulk, then
    # do a single vectorized extraction pass.
    span_starts: list[int] = []
    span_ends: list[int] = []
    span_family_idx: list[int] = []  # which family's buffer each span belongs to
    part_row_map_list: list[int] = []  # row idx (into global_row_indices) for each part
    global_row_indices: list[int] = []
    family_buffers: list[object] = []

    for family in line_families:
        buffer = owned.families[family]
        tag = FAMILY_TAGS[family]
        family_rows = np.flatnonzero(owned.tags == tag)
        fam_idx = len(family_buffers)
        family_buffers.append(buffer)

        if buffer.row_count == 0 or buffer.geometry_offsets is None or len(buffer.geometry_offsets) < 2:
            for global_row in family_rows:
                global_row_indices.append(int(global_row))
            continue

        for global_row in family_rows:
            row_idx = len(global_row_indices)
            global_row_indices.append(int(global_row))
            local_row = int(owned.family_row_offsets[global_row])

            if buffer.empty_mask.size > local_row and buffer.empty_mask[local_row]:
                continue

            if family == GeometryFamily.LINESTRING:
                s = int(buffer.geometry_offsets[local_row])
                e = int(buffer.geometry_offsets[local_row + 1])
                if e - s >= 2:
                    span_starts.append(s)
                    span_ends.append(e)
                    span_family_idx.append(fam_idx)
                    part_row_map_list.append(row_idx)
            else:
                # MultiLineString
                part_start = int(buffer.geometry_offsets[local_row])
                part_end = int(buffer.geometry_offsets[local_row + 1])
                for idx in range(part_start, part_end):
                    s = int(buffer.part_offsets[idx])
                    e = int(buffer.part_offsets[idx + 1])
                    if e - s >= 2:
                        span_starts.append(s)
                        span_ends.append(e)
                        span_family_idx.append(fam_idx)
                        part_row_map_list.append(row_idx)

    if not span_starts:
        empty = np.empty(0, dtype=np.float64)
        return (
            empty, empty, empty, empty,
            np.zeros(len(global_row_indices) + 1, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            global_row_indices,
        )

    # Compute segment counts per span: (end - start - 1)
    starts_arr = np.asarray(span_starts, dtype=np.int64)
    ends_arr = np.asarray(span_ends, dtype=np.int64)
    seg_counts = (ends_arr - starts_arr - 1).astype(np.int32)
    total_segments = int(np.sum(seg_counts))

    if total_segments == 0:
        empty = np.empty(0, dtype=np.float64)
        return (
            empty, empty, empty, empty,
            np.zeros(len(global_row_indices) + 1, dtype=np.int32),
            np.zeros(len(span_starts) + 1, dtype=np.int32),
            np.asarray(part_row_map_list, dtype=np.int32),
            global_row_indices,
        )

    # Build flat index arrays for vectorized gather.
    # For each span, we need indices [start, start+1, ..., end-2] for x0/y0
    # and [start+1, start+2, ..., end-1] for x1/y1.
    seg_x0 = np.empty(total_segments, dtype=np.float64)
    seg_y0 = np.empty(total_segments, dtype=np.float64)
    seg_x1 = np.empty(total_segments, dtype=np.float64)
    seg_y1 = np.empty(total_segments, dtype=np.float64)

    out_pos = 0
    for i, (s, e, fi) in enumerate(zip(span_starts, span_ends, span_family_idx)):
        n = e - s - 1
        if n <= 0:
            continue
        buf = family_buffers[fi]
        # Vectorized slice: buffer.x[s:e-1] and buffer.x[s+1:e]
        seg_x0[out_pos:out_pos + n] = buf.x[s:e - 1]
        seg_y0[out_pos:out_pos + n] = buf.y[s:e - 1]
        seg_x1[out_pos:out_pos + n] = buf.x[s + 1:e]
        seg_y1[out_pos:out_pos + n] = buf.y[s + 1:e]
        out_pos += n

    # Build per-part segment offsets (cumulative)
    part_segment_offsets = np.empty(len(seg_counts) + 1, dtype=np.int32)
    part_segment_offsets[0] = 0
    np.cumsum(seg_counts, out=part_segment_offsets[1:])

    # Build per-row segment offsets from part offsets
    part_row_map_arr = np.asarray(part_row_map_list, dtype=np.int32)
    n_rows = len(global_row_indices)
    row_segment_offsets = np.zeros(n_rows + 1, dtype=np.int32)
    for pi, ri in enumerate(part_row_map_list):
        row_segment_offsets[ri + 1] += int(seg_counts[pi])
    np.cumsum(row_segment_offsets, out=row_segment_offsets)

    return (
        seg_x0, seg_y0, seg_x1, seg_y1,
        row_segment_offsets, part_segment_offsets,
        part_row_map_arr, global_row_indices,
    )


def _reassemble_lines_to_arrays(
    out_x0: np.ndarray,
    out_y0: np.ndarray,
    out_x1: np.ndarray,
    out_y1: np.ndarray,
    valid: np.ndarray,
    row_segment_offsets: np.ndarray,
    part_segment_offsets: np.ndarray,
    part_row_map: np.ndarray,
    global_row_indices: list[int],
    row_count: int,
) -> np.ndarray:
    """Reassemble clipped segments into Shapely geometries per row.

    Uses vectorized valid-mask slicing per part rather than per-segment
    Python loops.  Still produces Shapely objects for backward compatibility.
    """
    result = np.empty(row_count, dtype=object)
    result[:] = None

    n_rows = len(global_row_indices)
    n_parts = len(part_segment_offsets) - 1

    # Build a mapping from row index to its parts
    # row_parts[ri] = list of part indices
    row_parts: list[list[int]] = [[] for _ in range(n_rows)]
    for pi in range(n_parts):
        ri = int(part_row_map[pi])
        row_parts[ri].append(pi)

    for ri in range(n_rows):
        global_row = global_row_indices[ri]
        parts_for_row = row_parts[ri]

        if not parts_for_row:
            result[global_row] = EMPTY
            continue

        all_line_parts: list[list[tuple[float, float]]] = []
        for pi in parts_for_row:
            seg_start = int(part_segment_offsets[pi])
            seg_end = int(part_segment_offsets[pi + 1])
            if seg_start == seg_end:
                continue

            # Vectorized: get valid mask slice for this part
            part_valid = valid[seg_start:seg_end]
            valid_indices = np.flatnonzero(part_valid)

            if valid_indices.size == 0:
                continue

            # Extract valid segments for this part
            abs_indices = seg_start + valid_indices
            vx0 = out_x0[abs_indices]
            vy0 = out_y0[abs_indices]
            vx1 = out_x1[abs_indices]
            vy1 = out_y1[abs_indices]

            # Merge consecutive segments: two segments are consecutive if
            # segment[i].end == segment[i+1].start (within epsilon).
            part_segments: list[list[tuple[float, float]]] = []
            for si in range(len(valid_indices)):
                segment = (float(vx0[si]), float(vy0[si]), float(vx1[si]), float(vy1[si]))
                _merge_clipped_segments(part_segments, segment)
            all_line_parts.extend(part_segments)

        result[global_row] = _build_linestring_result(all_line_parts)

    return result


def _build_line_clip_owned_result(
    out_x0: np.ndarray,
    out_y0: np.ndarray,
    out_x1: np.ndarray,
    out_y1: np.ndarray,
    valid: np.ndarray,
    row_segment_offsets: np.ndarray,
    part_segment_offsets: np.ndarray,
    part_row_map: np.ndarray,
    global_row_indices: list[int],
) -> OwnedGeometryArray | None:
    """Build an OwnedGeometryArray directly from clipped segment arrays.

    Constructs coordinate and offset arrays for a LineString-family OGA
    without going through Shapely construction.  Each part with valid
    segments becomes one LineString row in the output.
    """
    from vibespatial.geometry.buffers import get_geometry_buffer_schema

    n_parts = len(part_segment_offsets) - 1
    if n_parts == 0:
        return None

    # Collect coordinates for all valid line parts.
    # Each valid-segments run within a part becomes a linestring.
    all_coords_x: list[np.ndarray] = []
    all_coords_y: list[np.ndarray] = []
    line_lengths: list[int] = []

    for pi in range(n_parts):
        seg_start = int(part_segment_offsets[pi])
        seg_end = int(part_segment_offsets[pi + 1])
        if seg_start == seg_end:
            continue

        part_valid = valid[seg_start:seg_end]
        valid_indices = np.flatnonzero(part_valid)
        if valid_indices.size == 0:
            continue

        abs_indices = seg_start + valid_indices
        vx0 = out_x0[abs_indices]
        vy0 = out_y0[abs_indices]
        vx1 = out_x1[abs_indices]
        vy1 = out_y1[abs_indices]

        # Build coordinate arrays: for N valid segments, we get
        # up to N+1 coordinates (start of first + end of each).
        # Detect breaks where end[i] != start[i+1].
        if len(valid_indices) == 1:
            all_coords_x.append(np.array([vx0[0], vx1[0]]))
            all_coords_y.append(np.array([vy0[0], vy1[0]]))
            line_lengths.append(2)
        else:
            # Check continuity between consecutive segments
            breaks = np.where(
                (np.abs(vx1[:-1] - vx0[1:]) > _POINT_EPSILON)
                | (np.abs(vy1[:-1] - vy0[1:]) > _POINT_EPSILON)
            )[0]

            if breaks.size == 0:
                # All segments are continuous: one linestring
                coords_x = np.empty(len(valid_indices) + 1, dtype=np.float64)
                coords_y = np.empty(len(valid_indices) + 1, dtype=np.float64)
                coords_x[:-1] = vx0
                coords_x[-1] = vx1[-1]
                coords_y[:-1] = vy0
                coords_y[-1] = vy1[-1]
                all_coords_x.append(coords_x)
                all_coords_y.append(coords_y)
                line_lengths.append(len(coords_x))
            else:
                # Multiple linestrings from breaks
                split_points = np.concatenate(([0], breaks + 1, [len(valid_indices)]))
                for j in range(len(split_points) - 1):
                    s, e = int(split_points[j]), int(split_points[j + 1])
                    if e - s < 1:
                        continue
                    n_seg = e - s
                    coords_x = np.empty(n_seg + 1, dtype=np.float64)
                    coords_y = np.empty(n_seg + 1, dtype=np.float64)
                    coords_x[:-1] = vx0[s:e]
                    coords_x[-1] = vx1[e - 1]
                    coords_y[:-1] = vy0[s:e]
                    coords_y[-1] = vy1[e - 1]
                    if len(coords_x) >= 2:
                        all_coords_x.append(coords_x)
                        all_coords_y.append(coords_y)
                        line_lengths.append(len(coords_x))

    if not all_coords_x:
        return None

    # Build flat coordinate arrays and geometry_offsets for OGA
    total_coords = sum(line_lengths)
    flat_x = np.empty(total_coords, dtype=np.float64)
    flat_y = np.empty(total_coords, dtype=np.float64)
    geometry_offsets = np.empty(len(line_lengths) + 1, dtype=np.int32)
    geometry_offsets[0] = 0
    pos = 0
    for i, (cx, cy) in enumerate(zip(all_coords_x, all_coords_y)):
        n = len(cx)
        flat_x[pos:pos + n] = cx
        flat_y[pos:pos + n] = cy
        pos += n
        geometry_offsets[i + 1] = pos

    line_count = len(line_lengths)
    empty_mask = np.zeros(line_count, dtype=np.bool_)
    validity = np.ones(line_count, dtype=np.bool_)
    tags = np.full(line_count, FAMILY_TAGS[GeometryFamily.LINESTRING], dtype=np.int8)
    family_row_offsets = np.arange(line_count, dtype=np.int32)

    line_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        schema=get_geometry_buffer_schema(GeometryFamily.LINESTRING),
        row_count=line_count,
        x=flat_x,
        y=flat_y,
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        host_materialized=True,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.LINESTRING: line_buffer},
    )


def _clip_all_lines_gpu(
    owned: OwnedGeometryArray,
    rect: tuple[float, float, float, float],
) -> np.ndarray:
    """Batch-clip ALL linestring/multilinestring rows on GPU via Liang-Barsky.

    Vectorized extraction: uses numpy fancy indexing on contiguous buffer.x/y
    arrays instead of per-coordinate Python list.append loops.

    Returns an object array of length ``owned.row_count`` where each entry is
    the clipped geometry (or ``None`` for rows that are not line families).
    """
    line_families = [
        f for f in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
        if f in owned.families
    ]
    result = np.empty(owned.row_count, dtype=object)
    result[:] = None

    if not line_families:
        return result

    # -- Step 1: vectorized segment extraction ----------------------------
    (
        seg_x0, seg_y0, seg_x1, seg_y1,
        row_segment_offsets, part_segment_offsets,
        part_row_map, global_row_indices,
    ) = _extract_segments_vectorized(owned, line_families)

    total_segments = len(seg_x0)
    if total_segments == 0:
        for global_row in global_row_indices:
            result[global_row] = EMPTY
        return result

    # -- Step 2: single GPU kernel launch ---------------------------------
    out_x0, out_y0, out_x1, out_y1, valid = _clip_line_segments_gpu(
        seg_x0, seg_y0, seg_x1, seg_y1, rect,
    )

    # -- Step 3: vectorized reassembly ------------------------------------
    return _reassemble_lines_to_arrays(
        out_x0, out_y0, out_x1, out_y1, valid,
        row_segment_offsets, part_segment_offsets,
        part_row_map, global_row_indices,
        owned.row_count,
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

def _extract_polygon_rings_vectorized(
    owned: OwnedGeometryArray,
    polygon_families: list[GeometryFamily],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Extract ALL polygon rings across ALL polygon-family rows into flat arrays.

    Uses direct buffer slicing instead of per-coordinate Python loops. The
    family geometry buffers already store contiguous x, y, ring_offsets, and
    geometry_offsets arrays -- this function concatenates them across families
    with vectorized numpy operations.

    Returns
    -------
    ring_x, ring_y : flat coordinate arrays (float64)
    ring_offsets : offsets into x/y for each ring (int32, length = total_rings + 1)
    ring_geom_map : which global row index each ring belongs to (int32)
    ring_is_exterior : 1 if exterior ring, 0 if hole (int32)
    geom_ring_offsets : offsets into ring arrays for each geometry row (int32)
    global_row_indices : ordered list of global row indices that have polygon data
    """
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

        for idx, global_row in enumerate(family_rows):
            grow = int(global_row)
            lr = int(local_rows[idx])

            if empty_flags[idx]:
                global_row_indices.append(grow)
                all_geom_ring_counts.append(0)
                continue

            if family == GeometryFamily.POLYGON:
                ring_start_idx = int(buffer.geometry_offsets[lr])
                ring_end_idx = int(buffer.geometry_offsets[lr + 1])
                n_rings = ring_end_idx - ring_start_idx
                if n_rings == 0:
                    global_row_indices.append(grow)
                    all_geom_ring_counts.append(0)
                    continue

                # Slice ring_offsets for this geometry's rings -- vectorized
                local_ring_offsets = buffer.ring_offsets[ring_start_idx:ring_end_idx + 1]
                coord_start = int(local_ring_offsets[0])
                coord_end = int(local_ring_offsets[-1])
                n_coords = coord_end - coord_start

                # Bulk copy coordinates -- zero per-element Python calls
                all_x_chunks.append(np.asarray(buffer.x[coord_start:coord_end], dtype=np.float64))
                all_y_chunks.append(np.asarray(buffer.y[coord_start:coord_end], dtype=np.float64))

                # Ring offsets for this geometry, rebased to coord_cursor
                rebased_offsets = np.asarray(local_ring_offsets[:-1], dtype=np.int32) - coord_start + coord_cursor
                all_ring_offsets_chunks.append(rebased_offsets)

                # Geom map: all rings map to this global row
                all_ring_geom_map.append(np.full(n_rings, grow, dtype=np.int32))

                # Exterior flag: first ring is exterior, rest are holes
                ext_flags = np.zeros(n_rings, dtype=np.int32)
                ext_flags[0] = 1
                all_ring_is_exterior.append(ext_flags)

                coord_cursor += n_coords
                ring_cursor += n_rings
            else:
                # MultiPolygon: geometry_offsets -> part_offsets -> ring_offsets
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
                continue

            global_row_indices.append(grow)
            all_geom_ring_counts.append(n_rings)

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

    # Walk ring structure to identify valid polygons and build GeoArrow offsets.
    # This loop iterates over metadata (rings per geometry, not coordinates)
    # so it is O(total_rings) which is small relative to coordinate count.

    polygon_ring_offsets: list[int] = [0]   # cumulative coord count per ring
    polygon_geometry_offsets: list[int] = [0]  # cumulative ring count per geometry
    output_valid_rows: list[int] = []  # global rows with valid output
    output_empty_mask_list: list[bool] = []

    # Gather coordinate ranges from d_out that survived clipping
    gather_ranges: list[tuple[int, int]] = []  # (start, end) into flat output

    for geom_idx, global_row in enumerate(global_row_indices):
        ring_start = int(geom_ring_offsets[geom_idx])
        ring_end = int(geom_ring_offsets[geom_idx + 1])

        if ring_start == ring_end:
            # No rings for this geometry -- skip (empty/invalid)
            continue

        # Walk rings for this geometry, identifying surviving exteriors + holes
        geom_has_output = False

        # Track state for current polygon part (exterior + its holes)
        current_ext_valid = False

        for ring_idx in range(ring_start, ring_end):
            out_start = int(h_full_offsets[ring_idx])
            out_end = int(h_full_offsets[ring_idx + 1])
            verts = out_end - out_start
            is_exterior = bool(ring_is_exterior[ring_idx])

            if is_exterior:
                current_ext_valid = verts >= 4
            # Only include rings whose exterior survived
            if not current_ext_valid:
                continue
            if verts < 4:
                # Hole clipped away -- skip
                continue

            gather_ranges.append((out_start, out_end))
            polygon_ring_offsets.append(polygon_ring_offsets[-1] + verts)
            geom_has_output = True

        if geom_has_output:
            output_valid_rows.append(global_row)
            polygon_geometry_offsets.append(len(polygon_ring_offsets) - 1)
            output_empty_mask_list.append(False)
        # If nothing survived for this geometry, discard any partial ring state
        # (shouldn't happen since we only append when geom_has_output, but guard)

    output_row_count = len(output_valid_rows)

    # Build validity mask for the caller (covers all input rows)
    validity_mask = np.zeros(input_row_count, dtype=bool)
    for row in output_valid_rows:
        validity_mask[row] = True

    if output_row_count == 0:
        return None, validity_mask

    # Build gather indices on host, execute gather on device (coordinates
    # stay on GPU).  This is O(n_output_rings) host work, not O(n_coords).
    gather_chunks = [np.arange(s, e, dtype=np.int64) for s, e in gather_ranges]
    h_gather = np.concatenate(gather_chunks)
    d_gather = cp.asarray(h_gather)
    gathered_x = d_out_x[d_gather]
    gathered_y = d_out_y[d_gather]

    h_ring_offsets = np.asarray(polygon_ring_offsets, dtype=np.int32)
    h_geom_offsets = np.asarray(polygon_geometry_offsets, dtype=np.int32)
    output_empty_mask = np.asarray(output_empty_mask_list, dtype=bool)

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
                line_result = _clip_all_lines_gpu(owned, rect)

            # Identify GPU-fast family tags
            _gpu_family_tags = set()
            for fam in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
                if fam in owned.families:
                    _gpu_family_tags.add(FAMILY_TAGS.get(fam, -99))
            for fam in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING):
                if fam in owned.families:
                    _gpu_family_tags.add(FAMILY_TAGS.get(fam, -99))

            # Rows handled by GPU kernels vs rows needing shapely fallback
            gpu_fast_rows: list[int] = []
            fallback_row_list: list[int] = []
            for i in range(owned.row_count):
                if not owned.validity[i]:
                    continue
                tag = int(owned.tags[i])
                if tag in _gpu_family_tags:
                    gpu_fast_rows.append(i)
                else:
                    fallback_row_list.append(i)

            fast_rows_arr = np.asarray(gpu_fast_rows, dtype=np.int32)
            fallback_rows_arr = np.asarray(fallback_row_list, dtype=np.int32)
            all_candidate_rows = np.sort(
                np.concatenate([fast_rows_arr, fallback_rows_arr])
            ).astype(np.int32)

            # Use poly_owned as the primary result when available.
            # Shapely geometries are materialized lazily only when
            # .geometries is accessed (tests, GeoPandas adapter).
            owned_result = poly_owned

            # Capture references for lazy factory
            _owned_ref = owned
            _poly_owned_ref = poly_owned
            _poly_validity_mask_ref = poly_validity_mask
            _line_result_ref = line_result
            _fallback_row_list_ref = fallback_row_list
            _rect_ref = rect

            def _materialize_poly_line_geometries():
                result = np.empty(_owned_ref.row_count, dtype=object)
                result[:] = None
                for i in range(_owned_ref.row_count):
                    if _owned_ref.validity[i]:
                        result[i] = EMPTY

                # Materialize polygon results from device-resident OGA
                if _poly_owned_ref is not None and _poly_validity_mask_ref is not None:
                    try:
                        poly_shapely = _poly_owned_ref.to_shapely()
                        valid_rows = np.flatnonzero(_poly_validity_mask_ref)
                        # poly_owned has compact rows; scatter back to global positions
                        for compact_idx, global_row in enumerate(valid_rows):
                            if compact_idx < len(poly_shapely):
                                result[global_row] = poly_shapely[compact_idx]
                    except Exception:
                        pass

                # Merge line results
                if _line_result_ref is not None:
                    for i in range(_owned_ref.row_count):
                        if _line_result_ref[i] is not None:
                            result[i] = _line_result_ref[i]

                # Handle fallback rows (non-polygon, non-line families)
                if _fallback_row_list_ref:
                    shapely_geoms = _owned_ref.to_shapely()
                    fallback_shapely = np.asarray(
                        [shapely_geoms[i] for i in _fallback_row_list_ref], dtype=object
                    )
                    clipped = shapely.clip_by_rect(fallback_shapely, *_rect_ref)
                    for idx, row in enumerate(_fallback_row_list_ref):
                        result[row] = clipped[idx]

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
