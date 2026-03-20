from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from time import perf_counter

import numpy as np

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import compact_indices

request_warmup(["select_i32", "select_i64"])
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    DeviceArray,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray  # noqa: E402
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.runtime.adaptive import plan_dispatch_selection  # noqa: E402
from vibespatial.runtime.precision import (  # noqa: E402
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    select_precision_plan,
)
from vibespatial.runtime.robustness import RobustnessPlan, select_robustness_plan  # noqa: E402

_FLOAT_EPSILON = np.finfo(np.float64).eps
_ORIENTATION_ERRBOUND = (3.0 + 16.0 * _FLOAT_EPSILON) * _FLOAT_EPSILON
_SEGMENT_GPU_THRESHOLD = 4_096

_SEGMENT_INTERSECTION_KERNEL_SOURCE = f"""
extern "C" __device__ double abs_f64(double value) {{
  return value < 0.0 ? -value : value;
}}

extern "C" __global__ void classify_segment_pairs(
    const int* __restrict__ left_lookup,
    const int* __restrict__ right_lookup,
    const double* __restrict__ left_x0,
    const double* __restrict__ left_y0,
    const double* __restrict__ left_x1,
    const double* __restrict__ left_y1,
    const double* __restrict__ right_x0,
    const double* __restrict__ right_y0,
    const double* __restrict__ right_x1,
    const double* __restrict__ right_y1,
    signed char* __restrict__ out_kind,
    double* __restrict__ out_point_x,
    double* __restrict__ out_point_y,
    double* __restrict__ out_overlap_x0,
    double* __restrict__ out_overlap_y0,
    double* __restrict__ out_overlap_x1,
    double* __restrict__ out_overlap_y1,
    unsigned char* __restrict__ out_ambiguous,
    int row_count
) {{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {{
    return;
  }}

  const int left_index = left_lookup[row];
  const int right_index = right_lookup[row];
  const double nan_value = 0.0 / 0.0;
  const double ax = left_x0[left_index];
  const double ay = left_y0[left_index];
  const double bx = left_x1[left_index];
  const double by = left_y1[left_index];
  const double cx = right_x0[right_index];
  const double cy = right_y0[right_index];
  const double dx = right_x1[right_index];
  const double dy = right_y1[right_index];

  out_kind[row] = 0;
  out_point_x[row] = nan_value;
  out_point_y[row] = nan_value;
  out_overlap_x0[row] = nan_value;
  out_overlap_y0[row] = nan_value;
  out_overlap_x1[row] = nan_value;
  out_overlap_y1[row] = nan_value;

  const double abx = bx - ax;
  const double aby = by - ay;
  const double acx = cx - ax;
  const double acy = cy - ay;
  const double adx = dx - ax;
  const double ady = dy - ay;
  const double cdx = dx - cx;
  const double cdy = dy - cy;
  const double cax = ax - cx;
  const double cay = ay - cy;
  const double cbx = bx - cx;
  const double cby = by - cy;

  const double o1_term1 = abx * acy;
  const double o1_term2 = aby * acx;
  const double o1 = o1_term1 - o1_term2;
  const double o2_term1 = abx * ady;
  const double o2_term2 = aby * adx;
  const double o2 = o2_term1 - o2_term2;
  const double o3_term1 = cdx * cay;
  const double o3_term2 = cdy * cax;
  const double o3 = o3_term1 - o3_term2;
  const double o4_term1 = cdx * cby;
  const double o4_term2 = cdy * cbx;
  const double o4 = o4_term1 - o4_term2;

  const double err1 = {_ORIENTATION_ERRBOUND} * (abs_f64(o1_term1) + abs_f64(o1_term2));
  const double err2 = {_ORIENTATION_ERRBOUND} * (abs_f64(o2_term1) + abs_f64(o2_term2));
  const double err3 = {_ORIENTATION_ERRBOUND} * (abs_f64(o3_term1) + abs_f64(o3_term2));
  const double err4 = {_ORIENTATION_ERRBOUND} * (abs_f64(o4_term1) + abs_f64(o4_term2));

  const int sign1 = (o1 > 0.0) - (o1 < 0.0);
  const int sign2 = (o2 > 0.0) - (o2 < 0.0);
  const int sign3 = (o3 > 0.0) - (o3 < 0.0);
  const int sign4 = (o4 > 0.0) - (o4 < 0.0);

  const int ambiguous =
      (abs_f64(o1) <= err1) ||
      (abs_f64(o2) <= err2) ||
      (abs_f64(o3) <= err3) ||
      (abs_f64(o4) <= err4) ||
      ((ax == bx) && (ay == by)) ||
      ((cx == dx) && (cy == dy)) ||
      (sign1 == 0) ||
      (sign2 == 0) ||
      (sign3 == 0) ||
      (sign4 == 0);

  out_ambiguous[row] = ambiguous ? 1 : 0;
  if (ambiguous) {{
    return;
  }}

  if ((sign1 * sign2 < 0) && (sign3 * sign4 < 0)) {{
    const double denominator = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx);
    if (denominator == 0.0) {{
      out_ambiguous[row] = 1;
      return;
    }}
    const double left_det = ax * by - ay * bx;
    const double right_det = cx * dy - cy * dx;
    out_kind[row] = 1;
    out_point_x[row] = (left_det * (cx - dx) - (ax - bx) * right_det) / denominator;
    out_point_y[row] = (left_det * (cy - dy) - (ay - by) * right_det) / denominator;
  }}
}}
"""

_SEGMENT_INTERSECTION_WARP_SKIP_KERNEL_SOURCE = f"""
extern "C" __device__ double abs_f64(double value) {{
  return value < 0.0 ? -value : value;
}}

extern "C" __global__ void __launch_bounds__(256, 4)
classify_segment_pairs_warp_skip(
    const int* __restrict__ left_lookup,
    const int* __restrict__ right_lookup,
    const double* __restrict__ left_x0,
    const double* __restrict__ left_y0,
    const double* __restrict__ left_x1,
    const double* __restrict__ left_y1,
    const double* __restrict__ right_x0,
    const double* __restrict__ right_y0,
    const double* __restrict__ right_x1,
    const double* __restrict__ right_y1,
    signed char* __restrict__ out_kind,
    double* __restrict__ out_point_x,
    double* __restrict__ out_point_y,
    double* __restrict__ out_overlap_x0,
    double* __restrict__ out_overlap_y0,
    double* __restrict__ out_overlap_x1,
    double* __restrict__ out_overlap_y1,
    unsigned char* __restrict__ out_ambiguous,
    int row_count
) {{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = row < row_count;

  /* --- Phase 1: MBR overlap check with warp-cooperative skip --- */
  int has_overlap = 0;
  int left_index = 0;
  int right_index = 0;
  double ax, ay, bx, by, cx, cy, dx, dy;

  if (valid) {{
    left_index = left_lookup[row];
    right_index = right_lookup[row];

    /* Load endpoints */
    ax = left_x0[left_index];
    ay = left_y0[left_index];
    bx = left_x1[left_index];
    by = left_y1[left_index];
    cx = right_x0[right_index];
    cy = right_y0[right_index];
    dx = right_x1[right_index];
    dy = right_y1[right_index];

    /* Compute MBRs */
    const double left_minx = ax < bx ? ax : bx;
    const double left_maxx = ax > bx ? ax : bx;
    const double left_miny = ay < by ? ay : by;
    const double left_maxy = ay > by ? ay : by;
    const double right_minx = cx < dx ? cx : dx;
    const double right_maxx = cx > dx ? cx : dx;
    const double right_miny = cy < dy ? cy : dy;
    const double right_maxy = cy > dy ? cy : dy;

    has_overlap = (left_minx <= right_maxx) && (left_maxx >= right_minx) &&
                  (left_miny <= right_maxy) && (left_maxy >= right_miny);
  }}

  /* If no thread in this warp has an overlapping pair, skip all work */
  if (__ballot_sync(0xFFFFFFFF, has_overlap) == 0) {{
    if (valid) {{
      out_kind[row] = 0;          /* DISJOINT */
      const double nan_value = 0.0 / 0.0;
      out_point_x[row] = nan_value;
      out_point_y[row] = nan_value;
      out_overlap_x0[row] = nan_value;
      out_overlap_y0[row] = nan_value;
      out_overlap_x1[row] = nan_value;
      out_overlap_y1[row] = nan_value;
      out_ambiguous[row] = 0;
    }}
    return;
  }}

  if (!valid) return;

  /* --- Phase 2: Full orientation classification (same as original) --- */
  const double nan_value = 0.0 / 0.0;
  out_kind[row] = 0;
  out_point_x[row] = nan_value;
  out_point_y[row] = nan_value;
  out_overlap_x0[row] = nan_value;
  out_overlap_y0[row] = nan_value;
  out_overlap_x1[row] = nan_value;
  out_overlap_y1[row] = nan_value;

  /* Non-overlapping MBR -> disjoint (thread-level early exit) */
  if (!has_overlap) {{
    out_ambiguous[row] = 0;
    return;
  }}

  const double abx = bx - ax;
  const double aby = by - ay;
  const double acx = cx - ax;
  const double acy = cy - ay;
  const double adx = dx - ax;
  const double ady = dy - ay;
  const double cdx = dx - cx;
  const double cdy = dy - cy;
  const double cax = ax - cx;
  const double cay = ay - cy;
  const double cbx = bx - cx;
  const double cby = by - cy;

  const double o1_term1 = abx * acy;
  const double o1_term2 = aby * acx;
  const double o1 = o1_term1 - o1_term2;
  const double o2_term1 = abx * ady;
  const double o2_term2 = aby * adx;
  const double o2 = o2_term1 - o2_term2;
  const double o3_term1 = cdx * cay;
  const double o3_term2 = cdy * cax;
  const double o3 = o3_term1 - o3_term2;
  const double o4_term1 = cdx * cby;
  const double o4_term2 = cdy * cbx;
  const double o4 = o4_term1 - o4_term2;

  const double err1 = {_ORIENTATION_ERRBOUND} * (abs_f64(o1_term1) + abs_f64(o1_term2));
  const double err2 = {_ORIENTATION_ERRBOUND} * (abs_f64(o2_term1) + abs_f64(o2_term2));
  const double err3 = {_ORIENTATION_ERRBOUND} * (abs_f64(o3_term1) + abs_f64(o3_term2));
  const double err4 = {_ORIENTATION_ERRBOUND} * (abs_f64(o4_term1) + abs_f64(o4_term2));

  const int sign1 = (o1 > 0.0) - (o1 < 0.0);
  const int sign2 = (o2 > 0.0) - (o2 < 0.0);
  const int sign3 = (o3 > 0.0) - (o3 < 0.0);
  const int sign4 = (o4 > 0.0) - (o4 < 0.0);

  const int ambiguous =
      (abs_f64(o1) <= err1) ||
      (abs_f64(o2) <= err2) ||
      (abs_f64(o3) <= err3) ||
      (abs_f64(o4) <= err4) ||
      ((ax == bx) && (ay == by)) ||
      ((cx == dx) && (cy == dy)) ||
      (sign1 == 0) ||
      (sign2 == 0) ||
      (sign3 == 0) ||
      (sign4 == 0);

  out_ambiguous[row] = ambiguous ? 1 : 0;
  if (ambiguous) {{
    return;
  }}

  if ((sign1 * sign2 < 0) && (sign3 * sign4 < 0)) {{
    const double denominator = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx);
    if (denominator == 0.0) {{
      out_ambiguous[row] = 1;
      return;
    }}
    const double left_det = ax * by - ay * bx;
    const double right_det = cx * dy - cy * dx;
    out_kind[row] = 1;
    out_point_x[row] = (left_det * (cx - dx) - (ax - bx) * right_det) / denominator;
    out_point_y[row] = (left_det * (cy - dy) - (ay - by) * right_det) / denominator;
  }}
}}
"""


class SegmentIntersectionKind(IntEnum):
    DISJOINT = 0
    PROPER = 1
    TOUCH = 2
    OVERLAP = 3


@dataclass(frozen=True)
class SegmentTable:
    row_indices: np.ndarray
    part_indices: np.ndarray
    ring_indices: np.ndarray
    segment_indices: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    bounds: np.ndarray

    @property
    def count(self) -> int:
        return int(self.row_indices.size)


@dataclass(frozen=True)
class SegmentIntersectionResult:
    left_rows: np.ndarray
    left_segments: np.ndarray
    left_lookup: np.ndarray
    right_rows: np.ndarray
    right_segments: np.ndarray
    right_lookup: np.ndarray
    kinds: np.ndarray
    point_x: np.ndarray
    point_y: np.ndarray
    overlap_x0: np.ndarray
    overlap_y0: np.ndarray
    overlap_x1: np.ndarray
    overlap_y1: np.ndarray
    candidate_pairs: int
    ambiguous_rows: np.ndarray
    runtime_selection: RuntimeSelection
    precision_plan: PrecisionPlan
    robustness_plan: RobustnessPlan
    device_state: SegmentIntersectionDeviceState | None = None

    @property
    def count(self) -> int:
        return int(self.left_rows.size)

    def kind_names(self) -> list[str]:
        return [SegmentIntersectionKind(int(value)).name.lower() for value in self.kinds]


@dataclass(frozen=True)
class SegmentIntersectionBenchmark:
    rows_left: int
    rows_right: int
    candidate_pairs: int
    disjoint_pairs: int
    proper_pairs: int
    touch_pairs: int
    overlap_pairs: int
    ambiguous_pairs: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SegmentIntersectionDeviceState:
    left_rows: DeviceArray
    left_segments: DeviceArray
    left_lookup: DeviceArray
    right_rows: DeviceArray
    right_segments: DeviceArray
    right_lookup: DeviceArray
    kinds: DeviceArray
    point_x: DeviceArray
    point_y: DeviceArray
    overlap_x0: DeviceArray
    overlap_y0: DeviceArray
    overlap_x1: DeviceArray
    overlap_y1: DeviceArray
    ambiguous_rows: DeviceArray


@dataclass(frozen=True)
class SegmentIntersectionCandidates:
    left_rows: np.ndarray
    left_segments: np.ndarray
    left_lookup: np.ndarray
    right_rows: np.ndarray
    right_segments: np.ndarray
    right_lookup: np.ndarray
    pairs_examined: int
    tile_size: int

    @property
    def count(self) -> int:
        return int(self.left_rows.size)


_SEGMENT_INTERSECTION_KERNEL_NAMES = ("classify_segment_pairs",)
_SEGMENT_INTERSECTION_WARP_SKIP_KERNEL_NAMES = ("classify_segment_pairs_warp_skip",)

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("segment-intersections", _SEGMENT_INTERSECTION_KERNEL_SOURCE, _SEGMENT_INTERSECTION_KERNEL_NAMES),
    ("segment-intersections-warp-skip", _SEGMENT_INTERSECTION_WARP_SKIP_KERNEL_SOURCE, _SEGMENT_INTERSECTION_WARP_SKIP_KERNEL_NAMES),
])


def _segment_intersection_kernels():
    return compile_kernel_group("segment-intersections", _SEGMENT_INTERSECTION_KERNEL_SOURCE, _SEGMENT_INTERSECTION_KERNEL_NAMES)


def _segment_intersection_warp_skip_kernels():
    return compile_kernel_group("segment-intersections-warp-skip", _SEGMENT_INTERSECTION_WARP_SKIP_KERNEL_SOURCE, _SEGMENT_INTERSECTION_WARP_SKIP_KERNEL_NAMES)


def _select_segment_runtime(
    dispatch_mode: ExecutionMode | str,
    *,
    candidate_count: int,
) -> RuntimeSelection:
    return plan_dispatch_selection(
        kernel_name="segment_classify",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=candidate_count,
        requested_mode=dispatch_mode,
    )


def _valid_global_rows(geometry_array: OwnedGeometryArray, family_name: str) -> np.ndarray:
    tag = FAMILY_TAGS[family_name]
    return np.flatnonzero(geometry_array.validity & (geometry_array.tags == tag)).astype(np.int32, copy=False)


def _append_segments_for_span(
    *,
    row_index: int,
    part_index: int,
    ring_index: int,
    segment_counter: int,
    x: np.ndarray,
    y: np.ndarray,
    start: int,
    end: int,
    row_indices: list[int],
    part_indices: list[int],
    ring_indices: list[int],
    segment_indices: list[int],
    x0: list[float],
    y0: list[float],
    x1: list[float],
    y1: list[float],
    bounds: list[tuple[float, float, float, float]],
) -> int:
    if end - start < 2:
        return segment_counter

    xs0 = x[start : end - 1]
    ys0 = y[start : end - 1]
    xs1 = x[start + 1 : end]
    ys1 = y[start + 1 : end]
    count = int(xs0.size)
    if count == 0:
        return segment_counter

    row_indices.extend([row_index] * count)
    part_indices.extend([part_index] * count)
    ring_indices.extend([ring_index] * count)
    segment_indices.extend(range(segment_counter, segment_counter + count))
    x0.extend(xs0.tolist())
    y0.extend(ys0.tolist())
    x1.extend(xs1.tolist())
    y1.extend(ys1.tolist())
    bounds.extend(
        zip(
            np.minimum(xs0, xs1).tolist(),
            np.minimum(ys0, ys1).tolist(),
            np.maximum(xs0, xs1).tolist(),
            np.maximum(ys0, ys1).tolist(),
            strict=True,
        )
    )
    return segment_counter + count


def extract_segments(geometry_array: OwnedGeometryArray) -> SegmentTable:
    row_indices: list[int] = []
    part_indices: list[int] = []
    ring_indices: list[int] = []
    segment_indices: list[int] = []
    x0: list[float] = []
    y0: list[float] = []
    x1: list[float] = []
    y1: list[float] = []
    bounds: list[tuple[float, float, float, float]] = []

    for family_name, buffer in geometry_array.families.items():
        if family_name not in {"linestring", "polygon", "multilinestring", "multipolygon"}:
            continue

        global_rows = _valid_global_rows(geometry_array, family_name)
        for family_row, row_index in enumerate(global_rows.tolist()):
            if bool(buffer.empty_mask[family_row]):
                continue

            segment_counter = 0
            if family_name == "linestring":
                start = int(buffer.geometry_offsets[family_row])
                end = int(buffer.geometry_offsets[family_row + 1])
                segment_counter = _append_segments_for_span(
                    row_index=row_index,
                    part_index=0,
                    ring_index=0,
                    segment_counter=segment_counter,
                    x=buffer.x,
                    y=buffer.y,
                    start=start,
                    end=end,
                    row_indices=row_indices,
                    part_indices=part_indices,
                    ring_indices=ring_indices,
                    segment_indices=segment_indices,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    bounds=bounds,
                )
                del segment_counter
                continue

            if family_name == "polygon":
                ring_start = int(buffer.geometry_offsets[family_row])
                ring_end = int(buffer.geometry_offsets[family_row + 1])
                for ring_local, ring_index in enumerate(range(ring_start, ring_end)):
                    coord_start = int(buffer.ring_offsets[ring_index])
                    coord_end = int(buffer.ring_offsets[ring_index + 1])
                    segment_counter = _append_segments_for_span(
                        row_index=row_index,
                        part_index=0,
                        ring_index=ring_local,
                        segment_counter=segment_counter,
                        x=buffer.x,
                        y=buffer.y,
                        start=coord_start,
                        end=coord_end,
                        row_indices=row_indices,
                        part_indices=part_indices,
                        ring_indices=ring_indices,
                        segment_indices=segment_indices,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        bounds=bounds,
                    )
                continue

            if family_name == "multilinestring":
                part_start = int(buffer.geometry_offsets[family_row])
                part_end = int(buffer.geometry_offsets[family_row + 1])
                for part_local, part_index in enumerate(range(part_start, part_end)):
                    coord_start = int(buffer.part_offsets[part_index])
                    coord_end = int(buffer.part_offsets[part_index + 1])
                    segment_counter = _append_segments_for_span(
                        row_index=row_index,
                        part_index=part_local,
                        ring_index=-1,
                        segment_counter=segment_counter,
                        x=buffer.x,
                        y=buffer.y,
                        start=coord_start,
                        end=coord_end,
                        row_indices=row_indices,
                        part_indices=part_indices,
                        ring_indices=ring_indices,
                        segment_indices=segment_indices,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        bounds=bounds,
                    )
                continue

            polygon_start = int(buffer.geometry_offsets[family_row])
            polygon_end = int(buffer.geometry_offsets[family_row + 1])
            for polygon_local, polygon_index in enumerate(range(polygon_start, polygon_end)):
                ring_start = int(buffer.part_offsets[polygon_index])
                ring_end = int(buffer.part_offsets[polygon_index + 1])
                for ring_local, ring_index in enumerate(range(ring_start, ring_end)):
                    coord_start = int(buffer.ring_offsets[ring_index])
                    coord_end = int(buffer.ring_offsets[ring_index + 1])
                    segment_counter = _append_segments_for_span(
                        row_index=row_index,
                        part_index=polygon_local,
                        ring_index=ring_local,
                        segment_counter=segment_counter,
                        x=buffer.x,
                        y=buffer.y,
                        start=coord_start,
                        end=coord_end,
                        row_indices=row_indices,
                        part_indices=part_indices,
                        ring_indices=ring_indices,
                        segment_indices=segment_indices,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        bounds=bounds,
                    )

    if not row_indices:
        empty_i32 = np.asarray([], dtype=np.int32)
        empty_f64 = np.asarray([], dtype=np.float64)
        return SegmentTable(
            row_indices=empty_i32,
            part_indices=empty_i32,
            ring_indices=empty_i32,
            segment_indices=empty_i32,
            x0=empty_f64,
            y0=empty_f64,
            x1=empty_f64,
            y1=empty_f64,
            bounds=np.empty((0, 4), dtype=np.float64),
        )

    return SegmentTable(
        row_indices=np.asarray(row_indices, dtype=np.int32),
        part_indices=np.asarray(part_indices, dtype=np.int32),
        ring_indices=np.asarray(ring_indices, dtype=np.int32),
        segment_indices=np.asarray(segment_indices, dtype=np.int32),
        x0=np.asarray(x0, dtype=np.float64),
        y0=np.asarray(y0, dtype=np.float64),
        x1=np.asarray(x1, dtype=np.float64),
        y1=np.asarray(y1, dtype=np.float64),
        bounds=np.asarray(bounds, dtype=np.float64),
    )


def generate_segment_candidates(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    tile_size: int = 512,
) -> SegmentIntersectionCandidates:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    left_segments = extract_segments(left)
    right_segments = extract_segments(right)
    return _generate_segment_candidates_from_tables(left_segments, right_segments, tile_size=tile_size)


def _generate_segment_candidates_from_tables(
    left_segments: SegmentTable,
    right_segments: SegmentTable,
    *,
    tile_size: int = 512,
) -> SegmentIntersectionCandidates:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    left_rows_out: list[np.ndarray] = []
    left_segment_out: list[np.ndarray] = []
    left_lookup_out: list[np.ndarray] = []
    right_rows_out: list[np.ndarray] = []
    right_segment_out: list[np.ndarray] = []
    right_lookup_out: list[np.ndarray] = []
    pairs_examined = 0

    for left_start in range(0, left_segments.count, tile_size):
        left_bounds = left_segments.bounds[left_start : left_start + tile_size]
        left_rows = left_segments.row_indices[left_start : left_start + tile_size]
        left_ids = left_segments.segment_indices[left_start : left_start + tile_size]
        for right_start in range(0, right_segments.count, tile_size):
            right_bounds = right_segments.bounds[right_start : right_start + tile_size]
            right_rows = right_segments.row_indices[right_start : right_start + tile_size]
            right_ids = right_segments.segment_indices[right_start : right_start + tile_size]
            pairs_examined += int(left_bounds.shape[0] * right_bounds.shape[0])
            intersects = (
                (left_bounds[:, None, 0] <= right_bounds[None, :, 2])
                & (left_bounds[:, None, 2] >= right_bounds[None, :, 0])
                & (left_bounds[:, None, 1] <= right_bounds[None, :, 3])
                & (left_bounds[:, None, 3] >= right_bounds[None, :, 1])
            )
            left_local, right_local = np.nonzero(intersects)
            if left_local.size == 0:
                continue
            left_rows_out.append(left_rows[left_local].astype(np.int32, copy=False))
            left_segment_out.append(left_ids[left_local].astype(np.int32, copy=False))
            left_lookup_out.append((left_start + left_local).astype(np.int32, copy=False))
            right_rows_out.append(right_rows[right_local].astype(np.int32, copy=False))
            right_segment_out.append(right_ids[right_local].astype(np.int32, copy=False))
            right_lookup_out.append((right_start + right_local).astype(np.int32, copy=False))

    if not left_rows_out:
        empty = np.asarray([], dtype=np.int32)
        return SegmentIntersectionCandidates(
            left_rows=empty,
            left_segments=empty,
            left_lookup=empty,
            right_rows=empty,
            right_segments=empty,
            right_lookup=empty,
            pairs_examined=pairs_examined,
            tile_size=tile_size,
        )
    return SegmentIntersectionCandidates(
        left_rows=np.concatenate(left_rows_out),
        left_segments=np.concatenate(left_segment_out),
        left_lookup=np.concatenate(left_lookup_out),
        right_rows=np.concatenate(right_rows_out),
        right_segments=np.concatenate(right_segment_out),
        right_lookup=np.concatenate(right_lookup_out),
        pairs_examined=pairs_examined,
        tile_size=tile_size,
    )


def _orient2d_fast(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    abx = bx - ax
    aby = by - ay
    acx = cx - ax
    acy = cy - ay
    term1 = abx * acy
    term2 = aby * acx
    det = term1 - term2
    errbound = _ORIENTATION_ERRBOUND * (np.abs(term1) + np.abs(term2))
    return det, np.abs(det) <= errbound


def _line_intersection_point(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    denominator = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if denominator == 0.0:
        return float("nan"), float("nan")
    left_det = ax * by - ay * bx
    right_det = cx * dy - cy * dx
    x = (left_det * (cx - dx) - (ax - bx) * right_det) / denominator
    y = (left_det * (cy - dy) - (ay - by) * right_det) / denominator
    return float(x), float(y)


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _exact_orientation_sign(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> int:
    det = (_fraction(bx) - _fraction(ax)) * (_fraction(cy) - _fraction(ay)) - (
        _fraction(by) - _fraction(ay)
    ) * (_fraction(cx) - _fraction(ax))
    return int(det > 0) - int(det < 0)


def _point_on_segment_exact(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> bool:
    if _exact_orientation_sign(ax, ay, bx, by, px, py) != 0:
        return False
    pxf = _fraction(px)
    pyf = _fraction(py)
    axf = _fraction(ax)
    ayf = _fraction(ay)
    bxf = _fraction(bx)
    byf = _fraction(by)
    return min(axf, bxf) <= pxf <= max(axf, bxf) and min(ayf, byf) <= pyf <= max(ayf, byf)


def _exact_intersection_point(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    axf = _fraction(ax)
    ayf = _fraction(ay)
    bxf = _fraction(bx)
    byf = _fraction(by)
    cxf = _fraction(cx)
    cyf = _fraction(cy)
    dxf = _fraction(dx)
    dyf = _fraction(dy)
    denominator = (axf - bxf) * (cyf - dyf) - (ayf - byf) * (cxf - dxf)
    if denominator == 0:
        return float("nan"), float("nan")
    left_det = axf * byf - ayf * bxf
    right_det = cxf * dyf - cyf * dxf
    x = (left_det * (cxf - dxf) - (axf - bxf) * right_det) / denominator
    y = (left_det * (cyf - dyf) - (ayf - byf) * right_det) / denominator
    return float(x), float(y)


def _unique_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    seen: set[tuple[Fraction, Fraction]] = set()
    for x, y in points:
        key = (_fraction(x), _fraction(y))
        if key in seen:
            continue
        seen.add(key)
        unique.append((float(x), float(y)))
    return unique


def _sort_collinear_points(
    points: list[tuple[float, float]],
    *,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> list[tuple[float, float]]:
    use_x = abs(bx - ax) >= abs(by - ay)

    def _key(point: tuple[float, float]) -> tuple[Fraction, Fraction]:
        x, y = point
        if use_x:
            return (_fraction(x), _fraction(y))
        return (_fraction(y), _fraction(x))

    return sorted(points, key=_key)


def _classify_exact_pair(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> tuple[SegmentIntersectionKind, tuple[float, float], tuple[float, float, float, float]]:
    a_is_point = _fraction(ax) == _fraction(bx) and _fraction(ay) == _fraction(by)
    c_is_point = _fraction(cx) == _fraction(dx) and _fraction(cy) == _fraction(dy)

    if a_is_point and c_is_point:
        if _fraction(ax) == _fraction(cx) and _fraction(ay) == _fraction(cy):
            return SegmentIntersectionKind.TOUCH, (float(ax), float(ay)), (float("nan"),) * 4
        return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4

    if a_is_point:
        if _point_on_segment_exact(ax, ay, cx, cy, dx, dy):
            return SegmentIntersectionKind.TOUCH, (float(ax), float(ay)), (float("nan"),) * 4
        return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4

    if c_is_point:
        if _point_on_segment_exact(cx, cy, ax, ay, bx, by):
            return SegmentIntersectionKind.TOUCH, (float(cx), float(cy)), (float("nan"),) * 4
        return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4

    o1 = _exact_orientation_sign(ax, ay, bx, by, cx, cy)
    o2 = _exact_orientation_sign(ax, ay, bx, by, dx, dy)
    o3 = _exact_orientation_sign(cx, cy, dx, dy, ax, ay)
    o4 = _exact_orientation_sign(cx, cy, dx, dy, bx, by)

    if o1 * o2 < 0 and o3 * o4 < 0:
        point = _exact_intersection_point(ax, ay, bx, by, cx, cy, dx, dy)
        return SegmentIntersectionKind.PROPER, point, (float("nan"),) * 4

    if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
        shared = _unique_points(
            [
                point
                for point in ((ax, ay), (bx, by), (cx, cy), (dx, dy))
                if _point_on_segment_exact(point[0], point[1], ax, ay, bx, by)
                and _point_on_segment_exact(point[0], point[1], cx, cy, dx, dy)
            ]
        )
        if not shared:
            return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4
        shared = _sort_collinear_points(shared, ax=ax, ay=ay, bx=bx, by=by)
        if len(shared) == 1:
            x, y = shared[0]
            return SegmentIntersectionKind.TOUCH, (x, y), (float("nan"),) * 4
        (sx0, sy0), (sx1, sy1) = shared[0], shared[-1]
        return SegmentIntersectionKind.OVERLAP, (float("nan"), float("nan")), (sx0, sy0, sx1, sy1)

    if o1 == 0 and _point_on_segment_exact(cx, cy, ax, ay, bx, by):
        return SegmentIntersectionKind.TOUCH, (float(cx), float(cy)), (float("nan"),) * 4
    if o2 == 0 and _point_on_segment_exact(dx, dy, ax, ay, bx, by):
        return SegmentIntersectionKind.TOUCH, (float(dx), float(dy)), (float("nan"),) * 4
    if o3 == 0 and _point_on_segment_exact(ax, ay, cx, cy, dx, dy):
        return SegmentIntersectionKind.TOUCH, (float(ax), float(ay)), (float("nan"),) * 4
    if o4 == 0 and _point_on_segment_exact(bx, by, cx, cy, dx, dy):
        return SegmentIntersectionKind.TOUCH, (float(bx), float(by)), (float("nan"),) * 4

    return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4


def _classify_exact_rows(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    rows: np.ndarray,
    kinds: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
    overlap_x0: np.ndarray,
    overlap_y0: np.ndarray,
    overlap_x1: np.ndarray,
    overlap_y1: np.ndarray,
) -> None:
    for row in rows.tolist():
        kind, point, overlap = _classify_exact_pair(
            float(ax[row]),
            float(ay[row]),
            float(bx[row]),
            float(by[row]),
            float(cx[row]),
            float(cy[row]),
            float(dx[row]),
            float(dy[row]),
        )
        kinds[row] = int(kind)
        point_x[row], point_y[row] = point
        overlap_x0[row], overlap_y0[row], overlap_x1[row], overlap_y1[row] = overlap


def classify_segment_intersections(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    candidate_pairs: SegmentIntersectionCandidates | None = None,
    tile_size: int = 512,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> SegmentIntersectionResult:
    left_segments = extract_segments(left)
    right_segments = extract_segments(right)
    pairs = (
        candidate_pairs
        if candidate_pairs is not None
        else _generate_segment_candidates_from_tables(left_segments, right_segments, tile_size=tile_size)
    )
    runtime_selection = _select_segment_runtime(dispatch_mode, candidate_count=int(pairs.count))
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
        return _classify_segment_intersections_gpu(
            left_segments=left_segments,
            right_segments=right_segments,
            pairs=pairs,
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
        )
    return _classify_segment_intersections_from_tables(
        left_segments=left_segments,
        right_segments=right_segments,
        pairs=pairs,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
    )


def _classify_segment_intersections_gpu(
    *,
    left_segments: SegmentTable,
    right_segments: SegmentTable,
    pairs: SegmentIntersectionCandidates,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    robustness_plan: RobustnessPlan,
) -> SegmentIntersectionResult:
    if pairs.count == 0:
        empty_i32 = np.asarray([], dtype=np.int32)
        empty_f64 = np.asarray([], dtype=np.float64)
        return SegmentIntersectionResult(
            left_rows=empty_i32,
            left_segments=empty_i32,
            left_lookup=empty_i32,
            right_rows=empty_i32,
            right_segments=empty_i32,
            right_lookup=empty_i32,
            kinds=empty_i32,
            point_x=empty_f64,
            point_y=empty_f64,
            overlap_x0=empty_f64,
            overlap_y0=empty_f64,
            overlap_x1=empty_f64,
            overlap_y1=empty_f64,
            candidate_pairs=0,
            ambiguous_rows=empty_i32,
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
        )

    runtime = get_cuda_runtime()
    left_lookup = None
    right_lookup = None
    left_x0 = None
    left_y0 = None
    left_x1 = None
    left_y1 = None
    right_x0 = None
    right_y0 = None
    right_x1 = None
    right_y1 = None
    device_kinds = None
    device_point_x = None
    device_point_y = None
    device_overlap_x0 = None
    device_overlap_y0 = None
    device_overlap_x1 = None
    device_overlap_y1 = None
    device_ambiguous_mask = None
    device_ambiguous_rows = None
    device_left_rows = None
    device_left_segments = None
    device_right_rows = None
    device_right_segments = None
    success = False
    try:
        left_lookup = runtime.from_host(pairs.left_lookup)
        right_lookup = runtime.from_host(pairs.right_lookup)
        left_x0 = runtime.from_host(left_segments.x0)
        left_y0 = runtime.from_host(left_segments.y0)
        left_x1 = runtime.from_host(left_segments.x1)
        left_y1 = runtime.from_host(left_segments.y1)
        right_x0 = runtime.from_host(right_segments.x0)
        right_y0 = runtime.from_host(right_segments.y0)
        right_x1 = runtime.from_host(right_segments.x1)
        right_y1 = runtime.from_host(right_segments.y1)

        device_kinds = runtime.allocate((pairs.count,), np.int8)
        device_point_x = runtime.allocate((pairs.count,), np.float64)
        device_point_y = runtime.allocate((pairs.count,), np.float64)
        device_overlap_x0 = runtime.allocate((pairs.count,), np.float64)
        device_overlap_y0 = runtime.allocate((pairs.count,), np.float64)
        device_overlap_x1 = runtime.allocate((pairs.count,), np.float64)
        device_overlap_y1 = runtime.allocate((pairs.count,), np.float64)
        device_ambiguous_mask = runtime.allocate((pairs.count,), np.uint8)

        kernel = _segment_intersection_warp_skip_kernels()["classify_segment_pairs_warp_skip"]
        ptr = runtime.pointer
        params = (
            (
                ptr(left_lookup),
                ptr(right_lookup),
                ptr(left_x0),
                ptr(left_y0),
                ptr(left_x1),
                ptr(left_y1),
                ptr(right_x0),
                ptr(right_y0),
                ptr(right_x1),
                ptr(right_y1),
                ptr(device_kinds),
                ptr(device_point_x),
                ptr(device_point_y),
                ptr(device_overlap_x0),
                ptr(device_overlap_y0),
                ptr(device_overlap_x1),
                ptr(device_overlap_y1),
                ptr(device_ambiguous_mask),
                pairs.count,
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
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(kernel, pairs.count)
        runtime.launch(kernel, grid=grid, block=block, params=params)

        device_ambiguous_rows = compact_indices(device_ambiguous_mask).values

        kinds = runtime.copy_device_to_host(device_kinds)
        point_x = runtime.copy_device_to_host(device_point_x)
        point_y = runtime.copy_device_to_host(device_point_y)
        overlap_x0 = runtime.copy_device_to_host(device_overlap_x0)
        overlap_y0 = runtime.copy_device_to_host(device_overlap_y0)
        overlap_x1 = runtime.copy_device_to_host(device_overlap_x1)
        overlap_y1 = runtime.copy_device_to_host(device_overlap_y1)
        ambiguous_rows = runtime.copy_device_to_host(device_ambiguous_rows).astype(np.int32, copy=False)

        for row in ambiguous_rows.tolist():
            left_index = int(pairs.left_lookup[row])
            right_index = int(pairs.right_lookup[row])
            kind, point, overlap = _classify_exact_pair(
                float(left_segments.x0[left_index]),
                float(left_segments.y0[left_index]),
                float(left_segments.x1[left_index]),
                float(left_segments.y1[left_index]),
                float(right_segments.x0[right_index]),
                float(right_segments.y0[right_index]),
                float(right_segments.x1[right_index]),
                float(right_segments.y1[right_index]),
            )
            kinds[row] = int(kind)
            point_x[row], point_y[row] = point
            overlap_x0[row], overlap_y0[row], overlap_x1[row], overlap_y1[row] = overlap

        runtime.copy_host_to_device(np.asarray(kinds, dtype=np.int8), device_kinds)
        runtime.copy_host_to_device(np.asarray(point_x, dtype=np.float64), device_point_x)
        runtime.copy_host_to_device(np.asarray(point_y, dtype=np.float64), device_point_y)
        runtime.copy_host_to_device(np.asarray(overlap_x0, dtype=np.float64), device_overlap_x0)
        runtime.copy_host_to_device(np.asarray(overlap_y0, dtype=np.float64), device_overlap_y0)
        runtime.copy_host_to_device(np.asarray(overlap_x1, dtype=np.float64), device_overlap_x1)
        runtime.copy_host_to_device(np.asarray(overlap_y1, dtype=np.float64), device_overlap_y1)

        device_left_rows = runtime.from_host(pairs.left_rows)
        device_left_segments = runtime.from_host(pairs.left_segments)
        device_right_rows = runtime.from_host(pairs.right_rows)
        device_right_segments = runtime.from_host(pairs.right_segments)
        success = True
        return SegmentIntersectionResult(
            left_rows=pairs.left_rows.copy(),
            left_segments=pairs.left_segments.copy(),
            left_lookup=pairs.left_lookup.copy(),
            right_rows=pairs.right_rows.copy(),
            right_segments=pairs.right_segments.copy(),
            right_lookup=pairs.right_lookup.copy(),
            kinds=np.asarray(kinds, dtype=np.int8),
            point_x=np.asarray(point_x, dtype=np.float64),
            point_y=np.asarray(point_y, dtype=np.float64),
            overlap_x0=np.asarray(overlap_x0, dtype=np.float64),
            overlap_y0=np.asarray(overlap_y0, dtype=np.float64),
            overlap_x1=np.asarray(overlap_x1, dtype=np.float64),
            overlap_y1=np.asarray(overlap_y1, dtype=np.float64),
            candidate_pairs=int(pairs.count),
            ambiguous_rows=ambiguous_rows,
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
            device_state=SegmentIntersectionDeviceState(
                left_rows=device_left_rows,
                left_segments=device_left_segments,
                left_lookup=left_lookup,
                right_rows=device_right_rows,
                right_segments=device_right_segments,
                right_lookup=right_lookup,
                kinds=device_kinds,
                point_x=device_point_x,
                point_y=device_point_y,
                overlap_x0=device_overlap_x0,
                overlap_y0=device_overlap_y0,
                overlap_x1=device_overlap_x1,
                overlap_y1=device_overlap_y1,
                ambiguous_rows=device_ambiguous_rows,
            ),
        )
    finally:
        runtime.free(left_x0)
        runtime.free(left_y0)
        runtime.free(left_x1)
        runtime.free(left_y1)
        runtime.free(right_x0)
        runtime.free(right_y0)
        runtime.free(right_x1)
        runtime.free(right_y1)
        runtime.free(device_ambiguous_mask)
        if not success:
            runtime.free(left_lookup)
            runtime.free(right_lookup)
            runtime.free(device_ambiguous_rows)
            runtime.free(device_kinds)
            runtime.free(device_point_x)
            runtime.free(device_point_y)
            runtime.free(device_overlap_x0)
            runtime.free(device_overlap_y0)
            runtime.free(device_overlap_x1)
            runtime.free(device_overlap_y1)
            runtime.free(device_left_rows)
            runtime.free(device_left_segments)
            runtime.free(device_right_rows)
            runtime.free(device_right_segments)


def _classify_segment_intersections_from_tables(
    *,
    left_segments: SegmentTable,
    right_segments: SegmentTable,
    pairs: SegmentIntersectionCandidates,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    robustness_plan: RobustnessPlan,
) -> SegmentIntersectionResult:
    if pairs.count == 0:
        empty_i32 = np.asarray([], dtype=np.int32)
        empty_f64 = np.asarray([], dtype=np.float64)
        return SegmentIntersectionResult(
            left_rows=empty_i32,
            left_segments=empty_i32,
            left_lookup=empty_i32,
            right_rows=empty_i32,
            right_segments=empty_i32,
            right_lookup=empty_i32,
            kinds=empty_i32,
            point_x=empty_f64,
            point_y=empty_f64,
            overlap_x0=empty_f64,
            overlap_y0=empty_f64,
            overlap_x1=empty_f64,
            overlap_y1=empty_f64,
            candidate_pairs=0,
            ambiguous_rows=empty_i32,
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
        )

    left_lookup = pairs.left_lookup
    right_lookup = pairs.right_lookup

    ax = left_segments.x0[left_lookup]
    ay = left_segments.y0[left_lookup]
    bx = left_segments.x1[left_lookup]
    by = left_segments.y1[left_lookup]
    cx = right_segments.x0[right_lookup]
    cy = right_segments.y0[right_lookup]
    dx = right_segments.x1[right_lookup]
    dy = right_segments.y1[right_lookup]

    o1, a1 = _orient2d_fast(ax, ay, bx, by, cx, cy)
    o2, a2 = _orient2d_fast(ax, ay, bx, by, dx, dy)
    o3, a3 = _orient2d_fast(cx, cy, dx, dy, ax, ay)
    o4, a4 = _orient2d_fast(cx, cy, dx, dy, bx, by)

    zero_left = (ax == bx) & (ay == by)
    zero_right = (cx == dx) & (cy == dy)
    sign1 = np.sign(o1).astype(np.int8, copy=False)
    sign2 = np.sign(o2).astype(np.int8, copy=False)
    sign3 = np.sign(o3).astype(np.int8, copy=False)
    sign4 = np.sign(o4).astype(np.int8, copy=False)

    ambiguous_mask = (
        a1
        | a2
        | a3
        | a4
        | zero_left
        | zero_right
        | (sign1 == 0)
        | (sign2 == 0)
        | (sign3 == 0)
        | (sign4 == 0)
    )
    proper_mask = (~ambiguous_mask) & (sign1 * sign2 < 0) & (sign3 * sign4 < 0)
    disjoint_mask = (~ambiguous_mask) & ~proper_mask

    count = int(pairs.count)
    kinds = np.full(count, int(SegmentIntersectionKind.DISJOINT), dtype=np.int8)
    point_x = np.full(count, np.nan, dtype=np.float64)
    point_y = np.full(count, np.nan, dtype=np.float64)
    overlap_x0 = np.full(count, np.nan, dtype=np.float64)
    overlap_y0 = np.full(count, np.nan, dtype=np.float64)
    overlap_x1 = np.full(count, np.nan, dtype=np.float64)
    overlap_y1 = np.full(count, np.nan, dtype=np.float64)

    kinds[proper_mask] = int(SegmentIntersectionKind.PROPER)
    proper_rows = np.flatnonzero(proper_mask)
    for row in proper_rows.tolist():
        point_x[row], point_y[row] = _line_intersection_point(
            float(ax[row]),
            float(ay[row]),
            float(bx[row]),
            float(by[row]),
            float(cx[row]),
            float(cy[row]),
            float(dx[row]),
            float(dy[row]),
        )
    del disjoint_mask

    ambiguous_rows = np.flatnonzero(ambiguous_mask).astype(np.int32, copy=False)
    if ambiguous_rows.size:
        _classify_exact_rows(
            ax,
            ay,
            bx,
            by,
            cx,
            cy,
            dx,
            dy,
            ambiguous_rows,
            kinds,
            point_x,
            point_y,
            overlap_x0,
            overlap_y0,
            overlap_x1,
            overlap_y1,
        )

    return SegmentIntersectionResult(
        left_rows=pairs.left_rows.copy(),
        left_segments=pairs.left_segments.copy(),
        left_lookup=pairs.left_lookup.copy(),
        right_rows=pairs.right_rows.copy(),
        right_segments=pairs.right_segments.copy(),
        right_lookup=pairs.right_lookup.copy(),
        kinds=kinds,
        point_x=point_x,
        point_y=point_y,
        overlap_x0=overlap_x0,
        overlap_y0=overlap_y0,
        overlap_x1=overlap_x1,
        overlap_y1=overlap_y1,
        candidate_pairs=int(pairs.count),
        ambiguous_rows=ambiguous_rows,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
    )


def benchmark_segment_intersections(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    tile_size: int = 512,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> SegmentIntersectionBenchmark:
    started = perf_counter()
    result = classify_segment_intersections(left, right, tile_size=tile_size, dispatch_mode=dispatch_mode)
    elapsed = perf_counter() - started
    return SegmentIntersectionBenchmark(
        rows_left=left.row_count,
        rows_right=right.row_count,
        candidate_pairs=result.candidate_pairs,
        disjoint_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.DISJOINT))),
        proper_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.PROPER))),
        touch_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.TOUCH))),
        overlap_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.OVERLAP))),
        ambiguous_pairs=int(result.ambiguous_rows.size),
        elapsed_seconds=elapsed,
    )
