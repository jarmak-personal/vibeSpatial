from __future__ import annotations

import numpy as np

from vibespatial.cuda_runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.owned_geometry import FAMILY_TAGS, OwnedGeometryArray
from vibespatial.precision import PrecisionMode

_POINT_DISTANCE_KERNEL_SOURCE_TEMPLATE = """
typedef {compute_type} compute_t;

#if !defined(INFINITY)
#define INFINITY __longlong_as_double(0x7FF0000000000000LL)
#endif

// Centered coordinate read: subtract center in fp64, then cast to compute_t.
// When compute_t is double, this is a no-op identity.  When compute_t is float,
// the fp64 subtraction removes large-magnitude bias before the fp32 cast,
// preserving relative precision for the local arithmetic.
#define CX(val) ((compute_t)((val) - center_x))
#define CY(val) ((compute_t)((val) - center_y))

// ---------------------------------------------------------------------------
// Tier 1 NVRTC: point-to-segment squared distance (device helper)
// ---------------------------------------------------------------------------
extern "C" __device__ inline compute_t point_segment_sq_distance(
    compute_t px, compute_t py,
    compute_t ax, compute_t ay,
    compute_t bx, compute_t by
) {{
  const compute_t dx = bx - ax;
  const compute_t dy = by - ay;
  const compute_t len_sq = dx * dx + dy * dy;
  compute_t t;
  if (len_sq < (compute_t)1e-30) {{
    t = (compute_t)0.0;
  }} else {{
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
    if (t < (compute_t)0.0) t = (compute_t)0.0;
    else if (t > (compute_t)1.0) t = (compute_t)1.0;
  }}
  const compute_t cx = ax + t * dx;
  const compute_t cy = ay + t * dy;
  const compute_t ex = px - cx;
  const compute_t ey = py - cy;
  return ex * ex + ey * ey;
}}

// ---------------------------------------------------------------------------
// Tier 1 NVRTC: min squared distance from a point to a coordinate range
// ---------------------------------------------------------------------------
extern "C" __device__ inline compute_t point_coords_min_sq_distance(
    compute_t px, compute_t py,
    const double* x, const double* y,
    double center_x, double center_y,
    int coord_start, int coord_end
) {{
  compute_t best = (compute_t)INFINITY;
  for (int c = coord_start + 1; c < coord_end; ++c) {{
    const compute_t d = point_segment_sq_distance(
        px, py, CX(x[c - 1]), CY(y[c - 1]), CX(x[c]), CY(y[c]));
    if (d < best) best = d;
  }}
  return best;
}}

// ---------------------------------------------------------------------------
// Winding-number point-in-polygon test (even-odd rule).
// This test uses centered coordinates for consistency but the boolean
// result is not precision-sensitive for well-separated geometries.
// ---------------------------------------------------------------------------
extern "C" __device__ inline bool point_inside_polygon(
    compute_t px, compute_t py,
    const double* x, const double* y,
    double center_x, double center_y,
    const int* geometry_offsets,
    const int* ring_offsets,
    int polygon_row
) {{
  const int ring_start = geometry_offsets[polygon_row];
  const int ring_end   = geometry_offsets[polygon_row + 1];
  bool inside = false;
  for (int ring = ring_start; ring < ring_end; ++ring) {{
    const int cs = ring_offsets[ring];
    const int ce = ring_offsets[ring + 1];
    if ((ce - cs) < 2) continue;
    for (int c = cs + 1; c < ce; ++c) {{
      const compute_t ax = CX(x[c - 1]), ay = CY(y[c - 1]);
      const compute_t bx = CX(x[c]),     by = CY(y[c]);
      const compute_t cross_val = ((px - ax) * (by - ay)) - ((py - ay) * (bx - ax));
      const compute_t scale = (bx > ax ? bx - ax : ax - bx) + (by > ay ? by - ay : ay - by) + (compute_t)1.0;
      if ((cross_val > (compute_t)0.0 ? cross_val : -cross_val) <= ((compute_t)1e-7 * scale)) {{
        const compute_t minx = ax < bx ? ax : bx;
        const compute_t maxx = ax > bx ? ax : bx;
        const compute_t miny = ay < by ? ay : by;
        const compute_t maxy = ay > by ? ay : by;
        if (px >= (minx - (compute_t)1e-7) && px <= (maxx + (compute_t)1e-7) &&
            py >= (miny - (compute_t)1e-7) && py <= (maxy + (compute_t)1e-7)) {{
          return true;
        }}
      }}
      if (((ay > py) != (by > py)) &&
          (px <= (((bx - ax) * (py - ay)) / ((by - ay) + (compute_t)0.0)) + ax)) {{
        inside = !inside;
      }}
    }}
  }}
  return inside;
}}

// ---------------------------------------------------------------------------
// Tier 1 NVRTC kernels: point distance to linestring / polygon families
// ---------------------------------------------------------------------------

extern "C" __global__ void point_linestring_distance_from_owned(
    const unsigned char* query_validity,
    const signed char*   query_tags,
    const int*           query_family_row_offsets,
    const int*           query_geometry_offsets,
    const unsigned char* query_empty_mask,
    const double*        query_x,
    const double*        query_y,
    int                  query_point_tag,
    const unsigned char* tree_validity,
    const signed char*   tree_tags,
    const int*           tree_family_row_offsets,
    const int*           tree_geometry_offsets,
    const unsigned char* tree_empty_mask,
    const double*        tree_x,
    const double*        tree_y,
    int                  tree_line_tag,
    const int*           left_idx,
    const int*           right_idx,
    double*              out_distances,
    int                  exclusive,
    int                  pair_count,
    double               center_x,
    double               center_y
) {{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];

  if (!query_validity[li] || !tree_validity[ri]) {{
    out_distances[i] = INFINITY; return;
  }}
  if (query_tags[li] != query_point_tag || tree_tags[ri] != tree_line_tag) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qrow = query_family_row_offsets[li];
  const int trow = tree_family_row_offsets[ri];
  if (qrow < 0 || trow < 0 || query_empty_mask[qrow] || tree_empty_mask[trow]) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qcoord = query_geometry_offsets[qrow];
  const double raw_px = query_x[qcoord];
  const double raw_py = query_y[qcoord];
  if (isnan(raw_px) || isnan(raw_py)) {{ out_distances[i] = INFINITY; return; }}

  const compute_t px = CX(raw_px);
  const compute_t py = CY(raw_py);

  const int coord_start = tree_geometry_offsets[trow];
  const int coord_end   = tree_geometry_offsets[trow + 1];

  const compute_t sq = point_coords_min_sq_distance(px, py, tree_x, tree_y,
                                                     center_x, center_y,
                                                     coord_start, coord_end);
  out_distances[i] = (double)sqrt((double)sq);
}}

extern "C" __global__ void point_multilinestring_distance_from_owned(
    const unsigned char* query_validity,
    const signed char*   query_tags,
    const int*           query_family_row_offsets,
    const int*           query_geometry_offsets,
    const unsigned char* query_empty_mask,
    const double*        query_x,
    const double*        query_y,
    int                  query_point_tag,
    const unsigned char* tree_validity,
    const signed char*   tree_tags,
    const int*           tree_family_row_offsets,
    const int*           tree_geometry_offsets,
    const int*           tree_part_offsets,
    const unsigned char* tree_empty_mask,
    const double*        tree_x,
    const double*        tree_y,
    int                  tree_multiline_tag,
    const int*           left_idx,
    const int*           right_idx,
    double*              out_distances,
    int                  exclusive,
    int                  pair_count,
    double               center_x,
    double               center_y
) {{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];

  if (!query_validity[li] || !tree_validity[ri]) {{
    out_distances[i] = INFINITY; return;
  }}
  if (query_tags[li] != query_point_tag || tree_tags[ri] != tree_multiline_tag) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qrow = query_family_row_offsets[li];
  const int trow = tree_family_row_offsets[ri];
  if (qrow < 0 || trow < 0 || query_empty_mask[qrow] || tree_empty_mask[trow]) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qcoord = query_geometry_offsets[qrow];
  const double raw_px = query_x[qcoord];
  const double raw_py = query_y[qcoord];
  if (isnan(raw_px) || isnan(raw_py)) {{ out_distances[i] = INFINITY; return; }}

  const compute_t px = CX(raw_px);
  const compute_t py = CY(raw_py);

  const int part_start = tree_geometry_offsets[trow];
  const int part_end   = tree_geometry_offsets[trow + 1];

  compute_t best = (compute_t)INFINITY;
  for (int part = part_start; part < part_end; ++part) {{
    const int cs = tree_part_offsets[part];
    const int ce = tree_part_offsets[part + 1];
    const compute_t sq = point_coords_min_sq_distance(px, py, tree_x, tree_y,
                                                       center_x, center_y, cs, ce);
    if (sq < best) best = sq;
  }}
  out_distances[i] = (double)sqrt((double)best);
}}

extern "C" __global__ void point_polygon_distance_from_owned(
    const unsigned char* query_validity,
    const signed char*   query_tags,
    const int*           query_family_row_offsets,
    const int*           query_geometry_offsets,
    const unsigned char* query_empty_mask,
    const double*        query_x,
    const double*        query_y,
    int                  query_point_tag,
    const unsigned char* tree_validity,
    const signed char*   tree_tags,
    const int*           tree_family_row_offsets,
    const int*           tree_polygon_geometry_offsets,
    const int*           tree_ring_offsets,
    const unsigned char* tree_empty_mask,
    const double*        tree_x,
    const double*        tree_y,
    int                  tree_polygon_tag,
    const int*           left_idx,
    const int*           right_idx,
    double*              out_distances,
    int                  exclusive,
    int                  pair_count,
    double               center_x,
    double               center_y
) {{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];

  if (!query_validity[li] || !tree_validity[ri]) {{
    out_distances[i] = INFINITY; return;
  }}
  if (query_tags[li] != query_point_tag || tree_tags[ri] != tree_polygon_tag) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qrow = query_family_row_offsets[li];
  const int trow = tree_family_row_offsets[ri];
  if (qrow < 0 || trow < 0 || query_empty_mask[qrow] || tree_empty_mask[trow]) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qcoord = query_geometry_offsets[qrow];
  const double raw_px = query_x[qcoord];
  const double raw_py = query_y[qcoord];
  if (isnan(raw_px) || isnan(raw_py)) {{ out_distances[i] = INFINITY; return; }}

  const compute_t px = CX(raw_px);
  const compute_t py = CY(raw_py);

  if (point_inside_polygon(px, py, tree_x, tree_y, center_x, center_y,
                            tree_polygon_geometry_offsets, tree_ring_offsets, trow)) {{
    out_distances[i] = 0.0;
    return;
  }}

  const int ring_start = tree_polygon_geometry_offsets[trow];
  const int ring_end   = tree_polygon_geometry_offsets[trow + 1];
  compute_t best = (compute_t)INFINITY;
  for (int ring = ring_start; ring < ring_end; ++ring) {{
    const int cs = tree_ring_offsets[ring];
    const int ce = tree_ring_offsets[ring + 1];
    const compute_t sq = point_coords_min_sq_distance(px, py, tree_x, tree_y,
                                                       center_x, center_y, cs, ce);
    if (sq < best) best = sq;
  }}
  out_distances[i] = (double)sqrt((double)best);
}}

extern "C" __global__ void point_multipolygon_distance_from_owned(
    const unsigned char* query_validity,
    const signed char*   query_tags,
    const int*           query_family_row_offsets,
    const int*           query_geometry_offsets,
    const unsigned char* query_empty_mask,
    const double*        query_x,
    const double*        query_y,
    int                  query_point_tag,
    const unsigned char* tree_validity,
    const signed char*   tree_tags,
    const int*           tree_family_row_offsets,
    const int*           tree_geometry_offsets,
    const int*           tree_part_offsets,
    const int*           tree_ring_offsets,
    const unsigned char* tree_empty_mask,
    const double*        tree_x,
    const double*        tree_y,
    int                  tree_multipolygon_tag,
    const int*           left_idx,
    const int*           right_idx,
    double*              out_distances,
    int                  exclusive,
    int                  pair_count,
    double               center_x,
    double               center_y
) {{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];

  if (!query_validity[li] || !tree_validity[ri]) {{
    out_distances[i] = INFINITY; return;
  }}
  if (query_tags[li] != query_point_tag || tree_tags[ri] != tree_multipolygon_tag) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qrow = query_family_row_offsets[li];
  const int trow = tree_family_row_offsets[ri];
  if (qrow < 0 || trow < 0 || query_empty_mask[qrow] || tree_empty_mask[trow]) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qcoord = query_geometry_offsets[qrow];
  const double raw_px = query_x[qcoord];
  const double raw_py = query_y[qcoord];
  if (isnan(raw_px) || isnan(raw_py)) {{ out_distances[i] = INFINITY; return; }}

  const compute_t px = CX(raw_px);
  const compute_t py = CY(raw_py);

  const int polygon_start = tree_geometry_offsets[trow];
  const int polygon_end   = tree_geometry_offsets[trow + 1];

  compute_t best = (compute_t)INFINITY;
  for (int polygon = polygon_start; polygon < polygon_end; ++polygon) {{
    const int ring_start = tree_part_offsets[polygon];
    const int ring_end   = tree_part_offsets[polygon + 1];
    bool inside = false;
    compute_t poly_best = (compute_t)INFINITY;
    for (int ring = ring_start; ring < ring_end; ++ring) {{
      const int cs = tree_ring_offsets[ring];
      const int ce = tree_ring_offsets[ring + 1];
      if ((ce - cs) < 2) continue;
      bool ring_inside = false;
      bool on_boundary = false;
      for (int c = cs + 1; c < ce; ++c) {{
        const compute_t ax = CX(tree_x[c - 1]), ay = CY(tree_y[c - 1]);
        const compute_t bx = CX(tree_x[c]),     by = CY(tree_y[c]);
        const compute_t cross_val = ((px - ax) * (by - ay)) - ((py - ay) * (bx - ax));
        const compute_t scale = (bx > ax ? bx - ax : ax - bx) + (by > ay ? by - ay : ay - by) + (compute_t)1.0;
        if ((cross_val > (compute_t)0.0 ? cross_val : -cross_val) <= ((compute_t)1e-7 * scale)) {{
          const compute_t minx = ax < bx ? ax : bx;
          const compute_t maxx = ax > bx ? ax : bx;
          const compute_t miny = ay < by ? ay : by;
          const compute_t maxy = ay > by ? ay : by;
          if (px >= (minx - (compute_t)1e-7) && px <= (maxx + (compute_t)1e-7) &&
              py >= (miny - (compute_t)1e-7) && py <= (maxy + (compute_t)1e-7)) {{
            on_boundary = true;
          }}
        }}
        if (((ay > py) != (by > py)) &&
            (px <= (((bx - ax) * (py - ay)) / ((by - ay) + (compute_t)0.0)) + ax)) {{
          ring_inside = !ring_inside;
        }}
      }}
      if (on_boundary) {{ out_distances[i] = 0.0; return; }}
      if (ring_inside) inside = !inside;
      const compute_t sq = point_coords_min_sq_distance(px, py, tree_x, tree_y,
                                                         center_x, center_y, cs, ce);
      if (sq < poly_best) poly_best = sq;
    }}
    if (inside) {{ out_distances[i] = 0.0; return; }}
    if (poly_best < best) best = poly_best;
  }}
  out_distances[i] = (double)sqrt((double)best);
}}
"""

_POINT_DISTANCE_KERNEL_NAMES = (
    "point_linestring_distance_from_owned",
    "point_multilinestring_distance_from_owned",
    "point_polygon_distance_from_owned",
    "point_multipolygon_distance_from_owned",
)


def _format_distance_kernel_source(compute_type: str = "double") -> str:
    return _POINT_DISTANCE_KERNEL_SOURCE_TEMPLATE.format(compute_type=compute_type)


_POINT_DISTANCE_KERNEL_SOURCE = _format_distance_kernel_source("double")

from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("point-distance", _POINT_DISTANCE_KERNEL_SOURCE, _POINT_DISTANCE_KERNEL_NAMES),
])


def _point_distance_kernels(compute_type: str = "double"):
    source = _format_distance_kernel_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"point-distance-{compute_type}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_POINT_DISTANCE_KERNEL_NAMES,
    )


# Family -> (kernel name, needs part_offsets, needs ring_offsets)
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
        from vibespatial.adaptive_runtime import get_cached_snapshot
        snapshot = get_cached_snapshot()
        use_fp32 = not snapshot.device_profile.favors_native_fp64
    else:
        use_fp32 = compute_precision is PrecisionMode.FP32
    compute_type = "float" if use_fp32 else "double"

    from vibespatial.residency import Residency, TransferTrigger

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
