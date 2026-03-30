"""NVRTC kernel sources for make_valid_pipeline."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GPU validity detection kernel (ADR-0019: compact-invalid-row pattern)
# ---------------------------------------------------------------------------
# Checks per-ring: ring closure (first == last), minimum vertex count (>=4),
# and flags rings that fail.  Self-intersection detection reuses the
# classify_segment_intersections infrastructure from segment_primitives.

_VALIDITY_KERNEL_SOURCE_TEMPLATE = """
typedef {compute_type} compute_t;

extern "C" __global__ void check_ring_validity(
    const double* ring_x,
    const double* ring_y,
    const int* ring_offsets,
    unsigned char* ring_valid,
    int ring_count
) {{
  const int ring = blockIdx.x * blockDim.x + threadIdx.x;
  if (ring >= ring_count) {{
    return;
  }}
  const int start = ring_offsets[ring];
  const int end = ring_offsets[ring + 1];
  const int vertex_count = end - start;

  /* Minimum vertex count: a closed polygon ring needs >= 4 coords (3 unique + closure) */
  if (vertex_count < 4) {{
    ring_valid[ring] = 0;
    return;
  }}

  /* Ring closure check: first coord must equal last coord.
     Storage is always fp64; cast to compute_t for the comparison.
     Tolerance is precision-dependent: fp64 uses 1e-24, fp32 uses 1e-10
     (squared epsilon for respective precision floors). */
  const compute_t x0 = (compute_t)ring_x[start];
  const compute_t y0 = (compute_t)ring_y[start];
  const compute_t x_last = (compute_t)ring_x[end - 1];
  const compute_t y_last = (compute_t)ring_y[end - 1];
  const compute_t dx = x0 - x_last;
  const compute_t dy = y0 - y_last;
  if (dx * dx + dy * dy > (compute_t){closure_tolerance}) {{
    ring_valid[ring] = 0;
    return;
  }}

  ring_valid[ring] = 1;
}}

extern "C" __global__ void reduce_ring_to_polygon_validity(
    const unsigned char* ring_valid,
    const int* geometry_offsets,
    unsigned char* polygon_valid,
    int polygon_count
) {{
  const int poly = blockIdx.x * blockDim.x + threadIdx.x;
  if (poly >= polygon_count) {{
    return;
  }}
  const int ring_start = geometry_offsets[poly];
  const int ring_end = geometry_offsets[poly + 1];
  if (ring_start >= ring_end) {{
    polygon_valid[poly] = 0;
    return;
  }}
  for (int r = ring_start; r < ring_end; r++) {{
    if (!ring_valid[r]) {{
      polygon_valid[poly] = 0;
      return;
    }}
  }}
  polygon_valid[poly] = 1;
}}
"""
_VALIDITY_KERNEL_NAMES = ("check_ring_validity", "reduce_ring_to_polygon_validity")


def _format_validity_kernel_source(compute_type: str = "double") -> str:
    """Format kernel source with precision-dependent typedef and tolerance."""
    closure_tolerance = "1e-10" if compute_type == "float" else "1e-24"
    return _VALIDITY_KERNEL_SOURCE_TEMPLATE.format(
        compute_type=compute_type,
        closure_tolerance=closure_tolerance,
    )


# Precompile both fp64 and fp32 variants (ADR-0002 + ADR-0034)
_VALIDITY_KERNEL_SOURCE_FP64 = _format_validity_kernel_source("double")
_VALIDITY_KERNEL_SOURCE_FP32 = _format_validity_kernel_source("float")
