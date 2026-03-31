"""Shared CUDA device functions: point-in-ring even-odd ray-cast.

Three variants at increasing classification granularity:

- ``POINT_IN_RING_DEVICE`` -- fast bool (inside/outside), no boundary
  detection.  For overlay nesting, containment bypass, nonpolygon binary.
- ``POINT_IN_RING_BOUNDARY_DEVICE`` -- bool with boundary out-param.
  Requires ``POINT_ON_SEGMENT_DEVICE`` prepended before this string.
  For face labeling, validity, segment distance containment.
- ``POINT_IN_RING_KIND_DEVICE`` -- 3-way ``unsigned char``
  (0=outside, 1=boundary, 2=inside) for DE-9IM.
  Requires ``POINT_ON_SEGMENT_KIND_DEVICE`` prepended before this string.

All functions use the standard even-odd (ray-casting) rule with
``(start, end)`` coordinate-index convention where ``start`` is the first
coordinate index and ``end`` is one-past-the-last.

PREDICATE class, fp64 only.  No centering, no Kahan -- callers that need
precision dispatch handle centering at the call site and cast results.
"""

from __future__ import annotations

__all__ = [
    "POINT_IN_RING_DEVICE",
    "POINT_IN_RING_BOUNDARY_DEVICE",
    "POINT_IN_RING_KIND_DEVICE",
]

# ---------------------------------------------------------------------------
# Fast variant: inside/outside only (no boundary detection)
# ---------------------------------------------------------------------------
POINT_IN_RING_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Point-in-ring even-odd ray-cast (fast, no boundary detection)      */
/* Shared via vibespatial.cuda.device_functions.point_in_ring          */
/* ------------------------------------------------------------------ */

/* Test if point (px,py) is inside the ring defined by coordinates
   x[start..end-1], y[start..end-1] using the even-odd (ray-casting) rule.

   Returns true if the point is strictly inside (boundary is NOT detected).
   The ring should be closed (first == last coordinate).

   Parameters:
     px, py  -- query point coordinates
     x, y    -- coordinate arrays (SoA layout)
     start   -- first coordinate index of the ring
     end     -- one-past-the-last coordinate index
*/
__device__ inline bool vs_ring_contains_point(
    double px, double py,
    const double* __restrict__ x, const double* __restrict__ y,
    int start, int end
) {
    bool inside = false;
    for (int i = start, j = end - 1; i < end; j = i++) {
        double ay = y[j], by = y[i];
        if ((ay > py) != (by > py)) {
            double ax = x[j], bx = x[i];
            if (px < (bx - ax) * (py - ay) / (by - ay) + ax)
                inside = !inside;
        }
    }
    return inside;
}
"""

# ---------------------------------------------------------------------------
# Boundary variant: bool + on_boundary out-param
# Requires POINT_ON_SEGMENT_DEVICE to be prepended before this string.
# ---------------------------------------------------------------------------
POINT_IN_RING_BOUNDARY_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Point-in-ring even-odd ray-cast (with boundary detection)          */
/* Shared via vibespatial.cuda.device_functions.point_in_ring          */
/* Requires: vs_point_on_segment (from POINT_ON_SEGMENT_DEVICE)       */
/* ------------------------------------------------------------------ */

/* Test if point (px,py) is inside or on the boundary of the ring
   defined by coordinates x[start..end-1], y[start..end-1].

   Returns true if the point is inside OR on the boundary.
   Sets *on_boundary = true if the point lies on an edge.

   The ring should be closed (first == last coordinate).

   Parameters:
     px, py      -- query point coordinates
     x, y        -- coordinate arrays (SoA layout)
     start       -- first coordinate index of the ring
     end         -- one-past-the-last coordinate index
     tolerance   -- boundary tolerance for vs_point_on_segment
     on_boundary -- output: set to true if point is on ring edge
*/
__device__ inline bool vs_ring_contains_point_with_boundary(
    double px, double py,
    const double* __restrict__ x, const double* __restrict__ y,
    int start, int end,
    double tolerance,
    bool* on_boundary
) {
    *on_boundary = false;
    bool inside = false;
    for (int i = start, j = end - 1; i < end; j = i++) {
        double ax = x[j], ay = y[j], bx = x[i], by = y[i];
        if (vs_point_on_segment(px, py, ax, ay, bx, by, tolerance)) {
            *on_boundary = true;
            return true;
        }
        if ((ay > py) != (by > py)) {
            if (px < (bx - ax) * (py - ay) / (by - ay) + ax)
                inside = !inside;
        }
    }
    return inside;
}
"""

# ---------------------------------------------------------------------------
# 3-way kind variant: 0=outside, 1=boundary, 2=inside (for DE-9IM)
# Requires POINT_ON_SEGMENT_KIND_DEVICE to be prepended before this string.
# ---------------------------------------------------------------------------
POINT_IN_RING_KIND_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Point-in-ring 3-way classification (for DE-9IM)                    */
/* Shared via vibespatial.cuda.device_functions.point_in_ring          */
/* Requires: vs_point_on_segment_kind (from POINT_ON_SEGMENT_KIND_DEVICE) */
/* ------------------------------------------------------------------ */

/* Classify how point (px,py) relates to the ring defined by
   coordinates x[start..end-1], y[start..end-1].

   Returns:
     0 -- point is outside the ring
     1 -- point is on the boundary (on an edge)
     2 -- point is strictly inside the ring

   The ring should be closed (first == last coordinate).

   Parameters:
     px, py    -- query point coordinates
     x, y      -- coordinate arrays (SoA layout)
     start     -- first coordinate index of the ring
     end       -- one-past-the-last coordinate index
     tolerance -- boundary tolerance for vs_point_on_segment_kind
*/
__device__ inline unsigned char vs_ring_point_classify(
    double px, double py,
    const double* __restrict__ x, const double* __restrict__ y,
    int start, int end,
    double tolerance
) {
    bool inside = false;
    if ((end - start) < 2) {
        return 0;
    }
    for (int i = start, j = end - 1; i < end; j = i++) {
        double ax = x[j], ay = y[j], bx = x[i], by = y[i];
        const unsigned char seg_kind = vs_point_on_segment_kind(
            px, py, ax, ay, bx, by, tolerance);
        if (seg_kind != 0) {
            return 1;
        }
        if ((ay > py) != (by > py)) {
            if (px < (bx - ax) * (py - ay) / (by - ay) + ax)
                inside = !inside;
        }
    }
    return inside ? 2 : 0;
}
"""
