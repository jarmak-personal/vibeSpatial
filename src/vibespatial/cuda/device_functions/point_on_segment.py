"""Shared CUDA device function: point-on-segment boundary test."""

from __future__ import annotations

__all__ = ["POINT_ON_SEGMENT_DEVICE", "POINT_ON_SEGMENT_KIND_DEVICE"]

POINT_ON_SEGMENT_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Point-on-segment boundary test (boolean)                           */
/* Shared via vibespatial.cuda.device_functions.point_on_segment      */
/* ------------------------------------------------------------------ */

/* Test if point (px,py) lies on segment (ax,ay)-(bx,by) within tolerance.
   Returns true if the point is on the segment.

   The cross-product is scaled by the segment's Manhattan extent
   (|dx|+|dy|+1) so that the tolerance is dimensionless and does not
   depend on segment length.  The bounding-box check is padded by
   tolerance in each direction.

   Parameters:
     px, py    -- query point coordinates
     ax, ay    -- segment start coordinates
     bx, by    -- segment end coordinates
     tolerance -- absolute tolerance (e.g. 1e-7 or 1e-12)
*/
__device__ inline bool vs_point_on_segment(
    double px, double py,
    double ax, double ay, double bx, double by,
    double tolerance
) {{
    const double dx = bx - ax;
    const double dy = by - ay;
    const double cross = (px - ax) * dy - (py - ay) * dx;
    const double scale = fabs(dx) + fabs(dy) + 1.0;
    if (fabs(cross) > tolerance * scale) {{
        return false;
    }}
    const double minx = fmin(ax, bx) - tolerance;
    const double maxx = fmax(ax, bx) + tolerance;
    const double miny = fmin(ay, by) - tolerance;
    const double maxy = fmax(ay, by) + tolerance;
    return px >= minx && px <= maxx && py >= miny && py <= maxy;
}}
"""

POINT_ON_SEGMENT_KIND_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Point-on-segment 3-way classification (for DE-9IM)                 */
/* Shared via vibespatial.cuda.device_functions.point_on_segment      */
/* ------------------------------------------------------------------ */

/* Classify how point (px,py) relates to segment (ax,ay)-(bx,by).
   Returns:
     0 -- point is off the segment
     1 -- point is on an endpoint of the segment
     2 -- point is on the interior of the segment (not an endpoint)

   Uses the same cross-product + bbox algorithm as vs_point_on_segment.
   Endpoint proximity is tested via Manhattan distance <= tolerance.

   Parameters:
     px, py    -- query point coordinates
     ax, ay    -- segment start coordinates
     bx, by    -- segment end coordinates
     tolerance -- absolute tolerance (e.g. 1e-12)
*/
__device__ inline unsigned char vs_point_on_segment_kind(
    double px, double py,
    double ax, double ay, double bx, double by,
    double tolerance
) {{
    const double dx = bx - ax;
    const double dy = by - ay;
    const double cross = (px - ax) * dy - (py - ay) * dx;
    const double scale = fabs(dx) + fabs(dy) + 1.0;
    if (fabs(cross) > tolerance * scale) {{
        return 0;
    }}
    const double minx = fmin(ax, bx) - tolerance;
    const double maxx = fmax(ax, bx) + tolerance;
    const double miny = fmin(ay, by) - tolerance;
    const double maxy = fmax(ay, by) + tolerance;
    if (px < minx || px > maxx || py < miny || py > maxy) {{
        return 0;
    }}
    /* Distinguish endpoint vs interior using Manhattan distance. */
    const bool endpoint =
        (fabs(px - ax) <= tolerance && fabs(py - ay) <= tolerance) ||
        (fabs(px - bx) <= tolerance && fabs(py - by) <= tolerance);
    return endpoint ? (unsigned char)1 : (unsigned char)2;
}}
"""
