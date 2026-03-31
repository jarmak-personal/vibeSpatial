"""Shared CUDA device functions: segment crossing and collinear containment."""

from __future__ import annotations

__all__ = ["SEGMENT_CROSSING_DEVICE"]

SEGMENT_CROSSING_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Segment crossing helpers                                           */
/* Shared via vibespatial.cuda.device_functions.segment_crossing      */
/* ------------------------------------------------------------------ */

/* Test if point (px,py) lies on segment (ax,ay)-(bx,by) GIVEN that they
   are known collinear (orient2d returned 0).  No tolerance -- exact bbox
   containment check using double comparisons.

   Returns 1 if the point is within the segment's bounding box, 0 otherwise.

   Parameters:
     px, py  -- query point coordinates
     ax, ay  -- segment start coordinates
     bx, by  -- segment end coordinates
*/
__device__ inline int vs_point_on_segment_collinear(
    double px, double py,
    double ax, double ay,
    double bx, double by
) {{
    double minx = ax < bx ? ax : bx;
    double maxx = ax > bx ? ax : bx;
    double miny = ay < by ? ay : by;
    double maxy = ay > by ? ay : by;
    return (px >= minx && px <= maxx && py >= miny && py <= maxy) ? 1 : 0;
}}

/* Test if segments (p1,p2) and (q1,q2) properly cross.
   "Properly cross" means the crossing point is interior to both segments
   (not at an endpoint).

   Requires ORIENT2D_DEVICE to be prepended (provides vs_orient2d).

   Parameters:
     p1x,p1y - p2x,p2y  -- first segment endpoints
     q1x,q1y - q2x,q2y  -- second segment endpoints
*/
__device__ inline bool vs_segments_properly_cross(
    double p1x, double p1y, double p2x, double p2y,
    double q1x, double q1y, double q2x, double q2y
) {{
    int d1 = vs_orient2d(q1x, q1y, q2x, q2y, p1x, p1y);
    int d2 = vs_orient2d(q1x, q1y, q2x, q2y, p2x, p2y);
    int d3 = vs_orient2d(p1x, p1y, p2x, p2y, q1x, q1y);
    int d4 = vs_orient2d(p1x, p1y, p2x, p2y, q2x, q2y);
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0))
        && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
}}
"""
