from __future__ import annotations

from vibespatial.constructive.multipoint_polygon_constructive import (
    multipoint_polygon_difference,
    multipoint_polygon_intersection,
)

from .clip_rect import clip_by_rect_kernel
from .make_valid import make_valid_kernel
from .nonpolygon_binary import (
    linestring_linestring_intersection,
    linestring_polygon_difference,
    linestring_polygon_intersection,
    point_linestring_difference,
    point_linestring_intersection,
    point_point_difference,
    point_point_intersection,
    point_point_symmetric_difference,
    point_point_union,
)
from .polygon_difference import polygon_difference
from .polygon_intersection import polygon_intersection
from .polygon_rect_intersection import polygon_rect_intersection
from .segmented_union import segmented_union_all
from .stroke import offset_curve_kernel, point_buffer_kernel

__all__ = [
    "clip_by_rect_kernel",
    "linestring_linestring_intersection",
    "linestring_polygon_difference",
    "linestring_polygon_intersection",
    "make_valid_kernel",
    "multipoint_polygon_difference",
    "multipoint_polygon_intersection",
    "offset_curve_kernel",
    "point_buffer_kernel",
    "point_linestring_difference",
    "point_linestring_intersection",
    "point_point_difference",
    "point_point_intersection",
    "point_point_symmetric_difference",
    "point_point_union",
    "polygon_difference",
    "polygon_intersection",
    "polygon_rect_intersection",
    "segmented_union_all",
]
