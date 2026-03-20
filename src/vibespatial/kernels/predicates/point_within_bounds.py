from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds
from vibespatial.predicates.support import (
    PointSequence,
    coerce_geometry_array,
    extract_empty_rows,
    extract_point_coordinates,
    resolve_predicate_context,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode

BoundsValue = tuple[float, float, float, float] | list[float] | np.ndarray
BoundsSequence = Sequence[BoundsValue | None]


@dataclass(frozen=True)
class NormalizedBoundsInput:
    bounds: np.ndarray
    null_mask: np.ndarray
    empty_mask: np.ndarray
    geometry_array: OwnedGeometryArray | None = None


def _normalize_bounds_tuple(value: BoundsValue) -> tuple[float, float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,):
        raise ValueError("bounds values must be 4-tuples: (minx, miny, maxx, maxy)")
    return tuple(float(item) for item in array)


def _normalize_right_input(
    values: BoundsSequence | PointSequence,
    *,
    expected_len: int,
) -> NormalizedBoundsInput:
    if isinstance(values, OwnedGeometryArray):
        array = coerce_geometry_array(
            values,
            arg_name="polygons_or_bounds",
            expected_families=(GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON),
        )
        if array.row_count != expected_len:
            raise ValueError(
                "point_within_bounds requires aligned inputs; "
                f"got {expected_len} points and {array.row_count} bounds rows"
            )
        return NormalizedBoundsInput(
            bounds=compute_geometry_bounds(array),
            null_mask=~array.validity,
            empty_mask=extract_empty_rows(array),
            geometry_array=array,
        )

    if not isinstance(values, Sequence):
        raise TypeError("polygons_or_bounds must be an OwnedGeometryArray or a sequence")
    if len(values) != expected_len:
        raise ValueError(
            f"point_within_bounds requires aligned inputs; got {expected_len} points and {len(values)} rows"
        )

    non_null = [value for value in values if value is not None]
    if non_null and hasattr(non_null[0], "geom_type"):
        array = coerce_geometry_array(
            values,
            arg_name="polygons_or_bounds",
            expected_families=(GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON),
        )
        return NormalizedBoundsInput(
            bounds=compute_geometry_bounds(array),
            null_mask=~array.validity,
            empty_mask=extract_empty_rows(array),
            geometry_array=array,
        )

    bounds = np.full((expected_len, 4), np.nan, dtype=np.float64)
    null_mask = np.zeros(expected_len, dtype=bool)
    empty_mask = np.zeros(expected_len, dtype=bool)
    for index, value in enumerate(values):
        if value is None:
            null_mask[index] = True
            continue
        bounds[index] = _normalize_bounds_tuple(value)
    return NormalizedBoundsInput(bounds=bounds, null_mask=null_mask, empty_mask=empty_mask)


def _evaluate_point_within_bounds(
    points: OwnedGeometryArray,
    right: NormalizedBoundsInput,
) -> np.ndarray:
    point_coords, point_empty = extract_point_coordinates(points)
    result = np.zeros(points.row_count, dtype=object)
    result[:] = False
    null_mask = ~points.validity | right.null_mask
    result[null_mask] = None

    value_mask = ~(null_mask | point_empty | right.empty_mask)
    value_mask &= ~np.isnan(point_coords).any(axis=1)
    value_mask &= ~np.isnan(right.bounds).any(axis=1)
    if not value_mask.any():
        return result

    x = point_coords[:, 0]
    y = point_coords[:, 1]
    bounds = right.bounds
    inside = (
        (bounds[:, 0] <= x)
        & (x <= bounds[:, 2])
        & (bounds[:, 1] <= y)
        & (y <= bounds[:, 3])
    )
    result[value_mask & inside] = True
    return result


@register_kernel_variant(
    "point_within_bounds",
    "cpu",
    kernel_class=KernelClass.PREDICATE,
    geometry_families=("point", "polygon", "multipolygon"),
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=False,
    tags=("coarse-filter", "bounds"),
)
def point_within_bounds(
    points: PointSequence,
    polygons_or_bounds: BoundsSequence | PointSequence,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> list[bool | None]:
    left = coerce_geometry_array(
        points,
        arg_name="points",
        expected_families=(GeometryFamily.POINT,),
    )
    right = _normalize_right_input(polygons_or_bounds, expected_len=left.row_count)
    right_array = right.geometry_array if right.geometry_array is not None else left
    resolve_predicate_context(
        kernel_name="point_within_bounds",
        left=left,
        right=right_array,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
    result = _evaluate_point_within_bounds(left, right)
    return [None if value is None else bool(value) for value in result]
