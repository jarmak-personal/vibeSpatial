"""Public GeoPandas overlay API, including keep_geom_type dispatch semantics."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import GeometryCollection

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
    cp = None

from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._compat import PANDAS_GE_30
from vibespatial.api._native_grouped import NativeGrouped, NativeGroupedSelection
from vibespatial.api._native_relation import NativeRelation, NativeRelationSelection
from vibespatial.api._native_result_core import NativeTabularSelection
from vibespatial.api._native_results import (
    GeometryNativeResult,
    RelationIndexResult,
    _concat_native_tabular_results,
    _left_constructive_capacity_to_native_tabular_result,
    _left_constructive_result_to_native_tabular_result,  # noqa: F401
    _left_constructive_to_native_tabular_result,
    _pairwise_constructive_result_to_native_tabular_result,  # noqa: F401
    _pairwise_constructive_to_native_tabular_result,
    _relation_join_source_state,
    _relation_selection_constructive_to_native_tabular_result,
    _rename_native_tabular_result,
    _symmetric_difference_native_tabular_results,
)
from vibespatial.api._native_rowset import NativeDeviceSelection
from vibespatial.api.geometry_array import (
    LINE_GEOM_TYPES,
    POINT_GEOM_TYPES,
    POLYGON_GEOM_TYPES,
    GeometryArray,
    _check_crs,
    _crs_mismatch_warn,
)
from vibespatial.api.tools._pair_cache import get_cached_intersection_pairs
from vibespatial.constructive.boundary_remnants import (
    polygon_pair_boundary_remnants_capacity_device,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.overlay._host_boundary import (
    overlay_bool_scalar as _overlay_bool_scalar,
)
from vibespatial.overlay._host_boundary import (
    overlay_device_to_host as _overlay_device_to_host,
)
from vibespatial.overlay._host_boundary import (
    overlay_int_scalar as _overlay_int_scalar,
)
from vibespatial.overlay.strategies import plan_overlay_operation
from vibespatial.runtime._runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.config import (
    SPATIAL_EPSILON,
)
from vibespatial.runtime.crossover import (
    PhysicalWorkEstimate,
    WorkloadShape,
    estimate_physical_work_from_owned,
    estimate_relation_pair_work_from_owned,
)
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event, strict_native_mode_enabled
from vibespatial.runtime.hotpath_trace import (
    attach_work_amplification,
    hotpath_stage,
    hotpath_timing_enabled,
)
from vibespatial.runtime.precision import KernelClass
from vibespatial.spatial.indexing import generate_bounds_pairs
from vibespatial.spatial.query_types import DeviceSpatialJoinResult

logger = logging.getLogger(__name__)

_OVERLAY_BBOX_PAIR_FAST_PATH_MAX_PAIRS = 262_144
_OVERLAY_FEW_RIGHT_GROUP_MAX = 64
_OVERLAY_FEW_RIGHT_GROUP_MIN_AVG = 8.0
_OVERLAY_SEGMENT_TABLE_BYTES_PER_SEGMENT = 48
_OVERLAY_HOST_EXACT_PAIR_BATCH_MAX_WORK_UNITS = 1_280
_OVERLAY_EXACT_POLYGON_GPU_BOUNDARY_MAX_WORK_UNITS = 2_000_000
_GROUPED_DIFFERENCE_CONTAINMENT_MAX_RIGHT_SEGMENTS_PER_ROW = 4_096
_SHAPELY_TYPE_ID_POLYGON = 3
_SHAPELY_TYPE_ID_MULTIPOLYGON = 6
_SHAPELY_TYPE_ID_GEOMETRYCOLLECTION = 7
# GPU overlay can emit sub-nanometer-thickness polygons for projected
# boundary-touch cases that GEOS classifies as lower-dimensional.  Keep this
# relative to the smaller source polygon so legitimate small overlays survive.
_GROUPED_RECTANGLE_HOLE_DIFF_MAX_GROUP_SIZE = 32


_GROUPED_RECTANGLE_HOLE_DIFF_KERNEL_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4) validate_grouped_polygon_holes(
    const unsigned char* __restrict__ row_supported,
    const double* __restrict__ right_bounds,
    const long long* __restrict__ group_offsets,
    int group_count,
    int max_group_size,
    unsigned char* __restrict__ supported
) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= group_count) {
        return;
    }
    supported[group] = 0;
    const long long start = group_offsets[group];
    const long long end = group_offsets[group + 1];
    const int count = (int)(end - start);
    if (count <= 0 || count > max_group_size || count > 32) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        const long long row_i = start + i;
        if (!row_supported[row_i]) {
            return;
        }
        const double* a = right_bounds + row_i * 4;
        const double ax0 = a[0];
        const double ay0 = a[1];
        const double ax1 = a[2];
        const double ay1 = a[3];
        if (!(ax1 > ax0 && ay1 > ay0)) {
            return;
        }
        for (int j = i + 1; j < count; ++j) {
            const long long row_j = start + j;
            if (!row_supported[row_j]) {
                return;
            }
            const double* b = right_bounds + row_j * 4;
            const bool separated =
                ax1 < b[0] || b[2] < ax0 || ay1 < b[1] || b[3] < ay0;
            if (!separated) {
                return;
            }
        }
    }
    supported[group] = 1;
}

extern "C" __global__ void __launch_bounds__(256, 4) validate_grouped_rectangle_holes(
    const double* __restrict__ left_bounds,
    const double* __restrict__ right_bounds,
    const long long* __restrict__ group_offsets,
    int group_count,
    int max_group_size,
    unsigned char* __restrict__ supported
) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= group_count) {
        return;
    }
    supported[group] = 0;
    const long long start = group_offsets[group];
    const long long end = group_offsets[group + 1];
    const int count = (int)(end - start);
    if (count <= 0 || count > max_group_size || count > 32) {
        return;
    }
    const double* left = left_bounds + group * 4;
    const double lx0 = left[0];
    const double ly0 = left[1];
    const double lx1 = left[2];
    const double ly1 = left[3];
    if (!(lx1 > lx0 && ly1 > ly0)) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        const double* a = right_bounds + (start + i) * 4;
        const double ax0 = a[0];
        const double ay0 = a[1];
        const double ax1 = a[2];
        const double ay1 = a[3];
        if (!(ax1 > ax0 && ay1 > ay0)) {
            return;
        }
        if (!(ax0 > lx0 && ay0 > ly0 && ax1 < lx1 && ay1 < ly1)) {
            return;
        }
        for (int j = i + 1; j < count; ++j) {
            const double* b = right_bounds + (start + j) * 4;
            const bool separated =
                ax1 < b[0] || b[2] < ax0 || ay1 < b[1] || b[3] < ay0;
            if (!separated) {
                return;
            }
        }
    }
    supported[group] = 1;
}

extern "C" __global__ void __launch_bounds__(256, 4) emit_grouped_rectangle_holes(
    const double* __restrict__ left_bounds,
    const double* __restrict__ right_bounds,
    const long long* __restrict__ group_offsets,
    const int* __restrict__ geometry_offsets,
    int group_count,
    long long total_coords,
    double* __restrict__ out_x,
    double* __restrict__ out_y
) {
    const long long pos = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total_coords) {
        return;
    }
    const int ring = (int)(pos / 5);
    const int local = (int)(pos - ((long long)ring * 5));

    int lo = 0;
    int hi = group_count;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (geometry_offsets[mid + 1] <= ring) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int group = lo;
    const int group_ring_start = geometry_offsets[group];
    const bool exterior = ring == group_ring_start;
    const double* bounds = nullptr;
    if (exterior) {
        bounds = left_bounds + group * 4;
    } else {
        const int hole_local = ring - group_ring_start - 1;
        bounds = right_bounds + (group_offsets[group] + hole_local) * 4;
    }

    const double x0 = bounds[0];
    const double y0 = bounds[1];
    const double x1 = bounds[2];
    const double y1 = bounds[3];

    if (exterior) {
        if (local == 0 || local == 4) {
            out_x[pos] = x0;
            out_y[pos] = y0;
        } else if (local == 1) {
            out_x[pos] = x0;
            out_y[pos] = y1;
        } else if (local == 2) {
            out_x[pos] = x1;
            out_y[pos] = y1;
        } else {
            out_x[pos] = x1;
            out_y[pos] = y0;
        }
    } else {
        if (local == 0 || local == 4) {
            out_x[pos] = x1;
            out_y[pos] = y0;
        } else if (local == 1) {
            out_x[pos] = x1;
            out_y[pos] = y1;
        } else if (local == 2) {
            out_x[pos] = x0;
            out_y[pos] = y1;
        } else {
            out_x[pos] = x0;
            out_y[pos] = y0;
        }
    }
}
"""


class _GroupedOverlayDifferenceNativeDeclined(RuntimeError):
    """Raised when grouped difference cannot stay on native grouped carriers."""


@dataclass(frozen=True)
class _GroupedDifferenceCapacityPartition:
    """Row-capacity direct result and its device-resident ownership mask."""

    owned: Any
    support_mask: Any
    collective_mask: Any | None = None


class _OverlayNativeConstructiveDeclined(RuntimeError):
    """Raised when an owned overlay constructive path cannot stay native."""


def _series_polygon_mask(series: GeoSeries) -> np.ndarray:
    """Return a polygon-or-multipolygon membership mask for a geometry series."""
    owned = _series_owned(series)
    if owned is not None:
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import FAMILY_TAGS

        state = getattr(owned, "device_state", None)
        if state is not None and state.trusted_polygonal_only is True:
            if state.trusted_all_valid is True:
                return np.ones(int(owned.row_count), dtype=bool)
            host_validity = getattr(owned, "_validity", None)
            if host_validity is not None and int(host_validity.size) == int(owned.row_count):
                return np.asarray(host_validity, dtype=bool)
            if has_gpu_runtime():
                import cupy as cp

                return _overlay_device_to_host(
                    cp.asarray(
                        owned._ensure_device_state(preserve_indexed_view=True).validity,
                        dtype=cp.bool_,
                    ),
                    reason="overlay polygonal logical validity mask terminal export",
                    dtype=bool,
                )

        tags = np.asarray(owned.tags)
        return np.asarray(owned.validity, dtype=bool) & (
            (tags == FAMILY_TAGS[GeometryFamily.POLYGON])
            | (tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
        )
    composition = _series_native_composition(series)
    if composition is not None:
        return _composition_polygon_mask(composition)
    return np.asarray(series.geom_type.isin(POLYGON_GEOM_TYPES), dtype=bool)


def _owned_has_logical_polygon_rows(owned) -> bool:
    """Return whether the logical owned carrier can contain polygon rows."""
    state = getattr(owned, "device_state", None)
    if state is not None and state.trusted_polygonal_only is True:
        return int(owned.row_count) > 0
    return bool(_owned_logical_family_flags(owned)[1])


def _mark_owned_logical_polygon_valid_nonempty(
    owned,
    *,
    all_rows_valid: bool = True,
) -> None:
    """Record logical-domain proofs after keep-geom-type keeps positive polygons."""
    if owned is None or getattr(owned, "device_state", None) is None:
        return
    state = owned._ensure_device_state(preserve_indexed_view=True)
    state.trusted_polygonal_only = True
    state.trusted_all_valid = all_rows_valid
    state.trusted_all_non_empty = True
    state.trusted_nonempty_polygonal_positive_area = True
    from vibespatial.geometry.buffers import GeometryFamily

    state.trusted_family_domain = (
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    )


def _series_owned(series: GeoSeries):
    """Return only already-owned backing; never physicalize a composition."""
    from vibespatial.api._native_state import get_native_state

    native_state = get_native_state(series)
    if native_state is not None:
        geometry = getattr(native_state, "geometry", None)
        cached_owned = getattr(geometry, "cached_owned", None)
        owned = (
            cached_owned()
            if callable(cached_owned)
            else geometry.owned
            if isinstance(geometry, GeometryNativeResult)
            else None
        )
        if owned is not None:
            return owned
    values = series.values
    cached_owned = getattr(values, "cached_owned", None)
    if callable(cached_owned):
        return cached_owned()
    if isinstance(values, GeometryArray):
        return values._owned
    return None


def _series_native_composition(series: GeoSeries):
    """Return native geometry partitions without asking for contiguous storage."""
    from vibespatial.api._native_state import get_native_state

    native_state = get_native_state(series)
    if native_state is not None:
        composition = getattr(getattr(native_state, "geometry", None), "composition", None)
        if composition is not None:
            return composition
    return getattr(series.values, "native_composition", None)


def _composition_polygon_mask(composition) -> np.ndarray:
    """Classify polygon rows from concrete composition parts without joining them."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS
    from vibespatial.runtime.residency import Residency

    polygon_tag = np.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    multipolygon_tag = np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
    if composition.residency is Residency.DEVICE:
        import cupy as cp

        d_result = cp.zeros(int(composition.row_count), dtype=cp.uint32)
        for part in composition.parts:
            owned = part.geometry.cached_owned()
            if owned is None:
                continue
            state = owned._ensure_device_state(preserve_indexed_view=True)
            d_tags = cp.asarray(state.tags, dtype=cp.int8)
            d_validity = cp.asarray(state.validity, dtype=cp.bool_)
            d_part_polygon = d_validity & (
                (d_tags == polygon_tag) | (d_tags == multipolygon_tag)
            )
            d_rows = cp.asarray(part.output_rows, dtype=cp.int64)
            cp.maximum.at(
                d_result,
                d_rows,
                d_part_polygon.astype(cp.uint32, copy=False),
            )
        return _overlay_device_to_host(
            d_result,
            reason="overlay native composition polygon-row inspection boundary",
            dtype=bool,
        )

    result = np.zeros(int(composition.row_count), dtype=bool)
    for part in composition.parts:
        owned = part.geometry.cached_owned()
        if owned is None:
            continue
        tags = np.asarray(owned.tags, dtype=np.int8)
        validity = np.asarray(owned.validity, dtype=bool)
        part_polygon = validity & (
            (tags == polygon_tag) | (tags == multipolygon_tag)
        )
        np.logical_or.at(
            result,
            np.asarray(part.output_rows, dtype=np.int64),
            part_polygon,
        )
    return result


def _owned_logical_family_flags(
    owned,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Return family flags for the logical row carrier of an owned array."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    polygon_domain = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
    polygon_families = set(polygon_domain)
    line_families = {
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTILINESTRING,
    }
    point_families = {GeometryFamily.POINT, GeometryFamily.MULTIPOINT}

    state = getattr(owned, "device_state", None)
    if state is not None and state.trusted_polygonal_only is True:
        row_count = int(owned.row_count)
        if state.trusted_all_valid is True:
            present = row_count > 0
            return (
                True,
                present,
                False,
                False,
                present,
                present,
            )
        host_validity = getattr(owned, "_validity", None)
        if host_validity is not None and int(host_validity.size) == row_count:
            validity = np.asarray(host_validity, dtype=bool)
            present = bool(validity.any())
            return (
                bool(validity.all()),
                present,
                False,
                False,
                present,
                present,
            )

    if state is not None and state.trusted_family_domain is not None:
        family_domain = set(state.trusted_family_domain)
        row_count = int(owned.row_count)
        if state.trusted_all_valid is True:
            present = row_count > 0
            if family_domain and family_domain <= polygon_families:
                return True, present, False, False, present, present
            if family_domain and family_domain <= line_families:
                return False, False, present, False, False, present
            if family_domain and family_domain <= point_families:
                return False, False, False, present, False, present

    polygon_tags = tuple(np.int8(FAMILY_TAGS[family]) for family in polygon_families)
    line_tags = tuple(np.int8(FAMILY_TAGS[family]) for family in line_families)
    point_tags = tuple(np.int8(FAMILY_TAGS[family]) for family in point_families)
    host_tags = getattr(owned, "_tags", None)
    host_validity = getattr(owned, "_validity", None)
    if host_tags is not None and host_validity is not None:
        tags = np.asarray(host_tags, dtype=np.int8)
        validity = np.asarray(host_validity, dtype=bool)
        polygon_mask = validity & np.isin(tags, polygon_tags)
        line_mask = validity & np.isin(tags, line_tags)
        point_mask = validity & np.isin(tags, point_tags)
        present = bool(validity.any())
        nonmissing_polygon = (~validity) | np.isin(tags, polygon_tags)
        if (
            present
            and bool(nonmissing_polygon.all())
            and getattr(owned, "device_state", None) is not None
        ):
            owned.device_state.trusted_polygonal_only = True
            owned.device_state.trusted_family_domain = polygon_domain
        return (
            bool(polygon_mask.all()),
            bool(polygon_mask.any()),
            bool(line_mask.any()),
            bool(point_mask.any()),
            present and bool(nonmissing_polygon.all()),
            present,
        )

    if getattr(owned, "device_state", None) is not None:
        import cupy as cp

        state = owned._ensure_device_state(preserve_indexed_view=True)
        d_tags = cp.asarray(state.tags, dtype=cp.int8)
        d_validity = cp.asarray(state.validity, dtype=cp.bool_)
        d_polygon = ((d_tags == polygon_tags[0]) | (d_tags == polygon_tags[1])) & d_validity
        d_line = ((d_tags == line_tags[0]) | (d_tags == line_tags[1])) & d_validity
        d_point = ((d_tags == point_tags[0]) | (d_tags == point_tags[1])) & d_validity
        d_nonmissing_polygon = (~d_validity) | d_polygon
        d_flags = cp.empty(6, dtype=cp.bool_)
        d_flags[0] = cp.all(d_polygon)
        d_flags[1] = cp.any(d_polygon)
        d_flags[2] = cp.any(d_line)
        d_flags[3] = cp.any(d_point)
        d_flags[4] = cp.all(d_nonmissing_polygon) & cp.any(d_validity)
        d_flags[5] = cp.any(d_validity)
        flags = _overlay_device_to_host(
            d_flags,
            reason="overlay source geometry family-domain scalar fence",
            dtype=bool,
        )
        if bool(flags[4]):
            state.trusted_polygonal_only = True
            state.trusted_family_domain = polygon_domain
        return tuple(bool(flag) for flag in flags)  # type: ignore[return-value]

    tags = np.asarray(owned.tags, dtype=np.int8)
    validity = np.asarray(owned.validity, dtype=bool)
    polygon_mask = validity & np.isin(tags, polygon_tags)
    line_mask = validity & np.isin(tags, line_tags)
    point_mask = validity & np.isin(tags, point_tags)
    present = bool(validity.any())
    nonmissing_polygon = (~validity) | np.isin(tags, polygon_tags)
    if (
        present
        and bool(nonmissing_polygon.all())
        and getattr(owned, "device_state", None) is not None
    ):
        owned.device_state.trusted_polygonal_only = True
        owned.device_state.trusted_family_domain = polygon_domain
    return (
        bool(polygon_mask.all()),
        bool(polygon_mask.any()),
        bool(line_mask.any()),
        bool(point_mask.any()),
        present and bool(nonmissing_polygon.all()),
        present,
    )


def _composition_logical_family_flags(composition) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Reduce concrete part families into logical-row routing flags."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS
    from vibespatial.runtime.residency import Residency

    family_bits = {
        FAMILY_TAGS[GeometryFamily.POLYGON]: 1,
        FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]: 1,
        FAMILY_TAGS[GeometryFamily.LINESTRING]: 2,
        FAMILY_TAGS[GeometryFamily.MULTILINESTRING]: 2,
        FAMILY_TAGS[GeometryFamily.POINT]: 4,
        FAMILY_TAGS[GeometryFamily.MULTIPOINT]: 4,
    }
    use_device = composition.residency is Residency.DEVICE or any(
        hasattr(part.output_rows, "__cuda_array_interface__")
        for part in composition.parts
    )
    if use_device:
        import cupy as xp
    else:
        xp = np

    row_count = int(composition.row_count)
    row_families = xp.zeros(row_count, dtype=xp.uint32)
    for part in composition.parts:
        owned = part.geometry.cached_owned()
        if owned is None:
            continue
        state = owned._ensure_device_state(preserve_indexed_view=True) if use_device else None
        tags = xp.asarray(state.tags if state is not None else owned.tags, dtype=xp.int8)
        validity = xp.asarray(
            state.validity if state is not None else owned.validity,
            dtype=xp.bool_,
        )
        bits = xp.zeros(int(tags.size), dtype=xp.uint32)
        for family_tag, family_bit in family_bits.items():
            bits = xp.where(tags == np.int8(family_tag), xp.uint32(family_bit), bits)
        bits = xp.where(validity, bits, xp.uint32(0))
        xp.bitwise_or.at(
            row_families,
            xp.asarray(part.output_rows, dtype=xp.int64),
            bits,
        )

    present = row_families != 0
    polygon_only = row_families == xp.uint32(1)
    flags = xp.stack(
        (
            xp.any(present) & xp.all(polygon_only),
            xp.any((row_families & xp.uint32(1)) != 0),
            xp.any((row_families & xp.uint32(2)) != 0),
            xp.any((row_families & xp.uint32(4)) != 0),
            xp.any(present) & xp.all((~present) | polygon_only),
            xp.any(present),
        )
    ).astype(xp.bool_, copy=False)
    if use_device:
        flags = _overlay_device_to_host(
            flags,
            reason="overlay native composition family-domain scalar fence",
            dtype=bool,
        )
    return tuple(bool(flag) for flag in flags)  # type: ignore[return-value]


def _series_family_summary(series: GeoSeries) -> tuple[bool, bool, bool, bool]:
    """Return all/any polygon, lineal, and point family flags.

    Public ``GeoDataFrame.geom_type`` is a terminal export for native-backed
    frames.  Overlay only needs private routing flags here, so derive them from
    owned family cardinalities when available and leave public type strings to
    true user-facing exports.
    """
    owned = _series_owned(series)
    if owned is not None:
        return _owned_logical_family_flags(owned)[:4]
    composition = _series_native_composition(series)
    if composition is not None:
        return _composition_logical_family_flags(composition)[:4]

    geom_types = series.geom_type
    polygon_mask = geom_types.isin(POLYGON_GEOM_TYPES)
    return (
        bool(polygon_mask.all()),
        bool(polygon_mask.any()),
        bool(geom_types.isin(LINE_GEOM_TYPES).any()),
        bool(geom_types.isin(POINT_GEOM_TYPES).any()),
    )


def _series_non_missing_all_polygons(series: GeoSeries) -> tuple[bool, bool]:
    """Return (all non-null rows are polygonal, any non-null row exists)."""
    owned = _series_owned(series)
    if owned is not None:
        flags = _owned_logical_family_flags(owned)
        return flags[4], flags[5]
    composition = _series_native_composition(series)
    if composition is not None:
        flags = _composition_logical_family_flags(composition)
        return flags[4], flags[5]

    geom_types = series.geom_type.dropna()
    if geom_types.empty:
        return False, False
    return bool(geom_types.isin(POLYGON_GEOM_TYPES).all()), True


def _series_first_geom_type(series: GeoSeries) -> str | None:
    """Return the first geometry type without public Series export when owned."""
    owned = _series_owned(series)
    if owned is not None:
        if int(owned.row_count) == 0:
            return None
        state = getattr(owned, "device_state", None)
        if state is not None:
            homogeneous = state.trusted_homogeneous_family
            if homogeneous is not None:
                return {
                    "point": "Point",
                    "linestring": "LineString",
                    "polygon": "Polygon",
                    "multipoint": "MultiPoint",
                    "multilinestring": "MultiLineString",
                    "multipolygon": "MultiPolygon",
                }.get(homogeneous.value)
            if state.trusted_polygonal_only is True:
                # Overlay keep_geom_type only needs the source geometry
                # family. Polygon and MultiPolygon share the same filter set.
                return "Polygon"
            family_domain = state.trusted_family_domain
            if family_domain:
                from vibespatial.geometry.buffers import GeometryFamily

                domain = set(family_domain)
                if domain <= {
                    GeometryFamily.POLYGON,
                    GeometryFamily.MULTIPOLYGON,
                }:
                    return "Polygon"
                if domain <= {
                    GeometryFamily.LINESTRING,
                    GeometryFamily.MULTILINESTRING,
                }:
                    return "LineString"
                if domain <= {
                    GeometryFamily.POINT,
                    GeometryFamily.MULTIPOINT,
                }:
                    return "Point"
        from vibespatial.geometry.owned import TAG_FAMILIES

        if getattr(owned, "_tags", None) is not None:
            tag = int(owned._tags[0])
        elif getattr(owned, "device_state", None) is not None:
            import cupy as cp

            tag = int(
                _overlay_device_to_host(
                    cp.asarray(owned.device_state.tags)[:1],
                    reason="overlay source geometry type scalar fence",
                )[0]
            )
        else:
            tag = int(owned.tags[0])
        family = TAG_FAMILIES.get(tag)
        if family is None:
            return None
        return {
            "point": "Point",
            "linestring": "LineString",
            "polygon": "Polygon",
            "multipoint": "MultiPoint",
            "multilinestring": "MultiLineString",
            "multipolygon": "MultiPolygon",
        }.get(family.value)

    composition = _series_native_composition(series)
    if composition is not None:
        family_domain = composition.trusted_family_domain
        if family_domain:
            from vibespatial.geometry.buffers import GeometryFamily

            domain = set(family_domain)
            if domain <= {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}:
                return "Polygon"
            if domain <= {GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING}:
                return "LineString"
            if domain <= {GeometryFamily.POINT, GeometryFamily.MULTIPOINT}:
                return "Point"

    return series.geom_type.iloc[0]


def _series_total_bounds_private(series: GeoSeries) -> np.ndarray:
    """Return total bounds for internal overlay routing without public export."""
    return np.asarray(series.values.total_bounds, dtype=np.float64)


def _series_prefers_device_bounds_private(series: GeoSeries) -> bool:
    owned = _series_owned(series)
    composition = _series_native_composition(series)
    from vibespatial.runtime.residency import Residency

    if owned is None:
        return bool(
            composition is not None and composition.residency is Residency.DEVICE
        )
    return bool(
        owned.residency is Residency.DEVICE or getattr(owned, "device_state", None) is not None
    )


def _polygonal_collection_input_values(series: GeoSeries) -> np.ndarray | None:
    """Return polygon-only values when GeometryCollections only add lower-dimensional parts."""
    values = np.asarray(series.values, dtype=object)
    if values.size == 0:
        return None
    type_ids = shapely.get_type_id(values)
    collection_mask = type_ids == _SHAPELY_TYPE_ID_GEOMETRYCOLLECTION
    if not collection_mask.any():
        return None

    polygon_only = _strip_non_polygon_collection_parts(values)
    collection_values = polygon_only[collection_mask]
    collection_type_ids = shapely.get_type_id(collection_values)
    collection_has_polygonal_part = (
        (~shapely.is_missing(collection_values))
        & (~shapely.is_empty(collection_values))
        & (
            (collection_type_ids == _SHAPELY_TYPE_ID_POLYGON)
            | (collection_type_ids == _SHAPELY_TYPE_ID_MULTIPOLYGON)
        )
    )
    if not bool(np.all(collection_has_polygonal_part)):
        return None

    polygon_type_ids = shapely.get_type_id(polygon_only)
    missing_mask = shapely.is_missing(polygon_only)
    empty_mask = shapely.is_empty(polygon_only)
    polygon_like = (
        missing_mask
        | empty_mask
        | (polygon_type_ids == _SHAPELY_TYPE_ID_POLYGON)
        | (polygon_type_ids == _SHAPELY_TYPE_ID_MULTIPOLYGON)
    )
    if not bool(np.all(polygon_like)):
        return None
    return polygon_only


def _normalize_polygonal_collection_input(
    df: GeoDataFrame,
) -> tuple[GeoDataFrame, bool]:
    """Strip lower-dimensional remnants from polygonal GeometryCollection inputs."""
    polygon_only = _polygonal_collection_input_values(df.geometry)
    if polygon_only is None:
        return df, False

    from vibespatial.geometry.device_array import DeviceGeometryArray
    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.runtime.residency import Residency

    source_owned = getattr(df.geometry.values, "_owned", None)
    target_residency = source_owned.residency if source_owned is not None else Residency.HOST
    owned = from_shapely_geometries(
        list(polygon_only),
        residency=target_residency,
    )
    if target_residency is Residency.DEVICE:
        values = DeviceGeometryArray._from_owned(owned, crs=df.crs)
    else:
        values = GeometryArray.from_owned(owned, crs=df.crs)

    normalized = df.copy()
    geom_name = normalized._geometry_column_name
    normalized[geom_name] = GeoSeries(values, index=normalized.index, crs=df.crs)
    return normalized, True


def _series_all_polygons(series: GeoSeries) -> bool:
    owned = _series_owned(series)
    if owned is not None:
        return bool(_owned_logical_family_flags(owned)[0])
    composition = _series_native_composition(series)
    if composition is not None:
        return bool(_composition_logical_family_flags(composition)[0])
    return bool(_series_polygon_mask(series).all())


def _is_device_array(value) -> bool:
    return hasattr(value, "__cuda_array_interface__")


def _device_take_preserving_indexed_rows(owned, rows):
    """Take logical rows without forcing indexed carriers to physicalize."""
    if cp is not None and getattr(owned, "is_indexed_view", False):
        return owned._device_indexed_take(cp.asarray(rows, dtype=cp.int64))
    return owned.device_take(rows)


def _device_take_relation_rows(owned, rows):
    """Represent relation-aligned geometry rows without copying coordinates."""
    if cp is None:
        raise RuntimeError("CuPy is required for device relation row flow")
    return owned._device_indexed_take(cp.asarray(rows, dtype=cp.int64))


def _overlay_host_bool_mask_sparse_first(
    mask,
    *,
    length: int,
    dense_reason: str,
    sparse_reason: str,
) -> np.ndarray | None:
    """Copy a device bool mask as sparse rows when that is the smaller boundary."""
    if mask is None:
        return None
    length = int(length)
    if _is_device_array(mask) and has_gpu_runtime():
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            d_mask = cp.asarray(mask, dtype=cp.bool_).reshape(-1)
            if int(d_mask.size) != length:
                return None
            d_rows = cp.flatnonzero(d_mask).astype(cp.int64, copy=False)
            true_count = int(d_rows.size)
            if true_count == 0:
                return np.zeros(length, dtype=bool)
            if true_count * np.dtype(np.int64).itemsize < length:
                rows = _overlay_device_to_host(
                    d_rows,
                    reason=sparse_reason,
                    dtype=np.intp,
                )
                host_mask = np.zeros(length, dtype=bool)
                host_mask[rows] = True
                return host_mask
    host_mask = _overlay_device_to_host(mask, reason=dense_reason, dtype=bool)
    if host_mask.size != length:
        return None
    return host_mask


def _array_length(value) -> int:
    size = getattr(value, "size", None)
    if size is not None:
        return int(size)
    return len(value)


def _overlay_pair_work_estimate(
    left_owned,
    right_owned,
    *,
    pair_count: int,
    workload_shape: WorkloadShape | None = None,
) -> PhysicalWorkEstimate:
    """Estimate overlay pair work from coordinates/segments and pair rows."""
    pair_count = int(pair_count)
    left_estimate = (
        estimate_physical_work_from_owned(left_owned)
        if left_owned is not None
        else PhysicalWorkEstimate.from_rows(pair_count)
    )
    right_estimate = (
        estimate_physical_work_from_owned(right_owned)
        if right_owned is not None
        else PhysicalWorkEstimate.from_rows(pair_count)
    )
    if workload_shape in (WorkloadShape.BROADCAST_RIGHT, WorkloadShape.SCALAR_RIGHT):
        right_coordinate_work = int(right_estimate.coordinate_count) * max(pair_count, 1)
        right_segment_work = int(right_estimate.segment_count) * max(pair_count, 1)
    else:
        right_coordinate_work = int(right_estimate.coordinate_count)
        right_segment_work = int(right_estimate.segment_count)
    coordinate_count = int(left_estimate.coordinate_count) + right_coordinate_work
    segment_count = int(left_estimate.segment_count) + right_segment_work
    if workload_shape in (WorkloadShape.BROADCAST_RIGHT, WorkloadShape.SCALAR_RIGHT):
        right_ring_work = int(right_estimate.ring_count) * max(pair_count, 1)
    else:
        right_ring_work = int(right_estimate.ring_count)
    ring_count = int(left_estimate.ring_count) + right_ring_work
    output_bytes = coordinate_count * 16 + ring_count * 8 + pair_count * 32
    temporary_bytes = segment_count * _OVERLAY_SEGMENT_TABLE_BYTES_PER_SEGMENT
    dispatch_units = max(
        pair_count,
        coordinate_count,
        segment_count,
        ring_count,
        output_bytes // 64,
        temporary_bytes // 128,
    )
    return PhysicalWorkEstimate(
        row_count=pair_count,
        coordinate_count=coordinate_count,
        segment_count=segment_count,
        ring_count=ring_count,
        candidate_pair_count=pair_count,
        output_row_count=pair_count,
        output_byte_count=output_bytes,
        temporary_byte_count=temporary_bytes,
        primary_unit_count=dispatch_units,
        primary_unit_name="overlay-segment",
    )


def _overlay_broadcast_right_work_units(left_owned, right_owned) -> int:
    """Return shape-level exact work for one right geometry against every left row."""
    return _overlay_pair_work_estimate(
        left_owned,
        right_owned,
        pair_count=int(left_owned.row_count),
        workload_shape=WorkloadShape.BROADCAST_RIGHT,
    ).dispatch_unit_count()


def _overlay_relation_pair_work_estimate(
    left_owned,
    right_owned,
    *,
    pair_count: int,
) -> PhysicalWorkEstimate:
    """Estimate a candidate relation before aligned geometry rows are gathered."""
    return estimate_relation_pair_work_from_owned(
        left_owned,
        right_owned,
        pair_count=pair_count,
        temporary_bytes_per_segment=_OVERLAY_SEGMENT_TABLE_BYTES_PER_SEGMENT,
        primary_unit_name="overlay-relation-segment",
    )


def _global_positions_for_local_indices(global_positions, local_indices):
    """Gather row positions without crossing device positions to host."""
    if _is_device_array(global_positions) or _is_device_array(local_indices):
        import cupy as cp

        d_local = cp.asarray(local_indices, dtype=cp.int64)
        if not _is_device_array(global_positions):
            host_positions = np.asarray(global_positions)
            if host_positions.ndim == 1 and np.array_equal(
                host_positions,
                np.arange(host_positions.size, dtype=host_positions.dtype),
            ):
                return d_local.astype(cp.int64, copy=False)
        return cp.asarray(global_positions, dtype=cp.int64)[d_local]
    return np.asarray(global_positions, dtype=np.intp)[np.asarray(local_indices, dtype=np.intp)]


def _indexed_positions_to_host(indices, *, reason: str) -> np.ndarray:
    if _is_device_array(indices):
        return _overlay_device_to_host(indices, reason=reason, dtype=np.intp)
    return np.asarray(indices, dtype=np.intp)


def _maybe_seed_polygon_validity_cache(spatial) -> None:
    geometry = spatial.geometry if isinstance(spatial, GeoDataFrame) else spatial
    values = geometry.values
    if getattr(values, "native_composition", None) is not None:
        return
    cached_owned = getattr(values, "cached_owned", None)
    owned = (
        cached_owned()
        if callable(cached_owned)
        else getattr(values, "_owned", None)
    )
    if owned is None:
        return
    if not _series_family_summary(geometry)[0]:
        return

    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(owned)


def _seed_all_validity_cache_if_owned(owned) -> None:
    if owned is None:
        return

    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(owned)


def _candidate_rows_all_valid(series: GeoSeries, row_indices) -> bool:
    row_size = getattr(row_indices, "size", None)
    row_count = int(row_size if row_size is not None else len(row_indices))
    if row_count == 0:
        return True
    rows_are_device = hasattr(row_indices, "__cuda_array_interface__")

    owned = _series_owned(series)
    if owned is not None:
        from vibespatial.runtime.residency import Residency

        state = getattr(owned, "device_state", None)
        if state is not None and state.trusted_all_valid is True:
            return True

        def _device_selected_rows_all_valid(rows) -> bool | None:
            if (
                cp is None
                or owned.residency is not Residency.DEVICE
                or owned.device_state is None
                or not has_gpu_runtime()
            ):
                return None
            try:
                from vibespatial.constructive.validity import (
                    _public_validity_gpu_device_values,
                )

                d_rows = cp.asarray(rows, dtype=cp.int64)
                d_public_valid = _public_validity_gpu_device_values(
                    owned,
                    preserve_indexed_view=True,
                )
                if int(d_public_valid.size) != int(owned.row_count):
                    return None
                return _overlay_bool_scalar(
                    cp.all(d_public_valid[d_rows]),
                    reason="overlay candidate validity scalar fence",
                )
            except _OverlayNativeConstructiveDeclined:
                raise

        if rows_are_device:
            device_result = _device_selected_rows_all_valid(row_indices)
            if device_result is not None:
                return device_result
            return False

        def _take_owned_rows(rows: np.ndarray):
            rows = np.asarray(rows, dtype=np.int64)
            if rows.size == 0:
                return owned.take(rows)
            if owned.residency is Residency.DEVICE and has_gpu_runtime():
                try:
                    import cupy as cp
                except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
                    cp = None
                if cp is not None:
                    return owned.device_take(
                        cp.asarray(rows, dtype=cp.int64),
                        host_indices_for_sizing=rows,
                    )
            return owned.take(rows)

        cached = getattr(owned, "_cached_is_valid_mask", None)
        if cached is not None and int(cached.size) == int(owned.row_count):
            rows = np.asarray(row_indices, dtype=np.int64)
            cached_mask = np.asarray(cached[rows], dtype=bool)
            validity = getattr(owned, "_validity", None)
            if validity is not None and int(validity.size) == int(owned.row_count):
                validity = np.asarray(validity, dtype=bool)
                cached_mask = cached_mask.copy()
                cached_mask[~validity[rows]] = True
            return bool(cached_mask.all())

        rows = np.asarray(row_indices, dtype=np.int64)
        subset = _take_owned_rows(rows)
        if _owned_subset_is_known_valid_rectangles(subset):
            return True
        device_result = _device_selected_rows_all_valid(rows)
        if device_result is not None:
            return device_result

        from vibespatial.constructive.validity import is_valid_owned

        validate_full_source = (
            owned.residency is Residency.DEVICE
            and has_gpu_runtime()
            and (
                int(owned.row_count) <= 2048
                or (rows.size >= 64 and rows.size * 10 >= int(owned.row_count))
            )
        )
        if validate_full_source:
            full_valid_mask = np.asarray(is_valid_owned(owned), dtype=bool)
            if not bool(np.all(owned.validity)):
                full_valid_mask = full_valid_mask.copy()
                full_valid_mask[~owned.validity] = True
            if full_valid_mask.size == int(owned.row_count):
                owned._cached_is_valid_mask = full_valid_mask
                return bool(full_valid_mask[rows].all())

        valid_mask = np.asarray(is_valid_owned(subset), dtype=bool)
        if not bool(np.all(subset.validity)):
            valid_mask = valid_mask.copy()
            valid_mask[~subset.validity] = True
        return bool(valid_mask.all())

    if rows_are_device:
        return False
    return bool(series.iloc[np.asarray(row_indices, dtype=np.intp)].is_valid.all())


def _cached_intersection_pair_count(index_result) -> int:
    if isinstance(index_result, NativeRelationSelection):
        return index_result.capacity
    if isinstance(index_result, DeviceSpatialJoinResult):
        return index_result.size
    if index_result is None:
        return 0
    return int(len(index_result[0]))


def _cached_intersection_unique_rows(index_result):
    if isinstance(index_result, NativeRelationSelection):
        return None, None
    if isinstance(index_result, DeviceSpatialJoinResult):
        if cp is None:
            return None, None
        return (
            cp.unique(cp.asarray(index_result.d_left_idx, dtype=cp.int32)),
            cp.unique(cp.asarray(index_result.d_right_idx, dtype=cp.int32)),
        )
    if index_result is None:
        return (
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.int32),
        )
    return (
        np.unique(np.asarray(index_result[0], dtype=np.int32)),
        np.unique(np.asarray(index_result[1], dtype=np.int32)),
    )


def _cached_relation_selection_rows_all_valid(
    left_series,
    right_series,
    relation_selection,
) -> bool:
    """Prove validity only for rows participating in a dynamic relation."""
    if cp is None or not has_gpu_runtime():
        return False
    from vibespatial.constructive.validity import (
        _public_validity_gpu_device_values,
    )
    from vibespatial.runtime.residency import Residency

    def _side_valid(series, *, side: str) -> bool:
        owned = _series_owned(series)
        if owned is None or owned.residency is not Residency.DEVICE:
            return False
        state = owned._ensure_device_state(preserve_indexed_view=True)
        if state.trusted_all_valid is True:
            return True
        d_valid = _public_validity_gpu_device_values(
            owned,
            preserve_indexed_view=True,
        )
        expression = (
            relation_selection.left_match_count_expression()
            if side == "left"
            else relation_selection.right_match_count_expression()
        )
        d_counts = cp.asarray(expression.values, dtype=cp.int64)
        if int(d_counts.size) != int(d_valid.size):
            return False
        return _overlay_bool_scalar(
            cp.all((d_counts == 0) | cp.asarray(d_valid, dtype=cp.bool_)),
            reason=f"overlay cached {side} relation validity scalar proof",
        )

    return _side_valid(left_series, side="left") and _side_valid(
        right_series,
        side="right",
    )


def _overlay_relation_selection_intersection_native(
    df1,
    df2,
    relation_selection,
    *,
    preserve_lower_dimensional: bool,
    warn_on_dropped_lower_dimensional: bool,
):
    """Construct a cached dynamic relation at pair capacity on device."""
    from vibespatial.api._native_state import get_native_state
    from vibespatial.constructive.measurement import _area_gpu_device_fp64
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS
    from vibespatial.runtime.residency import Residency

    relation = relation_selection.relation
    cached_subset = relation.origin == "intersection-pair-cache-subset"
    left_state = get_native_state(df1)
    right_state = get_native_state(df2)
    if cached_subset:
        left_state = left_state or _relation_join_source_state(df1)
        right_state = right_state or _relation_join_source_state(df2)
    if left_state is None or right_state is None:
        return None
    result = _relation_selection_constructive_to_native_tabular_result(
        op="intersection",
        relation_selection=relation_selection,
        left_state=left_state,
        right_state=right_state,
        frame_attrs=(
            None if preserve_lower_dimensional else {"_vibespatial_keep_geom_type_applied": True}
        ),
    )
    if result is None:
        return None
    owned = result.capacity_result.geometry.owned
    if owned is None or owned.residency is not Residency.DEVICE:
        return None

    state = owned._ensure_device_state(preserve_indexed_view=True)
    d_active = result.selection.active_capacity_mask()
    if preserve_lower_dimensional:
        aligned_left, aligned_right = _aligned_pair_owned_from_area(owned)
        d_topology_remnants = getattr(
            owned,
            "_polygon_intersection_lower_dimensional_remnant",
            None,
        )
        if d_topology_remnants is not None:
            d_topology_remnants = (
                cp.asarray(d_topology_remnants, dtype=cp.bool_) & d_active
            )
        composition_result = polygon_pair_boundary_remnants_capacity_device(
            aligned_left,
            aligned_right,
            owned,
            crs=left_state.geometry.crs,
            remnant_mask=d_topology_remnants,
        )
        if composition_result is None:
            return None
        composition_geometry, d_composition_keep = composition_result
        capacity_result = replace(
            result.capacity_result,
            geometry=composition_geometry,
            geometry_metadata=None,
        )
        selected = type(result)(
            capacity_result=capacity_result,
            selection=NativeDeviceSelection.from_mask(
                d_active & cp.asarray(d_composition_keep, dtype=cp.bool_),
                source_row_count=result.capacity,
            ),
        )
        record_dispatch_event(
            surface="geopandas.overlay",
            operation="intersection",
            implementation="cached_relation_selection_composition_gpu",
            reason=(
                "cached subset relation preserved polygon area and boundary "
                "remnants through pair-capacity composition"
            ),
            detail=(
                f"pair_capacity={result.capacity}; "
                "physical_shape=relation_selection_pair_capacity_composition"
            ),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )
        return selected

    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_validity = cp.asarray(state.validity, dtype=cp.bool_)
    d_polygonal = (d_tags == cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])) | (
        d_tags == cp.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
    )
    d_areas = _area_gpu_device_fp64(owned)
    d_keep = d_active & d_validity & d_polygonal & cp.isfinite(d_areas) & (d_areas > 0.0)
    if warn_on_dropped_lower_dimensional:
        left_owned = left_state.geometry.cached_owned()
        right_owned = right_state.geometry.cached_owned()
        d_capacity_rows = cp.arange(result.capacity, dtype=cp.int64)
        dropped_count = _device_count_dropped_polygon_warning_rows_owned(
            owned,
            warning_rows=d_capacity_rows,
            warning_keep_mask=d_keep,
            warning_active_mask=d_active,
            left_owned=left_owned,
            right_owned=right_owned,
            warning_left_rows=result.provenance.left_rows,
            warning_right_rows=result.provenance.right_rows,
        )
        if dropped_count is None:
            return None
        if dropped_count > 0:
            warnings.warn(
                "`keep_geom_type=True` in overlay resulted in "
                f"{dropped_count} dropped geometries of different "
                "geometry types than df1 has. Set keep_geom_type=False "
                "to retain all geometries",
                UserWarning,
                stacklevel=4,
            )
    selected = type(result)(
        capacity_result=result.capacity_result,
        selection=NativeDeviceSelection.from_mask(
            d_keep,
            source_row_count=result.capacity,
            geometry_family_domain=(
                GeometryFamily.POLYGON,
                GeometryFamily.MULTIPOLYGON,
            ),
            trusted_all_valid_rows=True,
        ),
    )
    record_dispatch_event(
        surface="geopandas.overlay",
        operation="intersection",
        implementation="cached_relation_selection_constructive_gpu",
        reason=(
            "cached subset relation stayed capacity-backed through polygon "
            "construction and positive-area filtering"
        ),
        detail=(
            f"pair_capacity={result.capacity}; physical_shape=relation_selection_pair_capacity"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return selected


def _owned_subset_is_known_valid_rectangles(owned) -> bool:
    """Return True for dense positive-area rectangle polygon subsets."""
    if owned.row_count == 0:
        return True

    from vibespatial.runtime.residency import Residency

    if owned.residency is not Residency.DEVICE or owned.device_state is None:
        return False
    if getattr(owned, "is_indexed_view", False):
        return False
    state = owned._ensure_device_state(preserve_indexed_view=True)
    if state.trusted_all_valid is not True:
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by residency
            return False
        if not _overlay_bool_scalar(
            cp.asarray(state.validity, dtype=cp.bool_).all(),
            reason="overlay rectangle-validity validity scalar fence",
        ):
            return False

    if state.trusted_homogeneous_family is not None and state.trusted_all_non_empty is False:
        return False

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        _device_rectangle_bounds,
    )

    polygon_buf = state.families.get(GeometryFamily.POLYGON)
    bounds = _device_rectangle_bounds(polygon_buf, owned.row_count)
    if bounds is None:
        return False
    xmin, ymin, xmax, ymax = bounds
    return bool(
        _overlay_bool_scalar(
            ((xmax - xmin) > SPATIAL_EPSILON).all(),
            reason="overlay rectangle-validity x-span scalar fence",
        )
        and _overlay_bool_scalar(
            ((ymax - ymin) > SPATIAL_EPSILON).all(),
            reason="overlay rectangle-validity y-span scalar fence",
        )
    )


def _sync_hotpath() -> None:
    if hotpath_timing_enabled():
        from vibespatial.cuda._runtime import get_cuda_runtime

        get_cuda_runtime().synchronize()


def _geoseries_object_values(series: GeoSeries) -> np.ndarray:
    """Return a fast object array view of a GeoSeries-backed GeometryArray."""
    return np.asarray(series.array, dtype=object)


def _take_geoseries_object_values(series: GeoSeries, rows: np.ndarray) -> np.ndarray:
    """Materialize only the selected rows from a GeoSeries-backed GeometryArray."""
    rows = np.asarray(rows, dtype=np.int64)
    owned = getattr(series.values, "_owned", None)
    if owned is not None:
        return np.asarray(owned.take(rows).to_shapely(), dtype=object)
    return np.asarray(series.array.take(rows.astype(np.intp, copy=False)), dtype=object)


def _polygon_rect_overlap_mask(geometries: GeoSeries) -> np.ndarray | None:
    """Return normalized rectangle-overlap metadata when present."""
    mask = getattr(geometries.values, "_polygon_rect_boundary_overlap", None)
    if mask is None:
        owned = getattr(geometries.values, "_owned", None)
        if owned is not None:
            mask = getattr(owned, "_polygon_rect_boundary_overlap", None)
    if mask is None:
        return None
    mask = _overlay_host_bool_mask_sparse_first(
        mask,
        length=len(geometries),
        dense_reason="overlay polygon rectangle-overlap mask host boundary",
        sparse_reason="overlay polygon rectangle-overlap rows host boundary",
    )
    return mask


def _polygon_rect_exact_polygon_only_mask(geometries: GeoSeries) -> np.ndarray | None:
    """Return rows whose rectangle fast path is known polygon-complete."""
    mask = _polygon_rect_exact_polygon_only_mask_raw(geometries)
    if mask is None:
        return None
    mask = _overlay_host_bool_mask_sparse_first(
        mask,
        length=len(geometries),
        dense_reason="overlay polygon exact-polygon-only mask host boundary",
        sparse_reason="overlay polygon exact-polygon-only rows host boundary",
    )
    return mask


def _polygon_rect_exact_polygon_only_mask_raw(geometries: GeoSeries):
    """Return exact-polygon-only metadata without crossing the host boundary."""
    mask = getattr(geometries.values, "_polygon_rect_exact_polygon_only", None)
    if mask is None:
        owned = getattr(geometries.values, "_owned", None)
        if owned is not None:
            mask = getattr(owned, "_polygon_rect_exact_polygon_only", None)
    return mask


def _exact_intersection_cache_from_sparse_metadata(
    owner,
    *,
    length: int,
    positions_reason: str,
    allow_device_position_export: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return dense exact-intersection cache from sparse row-position metadata."""
    sparse_values = getattr(owner, "_exact_intersection_sparse_values", None)
    sparse_mask = getattr(owner, "_exact_intersection_sparse_value_mask", None)
    sparse_positions = getattr(owner, "_exact_intersection_sparse_positions", None)
    if sparse_values is None or sparse_mask is None or sparse_positions is None:
        return None, None

    sparse_values = np.asarray(sparse_values, dtype=object)
    sparse_mask = np.asarray(sparse_mask, dtype=bool)
    if sparse_values.size != sparse_mask.size:
        return None, None

    if _is_device_array(sparse_positions) and not allow_device_position_export:
        return None, None

    host_positions = _indexed_positions_to_host(
        sparse_positions,
        reason=positions_reason,
    )
    if host_positions.size != sparse_values.size:
        return None, None

    row_indices = np.asarray(host_positions, dtype=np.intp)
    if row_indices.size and (np.any(row_indices < 0) or np.any(row_indices >= int(length))):
        return None, None

    exact_values = np.empty(int(length), dtype=object)
    exact_values[:] = None
    exact_mask = np.zeros(int(length), dtype=bool)
    exact_values[row_indices] = sparse_values
    exact_mask[row_indices] = sparse_mask
    return exact_values, exact_mask


def _exact_intersection_cache_from_metadata_owner(
    owner,
    *,
    length: int,
    sparse_positions_reason: str,
    allow_device_position_export: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return exact-intersection cache from dense or sparse metadata."""
    exact_values = getattr(owner, "_exact_intersection_values", None)
    exact_mask = getattr(owner, "_exact_intersection_value_mask", None)
    if exact_values is not None and exact_mask is not None:
        return np.asarray(exact_values, dtype=object), np.asarray(exact_mask, dtype=bool)
    return _exact_intersection_cache_from_sparse_metadata(
        owner,
        length=length,
        positions_reason=sparse_positions_reason,
        allow_device_position_export=allow_device_position_export,
    )


def _can_defer_make_valid_to_rect_repair(geometries: GeoSeries) -> bool:
    """Return True when targeted rectangle repair can replace generic make_valid."""
    return _polygon_rect_overlap_mask(geometries) is not None


def _empty_owned_result_base(row_count: int, *, device: bool):
    """Build an all-null owned array for scatter assembly."""
    if device:
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover
            device = False
        else:
            from vibespatial.geometry.owned import build_device_resident_owned

            return build_device_resident_owned(
                device_families={},
                row_count=row_count,
                tags=cp.full(row_count, -1, dtype=cp.int8),
                validity=cp.zeros(row_count, dtype=cp.bool_),
                family_row_offsets=cp.full(row_count, -1, dtype=cp.int32),
                execution_mode="gpu",
            )

    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.runtime.residency import Residency

    return OwnedGeometryArray(
        validity=np.zeros(row_count, dtype=bool),
        tags=np.full(row_count, -1, dtype=np.int8),
        family_row_offsets=np.full(row_count, -1, dtype=np.int32),
        families={},
        residency=Residency.HOST,
    )


def _owned_valid_nonempty_mask(owned) -> np.ndarray:
    """Return a public-boundary keep mask without materializing full host geometry."""
    device_mask = _owned_valid_nonempty_mask_device(owned)
    if device_mask is not None:
        return _overlay_device_to_host(
            device_mask,
            reason="overlay valid non-empty terminal mask export",
            dtype=bool,
        )

    from vibespatial.geometry.owned import FAMILY_TAGS

    validity = np.asarray(owned.validity, dtype=bool)
    if not validity.any():
        return validity

    tags = np.asarray(owned.tags)
    row_offsets = np.asarray(owned.family_row_offsets)
    keep_mask = validity.copy()

    for family in owned.families:
        owned._ensure_host_family_structure(family)
        family_tag = FAMILY_TAGS[family]
        family_mask = validity & (tags == family_tag)
        if not family_mask.any():
            continue
        family_rows = row_offsets[family_mask]
        empty_mask = np.asarray(owned.families[family].empty_mask, dtype=bool)
        if empty_mask.size == 0 or np.any((family_rows < 0) | (family_rows >= empty_mask.size)):
            # Fall back to the GeometryArray path if metadata is inconsistent.
            return validity & ~np.asarray(GeometryArray.from_owned(owned).is_empty, dtype=bool)
        keep_mask[np.flatnonzero(family_mask)] = ~empty_mask[family_rows]

    return keep_mask


def _owned_valid_nonempty_mask_device(owned):
    """Build the non-empty keep mask from logical device metadata when possible."""
    if getattr(owned, "device_state", None) is None or not has_gpu_runtime():
        return None
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    state = owned._ensure_device_state(preserve_indexed_view=True)
    row_count = int(owned.row_count)
    if row_count == 0:
        return cp.empty(0, dtype=cp.bool_)
    if state.trusted_all_valid is True and getattr(state, "trusted_all_non_empty", None) is True:
        return cp.ones(row_count, dtype=cp.bool_)

    d_validity = cp.asarray(state.validity, dtype=cp.bool_)
    d_keep = d_validity.copy()
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)

    if state.trusted_polygonal_only is True:
        family_domain = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
    elif state.trusted_homogeneous_family is not None:
        family_domain = (state.trusted_homogeneous_family,)
    else:
        family_domain = tuple(getattr(state, "families", {}) or ())

    for family in family_domain:
        d_buf = state.families.get(family)
        if d_buf is None:
            continue
        d_empty_source = getattr(d_buf, "empty_mask", None)
        if d_empty_source is None:
            return None
        empty_count = int(getattr(d_empty_source, "size", 0))
        if empty_count == 0:
            continue
        tag = FAMILY_TAGS.get(family)
        if tag is None:
            return None
        d_family_mask = d_validity & (d_tags == np.int8(tag))
        d_in_bounds = d_family_mask & (d_family_rows >= 0) & (d_family_rows < np.int64(empty_count))
        d_safe_rows = cp.where(d_in_bounds, d_family_rows, 0)
        d_family_empty = cp.asarray(d_empty_source, dtype=cp.bool_)[d_safe_rows] & d_in_bounds
        d_keep &= ~d_family_empty

    return d_keep


def _geometry_native_result_from_geoseries(geoseries: GeoSeries) -> GeometryNativeResult:
    return GeometryNativeResult.from_geoseries(geoseries)


def _extract_owned_pair(
    df1,
    df2,
    *,
    how: str | None = None,
    left_all_polygons: bool | None = None,
    right_all_polygons: bool | None = None,
):
    """Return (left_owned, right_owned) if both DataFrames have owned backing, else (None, None)."""
    ga1 = df1.geometry.values
    ga2 = df2.geometry.values
    left_owned = _series_owned(df1.geometry)
    right_owned = _series_owned(df2.geometry)
    if left_all_polygons is None:
        left_all_polygons = _series_all_polygons(df1.geometry)
    if right_all_polygons is None:
        right_all_polygons = _series_all_polygons(df2.geometry)
    if left_all_polygons and right_all_polygons:
        already_owned_pair = left_owned is not None and right_owned is not None
        already_native_resident_pair = False
        if already_owned_pair:
            from vibespatial.runtime.residency import Residency, combined_residency

            already_native_resident_pair = (
                combined_residency(left_owned, right_owned) is Residency.DEVICE
            )
        within_direct_pair_cap = (len(df1) * len(df2)) <= _OVERLAY_BBOX_PAIR_FAST_PATH_MAX_PAIRS
        allow_large_owned_pair = already_owned_pair and (
            how == "difference" or strict_native_mode_enabled() or already_native_resident_pair
        )
        if not within_direct_pair_cap and not allow_large_owned_pair:
            return None, None
        try:
            if left_owned is None:
                left_owned = ga1.to_owned()
            if right_owned is None:
                right_owned = ga2.to_owned()
        except (AttributeError, NotImplementedError):
            return None, None
        if left_owned is not None and right_owned is not None:
            if has_gpu_runtime():
                from vibespatial.runtime.residency import Residency, TransferTrigger

                selected_gpu = False
                if (
                    ga1.__class__.__name__ == "DeviceGeometryArray"
                    or ga2.__class__.__name__ == "DeviceGeometryArray"
                    or getattr(left_owned, "residency", None) is Residency.DEVICE
                    or getattr(right_owned, "residency", None) is Residency.DEVICE
                ):
                    selected_gpu = True
                    if left_owned.residency is not Residency.DEVICE:
                        left_owned = left_owned.move_to(
                            Residency.DEVICE,
                            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                            reason=(
                                "overlay kept polygon pair on device after one input "
                                "already selected the device-native path"
                            ),
                        )
                    if right_owned.residency is not Residency.DEVICE:
                        right_owned = right_owned.move_to(
                            Residency.DEVICE,
                            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                            reason=(
                                "overlay kept polygon pair on device after one input "
                                "already selected the device-native path"
                            ),
                        )
                if selected_gpu:
                    record_dispatch_event(
                        surface="geopandas.overlay",
                        operation="extract_owned_pair",
                        implementation="owned_pair_device_promotion",
                        reason=(
                            "overlay preserved the polygon pair on device because "
                            "one input was already device-backed"
                        ),
                        detail=f"left_rows={len(df1)}, right_rows={len(df2)}",
                        requested=ExecutionMode.AUTO,
                        selected=ExecutionMode.GPU,
                    )
            return left_owned, right_owned
    return None, None


def _should_prefer_exact_polygon_gpu(
    df1,
    df2,
    left_owned,
    right_owned,
    *,
    left_all_polygons: bool | None = None,
    right_all_polygons: bool | None = None,
) -> bool:
    """Prefer the exact GPU polygon boundary whenever both polygon inputs have owned backing.

    Once we can represent both sides as owned polygon arrays, the exact
    polygon-intersection path should stay in the native execution family
    instead of dropping back to the host exact-intersection boundary.
    This keeps cheap host-to-owned polygon cases on the GPU path as well.
    """
    if not has_gpu_runtime():
        return False
    if left_all_polygons is None:
        left_all_polygons = _series_all_polygons(df1.geometry)
    if right_all_polygons is None:
        right_all_polygons = _series_all_polygons(df2.geometry)
    if not (left_all_polygons and right_all_polygons):
        return False
    if strict_native_mode_enabled():
        return True
    return left_owned is not None and right_owned is not None


def _should_use_owned_constructive_overlay(left_owned, right_owned) -> bool:
    """Use the owned constructive overlay path only when the workflow is truly device-native.

    Host-resident ``_owned`` backings on plain GeoPandas inputs are still a
    transitional cache layer, not a stable public constructive execution model.
    Auto-mode public overlay should only enter the owned constructive path when
    strict-native mode requires it or the data already lives on device.
    """
    if left_owned is None or right_owned is None:
        return False
    if strict_native_mode_enabled():
        return True

    from vibespatial.runtime.residency import Residency, combined_residency

    return combined_residency(left_owned, right_owned) is Residency.DEVICE


def _coerce_owned_pair_for_strict_overlay(df1, df2, left_owned, right_owned):
    """Materialize owned backing for strict overlay paths when GPU is available.

    Overlay does its spatial join before pairwise constructive work, so this
    coercion stays off the hot-path for non-strict runs. In strict mode we need
    the downstream pairwise overlay operations to use the repo-owned GPU
    dispatch instead of inheriting the generic small-workload crossover.
    """
    if not strict_native_mode_enabled() or not has_gpu_runtime():
        return left_owned, right_owned
    from vibespatial.runtime.residency import Residency, TransferTrigger

    try:
        if left_owned is None:
            left_owned = df1.geometry.values.to_owned()
        if right_owned is None:
            right_owned = df2.geometry.values.to_owned()
        if left_owned is not None and left_owned.residency is not Residency.DEVICE:
            left_owned = left_owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="strict overlay coerced owned left input to device residency",
            )
        if right_owned is not None and right_owned.residency is not Residency.DEVICE:
            right_owned = right_owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="strict overlay coerced owned right input to device residency",
            )
    except (AttributeError, NotImplementedError):
        return left_owned, right_owned
    return left_owned, right_owned


def _group_offsets_to_host(group_offsets, *, reason: str) -> np.ndarray:
    """Return host group offsets through a named overlay boundary if needed."""
    if hasattr(group_offsets, "__cuda_array_interface__"):
        return _overlay_device_to_host(group_offsets, reason=reason, dtype=np.int64)
    return np.asarray(group_offsets, dtype=np.int64)


def _group_offsets_metadata(group_offsets, *, total_rows: int | None = None):
    """Return host offsets plus exact group metadata for CPU execution."""
    if hasattr(group_offsets, "__cuda_array_interface__"):
        raise TypeError("device group offsets must be consumed through NativeGrouped metadata")

    offsets = np.asarray(group_offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("group_offsets must be a 1D array with length >= 1")
    counts = np.diff(offsets).astype(np.int64, copy=False)
    if np.any(counts < 0):
        raise ValueError("group_offsets must be monotonically nondecreasing")
    max_group_size = int(counts.max(initial=0))
    return offsets, counts, max_group_size, len(offsets) - 1


def _native_grouped_from_sorted_offsets(
    group_offsets,
    *,
    row_count: int,
    force_device: bool,
    all_groups_observed: bool | None = None,
    group_size_min: int | None = None,
    group_size_max: int | None = None,
) -> NativeGrouped:
    """Build the grouped constructive carrier for sorted right-side rows."""
    if force_device:
        if not has_gpu_runtime():
            raise _GroupedOverlayDifferenceNativeDeclined(
                "native grouped difference requires a GPU runtime"
            )
        import cupy as cp

        offsets = cp.asarray(group_offsets, dtype=cp.int64)
        return NativeGrouped.from_sorted_offsets(
            offsets,
            row_count=int(row_count),
            all_groups_observed=all_groups_observed,
            group_size_min=group_size_min,
            group_size_max=group_size_max,
        )

    return NativeGrouped.from_sorted_offsets(
        np.asarray(group_offsets, dtype=np.int64),
        row_count=int(row_count),
        all_groups_observed=all_groups_observed,
        group_size_min=group_size_min,
        group_size_max=group_size_max,
    )


def _native_grouped_max_group_size(grouped: NativeGrouped) -> int | None:
    if grouped.group_size_max is not None:
        return int(grouped.group_size_max)
    offsets = grouped.group_offsets
    if hasattr(offsets, "__cuda_array_interface__"):
        return None

    h_offsets = np.asarray(offsets, dtype=np.int64)
    if h_offsets.size <= 1:
        return 0
    return int(np.diff(h_offsets).max(initial=0))


def _native_grouped_fixed_group_size(
    grouped: NativeGrouped,
    *,
    row_count: int,
) -> int | None:
    """Return exact fixed group width when NativeGrouped metadata proves it."""
    n_groups = grouped.resolved_group_count
    if n_groups == 0:
        return 0 if int(row_count) == 0 else None

    group_size_min = None if grouped.group_size_min is None else int(grouped.group_size_min)
    group_size_max = None if grouped.group_size_max is None else int(grouped.group_size_max)
    if (
        group_size_min is not None
        and group_size_max is not None
        and group_size_min == group_size_max
    ):
        return group_size_min

    if grouped.all_groups_observed is not True:
        return None
    observed_group_count = 0 if grouped.group_ids is None else int(grouped.group_ids.size)
    if observed_group_count != n_groups:
        return None
    if n_groups == 1:
        return int(row_count)
    if group_size_min is not None and int(row_count) == n_groups * group_size_min:
        return group_size_min
    if group_size_max is not None and int(row_count) == n_groups * group_size_max:
        return group_size_max
    return None


def _native_grouped_all_groups_positive(grouped: NativeGrouped) -> bool:
    if grouped.all_groups_observed is True:
        return True
    if grouped.group_size_min is not None and int(grouped.group_size_min) > 0:
        return True
    return False


def _native_grouped_source_rows(
    grouped: NativeGrouped,
    *,
    total_count: int,
):
    """Expand a `NativeGrouped` carrier to one source row per grouped member."""
    if total_count == 0:
        if grouped.is_device:
            import cupy as cp

            return cp.empty(0, dtype=cp.int32)
        return np.empty(0, dtype=np.int32)

    offsets = grouped.group_offsets
    group_ids = grouped.group_ids
    if group_ids is None:
        raise _GroupedOverlayDifferenceNativeDeclined(
            "NativeGrouped source-row expansion requires observed group ids"
        )
    if hasattr(offsets, "__cuda_array_interface__") or hasattr(
        group_ids,
        "__cuda_array_interface__",
    ):
        import cupy as cp

        d_offsets = cp.asarray(offsets, dtype=cp.int64)
        d_group_ids = cp.asarray(group_ids, dtype=cp.int32)
        positions = cp.arange(int(total_count), dtype=cp.int64)
        compact_rows = cp.searchsorted(
            d_offsets[1:],
            positions,
            side="right",
        ).astype(cp.int64, copy=False)
        return d_group_ids[compact_rows].astype(cp.int32, copy=False)

    h_offsets = np.asarray(offsets, dtype=np.int64)
    h_group_ids = np.asarray(group_ids, dtype=np.int32)
    positions = np.arange(int(total_count), dtype=np.int64)
    compact_rows = np.searchsorted(h_offsets[1:], positions, side="right").astype(
        np.int64,
        copy=False,
    )
    return h_group_ids[compact_rows].astype(np.int32, copy=False)


def _host_nested_row_segment_counts(
    geometry_offsets: np.ndarray,
    leaf_offsets: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray | None:
    starts = geometry_offsets[rows]
    ends = geometry_offsets[rows + 1]
    if starts.size == 0:
        return np.empty(0, dtype=np.int64)
    if int(starts.min(initial=0)) < 0 or int(ends.max(initial=0)) >= int(leaf_offsets.size):
        return None
    leaf_segment_counts = np.maximum(np.diff(leaf_offsets) - 1, 0)
    prefix = np.empty(leaf_segment_counts.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(leaf_segment_counts, out=prefix[1:])
    return prefix[ends] - prefix[starts]


def _host_multipolygon_row_segment_counts(
    geometry_offsets: np.ndarray,
    part_offsets: np.ndarray,
    ring_offsets: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray | None:
    part_starts = geometry_offsets[rows]
    part_ends = geometry_offsets[rows + 1]
    if part_starts.size == 0:
        return np.empty(0, dtype=np.int64)
    if int(part_starts.min(initial=0)) < 0 or int(part_ends.max(initial=0)) >= int(
        part_offsets.size
    ):
        return None
    if int(part_offsets.min(initial=0)) < 0 or int(part_offsets.max(initial=0)) >= int(
        ring_offsets.size
    ):
        return None

    ring_segment_counts = np.maximum(np.diff(ring_offsets) - 1, 0)
    ring_prefix = np.empty(ring_segment_counts.size + 1, dtype=np.int64)
    ring_prefix[0] = 0
    np.cumsum(ring_segment_counts, out=ring_prefix[1:])

    part_segment_counts = ring_prefix[part_offsets[1:]] - ring_prefix[part_offsets[:-1]]
    part_prefix = np.empty(part_segment_counts.size + 1, dtype=np.int64)
    part_prefix[0] = 0
    np.cumsum(part_segment_counts, out=part_prefix[1:])
    return part_prefix[part_ends] - part_prefix[part_starts]


def _host_owned_row_segment_counts(owned) -> np.ndarray | None:
    """Return host-known per-row segment counts without materializing device state."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    row_count = int(getattr(owned, "row_count", 0))
    validity = getattr(owned, "_validity", None)
    tags = getattr(owned, "_tags", None)
    family_row_offsets = getattr(owned, "_family_row_offsets", None)
    if validity is None or tags is None or family_row_offsets is None:
        return None
    if _is_device_array(validity) or _is_device_array(tags) or _is_device_array(family_row_offsets):
        return None
    h_validity = np.asarray(validity, dtype=bool)
    h_tags = np.asarray(tags, dtype=np.int8)
    h_family_rows = np.asarray(family_row_offsets, dtype=np.int64)
    if h_validity.size != row_count or h_tags.size != row_count or h_family_rows.size != row_count:
        return None

    counts = np.zeros(row_count, dtype=np.int64)
    for family in (
        GeometryFamily.LINESTRING,
        GeometryFamily.POLYGON,
        GeometryFamily.MULTILINESTRING,
        GeometryFamily.MULTIPOLYGON,
    ):
        buffer = owned.families.get(family)
        if buffer is None:
            continue
        family_mask = h_validity & (h_tags == np.int8(FAMILY_TAGS[family]))
        if not np.any(family_mask):
            continue
        if (
            _is_device_array(buffer.geometry_offsets)
            or _is_device_array(buffer.empty_mask)
            or (
                getattr(buffer, "part_offsets", None) is not None
                and _is_device_array(buffer.part_offsets)
            )
            or (
                getattr(buffer, "ring_offsets", None) is not None
                and _is_device_array(buffer.ring_offsets)
            )
        ):
            return None
        family_rows = h_family_rows[family_mask]
        if family_rows.size == 0:
            continue
        if int(family_rows.min(initial=0)) < 0:
            return None
        geom_offsets = np.asarray(buffer.geometry_offsets, dtype=np.int64)
        empty_mask = np.asarray(buffer.empty_mask, dtype=bool)
        if int(family_rows.max(initial=-1)) + 1 >= int(geom_offsets.size):
            return None
        if int(family_rows.max(initial=-1)) >= int(empty_mask.size):
            return None
        active = ~empty_mask[family_rows]
        family_counts = np.zeros(family_rows.size, dtype=np.int64)
        if np.any(active):
            active_rows = family_rows[active]
            if family is GeometryFamily.LINESTRING:
                lengths = geom_offsets[active_rows + 1] - geom_offsets[active_rows]
                active_counts = np.maximum(lengths - 1, 0)
            elif family is GeometryFamily.POLYGON:
                ring_offsets = getattr(buffer, "ring_offsets", None)
                if ring_offsets is None:
                    return None
                active_counts = _host_nested_row_segment_counts(
                    geom_offsets,
                    np.asarray(ring_offsets, dtype=np.int64),
                    active_rows,
                )
            elif family is GeometryFamily.MULTILINESTRING:
                part_offsets = getattr(buffer, "part_offsets", None)
                if part_offsets is None:
                    return None
                active_counts = _host_nested_row_segment_counts(
                    geom_offsets,
                    np.asarray(part_offsets, dtype=np.int64),
                    active_rows,
                )
            elif family is GeometryFamily.MULTIPOLYGON:
                part_offsets = getattr(buffer, "part_offsets", None)
                ring_offsets = getattr(buffer, "ring_offsets", None)
                if part_offsets is None or ring_offsets is None:
                    return None
                active_counts = _host_multipolygon_row_segment_counts(
                    geom_offsets,
                    np.asarray(part_offsets, dtype=np.int64),
                    np.asarray(ring_offsets, dtype=np.int64),
                    active_rows,
                )
            else:  # pragma: no cover - guarded by family iteration above
                active_counts = None
            if active_counts is None:
                return None
            family_counts[active] = np.asarray(active_counts, dtype=np.int64)
        counts[np.flatnonzero(family_mask)] = family_counts
    return counts


def _grouped_difference_same_row_span_summary(
    left_batch,
    right_batch,
    group_offsets,
    *,
    max_group_size: int | None,
) -> tuple[int, int, int] | None:
    """Prove same-row segment spans from carried structural metadata."""
    left_counts = _host_owned_row_segment_counts(left_batch)
    right_counts = _host_owned_row_segment_counts(right_batch)
    if left_counts is None:
        from vibespatial.constructive.binary_constructive import (
            _polygon_segment_span_bound,
        )

        left_max_span = _polygon_segment_span_bound(left_batch)
    else:
        left_max_span = int(left_counts.max(initial=0))
    if left_max_span is None:
        return None
    left_max_span = int(left_max_span)
    if left_max_span <= 0:
        return None

    if right_counts is not None and not _is_device_array(group_offsets):
        offsets = np.asarray(group_offsets, dtype=np.int64)
        if offsets.ndim != 1 or offsets.size == 0:
            return None
        if np.any(offsets[1:] < offsets[:-1]):
            return None
        if int(offsets[-1]) > int(right_counts.size):
            return None
        prefix = np.empty(right_counts.size + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(right_counts, out=prefix[1:])
        group_segment_counts = prefix[offsets[1:]] - prefix[offsets[:-1]]
        right_max_span = int(group_segment_counts.max(initial=0))
    elif max_group_size is not None:
        if right_counts is None:
            from vibespatial.constructive.binary_constructive import (
                _polygon_segment_span_bound,
            )

            right_row_span = _polygon_segment_span_bound(right_batch)
        else:
            right_row_span = int(right_counts.max(initial=0))
        if right_row_span is None:
            return None
        right_max_span = int(right_row_span) * int(max_group_size)
    else:
        return None

    if right_max_span <= 0:
        return None
    return left_max_span, right_max_span, max(0, int(left_batch.row_count) - 1)


def _cpu_grouped_difference_owned(
    left_batch,
    right_batch,
    group_offsets,
    *,
    dispatch_mode: ExecutionMode,
):
    """Compute explicitly requested CPU grouped difference."""
    from vibespatial.constructive.binary_constructive import binary_constructive_owned

    with hotpath_stage(
        "overlay.diff.group_metadata",
        category="setup",
    ) as amplification_metadata:
        group_offsets = np.asarray(group_offsets, dtype=np.int64)
        group_lengths = np.diff(group_offsets).astype(np.int64, copy=False)
        max_group_size = int(group_lengths.max(initial=0))
        if amplification_metadata is not None:
            attach_work_amplification(
                amplification_metadata,
                operation="overlay.diff.group_metadata",
                metric_family="group_compression",
                sums={
                    "input_rows": int(right_batch.row_count),
                    "output_groups": int(left_batch.row_count),
                },
                maxima={"max_group_size": max_group_size},
                unavailable=(
                    "input_segments",
                    "input_coordinates",
                    "pre_reduction_fragments",
                    "output_parts",
                    "output_coordinates",
                ),
            )
    if max_group_size <= 0:
        return left_batch

    group_starts = group_offsets[:-1].astype(np.int64, copy=False)
    all_rows = np.arange(left_batch.row_count, dtype=np.int64)
    current_owned = left_batch

    for step in range(max_group_size):
        with hotpath_stage("overlay.diff.exact.active_rows", category="filter"):
            active_rows = all_rows[
                (group_lengths > step) & np.asarray(current_owned.validity, dtype=bool)
            ]
        if active_rows.size == 0:
            break
        _sync_hotpath()
        with hotpath_stage("overlay.diff.exact.left_take", category="refine"):
            active_left = current_owned.take(active_rows)
        _sync_hotpath()
        with hotpath_stage("overlay.diff.exact.right_take", category="refine"):
            right_step = right_batch.take(group_starts[active_rows] + step)
        _sync_hotpath()
        with hotpath_stage("overlay.diff.exact.binary_difference", category="refine"):
            active_diff = binary_constructive_owned(
                "difference",
                active_left,
                right_step,
                dispatch_mode=dispatch_mode,
            )
        _sync_hotpath()
        if active_rows.size == current_owned.row_count:
            current_owned = active_diff
        else:
            from vibespatial.geometry.owned import concat_owned_scatter

            _sync_hotpath()
            with hotpath_stage("overlay.diff.exact.scatter", category="refine"):
                current_owned = concat_owned_scatter(
                    current_owned,
                    active_diff,
                    active_rows,
                )
            _sync_hotpath()

    return current_owned


def _single_pair_grouped_difference_owned(
    left_batch,
    right_batch,
    *,
    dispatch_mode: ExecutionMode,
):
    """Compute one-candidate grouped difference as aligned pairwise work."""
    from vibespatial.constructive.binary_constructive import binary_constructive_owned

    return binary_constructive_owned(
        "difference",
        left_batch,
        right_batch,
        dispatch_mode=dispatch_mode,
    )


def _sparse_single_pair_grouped_difference_owned(
    left_batch,
    right_batch,
    grouped: NativeGrouped,
    *,
    dispatch_mode: ExecutionMode,
):
    """Difference sparse one-candidate groups and scatter them into left rows."""
    from vibespatial.constructive.binary_constructive import binary_constructive_owned
    from vibespatial.geometry.owned import concat_owned_scatter, device_concat_owned_scatter
    from vibespatial.runtime.residency import Residency

    group_ids = grouped.group_ids
    if group_ids is None:
        raise _GroupedOverlayDifferenceNativeDeclined(
            "sparse single-pair grouped difference requires observed group ids"
        )
    observed_count = int(group_ids.size)
    if observed_count == 0:
        return left_batch
    if observed_count != right_batch.row_count and grouped.group_offsets is not None:
        offsets = grouped.group_offsets
        if hasattr(offsets, "__cuda_array_interface__"):
            import cupy as cp

            d_offsets = cp.asarray(offsets, dtype=cp.int64)
            group_ids = cp.nonzero(d_offsets[1:] > d_offsets[:-1])[0].astype(
                cp.int32,
                copy=False,
            )
        else:
            host_offsets = np.asarray(offsets, dtype=np.int64)
            group_ids = np.flatnonzero(host_offsets[1:] > host_offsets[:-1]).astype(
                np.int32,
                copy=False,
            )
        observed_count = int(group_ids.size)
    if observed_count != right_batch.row_count:
        raise _GroupedOverlayDifferenceNativeDeclined(
            "sparse single-pair grouped difference requires one right row per observed group"
        )

    if (
        left_batch.residency is Residency.DEVICE
        and hasattr(group_ids, "__cuda_array_interface__")
        and has_gpu_runtime()
    ):
        left_subset = _device_take_preserving_indexed_rows(left_batch, group_ids)
    else:
        left_subset = left_batch.take(group_ids)
    diff_subset = binary_constructive_owned(
        "difference",
        left_subset,
        right_batch,
        dispatch_mode=dispatch_mode,
    )
    if diff_subset.row_count != observed_count:
        raise _GroupedOverlayDifferenceNativeDeclined(
            "sparse single-pair grouped difference produced an unexpected row count"
        )
    if (
        left_batch.residency is Residency.DEVICE
        and diff_subset.residency is Residency.DEVICE
        and hasattr(group_ids, "__cuda_array_interface__")
        and has_gpu_runtime()
    ):
        return device_concat_owned_scatter(left_batch, diff_subset, group_ids)
    return concat_owned_scatter(left_batch, diff_subset, group_ids)


def _native_grouped_union_difference_owned(
    left_batch,
    right_batch,
    grouped: NativeGrouped,
    *,
    dispatch_mode: ExecutionMode,
    stage: str,
):
    """Difference left rows by a native grouped union of their right members."""
    if not grouped.is_device:
        return None
    if grouped.resolved_group_count != left_batch.row_count:
        return None

    from vibespatial.constructive.binary_constructive import (
        _explode_polygonal_rows_to_polygon_capacity_gpu,
        _row_aligned_rectangle_partition_difference_gpu,
        binary_constructive_owned,
    )
    from vibespatial.overlay.dissolve import DissolveUnionMethod, execute_native_grouped_union

    grouped_union_input = right_batch
    if getattr(right_batch, "is_indexed_view", False):
        grouped_union_input = right_batch.physicalize_device_rows(
            allow_capacity_allocation=True,
        )
        record_dispatch_event(
            surface="geopandas.array.difference",
            operation="difference",
            implementation="grouped_union_input_physicalization_gpu",
            reason=(
                "the grouped union reducer requires contiguous family buffers, "
                "so indexed right rows were physicalized entirely on device"
            ),
            detail=(
                f"groups={left_batch.row_count}, pairs={right_batch.row_count}, "
                f"stage={stage}"
            ),
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
        )
    grouped_union = execute_native_grouped_union(
        grouped,
        _geometries=(),
        method=DissolveUnionMethod.UNARY,
        owned=grouped_union_input,
    )
    if grouped_union is None:
        grouped_union = execute_native_grouped_union(
            grouped,
            _geometries=(),
            method=DissolveUnionMethod.DISJOINT_SUBSET,
            owned=grouped_union_input,
        )
    if grouped_union is None or grouped_union.owned is None:
        return None
    unioned_right = grouped_union.owned
    if unioned_right.row_count != left_batch.row_count:
        return None
    strip_difference = _row_aligned_rectangle_partition_difference_gpu(
        left_batch,
        unioned_right,
    )
    if strip_difference is not None:
        strip_support = getattr(
            strip_difference,
            "_rectangle_partition_difference_support_mask",
            None,
        )
        if strip_support is None:
            return None
        strip_partition = _grouped_direct_difference_capacity_partition(
            strip_difference,
            strip_support,
        )
        if strip_partition is None:
            return None
        record_dispatch_event(
            surface="geopandas.array.difference",
            operation="difference",
            implementation="grouped_overlay_difference_rectangle_strip_gpu",
            reason=(
                "grouped rectangle-strip union fed a bounded rectangle "
                "partition difference carrier without rowwise exact topology"
            ),
            detail=(f"groups={left_batch.row_count}, pairs={right_batch.row_count}, stage={stage}"),
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
        )
        direct_partitions = [strip_partition]
    else:
        direct_partitions = []
    if cp is not None and left_batch.row_count > 0:
        d_single_offsets = cp.arange(left_batch.row_count + 1, dtype=cp.int64)
        single_grouped = NativeGrouped.from_sorted_offsets(
            d_single_offsets,
            row_count=unioned_right.row_count,
            all_groups_observed=True,
            group_size_min=1,
            group_size_max=1,
        )
        direct_donuts = _grouped_polygon_donut_difference_owned(
            left_batch,
            unioned_right,
            single_grouped,
            d_single_offsets,
            dispatch_mode=dispatch_mode,
            event_implementation="grouped_overlay_difference_unioned_polygon_donuts_gpu",
            event_reason=(
                "grouped exact overlay difference merged overlapping "
                "contained holed right rows with NativeGrouped union and "
                "emitted the merged polygon as MultiPolygon main parts plus "
                "retained islands from device buffers"
            ),
            event_pairs=right_batch.row_count,
            event_detail_extra=(f", unioned_rows={unioned_right.row_count}, stage={stage}"),
        )
        if direct_donuts is not None:
            direct_partitions.append(direct_donuts)
        direct_holes = _grouped_polygon_hole_difference_owned(
            left_batch,
            unioned_right,
            single_grouped,
            d_single_offsets,
            dispatch_mode=dispatch_mode,
            event_implementation="grouped_overlay_difference_unioned_polygon_holes_gpu",
            event_reason=(
                "grouped exact overlay difference merged overlapping "
                "contained right rows with NativeGrouped union and emitted "
                "the merged polygon as an interior ring from device buffers"
            ),
            event_pairs=right_batch.row_count,
            event_detail_extra=(f", unioned_rows={unioned_right.row_count}, stage={stage}"),
        )
        if direct_holes is not None:
            direct_partitions.append(direct_holes)
    topology_left = left_batch
    exact_unioned_result = None

    def _exact_unioned_difference():
        nonlocal exact_unioned_result
        if exact_unioned_result is None:
            exact_unioned_result = binary_constructive_owned(
                "difference",
                topology_left,
                unioned_right,
                dispatch_mode=dispatch_mode,
            )
            if exact_unioned_result.row_count != topology_left.row_count:
                return None
        return exact_unioned_result

    exploded_polygonal = _explode_polygonal_rows_to_polygon_capacity_gpu(
        unioned_right,
    )
    if exploded_polygonal is not None:
        unioned_parts = exploded_polygonal.geometry
        if unioned_parts.row_count > 0:
            d_union_source_rows = cp.asarray(
                exploded_polygonal.source_rows,
                dtype=cp.int32,
            )
            parts_grouped = NativeGroupedSelection(
                group_codes=d_union_source_rows,
                selection=exploded_polygonal.selection,
                group_count=topology_left.row_count,
            )
            from vibespatial.overlay.assemble import (
                assemble_grouped_polygonal_complement_gpu,
                classify_grouped_polygonal_complement_groups_gpu,
                classify_grouped_polygonal_complement_parts_gpu,
            )

            d_supported_groups = classify_grouped_polygonal_complement_groups_gpu(
                topology_left,
                unioned_parts,
                parts_grouped,
            )
            if d_supported_groups is not None:
                direct_polygonal = assemble_grouped_polygonal_complement_gpu(
                    topology_left,
                    unioned_parts,
                    parts_grouped,
                    support_mask=d_supported_groups,
                    right_ring_capacity=exploded_polygonal.ring_capacity,
                    right_coord_capacity=exploded_polygonal.coord_capacity,
                )
                if direct_polygonal is not None:
                    polygonal_partition = _grouped_direct_difference_capacity_partition(
                        direct_polygonal,
                        d_supported_groups,
                    )
                    if polygonal_partition is not None:
                        direct_partitions.append(polygonal_partition)

            d_part_contained = classify_grouped_polygonal_complement_parts_gpu(
                topology_left,
                unioned_parts,
                parts_grouped,
            )
            if d_part_contained is not None:
                from vibespatial.api._native_rowset import NativeDeviceSelection

                d_part_active = cp.asarray(
                    exploded_polygonal.selection.active_capacity_mask(),
                    dtype=cp.bool_,
                )
                d_safe_part_groups = cp.where(
                    d_part_active,
                    d_union_source_rows,
                    cp.int32(0),
                )
                d_group_part_counts = cp.bincount(
                    d_safe_part_groups,
                    weights=d_part_active.astype(cp.int32, copy=False),
                    minlength=topology_left.row_count,
                )[: topology_left.row_count].astype(cp.int32, copy=False)
                d_group_contained_counts = cp.bincount(
                    d_safe_part_groups,
                    weights=cp.asarray(d_part_contained, dtype=cp.int32),
                    minlength=topology_left.row_count,
                )[: topology_left.row_count].astype(cp.int32, copy=False)
                d_mixed_groups = (d_group_contained_counts > 0) & (
                    d_group_contained_counts < d_group_part_counts
                )
                d_mixed_contained_parts = (
                    cp.asarray(d_part_contained, dtype=cp.bool_)
                    & d_mixed_groups[d_safe_part_groups]
                )
                mixed_selection = NativeDeviceSelection.from_mask(
                    d_mixed_contained_parts,
                    source_row_count=unioned_parts.row_count,
                )
                mixed_union_parts = _device_take_preserving_indexed_rows(
                    unioned_parts,
                    mixed_selection.partition_capacity_positions(),
                )._apply_row_activity(mixed_selection.active_capacity_mask())
                mixed_grouped = NativeGroupedSelection(
                    group_codes=mixed_selection.gather_capacity(
                        d_union_source_rows,
                        fill_value=0,
                    ).astype(cp.int32, copy=False),
                    selection=mixed_selection.as_capacity_prefix(),
                    group_count=topology_left.row_count,
                )
                crossing_exact = _exact_unioned_difference()
                if crossing_exact is not None:
                    d_mixed_supported = classify_grouped_polygonal_complement_groups_gpu(
                        crossing_exact,
                        mixed_union_parts,
                        mixed_grouped,
                    )
                    if d_mixed_supported is not None:
                        d_mixed_supported &= d_mixed_groups
                        mixed_complement = assemble_grouped_polygonal_complement_gpu(
                            crossing_exact,
                            mixed_union_parts,
                            mixed_grouped,
                            support_mask=d_mixed_supported,
                            right_ring_capacity=exploded_polygonal.ring_capacity,
                            right_coord_capacity=exploded_polygonal.coord_capacity,
                        )
                        if mixed_complement is not None:
                            mixed_partition = _grouped_direct_difference_capacity_partition(
                                mixed_complement,
                                d_mixed_supported,
                            )
                            if mixed_partition is not None:
                                direct_partitions.append(mixed_partition)

    if direct_partitions:
        from vibespatial.geometry.owned import (
            device_select_owned_capacity_partitions,
        )

        d_claimed = cp.zeros(topology_left.row_count, dtype=cp.bool_)
        direct_replacements = []
        for partition in direct_partitions:
            d_partition = (
                cp.asarray(
                    partition.support_mask,
                    dtype=cp.bool_,
                )
                & ~d_claimed
            )
            direct_replacements.append((partition.owned, d_partition))
            d_claimed |= d_partition
        exact_result = _exact_unioned_difference()
        if exact_result is None:
            return None
        if exact_result.row_count != topology_left.row_count:
            return None
        partitioned_result = device_select_owned_capacity_partitions(
            exact_result,
            direct_replacements,
        )
        record_dispatch_event(
            surface="geopandas.array.difference",
            operation="difference",
            implementation=("grouped_overlay_difference_unioned_polygonal_capacity_partition_gpu"),
            reason=(
                "NativeGrouped union difference assigned donut, hole, exploded "
                "polygonal complement, and exact rows through one device ownership map"
            ),
            detail=(
                f"groups={topology_left.row_count}, pairs={right_batch.row_count}, "
                f"partitions={len(direct_partitions)}, stage={stage}"
            ),
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
        )
        return partitioned_result

    diff_owned = _exact_unioned_difference()
    if diff_owned is None:
        return None
    if diff_owned.row_count != left_batch.row_count:
        return None
    record_dispatch_event(
        surface="geopandas.array.difference",
        operation="difference",
        implementation="native_grouped_union_difference_gpu",
        reason=(
            "grouped exact overlay difference used NativeGrouped union plus "
            f"rowwise exact difference after {stage}"
        ),
        detail=(f"groups={left_batch.row_count}, pairs={right_batch.row_count}"),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return diff_owned


def _grouped_rectangle_hole_difference_kernels():
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(
        "grouped-rectangle-hole-difference",
        _GROUPED_RECTANGLE_HOLE_DIFF_KERNEL_SOURCE,
    )
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=_GROUPED_RECTANGLE_HOLE_DIFF_KERNEL_SOURCE,
        kernel_names=(
            "validate_grouped_polygon_holes",
            "validate_grouped_rectangle_holes",
            "emit_grouped_rectangle_holes",
        ),
    )


def _owned_all_valid_without_device_probe(owned) -> bool:
    state = getattr(owned, "device_state", None)
    if getattr(state, "trusted_all_valid", None) is True:
        return True
    cached = owned._current_cached_validity_mask()
    if cached is not None and int(cached.size) == int(owned.row_count):
        return bool(np.all(cached))
    validity = getattr(owned, "_validity", None)
    if validity is not None and int(validity.size) == int(owned.row_count):
        return bool(np.all(validity))
    return False


def _grouped_direct_difference_capacity_partition(
    result,
    support_mask,
    *,
    collective_mask=None,
):
    """Mask a direct grouped result without compacting its public row capacity."""
    if cp is None or result.device_state is None:
        return None
    row_count = int(result.row_count)
    d_support = cp.asarray(support_mask, dtype=cp.bool_)
    if d_support.ndim != 1 or int(d_support.size) != row_count:
        return None

    state = result.device_state
    state.validity = cp.asarray(state.validity, dtype=cp.bool_) & d_support
    state.tags = cp.where(
        state.validity,
        cp.asarray(state.tags, dtype=cp.int8),
        cp.int8(-1),
    )
    state.family_row_offsets = cp.where(
        state.validity,
        cp.asarray(state.family_row_offsets, dtype=cp.int32),
        cp.int32(-1),
    )
    if state.row_bounds is not None:
        state.row_bounds = cp.where(
            state.validity[:, None],
            cp.asarray(state.row_bounds, dtype=cp.float64).reshape(row_count, 4),
            cp.asarray(cp.nan, dtype=cp.float64),
        )
    for device_buffer in state.families.values():
        if int(device_buffer.empty_mask.size) == row_count:
            device_buffer.empty_mask = (
                cp.asarray(
                    device_buffer.empty_mask,
                    dtype=cp.bool_,
                )
                | ~d_support
            )
    state.trusted_all_valid = None
    state.trusted_all_non_empty = None
    result._cached_is_valid_mask = None
    return _GroupedDifferenceCapacityPartition(
        result,
        d_support,
        None if collective_mask is None else cp.asarray(collective_mask, dtype=cp.bool_),
    )


def _grouped_rectangle_hole_difference_owned(
    left_batch,
    right_batch,
    grouped: NativeGrouped,
    group_offsets,
    *,
    dispatch_mode: ExecutionMode,
):
    """Native grouped rectangle minus contained rectangle holes.

    Physical shape: dense `NativeGrouped` offsets with one axis-aligned
    rectangle left row per group and one or more disjoint contained rectangle
    right rows. The output is one polygon per group with a fixed exterior ring
    and one interior ring per right row, emitted directly as device buffers.
    """
    if cp is None or not grouped.is_device:
        return None
    if grouped.resolved_group_count != int(left_batch.row_count):
        return None
    if right_batch.row_count <= 0:
        return left_batch
    if not (
        _owned_all_valid_without_device_probe(left_batch)
        and _owned_all_valid_without_device_probe(right_batch)
    ):
        return None

    from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        FamilyGeometryBuffer,
        build_device_resident_owned,
    )
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    left_state = left_batch._ensure_device_state(preserve_indexed_view=True)
    right_state = right_batch._ensure_device_state(preserve_indexed_view=True)
    left_polygon = left_state.families.get(GeometryFamily.POLYGON)
    right_polygon = right_state.families.get(GeometryFamily.POLYGON)
    if left_polygon is None or right_polygon is None:
        return None
    if (
        int(getattr(left_polygon, "dense_single_ring_width", 0) or 0) != 5
        or int(getattr(right_polygon, "dense_single_ring_width", 0) or 0) != 5
        or not bool(getattr(left_polygon, "axis_aligned_rectangles", False))
        or not bool(getattr(right_polygon, "axis_aligned_rectangles", False))
    ):
        return None

    d_offsets = cp.asarray(group_offsets, dtype=cp.int64)
    if int(d_offsets.size) != int(left_batch.row_count) + 1:
        return None
    d_left_bounds = (
        cp.asarray(left_polygon.bounds, dtype=cp.float64).reshape(left_batch.row_count, 4)
        if left_polygon.bounds is not None
        else cp.asarray(
            compute_geometry_bounds_device(left_batch, preserve_indexed_view=True),
            dtype=cp.float64,
        ).reshape(left_batch.row_count, 4)
    )
    d_right_bounds = (
        cp.asarray(right_polygon.bounds, dtype=cp.float64).reshape(right_batch.row_count, 4)
        if right_polygon.bounds is not None
        else cp.asarray(
            compute_geometry_bounds_device(right_batch, preserve_indexed_view=True),
            dtype=cp.float64,
        ).reshape(right_batch.row_count, 4)
    )

    runtime = get_cuda_runtime()
    kernels = _grouped_rectangle_hole_difference_kernels()
    ptr = runtime.pointer
    d_supported = cp.zeros(left_batch.row_count, dtype=cp.bool_)
    validate = kernels["validate_grouped_rectangle_holes"]
    grid, block = runtime.launch_config(validate, left_batch.row_count)
    runtime.launch(
        validate,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_left_bounds),
                ptr(d_right_bounds),
                ptr(d_offsets),
                int(left_batch.row_count),
                _GROUPED_RECTANGLE_HOLE_DIFF_MAX_GROUP_SIZE,
                ptr(d_supported),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
            ),
        ),
    )
    d_geometry_offsets = (d_offsets + cp.arange(left_batch.row_count + 1, dtype=cp.int64)).astype(
        cp.int32, copy=False
    )
    total_rings = int(left_batch.row_count + right_batch.row_count)
    total_coords = total_rings * 5
    d_ring_offsets = cp.arange(total_rings + 1, dtype=cp.int32) * np.int32(5)
    d_x = cp.empty(total_coords, dtype=cp.float64)
    d_y = cp.empty(total_coords, dtype=cp.float64)

    emit = kernels["emit_grouped_rectangle_holes"]
    grid, block = runtime.launch_config(emit, total_coords)
    runtime.launch(
        emit,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_left_bounds),
                ptr(d_right_bounds),
                ptr(d_offsets),
                ptr(d_geometry_offsets),
                int(left_batch.row_count),
                int(total_coords),
                ptr(d_x),
                ptr(d_y),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
            ),
        ),
    )

    row_count = int(left_batch.row_count)
    d_empty = cp.zeros(row_count, dtype=cp.bool_)
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=d_x,
                y=d_y,
                geometry_offsets=d_geometry_offsets,
                empty_mask=d_empty,
                ring_offsets=d_ring_offsets,
                bounds=d_left_bounds,
            )
        },
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8),
        validity=cp.ones(row_count, dtype=cp.bool_),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    result.families[GeometryFamily.POLYGON] = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=np.empty(0, dtype=np.int32),
        empty_mask=np.empty(0, dtype=np.bool_),
        ring_offsets=None,
        bounds=None,
        host_materialized=False,
    )
    if result.device_state is not None:
        result.device_state.trusted_all_valid = True
        result.device_state.trusted_homogeneous_family = GeometryFamily.POLYGON
        result.device_state.trusted_all_non_empty = True
        result.device_state.row_bounds = d_left_bounds
    partition = _grouped_direct_difference_capacity_partition(result, d_supported)
    if partition is None:
        return None
    record_dispatch_event(
        surface="geopandas.array.difference",
        operation="difference",
        implementation="grouped_overlay_difference_rectangle_holes_gpu",
        reason=(
            "grouped exact overlay difference emitted rectangle exterior and "
            "interior rings directly from NativeGrouped device bounds"
        ),
        detail=(
            f"groups={left_batch.row_count}, pairs={right_batch.row_count}, "
            f"group_size_bound={_GROUPED_RECTANGLE_HOLE_DIFF_MAX_GROUP_SIZE}, "
            "support=NativeRowSet"
        ),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return partition


def _grouped_polygon_buffer_shape(
    batch,
    state,
    polygon_buffer,
    logical_rows: int,
) -> tuple[tuple[int, int], str] | None:
    """Bound rings and coordinates for a logical polygon row carrier."""
    if (
        not bool(getattr(batch, "is_indexed_view", False))
        and int(polygon_buffer.geometry_offsets.size) == int(logical_rows) + 1
    ):
        return (
            (
                max(int(polygon_buffer.ring_offsets.size) - 1, 0),
                int(polygon_buffer.x.size),
            ),
            "structural",
        )

    fixed_size = getattr(polygon_buffer, "fixed_size", None)
    exact_fixed_shape = bool(
        fixed_size is not None
        and fixed_size.first_level_count_per_row is not None
        and fixed_size.coord_count_per_row is not None
    )
    rings_per_row = (
        None
        if fixed_size is None
        else (
            fixed_size.first_level_count_per_row
            if fixed_size.first_level_count_per_row is not None
            else fixed_size.max_first_level_count_per_row
        )
    )
    coords_per_row = (
        None
        if fixed_size is None
        else (
            fixed_size.coord_count_per_row
            if fixed_size.coord_count_per_row is not None
            else fixed_size.max_coord_count_per_row
        )
    )
    dense_width = getattr(polygon_buffer, "dense_single_ring_width", None)
    if dense_width is not None:
        rings_per_row = 1 if rings_per_row is None else rings_per_row
        coords_per_row = int(dense_width) if coords_per_row is None else coords_per_row
        exact_fixed_shape = True
    if rings_per_row is not None and coords_per_row is not None:
        return (
            (
                int(logical_rows) * int(rings_per_row),
                int(logical_rows) * int(coords_per_row),
            ),
            "structural" if exact_fixed_shape else "capacity",
        )

    if state.trusted_unique_family_rows is True:
        return (
            (
                max(int(polygon_buffer.ring_offsets.size) - 1, 0),
                int(polygon_buffer.x.size),
            ),
            "capacity",
        )
    return None


def _grouped_polygon_hole_difference_owned(
    left_batch,
    right_batch,
    grouped: NativeGrouped,
    group_offsets,
    *,
    dispatch_mode: ExecutionMode,
    event_implementation: str = "grouped_overlay_difference_polygon_holes_gpu",
    event_reason: str | None = None,
    event_pairs: int | None = None,
    event_detail_extra: str = "",
):
    """Native grouped polygon minus disjoint contained polygon holes.

    Physical shape: dense `NativeGrouped` offsets with one polygon left row per
    group and one or more single-ring polygon right rows strictly contained in
    that left row. The output is one polygon per group containing every original
    left ring plus one reversed right exterior ring per subtracting row.
    """
    if cp is None or not grouped.is_device:
        return None
    if grouped.resolved_group_count != int(left_batch.row_count):
        return None
    if right_batch.row_count <= 0:
        return left_batch
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
    from vibespatial.predicates.binary import binary_predicate_expression

    left_state = left_batch._ensure_device_state(preserve_indexed_view=True)
    right_state = right_batch._ensure_device_state(preserve_indexed_view=True)
    left_polygon = left_state.families.get(GeometryFamily.POLYGON)
    right_polygon = right_state.families.get(GeometryFamily.POLYGON)
    if left_polygon is None or right_polygon is None:
        return None
    if left_polygon.ring_offsets is None or right_polygon.ring_offsets is None:
        return None

    d_offsets = cp.asarray(group_offsets, dtype=cp.int64)
    row_count = int(left_batch.row_count)
    if int(d_offsets.size) != row_count + 1:
        return None
    d_group_counts = (d_offsets[1:] - d_offsets[:-1]).astype(cp.int64, copy=False)

    d_right_group_rows = _native_grouped_source_rows(
        grouped,
        total_count=right_batch.row_count,
    )
    d_right_group_rows = cp.asarray(d_right_group_rows, dtype=cp.int64)
    if int(d_right_group_rows.size) != int(right_batch.row_count):
        return None

    d_left_rows = cp.asarray(left_state.family_row_offsets, dtype=cp.int64)
    d_right_rows = cp.asarray(right_state.family_row_offsets, dtype=cp.int64)
    d_left_valid = cp.asarray(left_state.validity, dtype=cp.bool_)
    d_right_valid = cp.asarray(right_state.validity, dtype=cp.bool_)
    d_left_tags = cp.asarray(left_state.tags, dtype=cp.int8)
    d_right_tags = cp.asarray(right_state.tags, dtype=cp.int8)
    polygon_tag = cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_left_polygon_rows = d_left_valid & (d_left_tags == polygon_tag) & (d_left_rows >= 0)
    d_right_polygon_rows = d_right_valid & (d_right_tags == polygon_tag) & (d_right_rows >= 0)
    d_safe_left_rows = cp.where(d_left_polygon_rows, d_left_rows, cp.int64(0))
    d_safe_right_rows = cp.where(d_right_polygon_rows, d_right_rows, cp.int64(0))
    d_left_geom_offsets = cp.asarray(left_polygon.geometry_offsets, dtype=cp.int64)
    d_right_geom_offsets = cp.asarray(right_polygon.geometry_offsets, dtype=cp.int64)
    d_left_ring_starts = d_left_geom_offsets[d_safe_left_rows].astype(
        cp.int64,
        copy=False,
    )
    d_left_ring_ends = d_left_geom_offsets[d_safe_left_rows + 1].astype(
        cp.int64,
        copy=False,
    )
    d_left_ring_counts = (d_left_ring_ends - d_left_ring_starts).astype(
        cp.int64,
        copy=False,
    )
    d_left_ring_counts = cp.where(d_left_polygon_rows, d_left_ring_counts, cp.int64(0))
    d_right_ring_starts = d_right_geom_offsets[d_safe_right_rows].astype(
        cp.int64,
        copy=False,
    )
    d_right_ring_ends = d_right_geom_offsets[d_safe_right_rows + 1].astype(
        cp.int64,
        copy=False,
    )
    d_right_ring_counts = (d_right_ring_ends - d_right_ring_starts).astype(
        cp.int64,
        copy=False,
    )
    d_right_ring_counts = cp.where(
        d_right_polygon_rows,
        d_right_ring_counts,
        cp.int64(0),
    )

    d_left_bounds = cp.asarray(
        compute_geometry_bounds_device(left_batch, preserve_indexed_view=True),
        dtype=cp.float64,
    ).reshape(row_count, 4)
    d_right_bounds = cp.asarray(
        compute_geometry_bounds_device(right_batch, preserve_indexed_view=True),
        dtype=cp.float64,
    ).reshape(right_batch.row_count, 4)
    d_pair_left_bounds = d_left_bounds[d_right_group_rows]
    scale = cp.maximum(
        cp.maximum(
            cp.max(cp.abs(d_pair_left_bounds), axis=1),
            cp.max(cp.abs(d_right_bounds), axis=1),
        ),
        1.0,
    )
    tol = cp.maximum(scale * 1.0e-12, 1.0e-12)
    d_strict_bbox_inside = (
        (d_right_bounds[:, 0] > d_pair_left_bounds[:, 0] + tol)
        & (d_right_bounds[:, 1] > d_pair_left_bounds[:, 1] + tol)
        & (d_right_bounds[:, 2] < d_pair_left_bounds[:, 2] - tol)
        & (d_right_bounds[:, 3] < d_pair_left_bounds[:, 3] - tol)
    )

    pair_left = _device_take_preserving_indexed_rows(left_batch, d_right_group_rows)
    coverage = binary_predicate_expression(
        "covers",
        pair_left,
        right_batch,
        dispatch_mode=ExecutionMode.GPU,
        operation="overlay.grouped_difference.polygon_hole_admission",
    )
    if coverage is None:
        return None
    d_covered = cp.asarray(coverage.values, dtype=cp.bool_)
    if int(d_covered.size) != int(right_batch.row_count):
        return None

    d_pair_left_ring_counts = d_left_ring_counts[d_right_group_rows]
    d_row_base_supported = (
        d_left_polygon_rows[d_right_group_rows]
        & d_right_polygon_rows
        & d_strict_bbox_inside
        & d_covered
    )
    d_collective_row_supported = d_row_base_supported & (
        ((d_right_ring_counts == 1) & (d_pair_left_ring_counts >= 1))
        | ((d_right_ring_counts >= 2) & (d_pair_left_ring_counts == 1))
    )
    d_row_supported = (
        d_row_base_supported & (d_pair_left_ring_counts >= 1) & (d_right_ring_counts == 1)
    )
    d_collective_supported_counts = cp.bincount(
        d_right_group_rows,
        weights=d_collective_row_supported.astype(cp.int32, copy=False),
        minlength=row_count,
    )[:row_count].astype(cp.int64, copy=False)
    d_collective_groups = (d_group_counts > 0) & (d_collective_supported_counts > 0)

    runtime = get_cuda_runtime()
    kernels = _grouped_rectangle_hole_difference_kernels()
    ptr = runtime.pointer
    d_supported_groups = cp.zeros(row_count, dtype=cp.bool_)
    d_row_supported_u8 = d_row_supported.astype(cp.uint8, copy=False)
    validate = kernels["validate_grouped_polygon_holes"]
    grid, block = runtime.launch_config(validate, row_count)
    runtime.launch(
        validate,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_row_supported_u8),
                ptr(d_right_bounds),
                ptr(d_offsets),
                row_count,
                _GROUPED_RECTANGLE_HOLE_DIFF_MAX_GROUP_SIZE,
                ptr(d_supported_groups),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
            ),
        ),
    )
    left_shape = _grouped_polygon_buffer_shape(
        left_batch,
        left_state,
        left_polygon,
        row_count,
    )
    right_shape = _grouped_polygon_buffer_shape(
        right_batch,
        right_state,
        right_polygon,
        int(right_batch.row_count),
    )
    if left_shape is None or right_shape is None:
        return None
    left_structural_shape, left_sizing = left_shape
    right_structural_shape, right_sizing = right_shape
    capacity_sizing = left_sizing == "capacity" or right_sizing == "capacity"
    sizing_label = "capacity" if capacity_sizing else "structural"
    d_output_ring_counts = (d_left_ring_counts + d_group_counts).astype(
        cp.int64,
        copy=False,
    )
    d_geometry_offsets = cp.empty(row_count + 1, dtype=cp.int64)
    d_geometry_offsets[0] = 0
    cp.cumsum(d_output_ring_counts, out=d_geometry_offsets[1:])
    total_rings = int(left_structural_shape[0]) + int(right_batch.row_count)
    if total_rings <= 0:
        return None

    d_ring_positions = cp.arange(total_rings, dtype=cp.int64)
    d_logical_ring_total = d_geometry_offsets[-1]
    d_ring_active = d_ring_positions < d_logical_ring_total
    d_safe_ring_positions = cp.where(
        d_ring_active,
        d_ring_positions,
        cp.zeros_like(d_ring_positions),
    )
    d_ring_group_rows = cp.searchsorted(
        d_geometry_offsets[1:],
        d_safe_ring_positions,
        side="right",
    ).astype(cp.int64, copy=False)
    d_local_ring = d_safe_ring_positions - d_geometry_offsets[d_ring_group_rows]
    d_ring_is_hole = d_local_ring >= d_left_ring_counts[d_ring_group_rows]
    d_safe_left_local_ring = cp.minimum(
        d_local_ring,
        d_left_ring_counts[d_ring_group_rows] - 1,
    )
    d_ring_left_source = (d_left_ring_starts[d_ring_group_rows] + d_safe_left_local_ring).astype(
        cp.int64, copy=False
    )
    d_right_local = cp.maximum(
        d_local_ring - d_left_ring_counts[d_ring_group_rows],
        0,
    )
    d_ring_right_row = (d_offsets[d_ring_group_rows] + d_right_local).astype(cp.int64, copy=False)
    d_ring_right_source = d_right_ring_starts[d_ring_right_row].astype(
        cp.int64,
        copy=False,
    )

    d_left_ring_offsets = cp.asarray(left_polygon.ring_offsets, dtype=cp.int64)
    d_right_ring_offsets = cp.asarray(right_polygon.ring_offsets, dtype=cp.int64)
    d_left_all_lengths = (d_left_ring_offsets[1:] - d_left_ring_offsets[:-1]).astype(
        cp.int64, copy=False
    )
    d_right_exterior_lengths = (
        d_right_ring_offsets[d_right_ring_starts + 1] - d_right_ring_offsets[d_right_ring_starts]
    ).astype(cp.int64, copy=False)
    d_ring_lengths = cp.where(
        d_ring_is_hole,
        d_right_exterior_lengths[d_ring_right_row],
        d_left_all_lengths[d_ring_left_source],
    ).astype(cp.int64, copy=False)
    d_ring_lengths = cp.where(
        d_ring_active,
        d_ring_lengths,
        cp.zeros((), dtype=cp.int64),
    )
    d_ring_offsets = cp.empty(total_rings + 1, dtype=cp.int64)
    d_ring_offsets[0] = 0
    cp.cumsum(d_ring_lengths, out=d_ring_offsets[1:])
    total_coords = int(left_structural_shape[1]) + int(right_structural_shape[1])
    if total_coords <= 0:
        return None

    d_out_positions = cp.arange(total_coords, dtype=cp.int64)
    d_logical_coord_total = d_ring_offsets[-1]
    d_coord_active = d_out_positions < d_logical_coord_total
    d_safe_out_positions = cp.where(
        d_coord_active,
        d_out_positions,
        cp.zeros_like(d_out_positions),
    )
    d_ring_ids = cp.searchsorted(
        d_ring_offsets[1:],
        d_safe_out_positions,
        side="right",
    ).astype(cp.int64, copy=False)
    d_local = d_safe_out_positions - d_ring_offsets[d_ring_ids]
    d_coord_is_hole = d_ring_is_hole[d_ring_ids]
    d_coord_left_source = d_ring_left_source[d_ring_ids]
    d_coord_right_source = d_ring_right_source[d_ring_ids]
    d_left_coord_rows = (d_left_ring_offsets[d_coord_left_source] + d_local).astype(
        cp.int64, copy=False
    )
    d_right_coord_rows = (d_right_ring_offsets[d_coord_right_source + 1] - 1 - d_local).astype(
        cp.int64, copy=False
    )

    d_x = cp.empty(total_coords, dtype=cp.float64)
    d_y = cp.empty(total_coords, dtype=cp.float64)
    d_normal = d_coord_active & ~d_coord_is_hole
    d_hole_coord = d_coord_active & d_coord_is_hole
    d_x[d_normal] = cp.asarray(left_polygon.x, dtype=cp.float64)[d_left_coord_rows[d_normal]]
    d_y[d_normal] = cp.asarray(left_polygon.y, dtype=cp.float64)[d_left_coord_rows[d_normal]]
    d_x[d_hole_coord] = cp.asarray(right_polygon.x, dtype=cp.float64)[
        d_right_coord_rows[d_hole_coord]
    ]
    d_y[d_hole_coord] = cp.asarray(right_polygon.y, dtype=cp.float64)[
        d_right_coord_rows[d_hole_coord]
    ]

    d_geometry_offsets_i32 = d_geometry_offsets.astype(cp.int32, copy=False)
    d_ring_offsets_i32 = d_ring_offsets.astype(cp.int32, copy=False)
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=d_x,
                y=d_y,
                geometry_offsets=d_geometry_offsets_i32,
                empty_mask=cp.zeros(row_count, dtype=cp.bool_),
                ring_offsets=d_ring_offsets_i32,
                bounds=d_left_bounds,
            )
        },
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8),
        validity=cp.ones(row_count, dtype=cp.bool_),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    if result.device_state is not None:
        result.device_state.trusted_all_valid = True
        result.device_state.trusted_homogeneous_family = GeometryFamily.POLYGON
        result.device_state.trusted_all_non_empty = True
        result.device_state.row_bounds = d_left_bounds
    partition = _grouped_direct_difference_capacity_partition(
        result,
        d_supported_groups,
        collective_mask=d_collective_groups,
    )
    if partition is None:
        return None
    record_dispatch_event(
        surface="geopandas.array.difference",
        operation="difference",
        implementation=event_implementation,
        reason=event_reason
        or (
            "grouped exact overlay difference emitted polygon exterior and "
            "interior rings directly from NativeGrouped device buffers"
        ),
        detail=(
            f"groups={left_batch.row_count}, pairs={right_batch.row_count}, "
            f"group_size_bound={_GROUPED_RECTANGLE_HOLE_DIFF_MAX_GROUP_SIZE}, "
            f"sizing={sizing_label}"
            f"{event_detail_extra}"
        )
        if event_pairs is None
        else (
            f"groups={left_batch.row_count}, pairs={event_pairs}, "
            f"group_size_bound={_GROUPED_RECTANGLE_HOLE_DIFF_MAX_GROUP_SIZE}, "
            f"sizing={sizing_label}"
            f"{event_detail_extra}"
        ),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return partition


def _grouped_polygon_donut_difference_owned(
    left_batch,
    right_batch,
    grouped: NativeGrouped,
    group_offsets,
    *,
    dispatch_mode: ExecutionMode,
    event_implementation: str = "grouped_overlay_difference_polygon_donuts_gpu",
    event_reason: str | None = None,
    event_pairs: int | None = None,
    event_detail_extra: str = "",
):
    """Native grouped polygon minus contained holed polygon rows.

    Physical shape: dense `NativeGrouped` offsets with one polygon left row per
    group and one or more contained right polygon rows. Each supported right
    row has one exterior and one or more interior rings. The output is one
    MultiPolygon per group: the main polygon is the single-ring left shell with
    right exteriors cut as holes, and every right interior ring is emitted as a
    retained island polygon.
    """
    if cp is None or not grouped.is_device:
        return None
    if grouped.resolved_group_count != int(left_batch.row_count):
        return None
    if right_batch.row_count <= 0:
        return left_batch

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
    from vibespatial.predicates.binary import binary_predicate_expression

    left_state = left_batch._ensure_device_state(preserve_indexed_view=True)
    right_state = right_batch._ensure_device_state(preserve_indexed_view=True)
    left_polygon = left_state.families.get(GeometryFamily.POLYGON)
    right_polygon = right_state.families.get(GeometryFamily.POLYGON)
    if left_polygon is None or right_polygon is None:
        return None
    if left_polygon.ring_offsets is None or right_polygon.ring_offsets is None:
        return None

    d_offsets = cp.asarray(group_offsets, dtype=cp.int64)
    row_count = int(left_batch.row_count)
    if int(d_offsets.size) != row_count + 1:
        return None
    d_group_counts = (d_offsets[1:] - d_offsets[:-1]).astype(cp.int64, copy=False)

    d_right_group_rows = _native_grouped_source_rows(
        grouped,
        total_count=right_batch.row_count,
    )
    d_right_group_rows = cp.asarray(d_right_group_rows, dtype=cp.int64)
    if int(d_right_group_rows.size) != int(right_batch.row_count):
        return None

    d_left_rows = cp.asarray(left_state.family_row_offsets, dtype=cp.int64)
    d_right_rows = cp.asarray(right_state.family_row_offsets, dtype=cp.int64)
    d_left_valid = cp.asarray(left_state.validity, dtype=cp.bool_)
    d_right_valid = cp.asarray(right_state.validity, dtype=cp.bool_)
    d_left_tags = cp.asarray(left_state.tags, dtype=cp.int8)
    d_right_tags = cp.asarray(right_state.tags, dtype=cp.int8)
    polygon_tag = cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_left_polygon_rows = d_left_valid & (d_left_tags == polygon_tag) & (d_left_rows >= 0)
    d_right_polygon_rows = d_right_valid & (d_right_tags == polygon_tag) & (d_right_rows >= 0)
    d_safe_left_rows = cp.where(d_left_polygon_rows, d_left_rows, cp.int64(0))
    d_safe_right_rows = cp.where(d_right_polygon_rows, d_right_rows, cp.int64(0))
    d_left_geom_offsets = cp.asarray(left_polygon.geometry_offsets, dtype=cp.int64)
    d_right_geom_offsets = cp.asarray(right_polygon.geometry_offsets, dtype=cp.int64)
    d_left_ring_starts = d_left_geom_offsets[d_safe_left_rows].astype(cp.int64, copy=False)
    d_left_ring_ends = d_left_geom_offsets[d_safe_left_rows + 1].astype(
        cp.int64,
        copy=False,
    )
    d_left_ring_counts = (d_left_ring_ends - d_left_ring_starts).astype(
        cp.int64,
        copy=False,
    )
    d_left_ring_counts = cp.where(d_left_polygon_rows, d_left_ring_counts, cp.int64(0))
    d_right_ring_starts = d_right_geom_offsets[d_safe_right_rows].astype(
        cp.int64,
        copy=False,
    )
    d_right_ring_ends = d_right_geom_offsets[d_safe_right_rows + 1].astype(
        cp.int64,
        copy=False,
    )
    d_right_ring_counts = (d_right_ring_ends - d_right_ring_starts).astype(
        cp.int64,
        copy=False,
    )
    d_right_ring_counts = cp.where(
        d_right_polygon_rows,
        d_right_ring_counts,
        cp.int64(0),
    )
    d_right_hole_counts = cp.maximum(
        d_right_ring_counts - cp.int64(1),
        cp.int64(0),
    ).astype(cp.int64, copy=False)
    d_group_hole_counts_i32 = cp.zeros(row_count, dtype=cp.int32)
    cp.add.at(
        d_group_hole_counts_i32,
        d_right_group_rows,
        d_right_hole_counts.astype(cp.int32, copy=False),
    )
    d_group_hole_counts = d_group_hole_counts_i32.astype(cp.int64, copy=False)

    d_left_bounds = cp.asarray(
        compute_geometry_bounds_device(left_batch, preserve_indexed_view=True),
        dtype=cp.float64,
    ).reshape(row_count, 4)
    d_right_bounds = cp.asarray(
        compute_geometry_bounds_device(right_batch, preserve_indexed_view=True),
        dtype=cp.float64,
    ).reshape(right_batch.row_count, 4)
    d_pair_left_bounds = d_left_bounds[d_right_group_rows]
    scale = cp.maximum(
        cp.maximum(
            cp.max(cp.abs(d_pair_left_bounds), axis=1),
            cp.max(cp.abs(d_right_bounds), axis=1),
        ),
        1.0,
    )
    tol = cp.maximum(scale * 1.0e-12, 1.0e-12)
    d_strict_bbox_inside = (
        (d_right_bounds[:, 0] > d_pair_left_bounds[:, 0] + tol)
        & (d_right_bounds[:, 1] > d_pair_left_bounds[:, 1] + tol)
        & (d_right_bounds[:, 2] < d_pair_left_bounds[:, 2] - tol)
        & (d_right_bounds[:, 3] < d_pair_left_bounds[:, 3] - tol)
    )

    pair_left = _device_take_preserving_indexed_rows(left_batch, d_right_group_rows)
    coverage = binary_predicate_expression(
        "covers",
        pair_left,
        right_batch,
        dispatch_mode=ExecutionMode.GPU,
        operation="overlay.grouped_difference.polygon_donut_admission",
    )
    if coverage is None:
        return None
    d_covered = cp.asarray(coverage.values, dtype=cp.bool_)
    if int(d_covered.size) != int(right_batch.row_count):
        return None

    d_pair_left_ring_counts = d_left_ring_counts[d_right_group_rows]
    d_row_supported = (
        d_left_polygon_rows[d_right_group_rows]
        & d_right_polygon_rows
        & (d_pair_left_ring_counts == 1)
        & (d_right_ring_counts >= 2)
        & d_strict_bbox_inside
        & d_covered
    )

    runtime = get_cuda_runtime()
    kernels = _grouped_rectangle_hole_difference_kernels()
    ptr = runtime.pointer
    d_supported_groups = cp.zeros(row_count, dtype=cp.bool_)
    d_row_supported_u8 = d_row_supported.astype(cp.uint8, copy=False)
    validate = kernels["validate_grouped_polygon_holes"]
    grid, block = runtime.launch_config(validate, row_count)
    runtime.launch(
        validate,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_row_supported_u8),
                ptr(d_right_bounds),
                ptr(d_offsets),
                row_count,
                _GROUPED_RECTANGLE_HOLE_DIFF_MAX_GROUP_SIZE,
                ptr(d_supported_groups),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
            ),
        ),
    )
    left_shape = _grouped_polygon_buffer_shape(
        left_batch,
        left_state,
        left_polygon,
        row_count,
    )
    right_shape = _grouped_polygon_buffer_shape(
        right_batch,
        right_state,
        right_polygon,
        int(right_batch.row_count),
    )
    if left_shape is None or right_shape is None:
        return None
    left_structural_shape, left_sizing = left_shape
    right_structural_shape, right_sizing = right_shape
    sizing_label = (
        "capacity" if left_sizing == "capacity" or right_sizing == "capacity" else "structural"
    )

    right_hole_capacity = max(
        int(right_structural_shape[0]) - int(right_batch.row_count),
        0,
    )
    if right_hole_capacity <= 0:
        return None

    d_output_part_counts = (1 + d_group_hole_counts).astype(cp.int64, copy=False)
    d_geometry_offsets = cp.empty(row_count + 1, dtype=cp.int64)
    d_geometry_offsets[0] = 0
    cp.cumsum(d_output_part_counts, out=d_geometry_offsets[1:])
    total_parts = row_count + right_hole_capacity
    if total_parts <= 0:
        return None

    d_right_hole_offsets = cp.empty(right_batch.row_count + 1, dtype=cp.int64)
    d_right_hole_offsets[0] = 0
    cp.cumsum(d_right_hole_counts, out=d_right_hole_offsets[1:])
    d_hole_positions = cp.arange(right_hole_capacity, dtype=cp.int64)
    d_hole_active = d_hole_positions < d_right_hole_offsets[-1]
    d_safe_hole_positions = cp.where(
        d_hole_active,
        d_hole_positions,
        cp.zeros_like(d_hole_positions),
    )
    d_hole_right_rows = cp.searchsorted(
        d_right_hole_offsets[1:],
        d_safe_hole_positions,
        side="right",
    ).astype(cp.int64, copy=False)
    d_hole_local = (d_safe_hole_positions - d_right_hole_offsets[d_hole_right_rows]).astype(
        cp.int64, copy=False
    )
    d_hole_ring_sources = (
        d_right_ring_starts[d_hole_right_rows] + cp.int64(1) + d_hole_local
    ).astype(cp.int64, copy=False)

    d_group_hole_offsets = cp.empty(row_count + 1, dtype=cp.int64)
    d_group_hole_offsets[0] = 0
    cp.cumsum(d_group_hole_counts, out=d_group_hole_offsets[1:])

    d_part_positions = cp.arange(total_parts, dtype=cp.int64)
    d_part_group_rows = cp.searchsorted(
        d_geometry_offsets[1:],
        d_part_positions,
        side="right",
    ).astype(cp.int64, copy=False)
    d_local_part = d_part_positions - d_geometry_offsets[d_part_group_rows]
    d_part_is_main = d_local_part == 0
    d_part_ring_counts = cp.where(
        d_part_is_main,
        d_left_ring_counts[d_part_group_rows] + d_group_counts[d_part_group_rows],
        cp.ones((), dtype=cp.int64),
    ).astype(cp.int64, copy=False)
    d_part_offsets = cp.empty(total_parts + 1, dtype=cp.int64)
    d_part_offsets[0] = 0
    cp.cumsum(d_part_ring_counts, out=d_part_offsets[1:])

    total_rings = int(left_structural_shape[0]) + int(right_structural_shape[0])
    if total_rings <= 0:
        return None

    d_ring_positions = cp.arange(total_rings, dtype=cp.int64)
    d_logical_ring_total = d_part_offsets[-1]
    d_ring_active = d_ring_positions < d_logical_ring_total
    d_safe_ring_positions = cp.where(
        d_ring_active,
        d_ring_positions,
        cp.zeros_like(d_ring_positions),
    )
    d_ring_part_rows = cp.searchsorted(
        d_part_offsets[1:],
        d_safe_ring_positions,
        side="right",
    ).astype(cp.int64, copy=False)
    d_ring_group_rows = d_part_group_rows[d_ring_part_rows]
    d_ring_local_part = d_local_part[d_ring_part_rows]
    d_ring_local_in_part = d_safe_ring_positions - d_part_offsets[d_ring_part_rows]
    d_ring_is_main_part = d_ring_local_part == 0
    d_ring_is_left = d_ring_is_main_part & (
        d_ring_local_in_part < d_left_ring_counts[d_ring_group_rows]
    )
    d_safe_left_local_ring = cp.minimum(
        d_ring_local_in_part,
        d_left_ring_counts[d_ring_group_rows] - 1,
    )
    d_ring_left_source = (d_left_ring_starts[d_ring_group_rows] + d_safe_left_local_ring).astype(
        cp.int64, copy=False
    )
    d_right_local_from_main = cp.maximum(
        d_ring_local_in_part - d_left_ring_counts[d_ring_group_rows],
        0,
    )
    d_hole_position_from_island = (
        d_group_hole_offsets[d_ring_group_rows] + cp.maximum(d_ring_local_part - 1, 0)
    ).astype(cp.int64, copy=False)
    d_safe_hole_position_from_island = cp.minimum(
        d_hole_position_from_island,
        cp.int64(right_hole_capacity - 1),
    )
    d_ring_right_row = cp.where(
        d_ring_is_main_part,
        d_offsets[d_ring_group_rows] + d_right_local_from_main,
        cp.zeros_like(d_ring_group_rows),
    ).astype(cp.int64, copy=False)
    d_ring_right_source = cp.where(
        d_ring_is_main_part,
        d_right_ring_starts[d_ring_right_row],
        d_hole_ring_sources[d_safe_hole_position_from_island],
    ).astype(cp.int64, copy=False)

    d_left_ring_offsets = cp.asarray(left_polygon.ring_offsets, dtype=cp.int64)
    d_right_ring_offsets = cp.asarray(right_polygon.ring_offsets, dtype=cp.int64)
    d_left_all_lengths = (d_left_ring_offsets[1:] - d_left_ring_offsets[:-1]).astype(
        cp.int64, copy=False
    )
    d_right_all_lengths = (d_right_ring_offsets[1:] - d_right_ring_offsets[:-1]).astype(
        cp.int64, copy=False
    )
    d_ring_lengths = cp.where(
        d_ring_is_left,
        d_left_all_lengths[d_ring_left_source],
        d_right_all_lengths[d_ring_right_source],
    ).astype(cp.int64, copy=False)
    d_ring_lengths = cp.where(
        d_ring_active,
        d_ring_lengths,
        cp.zeros((), dtype=cp.int64),
    )
    d_ring_offsets = cp.empty(total_rings + 1, dtype=cp.int64)
    d_ring_offsets[0] = 0
    cp.cumsum(d_ring_lengths, out=d_ring_offsets[1:])

    total_coords = int(left_structural_shape[1]) + int(right_structural_shape[1])
    if total_coords <= 0:
        return None

    d_out_positions = cp.arange(total_coords, dtype=cp.int64)
    d_logical_coord_total = d_ring_offsets[-1]
    d_coord_active = d_out_positions < d_logical_coord_total
    d_safe_out_positions = cp.where(
        d_coord_active,
        d_out_positions,
        cp.zeros_like(d_out_positions),
    )
    d_coord_ring_rows = cp.searchsorted(
        d_ring_offsets[1:],
        d_safe_out_positions,
        side="right",
    ).astype(cp.int64, copy=False)
    d_local = d_safe_out_positions - d_ring_offsets[d_coord_ring_rows]
    d_coord_is_left = d_ring_is_left[d_coord_ring_rows]
    d_coord_left_source = d_ring_left_source[d_coord_ring_rows]
    d_coord_right_source = d_ring_right_source[d_coord_ring_rows]
    d_left_coord_rows = (d_left_ring_offsets[d_coord_left_source] + d_local).astype(
        cp.int64, copy=False
    )
    d_right_coord_rows = (d_right_ring_offsets[d_coord_right_source + 1] - 1 - d_local).astype(
        cp.int64, copy=False
    )

    d_x = cp.empty(total_coords, dtype=cp.float64)
    d_y = cp.empty(total_coords, dtype=cp.float64)
    d_left_coord = d_coord_active & d_coord_is_left
    d_right_coord = d_coord_active & ~d_coord_is_left
    d_x[d_left_coord] = cp.asarray(left_polygon.x, dtype=cp.float64)[
        d_left_coord_rows[d_left_coord]
    ]
    d_y[d_left_coord] = cp.asarray(left_polygon.y, dtype=cp.float64)[
        d_left_coord_rows[d_left_coord]
    ]
    d_x[d_right_coord] = cp.asarray(right_polygon.x, dtype=cp.float64)[
        d_right_coord_rows[d_right_coord]
    ]
    d_y[d_right_coord] = cp.asarray(right_polygon.y, dtype=cp.float64)[
        d_right_coord_rows[d_right_coord]
    ]

    result = build_device_resident_owned(
        device_families={
            GeometryFamily.MULTIPOLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.MULTIPOLYGON,
                x=d_x,
                y=d_y,
                geometry_offsets=d_geometry_offsets.astype(cp.int32, copy=False),
                empty_mask=cp.zeros(row_count, dtype=cp.bool_),
                part_offsets=d_part_offsets.astype(cp.int32, copy=False),
                ring_offsets=d_ring_offsets.astype(cp.int32, copy=False),
                bounds=d_left_bounds,
            )
        },
        row_count=row_count,
        tags=cp.full(
            row_count,
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            dtype=cp.int8,
        ),
        validity=cp.ones(row_count, dtype=cp.bool_),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    if result.device_state is not None:
        result.device_state.trusted_all_valid = True
        result.device_state.trusted_homogeneous_family = GeometryFamily.MULTIPOLYGON
        result.device_state.trusted_all_non_empty = True
        result.device_state.row_bounds = d_left_bounds
    partition = _grouped_direct_difference_capacity_partition(
        result,
        d_supported_groups,
    )
    if partition is None:
        return None
    record_dispatch_event(
        surface="geopandas.array.difference",
        operation="difference",
        implementation=event_implementation,
        reason=event_reason
        or (
            "grouped exact overlay difference emitted contained holed right "
            "polygons as MultiPolygon main parts plus retained islands"
        ),
        detail=(
            f"groups={left_batch.row_count}, pairs={right_batch.row_count}, "
            f"group_size_bound={_GROUPED_RECTANGLE_HOLE_DIFF_MAX_GROUP_SIZE}, "
            f"sizing={sizing_label}"
            f"{event_detail_extra}"
        )
        if event_pairs is None
        else (
            f"groups={left_batch.row_count}, pairs={event_pairs}, "
            f"group_size_bound={_GROUPED_RECTANGLE_HOLE_DIFF_MAX_GROUP_SIZE}, "
            f"sizing={sizing_label}"
            f"{event_detail_extra}"
        ),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return partition


def _grouped_overlay_difference_owned(
    left_batch,
    right_batch,
    group_offsets,
    *,
    dispatch_mode: ExecutionMode,
    _native_grouped: NativeGrouped | None = None,
    _all_groups_observed: bool | None = None,
    _group_size_min: int | None = None,
    _group_size_max: int | None = None,
    _skip_containment_union: bool = False,
    _skip_direct_specializations: bool = False,
):
    """Compute grouped exact difference from one grouped overlay execution plan.

    The grouped workload shape is:
    - one left geometry row per group
    - many right geometry rows packed together

    The overlay planner already supports logical row isolation. By remapping
    every right geometry row to its owning left-group id, the existing
    same-row split, graph, and face-labeling pipeline becomes a true grouped
    exact-difference executor without per-pair replanning.
    """
    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    def _selected_grouped_difference_mode() -> ExecutionMode:
        requested = (
            dispatch_mode
            if isinstance(dispatch_mode, ExecutionMode)
            else ExecutionMode(dispatch_mode)
        )
        return (
            ExecutionMode.GPU
            if requested is not ExecutionMode.CPU and has_gpu_runtime()
            else ExecutionMode.CPU
        )

    selected_mode = _selected_grouped_difference_mode()
    if selected_mode is ExecutionMode.CPU:
        host_offsets = _group_offsets_to_host(
            group_offsets,
            reason="overlay grouped difference explicit CPU offsets export",
        )
        _host_offsets, _group_lengths, cpu_max_group_size, cpu_n_groups = _group_offsets_metadata(
            host_offsets, total_rows=right_batch.row_count
        )
        if cpu_max_group_size <= 0:
            return left_batch
        cpu_single_pair_aligned = (
            cpu_max_group_size <= 1
            and cpu_n_groups == left_batch.row_count
            and right_batch.row_count == left_batch.row_count
        )
        if cpu_single_pair_aligned:
            record_dispatch_event(
                surface="geopandas.array.difference",
                operation="difference",
                implementation="grouped_overlay_difference_single_pair_cpu",
                reason=(
                    "explicit CPU grouped exact overlay difference reduced to "
                    "the rowwise exact difference path because each group had "
                    "at most one candidate"
                ),
                detail=(
                    f"groups={left_batch.row_count}, "
                    f"pairs={right_batch.row_count}, "
                    f"max_group_size={cpu_max_group_size}, "
                    "aligned=True"
                ),
                requested=dispatch_mode,
                selected=ExecutionMode.CPU,
            )
            return _single_pair_grouped_difference_owned(
                left_batch,
                right_batch,
                dispatch_mode=ExecutionMode.CPU,
            )
        return _cpu_grouped_difference_owned(
            left_batch,
            right_batch,
            host_offsets,
            dispatch_mode=ExecutionMode.CPU,
        )

    with hotpath_stage(
        "overlay.diff.group_metadata",
        category="setup",
    ) as amplification_metadata:
        if not _is_device_array(group_offsets):
            host_offsets = np.asarray(group_offsets, dtype=np.int64)
            if host_offsets.ndim != 1 or host_offsets.size == 0:
                raise ValueError("group_offsets must be a 1D array with length >= 1")
            host_counts = np.diff(host_offsets).astype(np.int64, copy=False)
            if np.any(host_counts < 0):
                raise ValueError("group_offsets must be monotonically nondecreasing")
            if _all_groups_observed is None:
                _all_groups_observed = bool(np.all(host_counts > 0))
            if _group_size_min is None:
                _group_size_min = int(host_counts.min(initial=0))
            if _group_size_max is None:
                _group_size_max = int(host_counts.max(initial=0))
        elif right_batch.row_count == 0 and _group_size_max is None:
            _group_size_min = 0 if _group_size_min is None else _group_size_min
            _group_size_max = 0

        grouped = (
            _native_grouped
            if _native_grouped is not None
            else _native_grouped_from_sorted_offsets(
                group_offsets,
                row_count=right_batch.row_count,
                force_device=True,
                all_groups_observed=_all_groups_observed,
                group_size_min=_group_size_min,
                group_size_max=_group_size_max,
            )
        )
        max_group_size = _native_grouped_max_group_size(grouped)
        n_groups = grouped.resolved_group_count
        observed_group_count = 0 if grouped.group_ids is None else int(grouped.group_ids.size)
        fixed_group_size = _native_grouped_fixed_group_size(
            grouped,
            row_count=right_batch.row_count,
        )
        if max_group_size is None and fixed_group_size is not None:
            max_group_size = fixed_group_size
        if max_group_size is None:
            if right_batch.row_count == 0:
                max_group_size = 0
            elif (
                grouped.all_groups_observed is True
                and right_batch.row_count == n_groups
                and observed_group_count == n_groups
            ):
                max_group_size = 1
        if fixed_group_size is not None and (
            grouped.group_size_min != fixed_group_size or grouped.group_size_max != fixed_group_size
        ):
            from dataclasses import replace

            grouped = replace(
                grouped,
                group_size_min=fixed_group_size,
                group_size_max=fixed_group_size,
            )
        elif max_group_size is not None and grouped.group_size_max is None:
            from dataclasses import replace

            grouped = replace(grouped, group_size_max=max_group_size)
        same_row_span_summary = _grouped_difference_same_row_span_summary(
            left_batch,
            right_batch,
            group_offsets,
            max_group_size=max_group_size,
        )
        if amplification_metadata is not None:
            group_unavailable = [
                "input_segments",
                "input_coordinates",
                "pre_reduction_fragments",
                "output_parts",
                "output_coordinates",
            ]
            group_maxima = {}
            if max_group_size is None:
                group_unavailable.append("max_group_size")
            else:
                group_maxima["max_group_size"] = int(max_group_size)
            attach_work_amplification(
                amplification_metadata,
                operation="overlay.diff.group_metadata",
                metric_family="group_compression",
                sums={
                    "input_rows": int(right_batch.row_count),
                    "output_groups": int(n_groups),
                    "observed_groups": int(observed_group_count),
                },
                maxima=group_maxima,
                unavailable=tuple(group_unavailable),
            )

    def _decline_native(stage: str, exc: Exception | str) -> None:
        detail_error = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        record_dispatch_event(
            surface="geopandas.array.difference",
            operation="difference",
            implementation=f"grouped_overlay_difference_{stage}_declined_gpu",
            reason=(
                "native grouped overlay difference declined without entering a "
                "host-shaped exact fallback"
            ),
            detail=(
                f"groups={left_batch.row_count}, "
                f"pairs={right_batch.row_count}, "
                f"error={detail_error}"
            ),
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
        )
        raise _GroupedOverlayDifferenceNativeDeclined(detail_error)

    def _try_native_grouped_union_difference(stage: str):
        return _native_grouped_union_difference_owned(
            left_batch,
            right_batch,
            grouped,
            dispatch_mode=ExecutionMode.GPU,
            stage=stage,
        )

    def _grouped_union_difference_or_decline(stage: str):
        result = _try_native_grouped_union_difference(stage)
        if result is None:
            _decline_native(stage, "NativeGrouped union difference executor declined")
        return result

    def _row_indirected_grouped_topology_inputs():
        """Give mutable grouped topology independently owned device buffers."""
        left_was_indexed = bool(getattr(left_batch, "is_indexed_view", False))
        right_was_indexed = bool(getattr(right_batch, "is_indexed_view", False))
        if left_was_indexed or right_was_indexed:
            from vibespatial.geometry.owned import (
                build_null_owned_array,
                device_physicalize_owned_row_selections_exact,
            )
            from vibespatial.runtime.residency import Residency

            sources = (left_batch, right_batch)
            indexed_positions = tuple(
                index
                for index, is_indexed in enumerate((left_was_indexed, right_was_indexed))
                if is_indexed
            )
            exact_inputs = device_physicalize_owned_row_selections_exact(
                [
                    (
                        sources[index],
                        cp.ones(sources[index].row_count, dtype=cp.bool_),
                    )
                    for index in indexed_positions
                ],
                reason="grouped overlay topology input exact-allocation packet",
            )
            topology_inputs = list(sources)
            for index, physicalized in zip(
                indexed_positions,
                exact_inputs,
                strict=True,
            ):
                topology_inputs[index] = (
                    build_null_owned_array(
                        sources[index].row_count,
                        residency=Residency.DEVICE,
                    )
                    if physicalized is None
                    else physicalized
                )
            topology_left, topology_right = topology_inputs
            record_dispatch_event(
                surface="geopandas.array.difference",
                operation="difference",
                implementation="grouped_overlay_difference_topology_physicalization_gpu",
                reason=(
                    "grouped topology canonicalization uses mutable coordinate "
                    "workspaces, so indexed gathers received independent device buffers"
                ),
                detail=(
                    f"groups={left_batch.row_count}, "
                    f"pairs={right_batch.row_count}, "
                    f"left_indexed={left_was_indexed}, "
                    f"right_indexed={right_was_indexed}"
                ),
                requested=dispatch_mode,
                selected=ExecutionMode.GPU,
            )
            return topology_left, topology_right
        return left_batch, right_batch

    def _left_covered_group_mask():
        """Return device ownership for groups erased by any right row."""
        if _skip_containment_union or right_batch.row_count == 0:
            return None
        if not (
            set(left_batch.families).issubset(polygonal_families)
            and set(right_batch.families).issubset(polygonal_families)
        ):
            return None
        right_work = estimate_physical_work_from_owned(right_batch)
        right_segments_per_row = (
            int(right_work.segment_count) // max(int(right_batch.row_count), 1)
        )
        if (
            right_segments_per_row
            > _GROUPED_DIFFERENCE_CONTAINMENT_MAX_RIGHT_SEGMENTS_PER_ROW
        ):
            record_dispatch_event(
                surface="geopandas.array.difference",
                operation="difference",
                implementation="grouped_overlay_difference_parallel_topology_gpu",
                reason=(
                    "grouped difference skipped serial containment pruning for "
                    "detailed right boundaries and retained the parallel topology plan"
                ),
                detail=(
                    f"groups={left_batch.row_count}, pairs={right_batch.row_count}, "
                    f"right_segments_per_row={right_segments_per_row}"
                ),
                requested=dispatch_mode,
                selected=ExecutionMode.GPU,
            )
            return None
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by GPU dispatch
            return None

        from vibespatial.predicates.binary import binary_predicate_expressions
        from vibespatial.runtime.residency import Residency, TransferTrigger

        try:
            if left_batch.residency is not Residency.DEVICE:
                left_batch.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="grouped difference left-covered prune left rows",
                )
            if right_batch.residency is not Residency.DEVICE:
                right_batch.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="grouped difference left-covered prune right rows",
                )
            right_group_rows = _native_grouped_source_rows(
                grouped,
                total_count=right_batch.row_count,
            )
            right_group_rows = cp.asarray(right_group_rows, dtype=cp.int64)
            if int(right_group_rows.size) != int(right_batch.row_count):
                return None

            pair_left = _device_take_preserving_indexed_rows(left_batch, right_group_rows)
            expressions = binary_predicate_expressions(
                ("covered_by",),
                pair_left,
                right_batch,
                dispatch_mode=ExecutionMode.GPU,
                operation_prefix="overlay.grouped_difference.containment",
            )
            if expressions is None:
                return None
            d_left_covered = cp.asarray(
                expressions["covered_by"].values,
                dtype=cp.bool_,
            )
            if int(d_left_covered.size) != int(right_batch.row_count):
                return None

            group_count = int(left_batch.row_count)
            pair_count = int(right_batch.row_count)
            d_pair_lanes = cp.arange(pair_count, dtype=cp.int64)
            d_covered_destinations = cp.where(
                d_left_covered,
                right_group_rows,
                np.int64(group_count) + d_pair_lanes,
            )
            d_covered_extended = cp.zeros(
                group_count + pair_count,
                dtype=cp.bool_,
            )
            d_covered_extended[d_covered_destinations] = True
            d_covered_mask = d_covered_extended[:group_count]
            record_dispatch_event(
                surface="geopandas.array.difference",
                operation="difference",
                implementation="grouped_overlay_difference_left_covered_prune_gpu",
                reason=(
                    "grouped exact overlay difference assigned groups exactly "
                    "covered by right rows to the valid-empty capacity owner"
                ),
                detail=(
                    f"groups={left_batch.row_count}, "
                    f"pairs={right_batch.row_count}, "
                    "covered_groups=device-resident, "
                    "ownership=device-resident"
                ),
                requested=dispatch_mode,
                selected=ExecutionMode.GPU,
            )
            return d_covered_mask, right_group_rows
        except _GroupedOverlayDifferenceNativeDeclined:
            raise

    if max_group_size is not None and max_group_size <= 0:
        return left_batch
    if max_group_size is not None and max_group_size <= 1:
        single_pair_aligned = (
            n_groups == left_batch.row_count
            and right_batch.row_count == left_batch.row_count
            and observed_group_count == left_batch.row_count
        )
        record_dispatch_event(
            surface="geopandas.array.difference",
            operation="difference",
            implementation="grouped_overlay_difference_single_pair_gpu",
            reason=(
                "grouped exact overlay difference reduced to the rowwise exact difference "
                "path because each group had at most one candidate"
            ),
            detail=(
                f"groups={left_batch.row_count}, "
                f"pairs={right_batch.row_count}, "
                f"max_group_size={max_group_size}, "
                f"aligned={single_pair_aligned}"
            ),
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
        )
        if single_pair_aligned:
            return _single_pair_grouped_difference_owned(
                left_batch,
                right_batch,
                dispatch_mode=ExecutionMode.GPU,
            )
        return _sparse_single_pair_grouped_difference_owned(
            left_batch,
            right_batch,
            grouped,
            dispatch_mode=ExecutionMode.GPU,
        )

    from vibespatial.geometry.buffers import GeometryFamily

    polygonal_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    lineal_families = {GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING}
    if set(left_batch.families).issubset(lineal_families) and set(right_batch.families).issubset(
        polygonal_families
    ):
        return _grouped_union_difference_or_decline("mixed-dimensional")
    if set(left_batch.families).issubset(polygonal_families) and set(right_batch.families).issubset(
        polygonal_families
    ):
        if not _skip_direct_specializations:
            covered_group_ownership = _left_covered_group_mask()
            direct_partitions = []
            for direct_partition in (
                _grouped_rectangle_hole_difference_owned(
                    left_batch,
                    right_batch,
                    grouped,
                    group_offsets,
                    dispatch_mode=dispatch_mode,
                ),
                _grouped_polygon_donut_difference_owned(
                    left_batch,
                    right_batch,
                    grouped,
                    group_offsets,
                    dispatch_mode=dispatch_mode,
                ),
                _grouped_polygon_hole_difference_owned(
                    left_batch,
                    right_batch,
                    grouped,
                    group_offsets,
                    dispatch_mode=dispatch_mode,
                ),
            ):
                if direct_partition is not None:
                    direct_partitions.append(direct_partition)
            if direct_partitions or covered_group_ownership is not None:
                from vibespatial.geometry.owned import (
                    build_empty_polygon_rows_device,
                    device_mask_owned_capacity,
                    device_select_owned_capacity_partitions,
                )

                d_claimed = cp.zeros(left_batch.row_count, dtype=cp.bool_)
                direct_replacements = []
                if covered_group_ownership is not None:
                    d_covered_groups, right_group_rows = covered_group_ownership
                    d_covered_groups = cp.asarray(
                        d_covered_groups,
                        dtype=cp.bool_,
                    )
                    direct_replacements.append(
                        (
                            build_empty_polygon_rows_device(left_batch.row_count),
                            d_covered_groups,
                        )
                    )
                    d_claimed |= d_covered_groups
                else:
                    right_group_rows = cp.asarray(
                        _native_grouped_source_rows(
                            grouped,
                            total_count=right_batch.row_count,
                        ),
                        dtype=cp.int64,
                    )
                for partition in direct_partitions:
                    d_partition = (
                        cp.asarray(
                            partition.support_mask,
                            dtype=cp.bool_,
                        )
                        & ~d_claimed
                    )
                    direct_replacements.append((partition.owned, d_partition))
                    d_claimed |= d_partition
                d_exact_groups = ~d_claimed
                topology_left = device_mask_owned_capacity(
                    left_batch,
                    d_exact_groups,
                )
                topology_right = device_mask_owned_capacity(
                    right_batch,
                    d_exact_groups[right_group_rows],
                )
                exact_result = _grouped_overlay_difference_owned(
                    topology_left,
                    topology_right,
                    group_offsets,
                    dispatch_mode=ExecutionMode.GPU,
                    _native_grouped=grouped,
                    _all_groups_observed=False,
                    _group_size_min=0,
                    _group_size_max=max_group_size,
                    _skip_containment_union=True,
                    _skip_direct_specializations=True,
                )
                if exact_result is None:
                    _decline_native(
                        "direct-capacity-partition",
                        "complementary exact topology declined",
                    )
                if exact_result.row_count != left_batch.row_count:
                    _decline_native(
                        "direct-capacity-partition",
                        "complementary exact topology returned the wrong row capacity",
                    )
                result = device_select_owned_capacity_partitions(
                    exact_result,
                    direct_replacements,
                )
                record_dispatch_event(
                    surface="geopandas.array.difference",
                    operation="difference",
                    implementation=("grouped_overlay_difference_direct_capacity_partition_gpu"),
                    reason=(
                        "grouped exact difference kept rectangle-hole, polygon-hole, "
                        "polygon-donut, and exact topology rows in complementary "
                        "row-capacity carriers"
                    ),
                    detail=(
                        f"groups={left_batch.row_count}, pairs={right_batch.row_count}, "
                        f"direct_partitions={len(direct_partitions)}"
                    ),
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                )
                return result

    try:
        topology_left_batch, topology_right_batch = _row_indirected_grouped_topology_inputs()
        with hotpath_stage(
            "overlay.diff.group_rows.expand",
            category="setup",
        ) as amplification_metadata:
            right_group_rows = _native_grouped_source_rows(
                grouped,
                total_count=topology_right_batch.row_count,
            )
            if amplification_metadata is not None:
                maxima = {}
                unavailable = [
                    "input_segments",
                    "input_coordinates",
                    "pre_reduction_fragments",
                    "output_parts",
                    "output_coordinates",
                ]
                if max_group_size is None:
                    unavailable.append("max_group_size")
                else:
                    maxima["max_group_size"] = int(max_group_size)
                attach_work_amplification(
                    amplification_metadata,
                    operation="overlay.diff.group_rows.expand",
                    metric_family="group_compression",
                    sums={
                        "input_rows": int(topology_right_batch.row_count),
                        "output_groups": int(topology_left_batch.row_count),
                    },
                    maxima=maxima,
                    unavailable=tuple(unavailable),
                )
        _sync_hotpath()
        try:
            with hotpath_stage(
                "overlay.diff.grouped_plan.build",
                category="setup",
            ) as amplification_metadata:
                plan = _build_overlay_execution_plan(
                    topology_left_batch,
                    topology_right_batch,
                    dispatch_mode=dispatch_mode,
                    _cached_right_segments=None,
                    _row_isolated=True,
                    _use_same_row_fast_path=same_row_span_summary is not None,
                    _same_row_span_summary=same_row_span_summary,
                    _include_same_side_splits=True,
                    _right_geometry_source_rows=right_group_rows,
                    _right_segment_source_rows=right_group_rows,
                )
                if amplification_metadata is not None:
                    plan_maxima = {}
                    if hasattr(plan, "page_count"):
                        plan_maxima["topology_pages"] = int(plan.page_count)
                    attach_work_amplification(
                        amplification_metadata,
                        operation="overlay.diff.grouped_plan.build",
                        metric_family="group_compression",
                        sums={
                            "input_rows": int(topology_right_batch.row_count),
                            "output_groups": int(topology_left_batch.row_count),
                        },
                        maxima=plan_maxima,
                        unavailable=(
                            "input_segments",
                            "input_coordinates",
                            "pre_reduction_fragments",
                            "output_parts",
                            "output_coordinates",
                        ),
                    )
        except _GroupedOverlayDifferenceNativeDeclined:
            raise
        _sync_hotpath()
        try:
            with hotpath_stage(
                "overlay.diff.grouped_plan.materialize",
                category="refine",
            ) as amplification_metadata:
                diff_owned, _selected = _materialize_overlay_execution_plan(
                    plan,
                    operation="difference",
                    requested=ExecutionMode.GPU,
                    preserve_row_count=left_batch.row_count,
                    valid_empty_rows=cp.asarray(
                        topology_left_batch._ensure_device_state(
                            preserve_indexed_view=True,
                        ).validity,
                        dtype=cp.bool_,
                    ),
                )
                if amplification_metadata is not None:
                    attach_work_amplification(
                        amplification_metadata,
                        operation="overlay.diff.grouped_plan.materialize",
                        metric_family="group_compression",
                        sums={
                            "input_rows": int(topology_right_batch.row_count),
                            "output_groups": int(diff_owned.row_count),
                        },
                        maxima={"group_capacity": int(left_batch.row_count)},
                        unavailable=(
                            "input_segments",
                            "input_coordinates",
                            "pre_reduction_fragments",
                            "output_parts",
                            "output_coordinates",
                        ),
                    )
        except _GroupedOverlayDifferenceNativeDeclined:
            raise
        _sync_hotpath()
        if _selected is not ExecutionMode.GPU:
            _decline_native(
                "materialize-selected",
                f"grouped execution plan selected {_selected.value}",
            )
        record_dispatch_event(
            surface="geopandas.array.difference",
            operation="difference",
            implementation="grouped_overlay_difference_gpu",
            reason="grouped exact overlay difference used one row-isolated overlay plan",
            detail=(f"groups={left_batch.row_count}, pairs={right_batch.row_count}"),
            requested=dispatch_mode,
            selected=_selected,
        )
        if diff_owned.row_count != left_batch.row_count:
            raise RuntimeError(
                "grouped overlay difference produced "
                f"{diff_owned.row_count} rows for {left_batch.row_count} groups"
            )
        if diff_owned.row_count > 0:
            from vibespatial.runtime.residency import Residency

            if (
                left_batch.residency is Residency.DEVICE
                and diff_owned.residency is Residency.DEVICE
                and has_gpu_runtime()
            ):
                # Device grouped difference cannot spend a host scalar probe on
                # a defensive invariant. The row-isolated topology selector is
                # the correctness contract; host-owned execution keeps the
                # explicit area assertion below where it does not cross devices.
                pass
            else:
                from vibespatial.constructive.measurement import area_owned

                left_area = np.asarray(
                    area_owned(left_batch),
                    dtype=np.float64,
                )
                diff_area = np.asarray(
                    area_owned(diff_owned),
                    dtype=np.float64,
                )
                area_expanded = diff_area > (left_area + 1.0e-9)
                if bool(np.any(area_expanded)):
                    expanded_rows = np.flatnonzero(area_expanded).astype(
                        np.int64,
                        copy=False,
                    )
                    raise RuntimeError(
                        "grouped overlay difference expanded polygon area for rows "
                        f"{expanded_rows[:8].tolist()}"
                    )
        return diff_owned
    except _GroupedOverlayDifferenceNativeDeclined:
        raise


def _grouped_overlay_difference_capacity_owned(
    left_owned,
    right_owned,
    idx1,
    idx2,
    d_idx1,
    d_idx2,
    _has_device_indices: bool,
    _pairwise_mode,
):
    """Execute all difference pairs as one full source-row grouped carrier.

    Pair rows are sorted once and gathered at relation capacity. Dense group
    offsets retain every left source row, including zero-length groups, so the
    grouped topology planner returns the complete public-row result directly.
    Paging and temporary-memory ownership stay inside that planner.
    """
    from vibespatial.runtime.residency import Residency, TransferTrigger

    try:
        import cupy as cp
    except ImportError:  # pragma: no cover - exercised on CPU-only installs
        cp = None

    use_device_indices = (
        _has_device_indices and cp is not None and d_idx1 is not None and d_idx2 is not None
    )
    use_device_gather = (
        cp is not None and _pairwise_mode is not ExecutionMode.CPU and has_gpu_runtime()
    )
    if use_device_gather:
        left_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="overlay difference grouped left carrier",
        )
        right_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="overlay difference grouped right gather",
        )

    with hotpath_stage(
        "overlay.diff.group_index_build",
        category="setup",
    ) as amplification_metadata:
        if use_device_indices:
            left_pairs = cp.asarray(d_idx1, dtype=cp.int64)
            right_pairs = cp.asarray(d_idx2, dtype=cp.int64)
            order = _device_pair_lexicographic_order(left_pairs, right_pairs)
            grouped_left = left_pairs[order]
            grouped_right = right_pairs[order]
            xp = cp
        else:
            left_pairs = np.asarray(idx1, dtype=np.int64)
            right_pairs = np.asarray(idx2, dtype=np.int64)
            order = np.lexsort((right_pairs, left_pairs))
            grouped_left = left_pairs[order]
            grouped_right = right_pairs[order]
            xp = np

        row_count = int(left_owned.row_count)
        pair_count = int(grouped_right.size)
        group_counts = xp.bincount(
            grouped_left,
            minlength=row_count,
        )[:row_count].astype(xp.int64, copy=False)
        group_offsets = xp.empty(row_count + 1, dtype=xp.int64)
        group_offsets[0] = 0
        xp.cumsum(group_counts, out=group_offsets[1:])
        if amplification_metadata is not None:
            attach_work_amplification(
                amplification_metadata,
                operation="overlay.diff.group_index_build",
                metric_family="group_compression",
                sums={
                    "input_rows": pair_count,
                    "output_groups": row_count,
                },
                maxima={"relation_capacity": pair_count},
                unavailable=(
                    "max_group_size",
                    "input_segments",
                    "input_coordinates",
                    "pre_reduction_fragments",
                    "output_parts",
                    "output_coordinates",
                ),
            )

    _sync_hotpath()
    with hotpath_stage(
        "overlay.diff.right_gather",
        category="refine",
    ) as amplification_metadata:
        if use_device_gather and use_device_indices:
            right_gathered = right_owned.device_take(
                grouped_right.astype(cp.int64, copy=False),
            )
        elif use_device_gather:
            host_right = np.asarray(grouped_right, dtype=np.int64)
            right_gathered = right_owned.device_take(
                cp.asarray(host_right, dtype=cp.int64),
                host_indices_for_sizing=host_right,
            )
        else:
            right_gathered = right_owned.take(grouped_right)
        if amplification_metadata is not None:
            attach_work_amplification(
                amplification_metadata,
                operation="overlay.diff.right_gather",
                metric_family="group_compression",
                sums={
                    "input_rows": pair_count,
                    "pre_reduction_fragments": int(right_gathered.row_count),
                    "output_groups": row_count,
                },
                maxima={"relation_capacity": pair_count},
                unavailable=(
                    "max_group_size",
                    "input_segments",
                    "input_coordinates",
                    "output_parts",
                    "output_coordinates",
                ),
            )

    grouped = NativeGrouped.from_dense_sorted_offsets(
        group_offsets,
        row_count=pair_count,
        all_groups_observed=False,
        group_size_min=0,
        group_size_max=None,
    )
    _sync_hotpath()
    with hotpath_stage(
        "overlay.diff.grouped_difference",
        category="refine",
    ) as amplification_metadata:
        diff_owned = _grouped_overlay_difference_owned(
            left_owned,
            right_gathered,
            group_offsets,
            dispatch_mode=_pairwise_mode,
            _native_grouped=grouped,
            _all_groups_observed=False,
            _group_size_min=0,
            _group_size_max=None,
        )
        if amplification_metadata is not None:
            grouped_sums = {
                "input_rows": pair_count,
                "pre_reduction_fragments": int(right_gathered.row_count),
            }
            grouped_unavailable = [
                "max_group_size",
                "input_segments",
                "input_coordinates",
                "output_parts",
                "output_coordinates",
            ]
            if diff_owned is None:
                grouped_unavailable.append("output_groups")
            else:
                grouped_sums["output_groups"] = int(diff_owned.row_count)
            attach_work_amplification(
                amplification_metadata,
                operation="overlay.diff.grouped_difference",
                metric_family="group_compression",
                sums=grouped_sums,
                maxima={"group_capacity": row_count},
                unavailable=tuple(grouped_unavailable),
            )
    _sync_hotpath()
    record_dispatch_event(
        surface="geopandas.array.difference",
        operation="difference",
        implementation="grouped_overlay_difference_full_row_capacity",
        reason=(
            "overlay difference retained every source-left row in one dense "
            "NativeGrouped carrier and delegated paging to grouped topology"
        ),
        detail=f"rows={row_count}, pairs={pair_count}",
        requested=_pairwise_mode,
        selected=(ExecutionMode.GPU if use_device_gather else ExecutionMode.CPU),
    )
    return diff_owned


def _make_valid_geoseries(
    gs,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
):
    """Apply make_valid to polygon rows of a GeoSeries, preferring GPU path.

    When the GeoSeries has owned backing, routes through make_valid_owned to
    keep data device-resident and avoid Shapely materialisation.  Falls back
    to the standard GeoSeries.make_valid() path otherwise.
    """
    ga = gs.values
    owned = _series_owned(gs)
    if owned is not None:
        if _owned_all_valid_without_device_probe(owned):
            return gs
        from vibespatial.runtime.residency import Residency

        if owned.residency is Residency.DEVICE:
            if not _owned_has_logical_polygon_rows(owned):
                return gs
        else:
            poly_ix = _series_polygon_mask(gs)
            if not poly_ix.any():
                return gs

        if owned.residency is not Residency.DEVICE:
            gs = gs.copy()
            gs.loc[poly_ix] = gs[poly_ix].make_valid()
            try:
                from vibespatial.geometry.owned import from_shapely_geometries

                new_owned = from_shapely_geometries(
                    np.asarray(gs.array, dtype=object),
                    residency=Residency.HOST,
                )
                new_ga = GeometryArray.from_owned(new_owned, crs=ga.crs)
                return GeoSeries(new_ga, index=gs.index)
            except NotImplementedError:
                return gs

        from vibespatial.constructive.make_valid_pipeline import make_valid_owned

        mv_result = make_valid_owned(
            owned=owned,
            dispatch_mode=dispatch_mode,
        )
        if mv_result.repaired_rows.size > 0:
            # Native repair is complete-or-decline. Callers consume the aligned
            # result directly and never revalidate or patch residual rows.
            if mv_result.owned is not None:
                _seed_all_validity_cache_if_owned(mv_result.owned)
                new_ga = GeometryArray.from_owned(mv_result.owned, crs=ga.crs)
                return GeoSeries(new_ga, index=gs.index)
            # Fallback: rebuild from Shapely geometries
            try:
                from vibespatial.geometry.owned import from_shapely_geometries

                new_owned = from_shapely_geometries(list(mv_result.geometries))
                _seed_all_validity_cache_if_owned(new_owned)
                new_ga = GeometryArray.from_owned(new_owned, crs=ga.crs)
            except NotImplementedError:
                new_ga = GeometryArray(mv_result.geometries, crs=ga.crs)
            return GeoSeries(new_ga, index=gs.index)
        # All rows already valid — owned backing preserved, return as-is.
        return gs

    # Shapely fallback path: no owned backing available.
    poly_ix = _series_polygon_mask(gs)
    if not poly_ix.any():
        return gs
    gs = gs.copy()
    gs.loc[poly_ix] = gs[poly_ix].make_valid()
    return gs


def _ensure_geometry_column(df):
    """Ensure that the geometry column is called 'geometry'.

    If another column with that name exists, it will be dropped.
    """
    if not df._geometry_column_name == "geometry":
        if PANDAS_GE_30:
            if "geometry" in df.columns:
                df = df.drop("geometry", axis=1)
            df = df.rename_geometry("geometry")
        else:
            if "geometry" in df.columns:
                df.drop("geometry", axis=1, inplace=True)
            df.rename_geometry("geometry", inplace=True)
    return df


def _device_pair_lexicographic_order(d_left, d_right):
    """Return exact ``(left, right)`` order from two device SoA columns."""
    import cupy as cp

    from vibespatial.overlay.graph import _stable_radix_order_pass

    pair_count = int(d_left.size)
    order = cp.arange(pair_count, dtype=cp.int32)
    order = _stable_radix_order_pass(order, d_right)
    return _stable_radix_order_pass(order, d_left)


def _intersecting_index_pairs(
    df1,
    df2,
    *,
    left_owned=None,
    right_owned=None,
    capacity_output: bool = False,
):
    # ADR-0042 low-level contract: spatial indexing still produces index arrays.
    # sindex.query has its own owned-dispatch path (sindex.py lines 334-378)
    # that routes through query_spatial_index when both sides support owned.
    #
    # Phase 2 zero-copy: when both DataFrames have owned (device-resident)
    # backing, request device-resident index arrays from the spatial index
    # to eliminate the D->H->D round-trip when downstream take() re-uploads.
    # Returns DeviceSpatialJoinResult when device arrays are available,
    # otherwise returns the standard (2, n) numpy array or (idx1, idx2) tuple.
    left_all_polygons = _series_family_summary(df1.geometry)[0] if left_owned is not None else False
    right_all_polygons = (
        _series_family_summary(df2.geometry)[0] if right_owned is not None else False
    )
    if (
        left_owned is not None
        and right_owned is not None
        and left_all_polygons
        and right_all_polygons
        and (left_owned.row_count * right_owned.row_count) <= _OVERLAY_BBOX_PAIR_FAST_PATH_MAX_PAIRS
    ):
        candidate_pairs = generate_bounds_pairs(
            left_owned,
            right_owned,
            capacity_output=capacity_output,
        )
        d_left = getattr(candidate_pairs, "device_left_indices", None)
        d_right = getattr(candidate_pairs, "device_right_indices", None)
        if d_left is not None and d_right is not None:
            import cupy as cp

            d_left = cp.asarray(d_left, dtype=cp.int32)
            d_right = cp.asarray(d_right, dtype=cp.int32)
            device_selection = candidate_pairs.device_selection
            if device_selection is not None:
                pair_count = device_selection.capacity
                selected = ExecutionMode.GPU
                result = NativeRelationSelection(
                    relation=NativeRelation(
                        left_indices=d_left,
                        right_indices=d_right,
                        left_row_count=int(left_owned.row_count),
                        right_row_count=int(right_owned.row_count),
                        predicate="intersects",
                    ),
                    selection=device_selection,
                )
            elif int(d_left.size) > 0:
                order = _device_pair_lexicographic_order(d_left, d_right)
                d_left = d_left[order]
                d_right = d_right[order]
                pair_count = int(d_left.size)
                selected = ExecutionMode.GPU
                result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)
            else:
                pair_count = 0
                selected = ExecutionMode.GPU
                result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)
        else:
            left_idx = np.asarray(candidate_pairs.left_indices, dtype=np.int32)
            right_idx = np.asarray(candidate_pairs.right_indices, dtype=np.int32)
            if left_idx.size > 0:
                order = np.lexsort((right_idx, left_idx))
                left_idx = left_idx[order]
                right_idx = right_idx[order]
            pair_count = int(left_idx.size)
            selected = ExecutionMode.CPU
            result = (left_idx, right_idx)
        record_dispatch_event(
            surface="geopandas.overlay.sindex",
            operation="intersects",
            implementation="gpu_bbox_pairs_fast_path",
            reason=(
                "owned polygon overlay used direct bbox candidate pairs "
                f"for {left_owned.row_count}x{right_owned.row_count} rows"
            ),
            detail=(
                f"pair_capacity={pair_count}, "
                f"left_rows={left_owned.row_count}, right_rows={right_owned.row_count}"
            ),
            requested=ExecutionMode.AUTO,
            selected=selected,
        )
        return result

    request_device = left_owned is not None and right_owned is not None
    if request_device:
        relation, execution = df2.sindex.query_relation(
            left_owned,
            predicate="intersects",
            sort=True,
            return_device=True,
            query_row_count=int(left_owned.row_count),
        )
        left_indices = relation.left_indices
        right_indices = relation.right_indices
        if hasattr(left_indices, "__cuda_array_interface__") or hasattr(
            right_indices,
            "__cuda_array_interface__",
        ):
            import cupy as cp

            d_left = cp.asarray(left_indices, dtype=cp.int32)
            d_right = cp.asarray(right_indices, dtype=cp.int32)
            result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)
            selected = ExecutionMode.GPU
            pair_count = int(d_left.size)
        else:
            left_idx = np.asarray(left_indices, dtype=np.int32)
            right_idx = np.asarray(right_indices, dtype=np.int32)
            result = (left_idx, right_idx)
            selected = execution.selected
            pair_count = int(left_idx.size)
        record_dispatch_event(
            surface="geopandas.overlay.sindex",
            operation="intersects",
            implementation="native_spatial_index_relation",
            reason=(
                "overlay consumed NativeSpatialIndex relation pairs without "
                "public sindex.query export"
            ),
            detail=(
                f"rows={pair_count}, "
                f"left_rows={left_owned.row_count}, right_rows={right_owned.row_count}"
            ),
            requested=execution.requested,
            selected=selected,
        )
        return result

    result = df2.sindex.query(
        df1.geometry,
        predicate="intersects",
        sort=True,
        return_device=False,
    )
    return result


def _reverse_intersecting_index_pairs(index_result):
    """Derive the reverse intersects pair mapping from a forward query result."""
    if isinstance(index_result, DeviceSpatialJoinResult):
        import cupy as cp

        d_left = cp.asarray(index_result.d_right_idx, dtype=cp.int32)
        d_right = cp.asarray(index_result.d_left_idx, dtype=cp.int32)
        if d_left.size > 0:
            # Fancy indexing below creates the independent reverse carriers;
            # copying the unsorted inputs first only doubles relation traffic.
            order = _device_pair_lexicographic_order(d_left, d_right)
            d_left = d_left[order]
            d_right = d_right[order]
        return DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)

    if isinstance(index_result, np.ndarray) and index_result.ndim == 2:
        idx1, idx2 = index_result
    else:
        idx1, idx2 = index_result

    left = np.asarray(idx2)
    right = np.asarray(idx1)
    if left.size > 0:
        order = np.lexsort((right, left))
        left = left[order]
        right = right[order]
    return left, right


def _assemble_intersection_attributes(idx1, idx2, df1, df2):
    """ADR-0042 transitional boundary: attribute assembly from index arrays.

    Receives integer index arrays and attribute-only DataFrames (geometry
    columns already dropped).  Returns a merged DataFrame with attributes
    from both sides joined via the spatial index pairs.

    Indices may be CuPy arrays (Phase 3) — materialized to host here since
    pandas DataFrames are inherently host-side.
    """
    h_idx1 = _overlay_device_to_host(
        idx1,
        reason="overlay intersection terminal attribute left-index export",
    )
    h_idx2 = _overlay_device_to_host(
        idx2,
        reason="overlay intersection terminal attribute right-index export",
    )
    pairs = pd.DataFrame({"__idx1": h_idx1, "__idx2": h_idx2})
    result = pairs.merge(
        df1,
        left_on="__idx1",
        right_index=True,
    )
    result = result.merge(
        df2,
        left_on="__idx2",
        right_index=True,
        suffixes=("_1", "_2"),
    )
    return result


def _intersection_attribute_columns(df1: GeoDataFrame, df2: GeoDataFrame) -> list[str]:
    """Return the public intersection attribute schema without materializing rows."""
    empty = np.empty(0, dtype=np.int32)
    columns = _assemble_intersection_attributes(
        empty,
        empty,
        df1.drop(df1._geometry_column_name, axis=1),
        df2.drop(df2._geometry_column_name, axis=1),
    ).columns
    return list(columns)


def _assemble_polygon_intersection_rows_with_lower_dim(
    left_pairs: GeoSeries,
    right_pairs: GeoSeries,
    area_pairs: GeoSeries,
) -> GeoSeries:
    """Recover lower-dimensional polygon intersection remnants at the boundary.

    The polygon constructive intersection path returns only polygonal area.
    Public overlay semantics also need line/point remnants when polygon pairs
    touch without area overlap, and GeometryCollections when polygonal area has
    additional lower-dimensional pieces.
    """
    area_geoms = np.asarray(area_pairs, dtype=object)
    left_geoms = np.asarray(left_pairs, dtype=object)
    right_geoms = np.asarray(right_pairs, dtype=object)
    boundary_geoms = np.asarray(
        shapely.intersection(
            shapely.boundary(left_geoms),
            shapely.boundary(right_geoms),
        ),
        dtype=object,
    )

    assembled = np.empty(len(area_geoms), dtype=object)
    for row_index in range(len(area_geoms)):
        area_geom = area_geoms[row_index]
        if area_geom is not None and area_geom.is_empty:
            area_geom = None
        elif area_geom is not None and getattr(area_geom, "area", 0.0) == 0.0:
            area_geom = None

        edge_geom = boundary_geoms[row_index]
        if edge_geom is not None and edge_geom.is_empty:
            edge_geom = None

        if area_geom is not None and edge_geom is not None:
            edge_parts = shapely.get_parts(np.asarray([edge_geom], dtype=object))
            if len(edge_parts) > 0:
                cleaned_parts = np.asarray(
                    shapely.difference(
                        edge_parts,
                        np.full(len(edge_parts), area_geom.boundary, dtype=object),
                    ),
                    dtype=object,
                )
                edge_parts = shapely.get_parts(cleaned_parts)
                edge_parts = edge_parts[~shapely.is_empty(edge_parts)]
                edge_geom = shapely.union_all(edge_parts) if len(edge_parts) > 0 else None
            else:
                edge_geom = None

        if edge_geom is not None:
            edge_parts = [
                part
                for part in shapely.get_parts(np.asarray([edge_geom], dtype=object))
                if not part.is_empty
            ]
            if edge_parts:
                unique_parts = []
                seen_parts = set()
                for part in edge_parts:
                    normalized = shapely.normalize(part)
                    key = normalized.wkb
                    if key in seen_parts:
                        continue
                    seen_parts.add(key)
                    unique_parts.append(normalized)

                if len(unique_parts) == 1:
                    edge_geom = unique_parts[0]
                else:
                    edge_part_types = {part.geom_type for part in unique_parts}
                    merged_edges = shapely.union_all(np.asarray(unique_parts, dtype=object))
                    if edge_part_types <= {"LineString", "LinearRing", "MultiLineString"}:
                        edge_geom = shapely.line_merge(merged_edges)
                    else:
                        edge_geom = merged_edges
                if edge_geom is not None and edge_geom.is_empty:
                    edge_geom = None

        if area_geom is None and edge_geom is None:
            assembled[row_index] = None
            continue
        if area_geom is None:
            assembled[row_index] = edge_geom
            continue
        if edge_geom is None:
            assembled[row_index] = area_geom
            continue

        parts = [area_geom]
        parts.extend(
            part
            for part in shapely.get_parts(np.asarray([edge_geom], dtype=object))
            if not part.is_empty
        )
        assembled[row_index] = GeometryCollection(parts)

    return GeoSeries(assembled, index=area_pairs.index, crs=area_pairs.crs)


def _count_non_polygon_collection_parts(geometries: np.ndarray) -> int:
    """Count dropped non-polygon parts for GeometryCollection warning parity."""
    if len(geometries) == 0:
        return 0
    parts = shapely.get_parts(geometries)
    if len(parts) == 0:
        return 0
    part_type_ids = shapely.get_type_id(parts)
    non_empty_mask = ~shapely.is_empty(parts)
    polygon_mask = (part_type_ids == _SHAPELY_TYPE_ID_POLYGON) | (
        part_type_ids == _SHAPELY_TYPE_ID_MULTIPOLYGON
    )
    return int(np.count_nonzero(non_empty_mask & ~polygon_mask))


def _count_dropped_polygon_parts_from_exact_values(
    exact_values: np.ndarray,
) -> int:
    """Count dropped lower-dimensional pieces from exact host intersection output."""
    if len(exact_values) == 0:
        return 0

    missing_mask = shapely.is_missing(exact_values) | shapely.is_empty(exact_values)
    type_ids = shapely.get_type_id(exact_values)
    polygon_mask = (type_ids == _SHAPELY_TYPE_ID_POLYGON) | (
        type_ids == _SHAPELY_TYPE_ID_MULTIPOLYGON
    )
    collection_mask = type_ids == _SHAPELY_TYPE_ID_GEOMETRYCOLLECTION

    dropped = int(np.count_nonzero(~missing_mask & ~polygon_mask & ~collection_mask))
    if collection_mask.any():
        dropped += _count_non_polygon_collection_parts(exact_values[collection_mask])
    return dropped


def _count_dropped_polygon_intersection_parts(
    left_values: np.ndarray,
    right_values: np.ndarray,
    row_count: int,
    *,
    exact_values: np.ndarray | None = None,
) -> int:
    """Count lower-dimensional exact-intersection output dropped by keep_geom_type.

    Warning parity needs the exact host intersection shape, not just the
    polygon-only area output retained by the fast native path.
    """
    if row_count == 0:
        return 0

    if exact_values is None:
        exact_values = np.asarray(shapely.intersection(left_values, right_values), dtype=object)
    else:
        exact_values = np.asarray(exact_values, dtype=object)
    return _count_dropped_polygon_parts_from_exact_values(exact_values)


def _exact_keep_mask_and_dropped_count_for_polygon_intersection_warning_rows(
    left_values: np.ndarray,
    right_values: np.ndarray,
    *,
    exact_values: np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Classify kept polygon rows from the exact host intersection oracle."""
    left_values = np.asarray(left_values, dtype=object)
    right_values = np.asarray(right_values, dtype=object)
    if exact_values is None:
        exact_values = np.asarray(shapely.intersection(left_values, right_values), dtype=object)
    else:
        exact_values = np.asarray(exact_values, dtype=object)

    polygon_only_values = _strip_non_polygon_collection_parts(exact_values)

    keep_mask = np.array(
        [
            geom is not None
            and not shapely.is_empty(geom)
            and geom.geom_type in POLYGON_GEOM_TYPES
            and float(shapely.area(geom)) > 0.0
            for geom in polygon_only_values
        ],
        dtype=bool,
    )
    kept_rows = np.flatnonzero(keep_mask).astype(np.intp, copy=False)
    if kept_rows.size > 0:
        rep_points = np.asarray(
            shapely.point_on_surface(polygon_only_values[kept_rows]),
            dtype=object,
        )
        keep_mask = keep_mask.copy()
        keep_mask[kept_rows] &= np.asarray(
            shapely.contains(left_values[kept_rows], rep_points)
            & shapely.contains(right_values[kept_rows], rep_points),
            dtype=bool,
        )
    return keep_mask, _count_dropped_polygon_parts_from_exact_values(exact_values), exact_values


def _replace_geoseries_rows_with_exact_values(
    geometries: GeoSeries,
    rows: np.ndarray,
    exact_values: np.ndarray,
) -> GeoSeries:
    """Replace selected rows in a GeoSeries with exact host geometries."""
    rows = np.asarray(rows, dtype=np.intp)
    if rows.size == 0:
        return geometries

    values = _geoseries_object_values(geometries).copy()
    values[rows] = np.asarray(exact_values, dtype=object)
    rebuilt = GeoSeries(values, index=geometries.index, crs=geometries.crs)

    overlap_mask = getattr(geometries.values, "_polygon_rect_boundary_overlap", None)
    if overlap_mask is not None:
        rebuilt.values._polygon_rect_boundary_overlap = np.asarray(overlap_mask, dtype=bool)
    exact_polygon_only = getattr(geometries.values, "_polygon_rect_exact_polygon_only", None)
    if exact_polygon_only is not None:
        rebuilt.values._polygon_rect_exact_polygon_only = np.asarray(
            exact_polygon_only,
            dtype=bool,
        )
    return rebuilt


def _warning_candidate_mask_for_polygon_keep_geom_type(
    left_values: np.ndarray,
    right_values: np.ndarray,
    keep_mask: np.ndarray,
) -> np.ndarray:
    """Return rows that can affect the polygon keep-geom-type warning count.

    Rows already dropped by the polygon area filter always need exact host
    classification. Rows retained as polygonal area only matter when their
    boundaries still intersect, because that is the only way the exact
    intersection can carry lower-dimensional extras inside a kept row.
    """
    suspect_mask = ~keep_mask
    kept_rows = np.flatnonzero(keep_mask)
    if kept_rows.size == 0:
        return suspect_mask

    kept_left = left_values[kept_rows]
    kept_right = right_values[kept_rows]
    boundary_overlap = shapely.intersects(
        shapely.boundary(kept_left),
        shapely.boundary(kept_right),
    )
    if np.any(boundary_overlap):
        suspect_mask = suspect_mask.copy()
        suspect_mask[kept_rows[np.asarray(boundary_overlap, dtype=bool)]] = True
    return suspect_mask


def _warning_candidate_mask_from_exact_intersection_values(
    exact_values: np.ndarray,
    keep_mask: np.ndarray,
) -> np.ndarray:
    """Classify keep-geom-type warning rows directly from exact host output."""
    exact_values = np.asarray(exact_values, dtype=object)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if exact_values.size != keep_mask.size:
        raise ValueError("exact_values and keep_mask must have the same length")

    warning_mask = ~keep_mask
    kept_rows = np.flatnonzero(keep_mask).astype(np.intp, copy=False)
    if kept_rows.size == 0:
        return warning_mask

    kept_values = exact_values[kept_rows]
    missing_mask = shapely.is_missing(kept_values) | shapely.is_empty(kept_values)
    type_ids = shapely.get_type_id(kept_values)
    polygon_mask = (type_ids == _SHAPELY_TYPE_ID_POLYGON) | (
        type_ids == _SHAPELY_TYPE_ID_MULTIPOLYGON
    )
    collection_mask = type_ids == _SHAPELY_TYPE_ID_GEOMETRYCOLLECTION
    kept_warning_mask = ~missing_mask & ~polygon_mask & ~collection_mask

    if collection_mask.any():
        for local_row in np.flatnonzero(collection_mask).astype(np.intp, copy=False):
            kept_warning_mask[local_row] = (
                _count_non_polygon_collection_parts(
                    np.asarray([kept_values[local_row]], dtype=object)
                )
                > 0
            )

    warning_mask = warning_mask.copy()
    warning_mask[kept_rows] = kept_warning_mask
    return warning_mask


def _aligned_pair_owned_from_area(area_owned) -> tuple[object | None, object | None]:
    """Return row-aligned source pair arrays cached on an intersection result."""
    if area_owned is None:
        return None, None
    left_owned = getattr(area_owned, "_aligned_left_pairs_owned", None)
    right_owned = getattr(area_owned, "_aligned_right_pairs_owned", None)
    if left_owned is None or right_owned is None:
        return None, None
    if (
        getattr(left_owned, "row_count", None) != area_owned.row_count
        or getattr(right_owned, "row_count", None) != area_owned.row_count
    ):
        return None, None
    return left_owned, right_owned


def _device_count_dropped_polygon_warning_rows_owned(
    area_owned,
    *,
    warning_rows,
    warning_keep_mask,
    warning_active_mask,
    left_owned,
    right_owned,
    warning_left_rows,
    warning_right_rows,
) -> int | None:
    """Classify warning rows from device-owned pair and area carriers."""
    if not has_gpu_runtime():
        return None

    import cupy as cp

    from vibespatial.constructive.boundary_remnants import (
        polygon_pair_boundary_remnant_mask_capacity_device,
    )
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import device_mask_owned_capacity
    from vibespatial.runtime.residency import Residency

    area_owned = getattr(area_owned, "owned", area_owned)
    if area_owned is None or getattr(area_owned, "residency", None) is not Residency.DEVICE:
        return None

    d_warning_rows = cp.asarray(warning_rows, dtype=cp.int64)
    warning_count = int(d_warning_rows.size)
    if warning_count == 0:
        return 0
    d_warning_keep = cp.asarray(warning_keep_mask, dtype=cp.bool_)
    d_warning_active = (
        cp.ones(warning_count, dtype=cp.bool_)
        if warning_active_mask is None
        else cp.asarray(warning_active_mask, dtype=cp.bool_)
    )
    d_warning_left_rows = cp.asarray(warning_left_rows, dtype=cp.int64)
    d_warning_right_rows = cp.asarray(warning_right_rows, dtype=cp.int64)
    if any(
        int(values.size) != warning_count
        for values in (
            d_warning_keep,
            d_warning_active,
            d_warning_left_rows,
            d_warning_right_rows,
        )
    ):
        return None

    d_topology_remnants = getattr(
        area_owned,
        "_polygon_intersection_lower_dimensional_remnant",
        None,
    )
    if d_topology_remnants is not None:
        d_topology_remnants = cp.asarray(d_topology_remnants, dtype=cp.bool_)
        if int(d_topology_remnants.size) == int(area_owned.row_count):
            d_dropped = d_topology_remnants[d_warning_rows] & d_warning_active
            runtime = get_cuda_runtime()
            dropped_count = runtime.copy_device_to_host(
                cp.sum(d_dropped, dtype=cp.uint32).reshape(1),
                reason=(
                    "polygon keep-geom topology-remnant warning count scalar fence"
                ),
            )
            record_dispatch_event(
                surface="geopandas.overlay.intersection",
                operation="keep_geom_type_warning_count",
                implementation="polygon_intersection_topology_remnant_gpu",
                reason=(
                    "the original polygon intersection topology reduced exact "
                    "non-area components directly to aligned result metadata"
                ),
                detail=(
                    f"rows={warning_count}; "
                    "workload_shape=half_edge_face_capacity_to_result_rows"
                ),
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.GPU,
            )
            return int(dropped_count[0])

    if any(
        owned is None or getattr(owned, "residency", None) is not Residency.DEVICE
        for owned in (left_owned, right_owned)
    ):
        return None

    warning_left = device_mask_owned_capacity(
        left_owned._device_indexed_take(d_warning_left_rows),
        d_warning_active,
    )
    warning_right = device_mask_owned_capacity(
        right_owned._device_indexed_take(d_warning_right_rows),
        d_warning_active,
    )
    warning_area = device_mask_owned_capacity(
        area_owned._device_indexed_take(
            d_warning_rows,
            assume_unique_indices=True,
        ),
        d_warning_active,
    )
    d_explicit_boundary_overlap = cp.zeros(warning_count, dtype=cp.bool_)
    boundary_overlap = getattr(area_owned, "_polygon_rect_boundary_overlap", None)
    if boundary_overlap is not None:
        d_boundary_overlap = cp.asarray(boundary_overlap, dtype=cp.bool_)
        if int(d_boundary_overlap.size) == int(area_owned.row_count):
            d_explicit_boundary_overlap = d_boundary_overlap[d_warning_rows] & d_warning_active
    left_state = warning_left._ensure_device_state(preserve_indexed_view=True)
    right_state = warning_right._ensure_device_state(preserve_indexed_view=True)

    def _metadata_proves_rectangle_domain(state) -> bool:
        polygon = state.families.get(GeometryFamily.POLYGON)
        return (
            set(state.families) == {GeometryFamily.POLYGON}
            and polygon is not None
            and int(getattr(polygon, "dense_single_ring_width", 0) or 0) == 5
            and bool(getattr(polygon, "axis_aligned_rectangles", False))
        )

    if _metadata_proves_rectangle_domain(left_state) and _metadata_proves_rectangle_domain(
        right_state
    ):
        from vibespatial.kernels.constructive.polygon_rect_intersection import (
            device_polygon_shape_mask_bounds,
        )

        left_shape = device_polygon_shape_mask_bounds(warning_left)
        right_shape = device_polygon_shape_mask_bounds(warning_right)
        if left_shape is None or right_shape is None:
            return None
        _, d_left_rect, d_left_bounds = left_shape
        _, d_right_rect, d_right_bounds = right_shape
        d_x_span = cp.minimum(d_left_bounds[:, 2], d_right_bounds[:, 2]) - cp.maximum(
            d_left_bounds[:, 0],
            d_right_bounds[:, 0],
        )
        d_y_span = cp.minimum(d_left_bounds[:, 3], d_right_bounds[:, 3]) - cp.maximum(
            d_left_bounds[:, 1],
            d_right_bounds[:, 1],
        )
        d_supported = (
            d_warning_active
            & cp.asarray(d_left_rect, dtype=cp.bool_)
            & cp.asarray(d_right_rect, dtype=cp.bool_)
        )
        d_lower_dimensional_contact = (
            d_supported
            & (d_x_span >= 0.0)
            & (d_y_span >= 0.0)
            & ~((d_x_span > 0.0) & (d_y_span > 0.0))
        ) | d_explicit_boundary_overlap
        d_dropped = d_lower_dimensional_contact & ~d_warning_keep
    else:
        warning_result = polygon_pair_boundary_remnant_mask_capacity_device(
            warning_left,
            warning_right,
            warning_area,
            keep_area_mask=d_warning_keep & d_warning_active,
        )
        if warning_result is None:
            return None
        d_dropped, d_supported = warning_result
    d_dropped &= d_warning_active

    runtime = get_cuda_runtime()
    d_counts = cp.stack(
        (
            cp.sum(cp.asarray(d_warning_active, dtype=cp.uint32), dtype=cp.uint32),
            cp.sum(cp.asarray(d_supported, dtype=cp.uint32), dtype=cp.uint32),
            cp.sum(cp.asarray(d_dropped, dtype=cp.uint32), dtype=cp.uint32),
        )
    )
    counts = runtime.copy_device_to_host(
        d_counts,
        reason="polygon keep-geom candidate-remnant warning count scalar fence",
    )
    if int(counts[0]) != int(counts[1]):
        return None
    record_dispatch_event(
        surface="geopandas.overlay.intersection",
        operation="keep_geom_type_warning_count",
        implementation="polygon_pair_warning_candidate_remnants_gpu",
        reason=(
            "unbounded polygon rows lowered boundary contacts through same-row "
            "segment candidates and native constructive remnants"
        ),
        detail=(f"rows={warning_count}; workload_shape=segment_candidate_relation_to_row_capacity"),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return int(counts[2])


def _device_count_dropped_polygon_intersection_warning_rows(
    area_owned,
    keep_mask: np.ndarray,
    warning_rows: np.ndarray,
    *,
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    left_pairs: GeoSeries | None = None,
    right_pairs: GeoSeries | None = None,
    warning_keep_mask: np.ndarray | None = None,
) -> int | None:
    """Resolve public pair provenance into the owned warning classifier."""
    if not has_gpu_runtime():
        return None

    import cupy as cp

    from vibespatial.runtime.residency import Residency

    area_owned = getattr(area_owned, "owned", area_owned)
    if area_owned is None or getattr(area_owned, "residency", None) is not Residency.DEVICE:
        return None
    d_warning_rows = cp.asarray(warning_rows, dtype=cp.int64)
    warning_count = int(d_warning_rows.size)
    if warning_count == 0:
        return 0
    if warning_keep_mask is not None:
        d_warning_keep = cp.asarray(warning_keep_mask, dtype=cp.bool_)
    else:
        d_warning_keep = cp.asarray(keep_mask, dtype=cp.bool_)[d_warning_rows]
    if int(d_warning_keep.size) != warning_count:
        return None

    left_source_owned = (
        getattr(left_source.values, "_owned", None) if left_source is not None else None
    )
    right_source_owned = (
        getattr(right_source.values, "_owned", None) if right_source is not None else None
    )
    left_pairs_owned = (
        getattr(left_pairs.values, "_owned", None) if left_pairs is not None else None
    )
    right_pairs_owned = (
        getattr(right_pairs.values, "_owned", None) if right_pairs is not None else None
    )
    if left_pairs_owned is None or right_pairs_owned is None:
        aligned_left, aligned_right = _aligned_pair_owned_from_area(area_owned)
        if left_pairs_owned is None:
            left_pairs_owned = aligned_left
        if right_pairs_owned is None:
            right_pairs_owned = aligned_right

    use_aligned_pairs = (
        left_pairs_owned is not None
        and right_pairs_owned is not None
        and getattr(left_pairs_owned, "residency", None) is Residency.DEVICE
        and getattr(right_pairs_owned, "residency", None) is Residency.DEVICE
    )
    if use_aligned_pairs:
        device_left_owned = left_pairs_owned
        device_right_owned = right_pairs_owned
        d_warning_left_rows = d_warning_rows
        d_warning_right_rows = d_warning_rows
    elif (
        left_source_owned is not None
        and right_source_owned is not None
        and left_rows is not None
        and right_rows is not None
        and getattr(left_source_owned, "residency", None) is Residency.DEVICE
        and getattr(right_source_owned, "residency", None) is Residency.DEVICE
    ):
        device_left_owned = left_source_owned
        device_right_owned = right_source_owned
        d_warning_left_rows = cp.asarray(left_rows, dtype=cp.int64)[d_warning_rows]
        d_warning_right_rows = cp.asarray(right_rows, dtype=cp.int64)[d_warning_rows]
    else:
        return None

    return _device_count_dropped_polygon_warning_rows_owned(
        area_owned,
        warning_rows=d_warning_rows,
        warning_keep_mask=d_warning_keep,
        warning_active_mask=None,
        left_owned=device_left_owned,
        right_owned=device_right_owned,
        warning_left_rows=d_warning_left_rows,
        warning_right_rows=d_warning_right_rows,
    )


def _device_polygon_keep_geom_type_warning_mask_from_de9im(
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    keep_mask: np.ndarray,
    *,
    area_owned=None,
    left_pairs: GeoSeries | None = None,
    right_pairs: GeoSeries | None = None,
    return_device: bool = False,
    candidate_rows=None,
) -> np.ndarray | None:
    """Classify keep-geom-type warning candidates from source-pair DE-9IM bits.

    The warning path only needs to know which rows can have lower-dimensional
    exact-intersection output.  Building source boundaries and then running a
    generic boundary-intersects predicate is substantially heavier than asking
    the existing polygon DE-9IM kernel for the boundary-boundary bit directly.
    """
    if _is_device_array(keep_mask):
        keep_mask_size = int(getattr(keep_mask, "size", len(keep_mask)))
    else:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        keep_mask_size = int(keep_mask.size)
    if keep_mask_size == 0 or not has_gpu_runtime():
        return None

    from vibespatial.constructive.binary_constructive import (
        _device_family_domain_tag_pairs,
    )
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS, TAG_FAMILIES, unique_tag_pairs
    from vibespatial.predicates.polygon import (
        DE9IM_BB,
        DE9IM_BI,
        DE9IM_IB,
        DE9IM_II,
        compute_polygon_de9im_gpu,
    )
    from vibespatial.runtime.residency import Residency

    left_source_owned = (
        getattr(left_source.values, "_owned", None) if left_source is not None else None
    )
    right_source_owned = (
        getattr(right_source.values, "_owned", None) if right_source is not None else None
    )
    left_pairs_owned = (
        getattr(left_pairs.values, "_owned", None) if left_pairs is not None else None
    )
    right_pairs_owned = (
        getattr(right_pairs.values, "_owned", None) if right_pairs is not None else None
    )
    if left_pairs_owned is None or right_pairs_owned is None:
        left_pairs_owned, right_pairs_owned = _aligned_pair_owned_from_area(area_owned)

    if (
        left_source_owned is not None
        and right_source_owned is not None
        and left_rows is not None
        and right_rows is not None
        and getattr(left_source_owned, "residency", None) is Residency.DEVICE
        and getattr(right_source_owned, "residency", None) is Residency.DEVICE
    ):
        left_owned = left_source_owned
        right_owned = right_source_owned
        import cupy as cp

        left_index = cp.asarray(left_rows, dtype=cp.int64)
        right_index = cp.asarray(right_rows, dtype=cp.int64)
    elif (
        left_pairs_owned is not None
        and right_pairs_owned is not None
        and getattr(left_pairs_owned, "residency", None) is Residency.DEVICE
        and getattr(right_pairs_owned, "residency", None) is Residency.DEVICE
    ):
        left_owned = left_pairs_owned
        right_owned = right_pairs_owned
        left_index = np.arange(keep_mask_size, dtype=np.intp)
        right_index = left_index
    else:
        return None

    if left_index.size != keep_mask_size or right_index.size != keep_mask_size:
        return None

    polygon_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}

    try:
        import cupy as cp

        left_state = left_owned._ensure_device_state(preserve_indexed_view=True)
        right_state = right_owned._ensure_device_state(preserve_indexed_view=True)
        d_left_index = cp.asarray(left_index, dtype=cp.int32)
        d_right_index = cp.asarray(right_index, dtype=cp.int32)
        d_left_tags = cp.asarray(left_state.tags)[d_left_index]
        d_right_tags = cp.asarray(right_state.tags)[d_right_index]
        tag_pairs = _device_family_domain_tag_pairs(left_owned, right_owned)
        if tag_pairs is None:
            left_polygon_domain = tuple(
                family for family in left_state.families if family in polygon_families
            )
            right_polygon_domain = tuple(
                family for family in right_state.families if family in polygon_families
            )
            if (
                left_polygon_domain
                and right_polygon_domain
                and len(left_polygon_domain) == len(left_state.families)
                and len(right_polygon_domain) == len(right_state.families)
            ):
                tag_pairs = [
                    (FAMILY_TAGS[left_family], FAMILY_TAGS[right_family])
                    for left_family in left_polygon_domain
                    for right_family in right_polygon_domain
                ]
            else:
                tag_pairs = unique_tag_pairs(d_left_tags, d_right_tags)
        if not tag_pairs:
            return np.zeros(keep_mask.size, dtype=bool)
        for left_tag, right_tag in tag_pairs:
            if (
                TAG_FAMILIES.get(left_tag) not in polygon_families
                or TAG_FAMILIES.get(right_tag) not in polygon_families
            ):
                return None

        d_keep_full = cp.asarray(keep_mask, dtype=cp.bool_)
        d_candidate_rows = None
        if candidate_rows is not None:
            d_candidate_rows = cp.asarray(candidate_rows, dtype=cp.int64)
            candidate_size = int(d_candidate_rows.size)
            if candidate_size == 0:
                d_warning = cp.zeros(keep_mask_size, dtype=cp.bool_)
                if return_device:
                    record_dispatch_event(
                        surface="geopandas.overlay.intersection",
                        operation="keep_geom_type_warning_mask",
                        implementation="gpu_de9im_boundary_warning_mask",
                        reason=(
                            "device DE-9IM warning rowset was empty after "
                            "native exact-polygon metadata pruning"
                        ),
                        detail=f"rows={keep_mask_size}; warning_rows=device; candidates=0",
                        requested=ExecutionMode.GPU,
                        selected=ExecutionMode.GPU,
                    )
                    return d_warning
                return np.zeros(keep_mask_size, dtype=bool)
            if candidate_size > keep_mask_size:
                return None
            d_left_index = d_left_index[d_candidate_rows]
            d_right_index = d_right_index[d_candidate_rows]
            d_left_tags = d_left_tags[d_candidate_rows]
            d_right_tags = d_right_tags[d_candidate_rows]
            d_keep = d_keep_full[d_candidate_rows]
        else:
            candidate_size = keep_mask_size
            d_keep = d_keep_full
        d_warning = cp.zeros(
            candidate_size if d_candidate_rows is not None else keep_mask_size,
            dtype=cp.bool_,
        )
        contact_mask = np.uint16(DE9IM_II | DE9IM_IB | DE9IM_BI | DE9IM_BB)
        boundary_mask = np.uint16(DE9IM_BB)
        single_pair = len(tag_pairs) == 1
        for left_tag, right_tag in tag_pairs:
            left_family = TAG_FAMILIES[left_tag]
            right_family = TAG_FAMILIES[right_tag]
            if single_pair:
                d_sub_idx = None
                d_sub_left = d_left_index
                d_sub_right = d_right_index
            else:
                d_sub_mask = (d_left_tags == left_tag) & (d_right_tags == right_tag)
                d_sub_idx = cp.flatnonzero(d_sub_mask)
                if int(d_sub_idx.size) == 0:
                    continue
                d_sub_left = d_left_index[d_sub_idx]
                d_sub_right = d_right_index[d_sub_idx]

            d_masks = compute_polygon_de9im_gpu(
                left_owned,
                right_owned,
                query_family=left_family,
                tree_family=right_family,
                d_left=d_sub_left,
                d_right=d_sub_right,
                return_device=True,
            )
            if d_masks is None:
                return None
            d_sub_keep = d_keep if d_sub_idx is None else d_keep[d_sub_idx]
            d_sub_warning = ((~d_sub_keep) & ((d_masks & contact_mask) != 0)) | (
                d_sub_keep & ((d_masks & boundary_mask) != 0)
            )
            if d_sub_idx is None:
                d_warning = d_sub_warning.astype(cp.bool_, copy=False)
            else:
                d_warning[d_sub_idx] = d_sub_warning

        if d_candidate_rows is not None:
            d_full_warning = cp.zeros(keep_mask_size, dtype=cp.bool_)
            d_full_warning[d_candidate_rows] = d_warning
            d_warning = d_full_warning

        if return_device:
            record_dispatch_event(
                surface="geopandas.overlay.intersection",
                operation="keep_geom_type_warning_mask",
                implementation="gpu_de9im_boundary_warning_mask",
                reason=(
                    "device DE-9IM classified polygon keep-geom-type warning "
                    "candidates as a native rowset mask"
                ),
                detail=(f"rows={keep_mask_size}; warning_rows=device; candidates={candidate_size}"),
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.GPU,
            )
            return d_warning

        out = _overlay_host_bool_mask_sparse_first(
            d_warning,
            length=keep_mask_size,
            dense_reason="overlay keep-geom-type warning mask host boundary",
            sparse_reason="overlay keep-geom-type warning rows host boundary",
        )
        if out is None:
            return None
        record_dispatch_event(
            surface="geopandas.overlay.intersection",
            operation="keep_geom_type_warning_mask",
            implementation="gpu_de9im_boundary_warning_mask",
            reason="device DE-9IM classified polygon keep-geom-type warning candidates",
            detail=f"rows={keep_mask_size}; warning_rows={int(np.count_nonzero(out))}",
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )
        return np.asarray(out, dtype=bool)
    except _OverlayNativeConstructiveDeclined:
        raise


def _device_polygon_keep_geom_type_cover_mask(
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    warning_rows: np.ndarray,
    *,
    area_owned=None,
    left_pairs: GeoSeries | None = None,
    right_pairs: GeoSeries | None = None,
) -> np.ndarray | None:
    """Return rows provably equal to one source polygon from device predicates.

    When one polygon covers the other, the exact polygon-polygon intersection is
    exactly the covered polygon, so keep-geom-type warning classification does
    not need a host semantic probe for that row.
    """
    if warning_rows.size == 0 or not has_gpu_runtime():
        return None

    left_source_owned = (
        getattr(left_source.values, "_owned", None) if left_source is not None else None
    )
    right_source_owned = (
        getattr(right_source.values, "_owned", None) if right_source is not None else None
    )
    left_pairs_owned = (
        getattr(left_pairs.values, "_owned", None) if left_pairs is not None else None
    )
    right_pairs_owned = (
        getattr(right_pairs.values, "_owned", None) if right_pairs is not None else None
    )
    if left_pairs_owned is None or right_pairs_owned is None:
        left_pairs_owned, right_pairs_owned = _aligned_pair_owned_from_area(area_owned)

    from vibespatial.runtime.residency import Residency

    use_pair_rows = (
        left_pairs_owned is not None
        and right_pairs_owned is not None
        and left_pairs_owned.residency is Residency.DEVICE
        and right_pairs_owned.residency is Residency.DEVICE
    )
    use_source_rows = (
        not use_pair_rows
        and left_source is not None
        and right_source is not None
        and left_rows is not None
        and right_rows is not None
        and left_source_owned is not None
        and right_source_owned is not None
        and left_source_owned.residency is Residency.DEVICE
        and right_source_owned.residency is Residency.DEVICE
    )
    if not use_source_rows and not use_pair_rows:
        return None

    try:
        import cupy as cp

        from vibespatial.constructive.measurement import _area_gpu_device_fp64
        from vibespatial.predicates.binary import binary_predicate_expressions

        def _take_owned_rows(owned, rows):
            if _is_device_array(rows):
                return owned.device_take(cp.asarray(rows, dtype=cp.int64))
            rows64 = np.asarray(rows, dtype=np.int64)
            if rows64.size == 0:
                return owned.take(rows64)
            return owned.device_take(
                cp.asarray(rows64, dtype=cp.int64),
                host_indices_for_sizing=rows64,
            )

        def _area_candidate_masks(left_owned, right_input):
            d_left_area = _area_gpu_device_fp64(left_owned)
            d_right_area = _area_gpu_device_fp64(right_input)
            if int(d_right_area.size) == 1 and int(d_left_area.size) > 1:
                d_right_area = cp.full(
                    d_left_area.shape,
                    d_right_area[0],
                    dtype=cp.float64,
                )
            if int(d_left_area.size) != int(d_right_area.size):
                raise RuntimeError("keep_geom_type cover area precheck row-count mismatch")
            d_scale = cp.maximum(cp.abs(d_left_area), cp.abs(d_right_area))
            d_tol = cp.maximum(d_scale * 1.0e-12, 1.0e-12)
            d_masks = cp.stack(
                (
                    d_left_area + d_tol >= d_right_area,
                    d_left_area <= d_right_area + d_tol,
                )
            )
            return d_masks[0], d_masks[1]

        skip_covered_by_probe = bool(
            getattr(area_owned, "_many_vs_one_left_containment_bypass_applied", False)
        )

        def _evaluate_cover_mask(left_owned, right_input) -> np.ndarray | None:
            maybe_covers, maybe_covered_by = _area_candidate_masks(left_owned, right_input)
            predicates = ("covers",) if skip_covered_by_probe else ("covers", "covered_by")
            expressions = binary_predicate_expressions(
                predicates,
                left_owned,
                right_input,
                dispatch_mode=ExecutionMode.GPU,
                operation_prefix="overlay.keep_geom_type_cover",
            )
            if expressions is None or "covers" not in expressions:
                return None

            d_cover_mask = cp.asarray(expressions["covers"].values, dtype=cp.bool_) & maybe_covers
            if not skip_covered_by_probe:
                covered_by_expression = expressions.get("covered_by")
                if covered_by_expression is None:
                    return None
                d_cover_mask = d_cover_mask | (
                    cp.asarray(covered_by_expression.values, dtype=cp.bool_) & maybe_covered_by
                )

            return _overlay_device_to_host(
                d_cover_mask,
                reason=("overlay keep-geometry-type cover classification mask host boundary"),
                dtype=bool,
            )

        if use_source_rows:
            left_owned = left_source_owned
            right_owned = right_source_owned
            if _is_device_array(left_rows) or _is_device_array(right_rows):
                d_warning_rows = cp.asarray(warning_rows, dtype=cp.int64)
                source_left_rows = cp.asarray(left_rows, dtype=cp.int64)[d_warning_rows]
                source_right_rows = cp.asarray(right_rows, dtype=cp.int64)[d_warning_rows]
            else:
                source_left_rows = np.asarray(left_rows, dtype=np.intp)[warning_rows]
                source_right_rows = np.asarray(right_rows, dtype=np.intp)[warning_rows]
        else:
            left_owned = left_pairs_owned
            right_owned = right_pairs_owned
            source_left_rows = warning_rows.astype(np.intp, copy=False)
            source_right_rows = warning_rows.astype(np.intp, copy=False)

        if not _is_device_array(source_left_rows) and not _is_device_array(source_right_rows):
            unique_left_rows = np.unique(source_left_rows)
            unique_right_rows = np.unique(source_right_rows)
            if unique_right_rows.size == 1 and source_left_rows.size > 1:
                left_eval = _take_owned_rows(left_owned, source_left_rows)
                right_one = _take_owned_rows(right_owned, unique_right_rows)
                return _evaluate_cover_mask(left_eval, right_one)

            if unique_left_rows.size == 1 and source_right_rows.size > 1:
                right_eval = _take_owned_rows(right_owned, source_right_rows)
                left_one = _take_owned_rows(left_owned, unique_left_rows)
                return _evaluate_cover_mask(right_eval, left_one)

        left_eval = _take_owned_rows(left_owned, source_left_rows)
        right_eval = _take_owned_rows(right_owned, source_right_rows)
        return _evaluate_cover_mask(left_eval, right_eval)
    except _OverlayNativeConstructiveDeclined:
        raise


def _native_polygon_keep_geom_type_positive_area_mask(
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    kept_rows: np.ndarray,
    *,
    area_owned=None,
    overlap_area: np.ndarray | None = None,
    left_pairs: GeoSeries | None = None,
    right_pairs: GeoSeries | None = None,
) -> np.ndarray | None:
    """Return rows whose polygon output has finite, strictly positive area."""
    kept_rows = np.asarray(kept_rows, dtype=np.intp)
    if kept_rows.size == 0 or area_owned is None:
        return None
    from vibespatial.runtime.residency import Residency

    def _device_positive_area_mask() -> np.ndarray | None:
        try:
            d_keep = _native_polygon_keep_geom_type_positive_area_device_mask(
                left_source,
                right_source,
                left_rows,
                right_rows,
                kept_rows,
                area_owned=area_owned,
                overlap_area=overlap_area,
                left_pairs=left_pairs,
                right_pairs=right_pairs,
            )
            if d_keep is None:
                return None

            return np.asarray(
                globals()["get_cuda_runtime"]().copy_device_to_host(
                    d_keep,
                    reason=("overlay keep-geometry-type positive-area terminal mask export"),
                ),
                dtype=bool,
            )
        except _OverlayNativeConstructiveDeclined:
            raise

    device_mask = _device_positive_area_mask()
    if device_mask is not None:
        return device_mask

    if overlap_area is None:
        target = area_owned
        if getattr(area_owned, "residency", None) is Residency.DEVICE:
            try:
                import cupy as cp
            except ModuleNotFoundError:  # pragma: no cover
                target = area_owned.take(kept_rows.astype(np.int64, copy=False))
            else:
                target = area_owned.device_take(
                    cp.asarray(kept_rows, dtype=cp.int64),
                    host_indices_for_sizing=kept_rows.astype(np.int64, copy=False),
                )
        else:
            target = area_owned.take(kept_rows.astype(np.int64, copy=False))
        if getattr(target, "residency", None) is Residency.DEVICE and has_gpu_runtime():
            from vibespatial.constructive.measurement import _area_gpu_device_fp64

            overlap_area = np.asarray(
                globals()["get_cuda_runtime"]().copy_device_to_host(
                    _area_gpu_device_fp64(target),
                    reason="overlay keep-geometry-type positive-area fp64 terminal export",
                ),
                dtype=np.float64,
            )
        else:
            from vibespatial.constructive.measurement import area_owned as measure_area_owned

            overlap_area = np.asarray(measure_area_owned(target), dtype=np.float64)
    else:
        overlap_area = np.asarray(overlap_area, dtype=np.float64)
        if overlap_area.size != kept_rows.size:
            return None
    return np.isfinite(overlap_area) & (overlap_area > 0.0)


def _native_polygon_keep_geom_type_positive_area_device_mask(
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    kept_rows,
    *,
    area_owned=None,
    overlap_area=None,
    left_pairs: GeoSeries | None = None,
    right_pairs: GeoSeries | None = None,
):
    """Return a device mask preserving every finite positive-area row."""
    if area_owned is None or cp is None or not has_gpu_runtime():
        return None

    from vibespatial.runtime.residency import Residency

    if getattr(area_owned, "residency", None) is not Residency.DEVICE:
        return None

    try:
        from vibespatial.constructive.measurement import _area_gpu_device_fp64

        d_kept_rows = cp.asarray(kept_rows, dtype=cp.int64)
        if int(d_kept_rows.size) == 0:
            return cp.empty(0, dtype=cp.bool_)
        if overlap_area is None:
            d_overlap_area = _area_gpu_device_fp64(
                area_owned.device_take(d_kept_rows),
            )
        elif hasattr(overlap_area, "__cuda_array_interface__"):
            d_overlap_area = cp.asarray(overlap_area, dtype=cp.float64)
            if int(d_overlap_area.size) != int(d_kept_rows.size):
                return None
        else:
            overlap_array = np.asarray(overlap_area, dtype=np.float64)
            if overlap_array.size != int(d_kept_rows.size):
                return None
            d_overlap_area = cp.asarray(overlap_array, dtype=cp.float64)

        return cp.isfinite(d_overlap_area) & (d_overlap_area > 0.0)
    except _OverlayNativeConstructiveDeclined:
        raise


def _host_polygon_keep_geom_type_positive_area_mask(
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    kept_rows: np.ndarray,
    *,
    area_pairs: GeoSeries,
    left_pairs: GeoSeries | None = None,
    right_pairs: GeoSeries | None = None,
) -> np.ndarray | None:
    """Return kept polygon rows with finite positive area."""
    kept_rows = np.asarray(kept_rows, dtype=np.intp)
    if kept_rows.size == 0:
        return None

    area_values = _take_geoseries_object_values(area_pairs, kept_rows)
    overlap_area = np.asarray(shapely.area(area_values), dtype=np.float64)
    return np.isfinite(overlap_area) & (overlap_area > 0.0)


def _clear_device_exact_keep_geom_type_warnings(
    warning_mask: np.ndarray,
    keep_mask: np.ndarray,
    *,
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    area_owned=None,
    left_pairs: GeoSeries | None = None,
    right_pairs: GeoSeries | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop warning rows that device predicates prove do not need host probing."""
    warning_rows = np.flatnonzero(warning_mask).astype(np.intp, copy=False)
    if warning_rows.size == 0:
        return warning_mask, warning_rows

    kept_warning_rows = warning_rows[np.asarray(keep_mask[warning_rows], dtype=bool)]
    if kept_warning_rows.size == 0:
        return warning_mask, warning_rows

    if not _many_vs_one_keep_geom_type_cover_probe_needed(
        left_source,
        right_source,
        left_rows,
        right_rows,
        kept_warning_rows,
        area_owned=area_owned,
    ):
        return warning_mask, warning_rows

    cover_mask = _device_polygon_keep_geom_type_cover_mask(
        left_source,
        right_source,
        left_rows,
        right_rows,
        kept_warning_rows,
        area_owned=area_owned,
        left_pairs=left_pairs,
        right_pairs=right_pairs,
    )
    if cover_mask is None or cover_mask.size != kept_warning_rows.size:
        return warning_mask, warning_rows

    resolved_rows = np.asarray(cover_mask, dtype=bool)
    if not resolved_rows.any():
        return warning_mask, warning_rows

    warning_mask = np.asarray(warning_mask, dtype=bool).copy()
    warning_mask[kept_warning_rows[resolved_rows]] = False
    return warning_mask, np.flatnonzero(warning_mask).astype(np.intp, copy=False)


def _many_vs_one_keep_geom_type_cover_probe_needed(
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    warning_rows: np.ndarray,
    *,
    area_owned=None,
) -> bool:
    """Return True when kept warning rows might still cover the single right polygon."""
    if warning_rows.size == 0:
        return False
    if not bool(getattr(area_owned, "_many_vs_one_left_containment_bypass_applied", False)):
        return True
    if left_source is None or right_source is None or left_rows is None or right_rows is None:
        return True

    left_source_owned = getattr(left_source.values, "_owned", None)
    right_source_owned = getattr(right_source.values, "_owned", None)
    if left_source_owned is None or right_source_owned is None:
        return True

    from vibespatial.runtime.residency import Residency

    if (
        left_source_owned.residency is not Residency.DEVICE
        or right_source_owned.residency is not Residency.DEVICE
    ):
        return True

    if _is_device_array(left_rows) or _is_device_array(right_rows):
        return True

    source_right_rows = np.asarray(right_rows, dtype=np.intp)[warning_rows]
    unique_right_rows = np.unique(source_right_rows)
    if unique_right_rows.size != 1:
        return True

    try:
        import cupy as cp

        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

        compute_geometry_bounds_device(left_source_owned)
        compute_geometry_bounds_device(right_source_owned)
        d_left_bounds = cp.asarray(left_source_owned.device_state.row_bounds).reshape(
            left_source_owned.row_count,
            4,
        )
        d_right_bounds = cp.asarray(right_source_owned.device_state.row_bounds).reshape(
            right_source_owned.row_count,
            4,
        )
        d_warning_left_rows = cp.asarray(
            np.asarray(left_rows, dtype=np.intp)[warning_rows],
            dtype=cp.int64,
        )
        d_warning_left_bounds = d_left_bounds[d_warning_left_rows]
        d_right_bounds_one = d_right_bounds[int(unique_right_rows[0])]
        maybe_covers = (
            (d_warning_left_bounds[:, 0] <= d_right_bounds_one[0])
            & (d_warning_left_bounds[:, 1] <= d_right_bounds_one[1])
            & (d_warning_left_bounds[:, 2] >= d_right_bounds_one[2])
            & (d_warning_left_bounds[:, 3] >= d_right_bounds_one[3])
        )
        return _overlay_bool_scalar(
            cp.any(maybe_covers),
            reason="overlay many-vs-one keep-geom-type cover-probe scalar fence",
        )
    except _OverlayNativeConstructiveDeclined:
        raise


def _repair_invalid_polygon_output_rows(
    geometries: GeoSeries,
    *,
    preserve_lower_dimensional: bool = True,
) -> GeoSeries:
    """Repair invalid polygon rows from the rectangle exact path when present.

    The rectangle intersection kernel can emit polygon rows with zero-area
    duplicate-edge spikes on boundary-overlap cases. Those rows are still
    geometrically equivalent after ``make_valid`` but can change convex-hull
    fingerprints and public GeoPandas validity semantics. Repair only rows
    explicitly flagged by the kernel so the normal fast path stays untouched.
    """
    owned = getattr(geometries.values, "_owned", None)
    repair_complete = bool(
        getattr(owned, "_polygon_rect_boundary_repair_complete", False)
        or getattr(geometries.values, "_polygon_rect_boundary_repair_complete", False)
    )
    if repair_complete and owned is not None:
        cached_validity = owned._current_cached_validity_mask()
        if cached_validity is not None and bool(np.all(cached_validity)):
            return geometries

    overlap_mask = getattr(owned, "_polygon_rect_boundary_overlap", None)
    if overlap_mask is None:
        if len(geometries) > 5000:
            return geometries
        suspect_rows = np.arange(len(geometries), dtype=np.intp)
    else:
        overlap_mask = _overlay_host_bool_mask_sparse_first(
            overlap_mask,
            length=len(geometries),
            dense_reason="overlay rectangle-overlap repair mask host boundary",
            sparse_reason="overlay rectangle-overlap repair rows host boundary",
        )
        if overlap_mask is not None and overlap_mask.any():
            suspect_rows = np.flatnonzero(overlap_mask).astype(np.intp, copy=False)
        else:
            if len(geometries) > 5000:
                return geometries
            suspect_rows = np.arange(len(geometries), dtype=np.intp)

    suspect_values: np.ndarray | None = None
    if owned is not None:
        from vibespatial.runtime.residency import Residency

        suspect_owned = owned
        if suspect_rows.size != len(geometries):
            if owned.residency is Residency.DEVICE and has_gpu_runtime():
                try:
                    import cupy as cp

                    suspect_owned = owned.device_take(cp.asarray(suspect_rows, dtype=cp.int64))
                except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
                    suspect_owned = owned.take(suspect_rows.astype(np.int64, copy=False))
            else:
                suspect_owned = owned.take(suspect_rows.astype(np.int64, copy=False))
        if suspect_owned.residency is Residency.DEVICE and has_gpu_runtime():
            cached_validity = suspect_owned._current_cached_validity_mask()
            if cached_validity is not None:
                invalid_local_rows = np.flatnonzero(
                    ~np.asarray(cached_validity, dtype=bool),
                ).astype(np.int64, copy=False)
            else:
                try:
                    from vibespatial.constructive.make_valid_pipeline import (
                        _try_device_validity_expression_rows,
                    )

                    device_rows = _try_device_validity_expression_rows(
                        suspect_owned,
                        row_count=suspect_owned.row_count,
                    )
                except _OverlayNativeConstructiveDeclined:
                    raise
                if device_rows is None:
                    raise _OverlayNativeConstructiveDeclined(
                        "polygon output repair requires device validity rows"
                    )
                invalid_local_rows = device_rows.repaired_rows
            if invalid_local_rows.size == 0:
                return geometries
            if not preserve_lower_dimensional:
                from vibespatial.constructive.make_valid_pipeline import make_valid_owned

                mv_result = make_valid_owned(
                    owned=suspect_owned,
                    dispatch_mode=ExecutionMode.GPU,
                )
                if (
                    mv_result.owned is not None
                    and mv_result.owned.row_count == suspect_owned.row_count
                ):
                    if suspect_rows.size == len(geometries):
                        repaired_owned = mv_result.owned
                    else:
                        from vibespatial.geometry.owned import (
                            concat_owned_scatter,
                            device_concat_owned_scatter,
                        )

                        try:
                            import cupy as cp

                            d_suspect_rows = cp.asarray(suspect_rows, dtype=cp.int64)
                        except ModuleNotFoundError:  # pragma: no cover
                            d_suspect_rows = None
                        if (
                            owned.residency is Residency.DEVICE
                            and mv_result.owned.residency is Residency.DEVICE
                            and d_suspect_rows is not None
                        ):
                            repaired_owned = device_concat_owned_scatter(
                                owned,
                                mv_result.owned,
                                d_suspect_rows,
                            )
                        else:
                            repaired_owned = concat_owned_scatter(
                                owned,
                                mv_result.owned,
                                suspect_rows.astype(np.int64, copy=False),
                            )
                    repaired = GeoSeries(
                        GeometryArray.from_owned(repaired_owned, crs=geometries.crs),
                        index=geometries.index,
                        crs=geometries.crs,
                    )
                    return _attach_polygon_rect_overlap_mask(repaired, overlap_mask)

            record_fallback_event(
                surface="geopandas.overlay.intersection",
                reason=(
                    "polygon output repair requires GEOS make_valid to preserve "
                    "lower-dimensional public semantics"
                ),
                detail=(
                    f"rows={len(geometries)}, "
                    f"suspect_rows={suspect_rows.size}, "
                    f"invalid_rows={invalid_local_rows.size}"
                ),
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.CPU,
                pipeline="overlay._repair_invalid_polygon_output_rows",
                d2h_transfer=True,
            )
            if _is_device_array(invalid_local_rows):
                from vibespatial.cuda._runtime import get_cuda_runtime

                invalid_local_rows = get_cuda_runtime().copy_device_to_host(
                    invalid_local_rows,
                    reason=("overlay invalid polygon repair rows GEOS compatibility boundary"),
                )
            invalid_local_rows = np.asarray(
                invalid_local_rows,
                dtype=np.intp,
            )
            invalid_mask = np.zeros(suspect_owned.row_count, dtype=bool)
            invalid_mask[invalid_local_rows] = True

        from vibespatial.constructive.validity import is_valid_owned

        if suspect_owned.residency is not Residency.DEVICE or not has_gpu_runtime():
            invalid_mask = ~np.asarray(is_valid_owned(suspect_owned), dtype=bool)
    else:
        all_values = np.asarray(geometries.values._data, dtype=object)
        suspect_values = all_values[suspect_rows]
        invalid_mask = ~np.asarray(shapely.is_valid(suspect_values), dtype=bool)
    if not invalid_mask.any():
        return geometries

    all_values = np.asarray(geometries.values._data, dtype=object)
    if suspect_values is None:
        suspect_values = all_values[suspect_rows]
    repaired_values = np.asarray(
        shapely.make_valid(suspect_values[invalid_mask]),
        dtype=object,
    )
    all_values[suspect_rows[invalid_mask]] = repaired_values
    repaired = GeoSeries(all_values, index=geometries.index, crs=geometries.crs)
    return _attach_polygon_rect_overlap_mask(repaired, overlap_mask)


def _attach_polygon_rect_overlap_mask(
    geometries: GeoSeries,
    overlap_mask: np.ndarray | None,
) -> GeoSeries:
    if overlap_mask is None:
        return geometries
    overlap_mask = np.asarray(overlap_mask, dtype=bool)
    owned = getattr(geometries.values, "_owned", None)
    if owned is not None:
        owned._polygon_rect_boundary_overlap = overlap_mask
    geometries.values._polygon_rect_boundary_overlap = overlap_mask
    return geometries


def _owned_positive_polygon_mask_and_areas(
    owned,
    *,
    candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return polygon-family positive-area mask plus measured row areas."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS
    from vibespatial.runtime.residency import Residency

    if owned.residency is Residency.DEVICE and has_gpu_runtime():
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            from vibespatial.constructive.measurement import _area_gpu_device_fp64
            from vibespatial.cuda._runtime import get_cuda_runtime

            device_state = owned._ensure_device_state(preserve_indexed_view=True)
            d_validity = cp.asarray(device_state.validity)
            d_tags = cp.asarray(device_state.tags)
            if candidate_mask is None:
                d_candidate_mask = d_validity
            else:
                candidate_mask = np.asarray(candidate_mask, dtype=bool)
                if candidate_mask.size != owned.row_count:
                    raise ValueError("candidate_mask size must match owned row count")
                d_candidate_mask = cp.asarray(candidate_mask, dtype=cp.bool_) & d_validity
            d_polygon_mask = (d_tags == FAMILY_TAGS[GeometryFamily.POLYGON]) | (
                d_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
            )
            d_areas = _area_gpu_device_fp64(owned)
            d_keep = d_candidate_mask & d_polygon_mask & cp.isfinite(d_areas) & (d_areas > 0.0)
            row_count = int(owned.row_count)
            area_by_row = np.full(row_count, np.nan, dtype=np.float64)
            d_rows = cp.flatnonzero(d_keep).astype(cp.int64, copy=False)
            positive_count = int(d_rows.size)
            if positive_count == 0:
                return np.zeros(row_count, dtype=bool), area_by_row
            if positive_count == row_count:
                return np.ones(row_count, dtype=bool), area_by_row
            runtime = get_cuda_runtime()
            if positive_count * np.dtype(np.int64).itemsize < row_count:
                rows = np.asarray(
                    runtime.copy_device_to_host(
                        d_rows,
                        reason=(
                            "overlay keep-geometry-type polygonal positive-area "
                            "terminal rows export"
                        ),
                    ),
                    dtype=np.intp,
                )
                keep_mask = np.zeros(row_count, dtype=bool)
                keep_mask[rows] = True
                return keep_mask, area_by_row
            return (
                np.asarray(
                    runtime.copy_device_to_host(
                        d_keep,
                        reason=(
                            "overlay keep-geometry-type polygonal positive-area "
                            "terminal mask export"
                        ),
                    ),
                    dtype=bool,
                ),
                area_by_row,
            )

    validity = np.asarray(owned.validity, dtype=bool)
    area_by_row = np.full(validity.size, np.nan, dtype=np.float64)
    if not validity.any():
        return validity, area_by_row

    if candidate_mask is None:
        candidate_mask = validity.copy()
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.size != validity.size:
            raise ValueError("candidate_mask size must match owned row count")
        candidate_mask = candidate_mask & validity

    tags = np.asarray(owned.tags)
    row_offsets = np.asarray(owned.family_row_offsets)
    keep_mask = np.zeros(len(tags), dtype=bool)

    for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        family_tag = FAMILY_TAGS[family]
        family_mask = candidate_mask & (tags == family_tag)
        if not family_mask.any():
            continue
        owned._ensure_host_family_structure(family)
        family_indices = np.flatnonzero(family_mask)
        family_rows = row_offsets[family_mask]
        empty_mask = np.asarray(owned.families[family].empty_mask, dtype=bool)
        if empty_mask.size == 0 or np.any((family_rows < 0) | (family_rows >= empty_mask.size)):
            candidate_rows = family_indices
        else:
            candidate_rows = family_indices[~empty_mask[family_rows]]
        if candidate_rows.size == 0:
            continue
        from vibespatial.constructive.measurement import area_owned

        candidate_area = np.asarray(
            area_owned(owned.take(candidate_rows)),
            dtype=np.float64,
        )
        area_by_row[candidate_rows] = candidate_area
        keep_mask[candidate_rows] = candidate_area > 0.0

    return keep_mask, area_by_row


def _owned_positive_polygon_mask(
    owned,
    *,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return polygon-family rows backed by strictly positive area."""
    keep_mask, _ = _owned_positive_polygon_mask_and_areas(
        owned,
        candidate_mask=candidate_mask,
    )
    return keep_mask


def _strip_non_polygon_collection_parts(geometries: np.ndarray) -> np.ndarray:
    """Replace GeometryCollections with polygon-only equivalents."""
    if len(geometries) == 0:
        return geometries

    type_ids = shapely.get_type_id(geometries)
    collection_rows = np.flatnonzero(type_ids == _SHAPELY_TYPE_ID_GEOMETRYCOLLECTION)
    if collection_rows.size == 0:
        return geometries

    cleaned = geometries.copy()
    for row_index in collection_rows:
        pending = np.asarray([geometries[int(row_index)]], dtype=object)
        polygon_parts: list[object] = []
        while len(pending) > 0:
            pending_type_ids = shapely.get_type_id(pending)
            collection_mask = pending_type_ids == _SHAPELY_TYPE_ID_GEOMETRYCOLLECTION
            non_collection = pending[~collection_mask]
            if non_collection.size > 0:
                non_empty_mask = ~shapely.is_empty(non_collection)
                non_collection = non_collection[non_empty_mask]
                if non_collection.size > 0:
                    non_collection_type_ids = shapely.get_type_id(non_collection)
                    polygon_mask = (non_collection_type_ids == _SHAPELY_TYPE_ID_POLYGON) | (
                        non_collection_type_ids == _SHAPELY_TYPE_ID_MULTIPOLYGON
                    )
                    if np.any(polygon_mask):
                        polygon_parts.extend(non_collection[polygon_mask].tolist())
            if not np.any(collection_mask):
                break
            pending = shapely.get_parts(pending[collection_mask])

        if not polygon_parts:
            cleaned[int(row_index)] = None
        elif len(polygon_parts) == 1:
            cleaned[int(row_index)] = polygon_parts[0]
        else:
            cleaned[int(row_index)] = shapely.union_all(np.asarray(polygon_parts, dtype=object))
    return cleaned


def _filter_polygon_intersection_rows_for_keep_geom_type(
    left_pairs: GeoSeries | None,
    right_pairs: GeoSeries | None,
    area_pairs: GeoSeries,
    *,
    keep_geom_type_warning: bool,
    left_source: GeoSeries | None = None,
    right_source: GeoSeries | None = None,
    left_rows: np.ndarray | None = None,
    right_rows: np.ndarray | None = None,
) -> tuple[GeoSeries, int, np.ndarray]:
    """Keep polygonal area rows only and classify dropped lower-dimensional remnants."""
    area_overlap_mask = getattr(area_pairs.values, "_polygon_rect_boundary_overlap", None)
    area_exact_polygon_only_mask_raw = _polygon_rect_exact_polygon_only_mask_raw(area_pairs)
    area_exact_area_mask_raw = getattr(
        area_pairs.values,
        "_polygon_intersection_exact_area",
        None,
    )
    area_exact_polygon_only_mask = None

    def _area_exact_polygon_only_host_mask() -> np.ndarray | None:
        nonlocal area_exact_polygon_only_mask
        if area_exact_polygon_only_mask is None:
            area_exact_polygon_only_mask = _polygon_rect_exact_polygon_only_mask(area_pairs)
        return area_exact_polygon_only_mask

    if area_overlap_mask is not None:
        area_overlap_mask = _overlay_host_bool_mask_sparse_first(
            area_overlap_mask,
            length=len(area_pairs),
            dense_reason="overlay keep-geom-type area-overlap mask host boundary",
            sparse_reason="overlay keep-geom-type area-overlap rows host boundary",
        )

    area_owned = getattr(area_pairs.values, "_owned", None)
    allow_sparse_exact_position_export = True
    if area_owned is not None:
        from vibespatial.runtime.residency import Residency

        def _series_has_device_owned(series: GeoSeries | None) -> bool:
            if series is None:
                return False
            owned = getattr(series.values, "_owned", None)
            return owned is not None and getattr(owned, "residency", None) is Residency.DEVICE

        allow_sparse_exact_position_export = not (
            getattr(area_owned, "residency", None) is Residency.DEVICE
            or _series_has_device_owned(left_source)
            or _series_has_device_owned(right_source)
            or _series_has_device_owned(left_pairs)
            or _series_has_device_owned(right_pairs)
        )
    if area_overlap_mask is None and area_owned is not None:
        area_overlap_mask = getattr(area_owned, "_polygon_rect_boundary_overlap", None)
        if area_overlap_mask is not None:
            area_overlap_mask = _overlay_host_bool_mask_sparse_first(
                area_overlap_mask,
                length=len(area_pairs),
                dense_reason="overlay keep-geom-type owned area-overlap mask host boundary",
                sparse_reason="overlay keep-geom-type owned area-overlap rows host boundary",
            )
    if area_exact_area_mask_raw is None and area_owned is not None:
        area_exact_area_mask_raw = getattr(
            area_owned,
            "_polygon_intersection_exact_area",
            None,
        )

    area_exact_values, area_exact_mask = _exact_intersection_cache_from_metadata_owner(
        area_pairs.values,
        length=len(area_pairs),
        sparse_positions_reason=(
            "overlay keep-geom-type exact-cache sparse positions host boundary"
        ),
        allow_device_position_export=allow_sparse_exact_position_export,
    )
    area_containment_passthrough_mask = getattr(
        area_pairs.values,
        "_many_vs_one_containment_passthrough_mask",
        None,
    )

    if area_owned is not None:
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import FAMILY_TAGS
        from vibespatial.runtime.residency import Residency, TransferTrigger

        def _requires_device_to_host_probe(*series) -> bool:
            for series_obj in series:
                if series_obj is None:
                    continue
                owned = getattr(series_obj.values, "_owned", None)
                if owned is not None and getattr(owned, "residency", None) is Residency.DEVICE:
                    return True
            aligned_left_owned, aligned_right_owned = _aligned_pair_owned_from_area(
                area_owned,
            )
            return (
                aligned_left_owned is not None
                and aligned_right_owned is not None
                and getattr(aligned_left_owned, "residency", None) is Residency.DEVICE
                and getattr(aligned_right_owned, "residency", None) is Residency.DEVICE
            )

        def _decline_keep_geom_type_host_probe(detail: str) -> None:
            if not _requires_device_to_host_probe(
                left_source,
                right_source,
                left_pairs,
                right_pairs,
            ):
                return
            raise _OverlayNativeConstructiveDeclined(
                f"keep_geom_type semantic probe requires host geometry materialization ({detail})"
            )

        left_source_owned = (
            getattr(left_source.values, "_owned", None) if left_source is not None else None
        )
        right_source_owned = (
            getattr(right_source.values, "_owned", None) if right_source is not None else None
        )
        if (
            left_source_owned is not None
            and right_source_owned is not None
            and left_source_owned.residency is Residency.DEVICE
            and right_source_owned.residency is Residency.DEVICE
            and area_owned.residency is not Residency.DEVICE
        ):
            try:
                area_owned = area_owned.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="keep_geom_type classification promoted area rows to device",
                )
                area_pairs = GeoSeries(
                    GeometryArray.from_owned(area_owned, crs=area_pairs.crs),
                    index=area_pairs.index,
                    crs=area_pairs.crs,
                )
            except _OverlayNativeConstructiveDeclined:
                raise

        row_count = int(area_owned.row_count)
        tags = np.empty(row_count, dtype=np.int8)
        keep_mask = np.zeros(row_count, dtype=bool)
        overlap_area_by_row = np.full(row_count, np.nan, dtype=np.float64)
        owned_metadata_consistent = True
        rect_overlap_mask = None
        if area_exact_values is None or area_exact_mask is None:
            area_exact_values, area_exact_mask = _exact_intersection_cache_from_metadata_owner(
                area_owned,
                length=len(area_pairs),
                sparse_positions_reason=(
                    "overlay keep-geom-type owned exact-cache sparse positions host boundary"
                ),
                allow_device_position_export=allow_sparse_exact_position_export,
            )
        if area_containment_passthrough_mask is None:
            area_containment_passthrough_mask = getattr(
                area_owned,
                "_many_vs_one_containment_passthrough_mask",
                None,
            )
        if area_containment_passthrough_mask is not None:
            area_containment_passthrough_mask = _overlay_host_bool_mask_sparse_first(
                area_containment_passthrough_mask,
                length=len(area_pairs),
                dense_reason="overlay keep-geom-type containment-passthrough mask host boundary",
                sparse_reason="overlay keep-geom-type containment-passthrough rows host boundary",
            )

        def _device_keep_geom_type_filter_result():
            if area_owned.residency is not Residency.DEVICE or not has_gpu_runtime():
                return None
            cupy = globals().get("cp")
            if cupy is None:
                return None
            rect_overlap_has_warning_rows = area_overlap_mask is not None and bool(
                np.any(area_overlap_mask)
            )
            if (
                rect_overlap_has_warning_rows
                or area_exact_values is not None
                or area_exact_mask is not None
            ):
                return None
            if keep_geom_type_warning and not _requires_device_to_host_probe(
                left_source,
                right_source,
                left_pairs,
                right_pairs,
            ):
                return None
            try:
                from vibespatial.constructive.measurement import _area_gpu_device_fp64

                device_state = area_owned._ensure_device_state(
                    preserve_indexed_view=True,
                )
                d_validity = cupy.asarray(device_state.validity, dtype=cupy.bool_)
                d_tags = cupy.asarray(device_state.tags, dtype=cupy.int8)
                d_polygon_mask = (d_tags == FAMILY_TAGS[GeometryFamily.POLYGON]) | (
                    d_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
                )
                d_areas = _area_gpu_device_fp64(area_owned)
                d_keep = d_validity & d_polygon_mask & cupy.isfinite(d_areas) & (d_areas > 0.0)
                d_exact_area_proven = cupy.zeros(row_count, dtype=cupy.bool_)
                for exact_area_proof in (
                    area_exact_polygon_only_mask_raw,
                    area_exact_area_mask_raw,
                ):
                    if exact_area_proof is None:
                        continue
                    d_exact_candidate = cupy.asarray(
                        exact_area_proof,
                        dtype=cupy.bool_,
                    ).reshape(-1)
                    if int(d_exact_candidate.size) == row_count:
                        d_exact_area_proven |= d_exact_candidate
                d_positive_rows = cupy.flatnonzero(
                    d_keep & ~d_exact_area_proven,
                ).astype(
                    cupy.int64,
                    copy=False,
                )
                d_meaningful_area = _native_polygon_keep_geom_type_positive_area_device_mask(
                    left_source,
                    right_source,
                    left_rows,
                    right_rows,
                    d_positive_rows,
                    area_owned=area_owned,
                    overlap_area=d_areas[d_positive_rows],
                    left_pairs=left_pairs,
                    right_pairs=right_pairs,
                )
                if d_meaningful_area is not None and int(d_meaningful_area.size) == int(
                    d_positive_rows.size
                ):
                    d_keep = d_keep.copy()
                    d_keep[d_positive_rows] &= cupy.asarray(
                        d_meaningful_area,
                        dtype=cupy.bool_,
                    )
                    d_positive_rows = cupy.flatnonzero(d_keep).astype(
                        cupy.int64,
                        copy=False,
                    )
                from vibespatial.geometry.owned import device_mask_owned_capacity

                filtered_owned = device_mask_owned_capacity(area_owned, d_keep)
                _mark_owned_logical_polygon_valid_nonempty(
                    filtered_owned,
                    all_rows_valid=False,
                )
                filtered = GeoSeries(
                    GeometryArray.from_owned(
                        filtered_owned,
                        crs=area_pairs.crs,
                    ),
                    crs=area_pairs.crs,
                )
                dropped_count = 0
                warning_count_event_recorded = False
                if keep_geom_type_warning:
                    warning_candidate_rows = None
                    exact_polygon_only = None
                    if area_exact_polygon_only_mask_raw is not None:
                        exact_polygon_only = cupy.asarray(
                            area_exact_polygon_only_mask_raw,
                            dtype=cupy.bool_,
                        ).reshape(-1)
                        if int(exact_polygon_only.size) == row_count:
                            warning_candidate_rows = cupy.flatnonzero(
                                ~(d_keep & exact_polygon_only),
                            ).astype(cupy.int64, copy=False)
                        else:
                            exact_polygon_only = None
                    direct_warning_dropped = None
                    if warning_candidate_rows is not None:
                        if int(warning_candidate_rows.size) == 0:
                            direct_warning_dropped = 0
                        else:
                            direct_warning_dropped = (
                                _device_count_dropped_polygon_intersection_warning_rows(
                                    area_owned,
                                    d_keep,
                                    warning_candidate_rows,
                                    left_source=left_source,
                                    right_source=right_source,
                                    left_rows=left_rows,
                                    right_rows=right_rows,
                                    left_pairs=left_pairs,
                                    right_pairs=right_pairs,
                                    warning_keep_mask=d_keep[warning_candidate_rows],
                                )
                            )
                    if direct_warning_dropped is not None:
                        dropped_count = int(direct_warning_dropped)
                        if dropped_count > 0:
                            record_dispatch_event(
                                surface="geopandas.overlay.intersection",
                                operation="keep_geom_type_warning_count",
                                implementation="device_boundary_warning_count",
                                reason=(
                                    "native exact-polygon metadata pruned "
                                    "keep-geom-type warning candidates and the "
                                    "device rowset kernel counted dropped "
                                    "lower-dimensional pieces without DE-9IM "
                                    "classification"
                                ),
                                detail=f"rows={int(dropped_count)}",
                                requested=ExecutionMode.GPU,
                                selected=ExecutionMode.GPU,
                            )
                            warning_count_event_recorded = True
                    else:
                        warning_mask = _device_polygon_keep_geom_type_warning_mask_from_de9im(
                            left_source,
                            right_source,
                            left_rows,
                            right_rows,
                            d_keep,
                            area_owned=area_owned,
                            left_pairs=left_pairs,
                            right_pairs=right_pairs,
                            return_device=True,
                            candidate_rows=warning_candidate_rows,
                        )
                        if warning_mask is not None and exact_polygon_only is not None:
                            warning_mask = cupy.asarray(
                                warning_mask,
                                dtype=cupy.bool_,
                            ) & ~(d_keep & exact_polygon_only)
                        if warning_mask is not None:
                            d_warning_mask = cupy.asarray(
                                warning_mask,
                                dtype=cupy.bool_,
                            )
                            warning_count = _overlay_int_scalar(
                                cupy.count_nonzero(d_warning_mask),
                                reason=(
                                    "overlay keep-geom-type warning count terminal scalar fence"
                                ),
                            )
                        else:
                            warning_count = 0
                        if warning_mask is not None and warning_count > 0:
                            d_warning_rows = cupy.flatnonzero(d_warning_mask).astype(
                                cupy.int64,
                                copy=False,
                            )
                            device_dropped = (
                                _device_count_dropped_polygon_intersection_warning_rows(
                                    area_owned,
                                    d_keep,
                                    d_warning_rows,
                                    left_source=left_source,
                                    right_source=right_source,
                                    left_rows=left_rows,
                                    right_rows=right_rows,
                                    left_pairs=left_pairs,
                                    right_pairs=right_pairs,
                                    warning_keep_mask=d_keep[d_warning_rows],
                                )
                            )
                            if device_dropped is None:
                                dropped_count = int(warning_count)
                                record_dispatch_event(
                                    surface="geopandas.overlay.intersection",
                                    operation="keep_geom_type_warning_count",
                                    implementation="device_advisory_warning_count",
                                    reason=(
                                        "device-backed keep-geom-type warning rows used "
                                        "the native advisory count after DE-9IM "
                                        "classification declined boundary refinement"
                                    ),
                                    detail=f"rows={int(dropped_count)}",
                                    requested=ExecutionMode.GPU,
                                    selected=ExecutionMode.GPU,
                                )
                                warning_count_event_recorded = True
                            else:
                                dropped_count = int(device_dropped)
                                if dropped_count > 0:
                                    record_dispatch_event(
                                        surface="geopandas.overlay.intersection",
                                        operation="keep_geom_type_warning_count",
                                        implementation="device_boundary_warning_count",
                                        reason=(
                                            "device-backed keep-geom-type warning "
                                            "rows were refined with native boundary "
                                            "classification"
                                        ),
                                        detail=f"rows={int(dropped_count)}",
                                        requested=ExecutionMode.GPU,
                                        selected=ExecutionMode.GPU,
                                    )
                                    warning_count_event_recorded = True
                        elif warning_mask is None:
                            kept_count = _overlay_int_scalar(
                                cupy.count_nonzero(d_keep),
                                reason=(
                                    "overlay keep-geom-type warning count terminal scalar fence"
                                ),
                            )
                            if kept_count < row_count:
                                dropped_count = row_count - kept_count
                if (
                    keep_geom_type_warning
                    and dropped_count > 0
                    and not warning_count_event_recorded
                ):
                    record_dispatch_event(
                        surface="geopandas.overlay.intersection",
                        operation="keep_geom_type_warning_count",
                        implementation="device_advisory_warning_count",
                        reason=(
                            "device-backed keep-geom-type warning rows used "
                            "the native advisory count instead of boundary "
                            "geometry reconstruction"
                        ),
                        detail=f"rows={int(dropped_count)}",
                        requested=ExecutionMode.GPU,
                        selected=ExecutionMode.GPU,
                    )
                return filtered, int(dropped_count), d_keep
            except _OverlayNativeConstructiveDeclined:
                raise

        device_filter_result = _device_keep_geom_type_filter_result()
        if device_filter_result is not None:
            return device_filter_result

        if area_owned.residency is Residency.DEVICE and has_gpu_runtime():
            keep_mask, overlap_area_by_row = _owned_positive_polygon_mask_and_areas(
                area_owned,
            )
        else:
            tags = area_owned.tags
            validity = area_owned.validity
            row_offsets = area_owned.family_row_offsets
            keep_mask = np.zeros(len(tags), dtype=bool)

            for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
                family_tag = FAMILY_TAGS[family]
                family_mask = validity & (tags == family_tag)
                if not family_mask.any():
                    continue
                area_owned._ensure_host_family_structure(family)
                family_rows = row_offsets[family_mask]
                empty_mask = np.asarray(area_owned.families[family].empty_mask, dtype=bool)
                if empty_mask.size == 0:
                    family_count = int(getattr(area_owned.families[family], "row_count", 0))
                    if (
                        area_overlap_mask is not None
                        and family_count > 0
                        and not np.any((family_rows < 0) | (family_rows >= family_count))
                    ):
                        keep_mask[np.flatnonzero(family_mask)] = True
                        continue
                    owned_metadata_consistent = False
                    break
                if np.any((family_rows < 0) | (family_rows >= empty_mask.size)):
                    owned_metadata_consistent = False
                    break
                keep_mask[np.flatnonzero(family_mask)] = ~empty_mask[family_rows]

        if owned_metadata_consistent:
            if area_owned.residency is not Residency.DEVICE or not has_gpu_runtime():
                positive_polygon_mask, overlap_area_by_row = _owned_positive_polygon_mask_and_areas(
                    area_owned,
                    candidate_mask=keep_mask,
                )
                keep_mask &= positive_polygon_mask
            kept_rows = np.flatnonzero(keep_mask).astype(np.intp, copy=False)
            if kept_rows.size > 0:
                positive_area_rows = kept_rows
                if area_containment_passthrough_mask is not None:
                    passthrough_positive = np.asarray(
                        area_containment_passthrough_mask[positive_area_rows],
                        dtype=bool,
                    )
                    if passthrough_positive.any():
                        positive_area_rows = positive_area_rows[~passthrough_positive]
                positive_area_mask = _native_polygon_keep_geom_type_positive_area_mask(
                    left_source,
                    right_source,
                    left_rows,
                    right_rows,
                    positive_area_rows,
                    area_owned=area_owned,
                    overlap_area=(
                        None
                        if area_owned.residency is Residency.DEVICE and has_gpu_runtime()
                        else overlap_area_by_row[positive_area_rows]
                    ),
                    left_pairs=left_pairs,
                    right_pairs=right_pairs,
                )
                if positive_area_mask is None:
                    if area_overlap_mask is not None and keep_geom_type_warning:
                        # Rectangle-kernel warning mode must not materialize every
                        # source row just to classify advisory warning counts. When
                        # native source/pair areas are unavailable, keep the prior
                        # warning-row-only host probe behavior.
                        positive_area_rows = np.empty(0, dtype=np.intp)
                    if positive_area_rows.size > 0:
                        positive_area_mask = _host_polygon_keep_geom_type_positive_area_mask(
                            left_source,
                            right_source,
                            left_rows,
                            right_rows,
                            positive_area_rows,
                            area_pairs=area_pairs,
                            left_pairs=left_pairs,
                            right_pairs=right_pairs,
                        )
                if (
                    positive_area_mask is not None
                    and positive_area_mask.size == positive_area_rows.size
                ):
                    positive_area_mask = np.asarray(positive_area_mask, dtype=bool)
                    keep_mask = np.asarray(keep_mask, dtype=bool).copy()
                    keep_mask[positive_area_rows] &= positive_area_mask
            dropped = 0
            if keep_geom_type_warning and len(tags) > 0:
                if area_exact_values is not None and area_exact_mask is not None:
                    area_exact_values = np.asarray(area_exact_values, dtype=object)
                    area_exact_mask = np.asarray(area_exact_mask, dtype=bool)
                    if area_exact_values.size != len(tags) or area_exact_mask.size != len(tags):
                        area_exact_values = None
                        area_exact_mask = None

                rect_overlap_mask = (
                    getattr(area_owned, "_polygon_rect_boundary_overlap", None)
                    if area_overlap_mask is None
                    else area_overlap_mask
                )
                if rect_overlap_mask is not None:
                    rect_overlap_mask = _overlay_host_bool_mask_sparse_first(
                        rect_overlap_mask,
                        length=len(tags),
                        dense_reason="overlay keep-geom-type rect-overlap mask host boundary",
                        sparse_reason="overlay keep-geom-type rect-overlap rows host boundary",
                    )

                if (
                    bool(np.all(keep_mask))
                    and rect_overlap_mask is not None
                    and not rect_overlap_mask.any()
                ):
                    filtered = area_pairs.reset_index(drop=True)
                    filtered = _attach_polygon_rect_overlap_mask(filtered, rect_overlap_mask)
                    return filtered, 0, keep_mask

                if rect_overlap_mask is not None:
                    warning_mask = (~keep_mask) | rect_overlap_mask
                    if area_exact_values is not None and area_exact_mask is not None:
                        warning_mask = np.asarray(warning_mask, dtype=bool).copy()
                        warning_mask[area_exact_mask] = (
                            _warning_candidate_mask_from_exact_intersection_values(
                                area_exact_values[area_exact_mask],
                                keep_mask[area_exact_mask],
                            )
                        )
                elif area_exact_values is not None and area_exact_mask is not None:
                    warning_mask = np.zeros(len(tags), dtype=bool)
                    warning_mask[area_exact_mask] = (
                        _warning_candidate_mask_from_exact_intersection_values(
                            area_exact_values[area_exact_mask],
                            keep_mask[area_exact_mask],
                        )
                    )
                    warning_mask[~area_exact_mask & ~keep_mask] = True
                    probe_rows = np.flatnonzero(~area_exact_mask & keep_mask).astype(
                        np.intp, copy=False
                    )
                    if probe_rows.size > 0:
                        _decline_keep_geom_type_host_probe(
                            f"rows={len(tags)}, probe_rows={probe_rows.size}"
                        )
                        if (
                            left_source is not None
                            and right_source is not None
                            and left_rows is not None
                            and right_rows is not None
                        ):
                            left_values = _take_geoseries_object_values(
                                left_source,
                                np.asarray(left_rows, dtype=np.intp)[probe_rows],
                            )
                            right_values = _take_geoseries_object_values(
                                right_source,
                                np.asarray(right_rows, dtype=np.intp)[probe_rows],
                            )
                        else:
                            if left_pairs is None or right_pairs is None:
                                raise ValueError(
                                    "left_pairs/right_pairs or source rows are required when "
                                    "keep_geom_type_warning=True"
                                )
                            left_values = _take_geoseries_object_values(left_pairs, probe_rows)
                            right_values = _take_geoseries_object_values(right_pairs, probe_rows)
                        probe_mask = _warning_candidate_mask_for_polygon_keep_geom_type(
                            left_values,
                            right_values,
                            keep_mask[probe_rows],
                        )
                        warning_mask[probe_rows] = probe_mask
                else:
                    warning_mask = _device_polygon_keep_geom_type_warning_mask_from_de9im(
                        left_source,
                        right_source,
                        left_rows,
                        right_rows,
                        keep_mask,
                        area_owned=area_owned,
                        left_pairs=left_pairs,
                        right_pairs=right_pairs,
                    )
                    if warning_mask is None:
                        warning_mask = np.zeros(len(tags), dtype=bool)
                        if (
                            left_source is not None
                            and right_source is not None
                            and left_rows is not None
                            and right_rows is not None
                        ):
                            source_left_rows = np.asarray(left_rows, dtype=np.intp)
                            source_right_rows = np.asarray(right_rows, dtype=np.intp)
                            device_source_owned = False
                            if (
                                left_source_owned is not None
                                and right_source_owned is not None
                                and left_source_owned.residency is Residency.DEVICE
                                and right_source_owned.residency is Residency.DEVICE
                            ):
                                device_source_owned = True

                            def _take_owned_rows(owned, rows: np.ndarray):
                                import cupy as cp

                                rows64 = np.asarray(rows, dtype=np.int64)
                                if rows64.size == 0:
                                    return owned.take(rows64)
                                return owned.device_take(
                                    cp.asarray(rows64, dtype=cp.int64),
                                    host_indices_for_sizing=rows64,
                                )

                            empty_rows = np.flatnonzero(~keep_mask).astype(np.intp, copy=False)
                            if empty_rows.size > 0:
                                if device_source_owned:
                                    from vibespatial.predicates.binary import (
                                        evaluate_binary_predicate,
                                    )

                                    empty_left = _take_owned_rows(
                                        left_source_owned,
                                        source_left_rows[empty_rows],
                                    )
                                    empty_right = _take_owned_rows(
                                        right_source_owned,
                                        source_right_rows[empty_rows],
                                    )
                                    warning_mask[empty_rows] = np.asarray(
                                        evaluate_binary_predicate(
                                            "intersects",
                                            empty_left,
                                            empty_right,
                                            dispatch_mode=ExecutionMode.GPU,
                                        ).values,
                                        dtype=bool,
                                    )
                                else:
                                    _decline_keep_geom_type_host_probe(
                                        f"rows={len(tags)}, dropped_rows={empty_rows.size}"
                                    )
                                    empty_left_values = _take_geoseries_object_values(
                                        left_source,
                                        source_left_rows[empty_rows],
                                    )
                                    empty_right_values = _take_geoseries_object_values(
                                        right_source,
                                        source_right_rows[empty_rows],
                                    )
                                    warning_mask[empty_rows] = np.asarray(
                                        shapely.intersects(empty_left_values, empty_right_values),
                                        dtype=bool,
                                    )

                            kept_rows = np.flatnonzero(keep_mask).astype(np.intp, copy=False)
                            if kept_rows.size > 0:
                                if device_source_owned and area_owned.residency is Residency.DEVICE:
                                    from vibespatial.constructive.boundary import boundary_owned
                                    from vibespatial.predicates.binary import (
                                        evaluate_binary_predicate,
                                    )

                                    kept_left = _take_owned_rows(
                                        left_source_owned,
                                        source_left_rows[kept_rows],
                                    )
                                    kept_right = _take_owned_rows(
                                        right_source_owned,
                                        source_right_rows[kept_rows],
                                    )
                                    kept_left_boundary = boundary_owned(kept_left)
                                    kept_right_boundary = boundary_owned(kept_right)
                                    warning_mask[kept_rows] = np.asarray(
                                        evaluate_binary_predicate(
                                            "intersects",
                                            kept_left_boundary,
                                            kept_right_boundary,
                                            dispatch_mode=ExecutionMode.GPU,
                                        ).values,
                                        dtype=bool,
                                    )
                                else:
                                    _decline_keep_geom_type_host_probe(
                                        f"rows={len(tags)}, kept_rows={kept_rows.size}"
                                    )
                                    kept_left_values = _take_geoseries_object_values(
                                        left_source,
                                        source_left_rows[kept_rows],
                                    )
                                    kept_right_values = _take_geoseries_object_values(
                                        right_source,
                                        source_right_rows[kept_rows],
                                    )
                                    warning_mask[kept_rows] = np.asarray(
                                        shapely.intersects(
                                            shapely.boundary(kept_left_values),
                                            shapely.boundary(kept_right_values),
                                        ),
                                        dtype=bool,
                                    )
                        else:
                            if left_pairs is None or right_pairs is None:
                                raise ValueError(
                                    "left_pairs/right_pairs or source rows are required when "
                                    "keep_geom_type_warning=True"
                                )
                            _decline_keep_geom_type_host_probe(
                                f"rows={len(tags)}, warning_rows={len(tags)}"
                            )
                            row_positions = np.arange(len(tags), dtype=np.intp)
                            left_values = _take_geoseries_object_values(left_pairs, row_positions)
                            right_values = _take_geoseries_object_values(right_pairs, row_positions)
                            warning_mask = _warning_candidate_mask_for_polygon_keep_geom_type(
                                left_values,
                                right_values,
                                keep_mask,
                            )

                exact_polygon_only_mask = _area_exact_polygon_only_host_mask()
                if exact_polygon_only_mask is not None:
                    safe_rows = np.asarray(keep_mask, dtype=bool) & exact_polygon_only_mask
                    if safe_rows.any():
                        warning_mask = np.asarray(warning_mask, dtype=bool).copy()
                        warning_mask[safe_rows] = False
                warning_rows = np.empty(0, dtype=np.intp)
                if np.any(warning_mask):
                    warning_rows = np.flatnonzero(np.asarray(warning_mask, dtype=bool)).astype(
                        np.intp,
                        copy=False,
                    )
                    warning_rows_have_exact_values = (
                        warning_rows.size > 0
                        and area_exact_values is not None
                        and area_exact_mask is not None
                        and bool(np.all(np.asarray(area_exact_mask[warning_rows], dtype=bool)))
                    )
                    if not warning_rows_have_exact_values:
                        warning_mask, warning_rows = _clear_device_exact_keep_geom_type_warnings(
                            warning_mask,
                            keep_mask,
                            left_source=left_source,
                            right_source=right_source,
                            left_rows=left_rows,
                            right_rows=right_rows,
                            area_owned=area_owned,
                            left_pairs=left_pairs,
                            right_pairs=right_pairs,
                        )
                if warning_rows.size > 0:
                    cached_warning_mask = None
                    rect_warning_count_resolved = False
                    if area_exact_values is not None and area_exact_mask is not None:
                        cached_warning_mask = np.asarray(area_exact_mask[warning_rows], dtype=bool)
                        if cached_warning_mask.any():
                            dropped += _count_dropped_polygon_intersection_parts(
                                np.empty(0, dtype=object),
                                np.empty(0, dtype=object),
                                int(np.count_nonzero(cached_warning_mask)),
                                exact_values=area_exact_values[warning_rows][cached_warning_mask],
                            )
                    if rect_overlap_mask is not None:
                        need_rect_probe_values = (
                            cached_warning_mask is None or (~cached_warning_mask).any()
                        )
                        if need_rect_probe_values:
                            uncached_warning_rows = (
                                warning_rows
                                if cached_warning_mask is None
                                else warning_rows[~cached_warning_mask]
                            )
                            device_uncached_dropped = (
                                _device_count_dropped_polygon_intersection_warning_rows(
                                    area_owned,
                                    keep_mask,
                                    uncached_warning_rows,
                                    left_source=left_source,
                                    right_source=right_source,
                                    left_rows=left_rows,
                                    right_rows=right_rows,
                                    left_pairs=left_pairs,
                                    right_pairs=right_pairs,
                                )
                            )
                            if device_uncached_dropped is not None:
                                dropped += device_uncached_dropped
                                need_rect_probe_values = False
                                rect_warning_count_resolved = True
                            elif _requires_device_to_host_probe(
                                left_source,
                                right_source,
                                left_pairs,
                                right_pairs,
                            ):
                                # Keep device-selected rect-overlap batches on device even when
                                # the lower-dimensional warning counter cannot refine them yet.
                                # The warning count is advisory; falling back here only burns
                                # wall time after the exact geometry result is already native.
                                dropped += int(uncached_warning_rows.size)
                                need_rect_probe_values = False
                                rect_warning_count_resolved = True
                                record_dispatch_event(
                                    surface="geopandas.overlay.intersection",
                                    operation="keep_geom_type_warning_count",
                                    implementation="device_advisory_warning_count",
                                    reason=(
                                        "device-backed keep-geom-type warning rows used "
                                        "the native advisory count instead of boundary "
                                        "geometry reconstruction"
                                    ),
                                    detail=f"rows={int(uncached_warning_rows.size)}",
                                    requested=ExecutionMode.GPU,
                                    selected=ExecutionMode.GPU,
                                )
                            else:
                                pass
                            if need_rect_probe_values and (
                                left_pairs is not None and right_pairs is not None
                            ):
                                uncached_count = int(uncached_warning_rows.size)
                                _decline_keep_geom_type_host_probe(
                                    f"rows={len(tags)}, warning_rows={uncached_count}"
                                )
                                left_values = _take_geoseries_object_values(
                                    left_pairs,
                                    warning_rows.astype(np.intp, copy=False),
                                )
                                right_values = _take_geoseries_object_values(
                                    right_pairs,
                                    warning_rows.astype(np.intp, copy=False),
                                )
                            elif need_rect_probe_values and (
                                left_source is not None
                                and right_source is not None
                                and left_rows is not None
                                and right_rows is not None
                            ):
                                uncached_count = int(uncached_warning_rows.size)
                                _decline_keep_geom_type_host_probe(
                                    f"rows={len(tags)}, warning_rows={uncached_count}"
                                )
                                source_left_rows = np.asarray(left_rows, dtype=np.intp)[
                                    warning_rows
                                ]
                                source_right_rows = np.asarray(right_rows, dtype=np.intp)[
                                    warning_rows
                                ]
                                left_values = _take_geoseries_object_values(
                                    left_source, source_left_rows
                                )
                                right_values = _take_geoseries_object_values(
                                    right_source,
                                    source_right_rows,
                                )
                            elif need_rect_probe_values:
                                if left_pairs is None or right_pairs is None:
                                    raise ValueError(
                                        "left_pairs/right_pairs or source rows are required when "
                                        "keep_geom_type_warning=True"
                                    )
                                uncached_count = int(uncached_warning_rows.size)
                                _decline_keep_geom_type_host_probe(
                                    f"rows={len(tags)}, warning_rows={uncached_count}"
                                )
                                left_values = _take_geoseries_object_values(
                                    left_pairs,
                                    warning_rows.astype(np.intp, copy=False),
                                )
                                right_values = _take_geoseries_object_values(
                                    right_pairs,
                                    warning_rows.astype(np.intp, copy=False),
                                )
                    if area_exact_values is not None and area_exact_mask is not None:
                        if (~cached_warning_mask).any():
                            uncached_warning_rows = warning_rows[~cached_warning_mask]
                            if rect_overlap_mask is not None and not need_rect_probe_values:
                                pass
                            elif rect_overlap_mask is not None:
                                left_uncached = left_values[~cached_warning_mask]
                                right_uncached = right_values[~cached_warning_mask]
                            elif left_pairs is not None and right_pairs is not None:
                                left_uncached = _take_geoseries_object_values(
                                    left_pairs,
                                    uncached_warning_rows,
                                )
                                right_uncached = _take_geoseries_object_values(
                                    right_pairs,
                                    uncached_warning_rows,
                                )
                            elif (
                                left_source is not None
                                and right_source is not None
                                and left_rows is not None
                                and right_rows is not None
                            ):
                                left_uncached = _take_geoseries_object_values(
                                    left_source,
                                    np.asarray(left_rows, dtype=np.intp)[uncached_warning_rows],
                                )
                                right_uncached = _take_geoseries_object_values(
                                    right_source,
                                    np.asarray(right_rows, dtype=np.intp)[uncached_warning_rows],
                                )
                            else:
                                if left_pairs is None or right_pairs is None:
                                    raise ValueError(
                                        "left_pairs/right_pairs or source rows are required when "
                                        "keep_geom_type_warning=True"
                                    )
                                left_uncached = _take_geoseries_object_values(
                                    left_pairs,
                                    uncached_warning_rows,
                                )
                                right_uncached = _take_geoseries_object_values(
                                    right_pairs,
                                    uncached_warning_rows,
                                )
                            if rect_overlap_mask is None:
                                uncached_keep_mask, uncached_dropped, uncached_exact_values = (
                                    _exact_keep_mask_and_dropped_count_for_polygon_intersection_warning_rows(
                                        left_uncached,
                                        right_uncached,
                                    )
                                )
                                dropped += uncached_dropped
                                area_pairs = _replace_geoseries_rows_with_exact_values(
                                    area_pairs,
                                    uncached_warning_rows,
                                    uncached_exact_values,
                                )
                                area_owned = getattr(area_pairs.values, "_owned", area_owned)
                                dropped_uncached_rows = uncached_warning_rows[
                                    ~np.asarray(uncached_keep_mask, dtype=bool)
                                ]
                                if dropped_uncached_rows.size > 0:
                                    keep_mask = np.asarray(keep_mask, dtype=bool).copy()
                                    keep_mask[dropped_uncached_rows] = False
                                    warning_mask = np.asarray(warning_mask, dtype=bool).copy()
                                    warning_mask[dropped_uncached_rows] = True
                                    warning_rows = np.flatnonzero(warning_mask).astype(
                                        np.intp,
                                        copy=False,
                                    )
                            elif need_rect_probe_values:
                                dropped += _count_dropped_polygon_intersection_parts(
                                    left_uncached,
                                    right_uncached,
                                    int(np.count_nonzero(~cached_warning_mask)),
                                )
                    else:
                        if rect_overlap_mask is not None and rect_warning_count_resolved:
                            pass
                        elif _requires_device_to_host_probe(
                            left_source,
                            right_source,
                            left_pairs,
                            right_pairs,
                        ):
                            # The exact geometry result is already native and
                            # keep-geom-type has already selected the retained
                            # polygon rows. Do not rebuild lower-dimensional
                            # boundary geometry just to refine the advisory
                            # warning count for device-backed sources.
                            dropped = int(warning_rows.size)
                            record_dispatch_event(
                                surface="geopandas.overlay.intersection",
                                operation="keep_geom_type_warning_count",
                                implementation="device_advisory_warning_count",
                                reason=(
                                    "device-backed keep-geom-type warning rows used "
                                    "the native advisory count instead of boundary "
                                    "geometry reconstruction"
                                ),
                                detail=f"rows={int(warning_rows.size)}",
                                requested=ExecutionMode.GPU,
                                selected=ExecutionMode.GPU,
                            )
                        else:
                            device_dropped = (
                                _device_count_dropped_polygon_intersection_warning_rows(
                                    area_owned,
                                    keep_mask,
                                    warning_rows,
                                    left_source=left_source,
                                    right_source=right_source,
                                    left_rows=left_rows,
                                    right_rows=right_rows,
                                    left_pairs=left_pairs,
                                    right_pairs=right_pairs,
                                )
                            )
                            if device_dropped is not None:
                                dropped = device_dropped
                            else:
                                if rect_overlap_mask is None:
                                    if left_pairs is not None and right_pairs is not None:
                                        _decline_keep_geom_type_host_probe(
                                            f"rows={len(tags)}, warning_rows={warning_rows.size}"
                                        )
                                        left_values = _take_geoseries_object_values(
                                            left_pairs,
                                            warning_rows.astype(np.intp, copy=False),
                                        )
                                        right_values = _take_geoseries_object_values(
                                            right_pairs,
                                            warning_rows.astype(np.intp, copy=False),
                                        )
                                    elif (
                                        left_source is not None
                                        and right_source is not None
                                        and left_rows is not None
                                        and right_rows is not None
                                    ):
                                        _decline_keep_geom_type_host_probe(
                                            f"rows={len(tags)}, warning_rows={warning_rows.size}"
                                        )
                                        left_values = _take_geoseries_object_values(
                                            left_source,
                                            np.asarray(left_rows, dtype=np.intp)[warning_rows],
                                        )
                                        right_values = _take_geoseries_object_values(
                                            right_source,
                                            np.asarray(right_rows, dtype=np.intp)[warning_rows],
                                        )
                                    else:
                                        left_values = left_values[warning_mask]
                                        right_values = right_values[warning_mask]
                                if rect_overlap_mask is None:
                                    exact_keep_mask, dropped, exact_warning_values = (
                                        _exact_keep_mask_and_dropped_count_for_polygon_intersection_warning_rows(
                                            left_values,
                                            right_values,
                                        )
                                    )
                                    area_pairs = _replace_geoseries_rows_with_exact_values(
                                        area_pairs,
                                        warning_rows,
                                        exact_warning_values,
                                    )
                                    area_owned = getattr(area_pairs.values, "_owned", area_owned)
                                    dropped_warning_rows = warning_rows[
                                        ~np.asarray(exact_keep_mask, dtype=bool)
                                    ]
                                    if dropped_warning_rows.size > 0:
                                        keep_mask = np.asarray(keep_mask, dtype=bool).copy()
                                        keep_mask[dropped_warning_rows] = False
                                        warning_mask = np.asarray(warning_mask, dtype=bool).copy()
                                        warning_mask[dropped_warning_rows] = True
                                        warning_rows = np.flatnonzero(warning_mask).astype(
                                            np.intp,
                                            copy=False,
                                        )
                                else:
                                    dropped = _count_dropped_polygon_intersection_parts(
                                        left_values,
                                        right_values,
                                        int(warning_rows.size),
                                    )

            if bool(np.all(keep_mask)):
                if area_owned is not None:
                    _mark_owned_logical_polygon_valid_nonempty(area_owned)
                filtered = area_pairs.reset_index(drop=True)
                filtered = _attach_polygon_rect_overlap_mask(filtered, rect_overlap_mask)
                return filtered, dropped, keep_mask
            filtered_rows = np.flatnonzero(keep_mask).astype(np.int64, copy=False)
            if area_owned is not None:
                if area_owned.residency is Residency.DEVICE and has_gpu_runtime():
                    try:
                        import cupy as cp

                        filtered_owned = area_owned.device_take(
                            cp.asarray(filtered_rows, dtype=cp.int64),
                            host_indices_for_sizing=np.asarray(
                                filtered_rows,
                                dtype=np.int64,
                            ),
                        )
                    except ModuleNotFoundError:  # pragma: no cover
                        filtered_owned = area_owned.take(filtered_rows)
                else:
                    filtered_owned = area_owned.take(filtered_rows)
                if area_exact_values is not None and area_exact_mask is not None:
                    filtered_owned._exact_intersection_values = np.asarray(
                        area_exact_values,
                        dtype=object,
                    )[keep_mask]
                    filtered_owned._exact_intersection_value_mask = np.asarray(
                        area_exact_mask,
                        dtype=bool,
                    )[keep_mask]
                filtered = GeoSeries(
                    GeometryArray.from_owned(filtered_owned, crs=area_pairs.crs),
                    crs=area_pairs.crs,
                )
                _mark_owned_logical_polygon_valid_nonempty(filtered_owned)
            else:
                filtered = area_pairs.take(filtered_rows).reset_index(drop=True)
            filtered = _attach_polygon_rect_overlap_mask(
                filtered,
                rect_overlap_mask[keep_mask] if rect_overlap_mask is not None else None,
            )
            return filtered, dropped, keep_mask

    area_values = _geoseries_object_values(area_pairs)
    present_mask = ~(shapely.is_missing(area_values) | shapely.is_empty(area_values))
    keep_mask = present_mask & (shapely.area(area_values) > 0.0)
    kept_rows = np.flatnonzero(keep_mask).astype(np.intp, copy=False)
    if kept_rows.size > 0:
        positive_area_mask = _host_polygon_keep_geom_type_positive_area_mask(
            left_source,
            right_source,
            left_rows,
            right_rows,
            kept_rows,
            area_pairs=area_pairs,
            left_pairs=left_pairs,
            right_pairs=right_pairs,
        )
        if positive_area_mask is not None and positive_area_mask.size == kept_rows.size:
            keep_mask = np.asarray(keep_mask, dtype=bool).copy()
            keep_mask[kept_rows] &= np.asarray(positive_area_mask, dtype=bool)

    dropped = 0
    if keep_geom_type_warning and len(area_values) > 0:
        if area_overlap_mask is not None:
            warning_mask = (~keep_mask) | area_overlap_mask
        else:
            if (
                left_source is not None
                and right_source is not None
                and left_rows is not None
                and right_rows is not None
            ):
                left_values = _take_geoseries_object_values(
                    left_source,
                    np.asarray(left_rows, dtype=np.intp),
                )
                right_values = _take_geoseries_object_values(
                    right_source,
                    np.asarray(right_rows, dtype=np.intp),
                )
            else:
                if left_pairs is None or right_pairs is None:
                    raise ValueError(
                        "left_pairs/right_pairs or source rows are required when "
                        "keep_geom_type_warning=True"
                    )
                left_values = _geoseries_object_values(left_pairs)
                right_values = _geoseries_object_values(right_pairs)
            warning_mask = _warning_candidate_mask_for_polygon_keep_geom_type(
                left_values,
                right_values,
                keep_mask,
            )
        exact_polygon_only_mask = _area_exact_polygon_only_host_mask()
        if exact_polygon_only_mask is not None:
            safe_rows = np.asarray(keep_mask, dtype=bool) & exact_polygon_only_mask
            if safe_rows.any():
                warning_mask = np.asarray(warning_mask, dtype=bool).copy()
                warning_mask[safe_rows] = False
        warning_rows = np.empty(0, dtype=np.intp)
        if np.any(warning_mask):
            warning_mask, warning_rows = _clear_device_exact_keep_geom_type_warnings(
                warning_mask,
                keep_mask,
                left_source=left_source,
                right_source=right_source,
                left_rows=left_rows,
                right_rows=right_rows,
                area_owned=area_owned,
            )

        if warning_rows.size > 0:
            if area_exact_values is not None and area_exact_mask is not None:
                area_exact_values = np.asarray(area_exact_values, dtype=object)
                area_exact_mask = np.asarray(area_exact_mask, dtype=bool)
                if area_exact_values.size != len(area_pairs) or area_exact_mask.size != len(
                    area_pairs
                ):
                    area_exact_values = None
                    area_exact_mask = None

            if area_overlap_mask is not None:
                if (
                    left_source is not None
                    and right_source is not None
                    and left_rows is not None
                    and right_rows is not None
                ):
                    _decline_keep_geom_type_host_probe(
                        f"rows={len(area_values)}, warning_rows={warning_rows.size}"
                    )
                    left_values = _take_geoseries_object_values(
                        left_source,
                        np.asarray(left_rows, dtype=np.intp)[warning_rows],
                    )
                    right_values = _take_geoseries_object_values(
                        right_source,
                        np.asarray(right_rows, dtype=np.intp)[warning_rows],
                    )
                else:
                    if left_pairs is None or right_pairs is None:
                        raise ValueError(
                            "left_pairs/right_pairs or source rows are required when "
                            "keep_geom_type_warning=True"
                        )
                    _decline_keep_geom_type_host_probe(
                        f"rows={len(area_values)}, warning_rows={warning_rows.size}"
                    )
                    left_values = _take_geoseries_object_values(left_pairs, warning_rows)
                    right_values = _take_geoseries_object_values(right_pairs, warning_rows)
            else:
                left_values = left_values[warning_mask]
                right_values = right_values[warning_mask]

            if area_exact_values is not None and area_exact_mask is not None:
                cached_warning_mask = np.asarray(area_exact_mask[warning_rows], dtype=bool)
                if cached_warning_mask.any():
                    dropped += _count_dropped_polygon_intersection_parts(
                        np.empty(0, dtype=object),
                        np.empty(0, dtype=object),
                        int(np.count_nonzero(cached_warning_mask)),
                        exact_values=area_exact_values[warning_rows][cached_warning_mask],
                    )
                if (~cached_warning_mask).any():
                    dropped += _count_dropped_polygon_intersection_parts(
                        left_values[~cached_warning_mask],
                        right_values[~cached_warning_mask],
                        int(np.count_nonzero(~cached_warning_mask)),
                    )
            else:
                dropped = _count_dropped_polygon_intersection_parts(
                    left_values,
                    right_values,
                    int(warning_rows.size),
                )

    if bool(np.all(keep_mask)):
        return area_pairs.reset_index(drop=True), dropped, keep_mask

    filtered_values = area_values[keep_mask].copy()
    if filtered_values.size > 0:
        filtered_values = _strip_non_polygon_collection_parts(filtered_values)
    filtered = GeoSeries(filtered_values, crs=area_pairs.crs)
    return filtered, dropped, keep_mask


def _needs_host_overlay_difference_boundary_rebuild(df1, df2) -> bool:
    """Return True when public overlay difference needs boundary reconstruction.

    The polygon-polygon owned path preserves the overlay face topology we need.
    Mixed-dimensional overlay differences still need GeoPandas/GEOS boundary
    semantics at the public API boundary: polygon boundaries noded by linework,
    lines split into outside pieces, and exact lower-dimensional remnants.
    """
    left_all_polygons, left_present = _series_non_missing_all_polygons(df1.geometry)
    right_all_polygons, right_present = _series_non_missing_all_polygons(df2.geometry)
    if not left_present or not right_present:
        return False
    return not left_all_polygons or not right_all_polygons


def _grouped_overlay_difference_geoms(df1, df2, idx1, idx2) -> np.ndarray:
    """Vectorized grouped Shapely difference for non-native compatibility assembly."""
    left_geoms = np.asarray(df1.geometry, dtype=object)
    result_geoms = left_geoms.copy()

    if idx1.size == 0:
        return result_geoms

    h_idx1 = _overlay_device_to_host(
        idx1,
        reason="overlay difference assembly left-index host boundary",
    )
    h_idx2 = _overlay_device_to_host(
        idx2,
        reason="overlay difference assembly right-index host boundary",
    )
    if h_idx1.size:
        order = np.lexsort((h_idx2, h_idx1))
        h_idx1 = h_idx1[order]
        h_idx2 = h_idx2[order]

    right_geoms = np.asarray(df2.geometry, dtype=object)
    right_unions = np.empty(len(df1), dtype=object)
    right_unions.fill(None)

    idx1_unique, idx1_split_at = np.unique(h_idx1, return_index=True)
    idx2_groups = np.split(h_idx2, idx1_split_at[1:])
    for left_pos, neighbors_idx in zip(idx1_unique, idx2_groups, strict=True):
        neighbors = right_geoms[neighbors_idx]
        if len(neighbors) == 1:
            right_unions[left_pos] = neighbors[0]
        else:
            right_unions[left_pos] = shapely.union_all(neighbors)

    has_neighbors = np.zeros(len(df1), dtype=bool)
    has_neighbors[idx1_unique] = True
    result_geoms[has_neighbors] = shapely.difference(
        left_geoms[has_neighbors],
        right_unions[has_neighbors],
    )
    return result_geoms


def _host_exact_polygon_intersection_series_batch(
    left_source: GeoSeries,
    right_source: GeoSeries,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    *,
    crs,
    requested: ExecutionMode | str,
    reason: str,
):
    """Build an exact host GeoSeries batch and retain raw exact results."""
    from vibespatial.geometry.owned import from_shapely_geometries

    requested_mode = requested if isinstance(requested, ExecutionMode) else ExecutionMode(requested)
    left_rows = np.asarray(left_rows, dtype=np.intp)
    right_rows = np.asarray(right_rows, dtype=np.intp)
    record_fallback_event(
        surface="geopandas.overlay.intersection",
        reason=reason,
        detail=f"rows={left_rows.size}",
        requested=requested_mode,
        selected=ExecutionMode.CPU,
        pipeline="_host_exact_polygon_intersection_series_batch",
        d2h_transfer=True,
    )

    left_values = _take_geoseries_object_values(left_source, left_rows)
    right_values = _take_geoseries_object_values(right_source, right_rows)
    raw = np.asarray(shapely.intersection(left_values, right_values), dtype=object)

    try:
        owned = from_shapely_geometries(list(raw))
    except NotImplementedError:
        result = GeoSeries(raw, crs=crs)
        result.values._exact_intersection_values = raw
        result.values._exact_intersection_value_mask = np.ones(len(raw), dtype=bool)
        return result

    owned._exact_intersection_values = raw
    owned._exact_intersection_value_mask = np.ones(len(raw), dtype=bool)
    return GeoSeries(
        GeometryArray.from_owned(owned, crs=crs),
        crs=crs,
    )


def _many_vs_one_intersection_owned(
    left_pairs,
    right_owned,
    unique_right_idx,
):
    """Run broadcast-right polygon intersection at aligned pair capacity."""
    from vibespatial.constructive.binary_constructive import (
        broadcast_right_polygon_intersection_capacity_gpu,
    )
    from vibespatial.runtime.residency import Residency, TransferTrigger, combined_residency

    pair_count = int(left_pairs.row_count)
    if pair_count == 0:
        return left_pairs
    if not has_gpu_runtime():
        raise _OverlayNativeConstructiveDeclined(
            "many-vs-one broadcast-right constructive requires GPU runtime"
        )

    if right_owned.row_count == 1 and int(unique_right_idx) == 0:
        right_one = right_owned
    elif right_owned.residency is Residency.DEVICE:
        import cupy as cp

        right_one = right_owned.device_take(
            cp.asarray([unique_right_idx], dtype=cp.int64),
        )
    else:
        right_one = right_owned.take(np.asarray([unique_right_idx], dtype=np.int64))

    plan_dispatch_selection(
        kernel_name="overlay_pairwise",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=pair_count,
        current_residency=combined_residency(left_pairs, right_one),
        workload_shape=WorkloadShape.BROADCAST_RIGHT,
        work_estimate=_overlay_pair_work_estimate(
            left_pairs,
            right_one,
            pair_count=pair_count,
            workload_shape=WorkloadShape.BROADCAST_RIGHT,
        ),
    )
    if left_pairs.residency is not Residency.DEVICE:
        left_pairs = left_pairs.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="many-vs-one intersection promoted left pair capacity to device",
        )
    if right_one.residency is not Residency.DEVICE:
        right_one = right_one.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="many-vs-one intersection promoted broadcast polygon to device",
        )

    result = broadcast_right_polygon_intersection_capacity_gpu(
        left_pairs,
        right_one,
        right_row=0,
        dispatch_mode=ExecutionMode.GPU,
    )
    if result is None or int(result.row_count) != pair_count:
        raise _OverlayNativeConstructiveDeclined(
            "broadcast-right polygon capacity partitioner declined"
        )
    record_dispatch_event(
        surface="geopandas.overlay",
        operation="overlay_intersection",
        implementation="broadcast_right_capacity_partition_gpu",
        reason=(
            "broadcast-right polygon pairs consumed the canonical rectangle, "
            "SH, swapped-SH, and exact capacity partitioner"
        ),
        detail=(f"pair_capacity={pair_count}; physical_shape=aligned_broadcast_right_capacity"),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return result


def _few_right_partitioned_polygon_intersection_owned(
    left_pairs,
    right_pairs,
    *,
    dispatch_mode: ExecutionMode,
):
    """Partition gathered polygon pairs and scatter device results once."""
    if left_pairs.row_count != right_pairs.row_count or left_pairs.row_count == 0:
        return None

    try:
        import cupy  # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover
        return None

    from vibespatial.constructive.binary_constructive import (
        _dispatch_partitioned_polygon_intersection_gpu,
    )

    result = _dispatch_partitioned_polygon_intersection_gpu(
        left_pairs,
        right_pairs,
        dispatch_mode=ExecutionMode.GPU,
    )
    if result is None or result.row_count != left_pairs.row_count:
        return None

    record_dispatch_event(
        surface="geopandas.overlay",
        operation="overlay_intersection",
        implementation="few_right_polygon_partition_gpu",
        reason=(
            "few-right overlay partitioned rectangle, SH-eligible, and exact "
            "polygon rows and scattered the device rowsets once"
        ),
        detail=(
            f"rows={left_pairs.row_count}; "
            "workload_shape=aligned_pairwise_row_indirected_rect_sh_exact"
        ),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return result


def _few_right_exact_intersection_owned(
    left_pairs,
    right_pairs,
    *,
    dispatch_mode: ExecutionMode,
):
    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_exact_batch_gpu,
        binary_constructive_owned,
    )

    batch_result = _dispatch_polygon_intersection_overlay_exact_batch_gpu(
        left_pairs,
        right_pairs,
        dispatch_mode=dispatch_mode,
    )
    if batch_result is not None and batch_result.row_count == left_pairs.row_count:
        return batch_result
    return binary_constructive_owned(
        "intersection",
        left_pairs,
        right_pairs,
        dispatch_mode=dispatch_mode,
    )


def _few_right_intersection_owned(
    left_owned,
    right_owned,
    idx1,
    idx2,
    *,
    dispatch_mode=ExecutionMode.AUTO,
    _has_device_indices=False,
    d_idx1=None,
    d_idx2=None,
    _right_group_count: int | None = None,
    _preserve_lower_dim_polygon_results: bool = False,
):
    """Run few-right intersection as one gathered exact pairwise batch.

    The earlier grouped-by-right shape decomposed a logically single overlay
    into many per-right preparations and gathers. For warmed strict-native
    workloads that was slower than simply gathering the intersecting pairs
    once and running one exact row-isolated GPU intersection batch. Device-backed
    pair rows stay in the native relation carrier; host row arrays are accepted
    only for host-backed spatial-index outputs.
    """
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover
        cp = None

    from vibespatial.constructive.binary_constructive import (
        binary_constructive_owned,
    )

    if _has_device_indices:
        if cp is None or d_idx1 is None or d_idx2 is None or _right_group_count is None:
            return None
        d_right_rows = cp.asarray(d_idx2, dtype=cp.int32)
        unique_right_count = int(_right_group_count)
        pair_count = int(d_right_rows.size)
        if unique_right_count <= 1 or unique_right_count > _OVERLAY_FEW_RIGHT_GROUP_MAX:
            return None
        if (pair_count / unique_right_count) < _OVERLAY_FEW_RIGHT_GROUP_MIN_AVG:
            return None

        left_pairs = left_owned.device_take(cp.asarray(d_idx1, dtype=cp.int64))
        right_pairs = right_owned.device_take(d_right_rows.astype(cp.int64, copy=False))
    else:
        idx1_array = np.asarray(idx1, dtype=np.intp)
        idx2_array = np.asarray(idx2, dtype=np.intp)
        unique_right = np.unique(idx2_array)
        if unique_right.size <= 1 or unique_right.size > _OVERLAY_FEW_RIGHT_GROUP_MAX:
            return None
        if (len(idx2_array) / unique_right.size) < _OVERLAY_FEW_RIGHT_GROUP_MIN_AVG:
            return None

        if idx1_array.size == left_owned.row_count and np.array_equal(
            idx1_array,
            np.arange(left_owned.row_count, dtype=idx1_array.dtype),
        ):
            left_pairs = left_owned
        elif cp is not None:
            left_pairs = left_owned.device_take(
                cp.asarray(idx1_array, dtype=cp.int64),
                host_indices_for_sizing=idx1_array,
            )
        else:
            left_pairs = left_owned.take(idx1_array)
        if idx2_array.size == right_owned.row_count and np.array_equal(
            idx2_array,
            np.arange(right_owned.row_count, dtype=idx2_array.dtype),
        ):
            right_pairs = right_owned
        elif cp is not None:
            right_pairs = right_owned.device_take(
                cp.asarray(idx2_array, dtype=cp.int64),
                host_indices_for_sizing=idx2_array,
            )
        else:
            right_pairs = right_owned.take(idx2_array)

    def _attach_aligned_pair_sources(result):
        if result is not None and result.row_count == left_pairs.row_count:
            result._aligned_left_pairs_owned = left_pairs
            result._aligned_right_pairs_owned = right_pairs
        return result

    if _preserve_lower_dim_polygon_results:
        from vibespatial.constructive.binary_constructive import (
            _dispatch_polygon_overlay_row_isolated_batch_gpu,
        )

        exact_topology_result = _dispatch_polygon_overlay_row_isolated_batch_gpu(
            "intersection",
            left_pairs,
            right_pairs,
            dispatch_mode=dispatch_mode,
        )
        if (
            exact_topology_result is not None
            and exact_topology_result.row_count == left_pairs.row_count
        ):
            return _attach_aligned_pair_sources(exact_topology_result)

    if not _preserve_lower_dim_polygon_results:
        partitioned_result = _few_right_partitioned_polygon_intersection_owned(
            left_pairs,
            right_pairs,
            dispatch_mode=dispatch_mode,
        )
        if (
            partitioned_result is not None
            and partitioned_result.row_count == left_pairs.row_count
        ):
            return _attach_aligned_pair_sources(partitioned_result)

    rect_capable = False
    try:
        from vibespatial.kernels.constructive.polygon_rect_intersection import (
            polygon_rect_intersection_can_handle,
        )

        rect_capable = polygon_rect_intersection_can_handle(
            left_pairs, right_pairs
        ) or polygon_rect_intersection_can_handle(right_pairs, left_pairs)
    except _OverlayNativeConstructiveDeclined:
        raise

    if rect_capable and not _preserve_lower_dim_polygon_results:
        return _attach_aligned_pair_sources(
            binary_constructive_owned(
                "intersection",
                left_pairs,
                right_pairs,
                dispatch_mode=dispatch_mode,
            )
        )

    return _attach_aligned_pair_sources(
        _few_right_exact_intersection_owned(
            left_pairs,
            right_pairs,
            dispatch_mode=dispatch_mode,
        )
    )


def _overlay_intersection_native(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
    _index_result=None,
    _polygon_inputs: bool | None = None,
):
    """Build the native intersection result before host-side export.

    Returns
    -------
    tuple[PairwiseConstructiveResult, bool]
        Native constructive result plus whether the owned dispatch path was used.
    """
    left_owned, right_owned = _coerce_owned_pair_for_strict_overlay(
        df1,
        df2,
        left_owned,
        right_owned,
    )
    if _polygon_inputs is None:
        _polygon_inputs = _series_all_polygons(df1.geometry) and _series_all_polygons(df2.geometry)
    prefer_exact_polygon_gpu = _prefer_exact_polygon_gpu
    # Public polygon intersection can legitimately produce lower-dimensional
    # rows or GeometryCollections that must be filtered at the GeoPandas
    # boundary. Unless the strict exact-polygon GPU path is explicitly
    # requested, keep geometry construction on the host boundary and use the
    # owned path only for pairing/index acceleration.
    _use_host_exact_polygon_boundary = (
        _polygon_inputs
        and not prefer_exact_polygon_gpu
        and (not strict_native_mode_enabled() or (left_owned is None and right_owned is None))
    )
    # ADR-0042 low-level contract: spatial indexing may still emit index arrays.
    # Phase 2: pass owned arrays to request device-resident index pairs.
    #
    # When neither side has owned backing yet (plain Shapely GeoDataFrames),
    # force the Shapely STRtree path to avoid cold GPU spatial-index compile
    # cost.  If either public geometry column already carries owned backing,
    # let sindex.query choose the native relation path instead of materializing
    # an owned-backed STRtree at this compute boundary.
    _force_strtree = (
        _index_result is None
        and left_owned is None
        and right_owned is None
        and _series_owned(df1.geometry) is None
        and _series_owned(df2.geometry) is None
        and has_gpu_runtime()
    )
    if _force_strtree:
        df2.sindex._ensure_strtree()
        _raw_idx = df2.sindex.query(
            df1.geometry,
            predicate="intersects",
            sort=True,
        )
        if isinstance(_raw_idx, np.ndarray) and _raw_idx.ndim == 2:
            idx1, idx2 = _raw_idx
        else:
            idx1, idx2 = _raw_idx
        idx1 = np.asarray(idx1, dtype=np.int32)
        idx2 = np.asarray(idx2, dtype=np.int32)
        pair_count = int(idx1.size)
        d_idx1, d_idx2 = None, None
        _has_device_indices = False
    else:
        index_result = (
            _index_result
            if _index_result is not None
            else _intersecting_index_pairs(
                df1,
                df2,
                left_owned=left_owned,
                right_owned=right_owned,
                capacity_output=True,
            )
        )

        if isinstance(index_result, NativeRelationSelection):
            selected_result = _overlay_relation_selection_intersection_native(
                df1,
                df2,
                index_result,
                preserve_lower_dimensional=(_preserve_lower_dim_polygon_results),
                warn_on_dropped_lower_dimensional=(_warn_on_dropped_lower_dim_polygon_results),
            )
            if selected_result is None:
                index_result = _intersecting_index_pairs(
                    df1,
                    df2,
                    left_owned=left_owned,
                    right_owned=right_owned,
                    capacity_output=False,
                )
            else:
                return selected_result, True

        # Unpack result: DeviceSpatialJoinResult (device arrays) or numpy.
        if isinstance(index_result, DeviceSpatialJoinResult):
            d_idx1 = index_result.d_left_idx
            d_idx2 = index_result.d_right_idx
            idx1 = None
            idx2 = None
            pair_count = index_result.size
            _has_device_indices = True
        else:
            if isinstance(index_result, np.ndarray) and index_result.ndim == 2:
                idx1, idx2 = index_result
            else:
                idx1, idx2 = index_result
            idx1 = np.asarray(idx1, dtype=np.int32)
            idx2 = np.asarray(idx2, dtype=np.int32)
            pair_count = int(idx1.size)
            d_idx1, d_idx2 = None, None
            _has_device_indices = False

    _many_vs_one_unique_right_value: int | None = None
    pair_selection: NativeDeviceSelection | None = None

    def _ensure_host_intersection_pairs() -> tuple[np.ndarray, np.ndarray]:
        nonlocal idx1, idx2
        if idx1 is None or idx2 is None:
            if not isinstance(index_result, DeviceSpatialJoinResult):
                raise RuntimeError("device index result missing for host pair export")
            if _many_vs_one_unique_right_value is not None:
                idx1 = index_result.left_to_host()
                idx1 = np.asarray(idx1, dtype=np.int32)
                idx2 = np.full(
                    idx1.shape,
                    _many_vs_one_unique_right_value,
                    dtype=np.int32,
                )
            else:
                idx1, idx2 = index_result.to_host()
                idx1 = np.asarray(idx1, dtype=np.int32)
                idx2 = np.asarray(idx2, dtype=np.int32)
        return idx1, idx2

    def _apply_intersection_pair_keep_mask(keep_mask):
        nonlocal idx1, idx2, d_idx1, d_idx2, index_result, pair_count
        nonlocal _has_device_indices, pair_selection
        if hasattr(keep_mask, "__cuda_array_interface__"):
            import cupy as cp

            d_keep = cp.asarray(keep_mask, dtype=cp.bool_)
            if int(d_keep.size) != pair_count:
                raise ValueError(
                    f"intersection keep mask length mismatch: {int(d_keep.size)} != {pair_count}"
                )
            if pair_selection is not None:
                d_keep = d_keep & pair_selection.source_mask()
            if not _has_device_indices:
                if idx1 is None or idx2 is None:
                    idx1, idx2 = _ensure_host_intersection_pairs()
                d_idx1 = cp.asarray(idx1, dtype=cp.int32)
                d_idx2 = cp.asarray(idx2, dtype=cp.int32)
                index_result = DeviceSpatialJoinResult(d_idx1, d_idx2)
                _has_device_indices = True
            pair_selection = NativeDeviceSelection.from_mask(
                d_keep,
                source_row_count=pair_count,
            )
            return d_keep
        keep = np.asarray(keep_mask, dtype=bool)
        if keep.size != pair_count:
            raise ValueError(f"intersection keep mask length mismatch: {keep.size} != {pair_count}")
        if pair_selection is not None:
            import cupy as cp

            return _apply_intersection_pair_keep_mask(
                cp.asarray(keep, dtype=cp.bool_),
            )
        if bool(np.all(keep)):
            pair_count = int(keep.size)
            return keep
        if idx1 is not None or idx2 is not None:
            if idx1 is None or idx2 is None:
                idx1, idx2 = _ensure_host_intersection_pairs()
            idx1 = np.asarray(idx1, dtype=np.int32)[keep]
            idx2 = np.asarray(idx2, dtype=np.int32)[keep]
            pair_count = int(idx1.size)
            if (
                _has_device_indices
                and d_idx1 is not None
                and d_idx2 is not None
                and int(getattr(d_idx1, "size", -1)) == keep.size
                and int(getattr(d_idx2, "size", -1)) == keep.size
            ):
                import cupy as cp

                d_keep = cp.asarray(keep, dtype=cp.bool_)
                d_idx1 = d_idx1[d_keep]
                d_idx2 = d_idx2[d_keep]
                index_result = DeviceSpatialJoinResult(d_idx1, d_idx2)
            else:
                d_idx1 = None
                d_idx2 = None
                _has_device_indices = False
            return keep
        if _has_device_indices:
            import cupy as cp

            d_keep = cp.asarray(keep, dtype=cp.bool_)
            d_idx1 = d_idx1[d_keep]
            d_idx2 = d_idx2[d_keep]
            index_result = DeviceSpatialJoinResult(d_idx1, d_idx2)
            pair_count = int(np.count_nonzero(keep))
            return keep
        idx1, idx2 = _ensure_host_intersection_pairs()
        idx1 = np.asarray(idx1, dtype=np.int32)[keep]
        idx2 = np.asarray(idx2, dtype=np.int32)[keep]
        pair_count = int(idx1.size)
        return keep

    used_owned = False
    # Create pairs of geometries in both dataframes to be intersected
    if pair_count > 0:
        left_sub = None
        right_sub = None
        # Many-vs-one owned coercion: when the spatial join reveals a
        # many-vs-one pattern (all pairs reference the same single right
        # row) and neither side has owned backing, coerce to
        # OwnedGeometryArray to enable the GPU containment bypass and SH
        # batch clip.  This coercion is restricted to the many-vs-one
        # pattern to avoid changing behavior for general N-vs-M overlay.
        if _has_device_indices and idx2 is None:
            _is_many_vs_one_pre = (
                right_owned is not None and right_owned.row_count == 1 and pair_count > 1
            )
        else:
            _unique_right_pre = np.unique(idx2)
            _is_many_vs_one_pre = _unique_right_pre.size == 1 and idx1.size > 1
        if _is_many_vs_one_pre and has_gpu_runtime():
            _both_polygon = _polygon_inputs
            if _both_polygon:
                from vibespatial.geometry.owned import from_shapely_geometries
                from vibespatial.runtime.residency import (
                    Residency,
                    TransferTrigger,
                    combined_residency,
                )

                if left_owned is None:
                    try:
                        left_owned = from_shapely_geometries(
                            list(df1.geometry),
                        )
                    except (NotImplementedError, ValueError):
                        left_owned = None

                if right_owned is None:
                    try:
                        right_owned = from_shapely_geometries(
                            list(df2.geometry),
                        )
                    except (NotImplementedError, ValueError):
                        right_owned = None
                if left_owned is not None and right_owned is not None:
                    if combined_residency(left_owned, right_owned) is not Residency.DEVICE:
                        left_owned.move_to(
                            Residency.DEVICE,
                            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                            reason="many-vs-one polygon overlay promoted left input to device",
                        )
                        right_owned.move_to(
                            Residency.DEVICE,
                            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                            reason="many-vs-one polygon overlay promoted right input to device",
                        )
                    prefer_exact_polygon_gpu = True
                    _use_host_exact_polygon_boundary = False

        # Owned-path dispatch: OwnedGeometryArray.take() operates at buffer
        # level (no Shapely materialization), then binary_constructive_owned
        # routes to GPU when available.  GeoSeries.take() breaks the DGA chain
        # by materializing to Shapely, so we bypass it when owned is present.
        # Note: GeometryArray.copy() preserves _owned, and __setitem__
        # invalidates it on mutation, so owned survives _make_valid when
        # all geometries are already valid.
        intersections = None

        def _attach_aligned_pair_sources(result, left_pairs, right_pairs):
            if (
                result is not None
                and left_pairs is not None
                and right_pairs is not None
                and int(getattr(result, "row_count", -1))
                == int(getattr(left_pairs, "row_count", -2))
                == int(getattr(right_pairs, "row_count", -3))
            ):
                result._aligned_left_pairs_owned = left_pairs
                result._aligned_right_pairs_owned = right_pairs
            return result

        host_exact_pair_work = _overlay_relation_pair_work_estimate(
            left_owned,
            right_owned,
            pair_count=pair_count,
        )
        if (
            intersections is None
            and _polygon_inputs
            and not strict_native_mode_enabled()
            and not prefer_exact_polygon_gpu
            and pair_count > 0
            and host_exact_pair_work.dispatch_unit_count()
            <= _OVERLAY_HOST_EXACT_PAIR_BATCH_MAX_WORK_UNITS
            and not has_gpu_runtime()
        ):
            pairwise_selection = plan_dispatch_selection(
                kernel_name="overlay_pairwise",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=pair_count,
                requested_mode=ExecutionMode.AUTO,
                work_estimate=host_exact_pair_work,
            )
            if pairwise_selection.selected is ExecutionMode.CPU:
                idx1, idx2 = _ensure_host_intersection_pairs()
                intersections = _host_exact_polygon_intersection_series_batch(
                    df1.geometry,
                    df2.geometry,
                    np.asarray(idx1, dtype=np.intp),
                    np.asarray(idx2, dtype=np.intp),
                    crs=df1.crs,
                    requested=pairwise_selection.requested,
                    reason=("small polygon pair CPU compatibility boundary without GPU runtime"),
                )
        if (
            left_owned is not None
            and right_owned is not None
            and not _use_host_exact_polygon_boundary
            and intersections is None
        ):
            from vibespatial.constructive.binary_constructive import (
                binary_constructive_native,
                binary_constructive_owned,
            )

            # Free pool memory before device_take: spatial index and
            # prior pipeline stages leave freed-but-cached blocks in
            # the pool.  Forcing GC here ensures dead CuPy arrays
            # return their blocks before the large gather allocation.
            from vibespatial.cuda._runtime import maybe_trim_pool_memory

            maybe_trim_pool_memory()
            # Keep AUTO behavior for normal runs, but force the repo-owned
            # GPU path when the public overlay contract only needs polygon
            # output. AUTO host selection can yield GeometryCollection rows
            # that are valid at the constructive layer but wrong for the
            # keep_geom_type=True / default overlay boundary before collection
            # extraction runs.
            _pairwise_mode = (
                ExecutionMode.GPU
                if strict_native_mode_enabled() or prefer_exact_polygon_gpu
                else ExecutionMode.AUTO
            )

            # ---- Many-vs-one detection ----
            # Check BEFORE device_take: for many-vs-one (all pairs
            # reference the same single right row), gathering the right
            # side duplicates one polygon's ring data N times.  At 1M
            # scale this can be 5+ GiB and exceed VRAM.  The many-vs-one
            # fast path only needs left_sub gathered; right_owned is
            # passed by reference.
            if _has_device_indices and idx2 is None:
                if right_owned is not None and right_owned.row_count == 1 and pair_count > 1:
                    unique_right_count = 1
                    unique_right_value = 0
                    _many_vs_one_unique_right_value = unique_right_value
                    _unique_right = np.array([unique_right_value], dtype=np.int32)
                    _is_many_vs_one = True
                    _is_few_right = False
                else:
                    import cupy as cp

                    d_unique_right = cp.unique(d_idx2)
                    unique_right_count = int(d_unique_right.size)
                    unique_right_value = (
                        _overlay_int_scalar(
                            d_unique_right[0],
                            reason="overlay many-vs-one unique right-row scalar host boundary",
                        )
                        if unique_right_count == 1 and pair_count > 1
                        else None
                    )
                    if unique_right_value is not None:
                        _many_vs_one_unique_right_value = unique_right_value
                        _unique_right = np.array([unique_right_value], dtype=np.int32)
                        _is_many_vs_one = True
                        _is_few_right = False
                    elif (
                        unique_right_count > 1
                        and unique_right_count <= _OVERLAY_FEW_RIGHT_GROUP_MAX
                        and (pair_count / unique_right_count) >= _OVERLAY_FEW_RIGHT_GROUP_MIN_AVG
                        and _polygon_inputs
                        and not _use_host_exact_polygon_boundary
                    ):
                        _unique_right = d_unique_right
                        _is_many_vs_one = False
                        _is_few_right = True
                    else:
                        _unique_right = np.empty(0, dtype=np.int32)
                        _is_many_vs_one = False
                        _is_few_right = False
            else:
                idx1, idx2 = _ensure_host_intersection_pairs()
                _unique_right = np.unique(idx2)
                unique_right_count = int(_unique_right.size)
                _is_many_vs_one = unique_right_count == 1 and idx1.size > 1
                _is_few_right = (
                    unique_right_count > 1
                    and unique_right_count <= _OVERLAY_FEW_RIGHT_GROUP_MAX
                    and (idx1.size / unique_right_count) >= _OVERLAY_FEW_RIGHT_GROUP_MIN_AVG
                    and _polygon_inputs
                    and not _use_host_exact_polygon_boundary
                )

            if _is_many_vs_one:
                # Many-vs-one: only gather left side.
                if _has_device_indices:
                    left_sub = _device_take_relation_rows(left_owned, d_idx1)
                else:
                    left_sub = left_owned.take(np.asarray(idx1))
                right_sub = None  # deferred until fallback

                def _finalize_many_vs_one(result_owned):
                    nonlocal intersections, used_owned, left_sub
                    intersections = GeoSeries(
                        GeometryArray.from_owned(result_owned, crs=df1.crs),
                    )
                    used_owned = True

                result_owned = _many_vs_one_intersection_owned(
                    left_sub,
                    right_owned,
                    int(_unique_right[0]),
                )
                if result_owned is not None:
                    _finalize_many_vs_one(result_owned)
            elif _is_few_right:
                result_owned = _few_right_intersection_owned(
                    left_owned,
                    right_owned,
                    idx1,
                    idx2,
                    dispatch_mode=_pairwise_mode,
                    _has_device_indices=_has_device_indices,
                    d_idx1=d_idx1,
                    d_idx2=d_idx2,
                    _right_group_count=unique_right_count,
                    _preserve_lower_dim_polygon_results=(_preserve_lower_dim_polygon_results),
                )
                if result_owned is not None:
                    intersections = GeoSeries(
                        GeometryArray.from_owned(result_owned, crs=df1.crs),
                    )
                    used_owned = True
            else:
                # Phase 2 zero-copy: pass CuPy device arrays directly to
                # row-indirected geometry views. Relation pairs can repeat
                # either source row many times, so copying coordinate slices
                # has the wrong physical shape even when sampled rows look
                # mostly unique.
                if _has_device_indices:
                    left_sub = _device_take_relation_rows(left_owned, d_idx1)
                    right_sub = _device_take_relation_rows(right_owned, d_idx2)
                else:
                    left_sub = left_owned.take(np.asarray(idx1))
                    right_sub = right_owned.take(np.asarray(idx2))

            if intersections is None and not (_is_many_vs_one or _is_few_right):
                # Standard element-wise path for N-vs-M patterns.
                native_intersections = binary_constructive_native(
                    "intersection",
                    left_sub,
                    right_sub,
                    dispatch_mode=_pairwise_mode,
                )
                result_owned = native_intersections.owned
                if result_owned is not None:
                    result_owned = _attach_aligned_pair_sources(
                        result_owned,
                        left_sub,
                        right_sub,
                    )
                    intersections = GeoSeries(
                        GeometryArray.from_owned(result_owned, crs=df1.crs),
                    )
                    used_owned = True
                elif native_intersections.composition is not None:
                    d_nonempty = native_intersections.valid_nonempty_mask_device()
                    if d_nonempty is None:
                        raise RuntimeError(
                            "native overlay intersection composition lost its "
                            "valid/nonempty row metadata"
                        )
                    keep_geom_type_applied = False
                    if _polygon_inputs and not _preserve_lower_dim_polygon_results:
                        from vibespatial.geometry.buffers import GeometryFamily

                        family_selection = (
                            native_intersections.select_family_domain_device(
                                (
                                    GeometryFamily.POLYGON,
                                    GeometryFamily.MULTIPOLYGON,
                                )
                            )
                        )
                        if family_selection is None:
                            raise RuntimeError(
                                "native polygon overlay composition lost device "
                                "family metadata"
                            )
                        native_intersections, d_nonempty, d_dropped_count = (
                            family_selection
                        )
                        if _warn_on_dropped_lower_dim_polygon_results:
                            num_dropped = _overlay_int_scalar(
                                cp.asarray(d_dropped_count).reshape(1),
                                reason=(
                                    "native polygon composition keep-geometry-type "
                                    "warning count scalar fence"
                                ),
                            )
                            if num_dropped > 0:
                                warnings.warn(
                                    "`keep_geom_type=True` in overlay resulted in "
                                    f"{num_dropped} dropped geometries of different "
                                    "geometry types than df1 has. Set "
                                    "`keep_geom_type=False` to retain all geometries",
                                    UserWarning,
                                    stacklevel=4,
                                )
                        keep_geom_type_applied = True
                    _apply_intersection_pair_keep_mask(d_nonempty)
                    use_device_relation = _has_device_indices and (
                        pair_selection is not None or idx1 is None
                    )
                    capacity_result = _pairwise_constructive_to_native_tabular_result(
                        geometry=native_intersections.with_crs(df1.crs),
                        relation=RelationIndexResult(
                            d_idx1 if use_device_relation else idx1,
                            d_idx2 if use_device_relation else idx2,
                        ),
                        keep_geom_type_applied=keep_geom_type_applied,
                        left_df=df1,
                        right_df=df2,
                    )
                    record_dispatch_event(
                        surface="geopandas.overlay",
                        operation="intersection",
                        implementation="mixed_pair_composition_gpu",
                        reason=(
                            "mixed-family constructive output remained in its "
                            "row-aligned native composition"
                        ),
                        detail=(
                            f"pair_capacity={pair_count}; "
                            "physical_shape=relation_pair_capacity_composition"
                        ),
                        requested=_pairwise_mode,
                        selected=ExecutionMode.GPU,
                    )
                    if pair_selection is not None and keep_geom_type_applied:
                        from vibespatial.geometry.buffers import GeometryFamily

                        pair_selection = replace(
                            pair_selection,
                            geometry_family_domain=(
                                GeometryFamily.POLYGON,
                                GeometryFamily.MULTIPOLYGON,
                            ),
                            trusted_all_valid_rows=True,
                        )
                    return (
                        NativeTabularSelection(
                            capacity_result=capacity_result,
                            selection=pair_selection,
                        ),
                        True,
                    )

            if intersections is None and _is_few_right:
                if _has_device_indices:
                    left_sub = _device_take_relation_rows(left_owned, d_idx1)
                    right_sub = _device_take_relation_rows(right_owned, d_idx2)
                else:
                    left_sub = left_owned.take(np.asarray(idx1))
                    right_sub = right_owned.take(np.asarray(idx2))
                result_owned = _attach_aligned_pair_sources(
                    binary_constructive_owned(
                        "intersection",
                        left_sub,
                        right_sub,
                        dispatch_mode=_pairwise_mode,
                    ),
                    left_sub,
                    right_sub,
                )
                if result_owned is not None:
                    intersections = GeoSeries(
                        GeometryArray.from_owned(result_owned, crs=df1.crs),
                    )
                    used_owned = True

            if intersections is None and _is_many_vs_one:
                # Many-vs-one fast path failed -- fall back to element-wise.
                # Gather right side now (deferred from above to avoid OOM
                # on the many-vs-one fast path).
                if right_sub is None:
                    if _has_device_indices:
                        right_sub = _device_take_relation_rows(right_owned, d_idx2)
                    else:
                        right_sub = right_owned.take(np.asarray(idx2))
                result_owned = _attach_aligned_pair_sources(
                    binary_constructive_owned(
                        "intersection",
                        left_sub,
                        right_sub,
                        dispatch_mode=_pairwise_mode,
                    ),
                    left_sub,
                    right_sub,
                )
                if result_owned is not None:
                    intersections = GeoSeries(
                        GeometryArray.from_owned(result_owned, crs=df1.crs),
                    )
                    used_owned = True

        if intersections is None:
            # ADR-0042 transitional boundary: host exact path still uses GeoSeries ops.
            idx1, idx2 = _ensure_host_intersection_pairs()
            if _use_host_exact_polygon_boundary:
                public_boundary_work = _overlay_relation_pair_work_estimate(
                    left_owned,
                    right_owned,
                    pair_count=int(idx1.size),
                )
                if (
                    has_gpu_runtime()
                    and public_boundary_work.dispatch_unit_count()
                    <= _OVERLAY_EXACT_POLYGON_GPU_BOUNDARY_MAX_WORK_UNITS
                ):
                    from vibespatial.constructive.binary_constructive import (
                        binary_constructive_owned,
                    )

                    pair_rows_left = np.asarray(idx1, dtype=np.intp)
                    pair_rows_right = np.asarray(idx2, dtype=np.intp)
                    if left_owned is not None and right_owned is not None:
                        pair_left_owned = left_owned.take(pair_rows_left)
                        pair_right_owned = right_owned.take(pair_rows_right)
                    else:
                        pair_left = df1.geometry.take(pair_rows_left)
                        pair_left.reset_index(drop=True, inplace=True)
                        pair_right = df2.geometry.take(pair_rows_right)
                        pair_right.reset_index(drop=True, inplace=True)
                        pair_left_owned = pair_left.values.to_owned()
                        pair_right_owned = pair_right.values.to_owned()
                    result_owned = _attach_aligned_pair_sources(
                        binary_constructive_owned(
                            "intersection",
                            pair_left_owned,
                            pair_right_owned,
                            dispatch_mode=ExecutionMode.GPU,
                        ),
                        pair_left_owned,
                        pair_right_owned,
                    )
                    if result_owned is not None:
                        intersections = GeoSeries(
                            GeometryArray.from_owned(result_owned, crs=df1.crs),
                        )
                        used_owned = True

                if intersections is None:
                    left_values = _take_geoseries_object_values(
                        df1.geometry,
                        np.asarray(idx1, dtype=np.intp),
                    )
                    right_values = _take_geoseries_object_values(
                        df2.geometry,
                        np.asarray(idx2, dtype=np.intp),
                    )
                    intersections = GeoSeries(
                        shapely.intersection(left_values, right_values),
                        crs=df1.crs,
                    )
            else:
                left = df1.geometry.take(idx1)
                left.reset_index(drop=True, inplace=True)
                right = df2.geometry.take(idx2)
                right.reset_index(drop=True, inplace=True)
                intersections = left.intersection(right)

        # Post-intersection make_valid must run for both owned/GPU and
        # compatibility boundary paths. For polygon area-only output, filter
        # keep-geom-type rows first so lower-dimensional/sliver remnants do
        # not force repair or compatibility materialization before they are dropped.
        post_intersection_make_valid_needed = not (
            _polygon_inputs and _can_defer_make_valid_to_rect_repair(intersections)
        )
        defer_post_intersection_make_valid = (
            post_intersection_make_valid_needed
            and _polygon_inputs
            and not _preserve_lower_dim_polygon_results
            and (_warn_on_dropped_lower_dim_polygon_results or prefer_exact_polygon_gpu)
        )
        if post_intersection_make_valid_needed and not defer_post_intersection_make_valid:
            intersections = _make_valid_geoseries(
                intersections,
                dispatch_mode=(
                    ExecutionMode.GPU
                    if (used_owned or prefer_exact_polygon_gpu)
                    else ExecutionMode.AUTO
                ),
            )

        geom_intersect = intersections
        keep_geom_type_applied = False
        native_lower_dim_geometry = None
        if _polygon_inputs:
            pair_left = None
            pair_right = None
            source_idx1 = None
            source_idx2 = None
            left_source_geoms = df1.geometry
            right_source_geoms = df2.geometry
            if left_owned is not None:
                left_source_geoms = GeoSeries(
                    GeometryArray.from_owned(left_owned, crs=df1.crs),
                    crs=df1.crs,
                )
            if right_owned is not None:
                right_source_geoms = GeoSeries(
                    GeometryArray.from_owned(right_owned, crs=df2.crs),
                    crs=df2.crs,
                )

            if _preserve_lower_dim_polygon_results:
                area_owned = getattr(geom_intersect.values, "_owned", None)
                aligned_left_owned, aligned_right_owned = _aligned_pair_owned_from_area(
                    area_owned,
                )
                d_topology_remnants = getattr(
                    area_owned,
                    "_polygon_intersection_lower_dimensional_remnant",
                    None,
                )
                native_lower_dim_result = polygon_pair_boundary_remnants_capacity_device(
                    aligned_left_owned,
                    aligned_right_owned,
                    area_owned,
                    crs=df1.crs,
                    remnant_mask=d_topology_remnants,
                )
                if native_lower_dim_result is not None:
                    native_lower_dim_geometry, d_native_keep = native_lower_dim_result
                    import cupy as cp

                    _apply_intersection_pair_keep_mask(
                        cp.asarray(d_native_keep, dtype=cp.bool_),
                    )
                    record_dispatch_event(
                        surface="geopandas.overlay",
                        operation="intersection",
                        implementation="polygon_boundary_remnant_composition_gpu",
                        reason=(
                            "polygon area and lower-dimensional boundary remnants "
                            "assembled through the shared native composition carrier"
                        ),
                        detail=(
                            f"pair_capacity={int(d_native_keep.size)}; "
                            f"output_capacity={native_lower_dim_geometry.row_count}; "
                            "physical_shape=aligned_polygon_boundaries_capacity_composition"
                        ),
                        requested=ExecutionMode.GPU,
                        selected=ExecutionMode.GPU,
                    )

            host_pair_series_required = (
                _preserve_lower_dim_polygon_results and native_lower_dim_geometry is None
            ) or (_warn_on_dropped_lower_dim_polygon_results and _use_host_exact_polygon_boundary)
            pair_context_required = (
                host_pair_series_required
                or _warn_on_dropped_lower_dim_polygon_results
                or prefer_exact_polygon_gpu
            )
            host_indices_required = host_pair_series_required
            if pair_context_required and not host_indices_required:
                if idx1 is not None and idx2 is not None:
                    source_idx1 = np.asarray(idx1, dtype=np.intp)
                    source_idx2 = np.asarray(idx2, dtype=np.intp)
                elif _has_device_indices and d_idx1 is not None and d_idx2 is not None:
                    source_idx1 = d_idx1
                    source_idx2 = d_idx2
                else:
                    aligned_area_owned = getattr(geom_intersect.values, "_owned", None)
                    aligned_left_owned, aligned_right_owned = _aligned_pair_owned_from_area(
                        aligned_area_owned
                    )
                    if aligned_left_owned is not None and aligned_right_owned is not None:
                        pair_left = GeoSeries(
                            GeometryArray.from_owned(aligned_left_owned, crs=df1.crs),
                            crs=df1.crs,
                        )
                        pair_right = GeoSeries(
                            GeometryArray.from_owned(aligned_right_owned, crs=df2.crs),
                            crs=df2.crs,
                        )
                    else:
                        host_indices_required = True

            if host_indices_required:
                idx1, idx2 = _ensure_host_intersection_pairs()
                source_idx1 = np.asarray(idx1, dtype=np.intp)
                source_idx2 = np.asarray(idx2, dtype=np.intp)
            if host_pair_series_required:
                pair_left = df1.geometry.take(idx1)
                pair_left.reset_index(drop=True, inplace=True)
                pair_right = df2.geometry.take(idx2)
                pair_right.reset_index(drop=True, inplace=True)

            def _take_pair_source_rows(series, keep_mask):
                if series is None:
                    return None
                if hasattr(keep_mask, "__cuda_array_interface__"):
                    import cupy as cp

                    d_keep = cp.asarray(keep_mask, dtype=cp.bool_)
                    owned = getattr(series.values, "_owned", None)
                    if owned is not None:
                        from vibespatial.runtime.residency import Residency

                        if owned.residency is Residency.DEVICE and has_gpu_runtime():
                            from vibespatial.geometry.owned import (
                                device_mask_owned_capacity,
                            )

                            taken_owned = device_mask_owned_capacity(owned, d_keep)
                            return GeoSeries(
                                GeometryArray.from_owned(taken_owned, crs=series.crs),
                                crs=series.crs,
                            )
                    d_rows = cp.flatnonzero(d_keep).astype(cp.int64, copy=False)
                    rows = (
                        globals()["get_cuda_runtime"]()
                        .copy_device_to_host(
                            d_rows,
                            reason=(
                                "overlay keep-geometry-type pair source device rows terminal export"
                            ),
                        )
                        .astype(np.int64, copy=False)
                    )
                    taken = series.take(rows)
                    taken.reset_index(drop=True, inplace=True)
                    return taken
                keep = np.asarray(keep_mask, dtype=bool)
                if bool(np.all(keep)):
                    return series
                rows = np.flatnonzero(keep).astype(np.int64, copy=False)
                owned = getattr(series.values, "_owned", None)
                if owned is not None:
                    from vibespatial.runtime.residency import Residency

                    if owned.residency is Residency.DEVICE and has_gpu_runtime():
                        import cupy as cp

                        taken_owned = owned.device_take(
                            cp.asarray(rows, dtype=cp.int64),
                            host_indices_for_sizing=rows,
                        )
                    else:
                        taken_owned = owned.take(rows)
                    return GeoSeries(
                        GeometryArray.from_owned(taken_owned, crs=series.crs),
                        crs=series.crs,
                    )
                taken = series.take(rows)
                taken.reset_index(drop=True, inplace=True)
                return taken

            def _apply_polygon_keep_mask(keep_mask) -> None:
                nonlocal pair_left, pair_right
                capacity_keep_mask = _apply_intersection_pair_keep_mask(keep_mask)
                pair_left = _take_pair_source_rows(pair_left, capacity_keep_mask)
                pair_right = _take_pair_source_rows(pair_right, capacity_keep_mask)

            if native_lower_dim_geometry is not None:
                pass
            elif _preserve_lower_dim_polygon_results:
                geom_intersect = _assemble_polygon_intersection_rows_with_lower_dim(
                    pair_left,
                    pair_right,
                    geom_intersect,
                )
            elif _warn_on_dropped_lower_dim_polygon_results:
                geom_intersect, num_dropped, keep_mask = (
                    _filter_polygon_intersection_rows_for_keep_geom_type(
                        pair_left,
                        pair_right,
                        geom_intersect,
                        keep_geom_type_warning=True,
                        left_source=left_source_geoms,
                        right_source=right_source_geoms,
                        left_rows=source_idx1,
                        right_rows=source_idx2,
                    )
                )
                _apply_polygon_keep_mask(keep_mask)
                if num_dropped > 0:
                    warnings.warn(
                        "`keep_geom_type=True` in overlay resulted in "
                        f"{num_dropped} dropped geometries of different "
                        "geometry types than df1 has. Set `keep_geom_type=False` to retain all "
                        "geometries",
                        UserWarning,
                        stacklevel=4,
                    )
                keep_geom_type_applied = True
            elif prefer_exact_polygon_gpu:
                geom_intersect, _, keep_mask = _filter_polygon_intersection_rows_for_keep_geom_type(
                    pair_left,
                    pair_right,
                    geom_intersect,
                    keep_geom_type_warning=False,
                    left_source=left_source_geoms,
                    right_source=right_source_geoms,
                    left_rows=source_idx1,
                    right_rows=source_idx2,
                )
                _apply_polygon_keep_mask(keep_mask)
                keep_geom_type_applied = True

            if defer_post_intersection_make_valid and keep_geom_type_applied:
                current_pair_count = pair_count
                pre_make_valid_intersect = geom_intersect
                geom_intersect = _make_valid_geoseries(
                    geom_intersect,
                    dispatch_mode=(
                        ExecutionMode.GPU
                        if (used_owned or prefer_exact_polygon_gpu)
                        else ExecutionMode.AUTO
                    ),
                )
                if (
                    geom_intersect is not pre_make_valid_intersect
                    and len(geom_intersect) == current_pair_count
                ):
                    geom_intersect, _, post_repair_keep_mask = (
                        _filter_polygon_intersection_rows_for_keep_geom_type(
                            pair_left,
                            pair_right,
                            geom_intersect,
                            keep_geom_type_warning=False,
                            left_source=left_source_geoms,
                            right_source=right_source_geoms,
                            left_rows=(None if idx1 is None else np.asarray(idx1, dtype=np.intp)),
                            right_rows=(None if idx2 is None else np.asarray(idx2, dtype=np.intp)),
                        )
                    )
                    _apply_polygon_keep_mask(post_repair_keep_mask)

            if native_lower_dim_geometry is None:
                geom_intersect = _repair_invalid_polygon_output_rows(
                    geom_intersect,
                    preserve_lower_dimensional=_preserve_lower_dim_polygon_results,
                )
            if not _preserve_lower_dim_polygon_results:
                geom_intersect_owned = getattr(geom_intersect.values, "_owned", None)
                if geom_intersect_owned is None:
                    geom_values = _geoseries_object_values(geom_intersect)
                else:
                    geom_values = None
                if geom_values is not None and np.any(shapely.get_type_id(geom_values) == 7):
                    geom_intersect = GeoSeries(
                        _strip_non_polygon_collection_parts(geom_values),
                        index=geom_intersect.index,
                        crs=geom_intersect.crs,
                    )

        if native_lower_dim_geometry is None:
            geom_intersect_owned = getattr(geom_intersect.values, "_owned", None)
            if geom_intersect_owned is not None:
                device_state = getattr(geom_intersect_owned, "device_state", None)
                if (
                    device_state is not None
                    and device_state.trusted_all_valid is True
                    and device_state.trusted_all_non_empty is True
                ):
                    nonempty_mask = None
                else:
                    nonempty_mask = _owned_valid_nonempty_mask_device(
                        geom_intersect_owned,
                    )
                    if nonempty_mask is None:
                        nonempty_mask = _owned_valid_nonempty_mask(
                            geom_intersect_owned,
                        )
            else:
                nonempty_mask = ~(geom_intersect.isna() | geom_intersect.is_empty)
            if nonempty_mask is not None and hasattr(
                nonempty_mask,
                "__cuda_array_interface__",
            ):
                from vibespatial.geometry.owned import device_mask_owned_capacity

                capacity_keep_mask = _apply_intersection_pair_keep_mask(nonempty_mask)
                masked_owned = device_mask_owned_capacity(
                    geom_intersect_owned,
                    capacity_keep_mask,
                )
                geom_intersect = GeoSeries(
                    GeometryArray.from_owned(masked_owned, crs=geom_intersect.crs),
                    crs=geom_intersect.crs,
                )
            elif nonempty_mask is not None and not nonempty_mask.all():
                keep = np.asarray(nonempty_mask, dtype=bool)
                _apply_intersection_pair_keep_mask(keep)
                geom_intersect = geom_intersect[keep].reset_index(drop=True)
        use_device_relation = _has_device_indices and (pair_selection is not None or idx1 is None)
        relation_left = d_idx1 if use_device_relation else idx1
        relation_right = d_idx2 if use_device_relation else idx2
        relation_broadcast_right = (
            _many_vs_one_unique_right_value
            if _has_device_indices and idx2 is None and _many_vs_one_unique_right_value is not None
            else None
        )

        capacity_result = _pairwise_constructive_to_native_tabular_result(
            geometry=(
                native_lower_dim_geometry
                if native_lower_dim_geometry is not None
                else _geometry_native_result_from_geoseries(geom_intersect)
            ),
            relation=RelationIndexResult(
                relation_left,
                relation_right,
                broadcast_right_value=relation_broadcast_right,
            ),
            keep_geom_type_applied=keep_geom_type_applied,
            left_df=df1,
            right_df=df2,
        )
        if pair_selection is not None and keep_geom_type_applied:
            from vibespatial.geometry.buffers import GeometryFamily

            pair_selection = replace(
                pair_selection,
                geometry_family_domain=(
                    GeometryFamily.POLYGON,
                    GeometryFamily.MULTIPOLYGON,
                ),
                trusted_all_valid_rows=True,
            )
        result = (
            capacity_result
            if pair_selection is None
            else NativeTabularSelection(
                capacity_result=capacity_result,
                selection=pair_selection,
            )
        )
        return result, used_owned

    empty_geometry = GeoSeries([], index=pd.RangeIndex(0), crs=df1.crs, name="geometry")
    return (
        _pairwise_constructive_to_native_tabular_result(
            geometry=_geometry_native_result_from_geoseries(empty_geometry),
            relation=RelationIndexResult(
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            ),
            keep_geom_type_applied=False,
            left_df=df1,
            right_df=df2,
        ),
        used_owned,
    )


def _overlay_intersection(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
    _index_result=None,
    _polygon_inputs: bool | None = None,
):
    export_result, used_owned = _overlay_intersection_export_result(
        df1,
        df2,
        left_owned,
        right_owned,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
        _index_result=_index_result,
        _polygon_inputs=_polygon_inputs,
    )
    return export_result.to_geodataframe(), used_owned


def _overlay_difference_native(df1, df2, left_owned=None, right_owned=None, *, _index_result=None):
    """Build the native difference result before host-side export.

    Returns
    -------
    tuple[NativeTabularResult, bool]
        Native constructive result plus whether the owned dispatch path was used.
    """
    left_owned, right_owned = _coerce_owned_pair_for_strict_overlay(
        df1,
        df2,
        left_owned,
        right_owned,
    )
    # ADR-0042 low-level contract: spatial indexing may still emit index arrays.
    # Phase 2: pass owned arrays to request device-resident index pairs.
    index_result = (
        _index_result
        if _index_result is not None
        else _intersecting_index_pairs(
            df1,
            df2,
            left_owned=left_owned,
            right_owned=right_owned,
        )
    )

    # Unpack result: DeviceSpatialJoinResult (device arrays) or numpy. Keep
    # device pairs private for owned difference; host pairs are materialized
    # only for compatibility filters or the final public/index scatter boundary.
    if isinstance(index_result, DeviceSpatialJoinResult):
        d_idx1 = index_result.d_left_idx
        d_idx2 = index_result.d_right_idx
        idx1 = None
        idx2 = None
        pair_count = index_result.size
        _has_device_indices = True
    else:
        if isinstance(index_result, np.ndarray) and index_result.ndim == 2:
            idx1, idx2 = index_result
        else:
            idx1, idx2 = index_result
        idx1 = np.asarray(idx1, dtype=np.int32)
        idx2 = np.asarray(idx2, dtype=np.int32)
        pair_count = int(idx1.size)
        d_idx1, d_idx2 = None, None
        _has_device_indices = False

    def _ensure_host_difference_pairs() -> tuple[np.ndarray, np.ndarray]:
        nonlocal idx1, idx2
        if idx1 is None or idx2 is None:
            if not isinstance(index_result, DeviceSpatialJoinResult):
                raise RuntimeError("device index result missing for host pair export")
            idx1, idx2 = index_result.to_host()
            idx1 = np.asarray(idx1, dtype=np.int32)
            idx2 = np.asarray(idx2, dtype=np.int32)
        return idx1, idx2

    used_owned = False
    result_geoms = None
    result_owned = None

    # Owned-path dispatch: native grouped row-isolated overlay difference when
    # both DataFrames have owned backing. Avoids Shapely materialization and
    # keeps grouped constructive work on NativeGrouped/owned carriers.
    # Phase 18: uses concat_owned_scatter to keep the result device-resident
    # instead of materializing via to_shapely().
    #
    if pair_count == 0 and left_owned is not None:
        result_owned = left_owned
        used_owned = True
    elif pair_count > 0 and _should_use_owned_constructive_overlay(left_owned, right_owned):
        # Keep AUTO behavior for normal runs, but in strict-native mode force
        # the repo-owned GPU difference path here so overlay does not die on
        # the generic small-workload crossover before the polygon dispatcher
        # can choose its overlay-based GPU implementation for concave inputs.
        _pairwise_mode = ExecutionMode.GPU if strict_native_mode_enabled() else ExecutionMode.AUTO

        result_owned = _grouped_overlay_difference_capacity_owned(
            left_owned,
            right_owned,
            idx1,
            idx2,
            d_idx1,
            d_idx2,
            _has_device_indices,
            _pairwise_mode,
        )
        used_owned = True

    if result_owned is not None:
        # Device-resident path: wrap the full-row OwnedGeometryArray
        # directly in a GeoSeries, preserving the owned backing.
        differences = GeoSeries(
            GeometryArray.from_owned(result_owned, crs=df1.crs),
            index=df1.index,
        )
    else:
        if result_geoms is None:
            idx1, idx2 = _ensure_host_difference_pairs()
            result_geoms = _grouped_overlay_difference_geoms(df1, df2, idx1, idx2)

        differences = GeoSeries(result_geoms, index=df1.index, crs=df1.crs)

    # Post-difference make_valid: use GPU path when owned backing is
    # available to avoid Shapely materialisation on the critical path.
    differences = _make_valid_geoseries(
        differences,
        dispatch_mode=ExecutionMode.GPU if used_owned else ExecutionMode.AUTO,
    )
    differences_owned = getattr(differences.values, "_owned", None)
    if differences_owned is not None:
        device_keep_rows = _owned_valid_nonempty_mask_device(differences_owned)
        if device_keep_rows is not None:
            from vibespatial.geometry.owned import device_mask_owned_capacity

            masked_owned = device_mask_owned_capacity(
                differences_owned,
                device_keep_rows,
            )
            capacity_result = _left_constructive_capacity_to_native_tabular_result(
                geometry=GeometryNativeResult.from_owned(masked_owned, crs=df1.crs),
                df=df1,
                geometry_name=df1._geometry_column_name,
            )
            return NativeTabularSelection(
                capacity_result=capacity_result,
                selection=NativeDeviceSelection.from_mask(
                    device_keep_rows,
                    source_row_count=int(differences_owned.row_count),
                ),
            ), used_owned
        keep_rows = _owned_valid_nonempty_mask(differences_owned)
        keep_positions = np.flatnonzero(keep_rows).astype(np.int64, copy=False)
        geometry_result = GeometryNativeResult(
            crs=df1.crs,
            owned=differences_owned.take(keep_positions),
        )
    else:
        empty_mask = differences.is_empty
        if empty_mask.any():
            differences = differences.copy()
            differences.loc[empty_mask] = None
        keep_rows = ~empty_mask
        geom_diff = differences[keep_rows].copy()
        geometry_result = GeometryNativeResult.from_geoseries(geom_diff)
        keep_positions = np.flatnonzero(np.asarray(keep_rows, dtype=bool)).astype(
            np.int64, copy=False
        )
    return _left_constructive_to_native_tabular_result(
        geometry=geometry_result,
        row_positions=keep_positions,
        df=df1,
        geometry_name=df1._geometry_column_name,
    ), used_owned


def _overlay_difference(df1, df2, left_owned=None, right_owned=None, *, _index_result=None):
    export_result, used_owned = _overlay_difference_export_result(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=_index_result,
    )
    return export_result.to_geodataframe(), used_owned


def _overlay_intersection_export_result(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
    _index_result=None,
    _polygon_inputs: bool | None = None,
):
    """Build the native intersection export result before GeoDataFrame assembly."""
    return _overlay_intersection_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
        _index_result=_index_result,
        _polygon_inputs=_polygon_inputs,
    )


def _overlay_difference_export_result(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _index_result=None,
):
    """Build the native difference export result before GeoDataFrame assembly."""
    return _overlay_difference_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=_index_result,
    )


def _overlay_identity_native(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
):
    """Build the native identity result before the explicit GeoPandas export."""
    left_owned, right_owned = _coerce_owned_pair_for_strict_overlay(
        df1,
        df2,
        left_owned,
        right_owned,
    )
    forward_index_result = _intersecting_index_pairs(
        df1,
        df2,
        left_owned=left_owned,
        right_owned=right_owned,
    )
    intersection_native, used_inter = _overlay_intersection_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=forward_index_result,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
    )
    difference_native, used_diff = _overlay_difference_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=forward_index_result,
    )

    intersection_left = df1.reset_index(drop=True)
    intersection_right = df2.reset_index(drop=True)
    intersection_columns = _intersection_attribute_columns(intersection_left, intersection_right)
    difference_rename = {
        column: (column if column in intersection_columns else f"{column}_1")
        for column in df1.drop(df1._geometry_column_name, axis=1).columns
    }

    difference_native = _rename_native_tabular_result(
        difference_native,
        difference_rename,
        geometry_name="geometry",
    )
    native_result = _concat_native_tabular_results(
        [intersection_native, difference_native],
        geometry_name="geometry",
        crs=df1.crs,
    )
    return native_result, used_inter or used_diff


def _overlay_identity(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
):
    """Overlay Identity operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    native_result, used_owned = _overlay_identity_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
    )
    return native_result.to_geodataframe(), used_owned


def _overlay_symmetric_diff_native(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _forward_index_result=None,
    _reverse_index_result=None,
):
    """Build the native symmetric-difference result before explicit export."""
    left_owned, right_owned = _coerce_owned_pair_for_strict_overlay(
        df1,
        df2,
        left_owned,
        right_owned,
    )
    if _forward_index_result is None:
        _forward_index_result = _intersecting_index_pairs(
            df1,
            df2,
            left_owned=left_owned,
            right_owned=right_owned,
        )
    if _reverse_index_result is None:
        _reverse_index_result = _reverse_intersecting_index_pairs(
            _forward_index_result,
        )

    diff1_native, used1 = _overlay_difference_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=_forward_index_result,
    )
    diff2_native, used2 = _overlay_difference_native(
        df2,
        df1,
        right_owned,
        left_owned,
        _index_result=_reverse_index_result,
    )

    native_result = _symmetric_difference_native_tabular_results(
        _rename_native_tabular_result(
            diff1_native,
            None,
            geometry_name="geometry",
        ),
        _rename_native_tabular_result(
            diff2_native,
            None,
            geometry_name="geometry",
        ),
        geometry_name="geometry",
        crs=df1.crs,
    )
    return native_result, used1 or used2


def _overlay_symmetric_diff(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _forward_index_result=None,
    _reverse_index_result=None,
):
    """Overlay Symmetric Difference operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    native_result, used_owned = _overlay_symmetric_diff_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _forward_index_result=_forward_index_result,
        _reverse_index_result=_reverse_index_result,
    )
    return native_result.to_geodataframe(), used_owned


def _overlay_union_native(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
):
    """Build the native union result before the explicit GeoPandas export."""
    left_owned, right_owned = _coerce_owned_pair_for_strict_overlay(
        df1,
        df2,
        left_owned,
        right_owned,
    )
    forward_index_result = _intersecting_index_pairs(
        df1,
        df2,
        left_owned=left_owned,
        right_owned=right_owned,
    )
    intersection_native, used_inter = _overlay_intersection_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=forward_index_result,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
    )
    symmetric_native, used_sym = _overlay_symmetric_diff_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _forward_index_result=forward_index_result,
    )
    native_result = _concat_native_tabular_results(
        [intersection_native, symmetric_native],
        geometry_name="geometry",
        crs=df1.crs,
    )
    return native_result, used_inter or used_sym


def _overlay_union(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
):
    """Overlay Union operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    native_result, used_owned = _overlay_union_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
    )
    return native_result.to_geodataframe(), used_owned


def _reset_overlay_result_index(result: GeoDataFrame) -> GeoDataFrame:
    """Drop a non-default index without shedding owned geometry backing."""
    if isinstance(result.index, pd.RangeIndex):
        if result.index.start == 0 and result.index.step == 1 and result.index.stop == len(result):
            _maybe_seed_polygon_validity_cache(result)
            return result

    from vibespatial.api._native_state import attach_native_state, get_native_state

    native_state = get_native_state(result)
    geom_name = result._geometry_column_name
    geom_values = result.geometry.values
    attrs = result.attrs.copy()

    cached_owned = getattr(geom_values, "cached_owned", None)
    has_native_geometry = getattr(geom_values, "native_composition", None) is not None
    if not has_native_geometry:
        has_native_geometry = (
            cached_owned()
            if callable(cached_owned)
            else getattr(geom_values, "_owned", None)
        ) is not None

    if has_native_geometry:
        attrs_df = result.drop(columns=[geom_name]).reset_index(drop=True)
        geom_series = GeoSeries(
            geom_values,
            index=attrs_df.index,
            name=geom_name,
            crs=result.crs,
        )
        reset = GeoDataFrame(attrs_df).set_geometry(
            geom_series,
            crs=result.crs,
        )
    else:
        reset = result.reset_index(drop=True)
        if not isinstance(reset, GeoDataFrame):
            reset = GeoDataFrame(reset)
        if reset.crs is None and result.crs is not None:
            reset = reset.set_crs(result.crs)

    reset.attrs.update(attrs)
    _maybe_seed_polygon_validity_cache(reset)
    if native_state is not None and tuple(reset.columns) == native_state.column_order:
        try:
            attach_native_state(reset, native_state.with_index(reset.index))
        except ValueError:
            pass
    return reset


_KEEP_GEOM_TYPE_WARNING_MESSAGE = (
    "`keep_geom_type=True` in overlay resulted in dropped geometries of different "
    "geometry types than df1 has. Set `keep_geom_type=False` to retain all geometries"
)


def _overlay_keep_geom_type_warning_is_ignored() -> bool:
    """Return True when Python warning filters make the default overlay warning invisible."""
    module_name = __name__
    for action, message, category, module, lineno in warnings.filters:
        if not issubclass(UserWarning, category):
            continue
        if message is not None and message.match(_KEEP_GEOM_TYPE_WARNING_MESSAGE) is None:
            continue
        if module is not None and module.match(module_name) is None:
            continue
        if lineno not in (0, None):
            continue
        return action == "ignore"
    return False


def overlay(df1, df2, how="intersection", keep_geom_type=None, make_valid=True):
    """Perform spatial overlay between two GeoDataFrames.

    Currently only supports data GeoDataFrames with uniform geometry types,
    i.e. containing only (Multi)Polygons, or only (Multi)Points, or a
    combination of (Multi)LineString and LinearRing shapes.
    Implements several methods that are all effectively subsets of the union.

    See the User Guide page :doc:`../../user_guide/set_operations` for details.

    Parameters
    ----------
    df1 : GeoDataFrame
    df2 : GeoDataFrame
    how : string
        Method of spatial overlay: 'intersection', 'union',
        'identity', 'symmetric_difference' or 'difference'.
    keep_geom_type : bool
        If True, return only geometries of the same geometry type as df1 has,
        if False, return all resulting geometries. Default is None,
        which will set keep_geom_type to True but warn upon dropping
        geometries.
    make_valid : bool, default True
        If True, any invalid input geometries are corrected with a call to make_valid(),
        if False, a `ValueError` is raised if any input geometries are invalid.

    Returns
    -------
    df : GeoDataFrame
        GeoDataFrame with new set of polygons and attributes
        resulting from the overlay

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> polys1 = geopandas.GeoSeries([Polygon([(0,0), (2,0), (2,2), (0,2)]),
    ...                               Polygon([(2,2), (4,2), (4,4), (2,4)])])
    >>> polys2 = geopandas.GeoSeries([Polygon([(1,1), (3,1), (3,3), (1,3)]),
    ...                               Polygon([(3,3), (5,3), (5,5), (3,5)])])
    >>> df1 = geopandas.GeoDataFrame({'geometry': polys1, 'df1_data':[1,2]})
    >>> df2 = geopandas.GeoDataFrame({'geometry': polys2, 'df2_data':[1,2]})

    >>> geopandas.overlay(df1, df2, how='union')
        df1_data  df2_data                                           geometry
    0       1.0       1.0                POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1       2.0       1.0                POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2       2.0       2.0                POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))
    3       1.0       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    4       2.0       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...
    5       NaN       1.0  MULTIPOLYGON (((2 3, 2 2, 1 2, 1 3, 2 3)), ((3...
    6       NaN       2.0      POLYGON ((3 5, 5 5, 5 3, 4 3, 4 4, 3 4, 3 5))

    >>> geopandas.overlay(df1, df2, how='intersection')
       df1_data  df2_data                             geometry
    0         1         1  POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1         2         1  POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2         2         2  POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))

    >>> geopandas.overlay(df1, df2, how='symmetric_difference')
        df1_data  df2_data                                           geometry
    0       1.0       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    1       2.0       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...
    2       NaN       1.0  MULTIPOLYGON (((2 3, 2 2, 1 2, 1 3, 2 3)), ((3...
    3       NaN       2.0      POLYGON ((3 5, 5 5, 5 3, 4 3, 4 4, 3 4, 3 5))

    >>> geopandas.overlay(df1, df2, how='difference')
                                                geometry  df1_data
    0      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))         1
    1  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...         2

    >>> geopandas.overlay(df1, df2, how='identity')
       df1_data  df2_data                                           geometry
    0         1       1.0                POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1         2       1.0                POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2         2       2.0                POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))
    3         1       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    4         2       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...

    See Also
    --------
    sjoin : spatial join
    GeoDataFrame.overlay : equivalent method

    Notes
    -----
    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    # Allowed operations
    allowed_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",  # aka erase
    ]
    # Error Messages
    if how not in allowed_hows:
        raise ValueError(f"`how` was '{how}' but is expected to be in {allowed_hows}")

    if isinstance(df1, GeoSeries) or isinstance(df2, GeoSeries):
        raise NotImplementedError("overlay currently only implemented for GeoDataFrames")

    if not _check_crs(df1, df2):
        _crs_mismatch_warn(df1, df2, stacklevel=3)

    if keep_geom_type is None:
        keep_geom_type = True
        keep_geom_type_warning = not _overlay_keep_geom_type_warning_is_ignored()
    else:
        keep_geom_type_warning = False

    (
        left_all_polygons,
        left_has_polygons,
        left_has_lines,
        left_has_points,
    ) = _series_family_summary(df1.geometry)
    (
        right_all_polygons,
        right_has_polygons,
        right_has_lines,
        right_has_points,
    ) = _series_family_summary(df2.geometry)

    for i, df in enumerate([df1, df2]):
        if i == 0:
            poly_check = left_has_polygons
            lines_check = left_has_lines
            points_check = left_has_points
        else:
            poly_check = right_has_polygons
            lines_check = right_has_lines
            points_check = right_has_points
        if sum([poly_check, lines_check, points_check]) > 1:
            raise NotImplementedError(f"df{i + 1} contains mixed geometry types.")

    if how == "intersection" and not (
        _series_prefers_device_bounds_private(df1.geometry)
        or _series_prefers_device_bounds_private(df2.geometry)
    ):
        box_gdf1 = _series_total_bounds_private(df1.geometry)
        box_gdf2 = _series_total_bounds_private(df2.geometry)

        if not (
            ((box_gdf1[0] <= box_gdf2[2]) and (box_gdf2[0] <= box_gdf1[2]))
            and ((box_gdf1[1] <= box_gdf2[3]) and (box_gdf2[1] <= box_gdf1[3]))
        ):
            result = df1.iloc[:0].merge(
                df2.iloc[:0].drop(df2.geometry.name, axis=1),
                left_index=True,
                right_index=True,
                suffixes=("_1", "_2"),
            )
            return result[result.columns.drop(df1.geometry.name).tolist() + [df1.geometry.name]]

    # Computations
    boundary_left_owned = getattr(df1.geometry.values, "_owned", None)
    boundary_right_owned = getattr(df2.geometry.values, "_owned", None)
    boundary_prefers_exact_polygon_gpu = _should_prefer_exact_polygon_gpu(
        df1,
        df2,
        boundary_left_owned,
        boundary_right_owned,
        left_all_polygons=left_all_polygons,
        right_all_polygons=right_all_polygons,
    )
    boundary_make_valid_mode = (
        ExecutionMode.GPU if boundary_prefers_exact_polygon_gpu else ExecutionMode.AUTO
    )

    def _make_valid(df, *, dispatch_mode: ExecutionMode | str, all_polygons: bool):
        df = df.copy()
        if all_polygons:
            # GPU make_valid path: when owned backing is available, route
            # through make_valid_owned to keep data device-resident and
            # avoid Shapely materialisation on the overlay critical path.
            ga = df.geometry.values
            owned = getattr(ga, "_owned", None)
            if make_valid and owned is not None:
                from vibespatial.constructive.make_valid_pipeline import (
                    make_valid_owned,
                )

                mv_result = make_valid_owned(
                    owned=owned,
                    dispatch_mode=dispatch_mode,
                )
                if mv_result.repaired_rows.size == 0:
                    return df

                # Repair happened — prefer device-resident .owned
                # to avoid D->H transfer.
                new_ga = None
                if mv_result.owned is not None:
                    new_ga = GeometryArray.from_owned(
                        mv_result.owned,
                        crs=df.crs,
                    )
                if new_ga is None:
                    try:
                        from vibespatial.geometry.owned import (
                            from_shapely_geometries,
                        )

                        new_owned = from_shapely_geometries(
                            list(mv_result.geometries),
                        )
                        new_ga = GeometryArray.from_owned(
                            new_owned,
                            crs=df.crs,
                        )
                    except NotImplementedError:
                        new_ga = GeometryArray(
                            mv_result.geometries,
                            crs=df.crs,
                        )
                col = df._geometry_column_name
                df[col] = GeoSeries(new_ga, index=df.index)
                df = _collection_extract(df, geom_type="Polygon", keep_geom_type_warning=False)
                return df

            if owned is not None:
                from vibespatial.constructive.validity import is_valid_owned

                mask = ~np.asarray(is_valid_owned(owned), dtype=bool)
                if not bool(np.all(owned.validity)):
                    mask = mask.copy()
                    mask[~owned.validity] = False
                if mask.any():
                    raise ValueError(
                        "You have passed make_valid=False along with "
                        f"{mask.sum()} invalid input geometries. "
                        "Use make_valid=True or make sure that all geometries "
                        "are valid before using overlay."
                    )
                return df

            mask = ~df.geometry.is_valid
            col = df._geometry_column_name
            if make_valid:
                if mask.any():
                    df.loc[mask, col] = df.loc[mask, col].make_valid()
                    # Extract only the input geometry type, as make_valid may change it
                    df = _collection_extract(df, geom_type="Polygon", keep_geom_type_warning=False)

            elif mask.any():
                raise ValueError(
                    "You have passed make_valid=False along with "
                    f"{mask.sum()} invalid input geometries. "
                    "Use make_valid=True or make sure that all geometries "
                    "are valid before using overlay."
                )
        return df

    # Check the source geometry type before make_valid, as make_valid may change it.
    if keep_geom_type:
        geom_type = _series_first_geom_type(df1.geometry)
        if geom_type == "GeometryCollection":
            # GeoPandas defines keep_geom_type from df1's leading geometry
            # family. A leading source-side collection has no single family to
            # preserve. Later polygonal collections are normalized below.
            raise TypeError(
                "keep_geom_type can not be called on a GeoDataFrame with GeometryCollection."
            )

    cached_intersection_index_result = None
    if how == "intersection":
        cached_intersection_index_result = get_cached_intersection_pairs(
            df1,
            df2,
            return_device=has_gpu_runtime(),
        )
    cached_left_rows, cached_right_rows = _cached_intersection_unique_rows(
        cached_intersection_index_result,
    )
    cached_rows_valid = (
        _cached_relation_selection_rows_all_valid(
            df1.geometry,
            df2.geometry,
            cached_intersection_index_result,
        )
        if isinstance(cached_intersection_index_result, NativeRelationSelection)
        else (
            cached_left_rows is not None
            and cached_right_rows is not None
            and _candidate_rows_all_valid(df1.geometry, cached_left_rows)
            and _candidate_rows_all_valid(df2.geometry, cached_right_rows)
        )
    )
    reuse_cached_intersection_index = (
        cached_intersection_index_result is not None
        and make_valid
        and left_all_polygons
        and right_all_polygons
        and cached_rows_valid
    )
    if reuse_cached_intersection_index:
        df1 = df1.copy()
        df2 = df2.copy()
    else:
        cached_intersection_index_result = None
        df1 = _make_valid(
            df1,
            dispatch_mode=boundary_make_valid_mode,
            all_polygons=left_all_polygons,
        )
        df2 = _make_valid(
            df2,
            dispatch_mode=boundary_make_valid_mode,
            all_polygons=right_all_polygons,
        )

    if keep_geom_type and not left_all_polygons:
        df1, left_normalized = _normalize_polygonal_collection_input(df1)
        if left_normalized:
            (
                left_all_polygons,
                left_has_polygons,
                left_has_lines,
                left_has_points,
            ) = _series_family_summary(df1.geometry)
    if keep_geom_type and not right_all_polygons:
        df2, right_normalized = _normalize_polygonal_collection_input(df2)
        if right_normalized:
            (
                right_all_polygons,
                right_has_polygons,
                right_has_lines,
                right_has_points,
            ) = _series_family_summary(df2.geometry)

    candidate_pair_count = (
        _cached_intersection_pair_count(cached_intersection_index_result)
        if cached_intersection_index_result is not None
        else 0
    )

    # Extract owned arrays AFTER _make_valid.  GeometryArray.copy() now
    # preserves _owned backing, and __setitem__ invalidates it only for
    # mutated rows.  If _make_valid mutated all rows or dropped rows via
    # _collection_extract, _owned will already be None here.
    left_owned, right_owned = _extract_owned_pair(
        df1,
        df2,
        how=how,
        left_all_polygons=left_all_polygons,
        right_all_polygons=right_all_polygons,
    )
    if (
        how == "intersection"
        and cached_intersection_index_result is not None
        and not isinstance(
            cached_intersection_index_result,
            (DeviceSpatialJoinResult, NativeRelationSelection),
        )
        and left_owned is not None
        and right_owned is not None
        and has_gpu_runtime()
    ):
        from vibespatial.runtime.residency import Residency, combined_residency

        if combined_residency(left_owned, right_owned) is Residency.DEVICE:
            device_cached_intersection = get_cached_intersection_pairs(
                df1,
                df2,
                return_device=True,
            )
            if device_cached_intersection is not None:
                cached_intersection_index_result = device_cached_intersection
    prefer_exact_polygon_gpu = _should_prefer_exact_polygon_gpu(
        df1,
        df2,
        left_owned,
        right_owned,
        left_all_polygons=left_all_polygons,
        right_all_polygons=right_all_polygons,
    )
    overlay_plan = plan_overlay_operation(
        left_rows=len(df1),
        right_rows=len(df2),
        how=how,
        candidate_pair_count=candidate_pair_count,
        keep_geom_type=keep_geom_type,
        prefer_exact_polygon_gpu=prefer_exact_polygon_gpu,
        preserve_lower_dim_results=(keep_geom_type is False),
    )

    _used_owned = False
    with warnings.catch_warnings():  # CRS checked above, suppress array-level warning
        warnings.filterwarnings("ignore", message="CRS mismatch between the CRS")
        if how == "difference":
            result, _used_owned = _overlay_difference(
                df1,
                df2,
                left_owned,
                right_owned,
            )
        elif how == "intersection":
            result, _used_owned = _overlay_intersection(
                df1,
                df2,
                left_owned,
                right_owned,
                _prefer_exact_polygon_gpu=(prefer_exact_polygon_gpu),
                _preserve_lower_dim_polygon_results=(keep_geom_type is False),
                _warn_on_dropped_lower_dim_polygon_results=keep_geom_type_warning,
                _index_result=cached_intersection_index_result,
                _polygon_inputs=(left_all_polygons and right_all_polygons),
            )
        elif how == "symmetric_difference":
            result, _used_owned = _overlay_symmetric_diff(
                df1,
                df2,
                left_owned,
                right_owned,
            )
        elif how == "union":
            result, _used_owned = _overlay_union(
                df1,
                df2,
                left_owned,
                right_owned,
                _prefer_exact_polygon_gpu=(prefer_exact_polygon_gpu),
                _preserve_lower_dim_polygon_results=(keep_geom_type is False),
                _warn_on_dropped_lower_dim_polygon_results=keep_geom_type_warning,
            )
        elif how == "identity":
            result, _used_owned = _overlay_identity(
                df1,
                df2,
                left_owned,
                right_owned,
                _prefer_exact_polygon_gpu=(prefer_exact_polygon_gpu),
                _preserve_lower_dim_polygon_results=(keep_geom_type is False),
                _warn_on_dropped_lower_dim_polygon_results=keep_geom_type_warning,
            )

    record_dispatch_event(
        surface="geopandas.overlay",
        operation=f"overlay_{how}",
        implementation="owned_dispatch" if _used_owned else "shapely_host",
        reason=(
            f"{how} via owned-path dispatch"
            if _used_owned
            else "no owned backing or explicit CPU compatibility boundary"
        ),
        detail=(
            f"{overlay_plan.telemetry_detail(left_rows=len(df1), right_rows=len(df2), candidate_pair_count=candidate_pair_count)}, "
            f"owned={left_owned is not None}"
        ),
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.GPU if _used_owned else ExecutionMode.CPU,
    )

    if keep_geom_type and not result.attrs.get("_vibespatial_keep_geom_type_applied", False):
        result_values = result.geometry.values
        cached_owned = getattr(result_values, "cached_owned", None)
        result_owned = (
            cached_owned()
            if callable(cached_owned)
            else getattr(result_values, "_owned", None)
        )
        if result_owned is not None:
            result = _collection_extract_owned(result, geom_type, keep_geom_type_warning)
        elif getattr(result_values, "native_composition", None) is not None:
            result = _collection_extract_composition_native(
                result,
                geom_type,
                keep_geom_type_warning,
            )
        else:
            result = _collection_extract(result, geom_type, keep_geom_type_warning)

    if result.geometry.isna().any():
        result = result.loc[~result.geometry.isna()].copy()

    if how in ["intersection", "symmetric_difference", "union", "identity"]:
        drop_cols = [col for col in ("__idx1", "__idx2") if col in result.columns]
        if drop_cols:
            result.drop(drop_cols, axis=1, inplace=True)

    return _reset_overlay_result_index(result)


def _geom_type_to_target_families(geom_type: str) -> set[int] | None:
    """Map a Shapely geom_type string to the set of OwnedGeometryArray family tags to keep.

    Returns ``None`` if *geom_type* is not a recognized polygon, line, or point type.
    Imports are deferred to avoid circular dependencies.
    """
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    if geom_type in POLYGON_GEOM_TYPES:
        return {FAMILY_TAGS[GeometryFamily.POLYGON], FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]}
    if geom_type in LINE_GEOM_TYPES:
        return {
            FAMILY_TAGS[GeometryFamily.LINESTRING],
            FAMILY_TAGS[GeometryFamily.MULTILINESTRING],
        }
    if geom_type in POINT_GEOM_TYPES:
        return {FAMILY_TAGS[GeometryFamily.POINT], FAMILY_TAGS[GeometryFamily.MULTIPOINT]}
    return None


def _collection_extract_owned(df, geom_type, keep_geom_type_warning):
    """Device-resident collection extract: filter by geometry family tag.

    When the result GeoDataFrame's geometry column has OwnedGeometryArray
    backing, we can filter by the ``tags`` array directly -- no Shapely
    materialization, no ``.explode()``, no ``.dissolve()``.

    OwnedGeometryArray does not represent GeometryCollections; constituent
    geometries are stored as individual rows tagged by their concrete family.
    Filtering is a simple mask on the tags array followed by
    ``OwnedGeometryArray.take()``.
    """
    from vibespatial.geometry.owned import NULL_TAG

    ga = df.geometry.values
    owned = ga._owned

    target_tags = _geom_type_to_target_families(geom_type)
    if target_tags is None:
        raise TypeError(f"`geom_type` does not support {geom_type}.")

    tags = owned.tags
    keep_mask = np.zeros(len(tags), dtype=bool)
    for tag in target_tags:
        keep_mask |= tags == tag

    num_dropped = int((~keep_mask & (tags != NULL_TAG)).sum())

    if num_dropped > 0 and keep_geom_type_warning:
        warnings.warn(
            "`keep_geom_type=True` in overlay resulted in "
            f"{num_dropped} dropped geometries of different "
            "geometry types than df1 has. Set `keep_geom_type=False` to retain all "
            "geometries",
            UserWarning,
            stacklevel=2,
        )

    # Preserve null geometries only on the default keep_geom_type=None path,
    # which historically keeps bookkeeping rows that later warning-based
    # filtering may expose.  Explicit keep_geom_type=True should continue to
    # behave strictly and drop missing-geometry rows.
    if keep_geom_type_warning:
        keep_mask |= ~owned.validity

    if keep_mask.all():
        return df

    # Filter both the DataFrame rows and the owned geometry array together.
    # Use iloc for positional indexing -- the DataFrame may have a non-default
    # index after concat in overlay sub-operations.
    keep_indices = np.flatnonzero(keep_mask)
    result = df.iloc[keep_indices].copy()
    filtered_owned = owned.take(keep_indices)

    if (
        geom_type in POLYGON_GEOM_TYPES
        and "__idx1" in result.columns
        and "__idx2" in result.columns
        and len(result) > 1
    ):
        from vibespatial.kernels.constructive.segmented_union import segmented_union_all

        idx1_col = result["__idx1"]
        idx2_col = result["__idx2"]
        if idx1_col.notna().all() and idx2_col.notna().all():
            idx1 = idx1_col.to_numpy(dtype=np.int64, copy=False)
            idx2 = idx2_col.to_numpy(dtype=np.int64, copy=False)
            order = np.lexsort((idx2, idx1))
            if order.size > 1:
                idx1_sorted = idx1[order]
                idx2_sorted = idx2[order]
                group_starts = (
                    np.flatnonzero(
                        (idx1_sorted[1:] != idx1_sorted[:-1])
                        | (idx2_sorted[1:] != idx2_sorted[:-1])
                    )
                    + 1
                )
                group_offsets = np.concatenate(
                    [
                        np.asarray([0], dtype=np.int64),
                        group_starts.astype(np.int64, copy=False),
                        np.asarray([len(order)], dtype=np.int64),
                    ]
                )
                if len(group_offsets) - 1 != len(order):
                    result = result.iloc[order].iloc[group_offsets[:-1]].copy()
                    filtered_owned = segmented_union_all(filtered_owned.take(order), group_offsets)

    # Rebuild the GeoSeries with owned backing to avoid Shapely materialisation.
    geom_col = result._geometry_column_name
    result[geom_col] = GeoSeries(
        GeometryArray.from_owned(filtered_owned, crs=df.crs),
        index=result.index,
    )
    return result


def _collection_extract_composition_native(df, geom_type, keep_geom_type_warning):
    """Select target families from a native composition without host tag export."""
    from vibespatial.api._native_result_core import NativeTabularSelection
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.api._native_state import get_native_state
    from vibespatial.geometry.buffers import GeometryFamily

    if geom_type in POLYGON_GEOM_TYPES:
        families = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
    elif geom_type in LINE_GEOM_TYPES:
        families = (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
    elif geom_type in POINT_GEOM_TYPES:
        families = (GeometryFamily.POINT, GeometryFamily.MULTIPOINT)
    else:
        raise TypeError(f"`geom_type` does not support {geom_type}.")

    state = get_native_state(df)
    if state is None or state.geometry.composition is None:
        raise RuntimeError("native overlay composition lost its frame-state carrier")
    family_selection = state.geometry.select_family_domain_device(families)
    if family_selection is None:
        raise RuntimeError("native overlay composition lost device family metadata")
    selected_geometry, d_keep, d_drop_count = family_selection
    if keep_geom_type_warning:
        import cupy as cp

        num_dropped = _overlay_int_scalar(
            cp.asarray(d_drop_count).reshape(1),
            reason="native overlay composition keep-geometry-type warning count packet",
        )
        if num_dropped > 0:
            warnings.warn(
                "`keep_geom_type=True` in overlay resulted in "
                f"{num_dropped} dropped geometries of different "
                "geometry types than df1 has. Set `keep_geom_type=False` to retain all "
                "geometries",
                UserWarning,
                stacklevel=2,
            )

    capacity_result = replace(
        state.with_geometry_result(selected_geometry).to_native_tabular_result(),
        attrs={
            **state.attrs,
            "_vibespatial_keep_geom_type_applied": True,
        },
    )
    return NativeTabularSelection(
        capacity_result=capacity_result,
        selection=NativeDeviceSelection.from_mask(
            d_keep,
            source_token=state.lineage_token,
            source_row_count=state.row_count,
            geometry_family_domain=families,
            trusted_all_valid_rows=True,
        ),
    ).to_geodataframe()


def _collection_extract(df, geom_type, keep_geom_type_warning):
    # Check input
    if geom_type in POLYGON_GEOM_TYPES:
        geom_types = POLYGON_GEOM_TYPES
    elif geom_type in LINE_GEOM_TYPES:
        geom_types = LINE_GEOM_TYPES
    elif geom_type in POINT_GEOM_TYPES:
        geom_types = POINT_GEOM_TYPES
    else:
        raise TypeError(f"`geom_type` does not support {geom_type}.")

    result = df.copy()

    # First we filter the geometry types inside GeometryCollections objects
    # (e.g. GeometryCollection([polygon, point]) -> polygon)
    # we do this separately on only the relevant rows, as this is an expensive
    # operation (an expensive no-op for geometry types other than collections)
    is_collection = result.geom_type == "GeometryCollection"
    if is_collection.any():
        geom_col = result._geometry_column_name
        collections = result.loc[is_collection, [geom_col]]

        exploded = collections.reset_index(drop=True).explode(index_parts=True)
        exploded = exploded.reset_index(level=0)

        orig_num_geoms_exploded = exploded.shape[0]
        exploded.loc[~exploded.geom_type.isin(geom_types), geom_col] = None
        num_dropped_collection = orig_num_geoms_exploded - exploded.geometry.isna().sum()

        # level_0 created with above reset_index operation
        # and represents the original geometry collections
        # TODO avoiding dissolve to call union_all in this case could further
        # improve performance (we only need to collect geometries in their
        # respective Multi version)
        dissolved = exploded.dissolve(by="level_0")
        result.loc[is_collection, geom_col] = dissolved[geom_col].values
    else:
        num_dropped_collection = 0

    # Now we filter all geometries (in theory we don't need to do this
    # again for the rows handled above for GeometryCollections, but filtering
    # them out is probably more expensive as simply including them when this
    # is typically about only a few rows)
    orig_num_geoms = result.shape[0]
    geom_keep_mask = result.geom_type.isin(geom_types)
    if keep_geom_type_warning:
        geom_keep_mask = geom_keep_mask | result.geometry.isna()
    result = result.loc[geom_keep_mask]
    num_dropped = orig_num_geoms - result.shape[0]

    if (num_dropped > 0 or num_dropped_collection > 0) and keep_geom_type_warning:
        warnings.warn(
            "`keep_geom_type=True` in overlay resulted in "
            f"{num_dropped + num_dropped_collection} dropped geometries of different "
            "geometry types than df1 has. Set `keep_geom_type=False` to retain all "
            "geometries",
            UserWarning,
            stacklevel=2,
        )

    return result
