"""Module to clip vector data using GeoPandas."""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas.api.types
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._compat import PANDAS_GE_30
from vibespatial.api._native_results import (
    GeometryNativeResult,
    LeftConstructiveResult,
    NativeTabularResult,
    _clip_constructive_parts_to_native_tabular_result,
    _spatial_to_native_tabular_result,
)
from vibespatial.api.geometry_array import (
    LINE_GEOM_TYPES,
    POINT_GEOM_TYPES,
    POLYGON_GEOM_TYPES,
    GeometryArray,
    _check_crs,
    _crs_mismatch_warn,
)
from vibespatial.api.geometry_array import (
    from_shapely as _geometryarray_from_shapely,
)
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime._runtime import has_gpu_runtime
from vibespatial.runtime.config import SPATIAL_EPSILON
from vibespatial.runtime.fallbacks import record_fallback_event, strict_native_mode_enabled
from vibespatial.runtime.residency import Residency

_POLYGON_MASK_DIRECT_EXACT_MAX_ROWS = 8
_POLYGON_MASK_RECT_VALIDATED_MIN_ROWS = 128
_POLYGON_MASK_HOST_INTERSECTS_REPAIR_MAX_ROWS = 256
_DEVICE_CLIP_GEOM_TYPES = POINT_GEOM_TYPES | LINE_GEOM_TYPES | POLYGON_GEOM_TYPES
logger = logging.getLogger(__name__)


def _maybe_seed_polygon_validity_cache(spatial) -> None:
    geometry = spatial.geometry if isinstance(spatial, GeoDataFrame) else spatial
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None:
        return

    from vibespatial.geometry.buffers import GeometryFamily

    if not set(owned.families).issubset(
        {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    ):
        return

    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(owned)


def _mask_is_list_like_rectangle(mask):
    """
    Check if the input mask is list-like and not an instance of
    specific geometric types.

    Parameters
    ----------
    mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
        Polygon vector layer used to clip ``gdf``.

    Returns
    -------
    bool
        True if `mask` is list-like and not an instance of `GeoDataFrame`,
        `GeoSeries`, `Polygon`, or `MultiPolygon`, otherwise False.
    """
    return pandas.api.types.is_list_like(mask) and not isinstance(
        mask, GeoDataFrame | GeoSeries | Polygon | MultiPolygon
    )


def _rectangle_bounds_from_mask(mask):
    """Return rectangle bounds for axis-aligned rectangle masks, else None."""
    if _mask_is_list_like_rectangle(mask):
        return tuple(float(v) for v in mask)
    if isinstance(mask, MultiPolygon):
        return None
    if not isinstance(mask, Polygon) or mask.is_empty or len(mask.interiors) != 0:
        return None
    coords = np.asarray(mask.exterior.coords)
    if len(coords) != 5:
        return None
    xs = np.unique(coords[:4, 0])
    ys = np.unique(coords[:4, 1])
    if len(xs) != 2 or len(ys) != 2:
        return None
    return (float(xs[0]), float(ys[0]), float(xs[1]), float(ys[1]))


def _polygon_mask_allows_rectangle_kernel(mask) -> bool:
    """Return True when polygon-mask clip can safely use the rectangle kernel.

    The specialized polygon-rectangle kernel emits at most one polygon row per
    input row. Concave polygon masks can intersect a rectangle into multiple
    disjoint polygon parts, which violates public clip semantics. Keep this
    fast path only for convex scalar polygon masks.
    """
    if not isinstance(mask, Polygon) or mask.is_empty or len(mask.interiors) != 0:
        return False
    return bool(shapely.equals(mask, shapely.convex_hull(mask)))


def _bbox_candidate_rows_for_scalar_clip_mask(
    gdf,
    query_input,
    *,
    sort: bool = False,
) -> np.ndarray | None:
    """Return bbox candidate rows for scalar clip masks without building an sindex.

    The exact clip stage still performs the real geometric intersection. This
    helper only replaces the candidate query for the common scalar-mask cases
    where building/querying an index is more expensive than one vectorized
    bounds overlap pass. When the source geometry is already device-backed, the
    bounds overlap pass stays on the device instead of round-tripping through
    the generic candidate-query path.
    """
    if len(gdf) == 0:
        return np.empty(0, dtype=np.int32)

    if isinstance(query_input, GeoDataFrame | GeoSeries):
        if len(query_input) != 1:
            return None
        query_bounds = np.asarray(query_input.total_bounds, dtype=np.float64)
    elif isinstance(query_input, Polygon | MultiPolygon):
        query_bounds = np.asarray(query_input.bounds, dtype=np.float64)
    else:
        return None

    if query_bounds.shape != (4,):
        return None

    geometry = gdf.geometry if isinstance(gdf, GeoDataFrame) else gdf
    values = geometry.values
    owned = getattr(values, "_owned", None)
    device_promotion_supported = (
        has_gpu_runtime()
        and _clip_partition_supports_device_promotion(gdf)
    )
    if (
        owned is None
        and hasattr(values, "to_owned")
        and (
            strict_native_mode_enabled()
            or (device_promotion_supported and len(gdf) > 50_000)
        )
    ):
        try:
            owned = values.to_owned()
        except Exception:
            owned = None

    use_device_bounds = (
        device_promotion_supported
        and owned is not None
        and (
            strict_native_mode_enabled()
            or owned.residency is Residency.DEVICE
            or len(gdf) > 50_000
        )
    )
    if use_device_bounds:
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
            cp = None
        if cp is None:
            return None
        try:
            from vibespatial.kernels.core.geometry_analysis import (
                compute_geometry_bounds_device,
            )
            from vibespatial.runtime.residency import TransferTrigger

            if owned.residency is not Residency.DEVICE:
                owned = owned.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason=(
                        "clip scalar-mask bbox candidate query promoted supported "
                        "source geometry to device"
                    ),
                )
            d_bounds = cp.asarray(
                compute_geometry_bounds_device(owned),
                dtype=cp.float64,
            ).reshape(len(gdf), 4)
            xmin, ymin, xmax, ymax = query_bounds
            d_overlap_mask = (
                (d_bounds[:, 0] <= xmax)
                & (d_bounds[:, 2] >= xmin)
                & (d_bounds[:, 1] <= ymax)
                & (d_bounds[:, 3] >= ymin)
            )
            d_rows = cp.flatnonzero(d_overlap_mask).astype(cp.int32, copy=False)
            rows = cp.asnumpy(d_rows).astype(np.int32, copy=False)
            if rows.size <= 1:
                return rows
            if sort:
                order = np.argsort(np.asarray(gdf.index.take(rows)), kind="stable")
                return rows[order].astype(np.int32, copy=False)
            candidate_bounds = cp.asnumpy(d_bounds[d_rows]).reshape(-1, 4)
            order = np.lexsort(
                (
                    rows,
                    candidate_bounds[:, 3],
                    candidate_bounds[:, 2],
                    candidate_bounds[:, 1],
                    candidate_bounds[:, 0],
                )
            )
            return rows[order].astype(np.int32, copy=False)
        except Exception as exc:
            record_fallback_event(
                surface="geopandas.clip",
                reason=(
                    "device-backed scalar-mask bbox candidate query could not stay "
                    "on the device; falling back to the generic candidate query path"
                ),
                detail=f"{type(exc).__name__}: {exc}",
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.CPU,
                pipeline="_bbox_candidate_rows_for_scalar_clip_mask",
                d2h_transfer=True,
            )
            return None

    # Avoid O(n) bounds filtering for genuinely large clip workloads where the
    # flat spatial index amortizes its build/query cost better.
    if len(gdf) > 50_000:
        return None

    source_bounds = np.asarray(geometry.bounds, dtype=np.float64)
    if source_bounds.ndim != 2 or source_bounds.shape[1] != 4:
        return None

    xmin, ymin, xmax, ymax = query_bounds
    overlap_mask = (
        (source_bounds[:, 0] <= xmax)
        & (source_bounds[:, 2] >= xmin)
        & (source_bounds[:, 1] <= ymax)
        & (source_bounds[:, 3] >= ymin)
    )
    rows = np.flatnonzero(overlap_mask).astype(np.int32, copy=False)
    if rows.size <= 1:
        return rows

    if sort:
        order = np.argsort(np.asarray(gdf.index.take(rows)), kind="stable")
        return rows[order].astype(np.int32, copy=False)

    # Match the public "unsorted" contract by returning a deterministic spatial
    # encounter order rather than monotonic source-index order.
    candidate_bounds = source_bounds[rows]
    order = np.lexsort(
        (
            rows,
            candidate_bounds[:, 3],
            candidate_bounds[:, 2],
            candidate_bounds[:, 1],
            candidate_bounds[:, 0],
        )
    )
    return rows[order].astype(np.int32, copy=False)


def _geometry_series_from_values(values, *, index, crs, name=None):
    """Build a GeoSeries-like object without demoting extension-backed geometry."""
    if isinstance(values, GeometryArray | DeviceGeometryArray):
        result = pd.Series(values, index=index, copy=False, name=name)
        result.__class__ = GeoSeries
        result._crs = crs
        return result
    return GeoSeries(values, index=index, crs=crs, name=name)


def _geometry_column_series(values, *, index, crs, name):
    """Build a DataFrame geometry column while preserving extension backing."""
    if isinstance(values, GeometryArray | DeviceGeometryArray):
        return pd.Series(values, index=index, copy=False, name=name)
    return GeoSeries(values, index=index, crs=crs, name=name)


def _clip_partition_supports_device_promotion(partition) -> bool:
    if not isinstance(partition, GeoDataFrame | GeoSeries):
        return False
    geom_types = partition.geom_type
    supported_mask = geom_types.isna() | geom_types.isin(_DEVICE_CLIP_GEOM_TYPES)
    return bool(np.asarray(supported_mask, dtype=bool).all())


def _replace_geometry_column(frame, values):
    """Replace the active geometry column while preserving extension backing."""
    if isinstance(frame, GeoDataFrame):
        geom_name = frame._geometry_column_name
        geometry_series = _geometry_column_series(
            values,
            index=frame.index,
            crs=frame.crs,
            name=geom_name,
        )
        data_columns = {
            column_name: (geometry_series if column_name == geom_name else frame[column_name])
            for column_name in frame.columns
        }
        rebuilt = pd.DataFrame(data_columns, index=frame.index, copy=False)
        rebuilt.__class__ = GeoDataFrame
        rebuilt._geometry_column_name = geom_name
        rebuilt.attrs = frame.attrs.copy()
        return rebuilt

    return _geometry_series_from_values(
        values,
        index=frame.index,
        crs=frame.crs,
        name=getattr(frame, "name", None),
    )


def _take_spatial_rows(spatial, keep_mask):
    """Filter rows by position without forcing geometry object materialization."""
    keep_mask = np.asarray(keep_mask, dtype=bool)
    geometry = spatial.geometry if isinstance(spatial, GeoDataFrame) else spatial
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if keep_mask.all():
        if owned is None:
            return spatial
        if owned.residency is Residency.DEVICE and isinstance(values, DeviceGeometryArray):
            return spatial
        if owned.residency is not Residency.DEVICE and isinstance(values, GeometryArray):
            return spatial
        return _replace_geometry_column(
            spatial.copy(deep=not PANDAS_GE_30),
            _geometry_values_from_owned(owned, crs=getattr(spatial, "crs", None)),
        )
    keep_rows = np.flatnonzero(keep_mask).astype(np.intp, copy=False)
    filtered = spatial.iloc[keep_rows].copy(deep=not PANDAS_GE_30)
    if owned is None:
        return filtered

    if owned.residency is Residency.DEVICE and has_gpu_runtime():
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            taken_owned = owned.device_take(cp.asarray(keep_rows, dtype=cp.int64))
        else:
            taken_owned = owned.take(keep_rows)
    else:
        taken_owned = owned.take(keep_rows)
    taken_values = _geometry_values_from_owned(
        taken_owned,
        crs=getattr(spatial, "crs", None),
    )
    return _replace_geometry_column(filtered, taken_values)


def _record_clip_host_cleanup_fallback(*, detail: str, pipeline: str) -> None:
    record_fallback_event(
        surface="geopandas.clip",
        reason="clip requires host semantic cleanup before materialization",
        detail=detail,
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.CPU,
        pipeline=pipeline,
        d2h_transfer=True,
    )


def _geometry_values_from_owned(owned, *, crs):
    from vibespatial.runtime.residency import Residency

    if owned.residency is Residency.DEVICE:
        return DeviceGeometryArray._from_owned(owned, crs=crs)
    return GeometryArray.from_owned(owned, crs=crs)


def _take_geometry_object_values(values, rows: np.ndarray) -> np.ndarray:
    """Materialize only the selected rows as host geometry objects."""
    if rows.size == 0:
        return np.empty(0, dtype=object)
    return np.asarray(values.take(rows), dtype=object)


def _is_axis_aligned_rectangle_polygon(geom) -> bool:
    if geom is None or getattr(geom, "is_empty", False):
        return False
    if getattr(geom, "geom_type", None) != "Polygon" or len(geom.interiors) != 0:
        return False
    if len(geom.exterior.coords) != 5:
        return False
    return bool(geom.equals(geom.envelope))


def _all_axis_aligned_rectangle_polygons(values) -> bool:
    """Return True when every row is an axis-aligned rectangle polygon.

    Prefer the owned/native polygon metadata when available so rectangle-heavy
    parcel clips do not pay a per-row Shapely envelope check just to prove a
    shape that is already explicit in the coordinate buffers.
    """
    if len(values) == 0:
        return True

    owned = getattr(values, "_owned", None)
    if owned is None and hasattr(values, "to_owned"):
        try:
            owned = values.to_owned()
        except Exception:
            owned = None

    if owned is not None:
        try:
            from vibespatial.constructive.binary_constructive import (
                _host_rectangle_polygon_mask,
            )

            rect_mask = _host_rectangle_polygon_mask(owned)
        except Exception:
            rect_mask = None
        if rect_mask is not None and rect_mask.size == len(values):
            return bool(np.all(rect_mask))

    try:
        first_geom = values[0]
    except Exception:
        first_geom = None
    if first_geom is not None and not _is_axis_aligned_rectangle_polygon(first_geom):
        return False

    boundary_geoms = np.asarray(values, dtype=object)
    return all(_is_axis_aligned_rectangle_polygon(geom) for geom in boundary_geoms)


def _seed_rectangle_clip_validity_cache_if_safe(result_owned, source_values) -> None:
    """Mark rectangle clip output valid when the source rows are valid boxes."""
    if result_owned is None:
        return

    source_owned = getattr(source_values, "_owned", None)
    if source_owned is not None and source_owned.residency is Residency.DEVICE:
        try:
            from vibespatial.geometry.buffers import GeometryFamily
            from vibespatial.kernels.constructive.polygon_rect_intersection import (
                _device_rectangle_bounds,
            )

            device_polygon_buf = source_owned.device_state.families.get(
                GeometryFamily.POLYGON
            ) if source_owned.device_state is not None else None
            device_bounds = _device_rectangle_bounds(
                device_polygon_buf,
                source_owned.row_count,
            )
            if device_bounds is None:
                return
            xmin, ymin, xmax, ymax = device_bounds
            valid_rectangles = bool(
                ((xmax - xmin) > SPATIAL_EPSILON).all().item()
                and ((ymax - ymin) > SPATIAL_EPSILON).all().item()
            )
        except Exception:
            return
        if not valid_rectangles:
            return
    elif not _all_axis_aligned_rectangle_polygons(source_values):
        return

    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(result_owned)


def _exact_rectangle_clip_boundary_rows(
    boundary_values,
    boundary_bounds: np.ndarray,
    rectangle_bounds: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Return exact box-vs-box clip output when every boundary row is a rectangle."""
    if len(boundary_bounds) == 0:
        return np.empty(0, dtype=object)

    if not _all_axis_aligned_rectangle_polygons(boundary_values):
        return None

    rxmin, rymin, rxmax, rymax = rectangle_bounds
    result = np.empty(len(boundary_bounds), dtype=object)
    result[:] = None

    for row_index, bounds in enumerate(boundary_bounds):
        xmin = max(float(bounds[0]), rxmin)
        ymin = max(float(bounds[1]), rymin)
        xmax = min(float(bounds[2]), rxmax)
        ymax = min(float(bounds[3]), rymax)
        dx = xmax - xmin
        dy = ymax - ymin

        if dx < -SPATIAL_EPSILON or dy < -SPATIAL_EPSILON:
            continue
        if dx > SPATIAL_EPSILON and dy > SPATIAL_EPSILON:
            result[row_index] = box(xmin, ymin, xmax, ymax)
            continue
        if abs(dx) <= SPATIAL_EPSILON and abs(dy) <= SPATIAL_EPSILON:
            result[row_index] = Point(xmin, ymin)
            continue
        if abs(dx) <= SPATIAL_EPSILON:
            result[row_index] = LineString([(xmin, ymin), (xmin, ymax)])
            continue
        result[row_index] = LineString([(xmin, ymin), (xmax, ymin)])

    return result


def _exact_rectangle_clip_boundary_owned_rows(
    boundary_values,
    boundary_bounds: np.ndarray,
    rectangle_bounds: tuple[float, float, float, float],
):
    """Return device-native exact box-vs-box clip output when boundary rows are rectangles."""
    owned = getattr(boundary_values, "_owned", None)
    if owned is None and hasattr(boundary_values, "to_owned"):
        try:
            owned = boundary_values.to_owned()
        except Exception:
            return None
    if owned is None or owned.residency is not Residency.DEVICE:
        return None

    from vibespatial.constructive.envelope import _build_device_boxes_from_bounds
    from vibespatial.constructive.nonpolygon_binary_output import (
        build_device_backed_linestring_output,
    )
    from vibespatial.constructive.point import _build_device_backed_point_output
    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import build_null_owned_array, concat_owned_scatter
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        _device_rectangle_bounds,
        _host_rectangle_bounds,
    )

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by device residency
        cp = None

    boundary_rect_bounds = None
    if owned.device_state is not None:
        if cp is not None:
            device_polygon_buf = owned.device_state.families.get(GeometryFamily.POLYGON)
            if device_polygon_buf is not None:
                device_bounds = _device_rectangle_bounds(device_polygon_buf, owned.row_count)
                if device_bounds is not None:
                    rxmin, rymin, rxmax, rymax = rectangle_bounds
                    dxmin, dymin, dxmax, dymax = device_bounds
                    xmin = cp.maximum(dxmin, float(rxmin))
                    ymin = cp.maximum(dymin, float(rymin))
                    xmax = cp.minimum(dxmax, float(rxmax))
                    ymax = cp.minimum(dymax, float(rymax))
                    dx = xmax - xmin
                    dy = ymax - ymin

                    miss_mask = (dx < -SPATIAL_EPSILON) | (dy < -SPATIAL_EPSILON)
                    polygon_mask = (~miss_mask) & (dx > SPATIAL_EPSILON) & (dy > SPATIAL_EPSILON)
                    point_mask = (~miss_mask) & (cp.abs(dx) <= SPATIAL_EPSILON) & (
                        cp.abs(dy) <= SPATIAL_EPSILON
                    )
                    vertical_line_mask = (~miss_mask) & (cp.abs(dx) <= SPATIAL_EPSILON) & (
                        dy > SPATIAL_EPSILON
                    )
                    horizontal_line_mask = (~miss_mask) & (cp.abs(dy) <= SPATIAL_EPSILON) & (
                        dx > SPATIAL_EPSILON
                    )

                    result_owned = build_null_owned_array(
                        len(boundary_bounds),
                        residency=owned.residency,
                    )

                    polygon_rows = cp.flatnonzero(polygon_mask).astype(cp.int64, copy=False)
                    if int(polygon_rows.size) > 0:
                        polygon_bounds = cp.column_stack(
                            (
                                xmin[polygon_rows],
                                ymin[polygon_rows],
                                xmax[polygon_rows],
                                ymax[polygon_rows],
                            )
                        ).astype(cp.float64, copy=False)
                        polygon_owned = _build_device_boxes_from_bounds(
                            polygon_bounds,
                            row_count=int(polygon_rows.size),
                        )
                        result_owned = concat_owned_scatter(
                            result_owned,
                            polygon_owned,
                            polygon_rows,
                        )

                    point_rows = cp.flatnonzero(point_mask).astype(cp.int64, copy=False)
                    if int(point_rows.size) > 0:
                        point_owned = _build_device_backed_point_output(
                            xmin[point_rows].astype(cp.float64, copy=False),
                            ymin[point_rows].astype(cp.float64, copy=False),
                            row_count=int(point_rows.size),
                        )
                        result_owned = concat_owned_scatter(
                            result_owned,
                            point_owned,
                            point_rows,
                        )

                    line_rows = cp.flatnonzero(
                        vertical_line_mask | horizontal_line_mask
                    ).astype(cp.int64, copy=False)
                    if int(line_rows.size) > 0:
                        is_vertical = vertical_line_mask[line_rows]
                        x0 = xmin[line_rows]
                        y0 = ymin[line_rows]
                        x1 = cp.where(is_vertical, xmin[line_rows], xmax[line_rows])
                        y1 = cp.where(is_vertical, ymax[line_rows], ymin[line_rows])
                        line_x = cp.empty(int(line_rows.size) * 2, dtype=cp.float64)
                        line_y = cp.empty(int(line_rows.size) * 2, dtype=cp.float64)
                        line_x[0::2] = x0
                        line_x[1::2] = x1
                        line_y[0::2] = y0
                        line_y[1::2] = y1
                        line_owned = build_device_backed_linestring_output(
                            line_x,
                            line_y,
                            row_count=int(line_rows.size),
                            validity=np.ones(int(line_rows.size), dtype=bool),
                            geometry_offsets=np.arange(
                                0,
                                (int(line_rows.size) + 1) * 2,
                                2,
                                dtype=np.int32,
                            ),
                        )
                        result_owned = concat_owned_scatter(
                            result_owned,
                            line_owned,
                            line_rows,
                        )

                    return result_owned
    if boundary_rect_bounds is None and GeometryFamily.POLYGON in owned.families:
        host_bounds = _host_rectangle_bounds(
            owned.families[GeometryFamily.POLYGON],
            owned.row_count,
        )
        if host_bounds is not None:
            boundary_rect_bounds = np.column_stack(host_bounds).astype(np.float64, copy=False)
    if boundary_rect_bounds is None or len(boundary_rect_bounds) != len(boundary_bounds):
        return None

    rxmin, rymin, rxmax, rymax = rectangle_bounds
    xmin = np.maximum(boundary_rect_bounds[:, 0], rxmin)
    ymin = np.maximum(boundary_rect_bounds[:, 1], rymin)
    xmax = np.minimum(boundary_rect_bounds[:, 2], rxmax)
    ymax = np.minimum(boundary_rect_bounds[:, 3], rymax)
    dx = xmax - xmin
    dy = ymax - ymin

    miss_mask = (dx < -SPATIAL_EPSILON) | (dy < -SPATIAL_EPSILON)
    polygon_mask = (~miss_mask) & (dx > SPATIAL_EPSILON) & (dy > SPATIAL_EPSILON)
    point_mask = (~miss_mask) & (np.abs(dx) <= SPATIAL_EPSILON) & (np.abs(dy) <= SPATIAL_EPSILON)
    vertical_line_mask = (~miss_mask) & (np.abs(dx) <= SPATIAL_EPSILON) & (dy > SPATIAL_EPSILON)
    horizontal_line_mask = (~miss_mask) & (np.abs(dy) <= SPATIAL_EPSILON) & (dx > SPATIAL_EPSILON)

    runtime = get_cuda_runtime()
    result_owned = build_null_owned_array(len(boundary_bounds), residency=owned.residency)

    polygon_rows = np.flatnonzero(polygon_mask).astype(np.intp, copy=False)
    if polygon_rows.size:
        polygon_bounds = np.column_stack(
            (
                xmin[polygon_rows],
                ymin[polygon_rows],
                xmax[polygon_rows],
                ymax[polygon_rows],
            )
        ).astype(np.float64, copy=False)
        polygon_owned = _build_device_boxes_from_bounds(
            runtime.from_host(polygon_bounds),
            row_count=int(polygon_rows.size),
        )
        result_owned = concat_owned_scatter(result_owned, polygon_owned, polygon_rows)

    point_rows = np.flatnonzero(point_mask).astype(np.intp, copy=False)
    if point_rows.size:
        point_owned = _build_device_backed_point_output(
            runtime.from_host(xmin[point_rows].astype(np.float64, copy=False)),
            runtime.from_host(ymin[point_rows].astype(np.float64, copy=False)),
            row_count=int(point_rows.size),
        )
        result_owned = concat_owned_scatter(result_owned, point_owned, point_rows)

    line_rows = np.flatnonzero(vertical_line_mask | horizontal_line_mask).astype(np.intp, copy=False)
    if line_rows.size:
        is_vertical = vertical_line_mask[line_rows]
        x0 = xmin[line_rows]
        y0 = ymin[line_rows]
        x1 = np.where(is_vertical, xmin[line_rows], xmax[line_rows])
        y1 = np.where(is_vertical, ymax[line_rows], ymin[line_rows])
        line_x = np.empty(line_rows.size * 2, dtype=np.float64)
        line_y = np.empty(line_rows.size * 2, dtype=np.float64)
        line_x[0::2] = x0
        line_x[1::2] = x1
        line_y[0::2] = y0
        line_y[1::2] = y1
        line_owned = build_device_backed_linestring_output(
            runtime.from_host(line_x),
            runtime.from_host(line_y),
            row_count=int(line_rows.size),
            validity=np.ones(line_rows.size, dtype=bool),
            geometry_offsets=np.arange(0, (line_rows.size + 1) * 2, 2, dtype=np.int32),
        )
        result_owned = concat_owned_scatter(result_owned, line_owned, line_rows)

    return result_owned


@dataclass(frozen=True)
class ClipNativeResult:
    """Deferred clip export that preserves native geometry results until the boundary."""

    source: GeoDataFrame | GeoSeries
    parts: tuple[LeftConstructiveResult, ...]
    ordered_index: pd.Index
    ordered_row_positions: np.ndarray
    clipping_by_rectangle: bool
    has_non_point_candidates: bool
    keep_geom_type: bool

    def _materialize_parts(self):
        if not self.parts:
            return self.source.iloc[:0]

        if isinstance(self.source, GeoDataFrame):
            parts = [part.to_geodataframe(self.source) for part in self.parts]
        else:
            parts = [part.to_geoseries(self.source) for part in self.parts]

        if len(parts) == 1:
            return parts[0]

        concatenated = pd.concat(parts)
        all_row_positions = np.concatenate(
            [
                np.asarray(part.row_positions, dtype=np.intp)
                for part in self.parts
            ]
        )
        sorter = np.argsort(all_row_positions, kind="stable")
        order = sorter[
            np.searchsorted(
                all_row_positions[sorter],
                self.ordered_row_positions,
            )
        ]
        reordered = concatenated.iloc[order].copy(deep=not PANDAS_GE_30)
        reordered.index = self.ordered_index
        return reordered

    def _normalize_geometry_backing(self, clipped):
        from vibespatial.geometry.owned import from_shapely_geometries

        def _coerce_host_geometry_values(values):
            shapely_values = np.asarray(values, dtype=object)
            try:
                owned = from_shapely_geometries(
                    shapely_values,
                    residency=Residency.HOST,
                )
            except NotImplementedError:
                return _geometryarray_from_shapely(
                    shapely_values,
                    crs=self.source.crs,
                )
            return _geometry_values_from_owned(owned, crs=self.source.crs)

        if isinstance(clipped, GeoDataFrame):
            geom_name = clipped._geometry_column_name
            geom_values = clipped[geom_name].values
            if isinstance(geom_values, DeviceGeometryArray) or (
                isinstance(geom_values, GeometryArray) and geom_values._owned is not None
            ):
                return _replace_geometry_column(
                    clipped.copy(deep=not PANDAS_GE_30),
                    geom_values,
                )
            return _replace_geometry_column(
                clipped.copy(deep=not PANDAS_GE_30),
                _coerce_host_geometry_values(clipped[geom_name]),
            )

        values = clipped.values
        if isinstance(values, DeviceGeometryArray) or (
            isinstance(values, GeometryArray) and values._owned is not None
        ):
            return _replace_geometry_column(clipped, values)
        return _replace_geometry_column(
            clipped,
            _coerce_host_geometry_values(clipped),
        )

    def _filter_result(self, clipped):
        clipped_geometry = clipped.geometry if isinstance(clipped, GeoDataFrame) else clipped
        clipped_values = clipped_geometry.values

        def _coerce_owned_for_rows(row_mask, *, full_array: bool = False):
            current_owned = getattr(clipped_values, "_owned", None)
            if current_owned is not None:
                if current_owned.residency is not Residency.DEVICE:
                    if not has_gpu_runtime():
                        return None
                    from vibespatial.runtime.residency import TransferTrigger

                    try:
                        current_owned = current_owned.move_to(
                            Residency.DEVICE,
                            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                            reason=(
                                "clip cleanup promoted representable owned result "
                                "back to device"
                            ),
                        )
                    except Exception:
                        return None
                if full_array:
                    return current_owned
                row_ids = np.flatnonzero(np.asarray(row_mask, dtype=bool)).astype(
                    np.intp,
                    copy=False,
                )
                return current_owned.take(row_ids)

            if not strict_native_mode_enabled():
                return None

            from vibespatial.geometry.owned import from_shapely_geometries

            if full_array:
                values = np.asarray(clipped_geometry, dtype=object)
            else:
                values = np.asarray(clipped_geometry, dtype=object)[
                    np.asarray(row_mask, dtype=bool)
                ]
            if values.size == 0:
                return None
            try:
                return from_shapely_geometries(
                    values.tolist(),
                    residency=(
                        Residency.DEVICE
                        if has_gpu_runtime() and strict_native_mode_enabled()
                        else Residency.HOST
                    ),
                )
            except NotImplementedError:
                return None

        if self.clipping_by_rectangle:
            keep_mask = ~clipped_geometry.isna() & ~clipped_geometry.is_empty
            return _take_spatial_rows(clipped, keep_mask)

        keep = ~clipped_geometry.isna() & ~clipped_geometry.is_empty
        if self.has_non_point_candidates:
            current_owned = getattr(clipped_values, "_owned", None)
            poly_rows = clipped.geom_type.isin(POLYGON_GEOM_TYPES)
            if poly_rows.any():
                poly_mask = np.asarray(poly_rows, dtype=bool)
                poly_owned = _coerce_owned_for_rows(poly_mask)
                if poly_owned is not None:
                    from vibespatial.constructive.measurement import area_owned

                    nonpositive_area = (
                        np.asarray(area_owned(poly_owned), dtype=np.float64) <= 0.0
                    )
                else:
                    if (
                        strict_native_mode_enabled()
                        and current_owned is not None
                        and current_owned.residency is not Residency.DEVICE
                    ):
                        _record_clip_host_cleanup_fallback(
                            detail=(
                                "host polygon cleanup encountered host-backed "
                                "geometry values under strict native mode"
                            ),
                            pipeline="clip.to_spatial",
                        )
                    _record_clip_host_cleanup_fallback(
                        detail=(
                            "host polygon cleanup would materialize Shapely "
                            "objects for area filtering"
                        ),
                        pipeline="clip.to_spatial",
                    )
                    poly_values = np.asarray(clipped_geometry, dtype=object)[poly_mask]
                    nonpositive_area = np.asarray(
                        shapely.area(poly_values),
                        dtype=np.float64,
                    ) <= 0.0
                if np.any(nonpositive_area):
                    poly_bounds = np.asarray(clipped_geometry.bounds, dtype=np.float64)[poly_mask]
                    pointlike_zero_area = (
                        nonpositive_area
                        & (np.abs(poly_bounds[:, 2] - poly_bounds[:, 0]) <= SPATIAL_EPSILON)
                        & (np.abs(poly_bounds[:, 3] - poly_bounds[:, 1]) <= SPATIAL_EPSILON)
                    )
                    poly_keep = np.ones(poly_mask.sum(), dtype=bool)
                    poly_keep[nonpositive_area & ~pointlike_zero_area] = False
                    keep_array = np.array(keep, dtype=bool, copy=True)
                    keep_array[np.flatnonzero(poly_mask)] &= poly_keep
                    keep = keep_array

            line_rows = clipped.geom_type.isin(LINE_GEOM_TYPES)
            if line_rows.any():
                line_mask = np.asarray(line_rows, dtype=bool)
                line_owned = _coerce_owned_for_rows(line_mask)
                if line_owned is not None:
                    from vibespatial.constructive.measurement import length_owned

                    degenerate_lines = (
                        np.asarray(length_owned(line_owned), dtype=np.float64) == 0.0
                    )
                else:
                    if (
                        strict_native_mode_enabled()
                        and current_owned is not None
                        and current_owned.residency is not Residency.DEVICE
                    ):
                        multiline_rows = np.asarray(
                            clipped.geom_type == "MultiLineString",
                            dtype=bool,
                        )
                        if multiline_rows.any():
                            from vibespatial.constructive.measurement import length_owned

                            line_owned = current_owned.take(
                                np.flatnonzero(line_mask).astype(np.intp, copy=False)
                            )
                            degenerate_lines = (
                                np.asarray(length_owned(line_owned), dtype=np.float64) == 0.0
                            )
                        else:
                            _record_clip_host_cleanup_fallback(
                                detail=(
                                    "line cleanup encountered host-backed geometry "
                                    "values under strict native mode"
                                ),
                                pipeline="clip.to_spatial",
                            )
                            degenerate_lines = np.zeros(line_mask.sum(), dtype=bool)
                    else:
                        line_values = np.asarray(clipped_geometry, dtype=object)[line_mask]
                        degenerate_lines = np.asarray(
                            shapely.length(line_values),
                            dtype=np.float64,
                        ) == 0.0
                if np.any(degenerate_lines):
                    full_owned = _coerce_owned_for_rows(line_mask, full_array=True)
                    if full_owned is not None:
                        from vibespatial.constructive.centroid import centroid_owned
                        from vibespatial.constructive.extract_unique_points import (
                            extract_unique_points_owned,
                        )
                        from vibespatial.geometry.owned import concat_owned_scatter

                        degenerate_rows = np.flatnonzero(line_mask)[degenerate_lines].astype(
                            np.intp,
                            copy=False,
                        )
                        degenerate_owned = full_owned.take(degenerate_rows)
                        repaired_owned = centroid_owned(
                            extract_unique_points_owned(
                                degenerate_owned,
                                dispatch_mode=(
                                    ExecutionMode.GPU
                                    if degenerate_owned.residency is Residency.DEVICE
                                    else ExecutionMode.AUTO
                                ),
                            ),
                            dispatch_mode=(
                                ExecutionMode.GPU
                                if full_owned.residency is Residency.DEVICE
                                else ExecutionMode.AUTO
                            ),
                        )
                        full_owned = concat_owned_scatter(
                            full_owned,
                            repaired_owned,
                            degenerate_rows,
                        )
                        clipped = _replace_geometry_column(
                            clipped.copy(deep=not PANDAS_GE_30),
                            _geometry_values_from_owned(full_owned, crs=self.source.crs),
                        )
                    else:
                        _record_clip_host_cleanup_fallback(
                            detail=(
                                "line cleanup would materialize Shapely objects "
                                "for validity repair"
                            ),
                            pipeline="clip.to_spatial",
                        )
                        line_values = np.asarray(clipped_geometry, dtype=object)[line_mask]
                        repaired_values = np.asarray(clipped_geometry, dtype=object).copy()
                        repaired_values[np.flatnonzero(line_mask)[degenerate_lines]] = shapely.make_valid(
                            line_values[degenerate_lines]
                        )
                        clipped = _replace_geometry_column(
                            clipped.copy(deep=not PANDAS_GE_30),
                            _geometryarray_from_shapely(repaired_values, crs=self.source.crs),
                        )
                    clipped_geometry = clipped.geometry if isinstance(clipped, GeoDataFrame) else clipped
                    clipped_values = clipped_geometry.values
        return _take_spatial_rows(clipped, keep)

    def _apply_keep_geom_type(self, clipped):
        if not self.keep_geom_type:
            return clipped

        from vibespatial.api.tools.overlay import _strip_non_polygon_collection_parts

        geomcoll_concat = (clipped.geom_type == "GeometryCollection").any()
        geomcoll_orig = (self.source.geom_type == "GeometryCollection").any()
        new_collection = geomcoll_concat and not geomcoll_orig

        if geomcoll_orig:
            warnings.warn(
                "keep_geom_type can not be called on a "
                "GeoDataFrame with GeometryCollection.",
                stacklevel=3,
            )
            return clipped

        orig_types_total = sum(
            [
                self.source.geom_type.isin(POLYGON_GEOM_TYPES).any(),
                self.source.geom_type.isin(LINE_GEOM_TYPES).any(),
                self.source.geom_type.isin(POINT_GEOM_TYPES).any(),
            ]
        )
        clip_types_total = sum(
            [
                clipped.geom_type.isin(POLYGON_GEOM_TYPES).any(),
                clipped.geom_type.isin(LINE_GEOM_TYPES).any(),
                clipped.geom_type.isin(POINT_GEOM_TYPES).any(),
            ]
        )
        more_types = orig_types_total < clip_types_total

        if orig_types_total > 1:
            warnings.warn(
                "keep_geom_type can not be called on a mixed type GeoDataFrame.",
                stacklevel=3,
            )
            return clipped

        if new_collection or more_types:
            orig_type = self.source.geom_type.iloc[0]
            if orig_type in POLYGON_GEOM_TYPES:
                if new_collection:
                    _record_clip_host_cleanup_fallback(
                        detail=(
                            "keep_geom_type polygon cleanup would strip geometry "
                            "collection parts on the host"
                        ),
                        pipeline="clip.keep_geom_type",
                    )
                    geometry_values = np.asarray(
                        clipped.geometry if isinstance(clipped, GeoDataFrame) else clipped,
                        dtype=object,
                    )
                    cleaned = _strip_non_polygon_collection_parts(geometry_values)
                    keep = ~(
                        shapely.is_missing(cleaned)
                        | shapely.is_empty(cleaned)
                    )
                    if isinstance(clipped, GeoDataFrame):
                        clipped = _replace_geometry_column(
                            clipped.copy(deep=not PANDAS_GE_30),
                            _geometryarray_from_shapely(cleaned, crs=self.source.crs),
                        )
                    else:
                        clipped = GeoSeries(
                            _geometryarray_from_shapely(cleaned, crs=self.source.crs),
                            index=clipped.index,
                            crs=self.source.crs,
                            name=clipped.name,
                        )
                    clipped = clipped[keep]
                clipped = clipped.loc[clipped.geom_type.isin(POLYGON_GEOM_TYPES)]
            elif orig_type in LINE_GEOM_TYPES:
                if new_collection:
                    _record_clip_host_cleanup_fallback(
                        detail=(
                            "keep_geom_type line cleanup would explode geometry "
                            "collections on the host"
                        ),
                        pipeline="clip.keep_geom_type",
                    )
                    clipped = clipped.explode(index_parts=False)
                clipped = clipped.loc[clipped.geom_type.isin(LINE_GEOM_TYPES)]
        return clipped

    def to_spatial(self):
        clipped = self._materialize_parts()
        clipped = self._normalize_geometry_backing(clipped)
        clipped = self._filter_result(clipped)
        clipped = self._apply_keep_geom_type(clipped)
        _maybe_seed_polygon_validity_cache(clipped)
        return clipped

    def to_geodataframe(self) -> GeoDataFrame:
        clipped = self.to_spatial()
        if not isinstance(clipped, GeoDataFrame):
            raise TypeError("ClipNativeResult source is not a GeoDataFrame")
        return clipped

    def to_geoseries(self) -> GeoSeries:
        clipped = self.to_spatial()
        if not isinstance(clipped, GeoSeries):
            raise TypeError("ClipNativeResult source is not a GeoSeries")
        return clipped


@dataclass(frozen=True)
class _ClipPartitionOutput:
    geometry_values: object
    local_rows: np.ndarray


def _clip_native_part(source, row_positions: np.ndarray, geometry_values) -> LeftConstructiveResult:
    return LeftConstructiveResult(
        geometry=GeometryNativeResult.from_values(geometry_values, crs=source.crs),
        row_positions=np.asarray(row_positions, dtype=np.intp),
    )


def _as_geometry_values(values, *, crs):
    if isinstance(values, GeometryArray | DeviceGeometryArray):
        return values
    return _geometryarray_from_shapely(np.asarray(values, dtype=object), crs=crs)


def _promote_geometry_backing_to_device(frame, *, reason):
    """Rebuild a public geometry container with device-backed owned storage."""
    if not has_gpu_runtime() or not isinstance(frame, GeoDataFrame | GeoSeries):
        return frame

    values = frame.geometry.values if isinstance(frame, GeoDataFrame) else frame.values
    if isinstance(values, DeviceGeometryArray):
        return frame
    if not isinstance(values, GeometryArray):
        return frame

    from vibespatial.runtime.residency import Residency, TransferTrigger

    owned = values.to_owned()
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=reason,
    )
    return _replace_geometry_column(
        frame.copy(deep=not PANDAS_GE_30),
        DeviceGeometryArray._from_owned(owned, crs=frame.crs),
    )


def _clip_polygon_area_intersection_owned(
    left_owned,
    mask_owned,
    *,
    allow_rectangle_kernel: bool = True,
    prefer_exact_polygon_rect_batch: bool = False,
    prefer_many_vs_one_planner: bool = False,
):
    """Compute polygon-mask area intersections without shedding the GPU path.

    Once clip selects the polygon-owned execution family and a CUDA runtime is
    available, the area-producing step stays on the GPU many-vs-one path with
    containment bypass, SH clipping, and batched row-isolated exact remainder.
    The older direct exact path remains the CPU/no-GPU fallback.
    """
    from vibespatial.api.tools.overlay import _many_vs_one_intersection_owned
    from vibespatial.geometry.owned import materialize_broadcast, tile_single_row
    from vibespatial.runtime.residency import Residency

    if has_gpu_runtime() and left_owned.residency is Residency.DEVICE:
        from vibespatial.kernels.constructive.polygon_rect_intersection import (
            polygon_rect_intersection,
            polygon_rect_intersection_can_handle,
        )
        from vibespatial.runtime import ExecutionMode

        if (
            not prefer_many_vs_one_planner
            and (
                left_owned.row_count <= _POLYGON_MASK_DIRECT_EXACT_MAX_ROWS
                or prefer_exact_polygon_rect_batch
            )
        ):
            return _clip_polygon_area_intersection_gpu_owned(
                left_owned,
                mask_owned,
                allow_rectangle_kernel=allow_rectangle_kernel,
            )

        tiled_mask = (
            mask_owned
            if left_owned.row_count == 1
            else materialize_broadcast(
                tile_single_row(mask_owned, left_owned.row_count),
            )
        )
        if allow_rectangle_kernel and polygon_rect_intersection_can_handle(
            left_owned,
            tiled_mask,
        ):
            return polygon_rect_intersection(
                left_owned,
                tiled_mask,
                dispatch_mode=ExecutionMode.GPU,
            )
        if allow_rectangle_kernel and polygon_rect_intersection_can_handle(
            tiled_mask,
            left_owned,
        ):
            return polygon_rect_intersection(
                tiled_mask,
                left_owned,
                dispatch_mode=ExecutionMode.GPU,
            )
        if (
            left_owned.row_count >= _POLYGON_MASK_RECT_VALIDATED_MIN_ROWS
            and (
                polygon_rect_intersection_can_handle(left_owned, tiled_mask)
                or polygon_rect_intersection_can_handle(tiled_mask, left_owned)
            )
        ):
            validated = _clip_validated_polygon_rect_mask_intersection_owned(
                left_owned,
                mask_owned,
                tiled_mask,
            )
            if validated is not None:
                return validated

        return _many_vs_one_intersection_owned(
            left_owned,
            mask_owned,
            0,
        )

    from vibespatial.runtime import ExecutionMode

    try:
        return _many_vs_one_intersection_owned(
            left_owned,
            mask_owned,
            0,
        )
    except Exception as exc:
        record_fallback_event(
            surface="geopandas.clip",
            reason=(
                "many-vs-one polygon clip helper failed; "
                "falling back to host exact polygonal area extraction"
            ),
            detail=f"{type(exc).__name__}: {exc}",
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            pipeline="_clip_polygon_area_intersection_owned",
            d2h_transfer=False,
        )
        return _host_polygonal_area_intersection_owned(
            left_owned,
            mask_owned,
        )


def _clip_validated_polygon_rect_mask_intersection_owned(
    left_owned,
    mask_owned,
    tiled_mask_owned,
):
    """Use the rectangle kernel for scalar mask clips, repairing unsafe rows.

    The rectangle kernel is exact for rows where clipping the scalar polygon by
    the source rectangle yields a single valid polygon. Concave masks can yield
    disconnected output; those rows become invalid single-ring polygons, so they
    are detected with the GPU validity checker and replaced with the exact GPU
    many-vs-one intersection. This keeps the reusable rectangular-cell mask
    clip shape on device without broadening the unsafe fast path.
    """
    if left_owned.row_count == 0:
        return None
    if mask_owned.row_count != 1 or tiled_mask_owned.row_count != left_owned.row_count:
        return None
    if not has_gpu_runtime() or left_owned.residency is not Residency.DEVICE:
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover
        return None

    from vibespatial.constructive.make_valid_pipeline import make_valid_owned
    from vibespatial.constructive.validity import is_valid_owned
    from vibespatial.geometry.owned import device_concat_owned_scatter
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
        polygon_rect_intersection_can_handle,
        polygon_rect_split_boundary_components,
    )
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.dispatch import record_dispatch_event

    if polygon_rect_intersection_can_handle(left_owned, tiled_mask_owned):
        polygon_owned = left_owned
        rect_owned = tiled_mask_owned
        physical_shape = "mask_rectangle"
    elif polygon_rect_intersection_can_handle(tiled_mask_owned, left_owned):
        polygon_owned = tiled_mask_owned
        rect_owned = left_owned
        physical_shape = "source_rectangle"
    else:
        return None

    fast_owned = polygon_rect_intersection(
        polygon_owned,
        rect_owned,
        dispatch_mode=ExecutionMode.GPU,
    )
    if fast_owned.row_count != left_owned.row_count:
        return None

    valid_mask = np.asarray(
        is_valid_owned(fast_owned, dispatch_mode=ExecutionMode.GPU),
        dtype=bool,
    )
    if valid_mask.size != fast_owned.row_count:
        return None
    boundary_overlap = getattr(fast_owned, "_polygon_rect_boundary_overlap", None)
    boundary_repair_mask = None
    if boundary_overlap is not None:
        boundary_repair_mask = (
            boundary_overlap.get()
            if hasattr(boundary_overlap, "get")
            else np.asarray(boundary_overlap, dtype=bool)
        )
        boundary_repair_mask = np.asarray(boundary_repair_mask, dtype=bool)
        if boundary_repair_mask.size != fast_owned.row_count:
            boundary_repair_mask = None

    nonempty_mask = np.asarray(fast_owned.validity, dtype=bool)
    if nonempty_mask.size != fast_owned.row_count:
        return None

    repair_mask = nonempty_mask & ~valid_mask
    if boundary_repair_mask is not None:
        repair_mask = repair_mask | (nonempty_mask & boundary_repair_mask)
    repair_rows = np.flatnonzero(repair_mask).astype(np.intp, copy=False)
    if repair_rows.size == 0:
        record_dispatch_event(
            surface="geopandas.clip",
            operation="validated_polygon_rect_mask_clip",
            implementation="polygon_rect_intersection_validated_gpu",
            reason="rectangular-cell mask clip stayed on GPU with all rows validated",
            detail=(
                f"rows={left_owned.row_count}; repair_rows=0; "
                f"physical_shape={physical_shape}"
            ),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )
        return fast_owned

    d_repair_rows = cp.asarray(repair_rows, dtype=cp.int64)
    repair_fast_owned = fast_owned.device_take(d_repair_rows)
    repair_rect_owned = rect_owned.device_take(d_repair_rows)
    repair_owned = polygon_rect_split_boundary_components(
        repair_fast_owned,
        repair_rect_owned,
    )
    repair_impl = "polygon_rect_boundary_split_gpu"
    if repair_owned is not None and repair_owned.row_count == repair_rows.size:
        repair_valid = np.asarray(
            is_valid_owned(repair_owned, dispatch_mode=ExecutionMode.GPU),
            dtype=bool,
        )
        if repair_valid.size != repair_rows.size or not bool(repair_valid.all()):
            repair_owned = None

    if repair_owned is None or repair_owned.row_count != repair_rows.size:
        repair_result = make_valid_owned(
            owned=repair_fast_owned,
            method="structure",
            keep_collapsed=True,
            dispatch_mode=ExecutionMode.GPU,
        )
        repair_owned = repair_result.owned
        repair_impl = "gpu_make_valid_structure_repair"
        if (
            repair_result.selected is not ExecutionMode.GPU
            or repair_owned is None
            or repair_owned.row_count != repair_rows.size
        ):
            return None

    result = device_concat_owned_scatter(
        fast_owned,
        repair_owned,
        d_repair_rows,
    )

    if boundary_overlap is not None:
        if hasattr(boundary_overlap, "copy"):
            boundary_overlap = boundary_overlap.copy()
        if hasattr(boundary_overlap, "__setitem__"):
            boundary_overlap[d_repair_rows if hasattr(boundary_overlap, "get") else repair_rows] = False
        result._polygon_rect_boundary_overlap = boundary_overlap

    record_dispatch_event(
        surface="geopandas.clip",
        operation="validated_polygon_rect_mask_clip",
        implementation="polygon_rect_intersection_validated_gpu",
        reason=(
            "rectangular-cell mask clip used GPU rectangle kernel and repaired "
            "unsafe rows without leaving the device"
        ),
        detail=(
            f"rows={left_owned.row_count}; repair_rows={repair_rows.size}; "
            f"repair_impl={repair_impl}; "
            f"physical_shape={physical_shape}"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return result


def _clip_polygon_area_intersection_gpu_owned(
    left_owned,
    mask_owned,
    *,
    allow_rectangle_kernel: bool = True,
):
    """Run exact polygon-mask area intersection on GPU without CPU crossover."""
    from vibespatial.constructive.binary_constructive import _binary_constructive_gpu
    from vibespatial.geometry.owned import materialize_broadcast, tile_single_row
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
        polygon_rect_intersection_can_handle,
    )
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.dispatch import record_dispatch_event

    scalar_mask_owned = mask_owned
    if left_owned.row_count > 1:
        mask_owned = materialize_broadcast(
            tile_single_row(mask_owned, left_owned.row_count),
        )

    if allow_rectangle_kernel and polygon_rect_intersection_can_handle(
        left_owned,
        mask_owned,
    ):
        return polygon_rect_intersection(
            left_owned,
            mask_owned,
            dispatch_mode=ExecutionMode.GPU,
        )
    if allow_rectangle_kernel and polygon_rect_intersection_can_handle(
        mask_owned,
        left_owned,
    ):
        return polygon_rect_intersection(
            mask_owned,
            left_owned,
            dispatch_mode=ExecutionMode.GPU,
        )
    if (
        left_owned.row_count >= _POLYGON_MASK_RECT_VALIDATED_MIN_ROWS
        and (
            polygon_rect_intersection_can_handle(left_owned, mask_owned)
            or polygon_rect_intersection_can_handle(mask_owned, left_owned)
        )
    ):
        validated = _clip_validated_polygon_rect_mask_intersection_owned(
            left_owned,
            scalar_mask_owned,
            mask_owned,
        )
        if validated is not None:
            return validated

    result = _binary_constructive_gpu(
        "intersection",
        left_owned,
        mask_owned,
        dispatch_mode=ExecutionMode.GPU,
        _prefer_exact_polygon_intersection=True,
        _allow_rectangle_intersection_fast_path=allow_rectangle_kernel,
    )
    if result is None:
        raise RuntimeError(
            "GPU polygon-mask intersection returned no result for clip() "
            "after GPU execution was selected"
        )
    record_dispatch_event(
        surface="geopandas.array.intersection",
        operation="intersection",
        implementation="clip_polygon_exact_gpu",
        reason="clip polygon-mask exact subset stayed on the GPU exact path",
        detail=f"rows={left_owned.row_count}",
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return result


def _clip_polygon_single_pair_containment_owned(left_owned, mask_owned):
    """Return a device-native scalar polygon clip result for trivial containment.

    For the common ``1x1`` polygon clip shape, full exact overlay is wasted
    when one polygon wholly contains the other. Reuse the existing GPU
    containment bypass kernels in both directions:

    - ``left inside mask`` -> clip result is ``left``
    - ``mask inside left`` -> clip result is ``mask``

    Return ``None`` when neither bypass applies so the caller can continue to
    the exact constructive path.
    """
    from vibespatial.overlay.bypass import _containment_bypass_gpu
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.dispatch import record_dispatch_event
    from vibespatial.runtime.residency import Residency

    if (
        left_owned.row_count != 1
        or mask_owned.row_count != 1
        or not has_gpu_runtime()
        or left_owned.residency is not Residency.DEVICE
        or mask_owned.residency is not Residency.DEVICE
    ):
        return None

    try:
        left_inside_mask, left_remainder = _containment_bypass_gpu(
            left_owned,
            mask_owned,
            "intersection",
        )
        if left_inside_mask is not None and left_remainder is None:
            record_dispatch_event(
                surface="geopandas.clip",
                operation="intersection",
                implementation="clip_polygon_single_pair_containment_bypass",
                reason="single-row polygon clip returned the source polygon via GPU containment bypass",
                detail="left_inside_mask",
                selected=ExecutionMode.GPU,
            )
            return left_inside_mask

        from vibespatial.kernels.constructive.polygon_rect_intersection import (
            polygon_rect_intersection_can_handle,
        )

        if polygon_rect_intersection_can_handle(mask_owned, left_owned):
            mask_inside_left, mask_remainder = _containment_bypass_gpu(
                mask_owned,
                left_owned,
                "intersection",
            )
            if mask_inside_left is not None and mask_remainder is None:
                record_dispatch_event(
                    surface="geopandas.clip",
                    operation="intersection",
                    implementation="clip_polygon_single_pair_containment_bypass",
                    reason=(
                        "single-row polygon clip returned the mask polygon via "
                        "GPU containment bypass against a rectangular source"
                    ),
                    detail="mask_inside_rectangular_left",
                    selected=ExecutionMode.GPU,
                )
                return mask_inside_left
    except Exception:
        logger.debug(
            "single-row polygon clip containment bypass failed; continuing to exact constructive path",
            exc_info=True,
        )
    return None


def _host_polygonal_area_intersection_owned(left_owned, right_owned):
    """Host exact intersection for clip's polygonal area-only contract."""
    from vibespatial.api.tools.overlay import _strip_non_polygon_collection_parts
    from vibespatial.geometry.owned import from_shapely_geometries

    left_values = np.asarray(left_owned.to_shapely(), dtype=object)
    if right_owned.row_count == 1 and left_values.size > 1:
        right_geom = right_owned.to_shapely()[0]
        right_values = np.full(left_values.size, right_geom, dtype=object)
    else:
        right_values = np.asarray(right_owned.to_shapely(), dtype=object)

    raw = np.asarray(shapely.intersection(left_values, right_values), dtype=object)
    polygonal = _strip_non_polygon_collection_parts(raw)
    area_only = np.asarray(
        [
            geom
            if (
                geom is not None
                and getattr(geom, "geom_type", None) in POLYGON_GEOM_TYPES
                and not getattr(geom, "is_empty", False)
            )
            else None
            for geom in polygonal
        ],
        dtype=object,
    )
    return from_shapely_geometries(area_only.tolist(), residency=left_owned.residency)


def _clip_polygon_rectangle_area_intersection_owned(
    left_owned,
    rectangle_bounds: tuple[float, float, float, float],
):
    """Compute polygon-only area intersections for polygon-vs-rectangle clip.

    Rectangle clip wants a stricter contract than the generic many-vs-one
    helper: first build polygonal area only, then recover lower-dimensional
    remnants separately for public clip semantics. When a GPU is present, keep
    this stage on the exact GPU constructive path so the polygon-rectangle
    kernel can claim eligible rows and the fallback exact GPU polygon path can
    handle the rest without silently bouncing to CPU.
    """
    from vibespatial.constructive.binary_constructive import (
        _binary_constructive_gpu,
    )
    from vibespatial.geometry.owned import (
        build_null_owned_array,
        concat_owned_scatter,
        from_shapely_geometries,
        materialize_broadcast,
        tile_single_row,
    )
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.residency import Residency

    xmin, ymin, xmax, ymax = rectangle_bounds
    rectangle_mask = box(xmin, ymin, xmax, ymax)
    rectangle_owned = from_shapely_geometries(
        [rectangle_mask],
        residency=left_owned.residency,
    )
    if left_owned.row_count > 1:
        rectangle_owned = materialize_broadcast(
            tile_single_row(rectangle_owned, left_owned.row_count),
        )

    if has_gpu_runtime() and left_owned.residency is Residency.DEVICE:
        result = _binary_constructive_gpu(
            "intersection",
            left_owned,
            rectangle_owned,
            dispatch_mode=ExecutionMode.GPU,
            _prefer_exact_polygon_intersection=True,
        )
        if result is None:
            raise RuntimeError(
                "GPU polygon-rectangle intersection returned no result for clip() "
                "after GPU execution was selected"
            )
    else:
        record_fallback_event(
            surface="geopandas.clip",
            reason="polygon-rectangle clip used host exact polygonal area extraction",
            detail=f"rows={left_owned.row_count}",
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            pipeline="_clip_polygon_rectangle_area_intersection_owned",
            d2h_transfer=left_owned.residency is Residency.DEVICE,
        )
        result = _host_polygonal_area_intersection_owned(
            left_owned,
            rectangle_owned,
        )

    positive_rows = np.flatnonzero(_owned_nonempty_polygon_mask(result)).astype(
        np.intp,
        copy=False,
    )
    if positive_rows.size == result.row_count:
        return result
    if positive_rows.size == 0:
        return build_null_owned_array(
            left_owned.row_count,
            residency=result.residency,
        )
    return concat_owned_scatter(
        build_null_owned_array(
            left_owned.row_count,
            residency=result.residency,
        ),
        result.take(positive_rows),
        positive_rows,
    )


def _clip_multipolygon_rectangle_keep_geom_type_owned(
    left_owned,
    rectangle_bounds: tuple[float, float, float, float],
):
    """Recover polygonal rectangle clip output for multipolygon rows on device.

    Rectangle `keep_geom_type=True` only needs the polygonal area portion of
    the public intersection result. For multipolygon rows, the full-row exact
    intersection path can degrade into mixed GeometryCollection semantics that
    are hard to preserve in the owned family model. Explode the multipolygon
    into polygon parts on device, clip each part through the polygon-rectangle
    area path, then regroup the surviving polygon parts back to the original
    row ids without leaving the device.
    """
    if not has_gpu_runtime() or left_owned.row_count == 0:
        return None
    if left_owned.residency is not Residency.DEVICE:
        return None

    from vibespatial.constructive.binary_constructive import (
        _explode_multipolygon_rows_to_polygons_gpu,
        _regroup_intersection_parts_with_grouped_union_gpu,
    )
    from vibespatial.geometry.owned import build_null_owned_array

    try:
        import cupy as cp
    except Exception:  # pragma: no cover - exercised only on CPU-only installs
        return None

    try:
        exploded_left, d_source_rows = _explode_multipolygon_rows_to_polygons_gpu(
            left_owned,
        )
    except Exception:
        logger.debug(
            "device multipolygon rectangle keep_geom_type rescue failed during explode",
            exc_info=True,
        )
        return None

    if exploded_left.row_count == 0:
        return build_null_owned_array(
            left_owned.row_count,
            residency=left_owned.residency,
        )

    try:
        part_result = _clip_polygon_rectangle_area_intersection_owned(
            exploded_left,
            rectangle_bounds,
        )
    except Exception:
        logger.debug(
            "device multipolygon rectangle keep_geom_type rescue failed during part clip",
            exc_info=True,
        )
        return None

    positive_rows = np.flatnonzero(_owned_nonempty_polygon_mask(part_result)).astype(
        np.int64,
        copy=False,
    )
    if positive_rows.size == 0:
        return build_null_owned_array(
            left_owned.row_count,
            residency=left_owned.residency,
        )

    d_positive_rows = cp.asarray(positive_rows, dtype=cp.int64)
    valid_parts = part_result.take(d_positive_rows)
    try:
        regrouped = _regroup_intersection_parts_with_grouped_union_gpu(
            valid_parts,
            d_source_rows[d_positive_rows],
            output_row_count=left_owned.row_count,
            dispatch_mode=ExecutionMode.GPU,
        )
    except Exception:
        logger.debug(
            "device multipolygon rectangle keep_geom_type rescue failed during regroup",
            exc_info=True,
        )
        return None
    return regrouped


def _clip_polygon_boundary_touch_mask(
    source_values,
    left_owned,
    boundary_rows: np.ndarray,
    *,
    mask,
    mask_owned,
) -> np.ndarray:
    """Return exact mask intersections for clip boundary rows.

    For device-backed polygon clip workloads, this keeps the boundary cull on
    the GPU by promoting the scalar mask to a broadcasted owned array and
    forcing the predicate stack down the GPU refine path. Host-backed public
    clip still uses the existing Shapely predicate path.
    """
    if boundary_rows.size == 0:
        return np.empty(0, dtype=bool)

    from vibespatial.runtime.residency import Residency

    if has_gpu_runtime() and left_owned.residency is Residency.DEVICE:
        from vibespatial.predicates.binary import NullBehavior, evaluate_binary_predicate
        from vibespatial.runtime import ExecutionMode

        boundary_owned = left_owned.take(boundary_rows)
        result = evaluate_binary_predicate(
            "intersects",
            boundary_owned,
            mask,
            dispatch_mode=ExecutionMode.GPU,
            null_behavior=NullBehavior.FALSE,
        )
        return np.asarray(result.values, dtype=bool)

    return np.asarray(source_values.take(boundary_rows).intersects(mask), dtype=bool)


def _owned_nonempty_polygon_mask(owned) -> np.ndarray:
    """Return rows backed by polygonal output with strictly positive area."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    if owned.residency is Residency.DEVICE and has_gpu_runtime():
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            from vibespatial.constructive.measurement import _area_gpu_device_fp64
            from vibespatial.cuda._runtime import get_cuda_runtime

            device_state = owned._ensure_device_state()
            d_tags = cp.asarray(device_state.tags)
            d_validity = cp.asarray(device_state.validity)
            d_polygonal_mask = (
                (d_tags == FAMILY_TAGS[GeometryFamily.POLYGON])
                | (d_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
            )
            d_areas = _area_gpu_device_fp64(owned)
            d_keep = d_validity & d_polygonal_mask & cp.isfinite(d_areas) & (d_areas > 0.0)
            return np.asarray(get_cuda_runtime().copy_device_to_host(d_keep), dtype=bool)

    from vibespatial.constructive.measurement import area_owned

    validity = np.asarray(owned.validity, dtype=bool)
    if not validity.any():
        return validity

    tags = np.asarray(owned.tags)
    polygon_tags = np.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=tags.dtype if tags.size else np.int8,
    )
    polygonal_mask = validity & np.isin(tags, polygon_tags)
    if not polygonal_mask.any():
        return np.zeros(len(tags), dtype=bool)

    areas = np.asarray(area_owned(owned), dtype=np.float64)
    if areas.size != len(tags):
        return np.zeros(len(tags), dtype=bool)
    return polygonal_mask & np.isfinite(areas) & (areas > 0.0)


def _exact_polygon_clip_boundary_rows(
    left_values,
    right_values,
) -> np.ndarray:
    """Return the exact host boundary rows for polygon-mask clip semantics."""
    return np.asarray(
        shapely.intersection(
            np.asarray(left_values, dtype=object),
            np.asarray(right_values, dtype=object),
        ),
        dtype=object,
    )


def _clip_boundary_row_matches_area(assembled_geom, area_geom) -> bool:
    """Return True when area-only output already matches public clip semantics.

    Topological equality is too weak here: degenerate polygon outputs can be
    point-set-equal to lower-dimensional artifacts, and near-equal polygonal
    rows can still drift numerically from the exact host boundary result. Keep
    the cheaper area result only when it is the same public geometry after
    normalization.
    """
    if assembled_geom is None and area_geom is None:
        return True
    if assembled_geom is None or area_geom is None:
        return False
    if getattr(assembled_geom, "geom_type", None) != getattr(area_geom, "geom_type", None):
        return False
    return bool(
        shapely.equals_exact(
            assembled_geom,
            area_geom,
            tolerance=0.0,
            normalize=True,
        )
    )


def _clip_polygon_partition_with_rectangle_mask(
    partition,
    rectangle_bounds: tuple[float, float, float, float],
    *,
    keep_geom_type_only: bool = False,
):
    """Clip polygon rows to a rectangle mask while preserving sliver leftovers.

    Rows fully inside the rectangle are pass-through. Only rows that cross or
    touch the rectangle boundary require exact area intersection to
    preserve lower-dimensional public clip semantics.
    """
    partition = _promote_geometry_backing_to_device(
        partition,
        reason="clip rectangle-mask polygon partition selected GPU-native constructive path",
    )
    xmin, ymin, xmax, ymax = rectangle_bounds
    from vibespatial.api.tools.overlay import (
        _assemble_polygon_intersection_rows_with_lower_dim,
    )
    from vibespatial.geometry.owned import (
        build_null_owned_array,
        concat_owned_scatter,
        from_shapely_geometries,
    )
    from vibespatial.runtime.residency import Residency

    rectangle_mask = box(xmin, ymin, xmax, ymax)
    source_values = partition.geometry.values if isinstance(partition, GeoDataFrame) else partition.values
    source_is_native = (
        isinstance(source_values, DeviceGeometryArray)
        or getattr(source_values, "_owned", None) is not None
    )
    source_owned = source_values.to_owned() if source_is_native else None
    if keep_geom_type_only and source_owned is None:
        source_owned = source_values.to_owned()
    assembled = np.empty(len(partition), dtype=object)
    assembled[:] = None

    source_missing = np.asarray(source_values.isna() | source_values.is_empty, dtype=bool)
    source_bounds = np.asarray(source_values.bounds, dtype=np.float64)
    fully_inside_mask = (
        ~source_missing
        & (source_bounds[:, 0] >= xmin)
        & (source_bounds[:, 1] >= ymin)
        & (source_bounds[:, 2] <= xmax)
    )
    fully_inside_mask &= source_bounds[:, 3] <= ymax

    inside_rows = np.flatnonzero(fully_inside_mask).astype(np.intp, copy=False)
    if inside_rows.size > 0 and source_owned is None:
        assembled[inside_rows] = np.asarray(
            source_values.take(inside_rows),
            dtype=object,
        )

    boundary_rows = np.flatnonzero(~source_missing & ~fully_inside_mask).astype(np.intp, copy=False)
    if keep_geom_type_only:
        assert source_owned is not None
        result_owned = build_null_owned_array(
            len(partition),
            residency=source_owned.residency,
        )
        if inside_rows.size > 0:
            result_owned = concat_owned_scatter(
                result_owned,
                source_owned.take(inside_rows),
                inside_rows,
            )
        if boundary_rows.size > 0:
            boundary_values = source_values.take(boundary_rows)
            boundary_owned = source_owned.take(boundary_rows)
            handled_local_mask = np.zeros(boundary_rows.size, dtype=bool)
            tags = np.asarray(boundary_owned.tags)

            from vibespatial.geometry.buffers import GeometryFamily
            from vibespatial.geometry.owned import FAMILY_TAGS

            multipolygon_local_rows = np.flatnonzero(
                tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
            ).astype(np.intp, copy=False)
            if (
                boundary_owned.residency is Residency.DEVICE
                and multipolygon_local_rows.size > 0
            ):
                multipolygon_rescue = _clip_multipolygon_rectangle_keep_geom_type_owned(
                    boundary_owned.take(multipolygon_local_rows),
                    rectangle_bounds,
                )
                if multipolygon_rescue is None:
                    logger.debug(
                        "device multipolygon rectangle keep_geom_type rescue unavailable; "
                        "continuing to generic area path"
                    )
                else:
                    rescued_local_rows = np.flatnonzero(
                        _owned_nonempty_polygon_mask(multipolygon_rescue)
                    ).astype(np.intp, copy=False)
                    if rescued_local_rows.size > 0:
                        result_owned = concat_owned_scatter(
                            result_owned,
                            multipolygon_rescue.take(rescued_local_rows),
                            boundary_rows[multipolygon_local_rows[rescued_local_rows]],
                        )
                    handled_local_mask[multipolygon_local_rows] = True

            remaining_local_rows = np.flatnonzero(~handled_local_mask).astype(
                np.intp,
                copy=False,
            )
            missing_local_rows = remaining_local_rows
            if remaining_local_rows.size > 0:
                area_owned = _clip_polygon_rectangle_area_intersection_owned(
                    boundary_owned.take(remaining_local_rows),
                    rectangle_bounds,
                )
                area_positive_local = _owned_nonempty_polygon_mask(area_owned)
                positive_local_rows = np.flatnonzero(area_positive_local).astype(
                    np.intp,
                    copy=False,
                )
                if positive_local_rows.size > 0:
                    result_owned = concat_owned_scatter(
                        result_owned,
                        area_owned.take(positive_local_rows),
                        boundary_rows[remaining_local_rows[positive_local_rows]],
                    )
                missing_local_rows = remaining_local_rows[
                    np.flatnonzero(~area_positive_local).astype(
                        np.intp,
                        copy=False,
                    )
                ]
            if missing_local_rows.size > 0:
                multipolygon_local_rows = missing_local_rows[
                    tags[missing_local_rows] == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
                ]
                if multipolygon_local_rows.size > 0:
                    multipolygon_rescue = _clip_multipolygon_rectangle_keep_geom_type_owned(
                        boundary_owned.take(multipolygon_local_rows),
                        rectangle_bounds,
                    )
                    if multipolygon_rescue is not None:
                        rescued_local_rows = np.flatnonzero(
                            _owned_nonempty_polygon_mask(multipolygon_rescue)
                        ).astype(np.intp, copy=False)
                        if rescued_local_rows.size > 0:
                            result_owned = concat_owned_scatter(
                                result_owned,
                                multipolygon_rescue.take(rescued_local_rows),
                                boundary_rows[multipolygon_local_rows[rescued_local_rows]],
                            )
                            rescued_mask = np.zeros(
                                missing_local_rows.size,
                                dtype=bool,
                            )
                            rescued_mask[np.isin(missing_local_rows, multipolygon_local_rows[rescued_local_rows])] = True
                            missing_local_rows = missing_local_rows[~rescued_mask]

                if missing_local_rows.size > 0:
                    from vibespatial.api.tools.overlay import (
                        _strip_non_polygon_collection_parts,
                    )

                    repeated_mask = np.empty(missing_local_rows.size, dtype=object)
                    repeated_mask[:] = rectangle_mask
                    recovered = _strip_non_polygon_collection_parts(
                        _exact_polygon_clip_boundary_rows(
                            _take_geometry_object_values(
                                boundary_values,
                                missing_local_rows,
                            ),
                            repeated_mask,
                        )
                    )
                    recover_keep = ~(
                        shapely.is_missing(recovered)
                        | shapely.is_empty(recovered)
                    )
                    if np.any(recover_keep):
                        _record_clip_host_cleanup_fallback(
                            detail=(
                                "rectangle keep_geom_type polygon cleanup recovered "
                                "polygonal collection parts on the host"
                            ),
                            pipeline="clip.keep_geom_type",
                        )
                        replacement_owned = from_shapely_geometries(
                            recovered[recover_keep].tolist(),
                            residency=result_owned.residency,
                        )
                        result_owned = concat_owned_scatter(
                            result_owned,
                            replacement_owned,
                            boundary_rows[missing_local_rows[recover_keep]],
                        )
        return _geometry_values_from_owned(result_owned, crs=partition.crs)

    if boundary_rows.size > 0:
        boundary_index = partition.index.take(boundary_rows)
        boundary_values = source_values.take(boundary_rows)
        boundary_bounds = source_bounds[boundary_rows]
        if source_owned is not None:
            rectangle_boundary_owned = _exact_rectangle_clip_boundary_owned_rows(
                boundary_values,
                boundary_bounds,
                rectangle_bounds,
            )
            if rectangle_boundary_owned is not None:
                result_owned = build_null_owned_array(
                    len(partition),
                    residency=source_owned.residency,
                )
                if inside_rows.size > 0:
                    result_owned = concat_owned_scatter(
                        result_owned,
                        source_owned.take(inside_rows),
                        inside_rows,
                    )
                result_owned = concat_owned_scatter(
                    result_owned,
                    rectangle_boundary_owned,
                    boundary_rows,
                )
                _seed_rectangle_clip_validity_cache_if_safe(
                    result_owned,
                    source_values,
                )
                result_values = (
                    DeviceGeometryArray._from_owned(result_owned, crs=partition.crs)
                    if result_owned.residency is Residency.DEVICE
                    else GeometryArray.from_owned(result_owned, crs=partition.crs)
                )
                return result_values
        rectangle_boundary = _exact_rectangle_clip_boundary_rows(
            boundary_values,
            boundary_bounds,
            rectangle_bounds,
        )
        if rectangle_boundary is not None:
            if source_owned is not None:
                result_owned = build_null_owned_array(
                    len(partition),
                    residency=source_owned.residency,
                )
                if inside_rows.size > 0:
                    result_owned = concat_owned_scatter(
                        result_owned,
                        source_owned.take(inside_rows),
                        inside_rows,
                    )
                replacement_owned = from_shapely_geometries(
                    rectangle_boundary.tolist(),
                    residency=result_owned.residency,
                )
                result_owned = concat_owned_scatter(
                    result_owned,
                    replacement_owned,
                    boundary_rows,
                )
                _seed_rectangle_clip_validity_cache_if_safe(
                    result_owned,
                    source_values,
                )
                result_values = (
                    DeviceGeometryArray._from_owned(result_owned, crs=partition.crs)
                    if result_owned.residency is Residency.DEVICE
                    else GeometryArray.from_owned(result_owned, crs=partition.crs)
                )
                return result_values

            assembled[boundary_rows] = rectangle_boundary
            return _as_geometry_values(assembled, crs=partition.crs)

        left_pairs = GeoSeries(
            boundary_values,
            index=boundary_index,
            crs=partition.crs,
        )
        boundary_owned = boundary_values.to_owned()
        area_owned = _clip_polygon_rectangle_area_intersection_owned(
            boundary_owned,
            rectangle_bounds,
        )
        area_pairs = GeoSeries(
            _geometry_values_from_owned(area_owned, crs=partition.crs),
            index=boundary_index,
            crs=partition.crs,
        )
        repeated_mask = np.empty(boundary_rows.size, dtype=object)
        repeated_mask[:] = rectangle_mask
        right_pairs = GeoSeries(
            repeated_mask,
            index=boundary_index,
            crs=partition.crs,
        )
        assembled[boundary_rows] = np.asarray(
            _assemble_polygon_intersection_rows_with_lower_dim(
                left_pairs,
                right_pairs,
                area_pairs,
            ),
            dtype=object,
        )
        changed_boundary_geoms = assembled[boundary_rows]
        contains_collection = any(
            geom is not None and getattr(geom, "geom_type", None) == "GeometryCollection"
            for geom in changed_boundary_geoms
        )
        if (
            not contains_collection
            and source_owned is not None
        ):
            result_owned = concat_owned_scatter(
                source_owned,
                area_owned,
                boundary_rows,
            )
            area_objects = np.asarray(area_pairs, dtype=object)
            changed_mask = np.ones(boundary_rows.size, dtype=bool)
            for row_index, (assembled_geom, area_geom) in enumerate(
                zip(assembled[boundary_rows], area_objects, strict=True)
            ):
                if assembled_geom is None and area_geom is None:
                    changed_mask[row_index] = False
                    continue
                if assembled_geom is None or area_geom is None:
                    continue
                if bool(shapely.equals(assembled_geom, area_geom)):
                    changed_mask[row_index] = False
            changed_rows = boundary_rows[changed_mask]
            if changed_rows.size > 0:
                replacement_owned = from_shapely_geometries(
                    assembled[changed_rows].tolist(),
                    residency=result_owned.residency,
                )
                result_owned = concat_owned_scatter(
                    result_owned,
                    replacement_owned,
                    changed_rows,
                )

            result_values = (
                DeviceGeometryArray._from_owned(result_owned, crs=partition.crs)
                if result_owned.residency is Residency.DEVICE
                else GeometryArray.from_owned(result_owned, crs=partition.crs)
            )
            return result_values

    if source_owned is not None and boundary_rows.size == 0:
        result_owned = build_null_owned_array(
            len(partition),
            residency=source_owned.residency,
        )
        if inside_rows.size > 0:
            result_owned = concat_owned_scatter(
                result_owned,
                source_owned.take(inside_rows),
                inside_rows,
            )
        _seed_rectangle_clip_validity_cache_if_safe(result_owned, source_values)
        return _geometry_values_from_owned(result_owned, crs=partition.crs)

    if source_owned is not None and inside_rows.size > 0:
        assembled[inside_rows] = np.asarray(
            source_owned.take(inside_rows).to_shapely(),
            dtype=object,
        )
    return _as_geometry_values(assembled, crs=partition.crs)


def _clip_polygon_partition_with_polygon_mask(
    partition,
    mask,
    *,
    keep_geom_type_only: bool = False,
):
    """Clip polygon rows to a polygon mask while preserving owned backing.

    The bulk polygon area result stays on the owned/device path. Only rows
    without positive-area output pay the boundary reconstruction cost needed
    to preserve lower-dimensional public clip semantics.
    """
    from vibespatial.geometry.owned import (
        build_null_owned_array,
        concat_owned_scatter,
        from_shapely_geometries,
    )

    partition = _promote_geometry_backing_to_device(
        partition,
        reason="clip polygon-mask partition selected GPU-native constructive path",
    )
    source_values = partition.geometry.values if isinstance(partition, GeoDataFrame) else partition.values
    source_missing = np.asarray(source_values.isna() | source_values.is_empty, dtype=bool)
    allow_rectangle_kernel = _polygon_mask_allows_rectangle_kernel(mask)
    strict_native = strict_native_mode_enabled()

    left_owned = source_values.to_owned()
    mask_owned = from_shapely_geometries([mask], residency=left_owned.residency)

    scalar_bypass_owned = _clip_polygon_single_pair_containment_owned(
        left_owned,
        mask_owned,
    )
    if scalar_bypass_owned is not None:
        return _geometry_values_from_owned(scalar_bypass_owned, crs=partition.crs)

    # Device-backed polygon clip benefits from a cheap exact predicate refine
    # before exact intersection: bbox candidates include both false positives
    # and polygons already fully covered by the mask, and paying full exact
    # intersection for those rows dominates 10K-scale workflows.
    if has_gpu_runtime() and left_owned.residency is Residency.DEVICE:
        from vibespatial.geometry.owned import materialize_broadcast, tile_single_row
        from vibespatial.predicates.binary import (
            _evaluate_binary_predicates_fused_gpu,
            _evaluate_covered_by_single_polygonal_mask_gpu,
            evaluate_binary_predicate,
        )
        from vibespatial.runtime import ExecutionMode

        def _take_owned_rows(owned, rows: np.ndarray):
            if rows.size == 0:
                return owned.take(rows)
            if (
                rows.size == owned.row_count
                and np.array_equal(rows, np.arange(owned.row_count, dtype=rows.dtype))
            ):
                return owned
            if owned.residency is Residency.DEVICE:
                import cupy as cp

                return owned.device_take(cp.asarray(rows, dtype=cp.int64))
            return owned.take(rows)

        def _all_candidate_bounds_within_single_mask_bounds() -> bool:
            import cupy as cp

            from vibespatial.kernels.core.geometry_analysis import (
                compute_geometry_bounds_device,
            )

            left_bounds = cp.asarray(
                compute_geometry_bounds_device(left_owned),
                dtype=cp.float64,
            ).reshape(left_owned.row_count, 4)
            mask_bounds = cp.asarray(
                compute_geometry_bounds_device(mask_owned),
                dtype=cp.float64,
            ).reshape(1, 4)[0]
            left_valid = cp.asarray(left_owned._ensure_device_state().validity)
            within_bounds = (
                (left_bounds[:, 0] >= mask_bounds[0])
                & (left_bounds[:, 2] <= mask_bounds[2])
                & (left_bounds[:, 1] >= mask_bounds[1])
                & (left_bounds[:, 3] <= mask_bounds[3])
            )
            return bool(cp.all(within_bounds | ~left_valid))

        inside_rows = np.asarray([], dtype=np.intp)
        exact_rows = np.asarray([], dtype=np.intp)
        if left_owned.row_count == 1 and not source_missing[0] and not strict_native:
            exact_rows = np.asarray([0], dtype=np.intp)
        else:
            single_mask_covered_by = (
                _evaluate_covered_by_single_polygonal_mask_gpu(left_owned, mask_owned)
                if _all_candidate_bounds_within_single_mask_bounds()
                else None
            )
            if single_mask_covered_by is not None:
                covered_mask = np.asarray(single_mask_covered_by, dtype=bool)
                valid_source_mask = ~source_missing
                inside_rows = np.flatnonzero(valid_source_mask & covered_mask).astype(
                    np.intp,
                    copy=False,
                )
                remaining_rows = np.flatnonzero(valid_source_mask & ~covered_mask).astype(
                    np.intp,
                    copy=False,
                )
                if remaining_rows.size > 0:
                    remaining_owned = _take_owned_rows(left_owned, remaining_rows)
                    broadcast_remaining_mask = (
                        mask_owned
                        if remaining_owned.row_count == 1
                        else materialize_broadcast(
                            tile_single_row(mask_owned, remaining_owned.row_count),
                        )
                    )
                    remaining_intersects = np.asarray(
                        evaluate_binary_predicate(
                            "intersects",
                            remaining_owned,
                            broadcast_remaining_mask,
                            dispatch_mode=ExecutionMode.GPU,
                        ).values,
                        dtype=bool,
                    )
                    exact_rows = remaining_rows[remaining_intersects]
            else:
                broadcast_all_mask = (
                    mask_owned
                    if left_owned.row_count == 1
                    else materialize_broadcast(
                        tile_single_row(mask_owned, left_owned.row_count),
                    )
                )
                fused_masks = _evaluate_binary_predicates_fused_gpu(
                    ("intersects", "covered_by"),
                    left_owned,
                    broadcast_all_mask,
                )
                if fused_masks is not None:
                    intersects_mask = np.asarray(fused_masks["intersects"], dtype=bool)
                    inside_mask = np.asarray(fused_masks["covered_by"], dtype=bool)
                    hit_mask = ~source_missing & intersects_mask
                    inside_rows = np.flatnonzero(hit_mask & inside_mask).astype(
                        np.intp,
                        copy=False,
                    )
                    exact_rows = np.flatnonzero(hit_mask & ~inside_mask).astype(
                        np.intp,
                        copy=False,
                    )
                else:
                    intersects_mask = np.asarray(
                        evaluate_binary_predicate(
                            "intersects",
                            left_owned,
                            mask,
                            dispatch_mode=ExecutionMode.GPU,
                        ).values,
                        dtype=bool,
                    )
                    hit_rows = np.flatnonzero(~source_missing & intersects_mask).astype(
                        np.intp,
                        copy=False,
                    )
                    if hit_rows.size > 0:
                        hit_owned = _take_owned_rows(left_owned, hit_rows)
                        broadcast_mask = (
                            mask_owned
                            if hit_owned.row_count == 1
                            else materialize_broadcast(
                                tile_single_row(mask_owned, hit_owned.row_count),
                            )
                        )
                        inside_hits = np.asarray(
                            evaluate_binary_predicate(
                                "covered_by",
                                hit_owned,
                                broadcast_mask,
                                dispatch_mode=ExecutionMode.GPU,
                            ).values,
                            dtype=bool,
                        )
                        inside_rows = hit_rows[inside_hits]
                        exact_rows = hit_rows[~inside_hits]

        exact_area_owned = None
        exact_area_values = None
        exact_positive_local = np.asarray([], dtype=bool)

        def _ensure_exact_area_values():
            nonlocal exact_area_values
            if exact_area_values is None and exact_area_owned is not None:
                exact_area_values = _geometry_values_from_owned(
                    exact_area_owned,
                    crs=partition.crs,
                )
            return exact_area_values

        def _exact_area_objects_for_rows(local_rows: np.ndarray) -> np.ndarray:
            if local_rows.size == 0:
                return np.asarray([], dtype=object)
            if exact_area_owned is not None:
                area_subset = _take_owned_rows(exact_area_owned, local_rows)
                return np.asarray(
                    _geometry_values_from_owned(area_subset, crs=partition.crs),
                    dtype=object,
                )
            values = _ensure_exact_area_values()
            if values is None:
                return np.asarray([], dtype=object)
            return np.asarray(values.take(local_rows), dtype=object)

        if exact_rows.size > 0:
            exact_area_owned = _clip_polygon_area_intersection_owned(
                _take_owned_rows(left_owned, exact_rows),
                mask_owned,
                allow_rectangle_kernel=allow_rectangle_kernel,
                prefer_exact_polygon_rect_batch=False,
                prefer_many_vs_one_planner=(
                    keep_geom_type_only
                    and strict_native
                    and not allow_rectangle_kernel
                ),
            )
            exact_positive_local = _owned_nonempty_polygon_mask(exact_area_owned)
            if not allow_rectangle_kernel:
                exact_area_mutated = False
                positive_local_rows = np.flatnonzero(exact_positive_local).astype(
                    np.intp,
                    copy=False,
                )
                if positive_local_rows.size > 0:
                    from vibespatial.constructive.measurement import area_owned
                    from vibespatial.geometry.device_array import _compute_bounds_from_owned

                    positive_bounds = np.asarray(
                        _compute_bounds_from_owned(exact_area_owned),
                        dtype=np.float64,
                    ).reshape(-1, 4)[positive_local_rows]
                    positive_area = np.asarray(
                        area_owned(exact_area_owned),
                        dtype=np.float64,
                    )[positive_local_rows]
                    collapsed_positive = (
                        (np.abs(positive_bounds[:, 2] - positive_bounds[:, 0]) <= SPATIAL_EPSILON)
                        | (np.abs(positive_bounds[:, 3] - positive_bounds[:, 1]) <= SPATIAL_EPSILON)
                        | (positive_area <= 0.0)
                    )
                    if collapsed_positive.any():
                        exact_positive_local = exact_positive_local.copy()
                        exact_positive_local[positive_local_rows[collapsed_positive]] = False
                boundary_local_rows = np.flatnonzero(~exact_positive_local).astype(np.intp, copy=False)
                exact_area_objects = None
                if keep_geom_type_only or exact_rows.size <= 128:
                    exact_area_objects = np.asarray(
                        _ensure_exact_area_values(),
                        dtype=object,
                    )
                    triangleish_local_rows = np.asarray(
                        [
                            row_index
                            for row_index, geom in enumerate(exact_area_objects)
                            if (
                                geom is not None
                                and getattr(geom, "geom_type", None) == "Polygon"
                                and not getattr(geom, "is_empty", False)
                                and len(geom.exterior.coords) <= 4
                            )
                        ],
                        dtype=np.intp,
                    )
                else:
                    triangleish_local_rows = np.asarray([], dtype=np.intp)
                if keep_geom_type_only:
                    correction_local_rows = np.union1d(
                        boundary_local_rows,
                        triangleish_local_rows,
                    ).astype(np.intp, copy=False)
                elif exact_rows.size <= 128:
                    correction_local_rows = np.arange(exact_rows.size, dtype=np.intp)
                else:
                    correction_local_rows = np.union1d(
                        boundary_local_rows,
                        triangleish_local_rows,
                    ).astype(np.intp, copy=False)
                if correction_local_rows.size > 0:
                    correction_rows = exact_rows[correction_local_rows]
                    left_pairs = _take_geometry_object_values(
                        source_values,
                        correction_rows,
                    )
                    repeated_mask = np.empty(correction_rows.size, dtype=object)
                    repeated_mask[:] = mask
                    assembled_exact = _exact_polygon_clip_boundary_rows(
                        left_pairs,
                        repeated_mask,
                    )
                    area_subset_objects = (
                        exact_area_objects[correction_local_rows]
                        if exact_area_objects is not None
                        else _exact_area_objects_for_rows(correction_local_rows)
                    )
                    changed_mask = np.ones(correction_local_rows.size, dtype=bool)
                    for row_index, (assembled_geom, area_geom) in enumerate(
                        zip(assembled_exact, area_subset_objects, strict=True)
                    ):
                        if _clip_boundary_row_matches_area(assembled_geom, area_geom):
                            changed_mask[row_index] = False
                    changed_local_rows = correction_local_rows[changed_mask]
                    if changed_local_rows.size > 0:
                        changed_geoms = assembled_exact[changed_mask]
                        contains_collection = any(
                            geom is not None and getattr(geom, "geom_type", None) == "GeometryCollection"
                            for geom in changed_geoms
                        )
                        if contains_collection:
                            if keep_geom_type_only:
                                from vibespatial.api.tools.overlay import (
                                    _strip_non_polygon_collection_parts,
                                )

                                replacement_owned = from_shapely_geometries(
                                    _strip_non_polygon_collection_parts(
                                        changed_geoms,
                                    ).tolist(),
                                    residency=exact_area_owned.residency,
                                )
                                exact_area_owned = concat_owned_scatter(
                                    exact_area_owned,
                                    replacement_owned,
                                    changed_local_rows,
                                )
                                exact_area_values = _geometry_values_from_owned(
                                    exact_area_owned,
                                    crs=partition.crs,
                                )
                                exact_area_mutated = True
                            else:
                                if exact_area_objects is None:
                                    exact_area_objects = np.asarray(
                                        _ensure_exact_area_values(),
                                        dtype=object,
                                    )
                                corrected_exact = exact_area_objects.copy()
                                corrected_exact[correction_local_rows] = assembled_exact
                                exact_area_owned = None
                                exact_area_values = _as_geometry_values(
                                    corrected_exact,
                                    crs=partition.crs,
                                )
                                exact_area_mutated = True
                        else:
                            replacement_owned = from_shapely_geometries(
                                changed_geoms.tolist(),
                                residency=exact_area_owned.residency,
                            )
                            exact_area_owned = concat_owned_scatter(
                                exact_area_owned,
                                replacement_owned,
                                changed_local_rows,
                            )
                            exact_area_values = None
                            exact_area_mutated = True
                if exact_area_owned is not None:
                    if exact_area_mutated:
                        exact_positive_local = _owned_nonempty_polygon_mask(exact_area_owned)
                else:
                    exact_objects = np.asarray(
                        _ensure_exact_area_values(),
                        dtype=object,
                    )
                    exact_positive_local = np.asarray(
                        [getattr(geom, "geom_type", None) in POLYGON_GEOM_TYPES for geom in exact_objects],
                        dtype=bool,
                    ) & (
                        np.asarray(shapely.area(exact_objects), dtype=np.float64) > 0.0
                    )

        positive_local_rows = np.flatnonzero(exact_positive_local).astype(
            np.intp,
            copy=False,
        )
        positive_rows = exact_rows[positive_local_rows]
        if keep_geom_type_only:
            if exact_area_owned is None:
                assembled = np.empty(len(partition), dtype=object)
                assembled[:] = None
                if inside_rows.size > 0:
                    assembled[inside_rows] = np.asarray(
                        source_values.take(inside_rows),
                        dtype=object,
                    )
                if positive_local_rows.size > 0:
                    values = _ensure_exact_area_values()
                    if values is None:
                        return _as_geometry_values(assembled, crs=partition.crs)
                    assembled[positive_rows] = np.asarray(
                        values.take(positive_local_rows),
                        dtype=object,
                    )
                return _as_geometry_values(assembled, crs=partition.crs)
            sparse_result = _build_sparse_owned_clip_output(
                partition_crs=partition.crs,
                left_owned=left_owned,
                inside_rows=inside_rows,
                exact_area_owned=exact_area_owned,
                positive_local_rows=positive_local_rows,
                positive_rows=positive_rows,
            )
            if sparse_result.local_rows.size == len(partition):
                return sparse_result.geometry_values
            return sparse_result
        boundary_local_rows = np.flatnonzero(~exact_positive_local).astype(
            np.intp,
            copy=False,
        )
        boundary_rows = exact_rows[boundary_local_rows]

        if boundary_rows.size == 0:
            if exact_area_owned is None:
                assembled = np.empty(len(partition), dtype=object)
                assembled[:] = None
                if inside_rows.size > 0:
                    assembled[inside_rows] = np.asarray(
                        source_values.take(inside_rows),
                        dtype=object,
                    )
                if positive_local_rows.size > 0:
                    values = _ensure_exact_area_values()
                    if values is None:
                        return _as_geometry_values(assembled, crs=partition.crs)
                    assembled[positive_rows] = np.asarray(
                        values.take(positive_local_rows),
                        dtype=object,
                    )
                return _as_geometry_values(assembled, crs=partition.crs)
            sparse_result = _build_sparse_owned_clip_output(
                partition_crs=partition.crs,
                left_owned=left_owned,
                inside_rows=inside_rows,
                exact_area_owned=exact_area_owned,
                positive_local_rows=positive_local_rows,
                positive_rows=positive_rows,
            )
            if sparse_result.local_rows.size == len(partition) and not source_missing.any():
                return sparse_result.geometry_values
            return sparse_result

        left_pairs = _take_geometry_object_values(source_values, boundary_rows)
        area_objects = _exact_area_objects_for_rows(boundary_local_rows)
        repeated_mask = np.empty(boundary_rows.size, dtype=object)
        repeated_mask[:] = mask
        assembled_boundary = _exact_polygon_clip_boundary_rows(
            left_pairs,
            repeated_mask,
        )

        contains_collection = any(
            geom is not None and getattr(geom, "geom_type", None) == "GeometryCollection"
            for geom in assembled_boundary
        )
        if exact_area_owned is None:
            assembled = np.empty(len(partition), dtype=object)
            assembled[:] = None
            if inside_rows.size > 0:
                assembled[inside_rows] = np.asarray(
                    source_values.take(inside_rows),
                    dtype=object,
                )
            if positive_local_rows.size > 0:
                values = _ensure_exact_area_values()
                if values is None:
                    return _as_geometry_values(assembled, crs=partition.crs)
                assembled[positive_rows] = np.asarray(
                    values.take(positive_local_rows),
                    dtype=object,
                )
            assembled[boundary_rows] = assembled_boundary
            return _as_geometry_values(assembled, crs=partition.crs)
        if not contains_collection:
            preserve_mask = np.zeros(boundary_rows.size, dtype=bool)
            changed_mask = np.zeros(boundary_rows.size, dtype=bool)
            for row_index, (assembled_geom, area_geom) in enumerate(
                zip(assembled_boundary, area_objects, strict=True)
            ):
                if _clip_boundary_row_matches_area(assembled_geom, area_geom):
                    preserve_mask[row_index] = (
                        area_geom is not None
                        and not getattr(area_geom, "is_empty", False)
                    )
                else:
                    changed_mask[row_index] = True
            preserved_local_rows = boundary_local_rows[preserve_mask]
            preserved_rows = boundary_rows[preserve_mask]
            extra_parts = []
            if preserved_local_rows.size > 0 and exact_area_owned is not None:
                extra_parts.append((
                    preserved_rows,
                    _take_owned_rows(exact_area_owned, preserved_local_rows),
                ))
            changed_rows = boundary_rows[changed_mask]
            if changed_rows.size > 0:
                replacement_owned = from_shapely_geometries(
                    assembled_boundary[changed_mask].tolist(),
                    residency=left_owned.residency,
                )
                extra_parts.append((changed_rows, replacement_owned))

            return _build_sparse_owned_clip_output(
                partition_crs=partition.crs,
                left_owned=left_owned,
                inside_rows=inside_rows,
                exact_area_owned=exact_area_owned,
                positive_local_rows=positive_local_rows,
                positive_rows=positive_rows,
                extra_owned_parts=tuple(extra_parts),
            )

        assembled = np.empty(len(partition), dtype=object)
        assembled[:] = None
        if inside_rows.size > 0:
            assembled[inside_rows] = np.asarray(
                source_values.take(inside_rows),
                dtype=object,
            )
        if positive_local_rows.size > 0 and exact_area_values is not None:
            assembled[positive_rows] = np.asarray(
                exact_area_values.take(positive_local_rows),
                dtype=object,
            )
        assembled[boundary_rows] = assembled_boundary
        return _as_geometry_values(assembled, crs=partition.crs)

    area_owned = _clip_polygon_area_intersection_owned(
        left_owned,
        mask_owned,
        allow_rectangle_kernel=allow_rectangle_kernel,
    )

    area_values = _geometry_values_from_owned(area_owned, crs=partition.crs)
    area_positive = ~source_missing & _owned_nonempty_polygon_mask(area_owned)
    if keep_geom_type_only:
        positive_rows = np.flatnonzero(area_positive).astype(np.intp, copy=False)
        if positive_rows.size == 0:
            return _geometry_values_from_owned(
                build_null_owned_array(
                    len(partition),
                    residency=left_owned.residency,
                ),
                crs=partition.crs,
            )
        result_owned = concat_owned_scatter(
            build_null_owned_array(
                len(partition),
                residency=left_owned.residency,
            ),
            area_owned.take(positive_rows),
            positive_rows,
        )
        return _geometry_values_from_owned(result_owned, crs=partition.crs)
    positive_rows = np.flatnonzero(area_positive).astype(np.int64, copy=False)

    boundary_rows = np.flatnonzero(~source_missing & ~area_positive).astype(np.intp, copy=False)
    if boundary_rows.size == 0:
        return area_values

    touch_boundary_mask = _clip_polygon_boundary_touch_mask(
        source_values,
        left_owned,
        boundary_rows,
        mask=mask,
        mask_owned=mask_owned,
    )
    boundary_rows = boundary_rows[touch_boundary_mask]
    if boundary_rows.size == 0:
        if positive_rows.size == 0:
            return _geometry_values_from_owned(
                build_null_owned_array(
                    len(partition),
                    residency=left_owned.residency,
                ),
                crs=partition.crs,
            )
        sparse_result = _build_sparse_owned_clip_output(
            partition_crs=partition.crs,
            left_owned=left_owned,
            inside_rows=np.empty(0, dtype=np.intp),
            exact_area_owned=area_owned,
            positive_local_rows=positive_rows.astype(np.intp, copy=False),
            positive_rows=positive_rows.astype(np.intp, copy=False),
        )
        if sparse_result.local_rows.size == len(partition) and not source_missing.any():
            return sparse_result.geometry_values
        return sparse_result

    left_pairs = _take_geometry_object_values(source_values, boundary_rows)
    area_objects = _take_geometry_object_values(area_values, boundary_rows)
    repeated_mask = np.empty(boundary_rows.size, dtype=object)
    repeated_mask[:] = mask
    assembled_boundary = _exact_polygon_clip_boundary_rows(
        left_pairs,
        repeated_mask,
    )

    contains_collection = any(
        geom is not None and getattr(geom, "geom_type", None) == "GeometryCollection"
        for geom in assembled_boundary
    )
    if not contains_collection:
        result_owned = build_null_owned_array(
            len(partition),
            residency=left_owned.residency,
        )
        if positive_rows.size > 0:
            result_owned = concat_owned_scatter(
                result_owned,
                area_owned.take(positive_rows),
                positive_rows,
            )

        preserve_mask = np.zeros(boundary_rows.size, dtype=bool)
        changed_mask = np.zeros(boundary_rows.size, dtype=bool)
        for row_index, (assembled_geom, area_geom) in enumerate(
            zip(assembled_boundary, area_objects, strict=True)
        ):
            if _clip_boundary_row_matches_area(assembled_geom, area_geom):
                preserve_mask[row_index] = (
                    area_geom is not None
                    and not getattr(area_geom, "is_empty", False)
                )
            else:
                changed_mask[row_index] = True
        preserved_rows = boundary_rows[preserve_mask]
        if preserved_rows.size > 0:
            result_owned = concat_owned_scatter(
                result_owned,
                area_owned.take(preserved_rows),
                preserved_rows,
            )
        changed_rows = boundary_rows[changed_mask]
        if changed_rows.size > 0:
            replacement_owned = from_shapely_geometries(
                assembled_boundary[changed_mask].tolist(),
                residency=result_owned.residency,
            )
            result_owned = concat_owned_scatter(
                result_owned,
                replacement_owned,
                changed_rows,
            )

        return _geometry_values_from_owned(result_owned, crs=partition.crs)

    assembled = np.empty(len(partition), dtype=object)
    assembled[:] = None
    positive_rows = np.flatnonzero(area_positive).astype(np.intp, copy=False)
    if positive_rows.size > 0:
        assembled[positive_rows] = np.asarray(area_values.take(positive_rows), dtype=object)
    assembled[boundary_rows] = assembled_boundary
    return _as_geometry_values(assembled, crs=partition.crs)


def _clip_complex_polygon_partition_with_rectangle_mask(
    partition,
    rectangle_bounds: tuple[float, float, float, float],
):
    """Preserve area and lower-dimensional remnants for complex rectangle clip rows."""
    rectangle_mask = box(*rectangle_bounds)
    area_values = _clip_polygon_partition_with_polygon_mask(partition, rectangle_mask)
    source_values = partition.geometry.values if isinstance(partition, GeoDataFrame) else partition.values
    area_objects = np.asarray(area_values, dtype=object)
    source_objects = np.asarray(source_values, dtype=object)
    rectangle_boundary = shapely.boundary(rectangle_mask)
    boundary_objects = np.asarray(
        shapely.intersection(shapely.boundary(source_objects), rectangle_boundary),
        dtype=object,
    )
    combined = np.empty(len(partition), dtype=object)

    for row_index, (area_geom, boundary_geom) in enumerate(
        zip(area_objects, boundary_objects, strict=True)
    ):
        if area_geom is None or getattr(area_geom, "is_empty", False):
            combined[row_index] = boundary_geom
            continue
        if boundary_geom is None or getattr(boundary_geom, "is_empty", False):
            combined[row_index] = area_geom
            continue
        combined[row_index] = GeometryCollection([area_geom, boundary_geom])

    return _as_geometry_values(combined, crs=partition.crs)


def _build_clip_partition_result(source, row_positions, geometry_values):
    """Create a native row-preserving clip fragment without frame assembly."""
    if isinstance(geometry_values, _ClipPartitionOutput):
        local_rows = np.asarray(geometry_values.local_rows, dtype=np.intp)
        row_positions = np.asarray(row_positions, dtype=np.intp)[local_rows]
        geometry_values = geometry_values.geometry_values
    return _clip_native_part(
        source,
        row_positions,
        _as_geometry_values(
            geometry_values,
            crs=source.crs,
        ),
    )


def _clip_point_partition_with_polygon_mask(partition, mask):
    """Filter point candidates by exact polygon intersection without re-clipping."""
    geometry = partition.geometry if isinstance(partition, GeoDataFrame) else partition
    values = geometry.values
    keep_rows = np.flatnonzero(
        np.asarray(geometry.intersects(mask), dtype=bool)
    ).astype(np.intp, copy=False)
    return _ClipPartitionOutput(
        geometry_values=values.take(keep_rows),
        local_rows=keep_rows,
    )


def _clip_homogeneous_polygon_candidates_native(
    source,
    mask,
    candidate_rows: np.ndarray,
    *,
    clipping_by_rectangle: bool,
    rectangle_bounds,
    keep_geom_type: bool,
) -> NativeTabularResult | None:
    """Clip polygon-only candidate rows without building a candidate frame."""
    if (
        clipping_by_rectangle
        or rectangle_bounds is not None
        or not isinstance(mask, Polygon | MultiPolygon)
        or candidate_rows.size == 0
        or not has_gpu_runtime()
    ):
        return None

    geometry = source.geometry if isinstance(source, GeoDataFrame) else source
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None or owned.residency is not Residency.DEVICE:
        return None

    from vibespatial.geometry.buffers import GeometryFamily

    if not set(owned.families).issubset(
        {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    ):
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
        return None

    rows = np.asarray(candidate_rows, dtype=np.intp)
    if (
        rows.size == owned.row_count
        and np.array_equal(rows, np.arange(owned.row_count, dtype=rows.dtype))
    ):
        candidate_owned = owned
    else:
        candidate_owned = owned.device_take(cp.asarray(rows, dtype=cp.int64))

    candidate_values = DeviceGeometryArray._from_owned(candidate_owned, crs=geometry.crs)
    partition = _geometry_series_from_values(
        candidate_values,
        index=geometry.index.take(rows),
        crs=geometry.crs,
        name=getattr(geometry, "name", None),
    )
    geometry_values = _clip_polygon_partition_with_polygon_mask(
        partition,
        mask,
        keep_geom_type_only=keep_geom_type,
    )
    parts_tuple = (
        _build_clip_partition_result(
            source,
            rows,
            geometry_values,
        ),
    )
    ordered_rows = rows.astype(np.intp, copy=False)
    return _clip_constructive_parts_to_native_tabular_result(
        source=source,
        parts=parts_tuple,
        ordered_row_positions=ordered_rows,
        clipping_by_rectangle=False,
        has_non_point_candidates=True,
        keep_geom_type=keep_geom_type,
        spatial_materializer=lambda: ClipNativeResult(
            source=source,
            parts=parts_tuple,
            ordered_index=geometry.index.take(ordered_rows),
            ordered_row_positions=ordered_rows,
            clipping_by_rectangle=False,
            has_non_point_candidates=True,
            keep_geom_type=keep_geom_type,
        ).to_spatial(),
    )


def _clip_gdf_with_mask_native(
    gdf,
    mask,
    sort=False,
    *,
    query_geometry=None,
    keep_geom_type: bool = False,
) -> NativeTabularResult:
    """Build a native clip result and defer GeoPandas assembly to explicit export."""
    clipping_by_rectangle = _mask_is_list_like_rectangle(mask)
    rectangle_bounds = _rectangle_bounds_from_mask(mask)
    if clipping_by_rectangle:
        intersection_polygon = box(*mask)
    else:
        intersection_polygon = mask

    candidate_query_geometry = (
        query_geometry
        if isinstance(query_geometry, GeoDataFrame | GeoSeries | Polygon | MultiPolygon)
        else intersection_polygon
    )
    candidate_rows = _bbox_candidate_rows_for_scalar_clip_mask(
        gdf,
        candidate_query_geometry,
        sort=sort,
    )
    candidate_rows_from_scalar_bbox = candidate_rows is not None
    if candidate_rows is None:
        query_input = query_geometry if query_geometry is not None else intersection_polygon
        candidate_rows = np.asarray(
            gdf.sindex.query(query_input, predicate="intersects", sort=sort),
            dtype=np.int32,
        )
    if candidate_rows.ndim == 2:
        right_rows = candidate_rows[1]
        if sort:
            candidate_rows = np.unique(right_rows).astype(np.int32, copy=False)
        else:
            _unique_rows, first_hits = np.unique(right_rows, return_index=True)
            candidate_rows = right_rows[np.sort(first_hits)].astype(np.int32, copy=False)
    if not sort and candidate_rows.size > 1 and not candidate_rows_from_scalar_bbox:
        source_bounds = np.asarray(
            (gdf.geometry if isinstance(gdf, GeoDataFrame) else gdf).bounds,
            dtype=np.float64,
        )
        candidate_bounds = source_bounds[candidate_rows]
        order = np.lexsort(
            (
                candidate_rows,
                candidate_bounds[:, 3],
                candidate_bounds[:, 2],
                candidate_bounds[:, 1],
                candidate_bounds[:, 0],
            )
        )
        candidate_rows = candidate_rows[order].astype(np.int32, copy=False)
    ordered_row_positions = candidate_rows.astype(np.intp, copy=False)
    direct_polygon_result = _clip_homogeneous_polygon_candidates_native(
        gdf,
        intersection_polygon,
        ordered_row_positions,
        clipping_by_rectangle=clipping_by_rectangle,
        rectangle_bounds=rectangle_bounds,
        keep_geom_type=keep_geom_type,
    )
    if direct_polygon_result is not None:
        return direct_polygon_result

    gdf_sub = gdf.iloc[candidate_rows]

    geom_types = gdf_sub.geom_type
    point_mask = geom_types == "Point"
    non_point_mask = ~point_mask
    line_mask = geom_types.isin(LINE_GEOM_TYPES)
    multiline_mask = geom_types == "MultiLineString"
    simple_line_mask = line_mask & ~multiline_mask
    polygon_mask = geom_types.isin(POLYGON_GEOM_TYPES)
    generic_mask = non_point_mask & ~(simple_line_mask | multiline_mask | polygon_mask)
    rectangle_cleanup_safe = bool(
        rectangle_bounds is not None
        and (
            clipping_by_rectangle
            or (len(gdf_sub) > 0 and (geom_types == "Polygon").all())
        )
    )

    def _clip_partition_values(partition, *, use_rect_fast_path=False):
        if (
            not clipping_by_rectangle
            and (partition.geom_type == "Point").all()
        ):
            return _clip_point_partition_with_polygon_mask(
                partition,
                mask,
            )
        if (
            rectangle_bounds is not None
            and keep_geom_type
            and partition.geom_type.isin(POLYGON_GEOM_TYPES).all()
        ):
            return _clip_polygon_partition_with_rectangle_mask(
                partition,
                rectangle_bounds,
                keep_geom_type_only=True,
            )
        if (
            rectangle_bounds is not None
            and (partition.geom_type == "Polygon").all()
        ):
            return _clip_polygon_partition_with_rectangle_mask(partition, rectangle_bounds)
        if (
            rectangle_bounds is not None
            and partition.geom_type.isin(POLYGON_GEOM_TYPES).all()
        ):
            return _clip_complex_polygon_partition_with_rectangle_mask(
                partition,
                rectangle_bounds,
            )

        if isinstance(partition, GeoDataFrame):
            if not clipping_by_rectangle and partition.geom_type.isin(POLYGON_GEOM_TYPES).all():
                return _clip_polygon_partition_with_polygon_mask(
                    partition,
                    mask,
                    keep_geom_type_only=keep_geom_type,
                )

            return (
                partition.geometry.values.clip_by_rect(*rectangle_bounds)
                if use_rect_fast_path
                else partition.geometry.values.intersection(mask)
            )

        if not clipping_by_rectangle and partition.geom_type.isin(POLYGON_GEOM_TYPES).all():
            return _clip_polygon_partition_with_polygon_mask(
                partition,
                mask,
                keep_geom_type_only=keep_geom_type,
            )

        return (
            partition.values.clip_by_rect(*rectangle_bounds)
            if use_rect_fast_path
            else partition.values.intersection(mask)
        )

    parts: list[LeftConstructiveResult] = []

    def _append_part(selection_mask, *, use_rect_fast_path=False, passthrough=False):
        if not selection_mask.any():
            return
        local_mask = np.asarray(selection_mask, dtype=bool)
        if local_mask.size == len(gdf_sub) and bool(local_mask.all()):
            partition = gdf_sub
            row_positions = candidate_rows
        else:
            partition = gdf_sub[local_mask]
            row_positions = candidate_rows[local_mask]
        if (
            has_gpu_runtime()
            and _clip_partition_supports_device_promotion(partition)
            and (
                passthrough
                or use_rect_fast_path
                or not clipping_by_rectangle
            )
        ):
            partition = _promote_geometry_backing_to_device(
                partition,
                reason="clip selected candidate-limited GPU-native partition execution",
            )
        geometry_values = (
            partition.geometry.values if isinstance(partition, GeoDataFrame) else partition.values
        ) if passthrough else _clip_partition_values(
            partition,
            use_rect_fast_path=use_rect_fast_path,
        )
        parts.append(
            _build_clip_partition_result(
                gdf,
                row_positions,
                geometry_values,
            )
        )

    _append_part(point_mask, passthrough=clipping_by_rectangle)
    _append_part(
        simple_line_mask,
        use_rect_fast_path=clipping_by_rectangle,
    )
    _append_part(
        multiline_mask,
        use_rect_fast_path=rectangle_bounds is not None,
    )
    _append_part(
        polygon_mask,
        use_rect_fast_path=clipping_by_rectangle,
    )
    _append_part(
        generic_mask,
        use_rect_fast_path=clipping_by_rectangle,
    )

    parts_tuple = tuple(parts)
    return _clip_constructive_parts_to_native_tabular_result(
        source=gdf,
        parts=parts_tuple,
        ordered_row_positions=ordered_row_positions,
        clipping_by_rectangle=rectangle_cleanup_safe,
        has_non_point_candidates=bool(non_point_mask.any()),
        keep_geom_type=keep_geom_type,
        spatial_materializer=lambda: ClipNativeResult(
            source=gdf,
            parts=parts_tuple,
            ordered_index=gdf_sub.index,
            ordered_row_positions=ordered_row_positions,
            clipping_by_rectangle=rectangle_cleanup_safe,
            has_non_point_candidates=bool(non_point_mask.any()),
            keep_geom_type=keep_geom_type,
        ).to_spatial(),
    )


def _clip_source_bounds_geometry(bounds):
    bounds = tuple(float(value) for value in bounds)
    if not np.all(np.isfinite(bounds)):
        return None
    xmin, ymin, xmax, ymax = bounds
    if abs(xmax - xmin) <= SPATIAL_EPSILON and abs(ymax - ymin) <= SPATIAL_EPSILON:
        return Point(xmin, ymin)
    if abs(xmax - xmin) <= SPATIAL_EPSILON:
        return LineString([(xmin, ymin), (xmax, ymax)])
    if abs(ymax - ymin) <= SPATIAL_EPSILON:
        return LineString([(xmin, ymin), (xmax, ymax)])
    return box(xmin, ymin, xmax, ymax)


def _clip_mask_covers_source_bounds(mask, source_bounds) -> bool:
    source_extent = _clip_source_bounds_geometry(source_bounds)
    if source_extent is None:
        return False

    if _mask_is_list_like_rectangle(mask):
        try:
            xmin, ymin, xmax, ymax = (float(value) for value in mask)
        except (TypeError, ValueError):
            return False
        if not np.all(np.isfinite((xmin, ymin, xmax, ymax))):
            return False
        sxmin, symin, sxmax, symax = (float(value) for value in source_bounds)
        return bool(
            xmin <= sxmin + SPATIAL_EPSILON
            and ymin <= symin + SPATIAL_EPSILON
            and xmax + SPATIAL_EPSILON >= sxmax
            and ymax + SPATIAL_EPSILON >= symax
        )

    if not isinstance(mask, Polygon | MultiPolygon) or mask.is_empty:
        return False
    try:
        return bool(shapely.covers(mask, source_extent))
    except Exception:
        logger.debug(
            "clip mask source-bounds coverage probe failed; continuing with exact clip",
            exc_info=True,
        )
        return False


def _clip_mask_covers_source_bounds_passthrough_native(
    source,
    mask,
    source_bounds,
) -> NativeTabularResult | None:
    if not _clip_mask_covers_source_bounds(mask, source_bounds):
        return None

    geometry = source.geometry if isinstance(source, GeoDataFrame) else source
    keep_mask = ~(
        np.asarray(geometry.isna(), dtype=bool)
        | np.asarray(geometry.is_empty, dtype=bool)
    )
    passthrough = _take_spatial_rows(source, keep_mask)
    values = passthrough.geometry.values if isinstance(passthrough, GeoDataFrame) else passthrough.values
    owned = getattr(values, "_owned", None)
    selected = (
        ExecutionMode.GPU
        if owned is not None and owned.residency is Residency.DEVICE
        else ExecutionMode.CPU
    )

    from vibespatial.runtime.dispatch import record_dispatch_event

    record_dispatch_event(
        surface="geopandas.clip",
        operation="clip",
        implementation="mask_covers_source_bounds_passthrough",
        reason=(
            "mask clip physical shape reduced to source passthrough because the "
            "mask covers the source total-bounds extent"
        ),
        detail=(
            f"rows={len(source)}; kept_rows={int(np.count_nonzero(keep_mask))}; "
            "physical_shape=mask_clip"
        ),
        requested=ExecutionMode.AUTO,
        selected=selected,
    )
    return _spatial_to_native_tabular_result(passthrough)


def _clip_native_tabular_to_spatial(
    result: NativeTabularResult,
    *,
    source: GeoDataFrame | GeoSeries,
):
    if isinstance(source, GeoDataFrame):
        clipped = result.to_geodataframe()
        _maybe_seed_polygon_validity_cache(clipped)
        return clipped

    clipped = result.geometry.to_geoseries(
        index=result.attributes.index,
        name=getattr(source, "name", None) or result.geometry_name,
    )
    if result.attrs:
        clipped.attrs.update(result.attrs)
    _maybe_seed_polygon_validity_cache(clipped)
    return clipped


def _build_sparse_owned_clip_output(
    *,
    partition_crs,
    left_owned,
    inside_rows: np.ndarray,
    exact_area_owned,
    positive_local_rows: np.ndarray,
    positive_rows: np.ndarray,
    extra_owned_parts=(),
):
    from vibespatial.geometry.owned import OwnedGeometryArray, build_null_owned_array
    from vibespatial.runtime.residency import Residency

    extra_owned_parts = tuple(extra_owned_parts)

    def _take_owned_part(owned, rows: np.ndarray):
        rows = np.asarray(rows, dtype=np.intp)
        if rows.size == 0:
            return owned.take(rows)
        if (
            rows.size == owned.row_count
            and np.array_equal(rows, np.arange(owned.row_count, dtype=rows.dtype))
        ):
            return owned
        if owned.residency is Residency.DEVICE and has_gpu_runtime():
            try:
                import cupy as cp
            except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
                cp = None
            if cp is not None:
                return owned.device_take(cp.asarray(rows, dtype=cp.int64))
        return owned.take(rows)

    row_parts = [
        np.asarray(inside_rows, dtype=np.intp),
        np.asarray(positive_rows, dtype=np.intp),
        *[np.asarray(rows, dtype=np.intp) for rows, _owned in extra_owned_parts],
    ]
    kept_local_rows = np.concatenate(row_parts)
    if kept_local_rows.size == 0:
        return _ClipPartitionOutput(
            geometry_values=_geometry_values_from_owned(
                build_null_owned_array(0, residency=left_owned.residency),
                crs=partition_crs,
            ),
            local_rows=np.empty(0, dtype=np.intp),
        )

    owned_parts = []
    if inside_rows.size > 0:
        owned_parts.append(_take_owned_part(left_owned, np.asarray(inside_rows, dtype=np.intp)))
    if positive_local_rows.size > 0:
        owned_parts.append(
            _take_owned_part(exact_area_owned, np.asarray(positive_local_rows, dtype=np.intp))
        )
    owned_parts.extend(owned for _rows, owned in extra_owned_parts)

    result_owned = OwnedGeometryArray.concat(owned_parts)
    ordered_local_rows = kept_local_rows
    if np.any(ordered_local_rows[1:] < ordered_local_rows[:-1]):
        reorder = np.argsort(ordered_local_rows, kind="stable").astype(np.intp, copy=False)
        result_owned = _take_owned_part(result_owned, reorder)
        ordered_local_rows = ordered_local_rows[reorder]
    result_owned._clip_semantically_clean = True

    return _ClipPartitionOutput(
        geometry_values=_geometry_values_from_owned(result_owned, crs=partition_crs),
        local_rows=ordered_local_rows,
    )


def _clip_gdf_with_mask(gdf, mask, sort=False, *, query_geometry=None):
    """
    Clip geometry to the polygon/rectangle extent.

    Clip an input GeoDataFrame to the polygon extent of the polygon
    parameter.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        Dataframe to clip.

    mask : (Multi)Polygon, list-like
        Reference polygon/rectangle for clipping.

    sort : boolean, default False
        If True, the results will be sorted in ascending order using the
        geometries' indexes as the primary key.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a clipped subset of gdf
        that intersects with polygon/rectangle.
    """
    native_result = _clip_gdf_with_mask_native(
        gdf,
        mask,
        sort=sort,
        query_geometry=query_geometry,
        keep_geom_type=False,
    )
    return _clip_native_tabular_to_spatial(native_result, source=gdf)


def evaluate_geopandas_clip_native(
    gdf,
    mask,
    *,
    keep_geom_type: bool = False,
    sort: bool = False,
) -> NativeTabularResult:
    """Build a native clip result and defer GeoPandas export to the boundary."""
    original = gdf

    if not isinstance(gdf, GeoDataFrame | GeoSeries):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    clipping_by_rectangle = _mask_is_list_like_rectangle(mask)
    if (
        not isinstance(mask, GeoDataFrame | GeoSeries | Polygon | MultiPolygon)
        and not clipping_by_rectangle
    ):
        raise TypeError(
            "'mask' should be GeoDataFrame, GeoSeries,"
            f"(Multi)Polygon or list-like, got {type(mask)}"
        )

    if clipping_by_rectangle and len(mask) != 4:
        raise TypeError(
            "If 'mask' is list-like, it must have four values (minx, miny, maxx, maxy)"
        )

    if isinstance(mask, GeoDataFrame | GeoSeries) and not _check_crs(gdf, mask):
        _crs_mismatch_warn(gdf, mask, stacklevel=3)

    if clipping_by_rectangle:
        polygon_mask_bounds = None
    elif isinstance(mask, GeoDataFrame | GeoSeries) and len(mask) == 1:
        polygon_mask_bounds = _rectangle_bounds_from_mask(mask.geometry.iloc[0])
    else:
        polygon_mask_bounds = _rectangle_bounds_from_mask(mask)

    promote_polygon_mask_to_device = (
        has_gpu_runtime()
        and not clipping_by_rectangle
        and (
            polygon_mask_bounds is None
            or (
                strict_native_mode_enabled()
                and bool(gdf.geom_type.isin(POLYGON_GEOM_TYPES).all())
            )
        )
    )

    if promote_polygon_mask_to_device:
        gdf = _promote_geometry_backing_to_device(
            gdf,
            reason="clip(): GPU boundary selection for source geometry",
        )
        if isinstance(mask, GeoDataFrame | GeoSeries):
            mask = _promote_geometry_backing_to_device(
                mask,
                reason="clip(): GPU boundary selection for mask geometry",
            )

    if isinstance(mask, GeoDataFrame | GeoSeries):
        box_mask = mask.total_bounds
    elif clipping_by_rectangle:
        box_mask = mask
    else:
        box_mask = mask.bounds if not mask.is_empty else (np.nan,) * 4
    box_gdf = gdf.total_bounds
    if not (
        ((box_mask[0] <= box_gdf[2]) and (box_gdf[0] <= box_mask[2]))
        and ((box_mask[1] <= box_gdf[3]) and (box_gdf[1] <= box_mask[3]))
    ):
        return _clip_constructive_parts_to_native_tabular_result(
            source=original,
            parts=(),
            ordered_row_positions=np.empty(0, dtype=np.intp),
            clipping_by_rectangle=clipping_by_rectangle,
            has_non_point_candidates=False,
            keep_geom_type=keep_geom_type,
        )

    mask_query_geometry = None
    if isinstance(mask, GeoDataFrame | GeoSeries):
        mask_query_geometry = mask.geometry if isinstance(mask, GeoDataFrame) else mask
        if len(mask) == 1:
            combined_mask = mask.geometry.iloc[0]
        else:
            combined_mask = mask.geometry.union_all()
    else:
        combined_mask = mask

    passthrough_result = _clip_mask_covers_source_bounds_passthrough_native(
        gdf,
        combined_mask,
        box_gdf,
    )
    if passthrough_result is not None:
        return passthrough_result

    return _clip_gdf_with_mask_native(
        gdf,
        combined_mask,
        sort=sort,
        query_geometry=mask_query_geometry,
        keep_geom_type=keep_geom_type,
    )


def clip(gdf, mask, keep_geom_type=False, sort=False):
    """Clip points, lines, or polygon geometries to the mask extent.

    Both layers must be in the same Coordinate Reference System (CRS).
    The ``gdf`` will be clipped to the full extent of the clip object.

    If there are multiple polygons in mask, data from ``gdf`` will be
    clipped to the total boundary of all polygons in mask.

    If the ``mask`` is list-like with four elements ``(minx, miny, maxx, maxy)``, a
    faster rectangle clipping algorithm will be used. Note that this can lead to
    slightly different results in edge cases, e.g. if a line would be reduced to a
    point, this point might not be returned.
    The geometry is clipped in a fast but possibly dirty way. The output is not
    guaranteed to be valid. No exceptions will be raised for topological errors.

    Parameters
    ----------
    gdf : GeoDataFrame or GeoSeries
        Vector layer (point, line, polygon) to be clipped to mask.
    mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
        Polygon vector layer used to clip ``gdf``.
        The mask's geometry is dissolved into one geometric feature
        and intersected with ``gdf``.
        If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
        ``clip`` will use a faster rectangle clipping (:meth:`~GeoSeries.clip_by_rect`),
        possibly leading to slightly different results.
    keep_geom_type : boolean, default False
        If True, return only geometries of original type in case of intersection
        resulting in multiple geometry types or GeometryCollections.
        If False, return all resulting geometries (potentially mixed-types).
    sort : boolean, default False
        If True, the results will be sorted in ascending order using the
        geometries' indexes as the primary key.

    Returns
    -------
    GeoDataFrame or GeoSeries
         Vector data (points, lines, polygons) from ``gdf`` clipped to
         polygon boundary from mask.

    See Also
    --------
    GeoDataFrame.clip : equivalent GeoDataFrame method
    GeoSeries.clip : equivalent GeoSeries method

    Examples
    --------
    Clip points (grocery stores) with polygons (the Near West Side community):

    >>> import geodatasets
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... )
    >>> near_west_side = chicago[chicago["community"] == "NEAR WEST SIDE"]
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... ).to_crs(chicago.crs)
    >>> groceries.shape
    (148, 8)

    >>> nws_groceries = geopandas.clip(groceries, near_west_side)
    >>> nws_groceries.shape
    (7, 8)
    """
    native_result = evaluate_geopandas_clip_native(
        gdf,
        mask,
        keep_geom_type=keep_geom_type,
        sort=sort,
    )
    return _clip_native_tabular_to_spatial(native_result, source=gdf)
