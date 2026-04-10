"""Module to clip vector data using GeoPandas."""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas.api.types
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._compat import PANDAS_GE_30
from vibespatial.api._native_results import GeometryNativeResult, LeftConstructiveResult
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
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.residency import Residency


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
    bounds overlap pass.
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

    # Avoid O(n) bounds filtering for genuinely large clip workloads where the
    # flat spatial index amortizes its build/query cost better.
    if len(gdf) > 50_000:
        return None

    source_bounds = np.asarray(gdf.geometry.bounds, dtype=np.float64)
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
    if keep_mask.all():
        return spatial
    keep_rows = np.flatnonzero(keep_mask).astype(np.intp, copy=False)
    return spatial.iloc[keep_rows].copy(deep=not PANDAS_GE_30)


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

    boundary_geoms = np.asarray(values, dtype=object)
    return all(_is_axis_aligned_rectangle_polygon(geom) for geom in boundary_geoms)


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

    boundary_rect_bounds = None
    if owned.device_state is not None:
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by device residency
            cp = None
        if cp is not None:
            device_polygon_buf = owned.device_state.families.get(GeometryFamily.POLYGON)
            if device_polygon_buf is not None:
                device_bounds = _device_rectangle_bounds(device_polygon_buf, owned.row_count)
                if device_bounds is not None:
                    boundary_rect_bounds = np.column_stack(
                        [cp.asnumpy(axis) for axis in device_bounds]
                    ).astype(np.float64, copy=False)
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

        if self.clipping_by_rectangle:
            keep_mask = ~clipped_geometry.isna() & ~clipped_geometry.is_empty
            return _take_spatial_rows(clipped, keep_mask)

        keep = ~clipped_geometry.isna() & ~clipped_geometry.is_empty
        if self.has_non_point_candidates:
            poly_rows = clipped.geom_type.isin(POLYGON_GEOM_TYPES)
            if poly_rows.any():
                poly_mask = np.asarray(poly_rows, dtype=bool)
                if isinstance(clipped_values, DeviceGeometryArray):
                    nonpositive_area = (
                        np.asarray(clipped_values.area, dtype=np.float64)[poly_mask] <= 0.0
                    )
                else:
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
                    poly_keep = np.ones(poly_mask.sum(), dtype=bool)
                    poly_keep[nonpositive_area] = False
                    keep_array = np.array(keep, dtype=bool, copy=True)
                    keep_array[np.flatnonzero(poly_mask)] &= poly_keep
                    keep = keep_array

            line_rows = clipped.geom_type.isin(LINE_GEOM_TYPES)
            if line_rows.any():
                line_mask = np.asarray(line_rows, dtype=bool)
                if isinstance(clipped_values, DeviceGeometryArray | GeometryArray):
                    degenerate_lines = (
                        np.asarray(clipped_values.length, dtype=np.float64)[line_mask] == 0.0
                    )
                else:
                    line_values = np.asarray(clipped_geometry, dtype=object)[line_mask]
                    degenerate_lines = np.asarray(
                        shapely.length(line_values),
                        dtype=np.float64,
                    ) == 0.0
                if np.any(degenerate_lines):
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
                    if isinstance(clipped, GeoDataFrame):
                        geom_name = clipped._geometry_column_name
                        clipped = clipped.copy(deep=not PANDAS_GE_30)
                        clipped[geom_name] = GeoSeries(
                            _geometryarray_from_shapely(repaired_values, crs=self.source.crs),
                            index=clipped.index,
                            crs=self.source.crs,
                        )
                    else:
                        clipped = GeoSeries(
                            _geometryarray_from_shapely(repaired_values, crs=self.source.crs),
                            index=clipped.index,
                            crs=self.source.crs,
                            name=clipped.name,
                        )
                    clipped_geometry = clipped.geometry if isinstance(clipped, GeoDataFrame) else clipped
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
        return self._apply_keep_geom_type(clipped)

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
):
    """Compute polygon-mask area intersections without shedding the GPU path.

    Once clip selects the polygon-owned execution family and a CUDA runtime is
    available, the area-producing step stays on the GPU many-vs-one path with
    containment bypass, SH clipping, and batched row-isolated exact remainder.
    The older direct exact path remains the CPU/no-GPU fallback.
    """
    from vibespatial.api.tools.overlay import _many_vs_one_intersection_owned
    from vibespatial.constructive.binary_constructive import (
        binary_constructive_owned,
    )
    from vibespatial.geometry.owned import materialize_broadcast, tile_single_row
    from vibespatial.runtime.residency import Residency

    if has_gpu_runtime() and left_owned.residency is Residency.DEVICE:
        from vibespatial.kernels.constructive.polygon_rect_intersection import (
            polygon_rect_intersection,
            polygon_rect_intersection_can_handle,
        )
        from vibespatial.runtime import ExecutionMode

        if prefer_exact_polygon_rect_batch:
            return binary_constructive_owned(
                "intersection",
                left_owned,
                mask_owned,
                dispatch_mode=ExecutionMode.GPU,
                _prefer_exact_polygon_intersection=True,
                _allow_rectangle_intersection_fast_path=True,
            )

        tiled_mask = (
            mask_owned
            if left_owned.row_count == 1
            else materialize_broadcast(
                tile_single_row(mask_owned, left_owned.row_count),
            )
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
                "falling back to generic exact constructive path"
            ),
            detail=f"{type(exc).__name__}: {exc}",
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            pipeline="_clip_polygon_area_intersection_owned",
            d2h_transfer=False,
        )
        return binary_constructive_owned(
            "intersection",
            left_owned,
            mask_owned,
            dispatch_mode=ExecutionMode.CPU,
            _prefer_exact_polygon_intersection=True,
            _allow_rectangle_intersection_fast_path=allow_rectangle_kernel,
        )


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

    if left_owned.row_count > 1:
        mask_owned = materialize_broadcast(
            tile_single_row(mask_owned, left_owned.row_count),
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
    from vibespatial.api.tools.overlay import _strip_non_polygon_collection_parts
    from vibespatial.constructive.binary_constructive import _binary_constructive_gpu
    from vibespatial.geometry.owned import (
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
        raw = np.asarray(result.to_shapely(), dtype=object)
        cleaned = _strip_non_polygon_collection_parts(raw)
        for row_index, geom in enumerate(cleaned):
            if geom is None or geom.is_empty or getattr(geom, "area", 0.0) <= 0.0:
                cleaned[row_index] = None
        return from_shapely_geometries(
            cleaned.tolist(),
            residency=left_owned.residency,
        )

    raw = np.asarray(
        shapely.intersection(
            np.asarray(left_owned.to_shapely(), dtype=object),
            np.full(left_owned.row_count, rectangle_mask, dtype=object),
        ),
        dtype=object,
    )
    cleaned = _strip_non_polygon_collection_parts(raw)
    for row_index, geom in enumerate(cleaned):
        if geom is None or geom.is_empty or getattr(geom, "area", 0.0) <= 0.0:
            cleaned[row_index] = None
    return from_shapely_geometries(
        cleaned.tolist(),
        residency=left_owned.residency,
    )


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
        from vibespatial.geometry.owned import materialize_broadcast, tile_single_row
        from vibespatial.predicates.binary import NullBehavior, evaluate_binary_predicate
        from vibespatial.runtime import ExecutionMode

        boundary_owned = left_owned.take(boundary_rows)
        broadcast_mask = (
            mask_owned
            if boundary_owned.row_count == 1
            else materialize_broadcast(
                tile_single_row(mask_owned, boundary_owned.row_count),
            )
        )
        result = evaluate_binary_predicate(
            "intersects",
            boundary_owned,
            broadcast_mask,
            dispatch_mode=ExecutionMode.GPU,
            null_behavior=NullBehavior.FALSE,
        )
        return np.asarray(result.values, dtype=bool)

    return np.asarray(source_values.take(boundary_rows).intersects(mask), dtype=bool)


def _owned_nonempty_polygon_mask(owned) -> np.ndarray:
    """Return rows backed by polygonal output with strictly positive area."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    validity = np.asarray(owned.validity, dtype=bool)
    if not validity.any():
        return validity

    tags = np.asarray(owned.tags)
    row_offsets = np.asarray(owned.family_row_offsets)
    keep_mask = np.zeros(len(tags), dtype=bool)

    for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        family_tag = FAMILY_TAGS[family]
        family_mask = validity & (tags == family_tag)
        if not family_mask.any():
            continue
        owned._ensure_host_family_structure(family)
        family_rows = row_offsets[family_mask]
        empty_mask = np.asarray(owned.families[family].empty_mask, dtype=bool)
        if empty_mask.size == 0 or np.any((family_rows < 0) | (family_rows >= empty_mask.size)):
            values = np.asarray(owned.to_shapely(), dtype=object)
            return ~(shapely.is_missing(values) | shapely.is_empty(values)) & (shapely.area(values) > 0.0)
        family_indices = np.flatnonzero(family_mask)
        candidate_rows = family_indices[~empty_mask[family_rows]]
        if candidate_rows.size == 0:
            continue
        candidate_values = np.asarray(
            owned.take(candidate_rows).to_shapely(),
            dtype=object,
        )
        keep_mask[candidate_rows] = np.asarray(
            shapely.area(candidate_values),
            dtype=np.float64,
        ) > 0.0

    return keep_mask


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
):
    """Clip polygon rows to a rectangle mask while preserving sliver leftovers.

    Rows fully inside the rectangle are pass-through. Only rows that cross or
    touch the rectangle boundary require exact area intersection to
    preserve lower-dimensional public clip semantics.
    """
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
        return _geometry_values_from_owned(result_owned, crs=partition.crs)

    if source_owned is not None and inside_rows.size > 0:
        assembled[inside_rows] = np.asarray(
            source_owned.take(inside_rows).to_shapely(),
            dtype=object,
        )
    return _as_geometry_values(assembled, crs=partition.crs)


def _clip_polygon_partition_with_polygon_mask(partition, mask):
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

    left_owned = source_values.to_owned()
    mask_owned = from_shapely_geometries([mask], residency=left_owned.residency)

    # Device-backed polygon clip benefits from a cheap exact predicate refine
    # before exact intersection: bbox candidates include both false positives
    # and polygons already fully covered by the mask, and paying full exact
    # intersection for those rows dominates 10K-scale workflows.
    if has_gpu_runtime() and left_owned.residency is Residency.DEVICE:
        from vibespatial.geometry.owned import materialize_broadcast, tile_single_row
        from vibespatial.predicates.binary import evaluate_binary_predicate
        from vibespatial.runtime import ExecutionMode

        def _take_owned_rows(owned, rows: np.ndarray):
            if rows.size == 0:
                return owned.take(rows)
            if owned.residency is Residency.DEVICE:
                import cupy as cp

                return owned.device_take(cp.asarray(rows, dtype=cp.int64))
            return owned.take(rows)

        inside_rows = np.asarray([], dtype=np.intp)
        exact_rows = np.asarray([], dtype=np.intp)
        prefer_exact_polygon_rect_batch = (
            not allow_rectangle_kernel
            and _all_axis_aligned_rectangle_polygons(source_values)
        )
        if left_owned.row_count == 1 and not source_missing[0]:
            exact_rows = np.asarray([0], dtype=np.intp)
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
            miss_rows = np.flatnonzero(~source_missing & ~intersects_mask).astype(
                np.intp,
                copy=False,
            )
            if miss_rows.size > 0:
                miss_owned = _take_owned_rows(left_owned, miss_rows)
                miss_mask = (
                    mask_owned
                    if miss_owned.row_count == 1
                    else materialize_broadcast(
                        tile_single_row(mask_owned, miss_owned.row_count),
                    )
                )
                touch_hits = np.asarray(
                    evaluate_binary_predicate(
                        "touches",
                        miss_owned,
                        miss_mask,
                        dispatch_mode=ExecutionMode.GPU,
                    ).values,
                    dtype=bool,
                )
                if touch_hits.any():
                    intersects_mask = intersects_mask.copy()
                    intersects_mask[miss_rows[touch_hits]] = True
            hit_rows = np.flatnonzero(~source_missing & intersects_mask).astype(
                np.intp,
                copy=False,
            )
            if hit_rows.size > 0:
                if prefer_exact_polygon_rect_batch:
                    exact_rows = hit_rows
                elif allow_rectangle_kernel:
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
                else:
                    # Concave polygon masks have shown false covered_by positives on
                    # some polygon rows. Validate only this shortcut on host while
                    # keeping the exact constructive work on the owned/device path.
                    inside_hits = np.asarray(
                        shapely.covered_by(
                            np.asarray(source_values.take(hit_rows), dtype=object),
                            mask,
                            ),
                            dtype=bool,
                        )
                    inside_rows = hit_rows[inside_hits]
                    exact_rows = hit_rows[~inside_hits]

        exact_area_owned = None
        exact_area_values = None
        exact_positive_local = np.asarray([], dtype=bool)
        if exact_rows.size > 0:
            exact_area_owned = _clip_polygon_area_intersection_owned(
                _take_owned_rows(left_owned, exact_rows),
                mask_owned,
                allow_rectangle_kernel=allow_rectangle_kernel,
                prefer_exact_polygon_rect_batch=prefer_exact_polygon_rect_batch,
            )
            exact_area_values = _geometry_values_from_owned(exact_area_owned, crs=partition.crs)
            exact_positive_local = _owned_nonempty_polygon_mask(exact_area_owned)
            if not allow_rectangle_kernel:
                exact_area_objects = np.asarray(exact_area_values, dtype=object)
                boundary_local_rows = np.flatnonzero(~exact_positive_local).astype(np.intp, copy=False)
                if exact_rows.size <= 128:
                    correction_local_rows = np.arange(exact_rows.size, dtype=np.intp)
                else:
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
                    correction_local_rows = np.union1d(boundary_local_rows, triangleish_local_rows).astype(
                        np.intp,
                        copy=False,
                    )
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
                    area_subset_objects = exact_area_objects[correction_local_rows]
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
                            corrected_exact = exact_area_objects.copy()
                            corrected_exact[correction_local_rows] = assembled_exact
                            exact_area_owned = None
                            exact_area_values = _as_geometry_values(corrected_exact, crs=partition.crs)
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
                            exact_area_values = _geometry_values_from_owned(
                                exact_area_owned,
                                crs=partition.crs,
                            )
                if exact_area_owned is not None:
                    exact_positive_local = _owned_nonempty_polygon_mask(exact_area_owned)
                else:
                    exact_objects = np.asarray(exact_area_values, dtype=object)
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
                if positive_local_rows.size > 0 and exact_area_values is not None:
                    assembled[positive_rows] = np.asarray(
                        exact_area_values.take(positive_local_rows),
                        dtype=object,
                    )
                return _as_geometry_values(assembled, crs=partition.crs)
            result_owned = build_null_owned_array(
                len(partition),
                residency=left_owned.residency,
            )
            if inside_rows.size > 0:
                result_owned = concat_owned_scatter(
                    result_owned,
                    _take_owned_rows(left_owned, inside_rows),
                    inside_rows,
                )
            if positive_local_rows.size > 0 and exact_area_owned is not None:
                result_owned = concat_owned_scatter(
                    result_owned,
                    _take_owned_rows(exact_area_owned, positive_local_rows),
                    positive_rows,
                )
            return _geometry_values_from_owned(result_owned, crs=partition.crs)

        left_pairs = _take_geometry_object_values(source_values, boundary_rows)
        area_objects = _take_geometry_object_values(exact_area_values, boundary_local_rows)
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
            if positive_local_rows.size > 0 and exact_area_values is not None:
                assembled[positive_rows] = np.asarray(
                    exact_area_values.take(positive_local_rows),
                    dtype=object,
                )
            assembled[boundary_rows] = assembled_boundary
            return _as_geometry_values(assembled, crs=partition.crs)
        if not contains_collection:
            result_owned = build_null_owned_array(
                len(partition),
                residency=left_owned.residency,
            )
            if inside_rows.size > 0:
                result_owned = concat_owned_scatter(
                    result_owned,
                    _take_owned_rows(left_owned, inside_rows),
                    inside_rows,
                )
            if positive_local_rows.size > 0 and exact_area_owned is not None:
                result_owned = concat_owned_scatter(
                    result_owned,
                    _take_owned_rows(exact_area_owned, positive_local_rows),
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
            preserved_local_rows = boundary_local_rows[preserve_mask]
            preserved_rows = boundary_rows[preserve_mask]
            if preserved_local_rows.size > 0 and exact_area_owned is not None:
                result_owned = concat_owned_scatter(
                    result_owned,
                    _take_owned_rows(exact_area_owned, preserved_local_rows),
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
            result_owned = build_null_owned_array(
                len(partition),
                residency=left_owned.residency,
            )
        else:
            result_owned = concat_owned_scatter(
                build_null_owned_array(
                    len(partition),
                    residency=left_owned.residency,
                ),
                area_owned.take(positive_rows),
                positive_rows,
            )
        return _geometry_values_from_owned(result_owned, crs=partition.crs)

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
    boundary_values = _clip_polygon_partition_with_rectangle_mask(partition, rectangle_bounds)
    area_objects = np.asarray(area_values, dtype=object)
    boundary_objects = np.asarray(boundary_values, dtype=object)
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
        if bool(shapely.equals(area_geom, boundary_geom)):
            combined[row_index] = area_geom
            continue
        combined[row_index] = GeometryCollection([area_geom, boundary_geom])

    return _as_geometry_values(combined, crs=partition.crs)


def _build_clip_partition_result(source, row_positions, geometry_values):
    """Create a native row-preserving clip fragment without frame assembly."""
    return _clip_native_part(
        source,
        row_positions,
        _as_geometry_values(
            geometry_values,
            crs=source.crs,
        ),
    )


def _clip_gdf_with_mask_native(
    gdf,
    mask,
    sort=False,
    *,
    query_geometry=None,
    keep_geom_type: bool = False,
) -> ClipNativeResult:
    """Build a native clip result and defer GeoPandas assembly to explicit export."""
    clipping_by_rectangle = _mask_is_list_like_rectangle(mask)
    rectangle_bounds = _rectangle_bounds_from_mask(mask)
    if clipping_by_rectangle:
        intersection_polygon = box(*mask)
    else:
        intersection_polygon = mask

    query_input = query_geometry if query_geometry is not None else intersection_polygon
    candidate_rows = (
        _bbox_candidate_rows_for_scalar_clip_mask(
            gdf,
            query_input,
            sort=sort,
        )
        if sort
        else None
    )
    if candidate_rows is None:
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
    gdf_sub = gdf.iloc[candidate_rows]

    point_mask = gdf_sub.geom_type == "Point"
    non_point_mask = ~point_mask
    line_mask = gdf_sub.geom_type.isin(LINE_GEOM_TYPES)
    multiline_mask = gdf_sub.geom_type == "MultiLineString"
    simple_line_mask = line_mask & ~multiline_mask
    polygon_mask = gdf_sub.geom_type.isin(POLYGON_GEOM_TYPES)
    generic_mask = non_point_mask & ~(simple_line_mask | multiline_mask | polygon_mask)

    def _clip_partition_values(partition, *, use_rect_fast_path=False):
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
                return _clip_polygon_partition_with_polygon_mask(partition, mask)

            return (
                partition.geometry.values.clip_by_rect(*rectangle_bounds)
                if use_rect_fast_path
                else partition.geometry.values.intersection(mask)
            )

        if not clipping_by_rectangle and partition.geom_type.isin(POLYGON_GEOM_TYPES).all():
            return _clip_polygon_partition_with_polygon_mask(partition, mask)

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
        partition = gdf_sub[local_mask]
        row_positions = candidate_rows[local_mask]
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

    _append_part(point_mask, passthrough=True)
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

    return ClipNativeResult(
        source=gdf,
        parts=tuple(parts),
        ordered_index=gdf_sub.index,
        ordered_row_positions=candidate_rows.astype(np.intp, copy=False),
        clipping_by_rectangle=clipping_by_rectangle,
        has_non_point_candidates=bool(non_point_mask.any()),
        keep_geom_type=keep_geom_type,
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
    return native_result.to_spatial()


def evaluate_geopandas_clip_native(
    gdf,
    mask,
    *,
    keep_geom_type: bool = False,
    sort: bool = False,
) -> ClipNativeResult:
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

    if has_gpu_runtime() and not clipping_by_rectangle and polygon_mask_bounds is None:
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
        return ClipNativeResult(
            source=original,
            parts=(),
            ordered_index=original.iloc[:0].index,
            ordered_row_positions=np.empty(0, dtype=np.intp),
            clipping_by_rectangle=clipping_by_rectangle,
            has_non_point_candidates=False,
            keep_geom_type=keep_geom_type,
        )

    mask_query_geometry = None
    if isinstance(mask, GeoDataFrame | GeoSeries):
        if polygon_mask_bounds is None and has_gpu_runtime():
            mask_query_geometry = mask.geometry.values.to_owned()
        if len(mask) == 1:
            combined_mask = mask.geometry.iloc[0]
        else:
            combined_mask = mask.geometry.union_all()
    else:
        combined_mask = mask

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
    return evaluate_geopandas_clip_native(
        gdf,
        mask,
        keep_geom_type=keep_geom_type,
        sort=sort,
    ).to_spatial()
