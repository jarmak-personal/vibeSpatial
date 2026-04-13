"""Shared helpers for Florida OSM PBF read shootouts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[3]
PBF_PATH = ROOT / ".benchmark_fixtures" / "fl-latest-osm.pbf"


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    return path


def running_with_vibespatial() -> bool:
    return hasattr(gpd, "get_runtime_selection")


def _empty_frame(*, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"osm_element": pd.Series(dtype="object"), "geometry": gpd.GeoSeries([], crs=crs)},
        geometry="geometry",
        crs=crs,
    )


def _normalize_simple_layer(
    frame: gpd.GeoDataFrame,
    *,
    osm_element: str,
) -> gpd.GeoDataFrame:
    if frame.empty:
        return _empty_frame(crs=frame.crs or "EPSG:4326")
    return gpd.GeoDataFrame(
        {
            "osm_element": pd.Series(osm_element, index=frame.index, dtype="object"),
            "geometry": frame.geometry,
        },
        geometry="geometry",
        crs=frame.crs,
    )


def _normalize_multipolygons_layer(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        return _empty_frame(crs=frame.crs or "EPSG:4326")
    way_mask = (
        frame["osm_way_id"].notna()
        if "osm_way_id" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    osm_element = pd.Series("relation", index=frame.index, dtype="object")
    osm_element.loc[way_mask] = "way"
    return gpd.GeoDataFrame(
        {
            "osm_element": osm_element,
            "geometry": frame.geometry,
        },
        geometry="geometry",
        crs=frame.crs,
    )


def combine_supported_osm_layers(
    points: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    multilinestrings: gpd.GeoDataFrame,
    multipolygons: gpd.GeoDataFrame,
    other_relations: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    parts = [
        _normalize_simple_layer(points, osm_element="node"),
        _normalize_simple_layer(lines, osm_element="way"),
        _normalize_simple_layer(multilinestrings, osm_element="relation"),
        _normalize_multipolygons_layer(multipolygons),
        _normalize_simple_layer(other_relations, osm_element="relation"),
    ]
    non_empty = [part for part in parts if not part.empty]
    if not non_empty:
        return _empty_frame()
    crs = non_empty[0].crs
    combined = pd.concat(non_empty, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=crs)


def normalize_public_osm_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        return _empty_frame(crs=frame.crs or "EPSG:4326")
    if "osm_element" in frame.columns:
        osm_element = frame["osm_element"].astype("object")
    else:
        osm_element = pd.Series("unknown", index=frame.index, dtype="object")
    return gpd.GeoDataFrame(
        {
            "osm_element": osm_element,
            "geometry": frame.geometry,
        },
        geometry="geometry",
        crs=frame.crs,
    )


def normalize_single_layer_osm_frame(
    frame: gpd.GeoDataFrame,
    *,
    osm_element: str,
) -> gpd.GeoDataFrame:
    if frame.empty:
        return _empty_frame(crs=frame.crs or "EPSG:4326")
    return gpd.GeoDataFrame(
        {
            "osm_element": pd.Series(osm_element, index=frame.index, dtype="object"),
            "geometry": frame.geometry,
        },
        geometry="geometry",
        crs=frame.crs,
    )


def normalize_multipolygons_public_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        return _empty_frame(crs=frame.crs or "EPSG:4326")
    if "osm_element" in frame.columns:
        return normalize_public_osm_frame(frame)
    return _normalize_multipolygons_layer(frame)


def read_supported_layers_pyogrio(
    path: Path,
    *,
    geometry_only: bool,
) -> gpd.GeoDataFrame:
    import pyogrio

    def _read(layer: str) -> gpd.GeoDataFrame:
        kwargs = {"layer": layer}
        if geometry_only:
            kwargs["columns"] = []
        try:
            return pyogrio.read_dataframe(path, **kwargs)
        except Exception:
            if geometry_only:
                kwargs.pop("columns", None)
                frame = pyogrio.read_dataframe(path, **kwargs)
                return gpd.GeoDataFrame(geometry=frame.geometry, crs=frame.crs)
            raise

    return combine_supported_osm_layers(
        _read("points"),
        _read("lines"),
        _read("multilinestrings"),
        _read("multipolygons"),
        _read("other_relations"),
    )


def read_pyogrio_osm_layer(
    path: Path,
    *,
    layer: str,
    geometry_only: bool,
) -> gpd.GeoDataFrame:
    import pyogrio

    kwargs = {"layer": layer}
    if geometry_only:
        kwargs["columns"] = []
    try:
        return pyogrio.read_dataframe(path, **kwargs)
    except Exception:
        if geometry_only:
            kwargs.pop("columns", None)
            frame = pyogrio.read_dataframe(path, **kwargs)
            return gpd.GeoDataFrame(geometry=frame.geometry, crs=frame.crs)
        raise


def frame_from_osm_result(result) -> gpd.GeoDataFrame:
    from vibespatial.io.geoarrow import geoseries_from_owned

    parts: list[gpd.GeoDataFrame] = []
    if result.nodes is not None and result.n_nodes > 0:
        parts.append(
            gpd.GeoDataFrame(
                {
                    "osm_element": pd.Series("node", index=range(result.n_nodes), dtype="object"),
                },
                geometry=geoseries_from_owned(result.nodes, crs="EPSG:4326", name="geometry"),
                crs="EPSG:4326",
            )
        )
    if result.ways is not None and result.n_ways > 0:
        parts.append(
            gpd.GeoDataFrame(
                {
                    "osm_element": pd.Series("way", index=range(result.n_ways), dtype="object"),
                },
                geometry=geoseries_from_owned(result.ways, crs="EPSG:4326", name="geometry"),
                crs="EPSG:4326",
            )
        )
    if result.relations is not None and result.n_relations > 0:
        parts.append(
            gpd.GeoDataFrame(
                {
                    "osm_element": pd.Series(
                        "relation", index=range(result.n_relations), dtype="object"
                    ),
                },
                geometry=geoseries_from_owned(
                    result.relations, crs="EPSG:4326", name="geometry"
                ),
                crs="EPSG:4326",
            )
        )
    if not parts:
        return _empty_frame()
    combined = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")


def fingerprint(frame: gpd.GeoDataFrame) -> str:
    if frame.empty:
        bounds = (0.0, 0.0, 0.0, 0.0)
        geom_types = pd.Series(dtype="object")
        osm_elements = pd.Series(dtype="object")
    else:
        bounds = tuple(round(float(value), 6) for value in frame.total_bounds)
        geom_types = frame.geometry.geom_type.astype("object")
        osm_elements = frame["osm_element"].astype("object")

    point = int((geom_types == "Point").sum())
    linear = int(geom_types.isin(["LineString", "MultiLineString"]).sum())
    area = int(geom_types.isin(["Polygon", "MultiPolygon"]).sum())
    other = int(len(frame) - point - linear - area)

    node = int((osm_elements == "node").sum())
    way = int((osm_elements == "way").sum())
    relation = int((osm_elements == "relation").sum())
    unknown = int(len(frame) - node - way - relation)

    return (
        f"rows={len(frame)} "
        f"elements=(node={node},way={way},relation={relation},unknown={unknown}) "
        f"geom=(point={point},linear={linear},area={area},other={other}) "
        f"bounds={bounds}"
    )
