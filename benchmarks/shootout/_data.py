"""Shared fixture generation for shootout scripts.

Generates synthetic GIS data and writes it to a temp directory so each
workflow script starts with gpd.read_parquet() / gpd.read_file().

IMPORTANT: This module must use ONLY shapely, numpy, pandas, geopandas,
and stdlib.  The geopandas baseline runs in an isolated environment with
no access to vibespatial.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

import geopandas as gpd


def get_scale() -> int:
    """Read VSBENCH_SCALE env var, default 10_000."""
    raw = os.environ.get("VSBENCH_SCALE", "10000")
    # Accept shorthand like "10k", "100K", "1M"
    raw = raw.strip().upper()
    multipliers = {"K": 1_000, "M": 1_000_000}
    for suffix, mult in multipliers.items():
        if raw.endswith(suffix):
            return int(float(raw[:-1]) * mult)
    return int(raw)


# ---------------------------------------------------------------------------
# Geometry construction helpers
# ---------------------------------------------------------------------------

def _grid_side(count: int) -> int:
    return max(1, math.ceil(math.sqrt(count)))


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n == 1:
        return [(lo + hi) / 2.0]
    return list(np.linspace(lo, hi, n))


def _make_clustered_points(
    count: int, seed: int, *, clusters: int = 8,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1000.0, 1000.0),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bounds
    centers_x = rng.uniform(xmin, xmax, clusters)
    centers_y = rng.uniform(ymin, ymax, clusters)
    picks = rng.integers(0, clusters, count)
    xs = centers_x[picks] + rng.normal(0.0, (xmax - xmin) / 30.0, count)
    ys = centers_y[picks] + rng.normal(0.0, (ymax - ymin) / 30.0, count)
    xs = np.clip(xs, xmin, xmax)
    ys = np.clip(ys, ymin, ymax)
    return shapely.points(xs, ys)


def _make_grid_polygons(
    count: int, seed: int, *,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1000.0, 1000.0),
) -> np.ndarray:
    count = max(count, 1)
    xmin, ymin, xmax, ymax = bounds
    side = _grid_side(count)
    xs = _linspace(xmin, xmax, side + 1)
    ys = _linspace(ymin, ymax, side + 1)
    geoms = []
    for row in range(side):
        for col in range(side):
            if len(geoms) >= count:
                break
            geoms.append(Polygon([
                (xs[col], ys[row]),
                (xs[col + 1], ys[row]),
                (xs[col + 1], ys[row + 1]),
                (xs[col], ys[row + 1]),
            ]))
    arr = np.asarray(geoms, dtype=object)
    return shapely.make_valid(arr)


def _convexish_polygon(rng, cx: float, cy: float, radius: float, vertices: int) -> Polygon:
    angles = np.sort(rng.uniform(0.0, 2 * math.pi, vertices))
    scales = rng.uniform(0.5, 1.0, vertices)
    coords = [
        (cx + radius * s * math.cos(a), cy + radius * s * math.sin(a))
        for a, s in zip(angles, scales)
    ]
    return Polygon(coords)


def _make_convex_polygons(
    count: int, seed: int, *, clusters: int = 8, vertices: int = 6,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1000.0, 1000.0),
) -> np.ndarray:
    count = max(count, 1)
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bounds
    # Generate cluster centers for polygon placement
    centers = _make_clustered_points(count, seed, clusters=clusters, bounds=bounds)
    radius = min(xmax - xmin, ymax - ymin) / max(_grid_side(count) * 3, 2)
    geoms = []
    for pt in centers:
        cx, cy = shapely.get_x(pt), shapely.get_y(pt)
        geoms.append(_convexish_polygon(rng, cx, cy, radius, vertices))
    arr = np.asarray(geoms, dtype=object)
    return shapely.make_valid(arr)


def _star_polygon(cx: float, cy: float, outer_r: float, inner_r: float, vertices: int) -> Polygon:
    coords = []
    for i in range(vertices * 2):
        angle = math.pi * i / vertices
        r = outer_r if i % 2 == 0 else inner_r
        coords.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return Polygon(coords)


def _make_star_polygons(
    count: int, seed: int, *, vertices: int = 10,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1000.0, 1000.0),
) -> np.ndarray:
    count = max(count, 1)
    xmin, ymin, xmax, ymax = bounds
    side = _grid_side(count)
    xs = _linspace(xmin, xmax, side + 1)
    ys = _linspace(ymin, ymax, side + 1)
    radius = min(xmax - xmin, ymax - ymin) / max(side * 4, 2)
    geoms = []
    for row in range(side):
        for col in range(side):
            if len(geoms) >= count:
                break
            cx = (xs[col] + xs[min(col + 1, len(xs) - 1)]) / 2
            cy = (ys[row] + ys[min(row + 1, len(ys) - 1)]) / 2
            geoms.append(_star_polygon(cx, cy, radius, radius / 2.5, vertices))
    arr = np.asarray(geoms, dtype=object)
    return shapely.make_valid(arr)


def _make_river_lines(
    count: int, seed: int, *, vertices: int = 12,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1000.0, 1000.0),
) -> np.ndarray:
    count = max(count, 1)
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bounds
    xs = _linspace(xmin, xmax, vertices)
    amplitude = (ymax - ymin) / 8.0
    geoms = []
    for offset in rng.uniform(ymin, ymax, count):
        phase = rng.uniform(0.0, 2 * math.pi)
        coords = [
            (float(x), float(np.clip(offset + amplitude * math.sin(phase + i / 2.0), ymin, ymax)))
            for i, x in enumerate(xs)
        ]
        geoms.append(LineString(coords))
    return np.asarray(geoms, dtype=object)


def _make_grid_lines(
    count: int, seed: int, *,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1000.0, 1000.0),
) -> np.ndarray:
    count = max(count, 1)
    xmin, ymin, xmax, ymax = bounds
    side = _grid_side(count)
    xs = _linspace(xmin, xmax, side + 1)
    ys = _linspace(ymin, ymax, side + 1)
    geoms = []
    for i in range(count):
        if i % 2 == 0:
            y = ys[(i // 2) % len(ys)]
            geoms.append(LineString([(xmin, float(y)), (xmax, float(y))]))
        else:
            x = xs[(i // 2) % len(xs)]
            geoms.append(LineString([(float(x), ymin), (float(x), ymax)]))
    return np.asarray(geoms, dtype=object)


# ---------------------------------------------------------------------------
# Fixture file writers
# ---------------------------------------------------------------------------

def _write_parquet(gdf: gpd.GeoDataFrame, path: Path) -> None:
    gdf.to_parquet(path, geometry_encoding="WKB")


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    gdf.to_file(path, driver="GeoJSON")


def setup_fixtures(tmpdir: Path) -> dict[str, Path]:
    """Write all fixture files to tmpdir. Returns name -> path map."""
    scale = get_scale()
    paths: dict[str, Path] = {}

    # --- Lines (powerlines / network) ---
    line_count = max(scale // 50, 2)
    lines = _make_river_lines(line_count, seed=10, vertices=12)
    lines_gdf = gpd.GeoDataFrame(
        {"circuit_id": np.arange(len(lines), dtype=np.int32) % max(min(len(lines), 32), 1),
         "geometry": lines},
        geometry="geometry", crs="EPSG:4326",
    )
    paths["lines"] = tmpdir / "lines.parquet"
    _write_parquet(lines_gdf, paths["lines"])

    # --- Vegetation patches ---
    veg_count = max(scale // 10, 2)
    veg = _make_convex_polygons(veg_count, seed=11, clusters=8, vertices=6)
    veg_gdf = gpd.GeoDataFrame(
        {"species": np.arange(len(veg), dtype=np.int32) % 5,
         "geometry": veg},
        geometry="geometry", crs="EPSG:4326",
    )
    paths["vegetation"] = tmpdir / "vegetation.parquet"
    _write_parquet(veg_gdf, paths["vegetation"])

    # --- Utility poles (GeoJSON) ---
    pole_count = scale
    poles = _make_clustered_points(pole_count, seed=12, clusters=12)
    poles_gdf = gpd.GeoDataFrame(
        {"pole_type": np.arange(len(poles), dtype=np.int32) % 3,
         "geometry": poles},
        geometry="geometry", crs="EPSG:4326",
    )
    paths["poles"] = tmpdir / "poles.geojson"
    _write_geojson(poles_gdf, paths["poles"])

    # --- Parcels ---
    parcels = _make_grid_polygons(scale, seed=20)
    parcels_gdf = gpd.GeoDataFrame(
        {"parcel_id": np.arange(len(parcels), dtype=np.int64),
         "geometry": parcels},
        geometry="geometry", crs="EPSG:4326",
    )
    paths["parcels"] = tmpdir / "parcels.parquet"
    _write_parquet(parcels_gdf, paths["parcels"])

    # --- Zoning polygons ---
    zone_count = max(scale // 100, 2)
    zones = _make_convex_polygons(zone_count, seed=21, clusters=4, vertices=8)
    zones_gdf = gpd.GeoDataFrame(
        {"zone_type": np.arange(len(zones), dtype=np.int32) % 4,
         "geometry": zones},
        geometry="geometry", crs="EPSG:4326",
    )
    paths["zones"] = tmpdir / "zones.parquet"
    _write_parquet(zones_gdf, paths["zones"])

    # --- Buildings ---
    buildings = _make_grid_polygons(scale, seed=30)
    buildings_gdf = gpd.GeoDataFrame(
        {"building_id": np.arange(len(buildings), dtype=np.int64),
         "geometry": buildings},
        geometry="geometry", crs="EPSG:4326",
    )
    paths["buildings"] = tmpdir / "buildings.parquet"
    _write_parquet(buildings_gdf, paths["buildings"])

    # --- Flood zones (GeoJSON) ---
    flood_count = max(scale // 500, 4)
    flood = _make_star_polygons(flood_count, seed=31, vertices=10)
    flood_gdf = gpd.GeoDataFrame(
        {"zone_id": np.arange(len(flood), dtype=np.int64),
         "geometry": flood},
        geometry="geometry", crs="EPSG:4326",
    )
    paths["flood_zones"] = tmpdir / "flood_zones.geojson"
    _write_geojson(flood_gdf, paths["flood_zones"])

    # --- Network lines (grid pattern) ---
    net_count = max(scale // 10, 2)
    net_lines = _make_grid_lines(net_count, seed=40)
    net_gdf = gpd.GeoDataFrame(
        {"segment_id": np.arange(len(net_lines), dtype=np.int64),
         "geometry": net_lines},
        geometry="geometry", crs="EPSG:4326",
    )
    paths["network"] = tmpdir / "network.parquet"
    _write_parquet(net_gdf, paths["network"])

    # --- Admin boundary (GeoJSON, single polygon) ---
    admin = _make_star_polygons(1, seed=41, vertices=12,
                                bounds=(100.0, 100.0, 900.0, 900.0))
    admin_gdf = gpd.GeoDataFrame(
        {"admin_name": ["Region A"], "geometry": admin},
        geometry="geometry", crs="EPSG:4326",
    )
    paths["admin_boundary"] = tmpdir / "admin_boundary.geojson"
    _write_geojson(admin_gdf, paths["admin_boundary"])

    # --- Exclusion zones ---
    excl_count = max(scale // 20, 2)
    excl = _make_convex_polygons(excl_count, seed=50, clusters=6, vertices=8)
    excl_gdf = gpd.GeoDataFrame(
        {"exclusion_type": np.arange(len(excl), dtype=np.int32) % 3,
         "geometry": excl},
        geometry="geometry", crs="EPSG:4326",
    )
    paths["exclusion_zones"] = tmpdir / "exclusion_zones.parquet"
    _write_parquet(excl_gdf, paths["exclusion_zones"])

    # --- Transit stations (GeoJSON) ---
    transit_count = max(scale // 5, 2)
    transit = _make_clustered_points(transit_count, seed=51, clusters=8)
    transit_gdf = gpd.GeoDataFrame(
        {"station_id": np.arange(len(transit), dtype=np.int64),
         "geometry": transit},
        geometry="geometry", crs="EPSG:4326",
    )
    paths["transit"] = tmpdir / "transit.geojson"
    _write_geojson(transit_gdf, paths["transit"])

    return paths


def fingerprint(gdf: gpd.GeoDataFrame) -> str:
    """Deterministic summary for correctness comparison across engines."""
    rows = len(gdf)
    b = tuple(round(float(v), 2) for v in gdf.total_bounds)
    # The benchmark fingerprint runs after the timed body, so we can afford
    # to materialize a host-safe geometry snapshot explicitly here rather than
    # letting strict-native fingerprinting stumble into a CPU fallback on a
    # lazy device-backed series.
    geoms = np.asarray(gdf.geometry.values._data, dtype=object)
    hulls = shapely.convex_hull(geoms)
    area = float(np.asarray(shapely.area(hulls), dtype=np.float64).sum())
    return f"rows={rows} bounds={b} convex_hull_area={area:.2f}"
