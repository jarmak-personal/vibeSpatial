"""Transit-access redevelopment screening.

Read realistic WGS84 buildings, parcels, transit stops, exclusions, and
an admin boundary. Estimate a local UTM CRS, reproject via the public
API, clip to the study area, attach nearest transit to building
centroids, subtract exclusions from parcels, keep transit-served
parcels, intersect with nearby buildings, then dissolve, explode,
simplify, and convex-hull summarize the result before exporting back to
WGS84.

Usage:
    vsbench shootout benchmarks/shootout/accessibility_redevelopment.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures

import geopandas as gpd

POLYGONAL_TYPES = ["Polygon", "MultiPolygon"]
MAX_NEAREST_DISTANCE_M = 1_800.0
TRANSIT_BUFFER_M = 900.0
SIMPLIFY_TOLERANCE_M = 25.0

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_accessibility_redevelopment_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

buildings = gpd.read_file(fixtures["access_buildings"])
parcels = gpd.read_parquet(fixtures["access_parcels"])
transit = gpd.read_file(fixtures["access_transit"])
exclusions = gpd.read_parquet(fixtures["access_exclusions"])
admin = gpd.read_file(fixtures["access_admin_boundary"])

utm_crs = buildings.geometry.estimate_utm_crs()
buildings = buildings.to_crs(utm_crs)
parcels = parcels.to_crs(utm_crs)
transit = transit.to_crs(utm_crs)
exclusions = exclusions.to_crs(utm_crs)
admin = admin.to_crs(utm_crs)

buildings = gpd.clip(buildings, admin)
parcels = gpd.clip(parcels, admin)
buildings = buildings[buildings.geometry.geom_type.isin(POLYGONAL_TYPES)].copy()
parcels = parcels[parcels.geometry.geom_type.isin(POLYGONAL_TYPES)].copy()

building_points = buildings[["building_id", "geometry"]].copy()
building_points["geometry"] = buildings.geometry.centroid

nearest = building_points.sjoin_nearest(
    transit[["station_id", "geometry"]],
    how="inner",
    max_distance=MAX_NEAREST_DISTANCE_M,
    distance_col="station_distance_m",
)
nearest = nearest.sort_values(
    ["building_id", "station_distance_m", "station_id"]
).drop_duplicates("building_id")

nearby_building_ids = nearest.loc[
    nearest["station_distance_m"] <= MAX_NEAREST_DISTANCE_M,
    "building_id",
].drop_duplicates()
nearby_buildings = buildings[
    buildings["building_id"].isin(nearby_building_ids)
].copy()

transit_buffers = transit.copy()
transit_buffers["geometry"] = transit_buffers.geometry.buffer(TRANSIT_BUFFER_M)

developable = gpd.overlay(parcels, exclusions, how="difference")
developable = developable[
    developable.geometry.geom_type.isin(POLYGONAL_TYPES)
].copy()

served = gpd.sjoin(
    developable,
    transit_buffers[["station_id", "geometry"]],
    predicate="intersects",
)
served_rows = served.index.unique()
served_parcels = (
    developable.loc[served_rows].copy()
    if len(served_rows) > 0
    else developable.iloc[:0].copy()
)

if len(nearby_buildings) > 0 and len(served_parcels) > 0:
    occupied = gpd.sjoin(
        served_parcels[["parcel_id", "geometry"]],
        nearby_buildings[["building_id", "geometry"]],
        predicate="intersects",
    )
    occupied_rows = occupied.index.unique()
    occupied = (
        served_parcels.loc[occupied_rows].copy()
        if len(occupied_rows) > 0
        else served_parcels.iloc[:0].copy()
    )
else:
    occupied = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=parcels.crs)

if len(occupied) > 0:
    occupied["parcel_group"] = (occupied["parcel_id"] % 12).astype(str)
    station_zones = occupied.dissolve(by="parcel_group").reset_index()
    station_zones = station_zones.explode(ignore_index=True)
    station_zones["geometry"] = station_zones.geometry.simplify(
        SIMPLIFY_TOLERANCE_M,
        preserve_topology=True,
    )
    station_zones["geometry"] = station_zones.geometry.convex_hull
else:
    station_zones = occupied.copy()

output = station_zones.to_crs(epsg=4326)
output_path = tmpdir / "output.parquet"
output.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
