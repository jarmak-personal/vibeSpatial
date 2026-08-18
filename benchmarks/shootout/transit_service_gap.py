"""Transit service gap analysis.

Read buildings, parcels, transit stations, and an admin boundary.
Clip buildings to the admin area, buffer transit stops, find served
buildings, turn unserved building centroids into gap polygons, then
intersect those polygons with parcel boundaries.

Usage:
    vsbench shootout benchmarks/shootout/transit_service_gap.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures, spatial_antijoin

import geopandas as gpd

_tmpdir = tempfile.TemporaryDirectory(prefix="shootout_transit_gap_")
tmpdir = Path(_tmpdir.name)
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

buildings = gpd.read_parquet(fixtures["buildings"])
parcels = gpd.read_parquet(fixtures["parcels"])
transit = gpd.read_file(fixtures["transit"])
admin = gpd.read_file(fixtures["admin_boundary"])

buildings_in_admin = gpd.clip(buildings, admin)

transit_buffers = transit.copy()
transit_buffers["geometry"] = transit_buffers.geometry.buffer(125.0)

unserved = spatial_antijoin(
    buildings_in_admin,
    transit_buffers[["station_id", "geometry"]],
    predicate="intersects",
)

if len(unserved) > 0:
    gap_polygons = unserved.copy()
    gap_polygons["geometry"] = unserved.geometry.centroid.buffer(35.0)
    parcel_gaps = gpd.overlay(
        gap_polygons[["geometry"]],
        parcels[["parcel_id", "geometry"]],
        how="intersection",
    )
else:
    parcel_gaps = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=buildings.crs)

output_path = tmpdir / "output.parquet"
parcel_gaps.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
