"""Emergency response catchment analysis.

Read synthetic census blocks, incident points, emergency stations, and
an administrative boundary. Build station catchments, assign incidents
to catchments, intersect catchments with blocks, then dissolve covered
population districts for a response-planning summary.

Usage:
    vsbench shootout benchmarks/shootout/emergency_response_catchments.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures

import geopandas as gpd

POLYGONAL_TYPES = ["Polygon", "MultiPolygon"]
MAX_STATIONS = 48
CATCHMENT_DISTANCE = 110.0

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_emergency_response_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

blocks = gpd.read_parquet(fixtures["parcels"])
incidents = gpd.read_file(fixtures["poles"])
stations = gpd.read_file(fixtures["transit"])
admin = gpd.read_file(fixtures["admin_boundary"])

blocks = gpd.clip(blocks, admin)
blocks = blocks[blocks.geometry.geom_type.isin(POLYGONAL_TYPES)].copy()
blocks["district"] = (blocks["parcel_id"] % 9).astype(str)

incidents = incidents.copy()
incidents["incident_id"] = incidents.index.astype("int64")

station_step = max(len(stations) // MAX_STATIONS, 1)
stations = stations.iloc[::station_step].head(MAX_STATIONS).copy()
catchments = stations[["station_id", "geometry"]].copy()
catchments["geometry"] = catchments.geometry.buffer(CATCHMENT_DISTANCE)

served = gpd.sjoin(
    incidents[["incident_id", "geometry"]],
    catchments[["station_id", "geometry"]],
    predicate="within",
)
served_incident_rows = served.index.unique()
served_incidents = incidents.loc[served_incident_rows].copy()

if len(blocks) > 0 and len(catchments) > 0:
    coverage = gpd.overlay(
        blocks[["parcel_id", "district", "geometry"]],
        catchments[["station_id", "geometry"]],
        how="intersection",
    )
    coverage = coverage[coverage.geometry.geom_type.isin(POLYGONAL_TYPES)].copy()
else:
    coverage = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=blocks.crs)

if len(coverage) > 0:
    coverage["covered_area"] = coverage.geometry.area
    district_coverage = coverage.dissolve(
        by="district",
        aggfunc={"parcel_id": "count", "covered_area": "sum"},
    ).reset_index()
    district_coverage["geometry"] = district_coverage.geometry.convex_hull
else:
    district_coverage = coverage.copy()

district_coverage["served_incidents"] = len(served_incidents)

output_path = tmpdir / "output.parquet"
district_coverage.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
