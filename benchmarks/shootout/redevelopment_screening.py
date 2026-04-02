"""Transit-oriented redevelopment screening.

Read parcels, zoning polygons, exclusion zones, and transit stations.
Clip parcels to a study area, remove exclusion areas, keep parcels near
transit, intersect with zoning polygons, then dissolve by zone type.

Usage:
    vsbench shootout benchmarks/shootout/redevelopment_screening.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures
from shapely.geometry import box

import geopandas as gpd

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_redev_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

parcels = gpd.read_parquet(fixtures["parcels"])
zones = gpd.read_parquet(fixtures["zones"])
exclusions = gpd.read_parquet(fixtures["exclusion_zones"])
transit = gpd.read_file(fixtures["transit"])

bounds = parcels.total_bounds
dx = (bounds[2] - bounds[0]) * 0.15
dy = (bounds[3] - bounds[1]) * 0.15
clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)
study_parcels = gpd.clip(parcels, clip_box)

study_parcels = study_parcels[
    study_parcels.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
].copy()

developable = gpd.overlay(study_parcels, exclusions, how="difference")

transit_buffers = transit.copy()
transit_buffers["geometry"] = transit_buffers.geometry.buffer(150.0)

near_transit = gpd.sjoin(
    developable,
    transit_buffers[["station_id", "geometry"]],
    predicate="intersects",
)
candidate_rows = near_transit.index.unique()
candidates = (
    developable.loc[candidate_rows].copy()
    if len(candidate_rows) > 0
    else developable.iloc[:0].copy()
)

zoned = gpd.overlay(candidates, zones[["zone_type", "geometry"]], how="intersection")

if len(zoned) > 0:
    zoned["zone_group"] = zoned["zone_type"].astype(str)
    dissolved = zoned.dissolve(by="zone_group").reset_index()
else:
    dissolved = zoned.copy()

output_path = tmpdir / "output.parquet"
dissolved.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
