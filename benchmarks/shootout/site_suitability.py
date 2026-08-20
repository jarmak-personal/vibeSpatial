"""Site suitability analysis.

Read land parcels, environmental exclusion zones, and transit stations.
Clip parcels to a study area, remove exclusion zones via overlay
difference, buffer transit stations, then spatial-join to find
suitable parcels near transit.

Usage:
    vsbench shootout benchmarks/shootout/site_suitability.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_site_suitability_fixtures, spatial_semijoin
from shapely.geometry import box

import geopandas as gpd

_tmpdir = tempfile.TemporaryDirectory(prefix="shootout_site_")
tmpdir = Path(_tmpdir.name)
fixtures = setup_site_suitability_fixtures(tmpdir)

# --- timed work starts here ---

# Read
parcels = gpd.read_parquet(fixtures["parcels"])
exclusions = gpd.read_parquet(fixtures["exclusion_zones"])
transit = gpd.read_file(fixtures["transit"])

# Clip parcels to 60% study area
bounds = parcels.total_bounds
dx = (bounds[2] - bounds[0]) * 0.2
dy = (bounds[3] - bounds[1]) * 0.2
clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)
clipped = gpd.clip(parcels, clip_box)

# Filter to polygonal types before overlay
poly_mask = clipped.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
clipped_poly = clipped[poly_mask] if not poly_mask.all() else clipped

# Remove exclusion zones
suitable = gpd.overlay(clipped_poly, exclusions, how="difference")

# Buffer transit stations and keep each suitable parcel once when any station
# buffer intersects it. vibeSpatial lowers this public existential join to a
# bounded device selection; GeoPandas uses the equivalent public spatial join.
transit_access = transit[["geometry"]].copy()
transit_access["geometry"] = transit_access.geometry.buffer(200.0)
joined = spatial_semijoin(
    suitable,
    transit_access,
    predicate="intersects",
)

# Write result
output_path = tmpdir / "output.parquet"
joined.to_parquet(output_path)

# --- timed work ends here ---

# --- fingerprint (post-benchmark, not timed) ---
check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
