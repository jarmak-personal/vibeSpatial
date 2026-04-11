"""Flood exposure assessment.

Read building footprints and flood zone boundaries. Repair invalid
geometries, spatial-join to find at-risk buildings, filter to affected
rows, then buffer building centroids to create risk zones.

Usage:
    vsbench shootout benchmarks/shootout/flood_exposure.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures

import geopandas as gpd

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_flood_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

# Read
buildings = gpd.read_parquet(fixtures["buildings"])
flood_zones = gpd.read_file(fixtures["flood_zones"])

# Repair through the public GeoSeries path so warmed GPU-native make_valid
# stays on the device-native execution model before terminal export.
buildings["geometry"] = buildings.geometry.make_valid()

# Spatial join: buildings vs flood zones
joined = gpd.sjoin(buildings, flood_zones, predicate="intersects")

# Filter to at-risk buildings
hit_indices = joined.index.unique()
filtered = buildings.loc[hit_indices]

# Buffer building centroids to create risk zones
if len(filtered) > 0:
    risk_zones = filtered.copy()
    risk_zones["geometry"] = filtered.geometry.centroid.buffer(50.0)
else:
    risk_zones = filtered.copy()

# Write result
output_path = tmpdir / "output.parquet"
risk_zones.to_parquet(output_path)

# --- timed work ends here ---

# --- fingerprint (post-benchmark, not timed) ---
check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
