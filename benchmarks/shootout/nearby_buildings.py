"""Nearby buildings search with CRS reprojection.

Read realistic WGS84 building footprints, estimate the local UTM CRS,
reproject through the public API, find buildings within 1 km of a
deterministic seed building, then export the result back to WGS84.

Usage:
    vsbench shootout benchmarks/shootout/nearby_buildings.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures

import geopandas as gpd

SEARCH_RADIUS_M = 1_000

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_nearby_buildings_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

buildings = gpd.read_file(fixtures["nearby_buildings"])

utm_crs = buildings.geometry.estimate_utm_crs()
buildings_utm = buildings.to_crs(utm_crs)

seed_idx = len(buildings_utm) // 2
seed_building = buildings_utm.geometry.iloc[seed_idx]
seed_centroid = seed_building.centroid

mask = buildings_utm.geometry.dwithin(seed_centroid, SEARCH_RADIUS_M)
nearby = buildings_utm[mask].copy()

nearby_wgs84 = nearby.to_crs(epsg=4326)

output_path = tmpdir / "output.parquet"
nearby_wgs84.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
