"""Powerline vegetation corridor monitoring.

Read powerline linestrings, vegetation polygons, and utility poles.
Buffer lines to create maintenance corridor, dissolve into a single
polygon, intersect with vegetation, then find poles near clipped
vegetation patches.

Usage:
    vsbench shootout benchmarks/shootout/vegetation_corridor.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import shapely
from _data import fingerprint, setup_fixtures

import geopandas as gpd

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_vegcorr_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

# Read
lines = gpd.read_parquet(fixtures["lines"])
vegetation = gpd.read_parquet(fixtures["vegetation"])
poles = gpd.read_file(fixtures["poles"])

# Buffer lines to create corridor
lines["geometry"] = lines.geometry.buffer(10.0)

# Dissolve into single corridor polygon
lines["group"] = 0
dissolved = lines.dissolve(by="group")

# make_valid on both sides before overlay
dissolved["geometry"] = shapely.make_valid(dissolved.geometry.values)
vegetation["geometry"] = shapely.make_valid(vegetation.geometry.values)

# Intersect vegetation with corridor
try:
    clipped = gpd.overlay(vegetation, dissolved[["geometry"]], how="intersection")
except Exception:
    # Fallback: vectorized intersection when overlay hits GEOS
    # TopologyException or IllegalArgumentException at scale.
    corridor_geom = dissolved.geometry.values[0]
    veg_arr = np.asarray(vegetation.geometry.values, dtype=object)
    corridor_arr = np.full(len(veg_arr), corridor_geom, dtype=object)
    try:
        intersected = shapely.intersection(veg_arr, corridor_arr)
        keep = ~shapely.is_empty(intersected) & ~shapely.is_missing(intersected)
        results = list(intersected[keep])
    except Exception:
        results = []
    clipped = gpd.GeoDataFrame(
        {"geometry": results if results else []},
        geometry="geometry",
        crs=vegetation.crs,
    )

# Find poles near clipped vegetation
if len(clipped) > 0:
    buffered_veg = clipped.copy()
    buffered_veg["geometry"] = clipped.geometry.centroid.buffer(1.0)
    joined = gpd.sjoin(poles, buffered_veg[["geometry"]], predicate="within")
else:
    joined = poles.iloc[:0]

# Write result
output_path = tmpdir / "output.parquet"
clipped.to_parquet(output_path)

# --- timed work ends here ---

# --- fingerprint (post-benchmark, not timed) ---
check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
