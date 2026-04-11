"""Parcel-zoning compliance check.

Read land parcels and zoning boundaries. Clip parcels to a study area,
spatial-join against zoning polygons, then overlay intersection to find
exact parcel-zone overlaps.

Usage:
    vsbench shootout benchmarks/shootout/parcel_zoning.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures
from shapely.geometry import box

import geopandas as gpd

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_parcel_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

# Read
parcels = gpd.read_parquet(fixtures["parcels"])
zones = gpd.read_parquet(fixtures["zones"])

# Clip parcels to 60% study area
bounds = parcels.total_bounds
dx = (bounds[2] - bounds[0]) * 0.2
dy = (bounds[3] - bounds[1]) * 0.2
clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)
clipped = gpd.clip(parcels, clip_box)

# Filter to polygonal types once before the relational steps. The parcel-zone
# overlap workload only consumes polygonal fragments downstream, and overlay
# rejects mixed geometry families anyway.
poly_mask = clipped.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
clipped_poly = clipped[poly_mask] if not poly_mask.all() else clipped

# Spatial join: parcels vs zones
joined = gpd.sjoin(clipped_poly, zones, predicate="intersects")

# Overlay intersection
overlaid = gpd.overlay(clipped_poly, zones, how="intersection")

# Write result
output_path = tmpdir / "output.parquet"
overlaid.to_parquet(output_path)

# --- timed work ends here ---

# --- fingerprint (post-benchmark, not timed) ---
check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
