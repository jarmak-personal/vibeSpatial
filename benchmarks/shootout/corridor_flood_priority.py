"""Corridor flood-priority screening.

Read network lines, vegetation patches, flood zones, utility poles, and
an admin boundary. Buffer and dissolve the network into a corridor,
clip to the admin area, intersect with vegetation and flood polygons,
then spatial-join nearby poles.

Usage:
    vsbench shootout benchmarks/shootout/corridor_flood_priority.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures

import geopandas as gpd

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_corridor_flood_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

network = gpd.read_parquet(fixtures["network"])
vegetation = gpd.read_parquet(fixtures["vegetation"])
flood_zones = gpd.read_file(fixtures["flood_zones"])
poles = gpd.read_file(fixtures["poles"])
admin = gpd.read_file(fixtures["admin_boundary"])

network["geometry"] = network.geometry.buffer(18.0)
network["group"] = 0
corridor = network.dissolve(by="group").reset_index(drop=True)

admin_corridor = gpd.clip(corridor, admin)
admin_corridor = admin_corridor[admin_corridor.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

veg_in_corridor = gpd.overlay(
    vegetation[["species", "geometry"]],
    admin_corridor[["geometry"]],
    how="intersection",
)

priority = gpd.overlay(
    veg_in_corridor,
    flood_zones[["zone_id", "geometry"]],
    how="intersection",
)

if len(priority) > 0:
    priority_buffers = priority.copy()
    priority_buffers["geometry"] = priority_buffers.geometry.buffer(12.0)
    pole_hits = gpd.sjoin(
        poles[["pole_type", "geometry"]],
        priority_buffers[["geometry"]],
        predicate="intersects",
    )
    keep_rows = pole_hits.index.unique()
    near_priority = poles.loc[keep_rows].copy() if len(keep_rows) > 0 else poles.iloc[:0].copy()
else:
    near_priority = poles.iloc[:0].copy()

output_path = tmpdir / "output.parquet"
priority.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
