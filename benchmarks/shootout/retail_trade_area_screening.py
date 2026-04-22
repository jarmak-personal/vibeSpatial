"""Retail trade-area site screening.

Read parcels, transit stops, road centerlines, and competitor points.
Keep commercially viable parcels, require transit access and road
frontage, exclude parcels close to competitors, then dissolve remaining
candidate sites into reporting submarkets.

Usage:
    vsbench shootout benchmarks/shootout/retail_trade_area_screening.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures

import geopandas as gpd

POLYGONAL_TYPES = ["Polygon", "MultiPolygon"]
TRANSIT_ACCESS_DISTANCE = 120.0
ROAD_FRONTAGE_DISTANCE = 45.0
COMPETITOR_EXCLUSION_DISTANCE = 90.0

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_retail_trade_area_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

parcels = gpd.read_parquet(fixtures["parcels"])
transit = gpd.read_file(fixtures["transit"])
roads = gpd.read_parquet(fixtures["network"])
pois = gpd.read_file(fixtures["poles"])

parcels = parcels[parcels.geometry.geom_type.isin(POLYGONAL_TYPES)].copy()
parcels["zoning"] = parcels["parcel_id"] % 5
candidates = parcels[parcels["zoning"].isin([0, 1, 2])].copy()

transit_access = transit[["geometry"]].copy()
transit_access["geometry"] = transit_access.geometry.buffer(TRANSIT_ACCESS_DISTANCE)

frontage = roads[["geometry"]].copy()
frontage["geometry"] = frontage.geometry.buffer(ROAD_FRONTAGE_DISTANCE)
frontage["frontage_group"] = 0
frontage = frontage.dissolve(by="frontage_group").reset_index()

competitors = pois[pois["pole_type"] == 1][["geometry"]].copy()
competitors["geometry"] = competitors.geometry.buffer(COMPETITOR_EXCLUSION_DISTANCE)

near_transit_rows = gpd.sjoin(
    candidates,
    transit_access,
    predicate="intersects",
).index.unique()
near_transit = candidates.loc[near_transit_rows].copy()

near_road_rows = gpd.sjoin(
    near_transit,
    frontage[["geometry"]],
    predicate="intersects",
).index.unique()
near_roads = near_transit.loc[near_road_rows].copy()

if len(competitors) > 0 and len(near_roads) > 0:
    competitor_hits = gpd.sjoin(
        near_roads,
        competitors,
        predicate="intersects",
    )
    safe_sites = near_roads.loc[
        ~near_roads.index.isin(competitor_hits.index.unique())
    ].copy()
else:
    safe_sites = near_roads.copy()

if len(safe_sites) > 0:
    safe_sites["submarket"] = (safe_sites["parcel_id"] // 250).astype(str)
    trade_areas = safe_sites.dissolve(
        by="submarket",
        aggfunc={"parcel_id": "count"},
    ).reset_index()
else:
    trade_areas = safe_sites.copy()

output_path = tmpdir / "output.parquet"
trade_areas.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
