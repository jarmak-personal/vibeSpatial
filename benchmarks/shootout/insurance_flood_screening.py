"""Insurance parcel flood screening.

Read assessor parcels, flood hazard polygons, and a county boundary.
Clip to the underwriting area, prefilter parcel/flood candidates with a
spatial join, compute exact flood overlap by overlay intersection, then
retain parcels whose inundated area ratio crosses an underwriting
threshold.

Usage:
    vsbench shootout benchmarks/shootout/insurance_flood_screening.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures

import geopandas as gpd

POLYGONAL_TYPES = ["Polygon", "MultiPolygon"]
FLOOD_RATIO_THRESHOLD = 0.15

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_insurance_flood_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

parcels = gpd.read_parquet(fixtures["parcels"])
flood_zones = gpd.read_file(fixtures["flood_zones"])
admin = gpd.read_file(fixtures["admin_boundary"])

parcels = gpd.clip(parcels, admin)
parcels = parcels[parcels.geometry.geom_type.isin(POLYGONAL_TYPES)].copy()
parcels["parcel_area"] = parcels.geometry.area

hits = gpd.sjoin(
    parcels[["parcel_id", "parcel_area", "geometry"]],
    flood_zones[["zone_id", "geometry"]],
    predicate="intersects",
)
candidate_rows = hits.index.unique()
candidate_parcels = parcels.loc[candidate_rows].copy()

if len(candidate_parcels) > 0:
    exposure = gpd.overlay(
        candidate_parcels[["parcel_id", "parcel_area", "geometry"]],
        flood_zones[["zone_id", "geometry"]],
        how="intersection",
    )
    exposure = exposure[exposure.geometry.geom_type.isin(POLYGONAL_TYPES)].copy()
else:
    exposure = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=parcels.crs)

if len(exposure) > 0:
    exposure["wet_area"] = exposure.geometry.area
    exposure["flood_ratio"] = exposure["wet_area"] / exposure["parcel_area"]
    risk = exposure[exposure["flood_ratio"] >= FLOOD_RATIO_THRESHOLD].copy()
    risk = risk.sort_values(["parcel_id", "flood_ratio"], ascending=[True, False])
    risk = risk.drop_duplicates("parcel_id")
else:
    risk = exposure.copy()

output_path = tmpdir / "output.parquet"
risk.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
