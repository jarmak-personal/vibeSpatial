"""Riparian habitat compliance screening.

Read parcels, waterways, habitat patches, protected areas, and an
administrative boundary. Build a riparian setback corridor, clip parcels
to that corridor, spatially join against habitat and protected polygons,
then dissolve impacted parcels into compliance reporting regions.

Usage:
    vsbench shootout benchmarks/shootout/habitat_corridor_compliance.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures

import geopandas as gpd

POLYGONAL_TYPES = ["Polygon", "MultiPolygon"]
RIPARIAN_SETBACK_DISTANCE = 35.0

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_habitat_corridor_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

parcels = gpd.read_parquet(fixtures["parcels"])
waterways = gpd.read_parquet(fixtures["lines"])
habitat = gpd.read_parquet(fixtures["vegetation"])
protected = gpd.read_parquet(fixtures["exclusion_zones"])
admin = gpd.read_file(fixtures["admin_boundary"])

parcels = gpd.clip(parcels, admin)
parcels = parcels[parcels.geometry.geom_type.isin(POLYGONAL_TYPES)].copy()
parcels["reporting_region"] = (parcels["parcel_id"] % 7).astype(str)

riparian = waterways[["geometry"]].copy()
riparian["geometry"] = riparian.geometry.buffer(RIPARIAN_SETBACK_DISTANCE)
riparian["corridor_id"] = 0
riparian = riparian.dissolve(by="corridor_id").reset_index()

if len(parcels) > 0 and len(riparian) > 0:
    corridor_parcels = gpd.clip(parcels, riparian)
    corridor_parcels = corridor_parcels[
        corridor_parcels.geometry.geom_type.isin(POLYGONAL_TYPES)
    ].copy()
else:
    corridor_parcels = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=parcels.crs)

if len(corridor_parcels) > 0:
    habitat_hits = gpd.sjoin(
        corridor_parcels,
        habitat[["species", "geometry"]],
        predicate="intersects",
    )
    habitat_parcels = corridor_parcels.loc[habitat_hits.index.unique()].copy()
else:
    habitat_parcels = corridor_parcels.copy()

if len(habitat_parcels) > 0:
    protected_hits = gpd.sjoin(
        habitat_parcels,
        protected[["exclusion_type", "geometry"]],
        predicate="intersects",
    )
    impacted = habitat_parcels.loc[protected_hits.index.unique()].copy()
else:
    impacted = habitat_parcels.copy()

if len(impacted) > 0:
    impacted["impacted_area"] = impacted.geometry.area
    summary = impacted.dissolve(
        by="reporting_region",
        aggfunc={"parcel_id": "count", "impacted_area": "sum"},
    ).reset_index()
else:
    summary = impacted.copy()

output_path = tmpdir / "output.parquet"
summary.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
