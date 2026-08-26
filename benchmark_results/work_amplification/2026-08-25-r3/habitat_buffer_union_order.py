#!/usr/bin/env python3
"""Public-API counterfactual for scalar line-buffer/union ordering."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "shootout"))

from _data import fingerprint, setup_fixtures

import geopandas as gpd

POLYGONAL_TYPES = ["Polygon", "MultiPolygon"]
RIPARIAN_SETBACK_DISTANCE = 35.0
VARIANT = os.environ.get("VIBESPATIAL_R3_VARIANT", "buffer_then_union")
if VARIANT not in {"buffer_then_union", "union_then_buffer"}:
    raise ValueError(f"unsupported R3 variant: {VARIANT}")

_tmpdir = tempfile.TemporaryDirectory(prefix="r3_habitat_buffer_union_")
tmpdir = Path(_tmpdir.name)
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

waterways["corridor_id"] = 0
if VARIANT == "buffer_then_union":
    riparian = waterways[["corridor_id", "geometry"]].copy()
    riparian["geometry"] = riparian.geometry.buffer(RIPARIAN_SETBACK_DISTANCE)
    riparian = riparian.dissolve(by="corridor_id").reset_index()
else:
    riparian = waterways[["corridor_id", "geometry"]].dissolve(
        by="corridor_id"
    ).reset_index()
    riparian["geometry"] = riparian.geometry.buffer(RIPARIAN_SETBACK_DISTANCE)

if len(parcels) > 0 and len(riparian) > 0:
    corridor_parcels = gpd.clip(parcels, riparian)
    corridor_parcels = corridor_parcels[
        corridor_parcels.geometry.geom_type.isin(POLYGONAL_TYPES)
    ].copy()
else:
    corridor_parcels = gpd.GeoDataFrame(
        {"geometry": []}, geometry="geometry", crs=parcels.crs
    )

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

output_path = Path(os.environ.get("VIBESPATIAL_R3_OUTPUT", tmpdir / "output.parquet"))
summary.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"R3_VARIANT: {VARIANT}")
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
