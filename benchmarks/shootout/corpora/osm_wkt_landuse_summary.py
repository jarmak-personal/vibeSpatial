"""Real WKT ingest, GeoDataFrame assembly, explode, dissolve, and write."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _common import fingerprint, get_scale, require_asset

import geopandas as gpd

source = require_asset("osm_landuse_american_oceania")
scale = get_scale()
_tmpdir = tempfile.TemporaryDirectory(prefix="vsbench_osm_wkt_")
output_path = Path(_tmpdir.name) / "summary.parquet"

# --- timed work starts here ---
table = pd.read_parquet(source).head(scale)
geometry = gpd.GeoSeries.from_wkt(table.pop("geometry_wkt"), crs=4326)
areas = gpd.GeoDataFrame(table, geometry=geometry, crs=4326)
areas = areas.loc[areas["extract_status"] == "clean"]
parts = areas.explode(index_parts=False, ignore_index=True)
summary = parts.dissolve(
    by="size_bin",
    aggfunc={"osm_id": "count", "area_km2": "sum", "centroid_lon": "mean"},
)
summary.to_parquet(output_path)
# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
