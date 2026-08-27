"""Real MultiPolygon read, frame reshaping, metric, grouped reduction, and write."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import fingerprint, get_scale, require_asset

import geopandas as gpd

source = require_asset("osm_multipolygon_algeria")
scale = get_scale()
_tmpdir = tempfile.TemporaryDirectory(prefix="vsbench_osm_multipolygon_")
output_path = Path(_tmpdir.name) / "summary.parquet"

# --- timed work starts here ---
areas = gpd.read_parquet(
    source,
    columns=["osm_id", "osm_type", "area_m2", "geometry"],
).head(scale)
areas = areas.sort_values(["osm_type", "area_m2"], ascending=[True, False], kind="stable")
projected = areas.to_crs(3857)
parts = projected.explode(index_parts=False, ignore_index=False)
parts["perimeter_m"] = parts.length
summary = parts.dissolve(
    by="osm_type",
    aggfunc={"osm_id": "count", "area_m2": "sum", "perimeter_m": "mean"},
)
summary.to_parquet(output_path)
# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
