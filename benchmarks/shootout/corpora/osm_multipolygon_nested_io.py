"""Full nested-attribute GeoParquet read and write compatibility canary."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import fingerprint, get_scale, require_asset

import geopandas as gpd

source = require_asset("osm_multipolygon_algeria")
scale = get_scale()
_tmpdir = tempfile.TemporaryDirectory(prefix="vsbench_osm_nested_io_")
output_path = Path(_tmpdir.name) / "nested.parquet"

# --- timed work starts here ---
areas = gpd.read_parquet(source).head(scale)
areas.to_parquet(output_path)
# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
