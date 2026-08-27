"""Attributed points, variable buffer, spatial join, relation shaping, and write."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _common import fingerprint, get_scale, require_asset

import geopandas as gpd

source = require_asset("cmab_beijing_level1")
scale = get_scale()
_tmpdir = tempfile.TemporaryDirectory(prefix="vsbench_cmab_influence_")
output_path = Path(_tmpdir.name) / "matches.parquet"

# --- timed work starts here ---
table = pd.read_parquet(source).head(scale)
buildings = gpd.GeoDataFrame(
    table[["building_uid", "func", "func_code", "d_m", "height_m"]],
    geometry=gpd.points_from_xy(table["cx"], table["cy"]),
)
influence = gpd.GeoDataFrame(
    table[["building_uid", "func_code"]],
    geometry=buildings.geometry.buffer(table["d_m"]),
)
matches = gpd.sjoin(
    buildings,
    influence,
    predicate="within",
    how="inner",
    on_attribute="func_code",
)
matches = matches.sort_values(
    ["building_uid_left", "building_uid_right"], kind="stable"
).drop_duplicates(subset="building_uid_left", keep="first")
matches.to_parquet(output_path)
# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
