"""Tabular point construction, reprojection, nearest join, shaping, and write."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _common import (
    fingerprint,
    get_scale,
    power_nearest_correctness_packet,
    require_asset,
)

import geopandas as gpd

source = require_asset("osm_power_substations")
scale = get_scale()
_tmpdir = tempfile.TemporaryDirectory(prefix="vsbench_power_nearest_")
output_path = Path(_tmpdir.name) / "nearest.parquet"

# --- timed work starts here ---
table = pd.read_parquet(source).dropna(subset=["lon", "lat"]).head(scale)
substations = gpd.GeoDataFrame(
    table[["node_id", "power_value", "voltage", "operator", "name"]],
    geometry=gpd.points_from_xy(table["lon"], table["lat"], crs=4326),
    crs=4326,
).to_crs(3857)
left = substations.iloc[::2]
right = substations.iloc[1::2][["node_id", "geometry"]]
nearest = gpd.sjoin_nearest(
    left,
    right,
    how="inner",
    max_distance=100_000,
    distance_col="distance_m",
)
nearest = nearest.sort_values(
    ["node_id_left", "distance_m", "node_id_right"], kind="stable"
).drop_duplicates(subset="node_id_left", keep="first")
nearest.to_parquet(output_path)
# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
print(
    "SHOOTOUT_CORRECTNESS: "
    + json.dumps(power_nearest_correctness_packet(check), separators=(",", ":"))
)
