"""Large Parquet ingest, filtering, point construction, grouping, and write."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _common import fingerprint, get_scale, require_asset

import geopandas as gpd

source = require_asset("geolife_level1_part3")
scale = get_scale()
_tmpdir = tempfile.TemporaryDirectory(prefix="vsbench_geolife_tracks_")
output_path = Path(_tmpdir.name) / "tracks.parquet"

# --- timed work starts here ---
table = pd.read_parquet(source).head(scale)
table = table.loc[table["t_max_ms"] >= table["t_min_ms"]].sort_values(
    ["traj_id", "point_idx", "t_min_ms"], kind="stable"
)
points = gpd.GeoDataFrame(
    table[["rect_id", "traj_id", "user_id", "point_idx", "t_min_ms", "t_max_ms"]],
    geometry=gpd.points_from_xy(
        (table["x_min_cm"] + table["x_max_cm"]) / 200.0,
        (table["y_min_cm"] + table["y_max_cm"]) / 200.0,
    ),
)
tracks = points.dissolve(
    by="traj_id",
    aggfunc={"rect_id": "count", "user_id": "first", "t_min_ms": "min", "t_max_ms": "max"},
)
tracks.to_parquet(output_path)
# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
