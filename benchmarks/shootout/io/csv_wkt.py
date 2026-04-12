"""Florida CSV WKT read-only shootout."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _florida import FORMATS_DIR, fingerprint, require_path

import geopandas as gpd

source = require_path(FORMATS_DIR / "Florida.wkt.csv")

# --- timed work starts here ---
table = pd.read_csv(source)
frame = gpd.GeoDataFrame(
    table.drop(columns=["geometry"]),
    geometry=gpd.GeoSeries.from_wkt(table["geometry"]),
    crs="EPSG:4326",
)
# --- timed work ends here ---

print(f"SHOOTOUT_FINGERPRINT: {fingerprint(frame)}")
