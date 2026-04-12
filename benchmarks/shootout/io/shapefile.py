"""Florida Shapefile read-only shootout."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _florida import FORMATS_DIR, fingerprint, require_path

import geopandas as gpd

source = require_path(FORMATS_DIR / "Florida.shp")

# --- timed work starts here ---
frame = gpd.read_file(source)
# --- timed work ends here ---

print(f"SHOOTOUT_FINGERPRINT: {fingerprint(frame)}")
