"""Florida OSM PBF public multipolygons-layer shootout."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _osm import (
    PBF_PATH,
    fingerprint,
    normalize_multipolygons_public_frame,
    read_pyogrio_osm_layer,
    require_path,
    running_with_vibespatial,
)

source = require_path(PBF_PATH)

if running_with_vibespatial():
    import geopandas as gpd

    def _run():
        return gpd.read_file(source, layer="multipolygons")

    def _finalize(payload):
        return normalize_multipolygons_public_frame(payload)
else:
    def _run():
        return read_pyogrio_osm_layer(source, layer="multipolygons", geometry_only=False)

    def _finalize(payload):
        return normalize_multipolygons_public_frame(payload)


# --- timed work starts here ---
payload = _run()
# --- timed work ends here ---

frame = _finalize(payload)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(frame)}")
