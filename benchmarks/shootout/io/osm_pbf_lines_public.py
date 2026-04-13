"""Florida OSM PBF public lines-layer shootout."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _osm import (
    PBF_PATH,
    fingerprint,
    normalize_single_layer_osm_frame,
    read_pyogrio_osm_layer,
    require_path,
    running_with_vibespatial,
)

source = require_path(PBF_PATH)

if running_with_vibespatial():
    import geopandas as gpd

    def _run():
        return gpd.read_file(source, layer="lines")

    def _finalize(payload):
        return normalize_single_layer_osm_frame(payload, osm_element="way")
else:
    def _run():
        return read_pyogrio_osm_layer(source, layer="lines", geometry_only=False)

    def _finalize(payload):
        return normalize_single_layer_osm_frame(payload, osm_element="way")


# --- timed work starts here ---
payload = _run()
# --- timed work ends here ---

frame = _finalize(payload)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(frame)}")
