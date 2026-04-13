"""Florida OSM PBF public read shootout."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _osm import (
    PBF_PATH,
    fingerprint,
    normalize_public_osm_frame,
    read_supported_layers_pyogrio,
    require_path,
    running_with_vibespatial,
)

source = require_path(PBF_PATH)

if running_with_vibespatial():
    import geopandas as gpd

    def _run():
        return gpd.read_file(source)

    def _finalize(payload):
        return normalize_public_osm_frame(payload)
else:
    def _run():
        return read_supported_layers_pyogrio(source, geometry_only=False)

    def _finalize(payload):
        return payload


# --- timed work starts here ---
payload = _run()
# --- timed work ends here ---

frame = _finalize(payload)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(frame)}")
