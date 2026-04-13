"""Florida OSM PBF geometry-only read shootout."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _osm import (
    PBF_PATH,
    fingerprint,
    frame_from_osm_result,
    read_supported_layers_pyogrio,
    require_path,
    running_with_vibespatial,
)

source = require_path(PBF_PATH)

if running_with_vibespatial():
    from vibespatial.io.osm_gpu import read_osm_pbf

    def _run():
        return read_osm_pbf(source, geometry_only=True)

    def _finalize(payload):
        return frame_from_osm_result(payload)
else:
    def _run():
        return read_supported_layers_pyogrio(source, geometry_only=True)

    def _finalize(payload):
        return payload


# --- timed work starts here ---
payload = _run()
# --- timed work ends here ---

frame = _finalize(payload)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(frame)}")
