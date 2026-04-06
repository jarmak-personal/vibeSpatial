from __future__ import annotations

import json
import time
from pathlib import Path

from vibespatial.api import read_file
from vibespatial.overlay.contraction import summarize_overlay_contraction_canary

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "tests" / "upstream" / "geopandas" / "tests" / "data"
LEFT_NYBB = DATA_ROOT / "nybb_16a.zip"
RIGHT_NYBB = DATA_ROOT / "overlay" / "nybb_qgis" / "polydf2.shp"


def main() -> int:
    left = read_file(f"zip://{LEFT_NYBB}")
    right = read_file(str(RIGHT_NYBB))
    started = time.perf_counter()
    summary = summarize_overlay_contraction_canary(
        left.geometry.values.to_owned(),
        right.geometry.values.to_owned(),
    )
    elapsed = time.perf_counter() - started
    print(json.dumps({"wall_seconds": elapsed, "summary": summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
