"""Shared helpers for Florida IO read shootouts.

These scripts intentionally benchmark only the read path.  The surrounding
vsbench harness handles backend swapping between real geopandas and the
repo-local geopandas shim exposed by vibeSpatial.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "examples" / "data"
FORMATS_DIR = DATA_DIR / "florida_formats"


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run examples/nearby_buildings.py and "
            "examples/export_florida_formats.py first."
        )
    return path


def fingerprint(frame) -> str:
    columns = tuple(str(column) for column in frame.columns)
    bounds = tuple(round(float(value), 6) for value in frame.total_bounds)
    return f"rows={len(frame)} cols={columns} bounds={bounds}"
