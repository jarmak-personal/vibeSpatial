#!/usr/bin/env python
"""Benchmark Florida CSV/WKT read time against an explicit host baseline.

This benchmark exists because CSV/WKT is not a clean same-script shootout:
the portable baseline path is `pandas.read_csv + GeoSeries.from_wkt`, while
the repo-native path is `vibespatial.read_file(...)`.

Usage:
    uv run python examples/benchmark_florida_csv.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

CSV_PATH = Path("examples/data/florida_formats/Florida.wkt.csv")


def _run_subprocess(python_cmd: list[str], code: str) -> dict:
    proc = subprocess.run(
        [*python_cmd, "-c", code],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "benchmark subprocess failed")
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"benchmark subprocess did not emit JSON:\n{proc.stdout}")


def _setup_geopandas_venv() -> str:
    uv = shutil.which("uv")
    if uv is not None:
        return uv

    venv_dir = Path(tempfile.mkdtemp(prefix="vibespatial-csv-bench-"))
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    pip = str(venv_dir / "bin" / "pip")
    subprocess.run(
        [pip, "install", "-q", "geopandas", "pandas", "shapely", "pyarrow"],
        check=True,
        timeout=300,
    )
    return str(venv_dir / "bin" / "python")


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"Missing {CSV_PATH}. Run examples/export_florida_formats.py first."
        )

    baseline_code = textwrap.dedent(
        f"""\
        import json
        import time
        import pandas as pd
        import geopandas as gpd

        path = {str(CSV_PATH)!r}
        t0 = time.perf_counter()
        table = pd.read_csv(path)
        geometry = gpd.GeoSeries.from_wkt(table["geometry"], crs="EPSG:4326")
        frame = gpd.GeoDataFrame(table.drop(columns=["geometry"]), geometry=geometry, crs="EPSG:4326")
        dt = time.perf_counter() - t0
        bounds = tuple(round(float(v), 6) for v in frame.total_bounds)
        print(json.dumps({{
            "engine": "host-baseline",
            "elapsed_seconds": round(dt, 3),
            "rows": int(len(frame)),
            "bounds": bounds,
        }}))
        """
    )

    vibespatial_code = textwrap.dedent(
        f"""\
        import json
        import time
        from pathlib import Path
        import vibespatial.api as gpd
        from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
        from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events

        path = Path({str(CSV_PATH)!r})
        clear_dispatch_events()
        clear_fallback_events()
        t0 = time.perf_counter()
        frame = gpd.read_file(path)
        dt = time.perf_counter() - t0
        dispatch = [event.implementation for event in get_dispatch_events()]
        fallbacks = [event.reason for event in get_fallback_events()]
        payload = {{
            "engine": "vibespatial",
            "elapsed_seconds": round(dt, 3),
            "rows": int(len(frame)),
            "frame_type": type(frame).__name__,
            "columns": [str(c) for c in frame.columns],
            "dispatch": dispatch,
            "fallbacks": fallbacks,
        }}
        if hasattr(frame, "geometry"):
            payload["geometry_values_type"] = type(frame.geometry.values).__name__
            try:
                payload["bounds"] = tuple(round(float(v), 6) for v in frame.total_bounds)
            except Exception:
                payload["bounds"] = None
        else:
            payload["geometry_values_type"] = None
            payload["bounds"] = None
        print(json.dumps(payload))
        """
    )

    baseline_runner = _setup_geopandas_venv()
    if baseline_runner == shutil.which("uv"):
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        baseline_cmd = [
            baseline_runner,
            "run",
            "--no-project",
            "--python",
            py_ver,
            "--with",
            "geopandas",
            "--with",
            "pandas",
            "--with",
            "shapely",
            "--with",
            "pyarrow",
            "--",
            "python",
        ]
    else:
        baseline_cmd = [baseline_runner]

    baseline = _run_subprocess(baseline_cmd, baseline_code)
    vibespatial = _run_subprocess([sys.executable], vibespatial_code)

    print(json.dumps({"baseline": baseline, "vibespatial": vibespatial}, indent=2))


if __name__ == "__main__":
    main()
