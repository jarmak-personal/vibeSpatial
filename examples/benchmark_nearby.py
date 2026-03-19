#!/usr/bin/env python
"""Benchmark the nearby-buildings workflow across three configurations.

Runs the same pipeline (read, reproject, spatial query, write parquet)
with upstream GeoPandas, vibeSpatial CPU-fallback, and vibeSpatial GPU,
then prints a comparison table.

The script auto-creates a temporary venv with upstream GeoPandas for
the baseline comparison. No manual setup required.

Requires:
- examples/data/Florida.geojson (run nearby_buildings.py first to download)
- vibeSpatial installed with GPU extras (pip install vibespatial[cu12])

Usage
-----
    python examples/benchmark_nearby.py
    python examples/benchmark_nearby.py --skip-geopandas   # skip baseline
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

GEOJSON = "examples/data/Florida.geojson"
OUTPUT_DIR = Path("examples/data")
SEED_IDX = 3_000_000  # fixed index for reproducibility
RADIUS_M = 1_000


def _bench_script(label: str, python: str, script: str) -> dict:
    """Run a benchmark script in a subprocess and return parsed timings."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    result = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"FAILED:\n{result.stderr}")
        return {}
    # Parse JSON timings from last line
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {}


def _setup_geopandas_venv() -> str | None:
    """Create a temporary venv with upstream GeoPandas. Returns python path."""
    venv_dir = Path(tempfile.mkdtemp(prefix="vibespatial-bench-"))
    print(f"Creating GeoPandas baseline venv at {venv_dir} ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True, capture_output=True,
        )
        pip = str(venv_dir / "bin" / "pip")
        subprocess.run(
            [pip, "install", "-q", "geopandas", "pyogrio", "pyarrow", "pyproj"],
            check=True, capture_output=True, timeout=120,
        )
        python = str(venv_dir / "bin" / "python")
        # Verify it works
        subprocess.run(
            [python, "-c", "import geopandas; print(geopandas.__version__)"],
            check=True, capture_output=True,
        )
        return python
    except Exception as e:
        print(f"  Failed to create GeoPandas venv: {e}")
        shutil.rmtree(venv_dir, ignore_errors=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vibeSpatial vs GeoPandas")
    parser.add_argument("--skip-geopandas", action="store_true",
                        help="Skip the upstream GeoPandas baseline")
    args = parser.parse_args()

    if not Path(GEOJSON).exists():
        print(f"Missing {GEOJSON} -- run nearby_buildings.py first to download.")
        sys.exit(1)

    # Common benchmark body (uses whatever 'gpd' is bound to)
    bench_body = textwrap.dedent(f"""\
        import time, json

        t0 = time.monotonic()
        gdf = gpd.read_file("{GEOJSON}")
        t_read = time.monotonic() - t0
        n = len(gdf)
        print(f"  Read: {{n:,}} buildings in {{t_read:.1f}}s")

        t0 = time.monotonic()
        utm_crs = gdf.geometry.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        del gdf  # free the pre-projection copy to halve peak RAM
        t_proj = time.monotonic() - t0
        print(f"  Reproject to {{utm_crs}}: {{t_proj:.1f}}s")

        t0 = time.monotonic()
        seed = gdf_utm.geometry.iloc[{SEED_IDX}]
        mask = gdf_utm.geometry.dwithin(seed.centroid, {RADIUS_M})
        nearby = gdf_utm[mask].copy()
        t_query = time.monotonic() - t0
        print(f"  Select within {RADIUS_M}m: {{len(nearby):,}} buildings in {{t_query:.1f}}s")

        out = "{OUTPUT_DIR}/bench_output.parquet"
        t0 = time.monotonic()
        nearby.to_crs(epsg=4326).to_parquet(out)
        t_write = time.monotonic() - t0
        print(f"  Write parquet: {{t_write:.1f}}s")

        print(json.dumps({{"read": round(t_read, 1), "reproject": round(t_proj, 1),
                          "query": round(t_query, 1), "write": round(t_write, 1),
                          "buildings": n, "nearby": len(nearby)}}))
    """)

    gpd_python = None
    t_gpd: dict = {}

    # 1. Upstream GeoPandas
    if not args.skip_geopandas:
        gpd_python = _setup_geopandas_venv()
        if gpd_python:
            gpd_script = f"import geopandas as gpd\n{bench_body}"
            t_gpd = _bench_script("Upstream GeoPandas", gpd_python, gpd_script)
            # Clean up venv
            shutil.rmtree(Path(gpd_python).parent.parent, ignore_errors=True)
        else:
            print("Skipping GeoPandas baseline (venv setup failed)")
    else:
        print("Skipping GeoPandas baseline (--skip-geopandas)")

    # 2. vibeSpatial CPU fallback
    cpu_script = (
        "import os; os.environ['VIBESPATIAL_EXECUTION_MODE'] = 'cpu'\n"
        "import vibespatial.api as gpd\n"
        f"{bench_body}"
    )
    t_cpu = _bench_script("vibeSpatial (CPU fallback)", sys.executable, cpu_script)

    # 3. vibeSpatial GPU
    gpu_script = f"import vibespatial.api as gpd\n{bench_body}"
    t_gpu = _bench_script("vibeSpatial (GPU)", sys.executable, gpu_script)

    # Summary table
    print(f"\n{'=' * 60}")
    print("  RESULTS: 7.2M Florida building footprints")
    print(f"{'=' * 60}")
    print(f"{'Step':<25} {'GeoPandas':>10} {'vS CPU':>10} {'vS GPU':>10}")
    print(f"{'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}")
    for step in ("read", "reproject", "query", "write"):
        label = {
            "read": "Read GeoJSON",
            "reproject": "Reproject to UTM",
            "query": "Select within 1 km",
            "write": "Write GeoParquet",
        }[step]
        vals = []
        for t in (t_gpd, t_cpu, t_gpu):
            v = t.get(step)
            vals.append(f"{v:.1f}s" if v is not None else "N/A")
        print(f"{label:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")


if __name__ == "__main__":
    main()
