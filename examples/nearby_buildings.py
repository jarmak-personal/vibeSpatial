#!/usr/bin/env python
"""Find buildings near a random building in the Microsoft US Building Footprints dataset.

Downloads the Florida GeoJSON (~2 GB zipped) from Microsoft's open dataset,
reprojects to UTM, picks a random building, selects everything within 1 km,
and exports the neighbourhood to GeoParquet.

Source: https://github.com/microsoft/USBuildingFootprints

Usage
-----
    uv run python examples/nearby_buildings.py

The first run downloads and caches the zip; subsequent runs reuse it.
"""

from __future__ import annotations

import random
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

import vibespatial.api as gpd


DATA_DIR = Path("examples/data")
ZIP_PATH = DATA_DIR / "Florida.geojson.zip"
GEOJSON_PATH = DATA_DIR / "Florida.geojson"
OUTPUT_PATH = DATA_DIR / "nearby_buildings.parquet"

SOURCE_URL = (
    "https://minedbuildings.z5.web.core.windows.net/"
    "legacy/usbuildings-v2/Florida.geojson.zip"
)

SEARCH_RADIUS_M = 1_000  # 1 km


def download_if_missing() -> Path:
    """Download and extract the Florida buildings GeoJSON if not cached."""
    if GEOJSON_PATH.exists():
        print(f"Using cached {GEOJSON_PATH}")
        return GEOJSON_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"Downloading {SOURCE_URL} ...")
        t0 = time.monotonic()
        with urlopen(SOURCE_URL) as resp:
            ZIP_PATH.write_bytes(resp.read())
        elapsed = time.monotonic() - t0
        size_mb = ZIP_PATH.stat().st_size / 1e6
        print(f"  downloaded {size_mb:.0f} MB in {elapsed:.0f}s")

    print("Extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)

    print(f"  extracted to {GEOJSON_PATH}")
    return GEOJSON_PATH


def main() -> None:
    geojson_path = download_if_missing()

    # -- Read -----------------------------------------------------------
    print(f"\nReading {geojson_path} ...")
    t0 = time.monotonic()
    gdf = gpd.read_file(str(geojson_path))
    print(f"  {len(gdf):,} buildings loaded in {time.monotonic() - t0:.1f}s")

    # -- Reproject to UTM -----------------------------------------------
    print("\nEstimating UTM CRS and reprojecting ...")
    t0 = time.monotonic()
    utm_crs = gdf.geometry.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    print(f"  reprojected to {utm_crs} in {time.monotonic() - t0:.1f}s")

    # -- Pick a random building -----------------------------------------
    idx = random.randrange(len(gdf_utm))
    seed_building = gdf_utm.geometry.iloc[idx]
    centroid = seed_building.centroid
    print(f"\nSeed building index: {idx}")
    print(f"  centroid (UTM): ({centroid.x:.1f}, {centroid.y:.1f})")

    # -- Select within 1 km --------------------------------------------
    print(f"\nSelecting buildings within {SEARCH_RADIUS_M} m ...")
    t0 = time.monotonic()
    mask = gdf_utm.geometry.dwithin(centroid, SEARCH_RADIUS_M)
    nearby = gdf_utm[mask].copy()
    elapsed = time.monotonic() - t0
    print(f"  {len(nearby):,} buildings found in {elapsed:.1f}s")

    # -- Export to GeoParquet -------------------------------------------
    print(f"\nWriting {OUTPUT_PATH} ...")
    # Convert back to WGS 84 for interoperability
    nearby_wgs84 = nearby.to_crs(epsg=4326)
    nearby_wgs84.to_parquet(str(OUTPUT_PATH))
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  wrote {size_kb:.0f} KB ({len(nearby_wgs84):,} buildings)")

    print("\nDone!")


if __name__ == "__main__":
    main()
