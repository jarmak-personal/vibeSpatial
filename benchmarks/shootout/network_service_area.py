"""Telecom network service area coverage.

Read network line segments and an administrative boundary polygon.
Buffer lines to create coverage areas, dissolve into a single service
polygon, then clip to the admin boundary.

Usage:
    vsbench shootout benchmarks/shootout/network_service_area.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import fingerprint, setup_fixtures

import geopandas as gpd

tmpdir = Path(tempfile.mkdtemp(prefix="shootout_netsvc_"))
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

# Read
network = gpd.read_parquet(fixtures["network"])
admin = gpd.read_file(fixtures["admin_boundary"])

# Buffer network lines
network["geometry"] = network.geometry.buffer(25.0)

# Dissolve into single service area
network["group"] = 0
dissolved = network.dissolve(by="group")

# Clip to admin boundary
clipped = gpd.clip(dissolved, admin)

# Write result
output_path = tmpdir / "output.parquet"
clipped.to_parquet(output_path)

# --- timed work ends here ---

# --- fingerprint (post-benchmark, not timed) ---
check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
