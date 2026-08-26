#!/usr/bin/env python3
"""Public-API counterfactual for scalar line-buffer/union ordering."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "shootout"))

import numpy as np
import shapely
from _data import fingerprint, setup_fixtures

import geopandas as gpd

VARIANT = os.environ.get("VIBESPATIAL_R3_VARIANT", "buffer_then_union")
if VARIANT not in {
    "buffer_then_union",
    "union_then_buffer",
    "distribute_intersection_then_union",
}:
    raise ValueError(f"unsupported R3 variant: {VARIANT}")

_tmpdir = tempfile.TemporaryDirectory(prefix="r3_vegetation_buffer_union_")
tmpdir = Path(_tmpdir.name)
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

lines = gpd.read_parquet(fixtures["lines"])
vegetation = gpd.read_parquet(fixtures["vegetation"])
poles = gpd.read_file(fixtures["poles"])

lines["group"] = 0
if VARIANT == "buffer_then_union":
    lines["geometry"] = lines.geometry.buffer(10.0)
    dissolved = lines.dissolve(by="group")
elif VARIANT == "union_then_buffer":
    dissolved = lines.dissolve(by="group")
    dissolved["geometry"] = dissolved.geometry.buffer(10.0)
else:
    dissolved = None

vegetation["geometry"] = vegetation.geometry.make_valid()
if VARIANT == "distribute_intersection_then_union":
    vegetation["_r3_source_row"] = np.arange(len(vegetation), dtype=np.int64)
    buffered_lines = lines[["geometry"]].copy()
    buffered_lines["geometry"] = buffered_lines.geometry.buffer(10.0)
    pieces = gpd.overlay(
        vegetation,
        buffered_lines,
        how="intersection",
        make_valid=False,
    )
    clipped = pieces.dissolve(
        by="_r3_source_row",
        aggfunc={"species": "first"},
    ).reset_index(drop=True)
else:
    dissolved["geometry"] = dissolved.geometry.make_valid()
    try:
        clipped = gpd.overlay(
            vegetation,
            dissolved[["geometry"]],
            how="intersection",
            make_valid=False,
        )
    except Exception:
        corridor_geom = dissolved.geometry.values[0]
        veg_arr = np.asarray(vegetation.geometry.values, dtype=object)
        corridor_arr = np.full(len(veg_arr), corridor_geom, dtype=object)
        try:
            intersected = shapely.intersection(veg_arr, corridor_arr)
            keep = ~shapely.is_empty(intersected) & ~shapely.is_missing(intersected)
            results = list(intersected[keep])
        except Exception:
            results = []
        clipped = gpd.GeoDataFrame(
            {"geometry": results if results else []},
            geometry="geometry",
            crs=vegetation.crs,
        )

if len(clipped) > 0:
    buffered_veg = clipped.copy()
    buffered_veg["geometry"] = clipped.geometry.centroid.buffer(1.0)
    joined = gpd.sjoin(poles, buffered_veg[["geometry"]], predicate="within")
else:
    joined = poles.iloc[:0]

output_path = Path(os.environ.get("VIBESPATIAL_R3_OUTPUT", tmpdir / "output.parquet"))
clipped.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"R3_VARIANT: {VARIANT}")
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
