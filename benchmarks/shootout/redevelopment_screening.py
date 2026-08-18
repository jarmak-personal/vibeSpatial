"""Transit-oriented redevelopment screening.

Read parcels, zoning polygons, exclusion zones, and transit stations.
Clip parcels to a study area, remove exclusion areas, keep parcels near
transit, intersect with zoning polygons, then dissolve by zone type.

Usage:
    vsbench shootout benchmarks/shootout/redevelopment_screening.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _data import fingerprint, setup_fixtures, spatial_semijoin
from shapely.geometry import box

import geopandas as gpd

CANDIDATE_PAGE_ROWS = 10_000
PAGED_REDUCTION_MIN_ZONE_ROWS = 500

_tmpdir = tempfile.TemporaryDirectory(prefix="shootout_redev_")
tmpdir = Path(_tmpdir.name)
fixtures = setup_fixtures(tmpdir)

# --- timed work starts here ---

parcels = gpd.read_parquet(fixtures["parcels"])
zones = gpd.read_parquet(fixtures["zones"])
exclusions = gpd.read_parquet(fixtures["exclusion_zones"])
transit = gpd.read_file(fixtures["transit"])

bounds = parcels.total_bounds
dx = (bounds[2] - bounds[0]) * 0.15
dy = (bounds[3] - bounds[1]) * 0.15
clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)
study_parcels = gpd.clip(parcels, clip_box)

study_parcels = study_parcels[
    study_parcels.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
].copy()

developable = gpd.overlay(study_parcels, exclusions, how="difference")

transit_buffers = transit.copy()
transit_buffers["geometry"] = transit_buffers.geometry.buffer(150.0)

near_transit = spatial_semijoin(
    developable,
    transit_buffers[["station_id", "geometry"]],
    predicate="intersects",
)
candidates = near_transit

if len(candidates) > 0 and len(zones) >= PAGED_REDUCTION_MIN_ZONE_ROWS:
    candidate_input_path = tmpdir / "candidates.parquet"
    candidates[["geometry"]].to_parquet(candidate_input_path)
    candidates = gpd.read_parquet(candidate_input_path)
    del (
        parcels,
        study_parcels,
        exclusions,
        transit,
        transit_buffers,
        near_transit,
        developable,
    )

    # Intersection distributes over union. Union the four zone classes once,
    # then intersect spatially ordered candidate pages with only the portion of
    # those four boundaries inside the page envelope. Candidates are not cut at
    # artificial partition edges, and their interiors are disjoint, so coverage
    # union is exact within and across pages.
    unioned_zones = zones[["zone_type", "geometry"]].dissolve(
        by="zone_type",
    ).reset_index()
    unioned_zone_path = tmpdir / "unioned_zones.parquet"
    unioned_zones.to_parquet(unioned_zone_path)
    unioned_zones = gpd.read_parquet(unioned_zone_path)

    page_frames = []
    for page_start in range(0, len(candidates), CANDIDATE_PAGE_ROWS):
        candidate_page = candidates.iloc[
            page_start : page_start + CANDIDATE_PAGE_ROWS
        ]
        page_box = box(*candidate_page.total_bounds)
        zone_page = gpd.clip(unioned_zones, page_box)
        zone_page = zone_page[
            zone_page.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ]
        if len(zone_page) == 0:
            continue
        zoned_page = gpd.overlay(
            candidate_page[["geometry"]],
            zone_page[["zone_type", "geometry"]],
            how="intersection",
        )
        if len(zoned_page) == 0:
            continue
        zoned_page = zoned_page.dissolve(
            by="zone_type",
            method="coverage",
        ).reset_index()
        page_path = tmpdir / f"zoned_page_{page_start}.parquet"
        zoned_page.to_parquet(page_path)
        page_frames.append(gpd.read_parquet(page_path))
    reduced_pages = (
        pd.concat(page_frames, ignore_index=True)
        if page_frames
        else candidates.assign(zone_type=0).iloc[:0]
    )
    dissolved = (
        reduced_pages[["zone_type", "geometry"]]
        .dissolve(
            by="zone_type",
            method="coverage",
        )
        .reset_index()
        if len(reduced_pages) > 0
        else reduced_pages[["zone_type", "geometry"]]
    )
    del (
        zones,
        candidates,
        unioned_zones,
        page_frames,
        reduced_pages,
    )

    dissolved["zone_group"] = dissolved["zone_type"].astype(str)
    dissolved = dissolved[["zone_group", "zone_type", "geometry"]]
elif len(candidates) > 0:
    zoned = gpd.overlay(
        candidates,
        zones[["zone_type", "geometry"]],
        how="intersection",
    )
    if len(zoned) > 0:
        zoned["zone_group"] = zoned["zone_type"].astype(str)
        dissolved = zoned.dissolve(by="zone_group").reset_index()
    else:
        dissolved = zoned.copy()
else:
    dissolved = candidates.assign(zone_group="", zone_type=0)[
        ["zone_group", "zone_type", "geometry"]
    ]

output_path = tmpdir / "output.parquet"
dissolved.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
