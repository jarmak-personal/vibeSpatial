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

from _data import fingerprint, setup_fixtures, spatial_semijoin
from shapely.geometry import box

import geopandas as gpd

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

if len(candidates) > 0:
    # The terminal result keeps only zone_type and unions every fragment in
    # that group. Set intersection distributes over union, so reduce each
    # certified coverage before constructing intersections:
    #
    #   union_i (candidate_i intersection zone_g)
    #       == union(candidate_i) intersection union(zone_g)
    #
    # The parcel grid is interior-disjoint, and clip/difference preserve that
    # property, so coverage union is exact for the selected candidates. This
    # physical shape avoids materializing the much larger per-parcel fragment
    # relation while preserving the public overlay/dissolve semantics needed by
    # the output.
    candidate_coverage = candidates[["geometry"]].dissolve(
        method="coverage",
    ).reset_index(drop=True)
    unioned_zones = zones[["zone_type", "geometry"]].dissolve(
        by="zone_type",
    ).reset_index()
    zoned = gpd.overlay(
        candidate_coverage[["geometry"]],
        unioned_zones[["zone_type", "geometry"]],
        how="intersection",
    )
    dissolved = (
        zoned.dissolve(
            by="zone_type",
            method="coverage",
        ).reset_index()
        if len(zoned) > 0
        else zoned[["zone_type", "geometry"]]
    )

    dissolved["zone_group"] = dissolved["zone_type"].astype(str)
    dissolved = dissolved[["zone_group", "zone_type", "geometry"]]
else:
    dissolved = candidates.assign(zone_group="", zone_type=0)[
        ["zone_group", "zone_type", "geometry"]
    ]

output_path = tmpdir / "output.parquet"
dissolved.to_parquet(output_path)

# --- timed work ends here ---

check = gpd.read_parquet(output_path)
print(f"SHOOTOUT_FINGERPRINT: {fingerprint(check)}")
