"""Regression test for overlay index alignment with non-contiguous indices.

When ``_make_valid`` inside ``overlay()`` repairs rows via the GPU
``make_valid_owned`` path, the replacement GeoSeries must carry the
DataFrame's index so that pandas column assignment aligns correctly.

Before the fix (line 1454 of overlay.py), ``GeoSeries(new_ga)`` created a
default RangeIndex(0..N-1), which misaligned with non-contiguous DataFrame
indices (e.g. after ``gpd.clip()``).  Pandas column assignment by index
caused only overlapping indices to receive values — the rest became NaN —
and the subsequent ``_collection_extract()`` dropped those NaN rows.

Fingerprint: vibeSpatial-9u2
"""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from vibespatial.api import GeoDataFrame
from vibespatial.api.tools.overlay import overlay


def _make_non_contiguous_gdf(
    polygons: list[Polygon],
    data_col: str,
    data_values: list[int],
) -> GeoDataFrame:
    """Build a GeoDataFrame with a non-contiguous integer index.

    Mimics what happens after ``gpd.clip()`` or ``gdf.iloc[::2]`` — the
    resulting DataFrame retains the original index, which is no longer a
    contiguous RangeIndex(0..N-1).
    """
    gdf = GeoDataFrame({data_col: data_values}, geometry=polygons)
    # Assign a deliberately non-contiguous index: [10, 30, 50, ...]
    gdf.index = range(10, 10 + 20 * len(polygons), 20)
    return gdf


@pytest.fixture()
def non_contiguous_pair():
    """Two GeoDataFrames with overlapping polygons and non-contiguous indices."""
    left_polys = [
        Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
        Polygon([(3, 0), (6, 0), (6, 3), (3, 3)]),
        Polygon([(0, 3), (3, 3), (3, 6), (0, 6)]),
        Polygon([(3, 3), (6, 3), (6, 6), (3, 6)]),
    ]
    right_polys = [
        Polygon([(1, 1), (4, 1), (4, 4), (1, 4)]),
        Polygon([(4, 1), (7, 1), (7, 4), (4, 4)]),
    ]
    df1 = _make_non_contiguous_gdf(left_polys, "left_val", [1, 2, 3, 4])
    df2 = _make_non_contiguous_gdf(right_polys, "right_val", [10, 20])
    return df1, df2


def _reference_overlay(df1: GeoDataFrame, df2: GeoDataFrame, how: str) -> GeoDataFrame:
    """Compute the reference result using stock GeoPandas overlay (Shapely path).

    Converts to plain geopandas GeoDataFrames with Shapely-backed geometry
    to ensure the reference never touches the owned/GPU path.
    """
    import geopandas as gpd

    gdf1 = gpd.GeoDataFrame(
        df1.drop(columns="geometry"),
        geometry=gpd.GeoSeries.from_wkb(df1.geometry.to_wkb()),
    )
    gdf1.index = df1.index
    gdf2 = gpd.GeoDataFrame(
        df2.drop(columns="geometry"),
        geometry=gpd.GeoSeries.from_wkb(df2.geometry.to_wkb()),
    )
    gdf2.index = df2.index
    return gpd.overlay(gdf1, gdf2, how=how)


def test_overlay_intersection_preserves_all_rows_with_non_contiguous_index(
    non_contiguous_pair,
) -> None:
    """Overlay intersection must not lose rows when the DataFrame has a
    non-contiguous index (regression for vibeSpatial-9u2)."""
    df1, df2 = non_contiguous_pair
    result = overlay(df1, df2, how="intersection")
    reference = _reference_overlay(df1, df2, how="intersection")

    # The vibeSpatial result must contain at least as many rows as the
    # reference.  Before the fix, ~60-70% of rows were silently dropped.
    assert len(result) == len(reference), (
        f"Row count mismatch: vibeSpatial produced {len(result)} rows, "
        f"reference produced {len(reference)} rows"
    )

    # Verify no geometry column is entirely NaN (symptom of the old bug).
    assert result.geometry.notna().all(), "Result contains NaN geometries"

    # Verify all expected attribute columns survived.
    assert "left_val" in result.columns
    assert "right_val" in result.columns


@pytest.mark.parametrize("how", ["intersection", "union", "difference", "symmetric_difference"])
def test_overlay_all_modes_with_non_contiguous_index(
    non_contiguous_pair,
    how: str,
) -> None:
    """All overlay modes must produce correct row counts with non-contiguous indices."""
    df1, df2 = non_contiguous_pair
    result = overlay(df1, df2, how=how)
    reference = _reference_overlay(df1, df2, how=how)

    assert len(result) == len(reference), (
        f"[{how}] Row count mismatch: vibeSpatial={len(result)}, "
        f"reference={len(reference)}"
    )


def test_overlay_intersection_with_iloc_derived_index() -> None:
    """Overlay works correctly when the non-contiguous index comes from iloc slicing,
    which is the most common source (e.g. after gpd.clip())."""
    # Create a larger GeoDataFrame and slice every other row.
    polys = [
        Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)])
        for i in range(6)
    ]
    gdf = GeoDataFrame({"val": list(range(6))}, geometry=polys)
    sliced = gdf.iloc[::2]  # rows 0, 2, 4 — non-contiguous index [0, 2, 4]

    other = GeoDataFrame(
        {"other_val": [100]},
        geometry=[Polygon([(0.5, -0.5), (3.5, -0.5), (3.5, 1.5), (0.5, 1.5)])],
    )

    result = overlay(sliced, other, how="intersection")
    reference = _reference_overlay(sliced, other, how="intersection")

    assert len(result) == len(reference), (
        f"Row count mismatch after iloc slice: vibeSpatial={len(result)}, "
        f"reference={len(reference)}"
    )
    assert result.geometry.notna().all()
