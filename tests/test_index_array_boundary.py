"""ADR-0036: Index-array boundary contract tests.

These tests validate that spatial kernels produce only index arrays and that
attribute assembly happens entirely on the host via pandas.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, box

import vibespatial.api as geopandas
from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.runtime import has_gpu_runtime
from vibespatial.spatial.query import build_owned_spatial_index, query_spatial_index

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def left_gdf():
    """Left GeoDataFrame with string, int, and float attribute columns."""
    return GeoDataFrame(
        {
            "name": ["a", "b", "c"],
            "value_int": [10, 20, 30],
            "value_float": [1.1, 2.2, 3.3],
            "geometry": GeoSeries([box(0, 0, 2, 2), box(1, 1, 3, 3), box(4, 4, 6, 6)]),
        }
    )


@pytest.fixture
def right_gdf():
    """Right GeoDataFrame with string, int, and float attribute columns."""
    return GeoDataFrame(
        {
            "label": ["x", "y"],
            "count": [100, 200],
            "score": [9.9, 8.8],
            "geometry": GeoSeries([box(1, 1, 3, 3), box(5, 5, 7, 7)]),
        }
    )


# ---------------------------------------------------------------------------
# Phase 1: Spatial query returns integer numpy arrays
# ---------------------------------------------------------------------------

class TestSpatialQueryReturnsIntegerArrays:
    PREDICATES = ["intersects", "contains", "within", "touches", "covers", "covered_by"]

    @pytest.mark.parametrize("predicate", PREDICATES)
    def test_spatial_query_returns_integer_numpy_arrays(self, predicate):
        tree = np.asarray([box(0, 0, 2, 2), box(1, 1, 3, 3), box(10, 10, 11, 11)], dtype=object)
        query = np.asarray([box(0.5, 0.5, 1.5, 1.5)], dtype=object)
        owned, flat = build_owned_spatial_index(tree)

        result = query_spatial_index(
            owned, flat, query, predicate=predicate, sort=True
        )
        assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
        assert np.issubdtype(result.dtype, np.integer), (
            f"Expected integer dtype for predicate={predicate}, got {result.dtype}"
        )


class TestSpatialNearestReturnsIndexArrays:
    def test_spatial_nearest_returns_index_arrays_and_distances(self):
        """sjoin_nearest produces results with correct index types."""
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([Point(0, 0), Point(5, 0)]),
        )
        right = GeoDataFrame(
            {"b": [10, 20, 30]},
            geometry=GeoSeries([Point(1, 0), Point(6, 0), Point(20, 0)]),
        )
        result = geopandas.sjoin_nearest(left, right, how="inner", distance_col="dist")
        assert "dist" in result.columns
        assert result["dist"].dtype == np.float64
        # index_right should be integer-typed
        assert "index_right" in result.columns
        assert np.issubdtype(result["index_right"].dtype, np.integer)


# ---------------------------------------------------------------------------
# Phase 2: sjoin preserves attributes across all join types
# ---------------------------------------------------------------------------

class TestSjoinPreservesAttributes:
    @pytest.mark.parametrize("how", ["inner", "left", "right"])
    def test_sjoin_preserves_attributes_all_join_types(self, left_gdf, right_gdf, how):
        result = geopandas.sjoin(left_gdf, right_gdf, how=how)
        # All attribute columns from the appropriate side(s) must be present.
        if how in ("inner", "left"):
            for col in ["name", "value_int", "value_float"]:
                assert col in result.columns, f"Missing left column {col} in {how} join"
            for col in ["label", "count", "score"]:
                assert f"index_{col}" in result.columns or col in result.columns
        elif how == "right":
            for col in ["label", "count", "score"]:
                assert col in result.columns, f"Missing right column {col} in {how} join"

    def test_sjoin_no_geometry_in_attribute_reindex(self, left_gdf, right_gdf):
        """The joined result should have exactly one geometry column."""
        result = geopandas.sjoin(left_gdf, right_gdf, how="inner")
        geom_cols = [c for c in result.columns if result[c].dtype.name == "geometry"]
        assert len(geom_cols) == 1


# ---------------------------------------------------------------------------
# Phase 3: Overlay preserves attributes
# ---------------------------------------------------------------------------

class TestOverlayPreservesAttributes:
    def test_overlay_intersection_preserves_attributes(self, left_gdf, right_gdf):
        result = geopandas.overlay(left_gdf, right_gdf, how="intersection")
        if len(result) > 0:
            # Both sides' attributes should be present (possibly suffixed).
            left_attr_found = any(
                c.startswith("name") or c == "name_1" for c in result.columns
            )
            right_attr_found = any(
                c.startswith("label") or c == "label_2" for c in result.columns
            )
            assert left_attr_found, "Left attributes missing from intersection"
            assert right_attr_found, "Right attributes missing from intersection"

    def test_overlay_union_nan_fills_non_overlapping(self, left_gdf, right_gdf):
        result = geopandas.overlay(left_gdf, right_gdf, how="union")
        # Non-overlapping portions should have NaN in the other side's columns.
        # Find a column from right side.
        right_cols = [c for c in result.columns if "label" in c or "count" in c or "score" in c]
        assert len(right_cols) > 0, "No right-side attribute columns found"
        # At least some rows should have NaN (the non-overlapping portions).
        has_nan = result[right_cols[0]].isna().any()
        # Only assert if there are non-overlapping geometries.
        if len(result) > len(left_gdf):
            assert has_nan, "Expected NaN fills for non-overlapping portions"


# ---------------------------------------------------------------------------
# Phase 4: Dissolve separates attribute aggregation from geometry
# ---------------------------------------------------------------------------

class TestDissolveBoundary:
    def test_dissolve_separates_attribute_agg_from_geometry(self):
        gdf = GeoDataFrame(
            {
                "group": ["a", "a", "b"],
                "value": [10, 20, 30],
                "geometry": GeoSeries([box(0, 0, 1, 1), box(0.5, 0, 1.5, 1), box(5, 5, 6, 6)]),
            }
        )
        result = gdf.dissolve(by="group", aggfunc="sum")
        assert result.loc["a", "value"] == 30
        assert result.loc["b", "value"] == 30
        # Geometry should be a union.
        assert result.loc["a", "geometry"].is_valid
        assert result.loc["b", "geometry"].is_valid


# ---------------------------------------------------------------------------
# Phase 5: Clip preserves all attribute columns
# ---------------------------------------------------------------------------

class TestClipPreservesAttributes:
    def test_clip_preserves_all_attribute_columns(self, left_gdf):
        clip_box = box(0, 0, 5, 5)
        result = geopandas.clip(left_gdf, clip_box)
        for col in ["name", "value_int", "value_float"]:
            assert col in result.columns, f"Clip dropped attribute column {col}"
        # Values should be preserved for features within the clip region.
        if len(result) > 0:
            assert result["value_int"].notna().all()


# ---------------------------------------------------------------------------
# GPU-conditional tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestGPUBoundary:
    @pytest.mark.skipif(not has_gpu_runtime(), reason="No GPU runtime available")
    def test_gpu_query_returns_host_arrays(self):
        tree = np.asarray([box(0, 0, 2, 2), box(3, 3, 5, 5)], dtype=object)
        query = np.asarray([box(1, 1, 4, 4)], dtype=object)
        owned, flat = build_owned_spatial_index(tree)

        result = query_spatial_index(
            owned, flat, query, predicate="intersects", sort=True
        )
        assert isinstance(result, np.ndarray)
        assert np.issubdtype(result.dtype, np.integer)
        # Should be on host (numpy), not device.
        assert type(result).__module__ == "numpy"

    @pytest.mark.skipif(not has_gpu_runtime(), reason="No GPU runtime available")
    def test_gpu_overlay_preserves_attributes(self):
        left = GeoDataFrame(
            {"a": [1, 2], "geometry": GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)])}
        )
        right = GeoDataFrame(
            {"b": [10, 20], "geometry": GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)])}
        )
        result = geopandas.overlay(left, right, how="intersection")
        if len(result) > 0:
            a_cols = [c for c in result.columns if c.startswith("a")]
            b_cols = [c for c in result.columns if c.startswith("b")]
            assert len(a_cols) > 0
            assert len(b_cols) > 0
