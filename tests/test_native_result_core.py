from __future__ import annotations

import warnings

import pandas as pd
import pytest
from pandas.errors import Pandas4Warning
from shapely.geometry import Point

from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._native_result_core import (
    GeometryNativeResult,
    NativeAttributeTable,
    NativeGeometryColumn,
    NativeTabularResult,
)
from vibespatial.api.testing import assert_geodataframe_equal


def test_native_tabular_result_core_preserves_explicit_export_contract() -> None:
    index = pd.Index([10, 11], name="row_id")
    primary = GeoSeries(
        [Point(0, 0), Point(1, 1)],
        index=index,
        name="geom",
        crs="EPSG:4326",
    )
    secondary = GeoSeries(
        [Point(0, 1), Point(1, 2)],
        index=index,
        name="centroid",
        crs="EPSG:4326",
    )
    payload = NativeTabularResult(
        attributes=pd.DataFrame({"value": [1, 2]}, index=index),
        geometry=GeometryNativeResult.from_geoseries(primary),
        geometry_name="geom",
        column_order=("geom", "value", "centroid"),
        attrs={"source": "native-core"},
        secondary_geometry=(
            NativeGeometryColumn(
                "centroid",
                GeometryNativeResult.from_geoseries(secondary),
            ),
        ),
    )

    result = payload.to_geodataframe()

    expected = GeoDataFrame(
        {"geom": primary, "value": [1, 2], "centroid": secondary},
        index=index,
        geometry="geom",
        crs="EPSG:4326",
    )
    expected.attrs["source"] = "native-core"

    assert result._geometry_column_name == "geom"
    assert list(result.columns) == ["geom", "value", "centroid"]
    assert_geodataframe_equal(result, expected)


def test_native_tabular_result_core_resolves_late_attribute_columns_before_geometry() -> None:
    load_calls = 0

    def _load() -> pd.DataFrame:
        nonlocal load_calls
        load_calls += 1
        return pd.DataFrame(
            {"value": [1, 2], "kind": ["a", "b"]},
            index=pd.RangeIndex(2),
        )

    payload = NativeTabularResult(
        attributes=NativeAttributeTable.from_loader(
            _load,
            index_override=pd.RangeIndex(2),
            columns=(),
        ),
        geometry=GeometryNativeResult.from_geoseries(
            GeoSeries([Point(0, 0), Point(1, 1)], crs="EPSG:4326", name="geom")
        ),
        geometry_name="geom",
        column_order=("geom",),
    )

    assert payload.resolved_column_order == ("value", "kind", "geom")

    frame = payload.to_geodataframe()

    assert load_calls == 1
    assert list(frame.columns) == ["value", "kind", "geom"]
    assert frame.geometry.name == "geom"


def test_native_tabular_result_core_requires_all_geometry_names_in_column_order() -> None:
    index = pd.RangeIndex(2)
    with pytest.raises(
        ValueError,
        match="column_order must include every geometry column",
    ):
        NativeTabularResult(
            attributes=pd.DataFrame({"value": [1, 2]}, index=index),
            geometry=GeometryNativeResult.from_geoseries(
                GeoSeries([Point(0, 0), Point(1, 1)], index=index, crs="EPSG:4326", name="geom")
            ),
            geometry_name="geom",
            column_order=("value", "geom"),
            secondary_geometry=(
                NativeGeometryColumn(
                    "centroid",
                    GeometryNativeResult.from_geoseries(
                        GeoSeries(
                            [Point(0, 1), Point(1, 2)],
                            index=index,
                            crs="EPSG:4326",
                            name="centroid",
                        )
                    ),
                ),
            ),
        )


def test_native_tabular_result_core_export_avoids_pandas4_reindex_warning() -> None:
    payload = NativeTabularResult(
        attributes=pd.DataFrame({"value": [1, 2]}, index=pd.RangeIndex(2)),
        geometry=GeometryNativeResult.from_geoseries(
            GeoSeries([Point(0, 0), Point(1, 1)], crs="EPSG:4326", name="geom")
        ),
        geometry_name="geom",
        column_order=("geom", "value"),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", Pandas4Warning)
        frame = payload.to_geodataframe()

    assert frame.geometry.name == "geom"
    assert not any(isinstance(item.message, Pandas4Warning) for item in caught)


def test_native_attribute_table_concat_preserves_row_count_without_attribute_columns() -> None:
    left = NativeAttributeTable(dataframe=pd.DataFrame(index=pd.RangeIndex(2)))
    right = NativeAttributeTable(dataframe=pd.DataFrame(index=pd.RangeIndex(3)))

    combined = NativeAttributeTable.concat([left, right], ignore_index=True)

    assert len(combined) == 5
    assert list(combined.columns) == []
    assert combined.index.equals(pd.RangeIndex(5))
