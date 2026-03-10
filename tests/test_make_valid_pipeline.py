from __future__ import annotations

import vibespatial.api as geopandas
from vibespatial.api import GeoSeries
from vibespatial.api.testing import assert_geoseries_equal
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon

from vibespatial import benchmark_make_valid, fusion_plan_for_make_valid, make_valid_owned, plan_make_valid_pipeline
from vibespatial.fusion import IntermediateDisposition


def test_make_valid_plan_compacts_invalid_rows_before_repair() -> None:
    plan = plan_make_valid_pipeline()

    assert [stage.name for stage in plan.stages] == [
        "compute_validity_mask",
        "compact_invalid_rows",
        "repair_invalid_topology",
        "scatter_repaired_rows",
        "emit_geometry",
    ]
    assert plan.stages[-1].disposition is IntermediateDisposition.PERSIST


def test_make_valid_fusion_plan_persists_final_geometry_only() -> None:
    fusion = fusion_plan_for_make_valid(method="structure", keep_collapsed=False)

    assert fusion.stages[-1].disposition is IntermediateDisposition.PERSIST
    assert fusion.stages[-1].steps[-1].output_name == "geometry_buffers"


def test_make_valid_owned_repairs_only_invalid_subset() -> None:
    polygon1 = Polygon([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)])
    polygon2 = Polygon([(0, 2), (0, 1), (2, 0), (0, 0), (0, 2)])
    linestring = LineString([(0, 0), (1, 1), (1, 0)])
    result = make_valid_owned([polygon1, polygon2, linestring, None])

    expected = GeoSeries(
        [
            MultiPolygon(
                [
                    Polygon([(1, 1), (0, 0), (0, 2), (1, 1)]),
                    Polygon([(2, 0), (1, 1), (2, 2), (2, 0)]),
                ]
            ),
            GeometryCollection(
                [Polygon([(2, 0), (0, 0), (0, 1), (2, 0)]), LineString([(0, 2), (0, 1)])]
            ),
            linestring,
            None,
        ]
    )

    assert result.repaired_rows.tolist() == [0, 1]
    assert result.valid_rows.tolist() == [2]
    assert result.null_rows.tolist() == [3]
    assert_geoseries_equal(GeoSeries(result.geometries), expected)


def test_geopandas_make_valid_uses_compacted_pipeline() -> None:
    polygon = Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)])
    series = geopandas.GeoSeries([polygon])
    expected = GeoSeries([MultiLineString([[(0, 0), (1, 1)], [(1, 1), (1, 2)]])])

    result = series.make_valid()

    assert_geoseries_equal(result, expected, check_geom_type=True)


def test_make_valid_benchmark_reports_repaired_rows() -> None:
    values = [
        Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)]),
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
    ]
    benchmark = benchmark_make_valid(values)

    assert benchmark.rows == 2
    assert benchmark.repaired_rows == 1
    assert benchmark.compact_elapsed_seconds >= 0.0
    assert benchmark.baseline_elapsed_seconds >= 0.0
