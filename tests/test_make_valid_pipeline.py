from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon

import vibespatial.api as geopandas
from vibespatial import (
    benchmark_make_valid,
    fusion_plan_for_make_valid,
    make_valid_owned,
    plan_make_valid_pipeline,
)
from vibespatial.api import GeoSeries
from vibespatial.api.testing import assert_geoseries_equal
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.fusion import IntermediateDisposition


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


def test_make_valid_gpu_detection_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    owned = from_shapely_geometries([bowtie])

    def _boom(*args, **kwargs):
        raise RuntimeError("gpu-detect-boom")

    monkeypatch.setattr(
        "vibespatial.constructive.validity.is_valid_owned",
        lambda owned, **kwargs: np.array([False], dtype=bool),
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._detect_self_intersections_gpu",
        _boom,
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline.plan_dispatch_selection",
        lambda *args, **kwargs: SimpleNamespace(
            selected=ExecutionMode.GPU,
            requested=ExecutionMode.GPU,
            precision_plan=None,
            reason="test",
        ),
    )

    with pytest.raises(RuntimeError, match="gpu-detect-boom"):
        make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)


def test_make_valid_gpu_repair_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    owned = from_shapely_geometries([bowtie])

    def _boom(*args, **kwargs):
        raise RuntimeError("gpu-repair-boom")

    monkeypatch.setattr(
        "vibespatial.constructive.validity.is_valid_owned",
        lambda owned, **kwargs: np.array([False], dtype=bool),
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._detect_self_intersections_gpu",
        lambda owned, valid_mask: valid_mask,
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._make_valid_gpu_repair",
        _boom,
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline.plan_dispatch_selection",
        lambda *args, **kwargs: SimpleNamespace(
            selected=ExecutionMode.GPU,
            requested=ExecutionMode.GPU,
            precision_plan=None,
            reason="test",
        ),
    )

    with pytest.raises(RuntimeError, match="gpu-repair-boom"):
        make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)
