from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point, Polygon

import vibespatial.api as geopandas
from vibespatial import (
    StrokeOperation,
    benchmark_offset_curve,
    benchmark_point_buffer,
    fusion_plan_for_stroke,
    offset_curve_owned,
    plan_stroke_kernel,
    point_buffer_owned,
)
from vibespatial.api import GeoSeries
from vibespatial.api.testing import assert_geoseries_equal
from vibespatial.constructive.linestring import linestring_buffer_owned_array
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.fusion import IntermediateDisposition


def test_stroke_plan_uses_prefix_sum_and_persistent_geometry_buffers() -> None:
    plan = plan_stroke_kernel(StrokeOperation.BUFFER)

    assert plan.stages[0].name == "expand_distances"
    assert plan.stages[-1].name == "emit_geometry"
    assert plan.stages[-1].disposition is IntermediateDisposition.PERSIST
    assert plan.stages[-1].geometry_producing is True


def test_stroke_fusion_plan_persists_geometry_buffers() -> None:
    fusion = fusion_plan_for_stroke("offset_curve")

    assert fusion.stages[-1].disposition is IntermediateDisposition.PERSIST
    assert fusion.stages[-1].steps[-1].output_name == "geometry_buffers"


def test_point_buffer_owned_matches_expected_diamond_for_quad1() -> None:
    result = point_buffer_owned([Point(0, 0)], 5.0, quad_segs=1)
    expected = GeoSeries([Polygon(((5, 0), (0, -5), (-5, 0), (0, 5), (5, 0)))])

    assert result.fast_rows.tolist() == [0]
    assert result.fallback_rows.size == 0
    assert_geoseries_equal(GeoSeries(result.geometries), expected)


def test_offset_curve_owned_matches_simple_mitre_case() -> None:
    line = LineString([(0, 0), (0, 2), (2, 2)])
    result = offset_curve_owned([line], 1.0, join_style="mitre")
    expected = GeoSeries([LineString([(-1, 0), (-1, 3), (2, 3)])])

    assert result.fast_rows.tolist() == [0]
    assert result.fallback_rows.size == 0
    assert_geoseries_equal(GeoSeries(result.geometries), expected)


def test_geopandas_buffer_mixed_rows_route_to_host_surface() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = GeoSeries([Point(0, 0), LineString([(0, 0), (1, 0)])])

    result = series.buffer(1.0, quad_segs=1)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)

    assert len(result) == 2
    assert not events
    assert dispatch_events
    assert dispatch_events[-1].surface in {
        "geopandas.array.buffer",
        "DeviceGeometryArray.buffer",
    }
    assert dispatch_events[-1].implementation == "shapely_host"


def test_geopandas_buffer_dispatch_claims_point_surface() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = GeoSeries([Point(0, 0), Point(2, 2)])

    result = series.buffer(1.0, quad_segs=1)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)
    expected = GeoSeries(shapely.buffer(np.asarray(series.values._data, dtype=object), 1.0, quad_segs=1))

    assert all(
        bool(shapely.equals_exact(left, right, tolerance=1e-12))
        for left, right in zip(result, expected, strict=True)
    )
    assert not fallback_events
    assert dispatch_events
    assert dispatch_events[-1].surface in {
        "geopandas.array.buffer",
        "DeviceGeometryArray.buffer",
    }
    assert dispatch_events[-1].implementation == "owned_stroke_kernel"


def test_linestring_buffer_owned_gpu_matches_shapely_after_normalize() -> None:
    line = LineString([(0, 0), (0, 2), (2, 2)])
    owned = from_shapely_geometries([line])

    result = linestring_buffer_owned_array(
        owned,
        1.0,
        quad_segs=1,
        dispatch_mode=ExecutionMode.GPU,
    )
    actual = result.to_shapely()[0]
    expected = shapely.buffer(line, 1.0, quad_segs=1)

    assert bool(shapely.is_valid(actual))
    assert bool(
        shapely.normalize(actual).equals_exact(
            shapely.normalize(expected),
            tolerance=1e-12,
        )
    )


def test_linestring_buffer_owned_gpu_two_point_grid_segments_match_shapely() -> None:
    lines = np.asarray(
        [
            LineString([(0, 0), (10, 0)]),
            LineString([(0, 0), (0, 10)]),
        ],
        dtype=object,
    )
    owned = from_shapely_geometries(lines.tolist())

    result = linestring_buffer_owned_array(
        owned,
        1.0,
        quad_segs=16,
        dispatch_mode=ExecutionMode.GPU,
    )
    actual = np.asarray(result.to_shapely(), dtype=object)
    expected = shapely.buffer(lines, 1.0, quad_segs=16)

    assert bool(shapely.is_valid(actual).all())
    assert np.all(
        [
            bool(
                shapely.normalize(left).equals_exact(
                    shapely.normalize(right),
                    tolerance=1e-12,
                )
            )
            for left, right in zip(actual, expected, strict=True)
        ]
    )


def test_linestring_buffer_owned_gpu_square_cap_elbow_matches_shapely_after_normalize() -> None:
    line = LineString([(0, 0), (10, 0), (10, 10)])
    owned = from_shapely_geometries([line])

    result = linestring_buffer_owned_array(
        owned,
        1.0,
        quad_segs=4,
        cap_style="square",
        join_style="round",
        dispatch_mode=ExecutionMode.GPU,
    )
    actual = result.to_shapely()[0]
    expected = shapely.buffer(
        line,
        1.0,
        quad_segs=4,
        cap_style="square",
        join_style="round",
    )

    assert len(actual.exterior.coords) == len(expected.exterior.coords)
    assert bool(
        shapely.normalize(actual).equals_exact(
            shapely.normalize(expected),
            tolerance=1e-12,
        )
    )


def test_linestring_buffer_owned_gpu_real_vegetation_corridor_line_has_stable_vertex_count() -> None:
    line = LineString(
        [
            (0.0, 1000.0),
            (90.9090909090909, 1000.0),
            (181.8181818181818, 1000.0),
            (272.72727272727275, 997.297227770596),
            (363.6363636363636, 935.678498916312),
            (454.5454545454545, 879.0356008412407),
            (545.4545454545455, 841.2366904842921),
            (636.3636363636364, 831.5362593839325),
            (727.2727272727273, 852.3093113878917),
            (818.1818181818181, 898.4698788800844),
            (909.090909090909, 958.7162450323488),
            (1000.0, 1000.0),
        ]
    )
    owned = from_shapely_geometries([line])

    result = linestring_buffer_owned_array(
        owned,
        10.0,
        quad_segs=16,
        dispatch_mode=ExecutionMode.GPU,
    )
    actual = result.to_shapely()[0]

    assert bool(shapely.is_valid(actual))
    assert len(actual.exterior.coords) == 111


def test_geopandas_offset_curve_fallback_is_observable_for_multiline() -> None:
    geopandas.clear_fallback_events()
    series = GeoSeries(
        [
            LineString([(0, 0), (0, 2), (2, 2)]),
            MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 0)]]),
        ]
    )

    result = series.offset_curve(1.0, join_style="mitre")
    events = geopandas.get_fallback_events(clear=True)

    assert len(result) == 2
    assert events
    assert events[-1].surface == "geopandas.array.offset_curve"
    assert "explicit CPU fallback" in events[-1].reason


def test_geopandas_offset_curve_dispatch_claims_linestring_surface() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = GeoSeries([LineString([(0, 0), (0, 2), (2, 2)]), LineString([(1, 0), (1, 2), (3, 2)])])

    result = series.offset_curve(1.0, join_style="mitre")
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)
    expected = GeoSeries(
        shapely.offset_curve(
            np.asarray(series.values._data, dtype=object),
            1.0,
            join_style="mitre",
        )
    )

    assert_geoseries_equal(result, expected)
    assert not fallback_events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.array.offset_curve"
    assert dispatch_events[-1].implementation == "owned_stroke_kernel"


def test_stroke_benchmarks_report_row_counts() -> None:
    point_benchmark = benchmark_point_buffer([Point(0, 0), Point(1, 1)], distance=1.0, quad_segs=1)
    offset_benchmark = benchmark_offset_curve(
        [LineString([(0, 0), (0, 2), (2, 2)])],
        distance=1.0,
        join_style="mitre",
    )

    assert point_benchmark.rows == 2
    assert offset_benchmark.rows == 1
    assert point_benchmark.owned_elapsed_seconds >= 0.0
    assert offset_benchmark.shapely_elapsed_seconds >= 0.0
