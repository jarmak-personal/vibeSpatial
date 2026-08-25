from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box

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
from vibespatial.api._native_result_core import NativeGeometryProvenance
from vibespatial.api.testing import assert_geoseries_equal
from vibespatial.constructive.linestring import (
    linestring_buffer_native_tabular_result,
    linestring_buffer_owned_array,
)
from vibespatial.constructive.polygon import polygon_buffer_native_tabular_result
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode, has_gpu_runtime, set_requested_mode
from vibespatial.runtime.fallbacks import StrictNativeFallbackError
from vibespatial.runtime.fusion import IntermediateDisposition
from vibespatial.runtime.residency import Residency
from vibespatial.testing import strict_native_environment


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
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

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
    assert dispatch_events[-1].selected is ExecutionMode.GPU


def test_geopandas_buffer_gpu_point_surface_claims_non_quad1() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = GeoSeries([Point(0, 0), Point(2, 2)])

    result = series.buffer(1.0, quad_segs=4)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)
    expected = GeoSeries(shapely.buffer(np.asarray(series.values._data, dtype=object), 1.0, quad_segs=4))

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
    assert dispatch_events[-1].selected is ExecutionMode.GPU


def test_geopandas_buffer_gpu_preserves_null_point_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geopandas.clear_dispatch_events()
    points = [Point(float(index), float(index % 17)) for index in range(600)]
    values = [None, *points[:300], None, *points[300:], None]
    series = GeoSeries(values)
    distances = np.full(len(series), 0.25, dtype=np.float64)

    result = series.buffer(distances, quad_segs=4)
    dispatch_events = geopandas.get_dispatch_events(clear=True)

    assert dispatch_events[-1].surface == "geopandas.array.buffer"
    assert dispatch_events[-1].selected is ExecutionMode.GPU
    assert result.iloc[0] is None
    assert result.iloc[301] is None
    assert result.iloc[-1] is None
    expected = shapely.buffer(np.asarray(values[1:4], dtype=object), 0.25, quad_segs=4)
    assert all(
        bool(shapely.equals_exact(actual, expected_geom, tolerance=1e-12))
        for actual, expected_geom in zip(result.iloc[1:4], expected, strict=True)
    )


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
def test_public_polygonal_buffer_strict_native_handles_indexed_polygon_multipolygon() -> None:
    import cupy as cp

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    values = [
        box(20, 0, 21, 1),
        MultiPolygon([box(0, 0, 1, 1), box(1.5, 0, 2.5, 1)]),
        box(10, 0, 11, 1),
        MultiPolygon([box(30, 0, 31, 1), box(34, 0, 35, 1)]),
    ]
    source = from_shapely_geometries(values, residency=Residency.DEVICE)
    indexed = source._device_indexed_take(
        cp.asarray([1, 2, 3], dtype=cp.int64),
        assume_unique_indices=True,
    )
    assert indexed.is_indexed_view
    series = GeoSeries(DeviceGeometryArray._from_owned(indexed))

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    with strict_native_environment():
        result = series.buffer(0.5, quad_segs=4)
    transfer_reasons = [
        event.reason for event in get_d2h_transfer_events(clear=True)
    ]
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    expected = shapely.buffer(
        np.asarray([values[1], values[2], values[3]], dtype=object),
        0.5,
        quad_segs=4,
    )
    assert isinstance(result.values, DeviceGeometryArray)
    assert not fallback_events
    assert dispatch_events[-1].implementation == "polygonal_part_group_buffer_gpu"
    assert dispatch_events[-1].selected is ExecutionMode.GPU
    assert not any(
        "active-part exact allocation" in reason for reason in transfer_reasons
    )
    for actual, expected_geom in zip(result, expected, strict=True):
        assert actual.symmetric_difference(expected_geom).area < 1e-9


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
def test_public_polygonal_buffer_strict_native_preserves_null_empty_and_row_radii() -> None:
    values = [
        box(0, 0, 1, 1),
        MultiPolygon([box(3, 0, 4, 1), box(4.5, 0, 5.5, 1)]),
        None,
        Polygon(),
        MultiPolygon([box(10, 0, 11, 1), box(13, 0, 14, 1)]),
    ]
    distances = np.asarray([0.25, 0.5, 1.0, 2.0, 0.1], dtype=np.float64)
    source = from_shapely_geometries(values, residency=Residency.DEVICE)
    series = GeoSeries(DeviceGeometryArray._from_owned(source))

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    with strict_native_environment():
        result = series.buffer(distances, quad_segs=4)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    expected = shapely.buffer(
        np.asarray(values, dtype=object),
        distances,
        quad_segs=4,
    )
    assert isinstance(result.values, DeviceGeometryArray)
    assert not fallback_events
    assert dispatch_events[-1].implementation == "polygonal_part_group_buffer_gpu"
    for actual, expected_geom in zip(result, expected, strict=True):
        if expected_geom is None:
            assert actual is None
        else:
            assert actual.symmetric_difference(expected_geom).area < 1e-9


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
def test_public_polygonal_buffer_strict_native_propagates_all_null_rows() -> None:
    source = from_shapely_geometries(
        [None, None, None],
        residency=Residency.DEVICE,
    )
    series = GeoSeries(DeviceGeometryArray._from_owned(source))

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    with strict_native_environment():
        result = series.buffer(2.0)

    assert isinstance(result.values, DeviceGeometryArray)
    assert result.isna().all()
    assert not geopandas.get_fallback_events(clear=True)
    assert geopandas.get_dispatch_events(clear=True)[-1].implementation == (
        "null_buffer_identity"
    )


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
@pytest.mark.parametrize(
    ("geometry", "distance"),
    [
        (
            MultiPolygon([box(0, 0, 1, 1), box(3, 0, 4, 1)]),
            -0.6,
        ),
        (
            MultiPolygon(
                [
                    Polygon(
                        [(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)],
                        [[(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]],
                    ),
                    box(10, 0, 11, 1),
                ]
            ),
            0.6,
        ),
        (
            MultiPolygon(
                [
                    Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)]),
                    box(4, 0, 5, 1),
                ]
            ),
            0.0,
        ),
    ],
    ids=("negative-collapse", "hole-collapse", "invalid-repair"),
)
def test_public_polygonal_buffer_strict_native_declines_unproven_topology(
    geometry,
    distance: float,
) -> None:
    source = from_shapely_geometries(
        [geometry],
        residency=Residency.DEVICE,
    )
    series = GeoSeries(DeviceGeometryArray._from_owned(source))

    geopandas.clear_fallback_events()
    with strict_native_environment(), pytest.raises(StrictNativeFallbackError):
        series.buffer(distance, quad_segs=4)
    fallback_events = geopandas.get_fallback_events(clear=True)

    assert len(fallback_events) == 1
    assert fallback_events[0].surface == "DeviceGeometryArray.buffer"


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
def test_public_polygonal_buffer_explicit_cpu_request_does_not_launch_gpu() -> None:
    geometry = MultiPolygon([box(0, 0, 1, 1), box(3, 0, 4, 1)])
    source = from_shapely_geometries([geometry], residency=Residency.HOST)
    series = GeoSeries(DeviceGeometryArray._from_owned(source))

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    with set_requested_mode(ExecutionMode.CPU):
        result = series.buffer(0.25, quad_segs=4)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    expected = shapely.buffer(geometry, 0.25, quad_segs=4)
    assert result.iloc[0].symmetric_difference(expected).area < 1e-9
    assert result.values.to_owned().residency is Residency.HOST
    assert len(fallback_events) == 1
    assert dispatch_events[-1].implementation == "shapely_fallback"
    assert dispatch_events[-1].selected is ExecutionMode.CPU


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
def test_public_polygonal_buffer_decline_exports_device_radii_at_cpu_boundary() -> None:
    import cupy as cp

    geometry = MultiPolygon([box(0, 0, 1, 1), box(3, 0, 4, 1)])
    source = from_shapely_geometries([geometry], residency=Residency.DEVICE)
    series = GeoSeries(DeviceGeometryArray._from_owned(source))
    distance = cp.asarray([-0.6], dtype=cp.float64)

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    result = series.buffer(distance, quad_segs=4)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    expected = shapely.buffer(geometry, -0.6, quad_segs=4)
    assert result.iloc[0].equals(expected)
    assert result.values.to_owned().residency is Residency.HOST
    assert len(fallback_events) == 1
    assert dispatch_events[-1].implementation == "shapely_fallback"
    assert dispatch_events[-1].selected is ExecutionMode.CPU


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
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


def test_buffer_native_tabular_results_cover_row_aligned_stroke_families() -> None:
    line_owned = from_shapely_geometries([LineString([(0, 0), (4, 0)])])
    polygon_owned = from_shapely_geometries([Polygon([(0, 0), (4, 0), (4, 4), (0, 0)])])

    line_result = linestring_buffer_native_tabular_result(
        line_owned,
        1.0,
        quad_segs=1,
        dispatch_mode=ExecutionMode.CPU,
        source_rows=np.asarray([3], dtype=np.int32),
        source_tokens=("line-source",),
    )
    polygon_result = polygon_buffer_native_tabular_result(
        polygon_owned,
        0.5,
        quad_segs=1,
        dispatch_mode=ExecutionMode.CPU,
        source_rows=np.asarray([5], dtype=np.int32),
        source_tokens=("polygon-source",),
    )

    for result, source_row, token in (
        (line_result, 3, "line-source"),
        (polygon_result, 5, "polygon-source"),
    ):
        assert result.geometry.row_count == 1
        assert result.column_order == ("geometry",)
        assert isinstance(result.provenance, NativeGeometryProvenance)
        assert result.provenance.operation == "buffer"
        assert result.provenance.source_tokens == (token,)
        assert result.provenance.source_rows.tolist() == [source_row]
        assert result.geometry_metadata is not None
        assert result.geometry_metadata.row_count == 1


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
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


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
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


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
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


def test_geopandas_offset_curve_partial_fallback_keeps_owned_result() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = GeoSeries(
        [
            LineString([(0, 0), (0, 2), (2, 2)]),
            LineString([(0, 0), (10, 0), (9, 0.1)]),
        ]
    )

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
    assert fallback_events
    assert fallback_events[-1].surface == "geopandas.array.offset_curve"
    assert "fallback_rows=1" in fallback_events[-1].detail
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.array.offset_curve"
    assert dispatch_events[-1].implementation == "owned_stroke_kernel"


def test_device_offset_curve_partial_fallback_avoids_full_shapely_dispatch() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    lines = np.asarray(
        [
            LineString([(0, 0), (0, 2), (2, 2)]),
            LineString([(0, 0), (10, 0), (9, 0.1)]),
        ],
        dtype=object,
    )
    device_array = DeviceGeometryArray._from_owned(from_shapely_geometries(lines.tolist()))

    result = device_array.offset_curve(1.0, join_style="mitre")
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)
    actual = np.asarray(result, dtype=object)
    expected = shapely.offset_curve(lines, 1.0, join_style="mitre")

    assert np.all(shapely.equals_exact(actual, expected, tolerance=1e-12))
    assert fallback_events
    assert fallback_events[-1].surface == "DeviceGeometryArray.offset_curve"
    assert "fallback_rows=1" in fallback_events[-1].detail
    assert dispatch_events
    assert dispatch_events[-1].surface == "DeviceGeometryArray.offset_curve"
    assert dispatch_events[-1].implementation == "offset_curve_owned_partial_fallback"


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
