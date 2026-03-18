from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import vibespatial.api as geopandas
import vibespatial.io_arrow as io_arrow
import vibespatial.io_geoparquet as io_geoparquet
from vibespatial import (
    BufferSharingMode,
    ExecutionMode,
    benchmark_geoarrow_bridge,
    benchmark_geoparquet_planner,
    benchmark_geoparquet_scan_engine,
    benchmark_native_geometry_codec,
    benchmark_wkb_bridge,
    build_geoparquet_metadata_summary,
    decode_owned_geoarrow,
    decode_wkb_owned,
    encode_owned_geoarrow,
    encode_owned_geoarrow_array,
    encode_wkb_owned,
    has_gpu_runtime,
    has_pylibcudf_support,
    plan_geoparquet_scan,
    plan_wkb_partition,
    read_geoparquet,
    read_geoparquet_owned,
    select_row_groups,
    write_geoparquet,
)
from vibespatial.device_geometry_array import DeviceGeometryArray
from vibespatial.owned_geometry import DiagnosticKind, from_shapely_geometries
from vibespatial.point_constructive import clip_points_rect_owned
from vibespatial.residency import Residency, TransferTrigger


def _diagnostic_totals(*values) -> tuple[int, int]:
    seen: set[int] = set()
    transfer_count = 0
    materialization_count = 0

    def visit(value) -> None:
        nonlocal transfer_count, materialization_count
        if isinstance(value, DeviceGeometryArray):
            owned = value.to_owned()
            key = id(owned)
            if key not in seen:
                seen.add(key)
                transfer_count += sum(event.kind == DiagnosticKind.TRANSFER for event in owned.diagnostics)
                materialization_count += sum(
                    event.kind == DiagnosticKind.MATERIALIZATION for event in owned.diagnostics
                )
            return
        if isinstance(value, geopandas.GeoDataFrame):
            visit(value.geometry.values)
            return
        if isinstance(value, geopandas.GeoSeries):
            visit(value.values)
            return
        geometry = getattr(value, "geometry", None)
        if geometry is not None and geometry is not value:
            visit(geometry)
        values_attr = getattr(value, "values", None)
        if isinstance(values_attr, DeviceGeometryArray):
            visit(values_attr)

    for value in values:
        visit(value)
    return transfer_count, materialization_count


def _take_dga_frame(frame, indices: np.ndarray):
    indices = np.asarray(indices, dtype=np.intp)
    geometry_name = frame.geometry.name
    geometry = frame.geometry.values.take(indices)
    data: dict[str, object] = {}
    for column in frame.columns:
        if column == geometry_name:
            data[column] = pd.Series(geometry, copy=False, name=geometry_name)
        else:
            data[column] = frame[column].to_numpy(copy=False)[indices]
    result = pd.DataFrame(data, copy=False)
    result.__class__ = geopandas.GeoDataFrame
    result._geometry_column_name = geometry_name
    result[geometry_name].array.crs = frame.crs
    return result


def test_owned_geoarrow_bridge_roundtrips_and_is_observable() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    owned = from_shapely_geometries([Point(0, 0), None, Polygon()])

    view = encode_owned_geoarrow(owned)
    roundtripped = decode_owned_geoarrow(view)
    events = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert roundtripped.to_shapely()[0].equals(Point(0, 0))
    assert roundtripped.to_shapely()[1] is None
    assert roundtripped.to_shapely()[2].is_empty
    assert [event.surface for event in events[-2:]] == [
        "vibespatial.io.geoarrow",
        "vibespatial.io.geoarrow",
    ]
    assert len(fallbacks) >= 2
    assert "device-side GeoArrow" in fallbacks[-1].reason
    assert roundtripped.geoarrow_backed is True
    assert roundtripped.shares_geoarrow_memory is True


def test_from_geoarrow_auto_normalizes_misaligned_buffers() -> None:
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
    view = owned.to_geoarrow(sharing=BufferSharingMode.SHARE)
    misaligned = type(view)(
        validity=view.validity.astype(np.int8),
        tags=view.tags.astype(np.int16),
        family_row_offsets=view.family_row_offsets.astype(np.int64),
        families={
            family: type(buffer)(
                family=buffer.family,
                x=buffer.x.astype(np.float32),
                y=buffer.y.astype(np.float32),
                geometry_offsets=buffer.geometry_offsets.astype(np.int64),
                empty_mask=buffer.empty_mask.astype(np.int8),
                part_offsets=None if buffer.part_offsets is None else buffer.part_offsets.astype(np.int64),
                ring_offsets=None if buffer.ring_offsets is None else buffer.ring_offsets.astype(np.int64),
                bounds=buffer.bounds,
                shares_memory=False,
            )
            for family, buffer in view.families.items()
        },
        shares_memory=False,
    )

    adopted = decode_owned_geoarrow(misaligned)

    point_family = next(iter(adopted.families))
    assert adopted.validity.dtype == np.bool_
    assert adopted.tags.dtype == np.int8
    assert adopted.family_row_offsets.dtype == np.int32
    assert adopted.families[point_family].x.dtype == np.float64
    assert not np.shares_memory(misaligned.validity, adopted.validity)
    assert adopted.shares_geoarrow_memory is False


def test_wkb_bridge_roundtrips_and_records_explicit_fallback() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    source = from_shapely_geometries([Point(1, 2), Point(3, 4)])

    encoded = encode_wkb_owned(source)
    decoded = decode_wkb_owned(encoded)
    events = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert decoded.to_shapely()[0].equals(Point(1, 2))
    assert decoded.to_shapely()[1].equals(Point(3, 4))
    assert [event.surface for event in events[-2:]] == [
        "vibespatial.io.wkb",
        "vibespatial.io.wkb",
    ]
    assert fallbacks == []


def test_wkb_partition_reports_fallback_pool_for_big_endian_rows() -> None:
    values = [Point(0, 0).wkb, bytes.fromhex("00000000013ff00000000000004000000000000000")]

    plan = plan_wkb_partition(values)

    assert plan.native_rows == 1
    assert plan.fallback_rows == 1
    assert plan.fallback_indexes == (1,)


def test_wkb_bridge_falls_back_explicitly_for_big_endian_point() -> None:
    geopandas.clear_fallback_events()
    values = [bytes.fromhex("00000000013ff00000000000004000000000000000")]

    decoded = decode_wkb_owned(values)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert decoded.to_shapely()[0].equals(Point(1, 2))
    assert len(fallbacks) == 1
    assert "fallback pool" in fallbacks[0].detail


def test_wkb_bridge_roundtrips_polygon_fast_path_without_fallback() -> None:
    geopandas.clear_fallback_events()
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
    source = from_shapely_geometries([polygon, polygon.buffer(1)])

    encoded = encode_wkb_owned(source)
    decoded = decode_wkb_owned(encoded)

    assert decoded.to_shapely()[0].equals(polygon)
    assert geopandas.get_fallback_events(clear=True) == []


def test_geoparquet_scan_plan_prefers_bbox_pushdown_when_covering_exists() -> None:
    metadata = {
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "polygon",
                "covering": {
                    "bbox": {
                        "xmin": ["bbox", "xmin"],
                        "ymin": ["bbox", "ymin"],
                        "xmax": ["bbox", "xmax"],
                        "ymax": ["bbox", "ymax"],
                    }
                },
            }
        },
    }

    plan = plan_geoparquet_scan(bbox=(0.0, 0.0, 1.0, 1.0), geo_metadata=metadata)

    assert plan.row_group_pushdown is True
    assert plan.uses_covering_bbox is True
    assert plan.uses_pylibcudf is has_pylibcudf_support()


def test_geoparquet_scan_plan_supports_point_encoding_pushdown_without_covering() -> None:
    metadata = {
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "point",
            }
        },
    }

    plan = plan_geoparquet_scan(bbox=(0.0, 0.0, 1.0, 1.0), geo_metadata=metadata)

    assert plan.row_group_pushdown is True
    assert plan.uses_covering_bbox is False
    assert plan.uses_point_encoding_pushdown is True


def test_geoparquet_scan_plan_uses_metadata_summary_for_row_group_selection() -> None:
    summary = build_geoparquet_metadata_summary(
        source="covering_bbox",
        row_group_rows=[100, 100, 100, 100],
        xmin=[0.0, 10.0, 20.0, 30.0],
        ymin=[0.0, 10.0, 20.0, 30.0],
        xmax=[5.0, 15.0, 25.0, 35.0],
        ymax=[5.0, 15.0, 25.0, 35.0],
    )

    plan = plan_geoparquet_scan(
        bbox=(12.0, 12.0, 22.0, 22.0),
        geo_metadata={
            "primary_column": "geometry",
            "columns": {"geometry": {"encoding": "polygon", "covering": {"bbox": {}}}},
        },
        metadata_summary=summary,
    )

    assert plan.row_group_pushdown is True
    assert plan.metadata_summary_available is True
    assert plan.metadata_source == "covering_bbox"
    assert plan.selected_row_groups == (1, 2)
    assert plan.available_row_groups == 4
    assert plan.decoded_row_fraction_estimate == 0.5


def test_geoparquet_prune_strategies_return_same_selection() -> None:
    summary = build_geoparquet_metadata_summary(
        source="covering_bbox",
        row_group_rows=[100, 100, 100, 100],
        xmin=[0.0, 10.0, 20.0, 30.0],
        ymin=[0.0, 10.0, 20.0, 30.0],
        xmax=[5.0, 15.0, 25.0, 35.0],
        ymax=[5.0, 15.0, 25.0, 35.0],
    )
    bbox = (12.0, 12.0, 22.0, 22.0)

    loop_result = select_row_groups(summary, bbox, strategy="loop")
    vectorized_result = select_row_groups(summary, bbox, strategy="vectorized")
    auto_result = select_row_groups(summary, bbox, strategy="auto")

    assert loop_result.selected_row_groups == (1, 2)
    assert vectorized_result.selected_row_groups == loop_result.selected_row_groups
    assert auto_result.selected_row_groups == loop_result.selected_row_groups
    assert loop_result.decoded_row_fraction == 0.5
    assert vectorized_result.pruned_row_group_fraction == 0.5


def test_geoparquet_planner_benchmark_reports_all_strategies() -> None:
    summary = build_geoparquet_metadata_summary(
        source="covering_bbox",
        row_group_rows=[100] * 128,
        xmin=list(range(128)),
        ymin=list(range(128)),
        xmax=[value + 0.5 for value in range(128)],
        ymax=[value + 0.5 for value in range(128)],
    )

    results = benchmark_geoparquet_planner(summary, (16.0, 16.0, 24.0, 24.0), repeat=2)
    by_strategy = {result.strategy: result for result in results}

    assert set(by_strategy) == {"full_scan", "loop", "vectorized", "auto"}
    assert by_strategy["full_scan"].decoded_row_fraction == 1.0
    assert by_strategy["vectorized"].selected_row_groups == by_strategy["auto"].selected_row_groups


def test_geoarrow_bridge_benchmark_reports_copy_vs_zero_copy() -> None:
    benchmarks = benchmark_geoarrow_bridge(operation="decode", geometry_type="point", rows=1_000, repeat=2)
    by_mode = {item.sharing: item for item in benchmarks}

    assert set(by_mode) == {"copy", "auto", "share"}
    assert by_mode["copy"].shares_memory is False
    assert by_mode["auto"].shares_memory is True
    assert by_mode["share"].shares_memory is True


def test_read_geoparquet_uses_planned_row_groups_when_metadata_summary_exists(monkeypatch) -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()

    captured: dict[str, object] = {}
    summary = build_geoparquet_metadata_summary(
        source="covering_bbox",
        row_group_rows=[100, 100, 100, 100],
        xmin=[0.0, 10.0, 20.0, 30.0],
        ymin=[0.0, 10.0, 20.0, 30.0],
        xmax=[5.0, 15.0, 25.0, 35.0],
        ymax=[5.0, 15.0, 25.0, 35.0],
    )

    monkeypatch.setattr(io_geoparquet, "has_pyarrow_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: False)
    monkeypatch.setattr(
        io_geoparquet,
        "_build_geoparquet_metadata_summary_from_pyarrow",
        lambda path, filesystem, geo_metadata: summary,
    )

    def fake_read_geoparquet_with_pyarrow(*args, **kwargs):
        captured["row_groups"] = kwargs["row_groups"]
        return "ok"

    monkeypatch.setattr(io_geoparquet, "_read_geoparquet_with_pyarrow", fake_read_geoparquet_with_pyarrow)
    monkeypatch.setattr(
        io_geoparquet,
        "_load_geoparquet_metadata",
        lambda path, filesystem=None, storage_options=None: (
            filesystem,
            path,
            {b"geo": b"1"},
            {
                "primary_column": "geometry",
                "columns": {
                    "geometry": {
                        "encoding": "polygon",
                        "covering": {"bbox": {"xmin": ["bbox", "xmin"]}},
                    }
                },
            },
        ),
    )

    result = io_arrow.read_geoparquet("sample.parquet", bbox=(12.0, 12.0, 22.0, 22.0))
    events = geopandas.get_dispatch_events(clear=True)

    assert result == "ok"
    assert captured["row_groups"] == (1, 2)
    assert events[-1].operation == "row_group_pushdown"


def test_read_geoparquet_owned_roundtrips_geoarrow_point_file(tmp_path) -> None:
    gdf = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]})
    path = tmp_path / "points.parquet"
    gdf.to_parquet(path, geometry_encoding="geoarrow")

    owned = read_geoparquet_owned(path, backend="cpu")

    assert owned.row_count == 3
    assert owned.to_shapely()[1].equals(Point(1, 1))


def test_read_geoparquet_owned_roundtrips_geoarrow_polygon_file(tmp_path) -> None:
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
    gdf = geopandas.GeoDataFrame({"geometry": [polygon, polygon.buffer(1)]})
    path = tmp_path / "polygons.parquet"
    gdf.to_parquet(path, geometry_encoding="geoarrow")

    owned = read_geoparquet_owned(path, backend="cpu")

    assert owned.row_count == 2
    assert owned.to_shapely()[0].equals(polygon)


def test_read_geoparquet_owned_uses_chunked_backend_and_concatenates(monkeypatch) -> None:
    geopandas.clear_dispatch_events()
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)])
    view = owned.to_geoarrow(sharing=BufferSharingMode.SHARE)
    import pyarrow as pa

    coords = pa.FixedSizeListArray.from_arrays(pa.array(np.column_stack([view.families[next(iter(view.families))].x, view.families[next(iter(view.families))].y]).ravel()), 2)

    def build_table(start: int, stop: int):
        field = pa.field("geometry", coords.type, metadata={b"ARROW:extension:name": b"geoarrow.point", b"ARROW:extension:metadata": b"{}"})
        arr = pa.FixedSizeListArray.from_arrays(pa.array(np.column_stack([np.arange(start, stop, dtype=float), np.arange(start, stop, dtype=float)]).ravel()), 2)
        return pa.table([arr], schema=pa.schema([field]))

    monkeypatch.setattr(io_geoparquet, "has_pyarrow_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(
        io_geoparquet,
        "_load_geoparquet_metadata",
        lambda path, filesystem=None, storage_options=None: (
            filesystem,
            path,
            {b"geo": b"1"},
            {"primary_column": "geometry", "columns": {"geometry": {"encoding": "point"}}},
        ),
    )
    monkeypatch.setattr(
        io_geoparquet,
        "_build_geoparquet_metadata_summary_from_pyarrow",
        lambda path, filesystem, geo_metadata: build_geoparquet_metadata_summary(
            source="point_encoding",
            row_group_rows=[2, 2],
            xmin=[0.0, 2.0],
            ymin=[0.0, 2.0],
            xmax=[1.0, 3.0],
            ymax=[1.0, 3.0],
        ),
    )
    monkeypatch.setattr(
        io_geoparquet,
        "_read_geoparquet_table_with_pylibcudf",
        lambda path, columns=None, row_groups=None, filesystem=None: build_table(
            row_groups[0] * 2, row_groups[-1] * 2 + 2
        ),
    )

    result = read_geoparquet_owned("sample.parquet", backend="gpu", chunk_rows=2)
    events = geopandas.get_dispatch_events(clear=True)

    assert result.row_count == 4
    assert result.to_shapely()[3].equals(Point(3, 3))
    assert events[-1].selected == "gpu"


def test_read_geoparquet_gpu_path_returns_dga_without_geometry_to_arrow(monkeypatch) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    geopandas.clear_dispatch_events()

    class FakeGpuTable:
        column_names = ["value", "geometry", "name"]

        def __init__(self) -> None:
            self._columns = [object(), object(), object()]

        def columns(self):
            return self._columns

        def to_arrow(self):
            raise AssertionError("geometry GPU path must not call to_arrow()")

    fake_table = FakeGpuTable()
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2)])

    monkeypatch.setattr(io_geoparquet, "has_pyarrow_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "_supports_pylibcudf_geoparquet_read", lambda *args, **kwargs: (True, "test"))
    monkeypatch.setattr(
        io_geoparquet,
        "_load_geoparquet_metadata",
        lambda path, filesystem=None, storage_options=None: (
            filesystem,
            path,
            {b"geo": b"1"},
            {
                "primary_column": "geometry",
                "columns": {
                    "geometry": {"encoding": "point", "crs": "EPSG:4326"},
                },
            },
        ),
    )
    monkeypatch.setattr(io_geoparquet, "_read_geoparquet_table_with_pylibcudf", lambda *args, **kwargs: fake_table)
    monkeypatch.setattr(
        io_geoparquet,
        "_read_non_geometry_geoparquet_columns_as_arrow",
        lambda *args, **kwargs: pa.table({"value": [10, 20, 30], "name": ["a", "b", "c"]}),
    )
    monkeypatch.setattr(io_geoparquet, "_decode_pylibcudf_geoparquet_column_to_owned", lambda column, encoding: owned)
    monkeypatch.setattr(
        io_geoparquet,
        "_build_geoparquet_metadata_summary_from_pyarrow",
        lambda path, filesystem, geo_metadata: None,
    )

    original_read_schema = pq.read_schema
    monkeypatch.setattr(
        pq,
        "read_schema",
        lambda path, filesystem=None: pa.schema(
            [
                pa.field("value", pa.int64()),
                pa.field("geometry", pa.binary()),
                pa.field("name", pa.string()),
            ],
            metadata={b"PANDAS_ATTRS": b'{"source": "gpu"}'},
        ),
    )

    try:
        result = io_arrow.read_geoparquet("sample.parquet")
    finally:
        monkeypatch.setattr(pq, "read_schema", original_read_schema)

    assert list(result.columns) == ["value", "geometry", "name"]
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert list(result["value"]) == [10, 20, 30]
    assert list(result["name"]) == ["a", "b", "c"]
    assert result.attrs == {"source": "gpu"}


def test_read_geoparquet_gpu_backend_returns_dga_with_live_pylibcudf(tmp_path) -> None:
    if not has_pylibcudf_support():
        return

    gdf = geopandas.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )
    path = tmp_path / "gpu-read-public.parquet"
    gdf.to_parquet(path, geometry_encoding="geoarrow")

    result = io_arrow.read_geoparquet(path)

    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert list(result["value"]) == [10, 20, 30]
    assert result.geometry.iloc[2].equals(Point(2, 2))


def test_concatenate_owned_arrays_preserves_device_backed_point_chunks() -> None:
    if not has_gpu_runtime():
        return

    first = clip_points_rect_owned(
        from_shapely_geometries([Point(0, 0), Point(1, 1)]),
        -1.0,
        -1.0,
        2.0,
        2.0,
        dispatch_mode=ExecutionMode.GPU,
    )
    second = clip_points_rect_owned(
        from_shapely_geometries([Point(2, 2), Point(3, 3)]),
        1.0,
        1.0,
        4.0,
        4.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    combined = io_arrow._concatenate_owned_arrays([first, second])

    assert combined.residency is Residency.DEVICE
    assert combined.device_state is not None
    assert combined.families[next(iter(combined.families))].host_materialized is False
    assert combined.row_count == 4
    assert combined.to_shapely()[3].equals(Point(3, 3))


def test_geoparquet_scan_engine_benchmark_runs(tmp_path) -> None:
    benchmark = benchmark_geoparquet_scan_engine(
        geometry_type="point",
        rows=1_000,
        geometry_encoding="geoarrow",
        backend="cpu",
        repeat=1,
    )

    assert benchmark.rows == 1_000
    assert benchmark.rows_per_second > 0


def test_encode_owned_geoarrow_array_roundtrips_point_family() -> None:
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2)])

    field, geom_arr = encode_owned_geoarrow_array(owned, field_name="geometry")
    restored = io_arrow._decode_geoarrow_array_to_owned(field, geom_arr)

    assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.point"
    assert restored.to_shapely()[1].equals(Point(1, 1))


def test_encode_owned_geoarrow_array_roundtrips_polygon_family() -> None:
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
    owned = from_shapely_geometries([polygon, polygon.buffer(1)])

    field, geom_arr = encode_owned_geoarrow_array(owned, field_name="geometry")
    restored = io_arrow._decode_geoarrow_array_to_owned(field, geom_arr)

    assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.polygon"
    assert restored.to_shapely()[0].equals(polygon)


def test_geoseries_to_arrow_mixed_family_records_explicit_fallback() -> None:
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(0, 0), Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])])

    arrow_array = io_arrow.geoseries_to_arrow(series, geometry_encoding="geoarrow")
    fallbacks = geopandas.get_fallback_events(clear=True)
    schema_capsule, _ = arrow_array.__arrow_c_array__()
    import pyarrow as pa

    field = pa.Field._import_from_c_capsule(schema_capsule)

    assert arrow_array is not None
    assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"
    assert any("geometry mix" in event.reason or "geometry mix" in event.detail for event in fallbacks)


def test_geoseries_to_arrow_uses_native_point_geoarrow_fast_path() -> None:
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2)])

    arrow_array = io_arrow.geoseries_to_arrow(series, geometry_encoding="geoarrow")
    schema_capsule, array_capsule = arrow_array.__arrow_c_array__()
    import pyarrow as pa

    field = pa.Field._import_from_c_capsule(schema_capsule)
    pa_array = pa.Array._import_from_c_capsule(field.__arrow_c_schema__(), array_capsule)

    assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.point"
    assert pa_array.type.list_size == 2
    assert geopandas.get_fallback_events(clear=True) == []


def test_geodataframe_to_arrow_uses_native_point_geoarrow_fast_path() -> None:
    geopandas.clear_fallback_events()
    gdf = geopandas.GeoDataFrame(
        {"value": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]}
    )

    arrow_table = io_arrow.geodataframe_to_arrow(gdf, geometry_encoding="geoarrow")
    import pyarrow as pa

    table = pa.table(arrow_table)
    field = table.schema.field("geometry")

    assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.point"
    assert geopandas.get_fallback_events(clear=True) == []


def test_native_geometry_benchmark_reports_host_and_native_paths() -> None:
    benchmarks = benchmark_native_geometry_codec(operation="encode", geometry_type="point", rows=1_000, repeat=1)
    implementations = {item.implementation for item in benchmarks}

    assert implementations == {"host_bridge", "native_owned"}


def test_wkb_bridge_benchmark_reports_host_and_native_paths() -> None:
    benchmarks = benchmark_wkb_bridge(operation="decode", geometry_type="point", rows=1_000, repeat=1)
    implementations = {item.implementation for item in benchmarks}

    assert implementations == {"host_bridge", "native_owned"}


def test_read_geoparquet_owned_gpu_backend_with_live_pylibcudf(tmp_path) -> None:
    if not has_pylibcudf_support():
        return

    gdf = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]})
    path = tmp_path / "gpu-points.parquet"
    gdf.to_parquet(path, geometry_encoding="geoarrow")

    owned = read_geoparquet_owned(path, backend="gpu")

    assert owned.row_count == 3
    assert owned.residency is Residency.DEVICE
    assert owned.device_state is not None
    assert owned.families[next(iter(owned.families))].host_materialized is False
    assert owned.to_shapely()[2].equals(Point(2, 2))
    assert owned.families[next(iter(owned.families))].host_materialized is True


def test_read_geoparquet_table_with_pylibcudf_preserves_source_schema(tmp_path) -> None:
    if not has_pylibcudf_support() or not io_arrow.has_pyarrow_support():
        return

    gdf = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]})
    path = tmp_path / "gpu-schema.parquet"
    gdf.to_parquet(path, geometry_encoding="geoarrow")

    table = io_arrow._read_geoparquet_table_with_pylibcudf(path)
    arrow_table = table.to_arrow()

    assert table.num_columns() == 1
    assert arrow_table.schema.field(0).type.num_fields == 2


def test_read_geoparquet_table_with_pylibcudf_disables_unneeded_schema_metadata() -> None:
    calls: list[tuple[str, object]] = []

    class FakeOptions:
        def __init__(self) -> None:
            self.columns = None
            self.row_groups = None

        def set_columns(self, columns) -> None:
            self.columns = list(columns)

        def set_row_groups(self, row_groups) -> None:
            self.row_groups = row_groups

    class FakeBuilder:
        def __init__(self, source) -> None:
            calls.append(("source", source.paths))
            self.options = FakeOptions()

        def use_arrow_schema(self, enabled: bool):
            calls.append(("use_arrow_schema", enabled))
            return self

        def use_pandas_metadata(self, enabled: bool):
            calls.append(("use_pandas_metadata", enabled))
            return self

        def build(self) -> FakeOptions:
            calls.append(("build", None))
            return self.options

    class FakeSourceInfo:
        def __init__(self, paths) -> None:
            self.paths = list(paths)

    fake_result = types.SimpleNamespace(tbl="device-table")
    fake_module = types.SimpleNamespace(
        io=types.SimpleNamespace(
            types=types.SimpleNamespace(SourceInfo=FakeSourceInfo),
            parquet=types.SimpleNamespace(
                ParquetReaderOptions=types.SimpleNamespace(builder=lambda source: FakeBuilder(source)),
                read_parquet=lambda options: fake_result,
            ),
        )
    )

    original = sys.modules.get("pylibcudf")
    sys.modules["pylibcudf"] = fake_module
    try:
        table = io_arrow._read_geoparquet_table_with_pylibcudf(
            "sample.parquet",
            columns=["geometry"],
            row_groups=(1, 3),
        )
    finally:
        if original is None:
            del sys.modules["pylibcudf"]
        else:
            sys.modules["pylibcudf"] = original

    assert table == "device-table"
    assert ("use_arrow_schema", False) in calls
    assert ("use_pandas_metadata", False) in calls


# ---------------------------------------------------------------------------
# DGA-backed write path roundtrip tests
# ---------------------------------------------------------------------------


def _make_dga_gdf(geoms, crs="EPSG:4326"):
    """Build a GeoDataFrame backed by a DeviceGeometryArray."""
    owned = from_shapely_geometries(geoms)
    dga = DeviceGeometryArray._from_owned(owned)
    gs = geopandas.GeoSeries(dga, crs=crs)
    return geopandas.GeoDataFrame({"geometry": gs, "idx": range(len(geoms))})


def _make_device_dga_gdf(geoms, crs="EPSG:4326"):
    import pandas as pd

    owned = from_shapely_geometries(geoms)
    if not has_gpu_runtime():
        return geopandas.GeoDataFrame({"geometry": geopandas.GeoSeries(DeviceGeometryArray._from_owned(owned), crs=crs), "idx": range(len(geoms))}), owned
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test device-resident native parquet write path",
    )
    owned.diagnostics.clear()
    dga = DeviceGeometryArray._from_owned(owned, crs=crs)
    geometry_series = pd.Series(dga, copy=False, name="geometry")
    gdf = pd.DataFrame({"geometry": geometry_series, "idx": range(len(geoms))}, copy=False)
    gdf.__class__ = geopandas.GeoDataFrame
    gdf._geometry_column_name = "geometry"
    gdf["geometry"].array.crs = crs
    return gdf, owned


def test_write_geoparquet_roundtrips_point_geoarrow(tmp_path) -> None:
    """Point DGA → GeoArrow write → read yields identical geometry."""
    pts = [Point(i, i * 2) for i in range(5)]
    gdf = _make_dga_gdf(pts)
    path = tmp_path / "pts_ga.parquet"
    write_geoparquet(gdf, path, geometry_encoding="geoarrow")
    result = geopandas.read_parquet(path)
    assert len(result) == 5
    for i, pt in enumerate(pts):
        assert result.geometry.iloc[i].equals(pt)


def test_write_geoparquet_roundtrips_point_wkb(tmp_path) -> None:
    """Point DGA → WKB write → read yields identical geometry."""
    pts = [Point(i, i * 2) for i in range(5)]
    gdf = _make_dga_gdf(pts)
    path = tmp_path / "pts_wkb.parquet"
    write_geoparquet(gdf, path, geometry_encoding="WKB")
    result = geopandas.read_parquet(path)
    assert len(result) == 5
    for i, pt in enumerate(pts):
        assert result.geometry.iloc[i].equals(pt)


def test_write_geoparquet_roundtrips_polygon_geoarrow(tmp_path) -> None:
    """Polygon DGA → GeoArrow write → read roundtrip."""
    polys = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)]) for i in range(4)]
    gdf = _make_dga_gdf(polys)
    path = tmp_path / "polys_ga.parquet"
    write_geoparquet(gdf, path, geometry_encoding="geoarrow")
    result = geopandas.read_parquet(path)
    assert len(result) == 4
    for i, poly in enumerate(polys):
        assert result.geometry.iloc[i].equals(poly)


def test_write_geoparquet_roundtrips_polygon_wkb(tmp_path) -> None:
    """Polygon DGA → WKB write → read roundtrip."""
    polys = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)]) for i in range(4)]
    gdf = _make_dga_gdf(polys)
    path = tmp_path / "polys_wkb.parquet"
    write_geoparquet(gdf, path, geometry_encoding="WKB")
    result = geopandas.read_parquet(path)
    assert len(result) == 4
    for i, poly in enumerate(polys):
        assert result.geometry.iloc[i].equals(poly)


def test_write_geoparquet_roundtrips_linestring_wkb(tmp_path) -> None:
    """LineString DGA → WKB write → read roundtrip."""
    lines = [LineString([(i, 0), (i + 1, 1), (i + 2, 0)]) for i in range(3)]
    gdf = _make_dga_gdf(lines)
    path = tmp_path / "ls_wkb.parquet"
    write_geoparquet(gdf, path, geometry_encoding="WKB")
    result = geopandas.read_parquet(path)
    assert len(result) == 3
    for i, ls in enumerate(lines):
        assert result.geometry.iloc[i].equals(ls)


def test_write_geoparquet_roundtrips_multipolygon_wkb(tmp_path) -> None:
    """MultiPolygon DGA → WKB write → read roundtrip."""
    mpolys = [
        MultiPolygon([
            Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 0)]),
            Polygon([(i + 2, 0), (i + 3, 0), (i + 3, 1), (i + 2, 0)]),
        ])
        for i in range(3)
    ]
    gdf = _make_dga_gdf(mpolys)
    path = tmp_path / "mpoly_wkb.parquet"
    write_geoparquet(gdf, path, geometry_encoding="WKB")
    result = geopandas.read_parquet(path)
    assert len(result) == 3
    for i, mp in enumerate(mpolys):
        assert result.geometry.iloc[i].equals(mp)


def test_write_geoparquet_no_materialization(tmp_path) -> None:
    """Writing DGA-backed GDF with GeoArrow must NOT trigger Shapely materialization."""
    pts = [Point(i, i) for i in range(10)]
    owned = from_shapely_geometries(pts)
    dga = DeviceGeometryArray._from_owned(owned)
    # Clear any construction diagnostics
    owned.diagnostics.clear()
    gs = geopandas.GeoSeries(dga, crs="EPSG:4326")
    gdf = geopandas.GeoDataFrame({"geometry": gs})
    path = tmp_path / "no_mat.parquet"
    write_geoparquet(gdf, path, geometry_encoding="geoarrow")
    mat_events = [e for e in owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION]
    assert mat_events == [], f"Unexpected materialization events: {mat_events}"


def test_wkb_encode_from_owned_buffers_point() -> None:
    """Verify _encode_owned_wkb_array produces valid WKB for points."""
    import struct

    owned = from_shapely_geometries([Point(1.5, 2.5), Point(3.0, 4.0)])
    _field, arr = io_arrow._encode_owned_wkb_array(owned, field_name="g", crs=None)
    wkb0 = arr[0].as_py()
    # WKB little-endian, type 1 = Point, then x, y
    byte_order, wkb_type, x, y = struct.unpack("<BIdd", wkb0)
    assert byte_order == 1
    assert wkb_type == 1
    assert x == 1.5
    assert y == 2.5


def test_wkb_encode_from_owned_buffers_polygon() -> None:
    """Verify _encode_owned_wkb_array produces valid WKB for polygons."""
    import shapely

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    owned = from_shapely_geometries([poly])
    _field, arr = io_arrow._encode_owned_wkb_array(owned, field_name="g", crs=None)
    roundtripped = shapely.from_wkb(arr[0].as_py())
    assert roundtripped.equals(poly)


def test_wkb_encode_from_owned_buffers_multilinestring() -> None:
    """Verify _encode_owned_wkb_array produces valid WKB for multilinestrings."""
    import shapely

    mls = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
    owned = from_shapely_geometries([mls])
    _field, arr = io_arrow._encode_owned_wkb_array(owned, field_name="g", crs=None)
    roundtripped = shapely.from_wkb(arr[0].as_py())
    assert roundtripped.equals(mls)


def test_write_geoparquet_device_wkb_has_no_transfer_or_materialization(tmp_path) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    gdf, owned = _make_device_dga_gdf([Point(i, i * 2) for i in range(4)])
    path = tmp_path / "device_wkb.parquet"

    write_geoparquet(gdf, path, geometry_encoding="WKB")
    result = geopandas.read_parquet(path)

    assert len(result) == 4
    assert result.geometry.iloc[3].equals(Point(3, 6))
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.TRANSFER] == []
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION] == []


def test_write_geoparquet_device_geoarrow_outputs_true_geoarrow(tmp_path) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    import pyarrow as pa
    import pyarrow.parquet as pq

    gdf, owned = _make_device_dga_gdf([
        Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
        Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
    ])
    path = tmp_path / "device_geoarrow_fallback.parquet"

    write_geoparquet(gdf, path, geometry_encoding="geoarrow")
    result = geopandas.read_parquet(path)
    arrow_table = pq.read_table(path)

    assert len(result) == 2
    assert result.geometry.iloc[0].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))
    assert not arrow_table.schema.field("geometry").type.equals(pa.binary())
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.TRANSFER] == []
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION] == []


def test_zero_transfer_pipeline(tmp_path) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    source = geopandas.GeoDataFrame(
        {
            "value": list(range(8)),
            "geometry": [Point(float(i), float(i)) for i in range(8)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    source_path = tmp_path / "zero-transfer-input.parquet"
    output_path = tmp_path / "zero-transfer-output.parquet"
    source.to_parquet(source_path, geometry_encoding="geoarrow")

    result = read_geoparquet(source_path)
    assert isinstance(result.geometry.values, DeviceGeometryArray)

    indices = np.flatnonzero(
        np.asarray(result.geometry.values.intersects(box(0.0, 0.0, 3.5, 3.5)), dtype=bool)
    )
    filtered = _take_dga_frame(result, indices)

    assert isinstance(filtered.geometry.values, DeviceGeometryArray)
    write_geoparquet(filtered, output_path, geometry_encoding="geoarrow")

    transfer_count, materialization_count = _diagnostic_totals(result, filtered)

    assert len(filtered) == 4
    assert transfer_count == 0
    assert materialization_count == 0


def test_parquet_roundtrip_preserves_crs(tmp_path) -> None:
    """GeoParquet roundtrip via DGA write path preserves CRS metadata."""
    from pyproj import CRS

    pts = [Point(i, i) for i in range(3)]
    gdf = _make_dga_gdf(pts, crs="EPSG:32632")
    path = tmp_path / "crs.parquet"
    write_geoparquet(gdf, path, geometry_encoding="geoarrow")
    result = geopandas.read_parquet(path)
    assert CRS(result.crs) == CRS("EPSG:32632")


def test_parquet_roundtrip_preserves_attributes(tmp_path) -> None:
    """Non-geometry columns survive DGA write path roundtrip."""
    pts = [Point(i, i) for i in range(5)]
    gdf = _make_dga_gdf(pts)
    path = tmp_path / "attrs.parquet"
    write_geoparquet(gdf, path, geometry_encoding="WKB")
    result = geopandas.read_parquet(path)
    assert list(result["idx"]) == list(range(5))
