from __future__ import annotations

import sys
import types
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pytest
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import vibespatial.api as geopandas
import vibespatial.api.io._geoarrow as api_geoarrow
import vibespatial.io.arrow as io_arrow
import vibespatial.io.geoarrow as io_geoarrow
import vibespatial.io.geoparquet as io_geoparquet
import vibespatial.io.wkb as io_wkb
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
    read_geoparquet_native,
    read_geoparquet_owned,
    select_row_groups,
    write_geoparquet,
)
from vibespatial.api._native_results import (
    GeometryNativeResult,
    NativeAttributeTable,
    NativeTabularResult,
    native_attribute_table_from_arrow_table,
    to_native_tabular_result,
)
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.geometry_array import to_wkb as array_to_wkb
from vibespatial.constructive.point import clip_points_rect_owned
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import FAMILY_TAGS, DiagnosticKind, from_shapely_geometries
from vibespatial.io.wkb import encode_owned_wkb_device
from vibespatial.runtime.residency import Residency, TransferTrigger


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
    assert fallbacks == []
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


def test_native_tabular_to_arrow_supports_geometry_only_payload() -> None:
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
    payload = to_native_tabular_result(GeometryNativeResult.from_owned(owned, crs="EPSG:4326"))

    assert payload is not None

    table = pa.table(payload.to_arrow(geometry_encoding="WKB"))
    roundtrip = geopandas.GeoDataFrame.from_arrow(table)

    assert table.column_names == ["geometry"]
    assert len(roundtrip) == 2
    assert roundtrip.geometry.iloc[0].equals(Point(0, 0))
    assert roundtrip.geometry.iloc[1].equals(Point(1, 1))


def test_decode_wkb_owned_large_native_list_uses_direct_device_pipeline(monkeypatch) -> None:
    if not has_gpu_runtime():
        return

    import vibespatial.kernels.core.wkb_decode as io_wkb_decode

    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
    values = from_shapely_geometries([polygon] * 8_100).to_wkb()
    pipeline_calls: list[int] = []
    original = io_wkb_decode.decode_wkb_device_pipeline

    def _spy_decode(payload_device, record_offsets_device, record_count: int):
        pipeline_calls.append(record_count)
        return original(payload_device, record_offsets_device, record_count)

    monkeypatch.setattr(io_wkb_decode, "decode_wkb_device_pipeline", _spy_decode)

    decoded = decode_wkb_owned(values)

    assert pipeline_calls == [8_100]
    assert decoded.residency is Residency.DEVICE
    assert decoded.to_shapely()[0].equals(polygon)
    assert decoded.to_shapely()[-1].equals(polygon)


def test_decode_wkb_owned_large_native_list_prefers_arrow_gpu_bridge(monkeypatch) -> None:
    monkeypatch.setattr(io_wkb, "DEVICE_WKB_LIST_DECODE_MIN_ROWS", 1)
    monkeypatch.setattr(
        io_wkb,
        "get_cuda_runtime",
        lambda: types.SimpleNamespace(available=lambda: True),
    )
    monkeypatch.setattr(
        io_wkb,
        "_try_gpu_wkb_arrow_decode",
        lambda array, on_invalid="raise": io_wkb._GpuWkbDecodeAttempt(
            result=from_shapely_geometries([Point(0, 0), Point(1, 1)]),
        ),
    )
    monkeypatch.setattr(
        io_wkb,
        "_prepare_native_wkb_list_for_device",
        lambda values: (_ for _ in ()).throw(
            AssertionError("staged device prep should not run when Arrow GPU decode succeeds")
        ),
    )

    decoded = decode_wkb_owned([Point(0, 0).wkb, Point(1, 1).wkb])

    assert decoded.to_shapely()[0].equals(Point(0, 0))
    assert decoded.to_shapely()[1].equals(Point(1, 1))


def test_decode_wkb_owned_raises_on_invalid_device_row_when_on_invalid_raise(monkeypatch) -> None:
    import vibespatial.kernels.core.wkb_decode as io_wkb_decode

    class _FakeRuntime:
        def available(self) -> bool:
            return True

        def from_host(self, value):
            return value

    monkeypatch.setattr(io_wkb, "DEVICE_WKB_LIST_DECODE_MIN_ROWS", 1)
    monkeypatch.setattr(io_wkb, "get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr(
        io_wkb_decode,
        "decode_wkb_device_pipeline",
        lambda payload_device, record_offsets_device, record_count: from_shapely_geometries(
            [Point(0, 0), None]
        ),
    )

    with pytest.raises(ValueError, match="Invalid WKB geometry encountered during GPU decode at row 1"):
        decode_wkb_owned([Point(0, 0).wkb, b"\x01\x01\x00\x00\x00"], on_invalid="raise")


def test_decode_wkb_owned_records_fallback_when_staged_gpu_decode_misses(monkeypatch) -> None:
    geopandas.clear_fallback_events()
    monkeypatch.setattr(
        io_wkb,
        "_try_gpu_wkb_list_decode",
        lambda values, on_invalid="raise": io_wkb._GpuWkbDecodeAttempt(
            result=None,
            fallback_detail="staged device decode failed: RuntimeError: boom",
        ),
    )

    decoded = decode_wkb_owned([Point(0, 0).wkb])
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert decoded.to_shapely()[0].equals(Point(0, 0))
    assert any(
        event.surface == "vibespatial.io.wkb"
        and "staged GPU WKB decode could not complete" in event.reason
        and "RuntimeError: boom" in event.detail
        for event in fallbacks
    )


def test_decode_wkb_arrow_array_records_fallback_when_gpu_bridge_misses(monkeypatch) -> None:
    geopandas.clear_fallback_events()
    monkeypatch.setattr(
        io_wkb,
        "_try_gpu_wkb_arrow_decode",
        lambda array, on_invalid="raise": io_wkb._GpuWkbDecodeAttempt(
            result=None,
            fallback_detail="GPU Arrow WKB decode bridge failed: RuntimeError: boom",
        ),
    )

    decoded = io_wkb.decode_wkb_arrow_array_owned(pa.array([Point(0, 0).wkb], type=pa.binary()))
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert decoded.to_shapely()[0].equals(Point(0, 0))
    assert any(
        event.surface == "vibespatial.io.wkb"
        and "GPU Arrow WKB decode could not complete" in event.reason
        and "RuntimeError: boom" in event.detail
        for event in fallbacks
    )

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


def test_geoparquet_read_backend_plan_allows_to_pandas_kwargs_on_gpu_path(monkeypatch) -> None:
    metadata = {
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "point"}},
    }

    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_gpu_runtime", lambda: True)
    monkeypatch.setattr(io_geoparquet, "_is_local_geoparquet_file", lambda path: True)

    plan = io_geoparquet.plan_geoparquet_read_backend(
        "sample.parquet",
        backend="auto",
        bbox=None,
        columns=None,
        storage_options=None,
        filesystem=None,
        filters=None,
        to_pandas_kwargs={"types_mapper": lambda dtype: dtype},
        geo_metadata=metadata,
    )

    assert plan.selected_backend == "pylibcudf"
    assert plan.selected_mode is ExecutionMode.GPU
    assert plan.gpu_rejection_reason is None


def test_geoparquet_read_backend_plan_allows_local_arrow_filesystem_on_gpu_path(
    monkeypatch,
) -> None:
    metadata = {
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "point"}},
    }

    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_gpu_runtime", lambda: True)
    monkeypatch.setattr(
        io_geoparquet,
        "_is_local_geoparquet_file",
        lambda path, filesystem=None: True,
    )

    plan = io_geoparquet.plan_geoparquet_read_backend(
        "sample.parquet",
        backend="auto",
        bbox=None,
        columns=None,
        storage_options=None,
        filesystem=pafs.LocalFileSystem(),
        filters=None,
        to_pandas_kwargs=None,
        geo_metadata=metadata,
    )

    assert plan.selected_backend == "pylibcudf"
    assert plan.selected_mode is ExecutionMode.GPU
    assert plan.gpu_rejection_reason is None


def test_geoparquet_read_backend_plan_allows_bytesio_source_on_gpu_path(monkeypatch) -> None:
    metadata = {
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "point"}},
    }

    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_gpu_runtime", lambda: True)

    plan = io_geoparquet.plan_geoparquet_read_backend(
        BytesIO(b"PAR1"),
        backend="auto",
        bbox=None,
        columns=None,
        storage_options=None,
        filesystem=None,
        filters=None,
        to_pandas_kwargs=None,
        geo_metadata=metadata,
    )

    assert plan.selected_backend == "pylibcudf"
    assert plan.selected_mode is ExecutionMode.GPU
    assert plan.gpu_rejection_reason is None


def test_geoparquet_engine_plan_reports_actual_selected_backend_for_point_bbox_gpu_scan(
    monkeypatch,
) -> None:
    metadata = {
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "point"}},
    }

    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_gpu_runtime", lambda: True)
    monkeypatch.setattr(io_geoparquet, "_is_local_geoparquet_file", lambda path: True)

    scan_plan = plan_geoparquet_scan(bbox=(0.0, 0.0, 1.0, 1.0), geo_metadata=metadata)
    read_plan = io_geoparquet.plan_geoparquet_read_backend(
        "sample.parquet",
        backend="auto",
        bbox=(0.0, 0.0, 1.0, 1.0),
        columns=None,
        storage_options=None,
        filesystem=None,
        filters=None,
        to_pandas_kwargs=None,
        geo_metadata=metadata,
    )
    engine_plan = io_geoparquet.plan_geoparquet_engine(
        geo_metadata=metadata,
        scan_plan=scan_plan,
        chunk_plans=(
            io_geoparquet.GeoParquetChunkPlan(
                chunk_index=0,
                row_groups=(0,),
                estimated_rows=10,
            ),
        ),
        target_chunk_rows=None,
        read_plan=read_plan,
    )

    assert read_plan.selected_backend == "pylibcudf"
    assert read_plan.gpu_rejection_reason is None
    assert engine_plan.backend == "pylibcudf"
    assert "GPU GeoParquet scan backend" in engine_plan.reason


def test_geoparquet_engine_plan_tolerates_missing_primary_geometry_metadata() -> None:
    scan_plan = plan_geoparquet_scan(bbox=None, geo_metadata=None)
    read_plan = io_geoparquet.GeoParquetReadBackendPlan(
        requested_backend="auto",
        selected_backend="pyarrow",
        selected_mode=ExecutionMode.CPU,
        can_use_pylibcudf=False,
        gpu_rejection_reason="missing geometry metadata",
        reason="host scan path selected",
    )

    engine_plan = io_geoparquet.plan_geoparquet_engine(
        geo_metadata={"primary_column": "geometry", "columns": {}},
        scan_plan=scan_plan,
        chunk_plans=(
            io_geoparquet.GeoParquetChunkPlan(
                chunk_index=0,
                row_groups=(0,),
                estimated_rows=10,
            ),
        ),
        target_chunk_rows=None,
        read_plan=read_plan,
    )

    assert engine_plan.geometry_encoding is None


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

    def fake_read_geoparquet_table_with_pyarrow(*args, **kwargs):
        captured["row_groups"] = kwargs["row_groups"]
        return (
            pa.table({"geometry": [Point(12, 12).wkb]}),
            {"primary_column": "geometry", "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}}},
            None,
        )

    monkeypatch.setattr(
        io_geoparquet,
        "_read_geoparquet_table_with_pyarrow",
        fake_read_geoparquet_table_with_pyarrow,
    )
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

    assert len(result) == 1
    assert captured["row_groups"] == (1, 2)
    assert events[-1].operation == "row_group_pushdown"


def test_read_geoparquet_records_backend_selection_fallback_when_auto_gpu_scan_rejected(
    monkeypatch,
) -> None:
    geopandas.clear_fallback_events()

    monkeypatch.setattr(io_geoparquet, "has_pyarrow_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_gpu_runtime", lambda: True)
    monkeypatch.setattr(io_geoparquet, "_is_local_geoparquet_file", lambda path: True)
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
        lambda path, filesystem, geo_metadata: None,
    )
    monkeypatch.setattr(
        io_geoparquet,
        "_read_geoparquet_table_with_pyarrow",
        lambda *args, **kwargs: (
            pa.table({"geometry": [Point(0, 0).wkb]}),
            {"primary_column": "geometry", "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}}},
            None,
        ),
    )

    result = io_arrow.read_geoparquet("sample.parquet", storage_options={"foo": "bar"})
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert len(result) == 1
    assert result.geometry.iloc[0].equals(Point(0, 0))
    assert any(
        event.surface == "geopandas.read_parquet"
        and event.reason == "explicit CPU fallback for GeoParquet scan backend selection"
        and "filesystem-backed GeoParquet reads still route through host pyarrow" in event.detail
        for event in fallbacks
    )


def test_arrow_to_geopandas_records_explicit_wkb_decode_fallback(monkeypatch) -> None:
    import vibespatial.api.io.arrow as api_io_arrow

    geopandas.clear_fallback_events()
    table = pa.table(
        {
            "value": [1, 2],
            "geometry": pa.array(
                [
                    bytes.fromhex("010100000000000000000000000000000000000000"),
                    bytes.fromhex("0101000000000000000000f03f000000000000f03f"),
                ],
                type=pa.binary(),
            ),
        }
    )
    geo_metadata = {
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }

    monkeypatch.setattr(
        io_wkb,
        "decode_wkb_arrow_array_owned",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            NotImplementedError("test-only arrow WKB decode miss")
        ),
    )

    result = api_io_arrow._arrow_to_geopandas(
        table,
        geo_metadata,
        fallback_surface="geopandas.read_parquet",
        fallback_pipeline="io/read_parquet",
    )
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert len(result) == 2
    assert result.geometry.iloc[0].equals(Point(0, 0))
    assert result.geometry.iloc[1].equals(Point(1, 1))
    assert any(
        event.surface == "geopandas.read_parquet"
        and "Arrow geometry decode could not complete" in event.reason
        and "column=geometry" in event.detail
        and "encoding='WKB'" in event.detail
        and "test-only arrow WKB decode miss" in event.detail
        for event in fallbacks
    )


def test_arrow_to_geopandas_records_explicit_geoarrow_decode_fallback(monkeypatch) -> None:
    import vibespatial.api.io.arrow as api_io_arrow

    geopandas.clear_fallback_events()
    source = geopandas.GeoDataFrame(
        {"value": [1, 2], "geometry": [Point(0, 0), Point(1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    table = pa.table(source.to_arrow(geometry_encoding="geoarrow", interleaved=False))
    geo_metadata = {
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "point", "crs": "EPSG:4326"}},
    }

    monkeypatch.setattr(
        io_geoarrow,
        "_decode_geoarrow_array_to_owned",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            NotImplementedError("test-only GeoArrow decode miss")
        ),
    )

    result = api_io_arrow._arrow_to_geopandas(
        table,
        geo_metadata,
        fallback_surface="geopandas.read_feather",
        fallback_pipeline="io/read_feather",
    )
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert len(result) == 2
    assert result.geometry.iloc[0].equals(Point(0, 0))
    assert result.geometry.iloc[1].equals(Point(1, 1))
    assert any(
        event.surface == "geopandas.read_feather"
        and "Arrow geometry decode could not complete" in event.reason
        and "column=geometry" in event.detail
        and "encoding='point'" in event.detail
        and "test-only GeoArrow decode miss" in event.detail
        for event in fallbacks
    )


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


def test_read_geoparquet_owned_gpu_backend_uses_shared_capability_gate(monkeypatch) -> None:
    monkeypatch.setattr(io_geoparquet, "has_pyarrow_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_gpu_runtime", lambda: True)
    monkeypatch.setattr(io_geoparquet, "_is_local_geoparquet_file", lambda path: True)
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
        lambda path, filesystem, geo_metadata: None,
    )

    with pytest.raises(RuntimeError, match="filesystem-backed GeoParquet reads still route through host pyarrow"):
        read_geoparquet_owned("sample.parquet", backend="gpu", storage_options={"foo": "bar"})


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
    monkeypatch.setattr(io_geoparquet, "_is_local_geoparquet_file", lambda path: True)
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
        lambda path, columns=None, row_groups=None, filesystem=None, **kwargs: build_table(
            row_groups[0][0] * 2, row_groups[-1][-1] * 2 + 2
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
    scan_calls: list[list[str] | None] = []

    class FakeGpuTable:
        column_names = ["geometry"]

        def __init__(self) -> None:
            self._columns = [object()]

        def columns(self):
            return self._columns

        def to_arrow(self):
            raise AssertionError("geometry GPU path must not call to_arrow()")

    fake_table = FakeGpuTable()
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2)])
    if has_gpu_runtime():
        owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="test fake pylibcudf public read result should stay device-backed",
        )

    monkeypatch.setattr(io_geoparquet, "has_pyarrow_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(
        io_geoparquet,
        "plan_geoparquet_read_backend",
        lambda *args, **kwargs: io_geoparquet.GeoParquetReadBackendPlan(
            requested_backend="auto",
            selected_backend="pylibcudf",
            selected_mode=ExecutionMode.GPU,
            can_use_pylibcudf=True,
            gpu_rejection_reason=None,
            reason="test",
        ),
    )
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
    monkeypatch.setattr(
        io_geoparquet,
        "_read_geoparquet_table_with_pylibcudf",
        lambda *args, **kwargs: scan_calls.append(kwargs.get("columns")) or fake_table,
    )
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

    assert scan_calls == [["geometry"]]
    assert list(result.columns) == ["value", "geometry", "name"]
    expected_geometry_array_type = DeviceGeometryArray if has_gpu_runtime() else GeometryArray
    assert isinstance(result.geometry.values, expected_geometry_array_type)
    assert list(result["value"]) == [10, 20, 30]
    assert list(result["name"]) == ["a", "b", "c"]
    assert result.attrs == {"source": "gpu"}


def test_decode_geoparquet_table_to_owned_gpu_decode_miss_falls_back_to_arrow(monkeypatch) -> None:
    class FakeGpuTable:
        def __init__(self, arrow_table) -> None:
            self._arrow_table = arrow_table
            self._columns = [object()]

        def to_arrow(self):
            return self._arrow_table

        def columns(self):
            return self._columns

    geopandas.clear_fallback_events()
    arrow_table = pa.table({"geometry": [Point(0, 0).wkb, Point(2, 3).wkb]})
    fake_table = FakeGpuTable(arrow_table)
    geo_metadata = {
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }

    monkeypatch.setattr(io_geoparquet, "_is_pylibcudf_table", lambda table: table is fake_table)
    monkeypatch.setattr(
        io_geoparquet,
        "_decode_pylibcudf_geoparquet_column_to_owned",
        lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("test-only mixed family miss")),
    )

    owned = io_geoparquet._decode_geoparquet_table_to_owned(fake_table, geo_metadata)

    assert owned.row_count == 2
    assert [geom.wkt for geom in owned.to_shapely()] == ["POINT (0 0)", "POINT (2 3)"]
    fallbacks = geopandas.get_fallback_events(clear=True)
    assert any(
        event.pipeline == "io/read_parquet" and "test-only mixed family miss" in event.detail
        for event in fallbacks
    )


def test_read_geoparquet_gpu_decode_miss_falls_back_to_arrow_geometry_decode(monkeypatch) -> None:
    import pyarrow.parquet as pq

    geopandas.clear_fallback_events()
    scan_calls: list[list[str] | None] = []

    class FakeGpuTable:
        column_names = ["geometry"]

        def __init__(self, arrow_table) -> None:
            self._columns = [object()]
            self._arrow_table = arrow_table

        def columns(self):
            return self._columns

        def to_arrow(self):
            return self._arrow_table

    arrow_table = pa.table({"geometry": [Point(0, 0).wkb, Point(1, 2).wkb]})
    fake_table = FakeGpuTable(arrow_table)

    monkeypatch.setattr(io_geoparquet, "has_pyarrow_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(
        io_geoparquet,
        "plan_geoparquet_read_backend",
        lambda *args, **kwargs: io_geoparquet.GeoParquetReadBackendPlan(
            requested_backend="auto",
            selected_backend="pylibcudf",
            selected_mode=ExecutionMode.GPU,
            can_use_pylibcudf=True,
            gpu_rejection_reason=None,
            reason="test",
        ),
    )
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
                    "geometry": {"encoding": "WKB", "crs": "EPSG:4326"},
                },
            },
        ),
    )
    monkeypatch.setattr(
        io_geoparquet,
        "_read_geoparquet_table_with_pylibcudf",
        lambda *args, **kwargs: scan_calls.append(kwargs.get("columns")) or fake_table,
    )
    monkeypatch.setattr(
        io_geoparquet,
        "_read_non_geometry_geoparquet_columns_as_arrow",
        lambda *args, **kwargs: pa.table({"value": [10, 20]}),
    )
    monkeypatch.setattr(
        io_geoparquet,
        "_decode_pylibcudf_geoparquet_column_to_owned",
        lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("test-only mixed family miss")),
    )
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
            ]
        ),
    )

    try:
        result = io_arrow.read_geoparquet("sample.parquet")
    finally:
        monkeypatch.setattr(pq, "read_schema", original_read_schema)

    assert scan_calls == [["geometry"]]
    assert list(result["value"]) == [10, 20]
    assert [geom.wkt for geom in result.geometry.to_list()] == ["POINT (0 0)", "POINT (1 2)"]
    fallbacks = geopandas.get_fallback_events(clear=True)
    assert any(
        event.pipeline == "io/read_parquet" and "test-only mixed family miss" in event.detail
        for event in fallbacks
    )


def test_read_geoparquet_gpu_filter_projection_includes_filter_columns_without_leaking_them(
    monkeypatch,
) -> None:
    import vibespatial.api.io.arrow as api_io_arrow

    scan_calls: list[list[str] | None] = []
    attrs_calls: list[list[str]] = []

    class FakeGpuTable:
        column_names = ["geometry", "value"]

        def __init__(self) -> None:
            self._columns = [object(), object()]

        def columns(self):
            return self._columns

        def to_arrow(self):
            raise AssertionError("geometry GPU path must not call to_arrow()")

    fake_table = FakeGpuTable()
    owned = from_shapely_geometries([Point(1, 1), Point(2, 2)])
    if has_gpu_runtime():
        owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="test filter projection should keep geometry on device",
        )

    monkeypatch.setattr(io_geoparquet, "has_pyarrow_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(
        io_geoparquet,
        "plan_geoparquet_read_backend",
        lambda *args, **kwargs: io_geoparquet.GeoParquetReadBackendPlan(
            requested_backend="auto",
            selected_backend="pylibcudf",
            selected_mode=ExecutionMode.GPU,
            can_use_pylibcudf=True,
            gpu_rejection_reason=None,
            reason="test",
        ),
    )
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
    monkeypatch.setattr(
        io_geoparquet,
        "_build_geoparquet_metadata_summary_from_pyarrow",
        lambda path, filesystem, geo_metadata: None,
    )
    monkeypatch.setattr(
        api_io_arrow,
        "_read_parquet_schema_and_metadata",
        lambda path, filesystem=None: (
            pa.schema(
                [
                    pa.field("value", pa.int64()),
                    pa.field("geometry", pa.binary()),
                    pa.field("name", pa.string()),
                ],
                metadata={b"PANDAS_ATTRS": b'{"source": "gpu"}'},
            ),
            {b"geo": b"1"},
        ),
    )
    monkeypatch.setattr(
        io_geoparquet,
        "_read_geoparquet_table_with_pylibcudf",
        lambda *args, **kwargs: scan_calls.append(kwargs.get("columns")) or fake_table,
    )
    monkeypatch.setattr(
        io_geoparquet,
        "_read_non_geometry_geoparquet_columns_as_arrow",
        lambda *args, **kwargs: attrs_calls.append(list(kwargs["columns"]))
        or pa.table({"name": ["b", "c"]}),
    )
    monkeypatch.setattr(io_geoparquet, "_decode_pylibcudf_geoparquet_column_to_owned", lambda column, encoding: owned)

    result = io_arrow.read_geoparquet(
        "sample.parquet",
        columns=["name", "geometry"],
        filters=[("value", ">", 15)],
    )

    assert scan_calls == [["geometry", "value"]]
    assert attrs_calls == [["name"]]
    assert list(result.columns) == ["name", "geometry"]
    assert list(result["name"]) == ["b", "c"]


def test_read_geoparquet_gpu_geometry_only_skips_non_geometry_sidecar_read(monkeypatch) -> None:
    import vibespatial.api.io.arrow as api_io_arrow

    class FakeGpuTable:
        column_names = ["geometry"]

        def __init__(self) -> None:
            self._columns = [object()]

        def columns(self):
            return self._columns

        def num_rows(self):
            return 3

        def to_arrow(self):
            raise AssertionError("geometry-only GPU path must not materialize Arrow geometry")

    fake_table = FakeGpuTable()
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2)])
    if has_gpu_runtime():
        owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="test geometry-only geoparquet read should stay device-backed",
        )

    monkeypatch.setattr(io_geoparquet, "has_pyarrow_support", lambda: True)
    monkeypatch.setattr(io_geoparquet, "has_pylibcudf_support", lambda: True)
    monkeypatch.setattr(
        io_geoparquet,
        "plan_geoparquet_read_backend",
        lambda *args, **kwargs: io_geoparquet.GeoParquetReadBackendPlan(
            requested_backend="auto",
            selected_backend="pylibcudf",
            selected_mode=ExecutionMode.GPU,
            can_use_pylibcudf=True,
            gpu_rejection_reason=None,
            reason="test",
        ),
    )
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
    monkeypatch.setattr(
        io_geoparquet,
        "_build_geoparquet_metadata_summary_from_pyarrow",
        lambda path, filesystem, geo_metadata: None,
    )
    monkeypatch.setattr(
        api_io_arrow,
        "_read_parquet_schema_and_metadata",
        lambda path, filesystem=None: (
            pa.schema([pa.field("geometry", pa.binary())]),
            {b"geo": b"1"},
        ),
    )
    monkeypatch.setattr(
        io_geoparquet,
        "_read_geoparquet_table_with_pylibcudf",
        lambda *args, **kwargs: fake_table,
    )
    monkeypatch.setattr(
        io_geoparquet,
        "_read_non_geometry_geoparquet_columns_as_arrow",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("geometry-only GPU path should not read non-geometry sidecar columns")
        ),
    )
    monkeypatch.setattr(io_geoparquet, "_decode_pylibcudf_geoparquet_column_to_owned", lambda column, encoding: owned)

    result = io_arrow.read_geoparquet("sample.parquet")

    expected_geometry_array_type = DeviceGeometryArray if has_gpu_runtime() else GeometryArray
    assert list(result.columns) == ["geometry"]
    assert isinstance(result.geometry.values, expected_geometry_array_type)
    assert list(result.index) == [0, 1, 2]


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


def test_geoparquet_geometry_column_crs_caches_repeated_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from pyproj import CRS
    except ImportError:
        pytest.skip("pyproj not available")

    io_geoparquet._cached_geoparquet_crs_from_user_input.cache_clear()
    io_geoparquet._cached_geoparquet_crs_from_json.cache_clear()

    original = CRS.from_user_input
    calls: list[object] = []

    def _wrapped(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(CRS, "from_user_input", _wrapped)

    default_first = io_geoparquet._geoparquet_geometry_column_crs({})
    default_call_count = len(calls)
    default_second = io_geoparquet._geoparquet_geometry_column_crs({})
    default_cached_call_count = len(calls)
    explicit_first = io_geoparquet._geoparquet_geometry_column_crs({"crs": "EPSG:4326"})
    explicit_call_count = len(calls)
    explicit_second = io_geoparquet._geoparquet_geometry_column_crs({"crs": "EPSG:4326"})
    explicit_cached_call_count = len(calls)

    assert default_first == default_second
    assert explicit_first == explicit_second
    assert default_call_count > 0
    assert explicit_call_count > default_cached_call_count
    assert default_cached_call_count == default_call_count
    assert explicit_cached_call_count == explicit_call_count


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
    assert restored.geoarrow_backed is True
    assert restored.to_shapely()[1].equals(Point(1, 1))


def test_decode_geoarrow_point_nan_nan_roundtrips_empty_point() -> None:
    import pyarrow as pa

    arr = pa.StructArray.from_arrays(
        [
            pa.array([0.0, np.nan, None, 1.0], type=pa.float64()),
            pa.array([0.0, np.nan, None, 1.0], type=pa.float64()),
        ],
        fields=[
            pa.field("x", pa.float64(), nullable=False),
            pa.field("y", pa.float64(), nullable=False),
        ],
        mask=pa.array([False, False, True, False], type=pa.bool_()),
    )
    field = pa.field(
        "geometry",
        arr.type,
        metadata={b"ARROW:extension:name": b"geoarrow.point", b"ARROW:extension:metadata": b"{}"},
    )

    restored = io_arrow._decode_geoarrow_array_to_owned(field, arr).to_shapely()

    assert restored[0].equals(Point(0, 0))
    assert restored[1].is_empty
    assert restored[2] is None
    assert restored[3].equals(Point(1, 1))


def test_encode_wkb_owned_preserves_empty_point_rows() -> None:
    import shapely

    owned = from_shapely_geometries([Point(0, 0), Point(), None, Point(1, 1)])

    encoded = encode_wkb_owned(owned)
    restored = [shapely.from_wkb(value) if value is not None else None for value in encoded]

    assert restored[0].equals(Point(0, 0))
    assert restored[1].is_empty
    assert restored[2] is None
    assert restored[3].equals(Point(1, 1))


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
def test_encode_owned_wkb_device_preserves_empty_point_rows() -> None:
    import pyarrow as pa
    import shapely

    owned = from_shapely_geometries([Point(0, 0), Point(), None, Point(1, 1)])

    column = encode_owned_wkb_device(owned)
    arrow = column.to_arrow().cast(pa.binary())
    restored = [
        shapely.from_wkb(value.as_py()) if value.as_py() is not None else None
        for value in arrow
    ]

    assert restored[0].equals(Point(0, 0))
    assert restored[1].is_empty
    assert restored[2] is None
    assert restored[3].equals(Point(1, 1))


def test_encode_owned_geoarrow_array_roundtrips_polygon_family() -> None:
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
    owned = from_shapely_geometries([polygon, polygon.buffer(1)])

    field, geom_arr = encode_owned_geoarrow_array(owned, field_name="geometry")
    restored = io_arrow._decode_geoarrow_array_to_owned(field, geom_arr)

    assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.polygon"
    assert restored.geoarrow_backed is True
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


def test_geoseries_to_arrow_point_fast_path_preserves_empty_point_rows() -> None:
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(0, 0), Point(), None, Point(2, 2)])

    arrow_array = io_arrow.geoseries_to_arrow(series, geometry_encoding="geoarrow")
    schema_capsule, array_capsule = arrow_array.__arrow_c_array__()
    import pyarrow as pa

    field = pa.Field._import_from_c_capsule(schema_capsule)
    pa_array = pa.Array._import_from_c_capsule(field.__arrow_c_schema__(), array_capsule)
    restored = io_arrow._decode_geoarrow_array_to_owned(field, pa_array).to_shapely()

    assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.point"
    assert restored[0].equals(Point(0, 0))
    assert restored[1].is_empty
    assert restored[2] is None
    assert restored[3].equals(Point(2, 2))
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


def test_geodataframe_to_arrow_device_geoarrow_records_fallback_before_host_materialization(
    monkeypatch,
) -> None:
    geopandas.clear_fallback_events()
    gdf, _owned = _make_device_dga_gdf([Point(0, 0), Point(1, 1)])

    original_asarray = io_geoarrow.np.asarray

    def _spy_asarray(value, *args, **kwargs):
        if isinstance(value, DeviceGeometryArray):
            assert geopandas.get_fallback_events(), (
                "GeoArrow export must record the compatibility fallback before host materialization"
            )
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(io_geoarrow.np, "asarray", _spy_asarray)

    arrow_table = io_arrow.geodataframe_to_arrow(gdf, geometry_encoding="geoarrow", include_z=True)
    table = pa.table(arrow_table)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert table.schema.field("geometry").metadata[b"ARROW:extension:name"] == b"geoarrow.point"
    assert any(
        event.surface == "geopandas.geodataframe.to_arrow"
        and "compatibility export" in event.reason
        for event in fallbacks
    )


def test_geodataframe_to_arrow_device_geoarrow_raises_strict_native_before_host_materialization(
    monkeypatch,
) -> None:
    from vibespatial.runtime.fallbacks import StrictNativeFallbackError
    from vibespatial.testing import strict_native_environment

    geopandas.clear_fallback_events()
    gdf, _owned = _make_device_dga_gdf([Point(0, 0), Point(1, 1)])

    original_asarray = io_geoarrow.np.asarray

    def _spy_asarray(value, *args, **kwargs):
        if isinstance(value, DeviceGeometryArray):
            raise AssertionError(
                "GeoArrow export should fail before host materialization in strict-native mode"
            )
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(io_geoarrow.np, "asarray", _spy_asarray)

    with pytest.raises(StrictNativeFallbackError):
        with strict_native_environment():
            io_arrow.geodataframe_to_arrow(gdf, geometry_encoding="geoarrow", include_z=True)

    fallbacks = geopandas.get_fallback_events(clear=True)
    assert any(
        event.surface == "geopandas.geodataframe.to_arrow"
        and "compatibility export" in event.reason
        for event in fallbacks
    )


def _interleaved_leaf_name(data_type) -> str | None:
    import pyarrow as pa

    current = data_type
    while True:
        if pa.types.is_fixed_size_list(current):
            return current.value_field.name
        if pa.types.is_list(current):
            current = current.value_field.type
            continue
        return None


@pytest.mark.parametrize(
    ("geometry", "extension_name"),
    [
        (LineString([(0, 0), (1, 1), (2, 0)]), "geoarrow.linestring"),
        (Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]), "geoarrow.polygon"),
        (MultiPoint([(0, 0), (1, 1)]), "geoarrow.multipoint"),
        (
            MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
            "geoarrow.multilinestring",
        ),
        (
            MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]),
            "geoarrow.multipolygon",
        ),
    ],
)
def test_geodataframe_to_arrow_interleaved_geoarrow_preserves_xy_leaf_name(
    geometry,
    extension_name,
) -> None:
    import pyarrow as pa

    gdf = geopandas.GeoDataFrame({"geometry": [geometry]})

    table = io_arrow.geodataframe_to_arrow(gdf, geometry_encoding="geoarrow", interleaved=True)
    field = pa.table(table).schema.field("geometry")

    assert field.metadata[b"ARROW:extension:name"] == extension_name.encode()
    assert _interleaved_leaf_name(field.type) == "xy"


def test_geodataframe_to_arrow_geoarrow_rejects_empty_or_all_missing_geometry() -> None:
    empty = geopandas.GeoDataFrame(columns=["value", "geometry"], geometry="geometry")
    all_missing = geopandas.GeoDataFrame({"value": [1], "geometry": [None]}, geometry="geometry")

    with pytest.raises(NotImplementedError, match="infer the geometry type"):
        io_arrow.geodataframe_to_arrow(empty, geometry_encoding="geoarrow")

    with pytest.raises(NotImplementedError, match="infer the geometry type"):
        io_arrow.geodataframe_to_arrow(all_missing, geometry_encoding="geoarrow")


def test_native_geometry_benchmark_reports_host_and_native_paths() -> None:
    benchmarks = benchmark_native_geometry_codec(operation="encode", geometry_type="point", rows=1_000, repeat=1)
    implementations = {item.implementation for item in benchmarks}

    assert implementations == {"host_bridge", "native_owned"}


def test_wkb_bridge_benchmark_reports_host_and_native_paths() -> None:
    benchmarks = benchmark_wkb_bridge(operation="decode", geometry_type="point", rows=1_000, repeat=1)
    implementations = {item.implementation for item in benchmarks}

    assert implementations == {"host_bridge", "native_owned"}


def test_wkb_bridge_benchmark_decode_uses_public_decode_path(monkeypatch) -> None:
    decode_calls: list[int] = []
    plan_calls: list[int] = []

    def _fake_decode(values):
        decode_calls.append(len(values))
        return object()

    def _fake_plan(values):
        plan_calls.append(len(values))
        return types.SimpleNamespace(fallback_rows=17)

    monkeypatch.setattr("vibespatial.io.geoarrow.decode_wkb_owned", _fake_decode)
    monkeypatch.setattr("vibespatial.io.geoarrow.plan_wkb_partition", _fake_plan)

    benchmarks = benchmark_wkb_bridge(operation="decode", geometry_type="point", rows=1_000, repeat=1)
    native = next(item for item in benchmarks if item.implementation == "native_owned")

    assert decode_calls == [1_000] * 6
    assert plan_calls == [1_000]
    assert native.fallback_rows == 17


def test_wkb_bridge_benchmark_encode_uses_host_bridge_and_native_arrow_encode(monkeypatch) -> None:
    class _FakeOwned:
        def to_wkb(self, *, hex: bool = False):
            raise AssertionError("encode benchmark host bridge should not route through owned.to_wkb()")

        def to_shapely(self):
            return [Point(0, 0), Point(1, 1)]

    host_calls: list[int] = []
    native_calls: list[int] = []
    plan_calls: list[int] = []

    monkeypatch.setattr(
        "vibespatial.io.geoarrow._sample_owned_for_geoarrow_benchmark",
        lambda geometry_type, rows, seed=0: _FakeOwned(),
    )
    monkeypatch.setattr(
        "shapely.to_wkb",
        lambda values: host_calls.append(len(values)) or [b"a", b"b"],
    )
    monkeypatch.setattr(
        "pyarrow.array",
        lambda values, type=None: list(values),
    )
    monkeypatch.setattr(
        "vibespatial.io.geoarrow._encode_native_wkb",
        lambda owned: (plan_calls.append(2) or [b"a", b"b"], types.SimpleNamespace(fallback_rows=19)),
    )
    monkeypatch.setattr(
        "vibespatial.io.geoarrow._encode_owned_wkb_array",
        lambda owned: (None, native_calls.append(2) or [b"a", b"b"]),
    )

    benchmarks = benchmark_wkb_bridge(operation="encode", geometry_type="point", rows=2, repeat=1)
    native = next(item for item in benchmarks if item.implementation == "native_owned")

    assert host_calls == [2] * 6
    assert native_calls == [2] * 6
    assert plan_calls == [2]
    assert native.fallback_rows == 19


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


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
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


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
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


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
def test_write_geoparquet_supports_gzip_compression_without_to_parquet_fallback(tmp_path) -> None:
    import pyarrow.parquet as pq

    geopandas.clear_fallback_events()
    pts = [Point(i, i * 2) for i in range(5)]
    gdf = _make_dga_gdf(pts)
    path = tmp_path / "pts_gzip.parquet"

    write_geoparquet(gdf, path, compression="gzip")
    result = geopandas.read_parquet(path)
    fallbacks = geopandas.get_fallback_events(clear=True)
    parquet_file = pq.ParquetFile(path)
    row_group = parquet_file.metadata.row_group(0)

    assert not any(
        event.surface == "geopandas.geodataframe.to_parquet" and "compression codec" in event.reason
        for event in fallbacks
    )
    assert row_group.column(0).compression == "GZIP"
    assert row_group.column(1).compression == "GZIP"
    for i, pt in enumerate(pts):
        assert result.geometry.iloc[i].equals(pt)


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
def test_write_geoparquet_defaults_to_snappy_compression(tmp_path) -> None:
    import pyarrow.parquet as pq

    pts = [Point(i, i * 2) for i in range(5)]
    gdf = _make_dga_gdf(pts)
    path = tmp_path / "pts_default_codec.parquet"

    write_geoparquet(gdf, path, geometry_encoding="geoarrow")

    parquet_file = pq.ParquetFile(path)
    row_group = parquet_file.metadata.row_group(0)

    assert row_group.column(0).compression == "SNAPPY"
    assert row_group.column(1).compression == "SNAPPY"


def test_read_geoparquet_restores_named_index_from_parquet_metadata(tmp_path) -> None:
    gdf = geopandas.GeoDataFrame(
        {
            "iso": ["AAA", "BBB", "CCC"],
            "value": [1, 2, 3],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    ).set_index("iso")
    path = tmp_path / "indexed.parquet"

    gdf.to_parquet(path, index=True)
    result = geopandas.read_parquet(path)

    assert result.index.name == "iso"
    assert list(result.index) == ["AAA", "BBB", "CCC"]
    assert list(result.columns) == ["value", "geometry"]
    for left, right in zip(gdf.geometry, result.geometry, strict=True):
        assert left.equals(right)


def test_read_non_geometry_geoparquet_columns_as_arrow_preserves_hidden_index_columns(tmp_path) -> None:
    gdf = geopandas.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
        crs="EPSG:4326",
        index=pd.Index(["AAA", "BBB", "CCC"], name="iso"),
    )
    path = tmp_path / "geometry-only-indexed.parquet"

    gdf.to_parquet(path, index=True)
    attrs_arrow = io_geoparquet._read_non_geometry_geoparquet_columns_as_arrow(
        path,
        columns=[],
    )
    attributes = native_attribute_table_from_arrow_table(attrs_arrow)

    assert attrs_arrow.column_names == ["iso"]
    assert list(attributes.columns) == []
    assert attributes.index.name == "iso"
    assert list(attributes.index) == ["AAA", "BBB", "CCC"]


def test_read_geoparquet_native_chunked_preserves_secondary_geometry_and_index(tmp_path) -> None:
    from pandas import ArrowDtype

    gdf = geopandas.GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
        index=pd.Index(["AAA", "BBB", "CCC"], name="iso"),
    )
    gdf["geom2"] = geopandas.GeoSeries(
        [Point(10, 10), Point(11, 11), Point(12, 12)],
        index=gdf.index,
        crs=gdf.crs,
        name="geom2",
    )
    path = tmp_path / "native-read.parquet"

    gdf.to_parquet(path, index=True, row_group_size=1)
    payload = read_geoparquet_native(
        path,
        backend="cpu",
        chunk_rows=1,
        to_pandas_kwargs={"types_mapper": ArrowDtype},
    )

    assert isinstance(payload, NativeTabularResult)
    assert payload.geometry_name == "geometry"
    assert [column.name for column in payload.secondary_geometry] == ["geom2"]
    assert payload.provenance is not None
    assert payload.provenance.surface == "vibespatial.read_geoparquet_native"
    assert payload.provenance.backend == "pyarrow"
    assert payload.provenance.chunk_rows == 1
    assert payload.attributes.index.name == "iso"

    materialized = payload.to_geodataframe()

    assert str(materialized["value"].dtype) == "int64[pyarrow]"
    assert list(materialized.columns) == ["value", "geometry", "geom2"]
    assert materialized.index.name == "iso"
    assert list(materialized.index.astype(str)) == list(gdf.index.astype(str))
    for left, right in zip(materialized.geometry, gdf.geometry, strict=True):
        assert left.equals(right)
    for left, right in zip(materialized["geom2"], gdf["geom2"], strict=True):
        assert left.equals(right)


def test_read_non_geometry_geoparquet_columns_as_arrow_supports_filter_only_columns_with_row_groups(
    tmp_path,
) -> None:
    gdf = geopandas.GeoDataFrame(
        {
            "name": ["a", "b", "c"],
            "value": [10, 20, 30],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )
    path = tmp_path / "attrs-row-groups.parquet"
    gdf.to_parquet(path, row_group_size=1)

    table = io_geoparquet._read_non_geometry_geoparquet_columns_as_arrow(
        path,
        columns=["name"],
        row_groups=(0, 1, 2),
        filters=[("value", ">", 15)],
    )

    assert table.column_names == ["name"]
    assert table.to_pydict() == {"name": ["b", "c"]}


def test_to_wkb_accepts_device_geometry_array_values() -> None:
    gdf = geopandas.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1), None]},
        crs="EPSG:4326",
    )

    values = array_to_wkb(gdf.geometry.values)

    assert values[0] == Point(0, 0).wkb
    assert values[1] == Point(1, 1).wkb
    assert values[2] is None


def test_read_parquet_partitioned_directory_recovers_fragment_geo_metadata(tmp_path) -> None:
    gdf = geopandas.GeoDataFrame(
        {
            "value": [1, 2, 3, 4],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        },
        crs="EPSG:4326",
    )
    basedir = tmp_path / "partitioned_dataset"
    basedir.mkdir()

    gdf.iloc[:2].to_parquet(basedir / "part1.parquet")
    gdf.iloc[2:].to_parquet(basedir / "part2.parquet")
    result = geopandas.read_parquet(basedir)

    assert list(result["value"]) == [1, 2, 3, 4]
    assert result.crs == gdf.crs
    for left, right in zip(gdf.geometry, result.geometry, strict=True):
        assert left.equals(right)


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
def test_device_backed_to_parquet_preserves_arrow_schema_metadata(tmp_path) -> None:
    import pyarrow.parquet as pq

    gdf, _owned = _make_device_dga_gdf([Point(0, 0), Point(1, 1)])
    path = tmp_path / "device-backed.parquet"

    gdf.to_parquet(path, write_covering_bbox=True)
    table = pq.read_table(path)

    assert table.schema.metadata is not None
    assert b"geo" in table.schema.metadata
    assert b"pandas" in table.schema.metadata


def test_native_tabular_geometry_only_to_parquet_populates_schema_metadata(tmp_path) -> None:
    import pyarrow.parquet as pq

    gdf = geopandas.GeoDataFrame(
        geometry=geopandas.GeoSeries([Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    )
    payload = to_native_tabular_result(GeometryNativeResult.from_geoseries(gdf.geometry))
    assert payload is not None

    path = tmp_path / "native-geometry-only.parquet"
    payload.to_parquet(path)
    table = pq.read_table(path)

    assert table.schema.metadata is not None
    assert b"geo" in table.schema.metadata


def test_native_tabular_to_parquet_records_fallback_when_device_writer_declines(
    monkeypatch,
    tmp_path,
) -> None:
    geopandas.clear_fallback_events()
    payload = to_native_tabular_result(
        GeometryNativeResult.from_owned(
            from_shapely_geometries([Point(0, 0)]),
            crs="EPSG:4326",
        )
    )
    assert payload is not None

    monkeypatch.setattr(
        io_geoparquet,
        "_write_geoparquet_native_device_payload",
        lambda *args, **kwargs: io_wkb._NativeDeviceWriteStatus(
            written=False,
            fallback_detail="test-only native payload writer miss",
        ),
    )

    path = tmp_path / "native-payload-writer-fallback.parquet"
    payload.to_parquet(path)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert path.exists()
    assert any(
        event.surface == "geopandas.geodataframe.to_parquet"
        and "native device GeoParquet payload writer" in event.reason
        and "test-only native payload writer miss" in event.detail
        for event in fallbacks
    )


def test_native_tabular_to_parquet_compatibility_decline_records_dispatch_not_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    payload = to_native_tabular_result(
        GeometryNativeResult.from_owned(
            from_shapely_geometries([Point(0, 0)], residency=Residency.DEVICE),
            crs="EPSG:4326",
        )
    )
    assert payload is not None

    monkeypatch.setattr(
        io_geoparquet,
        "_write_geoparquet_native_device_payload",
        lambda *args, **kwargs: io_wkb._NativeDeviceWriteStatus(
            written=False,
            compatibility_detail="test-only filesystem-path sink miss",
        ),
    )

    path = tmp_path / "native-payload-writer-compatibility.parquet"
    payload.to_parquet(path)
    dispatches = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert path.exists()
    assert fallbacks == []
    assert any(
        event.surface == "geopandas.geodataframe.to_parquet"
        and event.implementation == "native_payload_arrow_compatibility_export"
        and "test-only filesystem-path sink miss" in event.detail
        for event in dispatches
    )


def test_device_geodataframe_to_parquet_compatibility_decline_records_dispatch_not_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    if not has_gpu_runtime():
        return

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    gdf, _owned = _make_device_dga_gdf([Point(0, 0)])

    monkeypatch.setattr(
        io_geoparquet,
        "_write_geoparquet_native_device",
        lambda *args, **kwargs: io_wkb._NativeDeviceWriteStatus(
            written=False,
            compatibility_detail="test-only public filesystem-path sink miss",
        ),
    )

    path = tmp_path / "device-gdf-compatibility.parquet"
    gdf.to_parquet(path)
    dispatches = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert path.exists()
    assert fallbacks == []
    assert any(
        event.surface == "geopandas.geodataframe.to_parquet"
        and event.implementation == "native_geodataframe_arrow_compatibility_export"
        and "test-only public filesystem-path sink miss" in event.detail
        for event in dispatches
    )


def test_device_geodataframe_small_terminal_write_prefers_arrow_export(
    monkeypatch,
    tmp_path,
) -> None:
    if not has_gpu_runtime():
        return

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    geoms = [box(float(i), float(i), float(i) + 1.0, float(i) + 1.0) for i in range(300)]
    gdf, _owned = _make_device_dga_gdf(geoms)

    def _fail(*_args, **_kwargs):
        raise AssertionError("small terminal write should skip the native device writer")

    monkeypatch.setattr(io_geoparquet, "_write_geoparquet_native_device", _fail)

    path = tmp_path / "device-gdf-small-terminal.parquet"
    gdf.to_parquet(path)
    dispatches = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert path.exists()
    assert fallbacks == []
    assert any(
        event.surface == "geopandas.geodataframe.to_parquet"
        and event.implementation == "native_geodataframe_arrow_terminal_export"
        and "small terminal GeoParquet write prefers the explicit Arrow export" in event.detail
        for event in dispatches
    )


def test_device_geodataframe_small_terminal_write_skips_arrow_shortcut_for_multi_geometry(
    monkeypatch,
    tmp_path,
) -> None:
    if not has_gpu_runtime():
        return

    import pandas as pd

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    geoms = [box(float(i), float(i), float(i) + 1.0, float(i) + 1.0) for i in range(2)]
    gdf, _owned = _make_device_dga_gdf(geoms)
    alt_owned = from_shapely_geometries([geom.buffer(0.1) for geom in geoms])
    alt_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test secondary geometry native parquet write path",
    )
    alt_dga = DeviceGeometryArray._from_owned(alt_owned, crs=gdf.crs)
    gdf["alt_geom"] = pd.Series(alt_dga, copy=False, name="alt_geom")
    gdf["alt_geom"].array.crs = gdf.crs

    def _fail(*_args, **_kwargs):
        raise AssertionError("multi-geometry small write must skip native tabular shortcut")

    monkeypatch.setattr(io_geoparquet, "_write_geoparquet_native_tabular_result", _fail)

    path = tmp_path / "device-gdf-multi-geometry-small-terminal.parquet"
    gdf.to_parquet(path)

    assert path.exists()


def test_device_geodataframe_small_terminal_write_skips_arrow_shortcut_for_unsupported_secondary_geometry(
    monkeypatch,
    tmp_path,
) -> None:
    if not has_gpu_runtime():
        return

    import pandas as pd

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    geoms = [box(float(i), float(i), float(i) + 1.0, float(i) + 1.0) for i in range(2)]
    gdf, _owned = _make_device_dga_gdf(geoms)
    gdf["alt_geom"] = geopandas.GeoSeries(
        [
            GeometryCollection([Point(0.0, 0.0), LineString([(0.0, 0.0), (1.0, 1.0)])]),
            GeometryCollection([Point(1.0, 1.0)]),
        ],
        crs=gdf.crs,
    )
    gdf["alt_geom"] = pd.Series(gdf["alt_geom"].array, copy=False, name="alt_geom")

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "unsupported secondary geometry must skip native tabular small-write shortcut"
        )

    monkeypatch.setattr(io_geoparquet, "_write_geoparquet_native_tabular_result", _fail)

    path = tmp_path / "device-gdf-unsupported-secondary-small-terminal.parquet"
    gdf.to_parquet(path)

    assert path.exists()


def test_device_geodataframe_to_parquet_fallback_decline_raises_in_strict_native(
    monkeypatch,
    tmp_path,
) -> None:
    if not has_gpu_runtime():
        return

    from vibespatial.runtime.fallbacks import StrictNativeFallbackError
    from vibespatial.testing import strict_native_environment

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    gdf, _owned = _make_device_dga_gdf([Point(0, 0)])

    monkeypatch.setattr(
        io_geoparquet,
        "_write_geoparquet_native_device",
        lambda *args, **kwargs: io_wkb._NativeDeviceWriteStatus(
            written=False,
            fallback_detail="test-only missing pylibcudf support",
        ),
    )

    path = tmp_path / "device-gdf-strict-fallback.parquet"
    with pytest.raises(StrictNativeFallbackError):
        with strict_native_environment():
            gdf.to_parquet(path)

    fallbacks = geopandas.get_fallback_events(clear=True)
    assert any("test-only missing pylibcudf support" in event.detail for event in fallbacks)


def test_native_tabular_small_terminal_write_prefers_arrow_export(
    monkeypatch,
    tmp_path,
) -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    payload = to_native_tabular_result(
        GeometryNativeResult.from_owned(
            from_shapely_geometries(
                [box(float(i), float(i), float(i) + 1.0, float(i) + 1.0) for i in range(300)],
                residency=Residency.DEVICE,
            ),
            crs="EPSG:4326",
        )
    )
    assert payload is not None

    def _fail(*_args, **_kwargs):
        raise AssertionError("small terminal payload write should skip the native device writer")

    monkeypatch.setattr(io_geoparquet, "_write_geoparquet_native_device_payload", _fail)

    path = tmp_path / "native-payload-small-terminal.parquet"
    payload.to_parquet(path)
    dispatches = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert path.exists()
    assert fallbacks == []
    assert any(
        event.surface == "geopandas.geodataframe.to_parquet"
        and event.implementation == "native_payload_arrow_terminal_export"
        and "small terminal GeoParquet write prefers the explicit Arrow export" in event.detail
        for event in dispatches
    )


def test_arrow_backed_native_attributes_feather_without_pandas_materialization(
    tmp_path,
    monkeypatch,
) -> None:
    attr_table = NativeAttributeTable(
        arrow_table=pa.table({"value": [1, 2]}),
        index_override=pd.RangeIndex(2),
    )
    payload = NativeTabularResult(
        attributes=attr_table,
        geometry=GeometryNativeResult.from_geoseries(
            geopandas.GeoSeries([Point(0, 0), Point(1, 1)], crs="EPSG:4326")
        ),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("arrow-backed attribute export should not require pandas")

    monkeypatch.setattr(NativeAttributeTable, "to_pandas", _fail)

    path = tmp_path / "arrow-backed-native.feather"
    payload.to_feather(path, index=False)
    result = geopandas.read_feather(path)

    assert list(result["value"]) == [1, 2]
    assert result.crs == "EPSG:4326"


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
def test_arrow_backed_native_attributes_device_parquet_without_pandas_materialization(
    tmp_path,
    monkeypatch,
) -> None:
    import pyarrow.parquet as pq

    _gdf, owned = _make_device_dga_gdf([Point(0, 0), Point(1, 1)])
    payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            arrow_table=pa.table({"value": [1, 2]}),
            index_override=pd.RangeIndex(2),
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs="EPSG:4326"),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("device parquet write should not require pandas attributes")

    monkeypatch.setattr(NativeAttributeTable, "to_pandas", _fail)

    path = tmp_path / "arrow-backed-native-device.parquet"
    payload.to_parquet(path)
    table = pq.read_table(path)

    assert table.column_names == ["value", "geometry"]
    assert table.schema.metadata is not None
    assert b"geo" in table.schema.metadata


def test_create_metadata_uses_device_geometry_fast_path(monkeypatch) -> None:
    if not has_gpu_runtime():
        return

    import vibespatial.api.io.arrow as api_io_arrow

    gdf, _owned = _make_device_dga_gdf([Point(0, 0), Point(1, 1)])

    def _fail(*_args, **_kwargs):
        raise AssertionError("device metadata fast path should not consult shapely type ids")

    monkeypatch.setattr(api_io_arrow.shapely, "get_type_id", _fail)
    monkeypatch.setattr(
        DeviceGeometryArray,
        "total_bounds",
        property(lambda self: np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)),
    )

    metadata = api_io_arrow._create_metadata(
        gdf,
        schema_version=None,
        geometry_encoding={"geometry": "point"},
        write_covering_bbox=False,
    )

    assert metadata["columns"]["geometry"]["geometry_types"] == ["Point"]
    assert metadata["columns"]["geometry"]["bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_create_geometry_metadata_uses_device_geometry_fast_path(monkeypatch) -> None:
    if not has_gpu_runtime():
        return

    import vibespatial.api.io.arrow as api_io_arrow

    gdf, _owned = _make_device_dga_gdf([Point(0, 0), Point(1, 1)])
    series = gdf.geometry

    def _fail(*_args, **_kwargs):
        raise AssertionError("device metadata fast path should not consult shapely type ids")

    monkeypatch.setattr(api_io_arrow.shapely, "get_type_id", _fail)
    monkeypatch.setattr(
        DeviceGeometryArray,
        "total_bounds",
        property(lambda self: np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)),
    )

    metadata = api_io_arrow._create_geometry_metadata(
        {"geometry": series},
        primary_column="geometry",
        schema_version=None,
        geometry_encoding={"geometry": "point"},
        write_covering_bbox=False,
    )

    assert metadata["columns"]["geometry"]["geometry_types"] == ["Point"]
    assert metadata["columns"]["geometry"]["bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_native_to_parquet_rejects_existing_bbox_column(tmp_path) -> None:
    gdf = _make_dga_gdf([Point(0, 0), Point(1, 1)])
    gdf = gdf.assign(bbox=[0, 0])

    with pytest.raises(ValueError, match="An existing column 'bbox' already exists in the dataframe"):
        gdf.to_parquet(tmp_path / "existing-bbox.parquet", write_covering_bbox=True)


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
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


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
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


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
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


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
def test_write_geoparquet_roundtrips_multipoint_wkb(tmp_path) -> None:
    """MultiPoint DGA → WKB write → read roundtrip."""
    mpoints = [MultiPoint([(i, 0), (i + 1, 1)]) for i in range(3)]
    gdf = _make_dga_gdf(mpoints)
    path = tmp_path / "mpoint_wkb.parquet"
    write_geoparquet(gdf, path, geometry_encoding="WKB")
    result = geopandas.read_parquet(path)
    assert len(result) == 3
    for i, mp in enumerate(mpoints):
        assert result.geometry.iloc[i].equals(mp)


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
def test_write_geoparquet_roundtrips_multilinestring_wkb(tmp_path) -> None:
    """MultiLineString DGA → WKB write → read roundtrip."""
    mlines = [MultiLineString([[(i, 0), (i + 1, 1)], [(i + 2, 0), (i + 3, 1)]]) for i in range(3)]
    gdf = _make_dga_gdf(mlines)
    path = tmp_path / "mls_wkb.parquet"
    write_geoparquet(gdf, path, geometry_encoding="WKB")
    result = geopandas.read_parquet(path)
    assert len(result) == 3
    for i, mls in enumerate(mlines):
        assert result.geometry.iloc[i].equals(mls)


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
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


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
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


def test_geopandas_to_arrow_device_geoarrow_uses_native_path_without_host_materialization() -> None:
    geopandas.clear_fallback_events()
    gdf, owned = _make_device_dga_gdf([Point(0, 0), Point(1, 1)])
    owned.diagnostics.clear()

    table, geometry_encoding = api_geoarrow.geopandas_to_arrow(gdf, geometry_encoding="geoarrow")
    fallbacks = geopandas.get_fallback_events(clear=True)
    mat_events = [event for event in owned.diagnostics if event.kind == DiagnosticKind.MATERIALIZATION]

    assert table is not None
    assert geometry_encoding["geometry"] == "point"
    assert fallbacks == []
    assert mat_events == []


def test_geopandas_to_arrow_device_geoarrow_succeeds_in_strict_native() -> None:
    from vibespatial.testing import strict_native_environment

    geopandas.clear_fallback_events()
    gdf, owned = _make_device_dga_gdf([Point(0, 0), Point(1, 1)])
    owned.diagnostics.clear()

    with strict_native_environment():
        table, geometry_encoding = api_geoarrow.geopandas_to_arrow(
            gdf,
            geometry_encoding="geoarrow",
        )

    fallbacks = geopandas.get_fallback_events(clear=True)
    mat_events = [event for event in owned.diagnostics if event.kind == DiagnosticKind.MATERIALIZATION]

    assert table is not None
    assert geometry_encoding["geometry"] == "point"
    assert fallbacks == []
    assert mat_events == []


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


def test_wkb_encode_from_owned_buffers_point_preserves_empty_rows() -> None:
    import shapely

    owned = from_shapely_geometries([Point(0, 0), Point(), None, Point(1, 1)])
    _field, arr = io_arrow._encode_owned_wkb_array(owned, field_name="g", crs=None)
    restored = [shapely.from_wkb(value.as_py()) if value.as_py() is not None else None for value in arr]

    assert restored[0].equals(Point(0, 0))
    assert restored[1].is_empty
    assert restored[2] is None
    assert restored[3].equals(Point(1, 1))


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


def test_wkb_encode_from_device_owned_ignores_stale_host_metadata() -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    import shapely

    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])
    poly2 = Polygon([(3, 0), (5, 0), (5, 2), (3, 0)])
    mpoly = MultiPolygon(
        [
            Polygon([(6, 0), (8, 0), (8, 2), (6, 0)]),
            Polygon([(8, 2), (10, 2), (10, 4), (8, 2)]),
        ]
    )

    owned = from_shapely_geometries([poly, poly2, mpoly], residency=Residency.DEVICE)
    owned._validity = np.asarray([True, False, True], dtype=np.bool_)
    owned._tags = np.asarray([0, 0, 0], dtype=np.int8)
    owned._family_row_offsets = np.asarray([0, 99, 99], dtype=np.int32)

    _field, arr = io_arrow._encode_owned_wkb_array(owned, field_name="g", crs=None)
    restored = [shapely.from_wkb(value.as_py()) if value.as_py() is not None else None for value in arr]

    assert len(restored) == 3
    assert restored[0].equals(poly)
    assert restored[1].equals(poly2)
    assert restored[2].equals(mpoly)


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


def test_read_parquet_pylibcudf_keeps_device_geometry_unmaterialized(tmp_path) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    gdf, _ = _make_device_dga_gdf(
        [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
            Polygon([(3, 0), (5, 0), (5, 2), (3, 0)]),
        ]
    )
    path = tmp_path / "read_device_geometry_no_materialization.parquet"

    write_geoparquet(gdf, path, geometry_encoding="geoarrow")
    result = geopandas.read_parquet(path)

    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert result.geometry.values._shapely_cache is None
    owned = result.geometry.values.owned
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.TRANSFER] == []
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION] == []


def test_decode_arrow_geoparquet_table_to_owned_handles_unnamed_single_geometry_column() -> None:
    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
            MultiPolygon([Polygon([(3, 0), (5, 0), (5, 2), (3, 0)])]),
        ]
    )
    _field, geometry_array = io_arrow._encode_owned_wkb_array(
        owned,
        field_name="geometry",
        crs="EPSG:4326",
    )
    unnamed_table = pa.Table.from_arrays([geometry_array], names=[""])
    geo_metadata = {
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB"}},
    }

    restored = io_geoparquet._decode_arrow_geoparquet_table_to_owned(
        unnamed_table,
        geo_metadata,
        column_index=0,
    )

    restored_shapely = restored.to_shapely()
    expected_shapely = owned.to_shapely()
    assert restored_shapely[0].equals(expected_shapely[0])
    assert restored_shapely[1].equals(expected_shapely[1])


def test_read_parquet_pylibcudf_wkb_decode_miss_falls_back_explicitly(tmp_path, monkeypatch) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    geopandas.clear_fallback_events()
    gdf = geopandas.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
                Polygon([(2, 0), (4, 0), (4, 2), (2, 0)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    path = tmp_path / "pylibcudf_wkb_decode_fallback.parquet"

    write_geoparquet(gdf, path, geometry_encoding="WKB")
    monkeypatch.setattr(
        io_geoparquet,
        "_decode_pylibcudf_geoparquet_column_to_owned",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            NotImplementedError("test-only pylibcudf GeoParquet decode miss")
        ),
    )
    result = geopandas.read_parquet(path)

    assert len(result) == 2
    assert result.geometry.iloc[0].equals(gdf.geometry.iloc[0])
    assert result.geometry.iloc[1].equals(gdf.geometry.iloc[1])
    fallbacks = geopandas.get_fallback_events(clear=True)
    assert any(
        event.surface == "vibespatial.io.geoparquet"
        and "GPU GeoParquet geometry decode could not complete" in event.reason
        for event in fallbacks
    )
    assert any(
        event.surface == "vibespatial.io.geoparquet"
        and "test-only pylibcudf GeoParquet decode miss" in event.detail
        for event in fallbacks
    )


def test_read_parquet_geometrycollection_wkb_uses_terminal_host_compatibility_fallback(tmp_path) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    geopandas.clear_fallback_events()
    gdf = geopandas.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": [
                GeometryCollection([Point(0, 0), LineString([(0, 0), (1, 1)])]),
                Polygon([(2, 0), (4, 0), (4, 2), (2, 0)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    path = tmp_path / "geometrycollection_terminal_host_fallback.parquet"

    write_geoparquet(gdf, path, geometry_encoding="WKB")
    result = geopandas.read_parquet(path)

    assert len(result) == 2
    assert result.geometry.iloc[0].equals(gdf.geometry.iloc[0])
    assert result.geometry.iloc[1].equals(gdf.geometry.iloc[1])
    fallbacks = geopandas.get_fallback_events(clear=True)
    assert any(
        event.surface == "vibespatial.io.geoparquet"
        and "owned native result model" in event.reason
        and "GeometryCollection" in event.detail
        for event in fallbacks
    )


def test_write_geoparquet_strict_native_wkb_succeeds_for_geometry_array_with_device_owned(tmp_path) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    from vibespatial.testing import strict_native_environment

    gdf = geopandas.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
                Polygon([(3, 0), (5, 0), (5, 2), (3, 0)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    owned = gdf.geometry.values.to_owned()
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="strict native parquet WKB write regression",
    )

    path = tmp_path / "strict_native_geometry_array_device_owned_wkb.parquet"
    with strict_native_environment():
        gdf.to_parquet(path, geometry_encoding="WKB")

    result = geopandas.read_parquet(path)
    assert len(result) == 2
    assert result.geometry.iloc[0].equals(gdf.geometry.iloc[0])
    assert result.geometry.iloc[1].equals(gdf.geometry.iloc[1])


def test_write_geoparquet_strict_native_wkb_succeeds_for_geometry_array_with_host_owned(tmp_path) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    from vibespatial.testing import strict_native_environment

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
            MultiPolygon(
                [
                    Polygon([(3, 0), (5, 0), (5, 2), (3, 0)]),
                    Polygon([(6, 0), (8, 0), (8, 2), (6, 0)]),
                ]
            ),
        ]
    )
    ga = GeometryArray.from_owned(owned, crs="EPSG:4326")
    gdf = geopandas.GeoDataFrame(
        {"value": [1, 2], "geometry": geopandas.GeoSeries(ga, crs="EPSG:4326")},
        geometry="geometry",
        crs="EPSG:4326",
    )

    path = tmp_path / "strict_native_geometry_array_host_owned_wkb.parquet"
    with strict_native_environment():
        gdf.to_parquet(path, geometry_encoding="WKB")

    result = geopandas.read_parquet(path)
    assert len(result) == 2
    assert result.geometry.iloc[0].equals(gdf.geometry.iloc[0])
    assert result.geometry.iloc[1].equals(gdf.geometry.iloc[1])


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


def test_write_geoparquet_device_geoarrow_preserves_empty_point_rows(tmp_path) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    import pyarrow as pa
    import pyarrow.parquet as pq

    geopandas.clear_fallback_events()
    gdf, owned = _make_device_dga_gdf([Point(0, 0), Point(), None, Point(1, 1)])
    path = tmp_path / "device_geoarrow_empty_points.parquet"

    write_geoparquet(gdf, path, geometry_encoding="geoarrow")
    result = geopandas.read_parquet(path)
    arrow_table = pq.read_table(path)

    assert result.geometry.iloc[0].equals(Point(0, 0))
    assert result.geometry.iloc[1].is_empty
    assert result.geometry.iloc[2] is None
    assert result.geometry.iloc[3].equals(Point(1, 1))
    assert not arrow_table.schema.field("geometry").type.equals(pa.binary())
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.TRANSFER] == []
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION] == []


def test_write_geoparquet_device_geoarrow_geometry_only_has_no_transfer_or_materialization(tmp_path) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    import pandas as pd

    gdf, owned = _make_device_dga_gdf([Point(0, 0), Point(1, 1), Point(2, 2)])
    geometry_only = pd.DataFrame(
        {"geometry": pd.Series(gdf.geometry.values, copy=False, name="geometry")},
        copy=False,
    )
    geometry_only.__class__ = geopandas.GeoDataFrame
    geometry_only._geometry_column_name = "geometry"
    geometry_only["geometry"].array.crs = gdf.crs

    path = tmp_path / "device_geoarrow_geometry_only.parquet"
    write_geoparquet(geometry_only, path, geometry_encoding="geoarrow")
    result = geopandas.read_parquet(path)

    assert len(result) == 3
    assert result.geometry.iloc[0].equals(Point(0, 0))
    assert result.geometry.iloc[2].equals(Point(2, 2))
    assert isinstance(result.index, pd.RangeIndex)
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.TRANSFER] == []
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION] == []


def test_write_geoparquet_device_geoarrow_geometry_only_skips_from_pandas(tmp_path, monkeypatch) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    import pandas as pd

    gdf, _owned = _make_device_dga_gdf([Point(0, 0), Point(1, 1), Point(2, 2)])
    geometry_only = pd.DataFrame(
        {"geometry": pd.Series(gdf.geometry.values, copy=False, name="geometry")},
        copy=False,
    )
    geometry_only.__class__ = geopandas.GeoDataFrame
    geometry_only._geometry_column_name = "geometry"
    geometry_only["geometry"].array.crs = gdf.crs

    def _fail(*_args, **_kwargs):
        raise AssertionError("geometry-only native device parquet write should skip the from_pandas fallback")

    monkeypatch.setattr("vibespatial.io.wkb._native_host_attribute_table_from_pandas", _fail)

    path = tmp_path / "device_geoarrow_geometry_only_no_from_pandas.parquet"
    write_geoparquet(geometry_only, path, geometry_encoding="geoarrow")

    result = geopandas.read_parquet(path)
    assert len(result) == 3
    assert result.geometry.iloc[2].equals(Point(2, 2))


def test_write_geoparquet_device_geoarrow_numeric_attributes_skip_from_pandas(
    tmp_path,
    monkeypatch,
) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    gdf, _owned = _make_device_dga_gdf([Point(0, 0), Point(1, 1), Point(2, 2)])

    def _fail(*_args, **_kwargs):
        raise AssertionError("numeric attribute native device parquet write should skip the from_pandas fallback")

    monkeypatch.setattr("vibespatial.io.wkb._native_host_attribute_table_from_pandas", _fail)

    path = tmp_path / "device_geoarrow_numeric_attrs_no_from_pandas.parquet"
    write_geoparquet(gdf, path, geometry_encoding="geoarrow")

    result = geopandas.read_parquet(path)
    assert list(result["idx"]) == [0, 1, 2]
    assert result.geometry.iloc[0].equals(Point(0, 0))


def test_write_geoparquet_device_geoarrow_ignores_stale_host_metadata(tmp_path) -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return

    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])
    poly2 = Polygon([(3, 0), (5, 0), (5, 2), (3, 0)])

    owned = from_shapely_geometries([poly, poly2], residency=Residency.DEVICE)
    owned._validity = np.asarray([False, False], dtype=np.bool_)
    owned._tags = np.asarray([FAMILY_TAGS[GeometryFamily.POINT]] * 2, dtype=np.int8)
    owned._family_row_offsets = np.asarray([99, 99], dtype=np.int32)
    owned.diagnostics.clear()
    dga = DeviceGeometryArray._from_owned(owned, crs="EPSG:4326")
    geometry_series = pd.Series(dga, copy=False, name="geometry")
    gdf = pd.DataFrame({"geometry": geometry_series}, copy=False)
    gdf.__class__ = geopandas.GeoDataFrame
    gdf._geometry_column_name = "geometry"
    gdf["geometry"].array.crs = "EPSG:4326"

    path = tmp_path / "device_geoarrow_stale_host_metadata.parquet"
    write_geoparquet(gdf, path, geometry_encoding="geoarrow")
    result = geopandas.read_parquet(path)

    assert len(result) == 2
    assert result.geometry.iloc[0].equals(poly)
    assert result.geometry.iloc[1].equals(poly2)
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.TRANSFER] == []
    assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION] == []
    assert geopandas.get_fallback_events(clear=True) == []


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


def test_write_geoparquet_geometry_native_result_roundtrip(tmp_path) -> None:
    residency = Residency.DEVICE if has_gpu_runtime() else Residency.HOST
    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2)],
        residency=residency,
    )
    owned.diagnostics.clear()
    native = GeometryNativeResult.from_owned(owned, crs="EPSG:4326")

    path = tmp_path / "geometry-native-result.parquet"
    write_geoparquet(native, path, geometry_encoding="geoarrow")
    result = geopandas.read_parquet(path)

    assert len(result) == 3
    assert result.geometry.iloc[0].equals(Point(0, 0))
    assert result.geometry.iloc[2].equals(Point(2, 2))
    if residency is Residency.DEVICE:
        assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.TRANSFER] == []
        assert [e for e in owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION] == []


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
def test_parquet_roundtrip_preserves_crs(tmp_path) -> None:
    """GeoParquet roundtrip via DGA write path preserves CRS metadata."""
    from pyproj import CRS

    pts = [Point(i, i) for i in range(3)]
    gdf = _make_dga_gdf(pts, crs="EPSG:32632")
    path = tmp_path / "crs.parquet"
    write_geoparquet(gdf, path, geometry_encoding="geoarrow")
    result = geopandas.read_parquet(path)
    assert CRS(result.crs) == CRS("EPSG:32632")


@pytest.mark.skipif(not has_pylibcudf_support(), reason="pylibcudf not available")
def test_parquet_roundtrip_preserves_attributes(tmp_path) -> None:
    """Non-geometry columns survive DGA write path roundtrip."""
    pts = [Point(i, i) for i in range(5)]
    gdf = _make_dga_gdf(pts)
    path = tmp_path / "attrs.parquet"
    write_geoparquet(gdf, path, geometry_encoding="WKB")
    result = geopandas.read_parquet(path)
    assert list(result["idx"]) == list(range(5))
