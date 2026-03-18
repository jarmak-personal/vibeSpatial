"""Thin facade that re-exports the public API from the split io modules.

The original monolithic io_arrow module has been refactored into:
  - io_wkb: WKB encoding/decoding, CUDA kernels, device encode helpers
  - io_geoarrow: GeoArrow bridge, conversion, benchmarks
  - io_geoparquet: GeoParquet scanning, reading, writing
  - io_pylibcudf: pylibcudf integration layer for GPU-accelerated decoding

This file preserves backwards compatibility so that ``import vibespatial.io_arrow``
and ``from vibespatial.io_arrow import X`` continue to work.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-exports from core geometry modules (backwards compat)
# ---------------------------------------------------------------------------
from vibespatial.geometry_buffers import GeometryFamily  # noqa: F401

# ---------------------------------------------------------------------------
# io_geoarrow  – GeoArrow bridge, conversion, benchmarks
# ---------------------------------------------------------------------------
from vibespatial.io_geoarrow import (  # noqa: F401
    GeoArrowBridgeBenchmark,
    GeoArrowCodecPlan,
    NativeGeometryBenchmark,
    WKBBridgeBenchmark,
    _arrow_validity_mask,
    _build_single_family_owned,
    _child_selection_mask,
    _construct_geoarrow_array_with_explicit_fallback,
    _decode_geoarrow_array_to_owned,
    _decode_geoarrow_linestring_like,
    _decode_geoarrow_multilinestring,
    _decode_geoarrow_multipolygon,
    _decode_geoarrow_point,
    _decode_geoarrow_polygon,
    _encode_list_family,
    _encode_point_family,
    _extract_point_xy,
    _family_decode_state,
    _full_offsets_from_local,
    _geoarrow_field_metadata,
    _owned_geoarrow_fast_path_reason,
    _sample_owned_for_geoarrow_benchmark,
    benchmark_geoarrow_bridge,
    benchmark_native_geometry_codec,
    benchmark_wkb_bridge,
    decode_owned_geoarrow,
    encode_owned_geoarrow,
    encode_owned_geoarrow_array,
    geodataframe_from_arrow,
    geodataframe_to_arrow,
    geoseries_from_arrow,
    geoseries_from_owned,
    geoseries_to_arrow,
    plan_geoarrow_codec,
)

# ---------------------------------------------------------------------------
# io_geoparquet  – GeoParquet scanning, reading, writing
# ---------------------------------------------------------------------------
from vibespatial.io_geoparquet import (  # noqa: F401
    GeoParquetChunkPlan,
    GeoParquetEngineBenchmark,
    GeoParquetEnginePlan,
    GeoParquetScanPlan,
    _build_geoparquet_metadata_summary_from_pyarrow,
    _concat_offsets,
    _concatenate_owned_arrays,
    _decode_geoparquet_table_to_owned,
    _geoparquet_geometry_column_crs,
    _is_local_geoparquet_file,
    _load_geoparquet_metadata,
    _parquet_column_path,
    _plan_geoparquet_chunks,
    _project_arrow_schema,
    _pylibcudf_table_to_geopandas,
    _read_geoparquet_table_with_pyarrow,
    _read_geoparquet_table_with_pylibcudf,
    _read_geoparquet_with_pyarrow,
    _read_geoparquet_with_pylibcudf,
    _read_non_geometry_geoparquet_columns_with_pyarrow,
    _rebuild_arrow_array_with_schema_type,
    _rebuild_arrow_table_with_schema,
    _supports_pylibcudf_geoparquet_read,
    _write_geoparquet_native,
    benchmark_geoparquet_scan_engine,
    plan_geoparquet_engine,
    plan_geoparquet_scan,
    read_geoparquet,
    read_geoparquet_owned,
    write_geoparquet,
)

# ---------------------------------------------------------------------------
# io_pylibcudf  – pylibcudf integration layer
# ---------------------------------------------------------------------------
from vibespatial.io_pylibcudf import (  # noqa: F401
    _build_device_mixed_owned,
    _build_device_single_family_owned,
    _build_device_wkb_linestring_family,
    _build_device_wkb_multilinestring_family,
    _build_device_wkb_multipoint_family,
    _build_device_wkb_multipolygon_family,
    _build_device_wkb_polygon_family,
    _decode_pylibcudf_geoparquet_column_to_owned,
    _decode_pylibcudf_geoparquet_table_to_owned,
    _decode_pylibcudf_linestring_like_geoarrow_column_to_owned,
    _decode_pylibcudf_multilinestring_geoarrow_column_to_owned,
    _decode_pylibcudf_multipolygon_geoarrow_column_to_owned,
    _decode_pylibcudf_point_geoarrow_column_to_owned,
    _decode_pylibcudf_polygon_geoarrow_column_to_owned,
    _decode_pylibcudf_wkb_general_column_to_owned,
    _decode_pylibcudf_wkb_linestring_column_to_owned,
    _decode_pylibcudf_wkb_multilinestring_column_to_owned,
    _decode_pylibcudf_wkb_multipoint_column_to_owned,
    _decode_pylibcudf_wkb_multipolygon_column_to_owned,
    _decode_pylibcudf_wkb_point_column_to_owned,
    _decode_pylibcudf_wkb_point_linestring_column_to_owned,
    _decode_pylibcudf_wkb_polygon_column_to_owned,
    _device_child_selection_mask,
    _device_compact_offsets,
    _device_mask_count,
    _device_select_true,
    _is_pylibcudf_table,
    _pylibcudf_buffer_view,
    _pylibcudf_list_offsets,
    _pylibcudf_point_xy_children,
    _pylibcudf_unpack_le_float64,
    _pylibcudf_unpack_le_uint32,
    _pylibcudf_validity_mask,
    _pylibcudf_wkb_offsets,
    _pylibcudf_wkb_payload,
    _require_pylibcudf_zero_offset,
    _scan_pylibcudf_wkb_headers,
)

# ---------------------------------------------------------------------------
# io_wkb  – WKB encode/decode, CUDA kernels, device encode helpers
# ---------------------------------------------------------------------------
from vibespatial.io_wkb import (  # noqa: F401
    WKB_ID_FAMILIES,
    WKB_POINT_RECORD_DTYPE,
    WKB_TYPE_IDS,
    DeviceWKBHeaderScan,
    WKBBridgePlan,
    WKBPartitionPlan,
    _append_coordinate_range,
    _append_point_row,
    _append_shapely_geometry_state,
    _apply_geoarrow_child_metadata,
    _arrow_binary_offsets,
    _compression_type_from_name,
    _decode_arrow_wkb_linestring_fast,
    _decode_arrow_wkb_linestring_uniform_fast,
    _decode_arrow_wkb_point_fast,
    _decode_arrow_wkb_polygon_fast,
    _decode_arrow_wkb_polygon_uniform_fast,
    _decode_linestring_wkb_payload,
    _decode_multilinestring_wkb_payload,
    _decode_multipoint_wkb_payload,
    _decode_multipolygon_wkb_payload,
    _decode_native_wkb,
    _decode_point_batch,
    _decode_polygon_wkb_payload,
    _device_family_row_selection,
    _device_full_offsets_from_local,
    _device_geoarrow_fast_path_reason_owned,
    _device_list_column,
    _device_point_values_column,
    _device_validity_gpumask,
    _device_wkb_lengths_for_family,
    _encode_family_row_wkb,
    _encode_native_wkb,
    _encode_owned_geoarrow_column_device,
    _encode_owned_wkb_array,
    _encode_owned_wkb_column_device,
    _encode_point_wkb_batch,
    _finalize_wkb_family_buffer,
    _hexify_if_requested,
    _homogeneous_family,
    _launch_device_wkb_write_kernel,
    _new_wkb_family_state,
    _normalize_wkb_value,
    _pack_linestring_wkb,
    _pack_multilinestring_wkb,
    _pack_multipoint_wkb,
    _pack_multipolygon_wkb,
    _pack_polygon_wkb,
    _scan_wkb_value,
    _try_gpu_wkb_arrow_decode,
    _wkb_encode_kernels,
    _write_geoparquet_native_device,
    decode_wkb_arrow_array_owned,
    decode_wkb_owned,
    encode_wkb_owned,
    has_pyarrow_support,
    has_pylibcudf_support,
    plan_wkb_bridge,
    plan_wkb_partition,
)
from vibespatial.owned_geometry import FAMILY_TAGS  # noqa: F401
