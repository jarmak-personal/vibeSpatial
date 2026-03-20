from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter

import numpy as np

try:
    import orjson as _json_fast

    def _fast_json_loads(data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        return _json_fast.loads(data)
except ImportError:
    _json_fast = None  # type: ignore[assignment]

    def _fast_json_loads(data):
        return json.loads(data)

try:
    import simdjson as _simdjson

    _HAS_SIMDJSON = True
except ImportError:
    _simdjson = None
    _HAS_SIMDJSON = False


from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    FamilyGeometryBuffer,
    GeoArrowBufferView,
    MixedGeoArrowView,
    OwnedGeometryArray,
    from_geoarrow,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event


def _simdjson_loads(data):
    """Parse JSON using simdjson, materializing to native Python types."""
    if _simdjson is None:
        raise RuntimeError(
            "pysimdjson is required for the simdjson strategy; install with: pip install pysimdjson"
        )
    parser = _simdjson.Parser()
    if isinstance(data, str):
        data = data.encode("utf-8")
    parsed = parser.parse(data)
    return parsed.as_dict()  # materialize to native Python for Tier 1 compatibility


@dataclass(frozen=True)
class GeoJSONIngestPlan:
    implementation: str
    selected_strategy: str
    uses_stream_tokenizer: bool
    uses_pylibcudf: bool
    uses_native_geometry_assembly: bool
    reason: str


@dataclass
class GeoJSONOwnedBatch:
    geometry: OwnedGeometryArray
    _properties: list[dict[str, object]] | None = None
    _properties_json: list[str | None] | None = None
    _properties_loader: Callable[[], list[dict[str, object]]] | None = None

    @property
    def properties(self) -> list[dict[str, object]]:
        if self._properties is None:
            if self._properties_loader is not None:
                self._properties = self._properties_loader()
                self._properties_loader = None
            else:
                self._properties = _decode_geojson_property_json_values(self._properties_json or [])
            self._properties_json = None
        return self._properties


@dataclass(frozen=True)
class GeoJSONIngestBenchmark:
    implementation: str
    geometry_type: str
    rows: int
    elapsed_seconds: float
    rows_per_second: float


def plan_geojson_ingest(*, prefer: str = "auto") -> GeoJSONIngestPlan:
    use_pylibcudf = False
    if prefer == "auto":
        selected_strategy = "fast-json"
    else:
        selected_strategy = prefer
    implementation = {
        "fast-json": "geojson_fast_json_vectorized",
        "simdjson": "geojson_simdjson_vectorized",
        "chunked": "geojson_chunked_vectorized",
        "full-json": "geojson_full_json_native",
        "stream": "geojson_stream_native",
        "tokenizer": "geojson_tokenizer_native",
        "pylibcudf-arrays": "geojson_pylibcudf_arrays_experimental",
        "pylibcudf-rowized": "geojson_pylibcudf_rowized_experimental",
        "gpu-byte-classify": "geojson_gpu_byte_classify",
    }.get(selected_strategy, "geojson_stream_native")
    if selected_strategy == "fast-json":
        reason = (
            "Fast GeoJSON ingest using orjson (if available) for parsing and vectorized "
            "per-family coordinate extraction directly into numpy arrays. Eliminates "
            "per-feature geometry assembly loops and span-discovery overhead."
        )
    elif selected_strategy == "simdjson":
        if not _HAS_SIMDJSON:
            raise RuntimeError(
                "pysimdjson is required for the simdjson strategy; "
                "install with: pip install pysimdjson"
            )
        reason = (
            "GeoJSON ingest using pysimdjson for SIMD-accelerated JSON parsing and "
            "vectorized per-family coordinate extraction directly into numpy arrays. "
            "Requires the pysimdjson package."
        )
    elif selected_strategy == "chunked":
        reason = (
            "Chunked GeoJSON ingest: split the features array into byte-range chunks, "
            "parse each chunk with orjson, and extract coordinates with vectorized numpy. "
            "Reduces peak memory pressure for very large files."
        )
    else:
        reason = (
            "Standard GeoJSON FeatureCollection ingest uses repo-owned native geometry assembly. "
            "The current fastest host path is fast-json parse plus vectorized assembly, while the "
            "structural tokenizer paths remain the GPU-oriented seam for future optimization."
        )
    if selected_strategy == "gpu-byte-classify":
        reason = (
            "GPU byte-classification GeoJSON ingest: NVRTC kernels for byte classification, "
            "structural scanning, coordinate extraction, and ASCII-to-fp64 parsing. "
            "Property extraction stays on CPU (hybrid design)."
        )
    elif prefer == "pylibcudf":
        use_pylibcudf = True
        selected_strategy = "pylibcudf"
        implementation = "geojson_pylibcudf_native"
        reason = (
            "Use host feature-span planning for the FeatureCollection wrapper, then bulk-parse "
            "per-feature geometry payloads on GPU with pylibcudf. This keeps the tokenizer seam "
            "aligned with future CCCL-style partition, compaction, and decode passes."
        )
    elif prefer == "pylibcudf-arrays":
        use_pylibcudf = True
        selected_strategy = "pylibcudf-arrays"
        implementation = "geojson_pylibcudf_arrays_experimental"
        reason = (
            "Use the experimental wildcard-array GPU path. It extracts geometry arrays directly "
            "from the full FeatureCollection and assembles owned buffers without host feature "
            "splitting, but it stays explicit until the extraction path is faster than the "
            "current host-split GPU route."
        )
    elif prefer == "pylibcudf-rowized":
        use_pylibcudf = True
        selected_strategy = "pylibcudf-rowized"
        implementation = "geojson_pylibcudf_rowized_experimental"
        reason = (
            "Use the experimental pylibcudf feature-array rowization path. This is a prototype "
            "for future device-side rowization work, not the default GPU route, because current "
            "interleave-based rowization is correctness-limited and much slower than host-span "
            "planning on measured workloads."
        )
    return GeoJSONIngestPlan(
        implementation=implementation,
        selected_strategy=selected_strategy,
        uses_stream_tokenizer=selected_strategy in {"stream", "tokenizer"},
        uses_pylibcudf=use_pylibcudf,
        uses_native_geometry_assembly=True,
        reason=reason,
    )


def _new_family_state() -> dict[str, object]:
    return {
        "row_count": 0,
        "empty_mask": [],
        "geometry_offsets": [],
        "x_payload": [],
        "y_payload": [],
        "part_offsets": [],
        "ring_offsets": [],
    }


def _append_coords(state: dict[str, object], coords) -> None:
    x_payload = state["x_payload"]
    y_payload = state["y_payload"]
    for x, y in coords:
        x_payload.append(float(x))
        y_payload.append(float(y))


def _append_geojson_geometry(
    geometry: dict[str, object] | None,
    *,
    row_index: int,
    states: dict[GeometryFamily, dict[str, object]],
    validity: list[bool],
    tags: list[int],
    family_row_offsets: list[int],
) -> None:
    if geometry is None:
        validity[row_index] = False
        return
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    family_map = {
        "Point": GeometryFamily.POINT,
        "LineString": GeometryFamily.LINESTRING,
        "Polygon": GeometryFamily.POLYGON,
        "MultiPoint": GeometryFamily.MULTIPOINT,
        "MultiLineString": GeometryFamily.MULTILINESTRING,
        "MultiPolygon": GeometryFamily.MULTIPOLYGON,
    }
    family = family_map.get(geom_type)
    if family is None:
        raise NotImplementedError(f"unsupported GeoJSON geometry type: {geom_type}")
    state = states[family]
    local_row = int(state["row_count"])
    state["row_count"] = local_row + 1
    family_row_offsets[row_index] = local_row
    tags[row_index] = FAMILY_TAGS[family]

    if family is GeometryFamily.POINT:
        empty = not coords
        state["empty_mask"].append(empty)
        state["geometry_offsets"].append(len(state["x_payload"]))
        if not empty:
            x, y = coords[:2]
            state["x_payload"].append(float(x))
            state["y_payload"].append(float(y))
        return

    if family in {GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        coord_list = coords or []
        state["empty_mask"].append(len(coord_list) == 0)
        state["geometry_offsets"].append(len(state["x_payload"]))
        _append_coords(state, coord_list)
        return

    if family is GeometryFamily.POLYGON:
        rings = coords or []
        state["empty_mask"].append(len(rings) == 0)
        state["geometry_offsets"].append(len(state["ring_offsets"]))
        for ring in rings:
            state["ring_offsets"].append(len(state["x_payload"]))
            _append_coords(state, ring)
        return

    if family is GeometryFamily.MULTILINESTRING:
        parts = coords or []
        state["empty_mask"].append(len(parts) == 0)
        state["geometry_offsets"].append(len(state["part_offsets"]))
        for part in parts:
            state["part_offsets"].append(len(state["x_payload"]))
            _append_coords(state, part)
        return

    if family is GeometryFamily.MULTIPOLYGON:
        polygons = coords or []
        state["empty_mask"].append(len(polygons) == 0)
        state["geometry_offsets"].append(len(state["part_offsets"]))
        for polygon in polygons:
            state["part_offsets"].append(len(state["ring_offsets"]))
            for ring in polygon:
                state["ring_offsets"].append(len(state["x_payload"]))
                _append_coords(state, ring)
        return


def _finalize_family_buffer(family: GeometryFamily, state: dict[str, object]) -> FamilyGeometryBuffer:
    x = np.asarray(state["x_payload"], dtype=np.float64)
    y = np.asarray(state["y_payload"], dtype=np.float64)
    geometry_offsets = np.asarray([*state["geometry_offsets"], len(state["x_payload"])], dtype=np.int32)
    part_offsets = None
    ring_offsets = None
    if family is GeometryFamily.POLYGON:
        ring_offsets = np.asarray([*state["ring_offsets"], len(state["x_payload"])], dtype=np.int32)
        geometry_offsets = np.asarray([*state["geometry_offsets"], len(state["ring_offsets"])], dtype=np.int32)
    elif family is GeometryFamily.MULTILINESTRING:
        part_offsets = np.asarray([*state["part_offsets"], len(state["x_payload"])], dtype=np.int32)
        geometry_offsets = np.asarray([*state["geometry_offsets"], len(state["part_offsets"])], dtype=np.int32)
    elif family is GeometryFamily.MULTIPOLYGON:
        part_offsets = np.asarray([*state["part_offsets"], len(state["ring_offsets"])], dtype=np.int32)
        ring_offsets = np.asarray([*state["ring_offsets"], len(state["x_payload"])], dtype=np.int32)
        geometry_offsets = np.asarray([*state["geometry_offsets"], len(state["part_offsets"])], dtype=np.int32)
    return FamilyGeometryBuffer(
        family=family,
        schema=get_geometry_buffer_schema(family),
        row_count=int(state["row_count"]),
        x=x,
        y=y,
        geometry_offsets=geometry_offsets,
        empty_mask=np.asarray(state["empty_mask"], dtype=bool),
        part_offsets=part_offsets,
        ring_offsets=ring_offsets,
    )


def _iter_feature_collection(text: str):
    decoder = json.JSONDecoder()
    features_key = '"features"'
    key_index = text.find(features_key)
    if key_index == -1:
        raise ValueError("GeoJSON FeatureCollection missing features array")
    index = text.find("[", key_index)
    if index == -1:
        raise ValueError("GeoJSON FeatureCollection missing '[' for features")
    index += 1
    length = len(text)
    while index < length:
        while index < length and text[index] in " \r\n\t,":
            index += 1
        if index >= length or text[index] == "]":
            break
        feature, index = decoder.raw_decode(text, index)
        yield feature


def _feature_collection_spans(text: str) -> list[tuple[int, int]]:
    """Find (start, end) byte spans of each Feature object inside a FeatureCollection.

    Uses ``json.JSONDecoder.raw_decode`` so that the heavy scanning is
    done by the C-level JSON parser rather than a Python character loop.
    """
    features_key = '"features"'
    key_index = text.find(features_key)
    if key_index == -1:
        raise ValueError("GeoJSON FeatureCollection missing features array")
    arr_start = text.find("[", key_index)
    if arr_start == -1:
        raise ValueError("GeoJSON FeatureCollection missing '[' for features")

    decoder = json.JSONDecoder()
    spans: list[tuple[int, int]] = []
    index = arr_start + 1
    length = len(text)

    while index < length:
        # Skip whitespace and commas between features
        while index < length and text[index] in " \t\n\r,":
            index += 1
        if index >= length or text[index] == "]":
            break
        start = index
        try:
            _, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"GeoJSON feature tokenizer failed at position {index}: {exc}"
            ) from exc
        spans.append((start, end))
        index = end

    return spans


def _has_pylibcudf_geojson_support() -> bool:
    return find_spec("pylibcudf") is not None and find_spec("pyarrow") is not None


def _require_pylibcudf_geojson_support() -> None:
    if not _has_pylibcudf_geojson_support():
        raise RuntimeError("pylibcudf + pyarrow are required for GeoJSON GPU ingest")


def _geojson_records_column(records: list[str]):
    import pyarrow as pa
    import pylibcudf as plc

    return plc.Column.from_arrow(pa.array(records))


def _geojson_json_path_column(records_column, path: str):
    import pylibcudf as plc

    return plc.json.get_json_object(records_column, plc.Scalar.from_py(path))


def _decode_geojson_property_json_values(values: list[str | None]) -> list[dict[str, object]]:
    properties: list[dict[str, object]] = []
    for value in values:
        if value in (None, "null", "{}"):
            properties.append({})
        else:
            parsed = json.loads(value)
            properties.append(parsed if isinstance(parsed, dict) else {})
    return properties


def _decode_geojson_properties_column(properties_column) -> list[dict[str, object]]:
    return _decode_geojson_property_json_values(properties_column.to_arrow().to_pylist())


def _load_geojson_properties_from_text(text: str) -> list[dict[str, object]]:
    payload = json.loads(text)
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection missing features list")
    return [dict(feature.get("properties") or {}) for feature in features]


def _geometry_family_from_geojson_name(name: str | None) -> GeometryFamily | None:
    mapping = {
        "Point": GeometryFamily.POINT,
        "LineString": GeometryFamily.LINESTRING,
        "Polygon": GeometryFamily.POLYGON,
        "MultiPoint": GeometryFamily.MULTIPOINT,
        "MultiLineString": GeometryFamily.MULTILINESTRING,
        "MultiPolygon": GeometryFamily.MULTIPOLYGON,
    }
    return mapping.get(name)


def _coordinate_xy_from_point_lists(point_array) -> tuple[np.ndarray, np.ndarray]:
    point_offsets = np.asarray(point_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    point_lengths = np.diff(point_offsets)
    if point_lengths.size and not np.all(point_lengths == 2):
        raise NotImplementedError("GeoJSON GPU ingest currently supports only 2D coordinates")
    scalars = np.asarray(point_array.values.to_numpy(zero_copy_only=False), dtype=np.float64)
    if scalars.size % 2 != 0:
        raise ValueError("GeoJSON coordinates must contain paired x/y values")
    return scalars[0::2], scalars[1::2]


def _point_offsets_from_scalar_offsets(offsets: np.ndarray) -> np.ndarray:
    lengths = np.diff(offsets)
    if lengths.size and not np.all(lengths % 2 == 0):
        raise ValueError("GeoJSON coordinate buffers must contain paired x/y values")
    point_offsets = np.empty(len(offsets), dtype=np.int32)
    point_offsets[0] = 0
    if lengths.size:
        point_offsets[1:] = np.cumsum(lengths // 2, dtype=np.int32)
    return point_offsets


def _decode_geojson_point_view(x_array, y_array, family: GeometryFamily) -> GeoArrowBufferView:
    import pyarrow as pa

    non_empty = np.asarray(x_array.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    geometry_offsets = np.empty(len(non_empty) + 1, dtype=np.int32)
    geometry_offsets[0] = 0
    if non_empty.size:
        geometry_offsets[1:] = np.cumsum(non_empty.astype(np.int32), dtype=np.int32)
    return GeoArrowBufferView(
        family=family,
        x=np.asarray(x_array.filter(pa.array(non_empty)).to_numpy(zero_copy_only=False), dtype=np.float64),
        y=np.asarray(y_array.filter(pa.array(non_empty)).to_numpy(zero_copy_only=False), dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=~non_empty,
    )


def _decode_geojson_point_geometry_view(coords_array, family: GeometryFamily) -> GeoArrowBufferView:
    lengths = np.diff(np.asarray(coords_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32))
    if lengths.size and not np.all((lengths == 0) | (lengths == 2)):
        raise NotImplementedError("GeoJSON GPU point ingest currently supports only 2D coordinates")
    scalars = np.asarray(coords_array.values.to_numpy(zero_copy_only=False), dtype=np.float64)
    if scalars.size % 2 != 0:
        raise ValueError("GeoJSON point coordinates must contain paired x/y values")
    geometry_offsets = np.empty(len(lengths) + 1, dtype=np.int32)
    geometry_offsets[0] = 0
    if lengths.size:
        geometry_offsets[1:] = np.cumsum(lengths // 2, dtype=np.int32)
    return GeoArrowBufferView(
        family=family,
        x=scalars[0::2],
        y=scalars[1::2],
        geometry_offsets=geometry_offsets,
        empty_mask=lengths == 0,
    )


def _decode_geojson_linestring_like_view(x_array, y_array, family: GeometryFamily) -> GeoArrowBufferView:
    geometry_offsets = np.asarray(x_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    return GeoArrowBufferView(
        family=family,
        x=np.asarray(x_array.values.to_numpy(zero_copy_only=False), dtype=np.float64),
        y=np.asarray(y_array.values.to_numpy(zero_copy_only=False), dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
    )


def _decode_geojson_linestring_geometry_view(coords_array, family: GeometryFamily) -> GeoArrowBufferView:
    geometry_offsets = np.asarray(coords_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    x, y = _coordinate_xy_from_point_lists(coords_array.values)
    return GeoArrowBufferView(
        family=family,
        x=x,
        y=y,
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
    )


def _decode_geojson_polygon_view(x_array, y_array, family: GeometryFamily) -> GeoArrowBufferView:
    geometry_offsets = np.asarray(x_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    ring_offsets = np.asarray(x_array.values.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    return GeoArrowBufferView(
        family=family,
        x=np.asarray(x_array.values.values.to_numpy(zero_copy_only=False), dtype=np.float64),
        y=np.asarray(y_array.values.values.to_numpy(zero_copy_only=False), dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
        ring_offsets=ring_offsets,
    )


def _decode_geojson_multilinestring_view(x_array, y_array, family: GeometryFamily) -> GeoArrowBufferView:
    geometry_offsets = np.asarray(x_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    part_offsets = np.asarray(x_array.values.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    return GeoArrowBufferView(
        family=family,
        x=np.asarray(x_array.values.values.to_numpy(zero_copy_only=False), dtype=np.float64),
        y=np.asarray(y_array.values.values.to_numpy(zero_copy_only=False), dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
        part_offsets=part_offsets,
    )


def _decode_geojson_multilinestring_geometry_view(coords_array, family: GeometryFamily) -> GeoArrowBufferView:
    geometry_offsets = np.asarray(coords_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    part_array = coords_array.values
    part_offsets = np.asarray(part_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    x, y = _coordinate_xy_from_point_lists(part_array.values)
    return GeoArrowBufferView(
        family=family,
        x=x,
        y=y,
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
        part_offsets=part_offsets,
    )


def _decode_geojson_polygon_geometry_view(coords_array, family: GeometryFamily) -> GeoArrowBufferView:
    geometry_offsets = np.asarray(coords_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    ring_array = coords_array.values
    ring_offsets = np.asarray(ring_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    x, y = _coordinate_xy_from_point_lists(ring_array.values)
    return GeoArrowBufferView(
        family=family,
        x=x,
        y=y,
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
        ring_offsets=ring_offsets,
    )


def _decode_geojson_multipolygon_view(x_array, y_array, family: GeometryFamily) -> GeoArrowBufferView:
    geometry_offsets = np.asarray(x_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    part_offsets = np.asarray(x_array.values.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    ring_offsets = np.asarray(x_array.values.values.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    return GeoArrowBufferView(
        family=family,
        x=np.asarray(x_array.values.values.values.to_numpy(zero_copy_only=False), dtype=np.float64),
        y=np.asarray(y_array.values.values.values.to_numpy(zero_copy_only=False), dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
        part_offsets=part_offsets,
        ring_offsets=ring_offsets,
    )


def _decode_geojson_multipolygon_geometry_view(coords_array, family: GeometryFamily) -> GeoArrowBufferView:
    geometry_offsets = np.asarray(coords_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    polygon_array = coords_array.values
    part_offsets = np.asarray(polygon_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    ring_array = polygon_array.values
    ring_offsets = np.asarray(ring_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    x, y = _coordinate_xy_from_point_lists(ring_array.values)
    return GeoArrowBufferView(
        family=family,
        x=x,
        y=y,
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
        part_offsets=part_offsets,
        ring_offsets=ring_offsets,
    )


def _decode_geojson_gpu_coordinates_view(family: GeometryFamily, coordinates_column) -> GeoArrowBufferView:
    import pylibcudf as plc

    geometry_table = plc.io.json.read_json_from_string_column(
        coordinates_column,
        plc.Scalar.from_py("\n"),
        plc.Scalar.from_py("null"),
    ).tbl.to_arrow()
    x_array = geometry_table.column(0).combine_chunks()
    y_array = geometry_table.column(1).combine_chunks()
    if family is GeometryFamily.POINT:
        return _decode_geojson_point_view(x_array, y_array, family)
    if family in {GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        return _decode_geojson_linestring_like_view(x_array, y_array, family)
    if family is GeometryFamily.POLYGON:
        return _decode_geojson_polygon_view(x_array, y_array, family)
    if family is GeometryFamily.MULTILINESTRING:
        return _decode_geojson_multilinestring_view(x_array, y_array, family)
    if family is GeometryFamily.MULTIPOLYGON:
        return _decode_geojson_multipolygon_view(x_array, y_array, family)
    raise NotImplementedError(f"unsupported GeoJSON GPU family: {family.value}")


def _decode_geojson_gpu_geometry_view(family: GeometryFamily, geometry_column) -> GeoArrowBufferView:
    import pylibcudf as plc

    geometry_table = plc.io.json.read_json_from_string_column(
        geometry_column,
        plc.Scalar.from_py("\n"),
        plc.Scalar.from_py("null"),
    ).tbl.to_arrow()
    coords_array = geometry_table.column(1).combine_chunks()
    if family is GeometryFamily.POLYGON:
        return _decode_geojson_polygon_geometry_view(coords_array, family)
    if family is GeometryFamily.MULTIPOLYGON:
        return _decode_geojson_multipolygon_geometry_view(coords_array, family)
    raise NotImplementedError(f"unsupported GeoJSON geometry-object family: {family.value}")


def _decode_geojson_coordinate_array_view(family: GeometryFamily, coords_array) -> GeoArrowBufferView:
    if family is GeometryFamily.POINT:
        return _decode_geojson_point_geometry_view(coords_array, family)
    if family in {GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        return _decode_geojson_linestring_geometry_view(coords_array, family)
    if family is GeometryFamily.POLYGON:
        return _decode_geojson_polygon_geometry_view(coords_array, family)
    if family is GeometryFamily.MULTILINESTRING:
        return _decode_geojson_multilinestring_geometry_view(coords_array, family)
    if family is GeometryFamily.MULTIPOLYGON:
        return _decode_geojson_multipolygon_geometry_view(coords_array, family)
    raise NotImplementedError(f"unsupported GeoJSON coordinate-array family: {family.value}")


def _read_geojson_owned_pylibcudf_arrays(text: str) -> GeoJSONOwnedBatch | None:
    import pyarrow as pa
    import pylibcudf as plc

    feature_collection = plc.Column.from_arrow(pa.array([text]))
    type_array_json = plc.json.get_json_object(
        feature_collection,
        plc.Scalar.from_py("$.features[*].geometry.type"),
    )
    type_table = plc.io.json.read_json_from_string_column(
        type_array_json,
        plc.Scalar.from_py("\n"),
        plc.Scalar.from_py("null"),
    ).tbl
    geometry_type_array = plc.concatenate.concatenate(type_table.columns()).to_arrow()
    geometry_types = geometry_type_array.to_pylist()
    families = {_geometry_family_from_geojson_name(name) for name in geometry_types if name is not None}
    families.discard(None)
    if len(families) != 1:
        return None
    family = next(iter(families))

    coordinates_array_json = plc.json.get_json_object(
        feature_collection,
        plc.Scalar.from_py("$.features[*].geometry.coordinates"),
    )
    coordinate_table = plc.io.json.read_json_from_string_column(
        coordinates_array_json,
        plc.Scalar.from_py("\n"),
        plc.Scalar.from_py("null"),
    ).tbl
    coordinate_array = plc.concatenate.concatenate(coordinate_table.columns()).to_arrow()
    validity = np.asarray(geometry_type_array.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    tags = np.full(len(geometry_types), -1, dtype=np.int8)
    family_row_offsets = np.full(len(geometry_types), -1, dtype=np.int32)
    tags[validity] = FAMILY_TAGS[family]
    family_row_offsets[validity] = np.arange(int(validity.sum()), dtype=np.int32)

    geometry = from_geoarrow(
        MixedGeoArrowView(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families={family: _decode_geojson_coordinate_array_view(family, coordinate_array)},
            shares_memory=False,
        )
    )
    return GeoJSONOwnedBatch(
        geometry=geometry,
        _properties_loader=lambda text=text: _load_geojson_properties_from_text(text),
    )


def _select_feature_geometry_child(feature_struct) -> tuple[int, object] | None:
    geometry_names = {family.value.title().replace("string", "String") for family in GeometryFamily}
    geometry_names.update(
        {"Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon"}
    )
    for index in range(feature_struct.type.num_fields):
        child = feature_struct.field(index)
        if child.type.num_fields < 2:
            continue
        maybe_type = child.field(0)
        if not str(maybe_type.type).startswith("string"):
            continue
        values = [value for value in maybe_type.to_pylist() if value is not None]
        if values and set(values).issubset(geometry_names):
            return index, child
    return None


def _decode_geojson_feature_geometry_struct(
    family: GeometryFamily,
    geometry_struct,
) -> GeoArrowBufferView:
    coords_array = geometry_struct.field(1)
    if family is GeometryFamily.POINT:
        return _decode_geojson_point_geometry_view(coords_array, family)
    if family in {GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        return _decode_geojson_linestring_geometry_view(coords_array, family)
    if family is GeometryFamily.POLYGON:
        return _decode_geojson_polygon_geometry_view(coords_array, family)
    if family is GeometryFamily.MULTILINESTRING:
        return _decode_geojson_multilinestring_geometry_view(coords_array, family)
    if family is GeometryFamily.MULTIPOLYGON:
        return _decode_geojson_multipolygon_geometry_view(coords_array, family)
    raise NotImplementedError(f"unsupported GeoJSON rowized family: {family.value}")


def _read_geojson_owned_pylibcudf_rowized(text: str) -> GeoJSONOwnedBatch | None:
    import pyarrow as pa
    import pylibcudf as plc

    feature_collection = plc.Column.from_arrow(pa.array([text]))
    features_array = plc.json.get_json_object(feature_collection, plc.Scalar.from_py("$.features"))
    try:
        parsed = plc.io.json.read_json_from_string_column(
            features_array,
            plc.Scalar.from_py("\n"),
            plc.Scalar.from_py("null"),
        ).tbl
        rowized = plc.reshape.interleave_columns(parsed).to_arrow()
    except RuntimeError:
        return None

    if rowized.null_count:
        return None
    geometry_child = _select_feature_geometry_child(rowized)
    if geometry_child is None:
        return None
    _, geometry_struct = geometry_child
    geometry_types = geometry_struct.field(0).to_pylist()
    families = {
        _geometry_family_from_geojson_name(name)
        for name in geometry_types
        if name is not None
    }
    families.discard(None)
    if len(families) != 1:
        return None
    family = next(iter(families))
    validity = np.asarray(geometry_struct.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    tags = np.full(len(geometry_types), -1, dtype=np.int8)
    family_row_offsets = np.full(len(geometry_types), -1, dtype=np.int32)
    tags[validity] = FAMILY_TAGS[family]
    family_row_offsets[validity] = np.arange(int(validity.sum()), dtype=np.int32)

    geometry = from_geoarrow(
        MixedGeoArrowView(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families={family: _decode_geojson_feature_geometry_struct(family, geometry_struct)},
            shares_memory=False,
        )
    )
    return GeoJSONOwnedBatch(
        geometry=geometry,
        _properties_loader=lambda text=text: _load_geojson_properties_from_text(text),
    )


def _read_geojson_owned_pylibcudf(text: str) -> GeoJSONOwnedBatch:
    _require_pylibcudf_geojson_support()
    spans = _feature_collection_spans(text)
    records = [text[start:end] for start, end in spans]
    records_column = _geojson_records_column(records)
    geometry_type_column = _geojson_json_path_column(records_column, "$.geometry.type")
    properties_column = _geojson_json_path_column(records_column, "$.properties")
    geometry_types = geometry_type_column.to_arrow().to_pylist()
    properties_json = properties_column.to_arrow().to_pylist()

    row_count = len(records)
    validity = np.zeros(row_count, dtype=bool)
    tags = np.full(row_count, -1, dtype=np.int8)
    family_row_offsets = np.full(row_count, -1, dtype=np.int32)
    family_rows: dict[GeometryFamily, list[int]] = {family: [] for family in GeometryFamily}

    for row_index, name in enumerate(geometry_types):
        family = _geometry_family_from_geojson_name(name)
        if family is None:
            if name is None:
                continue
            raise NotImplementedError(f"unsupported GeoJSON geometry type: {name}")
        validity[row_index] = True
        family_rows[family].append(row_index)

    families: dict[GeometryFamily, GeoArrowBufferView] = {}
    for family, row_indexes in family_rows.items():
        if not row_indexes:
            continue
        family_records = [records[row_index] for row_index in row_indexes]
        family_records_column = _geojson_records_column(family_records)
        if family in {
            GeometryFamily.POINT,
            GeometryFamily.LINESTRING,
            GeometryFamily.MULTIPOINT,
            GeometryFamily.MULTILINESTRING,
        }:
            coordinates_column = _geojson_json_path_column(
                family_records_column,
                "$.geometry.coordinates",
            )
            families[family] = _decode_geojson_gpu_coordinates_view(family, coordinates_column)
        else:
            geometry_column = _geojson_json_path_column(
                family_records_column,
                "$.geometry",
            )
            families[family] = _decode_geojson_gpu_geometry_view(family, geometry_column)
        row_array = np.asarray(row_indexes, dtype=np.int32)
        tags[row_array] = FAMILY_TAGS[family]
        family_row_offsets[row_array] = np.arange(len(row_indexes), dtype=np.int32)

    geometry = from_geoarrow(
        MixedGeoArrowView(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families=families,
            shares_memory=False,
        )
    )
    return GeoJSONOwnedBatch(geometry=geometry, _properties_json=properties_json)


def _owned_batch_from_features(features: list[dict[str, object]]) -> GeoJSONOwnedBatch:
    properties: list[dict[str, object]] = []
    validity_list: list[bool] = []
    tags_list: list[int] = []
    family_row_offsets_list: list[int] = []
    states = {family: _new_family_state() for family in GeometryFamily}
    for row_index, feature in enumerate(features):
        properties.append(dict(feature.get("properties") or {}))
        validity_list.append(True)
        tags_list.append(-1)
        family_row_offsets_list.append(-1)
        _append_geojson_geometry(
            feature.get("geometry"),
            row_index=row_index,
            states=states,
            validity=validity_list,
            tags=tags_list,
            family_row_offsets=family_row_offsets_list,
        )
    families = {
        family: _finalize_family_buffer(family, state)
        for family, state in states.items()
        if int(state["row_count"]) > 0
    }
    owned = OwnedGeometryArray(
        validity=np.asarray(validity_list, dtype=bool),
        tags=np.asarray(tags_list, dtype=np.int8),
        family_row_offsets=np.asarray(family_row_offsets_list, dtype=np.int32),
        families=families,
    )
    return GeoJSONOwnedBatch(geometry=owned, _properties=properties)


# ---------------------------------------------------------------------------
# Vectorized per-family coordinate extraction (fast-json / chunked paths)
# ---------------------------------------------------------------------------

_GEOJSON_FAMILY_MAP: dict[str, GeometryFamily] = {
    "Point": GeometryFamily.POINT,
    "LineString": GeometryFamily.LINESTRING,
    "Polygon": GeometryFamily.POLYGON,
    "MultiPoint": GeometryFamily.MULTIPOINT,
    "MultiLineString": GeometryFamily.MULTILINESTRING,
    "MultiPolygon": GeometryFamily.MULTIPOLYGON,
}


def _extract_point_buffers(
    coords_list: list[list[float]],
) -> FamilyGeometryBuffer:
    """Vectorized point coordinate extraction into owned buffer."""
    n = len(coords_list)
    if n == 0:
        return FamilyGeometryBuffer(
            family=GeometryFamily.POINT,
            schema=get_geometry_buffer_schema(GeometryFamily.POINT),
            row_count=0,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=np.zeros(1, dtype=np.int32),
            empty_mask=np.empty(0, dtype=bool),
        )
    arr = np.empty((n, 2), dtype=np.float64)
    empty_flags = np.zeros(n, dtype=bool)
    valid_count = 0
    for i, c in enumerate(coords_list):
        if c:
            arr[valid_count, 0] = float(c[0])
            arr[valid_count, 1] = float(c[1])
            valid_count += 1
        else:
            empty_flags[i] = True
    x = arr[:valid_count, 0].copy()
    y = arr[:valid_count, 1].copy()
    offsets = np.empty(n + 1, dtype=np.int32)
    offsets[0] = 0
    if n > 0:
        offsets[1:] = np.cumsum(~empty_flags, dtype=np.int32)
    return FamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        schema=get_geometry_buffer_schema(GeometryFamily.POINT),
        row_count=n,
        x=x,
        y=y,
        geometry_offsets=offsets,
        empty_mask=empty_flags,
    )


def _extract_linestring_buffers(
    coords_list: list[list[list[float]]],
    family: GeometryFamily,
) -> FamilyGeometryBuffer:
    """Vectorized linestring / multipoint extraction."""
    n = len(coords_list)
    all_x: list[float] = []
    all_y: list[float] = []
    offsets = [0]
    empty_flags: list[bool] = []
    for ring in coords_list:
        if not ring:
            empty_flags.append(True)
            offsets.append(len(all_x))
            continue
        empty_flags.append(False)
        for pt in ring:
            all_x.append(float(pt[0]))
            all_y.append(float(pt[1]))
        offsets.append(len(all_x))
    return FamilyGeometryBuffer(
        family=family,
        schema=get_geometry_buffer_schema(family),
        row_count=n,
        x=np.array(all_x, dtype=np.float64),
        y=np.array(all_y, dtype=np.float64),
        geometry_offsets=np.array(offsets, dtype=np.int32),
        empty_mask=np.array(empty_flags, dtype=bool),
    )


def _extract_polygon_buffers(
    coords_list: list[list[list[list[float]]]],
) -> FamilyGeometryBuffer:
    """Vectorized polygon extraction preserving ring structure."""
    n = len(coords_list)
    all_x: list[float] = []
    all_y: list[float] = []
    ring_offsets = [0]
    geom_offsets = [0]
    empty_flags: list[bool] = []
    for rings in coords_list:
        if not rings:
            empty_flags.append(True)
            geom_offsets.append(len(ring_offsets) - 1)
            continue
        empty_flags.append(False)
        for ring in rings:
            for pt in ring:
                all_x.append(float(pt[0]))
                all_y.append(float(pt[1]))
            ring_offsets.append(len(all_x))
        geom_offsets.append(len(ring_offsets) - 1)
    return FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=n,
        x=np.array(all_x, dtype=np.float64),
        y=np.array(all_y, dtype=np.float64),
        geometry_offsets=np.array(geom_offsets, dtype=np.int32),
        empty_mask=np.array(empty_flags, dtype=bool),
        ring_offsets=np.array(ring_offsets, dtype=np.int32),
    )


def _extract_multilinestring_buffers(
    coords_list: list[list[list[list[float]]]],
) -> FamilyGeometryBuffer:
    """Vectorized multilinestring extraction."""
    n = len(coords_list)
    all_x: list[float] = []
    all_y: list[float] = []
    part_offsets = [0]
    geom_offsets = [0]
    empty_flags: list[bool] = []
    for parts in coords_list:
        if not parts:
            empty_flags.append(True)
            geom_offsets.append(len(part_offsets) - 1)
            continue
        empty_flags.append(False)
        for part in parts:
            for pt in part:
                all_x.append(float(pt[0]))
                all_y.append(float(pt[1]))
            part_offsets.append(len(all_x))
        geom_offsets.append(len(part_offsets) - 1)
    return FamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        schema=get_geometry_buffer_schema(GeometryFamily.MULTILINESTRING),
        row_count=n,
        x=np.array(all_x, dtype=np.float64),
        y=np.array(all_y, dtype=np.float64),
        geometry_offsets=np.array(geom_offsets, dtype=np.int32),
        empty_mask=np.array(empty_flags, dtype=bool),
        part_offsets=np.array(part_offsets, dtype=np.int32),
    )


def _extract_multipolygon_buffers(
    coords_list: list[list[list[list[list[float]]]]],
) -> FamilyGeometryBuffer:
    """Vectorized multipolygon extraction."""
    n = len(coords_list)
    all_x: list[float] = []
    all_y: list[float] = []
    ring_offsets = [0]
    part_offsets = [0]
    geom_offsets = [0]
    empty_flags: list[bool] = []
    for polygons in coords_list:
        if not polygons:
            empty_flags.append(True)
            geom_offsets.append(len(part_offsets) - 1)
            continue
        empty_flags.append(False)
        for rings in polygons:
            for ring in rings:
                for pt in ring:
                    all_x.append(float(pt[0]))
                    all_y.append(float(pt[1]))
                ring_offsets.append(len(all_x))
            part_offsets.append(len(ring_offsets) - 1)
        geom_offsets.append(len(part_offsets) - 1)
    return FamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.MULTIPOLYGON),
        row_count=n,
        x=np.array(all_x, dtype=np.float64),
        y=np.array(all_y, dtype=np.float64),
        geometry_offsets=np.array(geom_offsets, dtype=np.int32),
        empty_mask=np.array(empty_flags, dtype=bool),
        part_offsets=np.array(part_offsets, dtype=np.int32),
        ring_offsets=np.array(ring_offsets, dtype=np.int32),
    )


_FAMILY_BUFFER_EXTRACTORS = {
    GeometryFamily.POINT: lambda coords: _extract_point_buffers(coords),
    GeometryFamily.LINESTRING: lambda coords: _extract_linestring_buffers(
        coords, GeometryFamily.LINESTRING
    ),
    GeometryFamily.MULTIPOINT: lambda coords: _extract_linestring_buffers(
        coords, GeometryFamily.MULTIPOINT
    ),
    GeometryFamily.POLYGON: lambda coords: _extract_polygon_buffers(coords),
    GeometryFamily.MULTILINESTRING: lambda coords: _extract_multilinestring_buffers(coords),
    GeometryFamily.MULTIPOLYGON: lambda coords: _extract_multipolygon_buffers(coords),
}


def _owned_batch_from_parsed_features(
    features: list[dict[str, object]],
) -> GeoJSONOwnedBatch:
    """Build an OwnedGeometryArray from already-parsed feature dicts.

    Uses vectorized per-family coordinate extraction instead of the
    per-feature ``_append_geojson_geometry`` loop. At 100K point features
    this is ~3x faster than the old per-element assembly path.
    """
    n = len(features)
    validity = np.ones(n, dtype=bool)
    tags = np.full(n, -1, dtype=np.int8)
    family_row_offsets = np.full(n, -1, dtype=np.int32)

    # Classify features and collect per-family coordinate lists
    family_indices: dict[GeometryFamily, list[int]] = {f: [] for f in GeometryFamily}
    family_coords: dict[GeometryFamily, list[object]] = {f: [] for f in GeometryFamily}

    for i, feature in enumerate(features):
        geom = feature.get("geometry") if isinstance(feature, dict) else None
        if geom is None:
            validity[i] = False
            continue
        gtype = geom.get("type") if isinstance(geom, dict) else None
        family = _GEOJSON_FAMILY_MAP.get(gtype) if gtype else None
        if family is None:
            if gtype is None:
                validity[i] = False
                continue
            raise NotImplementedError(f"unsupported GeoJSON geometry type: {gtype}")
        family_indices[family].append(i)
        family_coords[family].append(geom.get("coordinates"))

    # Vectorized per-family buffer extraction
    families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    for family in GeometryFamily:
        indices = family_indices[family]
        if not indices:
            continue
        idx_arr = np.array(indices, dtype=np.int32)
        tags[idx_arr] = FAMILY_TAGS[family]
        family_row_offsets[idx_arr] = np.arange(len(indices), dtype=np.int32)
        families[family] = _FAMILY_BUFFER_EXTRACTORS[family](family_coords[family])

    owned = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
    )
    return GeoJSONOwnedBatch(
        geometry=owned,
        _properties_loader=lambda features=features: [
            dict(f.get("properties") or {}) if isinstance(f, dict) else {} for f in features
        ],
    )


def _read_geojson_owned_fast_json(text: str) -> GeoJSONOwnedBatch:
    """Fast GeoJSON ingest: orjson parse + vectorized coordinate extraction.

    Uses orjson (when available) for ~1.5x faster JSON parsing, combined with
    vectorized per-family coordinate extraction that avoids the per-feature
    ``_append_geojson_geometry`` loop. Together these give ~3-4x speedup over
    the previous ``full-json`` path at 1M features.
    """
    payload = _fast_json_loads(text)
    if isinstance(payload, dict) and payload.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON ingest currently expects a FeatureCollection root")
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection missing features list")
    return _owned_batch_from_parsed_features(features)


def _read_geojson_owned_simdjson(data: str | bytes) -> GeoJSONOwnedBatch:
    """GeoJSON ingest using simdjson + vectorized extraction.

    Uses pysimdjson for parsing and the same vectorized per-family coordinate
    extraction pipeline as the fast-json path.  The parser is created per-call
    for thread safety.
    """
    payload = _simdjson_loads(data)
    if isinstance(payload, dict) and payload.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON ingest currently expects a FeatureCollection root")
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection missing features list")
    return _owned_batch_from_parsed_features(features)


# ---------------------------------------------------------------------------
# Chunked GeoJSON ingest: split features array by byte range, parse chunks
# ---------------------------------------------------------------------------

def _find_feature_chunk_boundaries(
    text_bytes: bytes,
    n_chunks: int,
) -> list[int]:
    """Find byte positions that split the features array into *n_chunks* chunks.

    Uses ``bytes.find`` on the ``}, {`` inter-feature boundary pattern to
    locate split points near evenly-spaced byte offsets. Returns a list of
    ``n_chunks + 1`` byte positions.
    """
    features_marker = b'"features"'
    mpos = text_bytes.find(features_marker)
    if mpos == -1:
        raise ValueError("GeoJSON FeatureCollection missing features array")
    array_start = text_bytes.find(b"[", mpos)
    if array_start == -1:
        raise ValueError("GeoJSON FeatureCollection missing '[' for features")
    array_start += 1
    array_end = text_bytes.rfind(b"]")
    if array_end == -1:
        array_end = len(text_bytes)

    data_size = array_end - array_start
    if data_size <= 0 or n_chunks <= 1:
        return [array_start, array_end]

    chunk_size = data_size // n_chunks
    boundaries: list[int] = [array_start]

    for i in range(1, n_chunks):
        target = array_start + i * chunk_size
        # Fast C-level search for the inter-feature boundary pattern
        pos = text_bytes.find(b"}, {", target)
        if pos == -1:
            pos = text_bytes.find(b"},\n{", target)
        if pos == -1:
            pos = text_bytes.find(b"},\r\n{", target)
        if pos != -1 and pos < array_end:
            # Skip past '}, ' to point at the next feature's '{'
            sep = text_bytes[pos + 1 : pos + 4]
            skip = 3 if sep.startswith(b" ") or sep.startswith(b",") else 2
            boundary = pos + skip
            # Skip any remaining whitespace/commas
            while boundary < array_end and text_bytes[boundary : boundary + 1] in (
                b" ",
                b"\t",
                b"\n",
                b"\r",
                b",",
            ):
                boundary += 1
            boundaries.append(boundary)

    boundaries.append(array_end)
    return boundaries


def _read_geojson_owned_chunked(
    text: str,
    *,
    n_chunks: int = 4,
) -> GeoJSONOwnedBatch:
    """Chunked GeoJSON ingest: split features array, parse chunks, merge.

    Splits the FeatureCollection's features array into *n_chunks* byte-range
    segments. Each chunk is parsed independently with orjson, then coordinates
    are extracted per-chunk and the results merged. Reduces peak memory
    compared to a single monolithic parse and provides a seam for future
    parallel-stream GPU dispatch.
    """
    text_bytes = text.encode("utf-8") if isinstance(text, str) else text
    boundaries = _find_feature_chunk_boundaries(text_bytes, n_chunks)

    all_features: list[dict[str, object]] = []
    for i in range(len(boundaries) - 1):
        chunk = text_bytes[boundaries[i] : boundaries[i + 1]]
        # Strip trailing commas/whitespace and wrap as a valid JSON array
        stripped = chunk.rstrip(b", \t\n\r")
        buf = b"[" + stripped + b"]"
        try:
            chunk_features = _fast_json_loads(buf)
        except (json.JSONDecodeError, Exception):
            # Fallback: parse the entire text monolithically
            return _read_geojson_owned_fast_json(text)
        if isinstance(chunk_features, list):
            all_features.extend(chunk_features)

    return _owned_batch_from_parsed_features(all_features)


def _read_geojson_owned_full_json(text: str) -> GeoJSONOwnedBatch:
    payload = json.loads(text)
    if payload.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON ingest currently expects a FeatureCollection root")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection missing features list")
    return _owned_batch_from_features(features)


def _read_geojson_owned_stream(text: str) -> GeoJSONOwnedBatch:
    return _owned_batch_from_features(list(_iter_feature_collection(text)))


def _read_geojson_owned_tokenizer(text: str) -> GeoJSONOwnedBatch:
    spans = _feature_collection_spans(text)
    features = [json.loads(text[start:end]) for start, end in spans]
    return _owned_batch_from_features(features)


def read_geojson_owned(source: str | Path, *, prefer: str = "auto") -> GeoJSONOwnedBatch:
    plan = plan_geojson_ingest(prefer=prefer)
    if plan.selected_strategy == "gpu-byte-classify":
        from .geojson_gpu import read_geojson_gpu
        record_dispatch_event(
            surface="vibespatial.io.geojson",
            operation="read_owned",
            implementation=plan.implementation,
            reason=plan.reason,
            selected=ExecutionMode.GPU,
        )
        resolved = Path(source) if not isinstance(source, Path) else source
        result = read_geojson_gpu(resolved)
        return GeoJSONOwnedBatch(
            geometry=result.owned,
            _properties_loader=result.properties_loader(),
        )
    # For the simdjson strategy, prefer reading as bytes to avoid str->bytes encode overhead.
    if plan.selected_strategy == "simdjson":
        if isinstance(source, Path):
            data: str | bytes = source.read_bytes()
        elif isinstance(source, str) and (
            source.lstrip().startswith("{") or source.lstrip().startswith("[")
        ):
            data = source
        else:
            data = Path(str(source)).read_bytes()
    else:
        if isinstance(source, Path):
            data = source.read_text(encoding="utf-8")
        elif isinstance(source, str) and (
            source.lstrip().startswith("{") or source.lstrip().startswith("[")
        ):
            data = source
        else:
            data = Path(str(source)).read_text(encoding="utf-8")
    text: str  # type alias for non-simdjson paths that expect str
    record_dispatch_event(
        surface="vibespatial.io.geojson",
        operation="read_owned",
        implementation=plan.implementation,
        reason=plan.reason,
        selected=ExecutionMode.CPU,
    )
    if plan.selected_strategy == "simdjson":
        return _read_geojson_owned_simdjson(data)
    text = data if isinstance(data, str) else data.decode("utf-8")
    if plan.selected_strategy == "fast-json":
        return _read_geojson_owned_fast_json(text)
    if plan.selected_strategy == "chunked":
        return _read_geojson_owned_chunked(text)
    if plan.selected_strategy == "full-json":
        return _read_geojson_owned_full_json(text)
    if plan.selected_strategy == "pylibcudf":
        return _read_geojson_owned_pylibcudf(text)
    if plan.selected_strategy == "pylibcudf-arrays":
        array_batch = _read_geojson_owned_pylibcudf_arrays(text)
        if array_batch is None:
            raise ValueError(
                "experimental GeoJSON wildcard-array GPU path requires a homogeneous geometry family"
            )
        return array_batch
    if plan.selected_strategy == "pylibcudf-rowized":
        rowized = _read_geojson_owned_pylibcudf_rowized(text)
        if rowized is None:
            raise ValueError(
                "experimental GeoJSON rowized GPU path requires a homogeneous feature schema"
            )
        return rowized
    if plan.selected_strategy == "tokenizer":
        return _read_geojson_owned_tokenizer(text)
    return _read_geojson_owned_stream(text)


def benchmark_geojson_ingest(
    *,
    geometry_type: str = "point",
    rows: int = 100_000,
    repeat: int = 5,
    seed: int = 0,
) -> list[GeoJSONIngestBenchmark]:
    import pyogrio

    if geometry_type == "point":
        from vibespatial.testing.synthetic import SyntheticSpec, generate_points

        dataset = generate_points(SyntheticSpec("point", "uniform", count=rows, seed=seed))
    elif geometry_type == "polygon":
        from vibespatial.testing.synthetic import SyntheticSpec, generate_polygons

        dataset = generate_polygons(
            SyntheticSpec("polygon", "regular-grid", count=rows, seed=seed, vertices=6)
        )
    else:
        raise ValueError(f"Unsupported geometry_type: {geometry_type}")

    gdf = dataset.to_geodataframe()
    text = gdf.to_json()
    import tempfile

    results: list[GeoJSONIngestBenchmark] = []
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sample.geojson"
        path.write_text(text, encoding="utf-8")

        start = perf_counter()
        for _ in range(repeat):
            pyogrio.read_dataframe(path)
        elapsed = (perf_counter() - start) / repeat
        results.append(
            GeoJSONIngestBenchmark(
                implementation="pyogrio_host",
                geometry_type=geometry_type,
                rows=rows,
                elapsed_seconds=elapsed,
                rows_per_second=rows / elapsed if elapsed else float("inf"),
            )
        )

        start = perf_counter()
        for _ in range(repeat):
            payload = json.loads(text)
            features = payload["features"]
            values = [feature.get("geometry") for feature in features]
            validity = np.ones(len(values), dtype=bool)
            tags = np.full(len(values), -1, dtype=np.int8)
            family_row_offsets = np.full(len(values), -1, dtype=np.int32)
            states = {family: _new_family_state() for family in GeometryFamily}
            for row_index, geometry in enumerate(values):
                _append_geojson_geometry(
                    geometry,
                    row_index=row_index,
                    states=states,
                    validity=validity,
                    tags=tags,
                    family_row_offsets=family_row_offsets,
                )
            _ = OwnedGeometryArray(
                validity=validity,
                tags=tags,
                family_row_offsets=family_row_offsets,
                families={
                    family: _finalize_family_buffer(family, state)
                    for family, state in states.items()
                    if int(state["row_count"]) > 0
                },
            )
        elapsed = (perf_counter() - start) / repeat
        results.append(
            GeoJSONIngestBenchmark(
                implementation="full_json_baseline",
                geometry_type=geometry_type,
                rows=rows,
                elapsed_seconds=elapsed,
                rows_per_second=rows / elapsed if elapsed else float("inf"),
            )
        )

        start = perf_counter()
        for _ in range(repeat):
            read_geojson_owned(text, prefer="stream")
        elapsed = (perf_counter() - start) / repeat
        results.append(
            GeoJSONIngestBenchmark(
                implementation="stream_native",
                geometry_type=geometry_type,
                rows=rows,
                elapsed_seconds=elapsed,
                rows_per_second=rows / elapsed if elapsed else float("inf"),
            )
        )

        start = perf_counter()
        for _ in range(repeat):
            read_geojson_owned(text, prefer="tokenizer")
        elapsed = (perf_counter() - start) / repeat
        results.append(
            GeoJSONIngestBenchmark(
                implementation="tokenizer_native",
                geometry_type=geometry_type,
                rows=rows,
                elapsed_seconds=elapsed,
                rows_per_second=rows / elapsed if elapsed else float("inf"),
            )
        )

        start = perf_counter()
        for _ in range(repeat):
            read_geojson_owned(text, prefer="fast-json")
        elapsed = (perf_counter() - start) / repeat
        results.append(
            GeoJSONIngestBenchmark(
                implementation="fast_json_vectorized",
                geometry_type=geometry_type,
                rows=rows,
                elapsed_seconds=elapsed,
                rows_per_second=rows / elapsed if elapsed else float("inf"),
            )
        )

        start = perf_counter()
        for _ in range(repeat):
            read_geojson_owned(text, prefer="chunked")
        elapsed = (perf_counter() - start) / repeat
        results.append(
            GeoJSONIngestBenchmark(
                implementation="chunked_vectorized",
                geometry_type=geometry_type,
                rows=rows,
                elapsed_seconds=elapsed,
                rows_per_second=rows / elapsed if elapsed else float("inf"),
            )
        )

        if _HAS_SIMDJSON:
            start = perf_counter()
            for _ in range(repeat):
                read_geojson_owned(text, prefer="simdjson")
            elapsed = (perf_counter() - start) / repeat
            results.append(
                GeoJSONIngestBenchmark(
                    implementation="simdjson_vectorized",
                    geometry_type=geometry_type,
                    rows=rows,
                    elapsed_seconds=elapsed,
                    rows_per_second=rows / elapsed if elapsed else float("inf"),
                )
            )

        if _has_pylibcudf_geojson_support():
            start = perf_counter()
            for _ in range(repeat):
                read_geojson_owned(text, prefer="pylibcudf")
            elapsed = (perf_counter() - start) / repeat
            results.append(
                GeoJSONIngestBenchmark(
                    implementation="pylibcudf_native",
                    geometry_type=geometry_type,
                    rows=rows,
                    elapsed_seconds=elapsed,
                    rows_per_second=rows / elapsed if elapsed else float("inf"),
                )
            )
            start = perf_counter()
            for _ in range(repeat):
                read_geojson_owned(text, prefer="pylibcudf-arrays")
            elapsed = (perf_counter() - start) / repeat
            results.append(
                GeoJSONIngestBenchmark(
                    implementation="pylibcudf_arrays_experimental",
                    geometry_type=geometry_type,
                    rows=rows,
                    elapsed_seconds=elapsed,
                    rows_per_second=rows / elapsed if elapsed else float("inf"),
                )
            )
    return results
