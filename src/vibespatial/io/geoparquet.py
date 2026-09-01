from __future__ import annotations

import hashlib
import inspect
import io
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import lru_cache
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from vibespatial.api._native_metadata import NativeGeometryMetadata
from vibespatial.api._native_results import (
    GeometryNativeResult,
    NativeAttributeTable,
    NativeGeometryColumn,
    NativeReadProvenance,
    NativeTabularResult,
    NativeTabularSelection,
    _concat_native_tabular_results,
    native_attribute_table_from_arrow_table,
    to_native_tabular_result,
)
from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import (
    DeviceRegularGridRectMetadata,
    OwnedGeometryArray,
    concatenate_owned_arrays,
)
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.residency import Residency, TransferTrigger

from .geoarrow import (
    _authoritative_geoarrow_host_view,
    _decode_geoarrow_array_to_owned,
    _GeoArrowNativeCompatibilityRoute,
    _owned_geoarrow_fast_path_reason,
    encode_owned_geoarrow_array,
    native_tabular_to_arrow,
)
from .geoparquet_planner import (
    GeoParquetMetadataSummary,
    build_geoparquet_metadata_summary,
    select_row_groups,
)
from .pylibcudf import (
    _decode_pylibcudf_geoparquet_column_to_owned,
    _is_pylibcudf_table,
    _pylibcudf_validity_mask,
)
from .support import IOFormat, IOOperation, IOPathKind, plan_io_support
from .wkb import (
    _apply_arrow_nested_child_metadata,
    _compression_type_from_name,
    _encode_owned_wkb_array,
    _pylibcudf_sink,
    _write_geoparquet_native_device,
    _write_geoparquet_native_device_payload,
    _write_pylibcudf_parquet_table,
    has_pyarrow_support,
    has_pylibcudf_support,
)

_SMALL_TERMINAL_ARROW_EXPORT_MAX_ROWS = 2_048
_SMALL_TERMINAL_ARROW_EXPORT_FAMILIES = frozenset(
    {
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    }
)
_VIBESPATIAL_METADATA_KEY = "vibespatial"
_VIBESPATIAL_SHAPE_PROOF_KEY = "shape_proof"
_REGULAR_GRID_RECT_PROOF_KIND = "regular_grid_rect"
_REGULAR_GRID_RECT_PROOF_VERSION = 1
_GEOPARQUET_SCAN_DECODE_MULTIPLIER = 5
_GEOPARQUET_WKB_SCAN_DECODE_MULTIPLIER = 8
_GEOPARQUET_SCAN_ROW_SCRATCH_BYTES = 16
_GEOPARQUET_EAGER_SIZE_PROOF_MIN_BYTES = 1 << 20


def _native_partition_manifest_path(path) -> Path | None:
    if not isinstance(path, (str, PathLike)):
        return None
    parquet_path = Path(path)
    if not parquet_path.name:
        return None
    return parquet_path.with_suffix(parquet_path.suffix + ".partitions.json")


def _native_partition_file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    sample_bytes = 64 * 1024
    digest = hashlib.sha256()
    with path.open("rb") as source:
        head = source.read(sample_bytes)
        digest.update(head)
        if stat.st_size > sample_bytes:
            source.seek(max(stat.st_size - sample_bytes, sample_bytes))
            digest.update(source.read(sample_bytes))
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "sample_sha256": digest.hexdigest(),
    }


def _remove_native_partition_manifest(path) -> None:
    manifest_path = _native_partition_manifest_path(path)
    if manifest_path is not None:
        manifest_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class LegacyWKBGeoParquetTranscodeResult:
    """Bounded evidence from a metadata-only device-native transcode."""

    row_count: int
    column_count: int
    geometry_columns: tuple[str, ...]
    primary_geometry: str
    source_bytes: int
    output_bytes: int
    backend: str = "pylibcudf"
    schema_validated: bool = True
    values_validated: bool = True
    atomic_publication: bool = True


@dataclass(frozen=True, slots=True)
class NativePartitionedParquetSegment:
    """One partition-homogeneous row group in a clustered spill."""

    partition: int
    row_group: int
    batch: int
    row_count: int


@dataclass(frozen=True, slots=True)
class NativePartitionedParquetLayout:
    """Host-sized routing metadata for one device-clustered Parquet spill."""

    path: Path
    manifest_path: Path
    partition_column: str
    partition_count: int
    row_count: int
    segments: tuple[NativePartitionedParquetSegment, ...]

    def row_groups_for(self, partitions: set[int] | tuple[int, ...] | list[int]):
        selected = {int(value) for value in partitions}
        invalid = sorted(value for value in selected if not 0 <= value < self.partition_count)
        if invalid:
            raise ValueError(f"partition ids are outside the clustered spill: {invalid}")
        return tuple(
            segment.row_group for segment in self.segments if segment.partition in selected
        )


class NativePartitionedParquetSink:
    """Persistent device writer for partition-clustered Parquet row groups.

    This is an internal dynamic-output carrier. Each appended pylibcudf table
    is clustered by a device partition map, and each bounded homogeneous slice
    becomes one row group in one persistent output file. A small JSON manifest
    maps logical partitions to row groups for exact projection/pushdown reads.
    """

    def __init__(
        self,
        path,
        *,
        arrow_schema=None,
        partition_column: str,
        partition_count: int,
        compression: str | None = "snappy",
        max_row_group_rows: int = 1_000_000,
        max_input_chunk_rows: int = 32_000_000,
    ) -> None:
        self.path = Path(path)
        self.manifest_path = self.path.with_suffix(self.path.suffix + ".partitions.json")
        self.arrow_schema = None
        self.partition_column = str(partition_column)
        self.partition_count = int(partition_count)
        self.compression = compression
        self.max_row_group_rows = int(max_row_group_rows)
        self.max_input_chunk_rows = int(max_input_chunk_rows)
        if self.partition_count <= 0:
            raise ValueError("partition_count must be positive")
        if self.max_row_group_rows <= 0:
            raise ValueError("max_row_group_rows must be positive")
        if self.max_input_chunk_rows <= 0:
            raise ValueError("max_input_chunk_rows must be positive")
        self._partition_column_index = -1
        self._writer = None
        self._stream = None
        self._cupy_stream = None
        self._batch_count = 0
        self._row_count = 0
        self._segments: list[NativePartitionedParquetSegment] = []
        self._footer_metadata: dict[str, str] = {}
        if arrow_schema is not None:
            self._bind_arrow_schema(arrow_schema)

    def _bind_arrow_schema(self, arrow_schema) -> None:
        import base64

        schema_metadata = dict(arrow_schema.metadata or {})
        geo_metadata_bytes = schema_metadata.get(b"geo")
        if geo_metadata_bytes is not None:
            geo_metadata = json.loads(geo_metadata_bytes)
            for column_metadata in geo_metadata.get("columns", {}).values():
                # Each appended batch only knows its own bounds. Publishing
                # the first batch's bbox as the file-wide bbox would be false.
                column_metadata.pop("bbox", None)
            schema_metadata[b"geo"] = json.dumps(
                geo_metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            arrow_schema = arrow_schema.with_metadata(schema_metadata)
        if self.arrow_schema is not None:
            if not self.arrow_schema.equals(arrow_schema, check_metadata=True):
                raise ValueError("device spill batches do not share one Arrow schema")
            return
        if self.partition_column not in arrow_schema.names:
            raise ValueError("partition column is missing from the declared Arrow schema")
        self.arrow_schema = arrow_schema
        self._partition_column_index = arrow_schema.get_field_index(self.partition_column)
        self._footer_metadata = {
            (key.decode() if isinstance(key, bytes) else str(key)): (
                value.decode() if isinstance(value, bytes) else str(value)
            )
            for key, value in (arrow_schema.metadata or {}).items()
        }
        self._footer_metadata["ARROW:schema"] = base64.b64encode(
            arrow_schema.serialize().to_pybytes()
        ).decode()

    def _ensure_writer(self, table) -> None:
        if self._writer is not None:
            return
        import cupy as cp
        import pylibcudf as plc

        from vibespatial.cuda._runtime import pylibcudf_current_stream

        if self.arrow_schema is None:
            raise ValueError("device spill schema must be bound before the first append")
        if int(table.num_columns()) != len(self.arrow_schema):
            raise ValueError("device spill table does not match the declared column count")
        metadata = plc.io.types.TableInputMetadata(table)
        for index, field in enumerate(self.arrow_schema):
            column_metadata = metadata.column_metadata[index]
            column_metadata.set_name(field.name)
            if field.metadata and field.metadata.get(b"ARROW:extension:name") == b"geoarrow.wkb":
                column_metadata.set_output_as_binary(True)
            else:
                _apply_arrow_nested_child_metadata(column_metadata, field)
        _remove_native_partition_manifest(self.path)
        self._cupy_stream = cp.cuda.get_current_stream()
        self._stream = pylibcudf_current_stream(table)
        builder = plc.io.parquet.ChunkedParquetWriterOptions.builder(
            plc.io.types.SinkInfo([str(self.path)])
        )
        builder.metadata(metadata)
        builder.key_value_metadata([self._footer_metadata])
        builder.write_arrow_schema(False)
        builder.compression(_compression_type_from_name(self.compression))
        builder.row_group_size_rows(self.max_row_group_rows)
        self._writer = plc.io.parquet.ChunkedParquetWriter.from_options(
            builder.build(),
            stream=self._stream,
        )

    def append(self, table, *, partition_map=None, arrow_schema=None) -> None:
        """Cluster and append one device table without materializing its rows."""
        import cupy as cp
        import pylibcudf as plc

        from vibespatial.cuda._runtime import (
            cuda_stream_identity,
            get_cuda_completion_retainer,
            pylibcudf_current_stream,
        )

        row_count = int(table.num_rows())
        if row_count == 0:
            return
        if arrow_schema is not None:
            self._bind_arrow_schema(arrow_schema)
        if self.arrow_schema is None:
            raise ValueError("arrow_schema is required for the first device spill batch")
        if int(table.num_columns()) != len(self.arrow_schema):
            raise ValueError("device spill table does not match the declared column count")
        self._ensure_writer(table)
        assert self._stream is not None
        assert self._cupy_stream is not None
        mapping = (
            table.columns()[self._partition_column_index]
            if partition_map is None
            else partition_map
        )
        if int(mapping.size()) != row_count or int(mapping.null_count()) != 0:
            raise ValueError("partition map must be non-null and row-aligned")
        producer_stream = cp.cuda.get_current_stream()
        pylibcudf_current_stream(table, mapping)
        producer_event = None
        if cuda_stream_identity(producer_stream) != cuda_stream_identity(self._cupy_stream):
            producer_event = cp.cuda.Event(disable_timing=True)
            producer_event.record(producer_stream)
            self._cupy_stream.wait_event(producer_event)
        clustered, offsets = plc.partitioning.partition(
            table,
            mapping,
            self.partition_count,
            stream=self._stream,
        )
        completion_owners = [table, mapping, clustered, producer_event]
        for partition in range(self.partition_count):
            start = int(offsets[partition])
            stop = int(offsets[partition + 1])
            for chunk_start in range(start, stop, self.max_row_group_rows):
                chunk_stop = min(chunk_start + self.max_row_group_rows, stop)
                chunk = plc.copying.slice(
                    clustered,
                    [chunk_start, chunk_stop],
                    stream=self._stream,
                )[0]
                row_group = len(self._segments)
                assert self._writer is not None
                self._writer.write(chunk)
                self._segments.append(
                    NativePartitionedParquetSegment(
                        partition=partition,
                        row_group=row_group,
                        batch=self._batch_count,
                        row_count=chunk_stop - chunk_start,
                    )
                )
                completion_owners.append(chunk)
        get_cuda_completion_retainer().defer(
            self._cupy_stream,
            tuple(completion_owners),
            lambda _owners: None,
        )
        self._row_count += row_count
        self._batch_count += 1

    def _append_native_table(self, table, *, arrow_schema) -> None:
        """Consume one native writer chunk through the clustered sink protocol."""
        self.append(table, arrow_schema=arrow_schema)

    def close(self) -> NativePartitionedParquetLayout:
        """Close the persistent writer and publish the routing manifest."""
        if self._writer is None:
            raise ValueError("cannot close a clustered spill with no appended rows")
        self._writer.close([])
        self._writer = None
        manifest = {
            "schema_version": 2,
            "file": self.path.name,
            "file_identity": _native_partition_file_identity(self.path),
            "partition_column": self.partition_column,
            "partition_count": self.partition_count,
            "row_count": self._row_count,
            "segments": [
                {
                    "partition": segment.partition,
                    "row_group": segment.row_group,
                    "batch": segment.batch,
                    "row_count": segment.row_count,
                }
                for segment in self._segments
            ],
        }
        self.manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return NativePartitionedParquetLayout(
            path=self.path,
            manifest_path=self.manifest_path,
            partition_column=self.partition_column,
            partition_count=self.partition_count,
            row_count=self._row_count,
            segments=tuple(self._segments),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._writer is not None:
            self._writer.close([])
            self._writer = None


def load_native_partitioned_parquet_layout(path) -> NativePartitionedParquetLayout:
    """Load a clustered-spill routing manifest without scanning table rows."""
    parquet_path = Path(path)
    manifest_path = parquet_path.with_suffix(parquet_path.suffix + ".partitions.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError("clustered-spill manifest schema is unsupported")
    if payload.get("file") != parquet_path.name:
        raise ValueError("clustered-spill manifest names a different Parquet file")
    if payload.get("file_identity") != _native_partition_file_identity(parquet_path):
        raise ValueError("clustered-spill manifest does not match the Parquet file")
    return NativePartitionedParquetLayout(
        path=parquet_path,
        manifest_path=manifest_path,
        partition_column=str(payload["partition_column"]),
        partition_count=int(payload["partition_count"]),
        row_count=int(payload["row_count"]),
        segments=tuple(
            NativePartitionedParquetSegment(
                partition=int(segment["partition"]),
                row_group=int(segment["row_group"]),
                batch=int(segment["batch"]),
                row_count=int(segment["row_count"]),
            )
            for segment in payload["segments"]
        ),
    )


def _clustered_partition_filter_row_groups(path, filters):
    """Resolve one exact partition equality predicate through the sidecar."""
    if filters is None:
        return None
    predicate = filters
    if isinstance(filters, list) and len(filters) == 1:
        predicate = filters[0]
    if not isinstance(predicate, tuple) or len(predicate) != 3 or predicate[1] not in ("=", "=="):
        return None
    try:
        layout = load_native_partitioned_parquet_layout(path)
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None
    if str(predicate[0]) != layout.partition_column:
        return None
    try:
        partition = int(predicate[2])
    except (TypeError, ValueError):
        return None
    if not 0 <= partition < layout.partition_count:
        return None
    row_groups = layout.row_groups_for([partition])
    # Preserve the ordinary filter path for a logically empty partition so
    # the reader can still construct the projected zero-row schema.
    return row_groups or None


def read_native_partitioned_parquet_partitions(
    path,
    partitions,
    *,
    columns=None,
):
    """Read only row groups belonging to selected logical partitions."""
    layout = load_native_partitioned_parquet_layout(path)
    row_groups = layout.row_groups_for(partitions)
    return _read_geoparquet_table_with_pylibcudf(
        layout.path,
        columns=columns,
        row_groups=row_groups,
    )


def _regular_grid_rect_proof_from_column_metadata(
    column_meta: dict[str, Any] | None,
    *,
    row_count: int,
) -> DeviceRegularGridRectMetadata | None:
    if not isinstance(column_meta, dict):
        return None
    native_meta = column_meta.get(_VIBESPATIAL_METADATA_KEY)
    if not isinstance(native_meta, dict):
        return None
    proof = native_meta.get(_VIBESPATIAL_SHAPE_PROOF_KEY)
    if not isinstance(proof, dict):
        return None
    if proof.get("kind") != _REGULAR_GRID_RECT_PROOF_KIND:
        return None
    if int(proof.get("version", -1)) != _REGULAR_GRID_RECT_PROOF_VERSION:
        return None
    if int(proof.get("size", -1)) != int(row_count):
        return None
    total_bounds = proof.get("total_bounds")
    if not isinstance(total_bounds, (list, tuple)) or len(total_bounds) != 4:
        return None
    try:
        origin_x = float(proof["origin_x"])
        origin_y = float(proof["origin_y"])
        cell_width = float(proof["cell_width"])
        cell_height = float(proof["cell_height"])
        cols = int(proof["cols"])
        rows = int(proof["rows"])
        size = int(proof["size"])
        total_bounds_tuple = tuple(float(value) for value in total_bounds)
    except (KeyError, TypeError, ValueError):
        return None
    if cell_width <= 0.0 or cell_height <= 0.0 or cols <= 0 or rows <= 0:
        return None
    if size != int(row_count):
        return None
    if not np.isfinite(
        np.asarray(
            [
                origin_x,
                origin_y,
                cell_width,
                cell_height,
                *total_bounds_tuple,
            ],
            dtype=np.float64,
        )
    ).all():
        return None
    return DeviceRegularGridRectMetadata(
        origin_x=origin_x,
        origin_y=origin_y,
        cell_width=cell_width,
        cell_height=cell_height,
        cols=cols,
        rows=rows,
        size=size,
        total_bounds=total_bounds_tuple,
    )


def _attach_regular_grid_rect_proof_from_column_metadata(
    owned: OwnedGeometryArray,
    column_meta: dict[str, Any] | None,
) -> None:
    """Rehydrate a trusted terminal GeoParquet regular-grid proof."""
    if owned.residency is not Residency.DEVICE or owned.device_state is None:
        return
    if set(owned.device_state.families) != {GeometryFamily.POLYGON}:
        return
    polygon_buffer = owned.device_state.families.get(GeometryFamily.POLYGON)
    if polygon_buffer is None:
        return
    if int(getattr(polygon_buffer.geometry_offsets, "size", 0)) != int(owned.row_count) + 1:
        return
    if int(getattr(polygon_buffer.ring_offsets, "size", 0)) != int(owned.row_count) + 1:
        return
    proof = _regular_grid_rect_proof_from_column_metadata(
        column_meta,
        row_count=owned.row_count,
    )
    if proof is None:
        return
    polygon_buffer.dense_single_ring_width = 5
    polygon_buffer.axis_aligned_rectangles = True
    polygon_buffer.regular_grid_rect = proof


def _attach_device_geometry_planning_metadata(
    owned: OwnedGeometryArray,
    column_meta: dict[str, Any] | None,
) -> None:
    """Attach serialized proofs and bound exceptionally large decoded roots."""
    _attach_regular_grid_rect_proof_from_column_metadata(owned, column_meta)
    if owned.residency is not Residency.DEVICE or owned.device_state is None:
        return
    if not isinstance(column_meta, dict) or str(column_meta.get("encoding", "")).upper() != "WKB":
        # GeoArrow keeps its nested physical layout explicit. WKB decode is the
        # boundary that otherwise loses all host-visible per-row width proof.
        return
    polygon_storage_bytes = sum(
        int(getattr(value, "nbytes", 0))
        for family, buffer in owned.device_state.families.items()
        if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
        for value in (
            buffer.x,
            buffer.y,
            buffer.geometry_offsets,
            buffer.part_offsets,
            buffer.ring_offsets,
        )
        if value is not None
    )
    if polygon_storage_bytes < _GEOPARQUET_EAGER_SIZE_PROOF_MIN_BYTES:
        return

    # Large variable-width roots can be duplicated into relation-shaped
    # consumers before any operation-specific stage can bound them. Reduce the
    # already-resident offsets once at the IO/setup boundary so later compute
    # stays capacity-safe. The packet contains only three int64 shape scalars
    # per family; coordinates and row metadata remain device resident.
    from vibespatial.geometry.owned import ensure_device_geometry_size_bounds

    ensure_device_geometry_size_bounds(
        owned,
        reason="GeoParquet IO-time device geometry size-bound planning packet",
    )


def _record_public_geoparquet_dispatch(
    *,
    selected: ExecutionMode,
    implementation: str,
    reason: str,
    row_count: int,
    detail: str | None = None,
) -> None:
    row_detail = f"rows={row_count}"
    if detail:
        row_detail = f"{row_detail}, {detail}"
    record_dispatch_event(
        surface="geopandas.geodataframe.to_parquet",
        operation="to_parquet",
        implementation=implementation,
        reason=reason,
        detail=row_detail,
        selected=selected,
    )


def _payload_geometry_series(payload: NativeTabularResult):
    # GeoParquet geometry metadata depends on CRS, geometry type domain, and
    # optional shape proofs; public index labels are serialized by the Arrow
    # table export below.  Use a synthetic row index here so metadata assembly
    # does not duplicate a device-position host export for host-label plans.
    index = pd.RangeIndex(payload.geometry.row_count)
    return _authoritative_geometry_series(
        payload.geometry.to_geoseries(
            index=index,
            name=payload.geometry_name,
        )
    )


def _authoritative_owned_geometry_array(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray:
    if owned.device_state is None:
        return owned

    validity, tags, family_row_offsets, families = _authoritative_geoarrow_host_view(owned)
    authoritative = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
        residency=owned.residency,
        diagnostics=list(owned.diagnostics),
        runtime_history=list(owned.runtime_history),
        geoarrow_backed=owned.geoarrow_backed,
        shares_geoarrow_memory=owned.shares_geoarrow_memory,
        device_adopted=owned.device_adopted,
        device_state=owned.device_state,
        _row_count=owned.row_count,
    )
    if owned._base is not None:
        authoritative._base = owned._base
    if owned._index_map is not None:
        authoritative._index_map = owned._index_map
    return authoritative


def _authoritative_geometry_series(series):
    arr = series.array
    if isinstance(arr, DeviceGeometryArray) and arr.owned.device_state is not None:
        from vibespatial.api.geoseries import GeoSeries

        authoritative_owned = _authoritative_owned_geometry_array(arr.owned)
        authoritative_array = DeviceGeometryArray._from_owned(
            authoritative_owned,
            crs=series.crs,
        )
        return GeoSeries(authoritative_array, index=series.index, name=series.name, crs=series.crs)
    return series


def _authoritative_native_tabular_result(
    payload: NativeTabularResult,
) -> NativeTabularResult:
    def authoritative_geometry_result(geometry: GeometryNativeResult) -> GeometryNativeResult:
        if geometry.composition is not None:
            from vibespatial.api._native_result_core import (
                NativeGeometryComposition,
                NativeGeometryCompositionPart,
            )

            return GeometryNativeResult.from_composition(
                NativeGeometryComposition(
                    parts=tuple(
                        NativeGeometryCompositionPart(
                            authoritative_geometry_result(part.geometry),
                            part.output_rows,
                        )
                        for part in geometry.composition.parts
                    ),
                    row_count=geometry.composition.row_count,
                    crs=geometry.crs,
                ),
                crs=geometry.crs,
            )
        owned = geometry.owned
        if owned is None or owned.device_state is None:
            return geometry
        return GeometryNativeResult.from_owned(
            _authoritative_owned_geometry_array(owned),
            crs=geometry.crs,
        )

    authoritative_geometry = authoritative_geometry_result(payload.geometry)
    authoritative_secondary = tuple(
        NativeGeometryColumn(
            column.name,
            authoritative_geometry_result(column.geometry),
        )
        for column in payload.secondary_geometry
    )
    if (
        authoritative_geometry is payload.geometry
        and authoritative_secondary == payload.secondary_geometry
    ):
        return payload
    return NativeTabularResult(
        attributes=payload.attributes,
        geometry=authoritative_geometry,
        geometry_name=payload.geometry_name,
        column_order=payload.column_order,
        attrs=payload.attrs,
        secondary_geometry=authoritative_secondary,
        provenance=payload.provenance,
        geometry_metadata=payload.geometry_metadata,
        index_plan=payload.index_plan,
    )


def _record_terminal_geoparquet_compatibility_export(
    *,
    detail: str,
    implementation: str,
    row_count: int,
) -> None:
    _record_public_geoparquet_dispatch(
        selected=ExecutionMode.CPU,
        implementation=implementation,
        reason=(
            "terminal GeoParquet export used the explicit Arrow compatibility writer "
            "after the native device writer declined a sink feature"
        ),
        row_count=row_count,
        detail=detail,
    )


def _terminal_arrow_export_selected_mode(owned: OwnedGeometryArray | None) -> ExecutionMode:
    if owned is not None and (
        owned.device_state is not None or owned.residency is Residency.DEVICE
    ):
        return ExecutionMode.GPU
    return ExecutionMode.CPU


def _record_terminal_geoparquet_native_arrow_export(
    *,
    detail: str,
    implementation: str,
    row_count: int,
    owned: OwnedGeometryArray | None,
) -> None:
    selected = _terminal_arrow_export_selected_mode(owned)
    _record_public_geoparquet_dispatch(
        selected=selected,
        implementation=implementation,
        reason=(
            "terminal GeoParquet export used the shared native Arrow sink after "
            "owned geometry encoding"
        ),
        row_count=row_count,
        detail=detail,
    )


def _owned_prefers_small_terminal_arrow_export(owned: OwnedGeometryArray | None) -> bool:
    if owned is None:
        return False
    if owned.device_state is not None:
        families = frozenset(owned.device_state.families)
    else:
        families = frozenset(owned.families)
    return bool(families) and families.issubset(_SMALL_TERMINAL_ARROW_EXPORT_FAMILIES)


def _small_terminal_arrow_export_detail(
    *,
    row_count: int,
    polygonal_terminal_candidate: bool,
) -> str | None:
    if not polygonal_terminal_candidate or row_count > _SMALL_TERMINAL_ARROW_EXPORT_MAX_ROWS:
        return None
    return (
        "small terminal GeoParquet write prefers the shared native Arrow sink; "
        "polygonal outputs are faster through the Arrow sink at this size while "
        "geometry encoding stays owned/native; "
        f"row_count={row_count} <= {_SMALL_TERMINAL_ARROW_EXPORT_MAX_ROWS}"
    )


def _try_promote_geoparquet_geometry_columns_to_device(
    df,
    geometry_columns,
) -> tuple[bool, list]:
    """Build device-owned geometry columns for public GeoParquet writes."""
    if not has_gpu_runtime() or geometry_columns.size == 0:
        return False, []
    if not df.columns.is_unique:
        return False, []

    candidates = []
    snapshots = []
    try:
        for column_name in geometry_columns:
            array = df[column_name].array
            if isinstance(array, DeviceGeometryArray):
                owned = array.owned
            else:
                original_owned = getattr(array, "_owned", None)
                original_residency = (
                    original_owned.residency if original_owned is not None else None
                )
                snapshots.append((array, original_owned, original_residency))
                has_z = getattr(array, "has_z", None)
                if has_z is None:
                    raise ValueError("geometry array does not expose has_z")
                if callable(has_z):
                    has_z = has_z()
                if bool(np.any(np.asarray(has_z, dtype=bool))):
                    raise ValueError("3D geometry is not supported by owned GeoParquet promotion")
                to_owned = getattr(array, "to_owned", None)
                if not callable(to_owned):
                    raise ValueError("geometry array does not expose to_owned")
                owned = to_owned()
            if owned is None:
                raise ValueError("geometry column did not produce owned buffers")
            candidates.append((owned, owned.residency))
        for owned, _original_residency in candidates:
            owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason=(
                    "GeoParquet public write promotes supported host geometry "
                    "to device-owned buffers"
                ),
            )
    except (NotImplementedError, TypeError, ValueError):
        _restore_geoparquet_promoted_geometry_columns(snapshots)
        return False, []
    return True, snapshots


def _restore_geoparquet_promoted_geometry_columns(snapshots) -> None:
    for array, original_owned, original_residency in reversed(snapshots):
        if original_owned is not None and original_residency is not None:
            if original_owned.residency is not original_residency:
                original_owned.move_to(
                    original_residency,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason=(
                        "GeoParquet public write restores caller geometry "
                        "residency after temporary device-owned promotion"
                    ),
                )
        array._owned = original_owned


def _geometry_columns_are_device_owned(df, geometry_columns) -> bool:
    if geometry_columns.size == 0:
        return False
    for column_name in geometry_columns:
        array = df[column_name].array
        owned = (
            array.owned
            if isinstance(array, DeviceGeometryArray)
            else getattr(array, "_owned", None)
        )
        if owned is None:
            return False
        if owned.device_state is None and owned.residency is not Residency.DEVICE:
            return False
    return True


def _write_native_tabular_result_with_arrow(
    payload: NativeTabularResult,
    path,
    *,
    index,
    compression,
    geometry_encoding,
    schema_version,
    write_covering_bbox,
    **kwargs,
) -> None:
    import pyarrow.parquet as pq

    from vibespatial.api.io.arrow import _native_tabular_to_arrow

    table = _native_tabular_to_arrow(
        payload,
        index=index,
        geometry_encoding=geometry_encoding,
        schema_version=schema_version,
        write_covering_bbox=write_covering_bbox,
        force_device_geometry_encode=True,
    )
    pq.write_table(table, path, compression=compression, **kwargs)


def _decode_pylibcudf_geoparquet_column_with_arrow_fallback(
    table,
    *,
    column_name: str,
    column_index: int,
    encoding: str | None,
    column_meta: dict[str, Any] | None = None,
    schema=None,
) -> OwnedGeometryArray:
    import pyarrow as pa

    from vibespatial.cuda._runtime import pylibcudf_to_arrow

    try:
        decoder = _decode_pylibcudf_geoparquet_column_to_owned
        if "column_meta" in inspect.signature(decoder).parameters:
            return decoder(
                table.columns()[column_index],
                encoding,
                column_meta=column_meta,
            )
        return decoder(table.columns()[column_index], encoding)
    except NotImplementedError as exc:
        record_fallback_event(
            surface="vibespatial.io.geoparquet",
            reason="explicit CPU fallback after GPU GeoParquet geometry decode could not complete",
            detail=(
                f"column={column_name}, encoding={encoding!r}, detail={type(exc).__name__}: {exc}"
            ),
            selected=ExecutionMode.CPU,
            pipeline="io/read_parquet",
            d2h_transfer=True,
        )
        arrow_table = pylibcudf_to_arrow(table)
        arrow_column_index = arrow_table.schema.get_field_index(column_name)
        if arrow_column_index == -1:
            arrow_column_index = int(column_index)
        if schema is not None and column_name in schema.names:
            field = schema.field(column_name)
        else:
            field = arrow_table.schema.field(arrow_column_index)
        array = arrow_table.column(arrow_column_index).combine_chunks()
        normalized_encoding = None if encoding is None else str(encoding).lower()
        if normalized_encoding == "wkb":
            if pa.types.is_string(array.type):
                array = pa.Array.from_buffers(
                    pa.binary(),
                    len(array),
                    array.buffers(),
                    null_count=array.null_count,
                )
                field = pa.field(
                    field.name,
                    pa.binary(),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            elif pa.types.is_large_string(array.type):
                array = pa.Array.from_buffers(
                    pa.large_binary(),
                    len(array),
                    array.buffers(),
                    null_count=array.null_count,
                )
                field = pa.field(
                    field.name,
                    pa.large_binary(),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
        try:
            return _decode_geoarrow_array_to_owned(field, array, encoding=encoding)
        except _GeoArrowNativeCompatibilityRoute as geoarrow_exc:
            raise NotImplementedError(str(geoarrow_exc)) from geoarrow_exc


def _decode_arrow_geoparquet_column_to_host_geoseries(
    table,
    *,
    column_name: str,
    column_index: int,
    encoding: str | None,
    crs,
    index,
):
    import pyarrow as pa

    from vibespatial.api.geometry_array import from_wkb
    from vibespatial.api.geoseries import GeoSeries
    from vibespatial.cuda._runtime import pylibcudf_to_arrow

    arrow_table = pylibcudf_to_arrow(table) if _is_pylibcudf_table(table) else table
    arrow_column_index = arrow_table.schema.get_field_index(column_name)
    if arrow_column_index == -1:
        arrow_column_index = int(column_index)
    array = arrow_table.column(arrow_column_index).combine_chunks()
    normalized_encoding = None if encoding is None else str(encoding).lower()
    if normalized_encoding == "wkb":
        if pa.types.is_string(array.type):
            array = pa.Array.from_buffers(
                pa.binary(),
                len(array),
                array.buffers(),
                null_count=array.null_count,
            )
        elif pa.types.is_large_string(array.type):
            array = pa.Array.from_buffers(
                pa.large_binary(),
                len(array),
                array.buffers(),
                null_count=array.null_count,
            )
        values = np.asarray(array.to_pylist(), dtype=object)
        return GeoSeries(from_wkb(values, crs=crs), index=index, crs=crs, name=column_name)
    raise NotImplementedError(
        "host GeoSeries GeoParquet fallback currently supports WKB-encoded columns only"
    )


def _write_geoparquet_native_tabular_result(
    payload: NativeTabularResult,
    path,
    *,
    index,
    compression,
    geometry_encoding,
    schema_version,
    write_covering_bbox,
    **kwargs,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vibespatial.api.io.arrow import _create_geometry_metadata, _replace_table_schema_metadata

    small_write_detail = _small_terminal_arrow_export_detail(
        row_count=payload.geometry.row_count,
        polygonal_terminal_candidate=_owned_prefers_small_terminal_arrow_export(
            payload.geometry.owned
        ),
    )
    if small_write_detail is not None:
        _record_terminal_geoparquet_native_arrow_export(
            detail=small_write_detail,
            implementation="native_payload_arrow_terminal_export",
            row_count=payload.geometry.row_count,
            owned=payload.geometry.owned,
        )
        _write_native_tabular_result_with_arrow(
            payload,
            path,
            index=index,
            compression=compression,
            geometry_encoding=geometry_encoding,
            schema_version=schema_version,
            write_covering_bbox=write_covering_bbox,
            **kwargs,
        )
        return

    device_write = _write_geoparquet_native_device_payload(
        payload.attributes,
        payload.geometry.owned,
        path,
        geometry_composition=payload.geometry.composition,
        geometry_name=payload.geometry_name,
        geometry_crs=payload.geometry.crs,
        index=index,
        compression=compression,
        geometry_encoding=geometry_encoding,
        schema_version=schema_version,
        write_covering_bbox=write_covering_bbox,
        column_order=payload.resolved_column_order,
        frame_attrs=payload.attrs,
        index_plan=payload.index_plan,
        **kwargs,
    )
    if device_write.written:
        _record_public_geoparquet_dispatch(
            selected=ExecutionMode.GPU,
            implementation="native_payload_device_export",
            reason=(
                "GeoParquet export stayed on the native device payload writer and did not "
                "materialize a public GeoDataFrame-shaped Arrow export"
            ),
            row_count=payload.geometry.row_count,
        )
        return
    if device_write.fallback_detail is not None:
        record_fallback_event(
            surface="geopandas.geodataframe.to_parquet",
            reason="explicit CPU fallback from the native device GeoParquet payload writer to the Arrow writer",
            detail=device_write.fallback_detail,
            selected=ExecutionMode.CPU,
            pipeline="io/to_parquet",
            d2h_transfer=True,
        )
        _record_public_geoparquet_dispatch(
            selected=ExecutionMode.CPU,
            implementation="native_payload_arrow_fallback_export",
            reason=(
                "GeoParquet export fell back to the explicit Arrow writer after the "
                "native device payload writer declined the sink"
            ),
            row_count=payload.geometry.row_count,
            detail=device_write.fallback_detail,
        )
    elif device_write.compatibility_detail is not None:
        _record_terminal_geoparquet_compatibility_export(
            detail=device_write.compatibility_detail,
            implementation="native_payload_arrow_compatibility_export",
            row_count=payload.geometry.row_count,
        )
    else:
        _record_public_geoparquet_dispatch(
            selected=ExecutionMode.CPU,
            implementation="native_payload_arrow_export",
            reason=(
                "GeoParquet export used the explicit Arrow writer because the native "
                "device payload writer was unavailable for this payload"
            ),
            row_count=payload.geometry.row_count,
        )

    payload = _authoritative_native_tabular_result(payload)
    geometry_series = _payload_geometry_series(payload)

    table, geometry_encoding_dict = native_tabular_to_arrow(
        payload,
        index=index,
        geometry_encoding=geometry_encoding,
        interleaved=False,
        include_z=None,
        force_device_geometry_encode=True,
    )

    geo_metadata = _create_geometry_metadata(
        {payload.geometry_name: geometry_series},
        primary_column=payload.geometry_name,
        schema_version=schema_version,
        geometry_encoding=geometry_encoding_dict,
        write_covering_bbox=write_covering_bbox,
    )

    if write_covering_bbox:
        bounds = geometry_series.bounds
        bbox_array = pa.StructArray.from_arrays(
            [bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"]],
            names=["xmin", "ymin", "xmax", "ymax"],
        )
        table = table.append_column("bbox", bbox_array)

    table = _replace_table_schema_metadata(
        table,
        geo_metadata=geo_metadata,
        attrs=payload.attrs,
    )

    pq.write_table(table, path, compression=compression, **kwargs)


@dataclass(frozen=True)
class GeoParquetScanPlan:
    selected_path: IOPathKind
    canonical_gpu: bool
    uses_pylibcudf: bool
    bbox_requested: bool
    metadata_summary_available: bool
    metadata_source: str | None
    uses_covering_bbox: bool
    uses_point_encoding_pushdown: bool
    row_group_pushdown: bool
    planner_strategy: str
    available_row_groups: int | None
    selected_row_groups: tuple[int, ...] | None
    decoded_row_fraction_estimate: float | None
    pruned_row_group_fraction: float | None
    reason: str


@dataclass(frozen=True)
class GeoParquetChunkPlan:
    chunk_index: int
    row_groups: tuple[int, ...] | None
    estimated_rows: int
    estimated_uncompressed_bytes: int = 0


@dataclass(frozen=True)
class GeoParquetEnginePlan:
    selected_path: IOPathKind
    backend: str
    geometry_encoding: str | None
    chunk_count: int
    target_chunk_rows: int | None
    uses_row_group_pruning: bool
    reason: str


@dataclass(frozen=True)
class GeoParquetReadBackendPlan:
    requested_backend: str
    selected_backend: str
    selected_mode: ExecutionMode
    can_use_pylibcudf: bool
    gpu_rejection_reason: str | None
    reason: str
    explicit_host_compatibility: bool = False


@dataclass(frozen=True)
class GeoParquetEngineBenchmark:
    backend: str
    geometry_encoding: str
    rows: int
    chunk_rows: int | None
    chunk_count: int
    elapsed_seconds: float
    rows_per_second: float
    planning_elapsed_seconds: float = 0.0
    scan_elapsed_seconds: float = 0.0
    decode_elapsed_seconds: float = 0.0
    concat_elapsed_seconds: float = 0.0


_PYLIBCUDF_GEOPARQUET_ENCODINGS = frozenset(
    {
        "point",
        "linestring",
        "polygon",
        "multipoint",
        "multilinestring",
        "multipolygon",
        "wkb",
    }
)
_DEFAULT_GPU_GEOPARQUET_CHUNK_ROWS = 250_000


def _unsupported_pylibcudf_geoparquet_encoding(
    geo_metadata: dict[str, Any] | None,
    columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, str | None] | None:
    if geo_metadata is None:
        return None
    requested = set(columns or ())
    geometry_columns = [
        name for name in geo_metadata["columns"] if not requested or name in requested
    ]
    for column_name in geometry_columns:
        encoding = geo_metadata["columns"][column_name].get("encoding")
        normalized = None if encoding is None else str(encoding).lower()
        if normalized not in _PYLIBCUDF_GEOPARQUET_ENCODINGS:
            return column_name, normalized
    return None


def _rebuild_arrow_array_with_schema_type(array, expected_type):
    import pyarrow as pa

    if array.type == expected_type:
        return array
    if (
        pa.types.is_string(array.type)
        or pa.types.is_large_string(array.type)
        or pa.types.is_binary(array.type)
        or pa.types.is_large_binary(array.type)
    ) and (
        pa.types.is_string(expected_type)
        or pa.types.is_large_string(expected_type)
        or pa.types.is_binary(expected_type)
        or pa.types.is_large_binary(expected_type)
    ):
        return array.cast(expected_type)

    children = None
    if pa.types.is_struct(expected_type):
        children = [
            _rebuild_arrow_array_with_schema_type(array.field(index), expected_type[index].type)
            for index in range(expected_type.num_fields)
        ]
    elif (
        pa.types.is_list(expected_type)
        or pa.types.is_large_list(expected_type)
        or pa.types.is_fixed_size_list(expected_type)
    ):
        children = [_rebuild_arrow_array_with_schema_type(array.values, expected_type.value_type)]

    return pa.Array.from_buffers(
        expected_type,
        len(array),
        array.buffers()[: expected_type.num_buffers],
        null_count=array.null_count,
        offset=array.offset,
        children=children,
    )


def _rebuild_arrow_table_with_schema(table, schema):
    import pyarrow as pa

    columns = []
    for index, field in enumerate(schema):
        source_column = table.column(index)
        columns.append(
            pa.chunked_array(
                [
                    _rebuild_arrow_array_with_schema_type(chunk, field.type)
                    for chunk in source_column.chunks
                ],
                type=field.type,
            )
        )
    return pa.Table.from_arrays(columns, schema=schema)


def _project_arrow_schema(schema, columns):
    if columns is None:
        return schema

    import pyarrow as pa

    selected_fields = [schema.field(name) for name in columns if name in schema.names]
    return pa.schema(selected_fields, metadata=schema.metadata)


def _is_local_arrow_filesystem(filesystem) -> bool:
    if filesystem is None:
        return True
    try:
        import pyarrow.fs as pafs
    except ImportError:  # pragma: no cover - pyarrow present in normal envs
        return False
    if isinstance(filesystem, pafs.LocalFileSystem):
        return True
    base_fs = getattr(filesystem, "base_fs", None)
    if base_fs is not None and base_fs is not filesystem:
        return _is_local_arrow_filesystem(base_fs)
    return False


def _is_local_geoparquet_file(path, *, filesystem=None) -> bool:
    if filesystem is not None and not _is_local_arrow_filesystem(filesystem):
        return False
    if filesystem is not None and _is_local_arrow_filesystem(filesystem):
        try:
            import pyarrow.fs as pafs

            info = filesystem.get_file_info(path)
            return info.type == pafs.FileType.File
        except Exception:
            return False
    candidate = path
    if isinstance(candidate, PathLike):
        candidate = candidate.__fspath__()
    if not isinstance(candidate, str):
        candidate = str(candidate)
    if "://" in candidate:
        return False
    return Path(candidate).is_file()


def _is_local_geoparquet_source(path, *, filesystem=None) -> bool:
    if isinstance(path, (bytes, io.BytesIO)):
        return True
    if filesystem is not None and _is_local_arrow_filesystem(filesystem):
        try:
            import pyarrow.fs as pafs

            info = filesystem.get_file_info(path)
            if info.type in {pafs.FileType.File, pafs.FileType.Directory}:
                return True
        except Exception:
            pass
    try:
        is_file = _is_local_geoparquet_file(path, filesystem=filesystem)
    except TypeError as exc:
        if "unexpected keyword argument 'filesystem'" not in str(exc):
            raise
        is_file = _is_local_geoparquet_file(path)
    if is_file:
        return True
    candidate = path
    if isinstance(candidate, PathLike):
        candidate = candidate.__fspath__()
    if not isinstance(candidate, str):
        candidate = str(candidate)
    if "://" in candidate:
        return False
    return Path(candidate).is_dir()


def _validate_geoparquet_bbox_support(
    geo_metadata: dict[str, Any] | None,
    bbox,
) -> None:
    if bbox is None or geo_metadata is None:
        return
    from vibespatial.api.io.arrow import _get_parquet_bbox_filter

    _get_parquet_bbox_filter(geo_metadata, bbox)


def _normalize_parquet_filters(filters):
    if filters is None:
        return None
    import pyarrow.compute as pc
    import pyarrow.parquet as parquet

    if isinstance(filters, pc.Expression):
        return filters
    return parquet.filters_to_expression(filters)


def _hidden_index_fields_from_schema_metadata(
    schema_metadata: dict[bytes, bytes] | None,
) -> list[str]:
    if schema_metadata is None:
        return []
    pandas_metadata_raw = schema_metadata.get(b"pandas")
    if pandas_metadata_raw is None:
        return []
    pandas_metadata = json.loads(pandas_metadata_raw.decode("utf-8"))
    return [
        str(index_column)
        for index_column in (pandas_metadata.get("index_columns") or [])
        if not isinstance(index_column, dict)
    ]


def _physical_hidden_index_fields_from_schema(schema) -> list[str]:
    available = set(schema.names)
    return [
        field
        for field in _hidden_index_fields_from_schema_metadata(schema.metadata)
        if field in available
    ]


def _merge_column_projection(*column_groups) -> list[str] | None:
    merged: list[str] = []
    seen: set[str] = set()
    for group in column_groups:
        if group is None:
            continue
        for name in group:
            text = str(name)
            if text not in seen:
                seen.add(text)
                merged.append(text)
    return merged or None


def _range_index_from_arrow_schema(schema, *, row_count: int) -> pd.RangeIndex:
    """Recover a metadata-only pandas RangeIndex without exporting columns."""
    metadata = None if schema is None else schema.metadata
    pandas_metadata_raw = None if metadata is None else metadata.get(b"pandas")
    if pandas_metadata_raw is None:
        return pd.RangeIndex(row_count)
    try:
        pandas_metadata = json.loads(pandas_metadata_raw.decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return pd.RangeIndex(row_count)
    index_columns = pandas_metadata.get("index_columns") or []
    if len(index_columns) != 1 or not isinstance(index_columns[0], dict):
        return pd.RangeIndex(row_count)
    range_spec = index_columns[0]
    if range_spec.get("kind") != "range":
        return pd.RangeIndex(row_count)
    candidate = pd.RangeIndex(
        start=int(range_spec.get("start", 0)),
        stop=int(range_spec.get("stop", row_count)),
        step=int(range_spec.get("step", 1)),
        name=range_spec.get("name"),
    )
    # Row-group and predicate scans may return only part of the physical file.
    # Parquet's file-level range metadata cannot describe those selected rows.
    return candidate if len(candidate) == int(row_count) else pd.RangeIndex(row_count)


def _device_attributes_from_pylibcudf_scan(
    table,
    *,
    table_column_names,
    attribute_columns,
    hidden_index_fields,
    schema,
    to_pandas_kwargs=None,
) -> NativeAttributeTable:
    """Attach projected scan columns directly to the native attribute carrier.

    Only physical pandas index columns cross to host, because pandas requires a
    public Index object at the compatibility boundary. Ordinary attributes stay
    in their pylibcudf columns until an explicit user export.
    """
    import pyarrow as pa
    import pylibcudf as plc

    names = tuple(table_column_names)
    positions = {name: index for index, name in enumerate(names)}
    source_columns = table.columns()
    row_count = _table_row_count(table)

    missing_attributes = [name for name in attribute_columns if name not in positions]
    if missing_attributes:
        raise ValueError(
            f"pylibcudf GeoParquet scan omitted requested attribute columns: {missing_attributes!r}"
        )

    index_override: pd.Index = _range_index_from_arrow_schema(
        schema,
        row_count=row_count,
    )
    scanned_index_fields = [name for name in hidden_index_fields if name in positions]
    if scanned_index_fields:
        from vibespatial.cuda._runtime import pylibcudf_to_arrow

        index_columns = [source_columns[positions[name]] for name in scanned_index_fields]
        host_index_table = pylibcudf_to_arrow(plc.Table(index_columns))
        index_fields = [schema.field(name) for name in scanned_index_fields]
        host_index_table = pa.Table.from_arrays(
            [host_index_table.column(index) for index in range(host_index_table.num_columns)],
            schema=pa.schema(index_fields, metadata=schema.metadata),
        )
        index_override = native_attribute_table_from_arrow_table(
            host_index_table,
            to_pandas_kwargs=to_pandas_kwargs,
        ).index

    if not attribute_columns:
        return NativeAttributeTable(
            dataframe=pd.DataFrame(index=index_override),
            to_pandas_kwargs=to_pandas_kwargs,
        )

    device_columns = [source_columns[positions[name]] for name in attribute_columns]
    fields = [schema.field(name) for name in attribute_columns]
    return NativeAttributeTable(
        device_table=plc.Table(device_columns),
        index_override=index_override,
        column_override=tuple(attribute_columns),
        schema_override=pa.schema(fields, metadata=schema.metadata),
        to_pandas_kwargs=to_pandas_kwargs,
    )


def _parquet_filter_column_names(
    filters,
    *,
    available_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    if filters is None:
        return ()

    ordered: list[str] = []
    seen: set[str] = set()

    def add_name(name: str) -> None:
        text = str(name)
        if text not in seen:
            seen.add(text)
            ordered.append(text)

    def walk_filter_tree(node) -> None:
        if isinstance(node, list):
            for item in node:
                walk_filter_tree(item)
            return
        if isinstance(node, tuple):
            if len(node) == 3 and isinstance(node[0], str):
                add_name(node[0])
                return
            for item in node:
                walk_filter_tree(item)

    walk_filter_tree(filters)
    if available_columns is None:
        return tuple(ordered)

    expr_text = str(_normalize_parquet_filters(filters))
    resolved: list[str] = []
    for name in available_columns:
        text = str(name)
        if text in seen or re.search(
            rf"(?<![0-9A-Za-z_]){re.escape(text)}(?![0-9A-Za-z_])",
            expr_text,
        ):
            resolved.append(text)
    return tuple(resolved)


def _compile_pylibcudf_parquet_filter(
    filters,
    *,
    available_columns: tuple[str, ...] | list[str],
):
    if filters is None:
        return None
    import pylibcudf as plc

    normalized = _normalize_parquet_filters(filters)
    return plc.expressions.to_expression(
        str(normalized), tuple(str(name) for name in available_columns)
    )


def _supports_pylibcudf_geoparquet_read(
    path,
    *,
    bbox,
    columns,
    storage_options,
    filesystem,
    filters,
    to_pandas_kwargs,
    geo_metadata,
    available_columns=None,
) -> tuple[bool, str]:
    if not has_pylibcudf_support():
        return False, "pylibcudf is not installed for the GPU GeoParquet reader"
    if not has_gpu_runtime():
        return False, "GPU GeoParquet reader requires an available CUDA runtime"
    if to_pandas_kwargs:
        return False, "to_pandas_kwargs require the host pyarrow compatibility reader"
    if storage_options is not None:
        return False, "filesystem-backed GeoParquet reads still route through host pyarrow"
    if filesystem is not None and not _is_local_arrow_filesystem(filesystem):
        return False, "filesystem-backed GeoParquet reads still route through host pyarrow"
    if geo_metadata is not None:
        primary = geo_metadata["primary_column"]
        if primary not in geo_metadata["columns"]:
            return (
                False,
                "GeoParquet metadata without a readable primary geometry routes through host pyarrow",
            )
        if bbox is not None:
            try:
                _validate_geoparquet_bbox_support(geo_metadata, bbox)
            except ValueError as exc:
                return False, str(exc)
        unsupported = _unsupported_pylibcudf_geoparquet_encoding(geo_metadata, columns)
        if unsupported is not None:
            column_name, encoding = unsupported
            return (
                False,
                f"geometry column {column_name!r} with GeoParquet encoding {encoding!r} still routes through host pyarrow",
            )
    if filters is not None and available_columns is not None:
        try:
            _compile_pylibcudf_parquet_filter(filters, available_columns=available_columns)
        except Exception as exc:
            return (
                False,
                f"predicate filter could not be compiled for the pylibcudf scan backend: {exc}",
            )
    if not _is_local_geoparquet_source(path, filesystem=filesystem):
        return False, "dataset and non-local GeoParquet paths still route through host pyarrow"
    return True, "local GeoParquet scan can use the pylibcudf reader"


def plan_geoparquet_read_backend(
    path,
    *,
    backend: str,
    bbox,
    columns,
    storage_options,
    filesystem,
    filters,
    to_pandas_kwargs,
    geo_metadata,
    available_columns=None,
) -> GeoParquetReadBackendPlan:
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unsupported GeoParquet backend request: {backend!r}")

    can_use_pylibcudf, gpu_reason = _supports_pylibcudf_geoparquet_read(
        path,
        bbox=bbox,
        columns=columns,
        storage_options=storage_options,
        filesystem=filesystem,
        filters=filters,
        to_pandas_kwargs=to_pandas_kwargs,
        geo_metadata=geo_metadata,
        available_columns=available_columns,
    )
    explicit_host_compatibility = backend == "auto" and (
        bool(to_pandas_kwargs)
        or storage_options is not None
        or (filesystem is not None and not _is_local_arrow_filesystem(filesystem))
    )
    if backend == "cpu":
        return GeoParquetReadBackendPlan(
            requested_backend=backend,
            selected_backend="pyarrow",
            selected_mode=ExecutionMode.CPU,
            can_use_pylibcudf=can_use_pylibcudf,
            gpu_rejection_reason=None,
            reason="explicit CPU backend requested for the GeoParquet read path",
        )
    if can_use_pylibcudf:
        return GeoParquetReadBackendPlan(
            requested_backend=backend,
            selected_backend="pylibcudf",
            selected_mode=ExecutionMode.GPU,
            can_use_pylibcudf=True,
            gpu_rejection_reason=None,
            reason=(
                "explicit GPU backend requested for the GeoParquet read path"
                if backend == "gpu"
                else "auto selected the GPU GeoParquet scan backend"
            ),
        )

    reason = gpu_reason or "GPU GeoParquet scan backend is unavailable"
    if backend == "gpu":
        return GeoParquetReadBackendPlan(
            requested_backend=backend,
            selected_backend="pyarrow",
            selected_mode=ExecutionMode.CPU,
            can_use_pylibcudf=False,
            gpu_rejection_reason=reason,
            reason=f"explicit GPU backend requested but {reason}",
        )
    return GeoParquetReadBackendPlan(
        requested_backend=backend,
        selected_backend="pyarrow",
        selected_mode=ExecutionMode.CPU,
        can_use_pylibcudf=False,
        gpu_rejection_reason=reason,
        reason=(
            f"explicit filesystem compatibility transport selected the host GeoParquet "
            f"scan backend because {reason}"
            if explicit_host_compatibility
            else f"auto selected the host GeoParquet scan backend because {reason}"
        ),
        explicit_host_compatibility=explicit_host_compatibility,
    )


def _record_geoparquet_scan_backend_fallback(*, surface: str, detail: str) -> None:
    record_fallback_event(
        surface=surface,
        reason="explicit CPU fallback for GeoParquet scan backend selection",
        detail=detail,
        selected=ExecutionMode.CPU,
        pipeline="io/read_parquet",
        d2h_transfer=False,
    )


def _is_geoparquet_scan_ineligible_for_gpu_fallback(geo_metadata) -> bool:
    if geo_metadata is None:
        return False
    primary = geo_metadata.get("primary_column")
    columns = geo_metadata.get("columns", {})
    return primary not in columns


def plan_geoparquet_scan(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    geo_metadata: dict[str, Any] | None = None,
    metadata_summary: GeoParquetMetadataSummary | None = None,
    planner_strategy: str = "auto",
) -> GeoParquetScanPlan:
    plan = plan_io_support(IOFormat.GEOPARQUET, IOOperation.SCAN)
    uses_covering_bbox = False
    uses_point_encoding_pushdown = False
    if geo_metadata is not None and bbox is not None:
        primary = geo_metadata["primary_column"]
        column_meta = geo_metadata["columns"][primary]
        uses_covering_bbox = "covering" in column_meta
        uses_point_encoding_pushdown = column_meta.get("encoding") == "point"
    prune_result = None
    if metadata_summary is not None and metadata_summary.has_spatial_bounds and bbox is not None:
        prune_result = select_row_groups(metadata_summary, bbox, strategy=planner_strategy)
    return GeoParquetScanPlan(
        selected_path=plan.selected_path,
        canonical_gpu=plan.canonical_gpu,
        uses_pylibcudf=has_pylibcudf_support() and has_gpu_runtime(),
        bbox_requested=bbox is not None,
        metadata_summary_available=metadata_summary is not None,
        metadata_source=metadata_summary.source if metadata_summary is not None else None,
        uses_covering_bbox=uses_covering_bbox,
        uses_point_encoding_pushdown=uses_point_encoding_pushdown,
        row_group_pushdown=(
            bbox is not None
            and (prune_result is not None or uses_covering_bbox or uses_point_encoding_pushdown)
        ),
        planner_strategy=prune_result.strategy if prune_result is not None else planner_strategy,
        available_row_groups=metadata_summary.row_group_count
        if metadata_summary is not None
        else None,
        selected_row_groups=prune_result.selected_row_groups if prune_result is not None else None,
        decoded_row_fraction_estimate=prune_result.decoded_row_fraction
        if prune_result is not None
        else None,
        pruned_row_group_fraction=prune_result.pruned_row_group_fraction
        if prune_result is not None
        else None,
        reason=(
            "GeoParquet reads should prefer a GPU scanner plus metadata-first pushdown; "
            "without pylibcudf the current path falls back to host pyarrow scanning, "
            "but row groups can still be pruned before full geometry decode."
        ),
    )


def plan_geoparquet_engine(
    *,
    geo_metadata: dict[str, Any] | None,
    scan_plan: GeoParquetScanPlan,
    chunk_plans: tuple[GeoParquetChunkPlan, ...],
    target_chunk_rows: int | None,
    read_plan: GeoParquetReadBackendPlan,
) -> GeoParquetEnginePlan:
    primary_column = None if geo_metadata is None else geo_metadata["primary_column"]
    geometry_encoding = None
    if (
        geo_metadata is not None
        and primary_column is not None
        and primary_column in geo_metadata["columns"]
    ):
        geometry_encoding = geo_metadata["columns"][primary_column].get("encoding")
    return GeoParquetEnginePlan(
        selected_path=scan_plan.selected_path,
        backend=read_plan.selected_backend,
        geometry_encoding=geometry_encoding,
        chunk_count=len(chunk_plans),
        target_chunk_rows=target_chunk_rows,
        uses_row_group_pruning=scan_plan.row_group_pushdown,
        reason=(
            f"{read_plan.reason}; keep row-group pruning from the metadata planner and "
            "decode supported geometry encodings directly into owned buffers after scan."
        ),
    )


def _plan_geoparquet_chunks(
    *,
    metadata_summary: GeoParquetMetadataSummary | None,
    selected_row_groups: tuple[int, ...] | list[int] | None,
    target_chunk_rows: int | None,
    target_uncompressed_bytes: int | None = None,
) -> tuple[GeoParquetChunkPlan, ...]:
    if selected_row_groups is None:
        estimated_rows = metadata_summary.total_rows if metadata_summary is not None else 0
        estimated_bytes = (
            int(metadata_summary.row_group_uncompressed_bytes.sum(dtype=np.int64))
            if metadata_summary is not None
            and metadata_summary.row_group_uncompressed_bytes is not None
            else 0
        )
        return (
            GeoParquetChunkPlan(
                chunk_index=0,
                row_groups=None,
                estimated_rows=estimated_rows,
                estimated_uncompressed_bytes=estimated_bytes,
            ),
        )
    if len(selected_row_groups) == 0:
        return (GeoParquetChunkPlan(chunk_index=0, row_groups=tuple(), estimated_rows=0),)
    row_groups = tuple(selected_row_groups)
    if metadata_summary is None or (
        (target_chunk_rows is None or target_chunk_rows <= 0)
        and (target_uncompressed_bytes is None or target_uncompressed_bytes <= 0)
    ):
        estimated_rows = (
            int(sum(metadata_summary.row_group_rows[list(row_groups)]))
            if metadata_summary is not None
            else 0
        )
        estimated_bytes = (
            int(metadata_summary.row_group_uncompressed_bytes[list(row_groups)].sum(dtype=np.int64))
            if metadata_summary is not None
            and metadata_summary.row_group_uncompressed_bytes is not None
            else 0
        )
        return (
            GeoParquetChunkPlan(
                chunk_index=0,
                row_groups=row_groups,
                estimated_rows=estimated_rows,
                estimated_uncompressed_bytes=estimated_bytes,
            ),
        )
    chunks: list[GeoParquetChunkPlan] = []
    current: list[int] = []
    current_rows = 0
    current_bytes = 0
    for row_group in row_groups:
        group_rows = int(metadata_summary.row_group_rows[row_group])
        group_bytes = (
            int(metadata_summary.row_group_uncompressed_bytes[row_group])
            if metadata_summary.row_group_uncompressed_bytes is not None
            else 0
        )
        rows_exceeded = (
            target_chunk_rows is not None
            and target_chunk_rows > 0
            and current_rows + group_rows > target_chunk_rows
        )
        bytes_exceeded = (
            target_uncompressed_bytes is not None
            and target_uncompressed_bytes > 0
            and current_bytes + group_bytes > target_uncompressed_bytes
        )
        if current and (rows_exceeded or bytes_exceeded):
            chunks.append(
                GeoParquetChunkPlan(
                    chunk_index=len(chunks),
                    row_groups=tuple(current),
                    estimated_rows=current_rows,
                    estimated_uncompressed_bytes=current_bytes,
                )
            )
            current = []
            current_rows = 0
            current_bytes = 0
        current.append(row_group)
        current_rows += group_rows
        current_bytes += group_bytes
    if current:
        chunks.append(
            GeoParquetChunkPlan(
                chunk_index=len(chunks),
                row_groups=tuple(current),
                estimated_rows=current_rows,
                estimated_uncompressed_bytes=current_bytes,
            )
        )
    return tuple(chunks)


def _effective_geoparquet_chunk_rows(
    chunk_rows: int | None,
    *,
    selected_backend: str,
) -> int | None:
    if chunk_rows is not None:
        return chunk_rows
    if selected_backend == "pylibcudf":
        return _DEFAULT_GPU_GEOPARQUET_CHUNK_ROWS
    return None


def _geoparquet_scan_decode_multiplier(
    geo_metadata: dict[str, Any],
    columns,
) -> int:
    requested = None if columns is None else set(columns)
    geometry_columns = geo_metadata.get("columns", {})
    wkb_column_count = sum(
        (requested is None or name in requested)
        and str(metadata.get("encoding", "")).upper() == "WKB"
        for name, metadata in geometry_columns.items()
    )
    if not wkb_column_count:
        return _GEOPARQUET_SCAN_DECODE_MULTIPLIER
    # The base WKB envelope covers the Parquet scan plus one structural plan
    # and owned decode. Every additional requested WKB geometry column has its
    # own simultaneous plan/output state, so charge that incremental share per
    # column instead of admitting a two-geometry scan as though it had one.
    incremental_wkb_share = (
        _GEOPARQUET_WKB_SCAN_DECODE_MULTIPLIER
        - _GEOPARQUET_SCAN_DECODE_MULTIPLIER
    )
    return _GEOPARQUET_SCAN_DECODE_MULTIPLIER + (
        incremental_wkb_share * wkb_column_count
    )


def _geoparquet_target_uncompressed_bytes(
    selected_backend: str,
    *,
    decode_multiplier: int = _GEOPARQUET_SCAN_DECODE_MULTIPLIER,
) -> int | None:
    if selected_backend != "pylibcudf" or not has_gpu_runtime():
        return None
    from vibespatial.cuda._runtime import get_cuda_runtime

    remaining = get_cuda_runtime().query_memory_remaining_bytes()
    # WKB decode retains more simultaneous structural and owned state than
    # native GeoArrow adoption, and Parquet row groups are indivisible. Reserve
    # four planning shares: one for whole-row-group overshoot and the remainder
    # for libcudf's large temporary allocations plus repeated-run pool
    # fragmentation. SF100 profiling demonstrated that two shares could leave
    # ample aggregate free bytes without a contiguous 1.5 GiB pool block.
    reserve_shares = (
        4
        if int(decode_multiplier) >= _GEOPARQUET_WKB_SCAN_DECODE_MULTIPLIER
        else 1
    )
    return max(remaining // (int(decode_multiplier) + reserve_shares), 1)


def _admit_geoparquet_chunk(
    chunk: GeoParquetChunkPlan,
    *,
    decode_multiplier: int = _GEOPARQUET_SCAN_DECODE_MULTIPLIER,
) -> None:
    if chunk.estimated_uncompressed_bytes <= 0 or not has_gpu_runtime():
        return
    from vibespatial.cuda._runtime import get_cuda_runtime

    required = (
        chunk.estimated_uncompressed_bytes * int(decode_multiplier)
        + chunk.estimated_rows * _GEOPARQUET_SCAN_ROW_SCRATCH_BYTES
    )
    admission = get_cuda_runtime().admit_device_memory(
        stage="geoparquet.scan_decode",
        required_bytes=required,
        requested_units=chunk.estimated_rows,
    )
    if not admission.admitted:
        raise MemoryError(
            "GeoParquet scan chunk exceeded the device query budget after "
            f"metadata planning: required={required}, "
            f"remaining={admission.remaining_bytes}, "
            f"admitted_rows={admission.admitted_units}"
        )


def _geoparquet_scan_sources(path, *, metadata_summary: GeoParquetMetadataSummary | None):
    if metadata_summary is not None and metadata_summary.source_paths is not None:
        return list(metadata_summary.source_paths)
    if isinstance(path, (bytes, io.BytesIO)):
        return [path]
    if isinstance(path, (str, PathLike)) and Path(path).is_dir():
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        return [str(fragment.path) for fragment in dataset.get_fragments()]
    return [path]


def _geoparquet_scan_row_groups(
    *,
    metadata_summary: GeoParquetMetadataSummary | None,
    selected_row_groups: tuple[int, ...] | list[int] | None,
):
    if selected_row_groups is None:
        return None
    if metadata_summary is not None and metadata_summary.source_paths is not None:
        source_count = len(metadata_summary.source_paths)
        grouped = [[] for _ in range(source_count)]
        source_indices = metadata_summary.row_group_source_indices
        source_row_groups = metadata_summary.row_group_source_row_groups
        if source_indices is None or source_row_groups is None:
            raise ValueError("dataset row-group selection requires per-source row-group metadata")
        for row_group in selected_row_groups:
            source_index = int(source_indices[row_group])
            grouped[source_index].append(int(source_row_groups[row_group]))
        return grouped
    return [list(selected_row_groups)]


def _native_bbox_row_positions_and_metadata(payload: NativeTabularResult, bbox):
    if bbox is None:
        return None, payload.geometry_metadata
    geometry = payload.geometry
    if geometry.owned is not None:
        import cupy as cp

        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

        d_bounds = compute_geometry_bounds_device(geometry.owned)
        d_keep = ~(
            (d_bounds[:, 0] > bbox[2])
            | (d_bounds[:, 1] > bbox[3])
            | (d_bounds[:, 2] < bbox[0])
            | (d_bounds[:, 3] < bbox[1])
        )
        metadata = payload.geometry_metadata
        if metadata is None:
            metadata = NativeGeometryMetadata.from_cached_owned(geometry.owned)
        if metadata.bounds is None:
            metadata = replace(
                metadata,
                bounds=d_bounds,
                residency=Residency.DEVICE,
            )
        return cp.flatnonzero(d_keep).astype(cp.int64, copy=False), metadata

    geometry_series = geometry.to_geoseries(
        index=pd.RangeIndex(geometry.row_count),
        name="geometry",
    )
    bounds = np.asarray(geometry_series.bounds, dtype=np.float64)
    keep = ~(
        (bounds[:, 0] > bbox[2])
        | (bounds[:, 1] > bbox[3])
        | (bounds[:, 2] < bbox[0])
        | (bounds[:, 3] < bbox[1])
    )
    return np.flatnonzero(keep).astype(np.int64, copy=False), payload.geometry_metadata


def _native_bbox_row_positions(payload: NativeTabularResult, bbox):
    row_positions, _metadata = _native_bbox_row_positions_and_metadata(payload, bbox)
    return row_positions


def _apply_native_bbox_filter(payload: NativeTabularResult, bbox) -> NativeTabularResult:
    if bbox is None:
        return payload
    row_positions, metadata = _native_bbox_row_positions_and_metadata(payload, bbox)
    if metadata is not payload.geometry_metadata:
        payload = replace(payload, geometry_metadata=metadata)
    return payload.take(row_positions)


def _apply_owned_bbox_filter(owned: OwnedGeometryArray, bbox) -> OwnedGeometryArray:
    if bbox is None:
        return owned
    import cupy as cp

    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    d_bounds = compute_geometry_bounds_device(owned)
    d_keep = ~(
        (d_bounds[:, 0] > bbox[2])
        | (d_bounds[:, 1] > bbox[3])
        | (d_bounds[:, 2] < bbox[0])
        | (d_bounds[:, 3] < bbox[1])
    )
    return owned.take(cp.flatnonzero(d_keep).astype(cp.int64, copy=False))


def _table_row_count(table) -> int:
    num_rows = getattr(table, "num_rows", None)
    if num_rows is None:
        raise ValueError("table does not expose row count")
    if callable(num_rows):
        return int(num_rows())
    return int(num_rows)


def _geoparquet_column_total_bounds(
    column_meta: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    bbox = column_meta.get("bbox")
    if bbox is None:
        return None
    if len(bbox) != 4:
        return None
    return tuple(float(value) for value in bbox)


def _pylibcudf_table_to_geopandas(
    table,
    *,
    path,
    row_groups=None,
    filesystem=None,
    geo_metadata: dict[str, Any] | None,
    schema=None,
    table_column_names=None,
    to_pandas_kwargs=None,
    df_attrs=None,
):
    payload = _geoparquet_table_to_native_tabular_result(
        table,
        path=path,
        row_groups=row_groups,
        filesystem=filesystem,
        geo_metadata=geo_metadata,
        schema=schema,
        table_column_names=table_column_names,
        to_pandas_kwargs=to_pandas_kwargs,
        df_attrs=df_attrs,
    )
    frame = payload.to_geodataframe()
    attach_native_state_from_native_tabular_result(frame, payload)
    return frame


def _geoparquet_table_to_native_tabular_result(
    table,
    *,
    path,
    row_groups=None,
    filesystem=None,
    geo_metadata: dict[str, Any] | None,
    schema=None,
    table_column_names=None,
    requested_columns=None,
    to_pandas_kwargs=None,
    df_attrs=None,
    attrs_arrow=None,
    provenance: NativeReadProvenance | None = None,
    scanned_with_pylibcudf: bool | None = None,
    filters=None,
    sources=None,
) -> NativeTabularResult:
    import warnings

    import pandas as pd

    if geo_metadata is None:
        raise ValueError("GeoParquet metadata is required for native tabular decode")
    if to_pandas_kwargs is None:
        to_pandas_kwargs = {}
    if scanned_with_pylibcudf is None:
        scanned_with_pylibcudf = _is_pylibcudf_table(table)

    if schema is not None and requested_columns is not None:
        result_column_names = [name for name in requested_columns if name in schema.names]
    elif schema is not None:
        result_column_names = list(schema.names)
    elif hasattr(table, "column_names"):
        result_column_names = list(table.column_names)
    else:
        raise ValueError("GeoParquet native decode requires schema metadata")
    schema_metadata = None
    if schema is not None:
        schema_metadata = schema.metadata
    elif hasattr(table, "schema"):
        schema_metadata = table.schema.metadata
    hidden_index_fields = _hidden_index_fields_from_schema_metadata(schema_metadata)
    schema_names = set(schema.names) if schema is not None else set(result_column_names)
    hidden_index_fields = [field for field in hidden_index_fields if field in schema_names]
    if hidden_index_fields:
        hidden_index_field_set = set(hidden_index_fields)
        result_column_names = [
            column_name
            for column_name in result_column_names
            if column_name not in hidden_index_field_set
        ]
    if requested_columns is None:
        from vibespatial.api.io.arrow import (
            _check_if_covering_in_geo_metadata,
            _get_bbox_encoding_column_name,
        )

        if geo_metadata is not None and _check_if_covering_in_geo_metadata(geo_metadata):
            bbox_column_name = _get_bbox_encoding_column_name(geo_metadata)
            result_column_names = [
                column_name
                for column_name in result_column_names
                if column_name != bbox_column_name
            ]

    geometry_columns = [col for col in geo_metadata["columns"] if col in result_column_names]
    geometry_columns.sort(key=result_column_names.index)
    if not geometry_columns:
        raise ValueError(
            """No geometry columns are included in the columns read from
            the Parquet/Feather file.  To read this file without geometry columns,
            use pandas.read_parquet/read_feather() instead."""
        )

    geometry_name = geo_metadata["primary_column"]
    if geometry_name not in geometry_columns:
        geometry_name = geometry_columns[0]
        if len(geometry_columns) > 1:
            warnings.warn(
                "Multiple non-primary geometry columns read from Parquet/Feather "
                "file. The first column read was promoted to the primary geometry.",
                stacklevel=3,
            )

    non_geometry_columns = [col for col in result_column_names if col not in geometry_columns]
    if table_column_names is None:
        if hasattr(table, "column_names"):
            table_column_names = list(table.column_names)
        else:
            table_column_names = list(geometry_columns)
    else:
        table_column_names = list(table_column_names)

    if scanned_with_pylibcudf and attrs_arrow is None:
        attributes = _device_attributes_from_pylibcudf_scan(
            table,
            table_column_names=table_column_names,
            attribute_columns=non_geometry_columns,
            hidden_index_fields=hidden_index_fields,
            schema=schema,
            to_pandas_kwargs=to_pandas_kwargs,
        )
    elif attrs_arrow is None and not non_geometry_columns and not hidden_index_fields:
        attributes = NativeAttributeTable(
            dataframe=pd.DataFrame(index=pd.RangeIndex(_table_row_count(table))),
            to_pandas_kwargs=to_pandas_kwargs,
        )
    else:
        if attrs_arrow is None:
            attrs_arrow = table.drop(geometry_columns)

        attributes = native_attribute_table_from_arrow_table(
            attrs_arrow,
            to_pandas_kwargs=to_pandas_kwargs,
        )

    decoded_geometry: dict[str, GeometryNativeResult] = {}
    decoded_metadata: dict[str, NativeGeometryMetadata] = {}
    row_count = None
    for column_name in geometry_columns:
        column_meta = geo_metadata["columns"][column_name]
        crs = _geoparquet_geometry_column_crs(column_meta)
        scan_column_index = table_column_names.index(column_name)
        try:
            if scanned_with_pylibcudf:
                owned = _decode_pylibcudf_geoparquet_column_with_arrow_fallback(
                    table,
                    column_name=column_name,
                    column_index=scan_column_index,
                    encoding=column_meta.get("encoding"),
                    column_meta=column_meta,
                    schema=schema,
                )
            else:
                owned = _decode_arrow_geoparquet_table_to_owned(
                    table,
                    geo_metadata,
                    column_index=scan_column_index,
                )
            _attach_device_geometry_planning_metadata(
                owned,
                column_meta,
            )
            if row_count is None:
                row_count = owned.row_count
            decoded_geometry[column_name] = GeometryNativeResult.from_owned(owned, crs=crs)
            decoded_metadata[column_name] = NativeGeometryMetadata.from_cached_owned(
                owned,
                total_bounds=_geoparquet_column_total_bounds(column_meta),
            )
        except NotImplementedError as exc:
            record_fallback_event(
                surface="vibespatial.io.geoparquet",
                reason=(
                    "explicit CPU compatibility fallback after GeoParquet geometry decode "
                    "produced families outside the owned native result model"
                ),
                detail=(
                    f"column={column_name}, encoding={column_meta.get('encoding')!r}, "
                    f"detail={type(exc).__name__}: {exc}"
                ),
                selected=ExecutionMode.CPU,
                pipeline="io/read_parquet",
                d2h_transfer=True,
            )
            if scanned_with_pylibcudf:
                from vibespatial.cuda._runtime import pylibcudf_to_arrow

                host_decode_table = pylibcudf_to_arrow(table)
            else:
                host_decode_table = table
            host_series = _decode_arrow_geoparquet_column_to_host_geoseries(
                host_decode_table,
                column_name=column_name,
                column_index=scan_column_index,
                encoding=column_meta.get("encoding"),
                crs=crs,
                index=attributes.index if len(attributes.index) else None,
            )
            if row_count is None:
                row_count = len(host_series)
            decoded_geometry[column_name] = GeometryNativeResult.from_geoseries(host_series)

    if row_count is not None and len(attributes.index) != row_count and not len(attributes.columns):
        attributes = NativeAttributeTable(
            dataframe=pd.DataFrame(index=pd.RangeIndex(row_count)),
            to_pandas_kwargs=to_pandas_kwargs,
        )

    secondary_geometry = tuple(
        NativeGeometryColumn(column_name, decoded_geometry[column_name])
        for column_name in geometry_columns
        if column_name != geometry_name
    )
    attrs = None if df_attrs is None else json.loads(df_attrs)
    return NativeTabularResult(
        attributes=attributes,
        geometry=decoded_geometry[geometry_name],
        geometry_name=geometry_name,
        column_order=tuple(result_column_names),
        attrs=attrs,
        secondary_geometry=secondary_geometry,
        provenance=provenance,
        geometry_metadata=decoded_metadata.get(geometry_name),
    )


def _read_geoparquet_with_pylibcudf(
    path,
    *,
    columns=None,
    row_groups=None,
    filesystem=None,
    geo_metadata=None,
    to_pandas_kwargs=None,
):
    geometry_scan_columns = None
    if geo_metadata is not None:
        if columns is None:
            geometry_scan_columns = list(geo_metadata["columns"])
        else:
            requested_columns = set(columns)
            geometry_scan_columns = [
                name for name in geo_metadata["columns"] if name in requested_columns
            ]
        if not geometry_scan_columns:
            geometry_scan_columns = None
    gpu_table = _read_geoparquet_table_with_pylibcudf(
        path,
        columns=geometry_scan_columns or columns,
        row_groups=row_groups,
        filesystem=filesystem,
    )
    schema = None
    df_attrs = None
    if has_pyarrow_support():
        import pyarrow.parquet as pq

        schema = _project_arrow_schema(
            pq.read_schema(path, filesystem=filesystem),
            columns,
        )
        metadata = schema.metadata
        df_attrs = None if metadata is None else metadata.get(b"PANDAS_ATTRS")
    return _pylibcudf_table_to_geopandas(
        gpu_table,
        path=path,
        row_groups=row_groups,
        filesystem=filesystem,
        geo_metadata=geo_metadata,
        schema=schema,
        table_column_names=geometry_scan_columns,
        to_pandas_kwargs=to_pandas_kwargs,
        df_attrs=df_attrs,
    )


def _read_non_geometry_geoparquet_columns_as_arrow(
    path,
    *,
    columns,
    row_groups=None,
    filesystem=None,
    filters=None,
    sources=None,
):
    """ADR-0042: Read non-geometry columns as an Arrow table.

    Returns a PyArrow Table instead of a pandas DataFrame so that the
    caller can defer ``.to_pandas()`` to the GeoDataFrame construction
    boundary.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from vibespatial.api.io.arrow import _coerce_pyarrow_parquet_source

    scan_sources = [_coerce_pyarrow_parquet_source(source) for source in (sources or [path])]
    schema = pq.read_schema(scan_sources[0], filesystem=filesystem)
    available_columns = tuple(schema.names)
    available_column_set = set(available_columns)
    requested_columns = (
        _merge_column_projection(
            list(columns),
            _physical_hidden_index_fields_from_schema(schema),
        )
        or []
    )
    requested_columns = [column for column in requested_columns if column in available_column_set]
    filter_expression = _normalize_parquet_filters(filters)
    filter_columns = _parquet_filter_column_names(
        filters,
        available_columns=available_columns,
    )
    scan_columns = _merge_column_projection(requested_columns, filter_columns)

    def _select_requested_columns(table):
        selected = table.select(requested_columns)
        if schema.metadata is not None:
            selected = selected.replace_schema_metadata(schema.metadata)
        return _normalize_arrow_pandas_range_metadata(selected)

    if row_groups is None:
        if len(scan_sources) == 1:
            table = pq.read_table(
                scan_sources[0],
                columns=scan_columns,
                filesystem=filesystem,
                filters=filter_expression,
                use_pandas_metadata=False,
            )
            return _select_requested_columns(table)
        tables = [
            _select_requested_columns(
                pq.read_table(
                    source,
                    columns=scan_columns,
                    filesystem=filesystem,
                    filters=filter_expression,
                    use_pandas_metadata=False,
                )
            )
            for source in scan_sources
        ]
    else:
        if len(scan_sources) == 1 and (not row_groups or isinstance(row_groups[0], int)):
            grouped_row_groups = [list(row_groups)]
        else:
            grouped_row_groups = [list(groups) for groups in row_groups]
        tables = []
        for source, source_row_groups in zip(scan_sources, grouped_row_groups, strict=True):
            parquet_file = pq.ParquetFile(source, filesystem=filesystem)
            table = parquet_file.read_row_groups(
                source_row_groups,
                columns=scan_columns,
                use_threads=True,
                use_pandas_metadata=False,
            )
            if filter_expression is not None:
                if not isinstance(filter_expression, pc.Expression):
                    filter_expression = pq.filters_to_expression(filter_expression)
                table = table.filter(filter_expression)
            tables.append(_select_requested_columns(table))

    if not tables:
        return pa.table({name: pa.array([], type=pa.null()) for name in requested_columns})
    if len(tables) == 1:
        return _normalize_arrow_pandas_range_metadata(tables[0])
    return _normalize_arrow_pandas_range_metadata(pa.concat_tables(tables))


def _read_non_geometry_geoparquet_columns_with_pyarrow(
    path,
    *,
    columns,
    row_groups=None,
    filesystem=None,
    to_pandas_kwargs=None,
):
    """Legacy wrapper that returns pandas DataFrame directly.

    Kept for callers that need immediate pandas conversion.
    """
    if to_pandas_kwargs is None:
        to_pandas_kwargs = {}
    arrow_table = _read_non_geometry_geoparquet_columns_as_arrow(
        path,
        columns=columns,
        row_groups=row_groups,
        filesystem=filesystem,
    )
    return arrow_table.to_pandas(**to_pandas_kwargs)


@lru_cache(maxsize=32)
def _cached_geoparquet_crs_from_user_input(crs_value: str) -> Any:
    from pyproj import CRS

    return CRS.from_user_input(crs_value)


@lru_cache(maxsize=32)
def _cached_geoparquet_crs_from_json(crs_json: str) -> Any:
    from pyproj import CRS

    from vibespatial.api.io.arrow import _remove_id_from_member_of_ensembles

    crs = json.loads(crs_json)
    _remove_id_from_member_of_ensembles(crs)
    return CRS.from_user_input(crs)


def _geoparquet_geometry_column_crs(column_metadata: dict[str, Any]) -> Any:
    if "crs" in column_metadata:
        crs = column_metadata["crs"]
        if isinstance(crs, dict):
            try:
                return _cached_geoparquet_crs_from_json(
                    json.dumps(crs, sort_keys=True, separators=(",", ":"))
                )
            except Exception:
                return crs
        try:
            return _cached_geoparquet_crs_from_user_input(str(crs))
        except Exception:
            return crs
    try:
        return _cached_geoparquet_crs_from_user_input("OGC:CRS84")
    except Exception:
        return "OGC:CRS84"


def _read_geoparquet_table_with_pylibcudf(
    path,
    *,
    columns=None,
    row_groups=None,
    filesystem=None,
    filters=None,
    sources=None,
    available_columns=None,
):
    import pylibcudf as plc

    from vibespatial.cuda._runtime import pylibcudf_current_stream

    scan_sources = list(sources or [path])
    source = plc.io.types.SourceInfo(scan_sources)
    builder = plc.io.parquet.ParquetReaderOptions.builder(source)
    # GeoParquet metadata is loaded separately, and host bridge paths rebuild the
    # projected Arrow schema explicitly. Leaving these defaults enabled adds a
    # large one-time scan penalty in pylibcudf without helping the owned decode.
    builder.use_arrow_schema(False)
    builder.use_pandas_metadata(False)
    options = builder.build()
    if columns is not None:
        set_column_names = getattr(options, "set_column_names", None)
        if set_column_names is not None:
            set_column_names(list(columns))
        else:  # pragma: no cover - compatibility with older pylibcudf
            options.set_columns(list(columns))
    if row_groups is not None:
        if len(scan_sources) == 1 and (not row_groups or isinstance(row_groups[0], int)):
            grouped_row_groups = [list(row_groups)]
        else:
            grouped_row_groups = [list(groups) for groups in row_groups]
        options.set_row_groups(grouped_row_groups)
    if filters is not None:
        options.set_filter(
            _compile_pylibcudf_parquet_filter(
                filters,
                available_columns=tuple(available_columns or ()),
            )
        )
    table_with_metadata = plc.io.parquet.read_parquet(
        options,
        stream=pylibcudf_current_stream(),
    )
    return table_with_metadata.tbl


def _parquet_column_path(*components: str) -> str:
    return ".".join(components)


def _load_geoparquet_metadata(path, *, filesystem=None, storage_options=None):
    from vibespatial.api.io.arrow import (
        _get_filesystem_path,
        _read_parquet_schema_and_metadata,
        _validate_and_decode_metadata,
    )

    filesystem, normalized_path = _get_filesystem_path(
        path,
        filesystem=filesystem,
        storage_options=storage_options,
    )
    _, metadata = _read_parquet_schema_and_metadata(normalized_path, filesystem)
    geo_metadata = None
    if metadata is not None and b"geo" in metadata:
        geo_metadata = _validate_and_decode_metadata(metadata)
    return filesystem, normalized_path, metadata, geo_metadata


def _build_geoparquet_metadata_summary_from_pyarrow(
    path,
    *,
    filesystem,
    geo_metadata: dict[str, Any],
) -> GeoParquetMetadataSummary | None:
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs
    import pyarrow.parquet as parquet

    from vibespatial.api.io.arrow import _coerce_pyarrow_parquet_source, _ensure_arrow_fs

    filesystem = _ensure_arrow_fs(filesystem)

    primary = geo_metadata["primary_column"]
    if primary not in geo_metadata["columns"]:
        return None
    column_meta = geo_metadata["columns"][primary]

    if "covering" in column_meta:
        bbox_meta = column_meta["covering"]["bbox"]
        xmin_path = _parquet_column_path(*bbox_meta["xmin"])
        ymin_path = _parquet_column_path(*bbox_meta["ymin"])
        xmax_path = _parquet_column_path(*bbox_meta["xmax"])
        ymax_path = _parquet_column_path(*bbox_meta["ymax"])
        source = "covering_bbox"
    elif column_meta.get("encoding") == "point":
        xmin_path = _parquet_column_path(primary, "x")
        ymin_path = _parquet_column_path(primary, "y")
        xmax_path = xmin_path
        ymax_path = ymin_path
        source = "point_encoding"
    else:
        xmin_path = ymin_path = xmax_path = ymax_path = None
        source = "row_group_metadata"

    row_group_rows: list[int] = []
    row_group_uncompressed_bytes: list[int] = []
    xmin: list[float] = []
    ymin: list[float] = []
    xmax: list[float] = []
    ymax: list[float] = []
    source_paths: list[str] | None = None
    row_group_source_indices: list[int] | None = None
    row_group_source_row_groups: list[int] | None = None
    required = None if xmin_path is None else (xmin_path, ymin_path, xmax_path, ymax_path)
    complete_spatial_bounds = required is not None

    def append_metadata(file_metadata, *, source_index: int | None = None) -> None:
        nonlocal complete_spatial_bounds
        for row_group_index in range(file_metadata.num_row_groups):
            group = file_metadata.row_group(row_group_index)
            row_group_rows.append(int(group.num_rows))
            row_group_uncompressed_bytes.append(int(group.total_byte_size))
            if source_index is not None:
                assert row_group_source_indices is not None
                assert row_group_source_row_groups is not None
                row_group_source_indices.append(int(source_index))
                row_group_source_row_groups.append(int(row_group_index))
            if not complete_spatial_bounds or required is None:
                continue
            stats_by_path: dict[str, tuple[float, float]] = {}
            for column_index in range(group.num_columns):
                column = group.column(column_index)
                if column.path_in_schema not in required:
                    continue
                stats = column.statistics
                if stats is None or not getattr(stats, "has_min_max", False):
                    continue
                stats_by_path[column.path_in_schema] = (float(stats.min), float(stats.max))
            if any(path_name not in stats_by_path for path_name in required):
                complete_spatial_bounds = False
                continue
            assert xmin_path is not None
            assert ymin_path is not None
            assert xmax_path is not None
            assert ymax_path is not None
            xmin.append(stats_by_path[xmin_path][0])
            ymin.append(stats_by_path[ymin_path][0])
            xmax.append(stats_by_path[xmax_path][1])
            ymax.append(stats_by_path[ymax_path][1])

    if filesystem is not None and hasattr(filesystem, "get_file_info"):
        info = filesystem.get_file_info(path)
        if info.type == pafs.FileType.Directory:
            dataset = ds.dataset(path, filesystem=filesystem, format="parquet")
            fragments = list(dataset.get_fragments())
            source_paths = []
            row_group_source_indices = []
            row_group_source_row_groups = []
            for source_index, fragment in enumerate(fragments):
                file_metadata = getattr(fragment, "metadata", None)
                if file_metadata is None:
                    return None
                source_paths.append(str(fragment.path))
                append_metadata(file_metadata, source_index=source_index)
        elif info.type == pafs.FileType.File:
            path = _coerce_pyarrow_parquet_source(path)
            append_metadata(parquet.ParquetFile(path, filesystem=filesystem).metadata)
        else:
            return None
    elif filesystem is None and isinstance(path, (str, PathLike)) and Path(path).is_dir():
        dataset = ds.dataset(path, filesystem=filesystem, format="parquet")
        fragments = list(dataset.get_fragments())
        source_paths = []
        row_group_source_indices = []
        row_group_source_row_groups = []
        for source_index, fragment in enumerate(fragments):
            file_metadata = getattr(fragment, "metadata", None)
            if file_metadata is None:
                return None
            source_paths.append(str(fragment.path))
            append_metadata(file_metadata, source_index=source_index)
    else:
        path = _coerce_pyarrow_parquet_source(path)
        append_metadata(parquet.ParquetFile(path, filesystem=filesystem).metadata)

    if not row_group_rows:
        return None
    has_spatial_bounds = complete_spatial_bounds and len(xmin) == len(row_group_rows)

    return build_geoparquet_metadata_summary(
        source=source if has_spatial_bounds else "row_group_metadata",
        row_group_rows=row_group_rows,
        xmin=xmin if has_spatial_bounds else None,
        ymin=ymin if has_spatial_bounds else None,
        xmax=xmax if has_spatial_bounds else None,
        ymax=ymax if has_spatial_bounds else None,
        source_paths=source_paths,
        row_group_source_indices=row_group_source_indices,
        row_group_source_row_groups=row_group_source_row_groups,
        row_group_uncompressed_bytes=row_group_uncompressed_bytes,
    )


def _read_geoparquet_with_pyarrow(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    to_pandas_kwargs=None,
    row_groups: tuple[int, ...] | list[int] | None = None,
    **kwargs,
):
    table, geo_metadata, df_attrs = _read_geoparquet_table_with_pyarrow(
        path,
        columns=columns,
        storage_options=storage_options,
        bbox=bbox,
        row_groups=row_groups,
        **kwargs,
    )
    from vibespatial.api.io.arrow import _arrow_to_geopandas

    return _arrow_to_geopandas(
        table,
        geo_metadata,
        to_pandas_kwargs,
        df_attrs,
        fallback_surface="geopandas.read_parquet",
        fallback_pipeline="io/read_parquet",
    )


def _normalize_arrow_pandas_range_metadata(table):
    effective_metadata = table.schema.metadata
    if effective_metadata is not None:
        effective_metadata = dict(effective_metadata)
        pandas_metadata_raw = effective_metadata.get(b"pandas")
        if pandas_metadata_raw is not None:
            pandas_metadata = json.loads(pandas_metadata_raw.decode("utf-8"))
            index_columns = pandas_metadata.get("index_columns") or []
            if len(index_columns) == 1 and isinstance(index_columns[0], dict):
                range_spec = dict(index_columns[0])
                if range_spec.get("kind") == "range":
                    range_start = int(range_spec.get("start", 0))
                    range_stop = int(range_spec.get("stop", table.num_rows))
                    range_step = int(range_spec.get("step", 1))
                    expected_rows = max(
                        0, (range_stop - range_start + (range_step - 1)) // range_step
                    )
                    if expected_rows != int(table.num_rows):
                        range_spec["start"] = 0
                        range_spec["stop"] = int(table.num_rows)
                        range_spec["step"] = 1
                        pandas_metadata["index_columns"] = [range_spec]
                        effective_metadata[b"pandas"] = json.dumps(pandas_metadata).encode("utf-8")
    if effective_metadata is not None:
        return table.replace_schema_metadata(effective_metadata)
    return table


def _read_geoparquet_table_with_pyarrow(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    row_groups: tuple[int, ...] | list[int] | None = None,
    **kwargs,
):
    import pyarrow.parquet as parquet

    from vibespatial.api.io.arrow import (
        _check_if_covering_in_geo_metadata,
        _coerce_pyarrow_parquet_source,
        _get_bbox_encoding_column_name,
        _get_filesystem_path,
        _get_non_bbox_columns,
        _get_parquet_bbox_filter,
        _read_parquet_schema_and_metadata,
        _splice_bbox_and_filters,
        _validate_and_decode_metadata,
    )
    from vibespatial.api.io.file import _expand_user

    filesystem = kwargs.pop("filesystem", None)
    filesystem, normalized_path = _get_filesystem_path(
        path,
        filesystem=filesystem,
        storage_options=storage_options,
    )
    normalized_path = _expand_user(normalized_path)
    schema, metadata = _read_parquet_schema_and_metadata(normalized_path, filesystem)
    parquet_source = _coerce_pyarrow_parquet_source(normalized_path)
    geo_metadata = _validate_and_decode_metadata(metadata)
    if len(geo_metadata["columns"]) == 0:
        raise ValueError(
            """No geometry columns are included in the columns read from
            the Parquet/Feather file.  To read this file without geometry columns,
            use pandas.read_parquet/read_feather() instead."""
        )
    bbox_filter = _get_parquet_bbox_filter(geo_metadata, bbox) if bbox is not None else None
    if not columns and _check_if_covering_in_geo_metadata(geo_metadata):
        columns = _get_non_bbox_columns(schema, geo_metadata)
    if "filters" in kwargs:
        filters = _splice_bbox_and_filters(kwargs.pop("filters"), bbox_filter)
    else:
        filters = bbox_filter
    kwargs["use_pandas_metadata"] = True
    is_directory_dataset = False
    if isinstance(normalized_path, (str, PathLike)):
        if filesystem is None:
            is_directory_dataset = Path(normalized_path).is_dir()
        else:
            from vibespatial.api.io.arrow import _ensure_arrow_fs

            arrow_filesystem = _ensure_arrow_fs(filesystem)
            if hasattr(arrow_filesystem, "get_file_info"):
                import pyarrow.fs as pafs

                is_directory_dataset = (
                    arrow_filesystem.get_file_info(normalized_path).type == pafs.FileType.Directory
                )
    added_bbox_column = None
    row_group_columns = columns
    if row_group_columns is not None and filters is not None:
        row_group_columns = _merge_column_projection(
            row_group_columns,
            _parquet_filter_column_names(filters, available_columns=tuple(schema.names)),
        )
    if (
        row_groups is not None
        and bbox_filter is not None
        and _check_if_covering_in_geo_metadata(geo_metadata)
    ):
        bbox_column_name = _get_bbox_encoding_column_name(geo_metadata)
        if row_group_columns is not None and bbox_column_name not in row_group_columns:
            row_group_columns = [*row_group_columns, bbox_column_name]
            added_bbox_column = bbox_column_name
    if row_groups is None or is_directory_dataset:
        table = parquet.read_table(
            parquet_source,
            columns=columns,
            filesystem=filesystem,
            filters=filters,
            **kwargs,
        )
    else:
        import pyarrow.compute as pc

        parquet_file = parquet.ParquetFile(parquet_source, filesystem=filesystem)
        table = parquet_file.read_row_groups(
            list(row_groups),
            columns=row_group_columns,
            use_threads=kwargs.get("use_threads", True),
            use_pandas_metadata=kwargs["use_pandas_metadata"],
        )
        if filters is not None:
            if not isinstance(filters, pc.Expression):
                filters = parquet.filters_to_expression(filters)
            table = table.filter(filters)
        if added_bbox_column is not None and added_bbox_column in table.column_names:
            table = table.drop([added_bbox_column])
        if columns is not None:
            table = table.select(columns)
    effective_metadata = metadata
    if effective_metadata is not None:
        # `pq.read_table(..., use_pandas_metadata=True)` does not reliably
        # carry file footer metadata onto the returned Table schema. Reattach
        # it here so downstream `.to_pandas()` can restore index columns.
        table = table.replace_schema_metadata(effective_metadata)
    table = _normalize_arrow_pandas_range_metadata(table)
    effective_metadata = table.schema.metadata
    if effective_metadata and b"PANDAS_ATTRS" in effective_metadata:
        df_attrs = effective_metadata[b"PANDAS_ATTRS"]
    else:
        df_attrs = None
    return table, geo_metadata, df_attrs


def _decode_arrow_geoparquet_table_to_owned(
    table,
    geo_metadata: dict[str, Any],
    *,
    column_index: int | None = None,
) -> OwnedGeometryArray:
    primary = geo_metadata["primary_column"]
    field_index = (
        table.schema.get_field_index(primary) if column_index is None else int(column_index)
    )
    if field_index == -1:
        field_index = 0
    field = table.schema.field(field_index)
    array = table.column(field_index).combine_chunks()
    column_name = field.name
    if column_name not in geo_metadata["columns"]:
        if column_index is None:
            column_name = primary
        elif len(geo_metadata["columns"]) == 1:
            column_name = next(iter(geo_metadata["columns"]))
        else:
            raise KeyError(column_name)
    encoding = geo_metadata["columns"][column_name].get("encoding")
    owned = _decode_geoarrow_array_to_owned(field, array, encoding=encoding)
    _attach_device_geometry_planning_metadata(
        owned,
        geo_metadata["columns"][column_name],
    )
    return owned


def _decode_geoparquet_table_to_owned(
    table,
    geo_metadata: dict[str, Any],
    *,
    column_index: int | None = None,
) -> OwnedGeometryArray:
    if _is_pylibcudf_table(table):
        unsupported = _unsupported_pylibcudf_geoparquet_encoding(geo_metadata)
        if unsupported is not None:
            column_name, encoding = unsupported
            record_fallback_event(
                surface="vibespatial.io.geoparquet",
                reason="explicit CPU fallback until pylibcudf device decode covers the current GeoParquet encoding",
                detail=f"column={column_name}, encoding={encoding!r}",
                selected=ExecutionMode.CPU,
                pipeline="io/read_parquet",
                d2h_transfer=True,
            )
            from vibespatial.cuda._runtime import pylibcudf_to_arrow

            table = pylibcudf_to_arrow(table)
        else:
            primary = geo_metadata["primary_column"]
            decode_index = 0 if column_index is None else int(column_index)
            owned = _decode_pylibcudf_geoparquet_column_with_arrow_fallback(
                table,
                column_name=primary,
                column_index=decode_index,
                encoding=geo_metadata["columns"][primary].get("encoding"),
                column_meta=geo_metadata["columns"][primary],
            )
            _attach_device_geometry_planning_metadata(
                owned,
                geo_metadata["columns"][primary],
            )
            return owned

    return _decode_arrow_geoparquet_table_to_owned(
        table,
        geo_metadata,
        column_index=column_index,
    )


def _geoparquet_native_provenance(
    *,
    surface: str,
    path,
    backend: str,
    selected_row_groups: tuple[int, ...] | list[int] | None,
    bbox: tuple[float, float, float, float] | None,
    metadata_source: str | None,
    planner_strategy: str | None,
    chunk_rows: int | None,
) -> NativeReadProvenance:
    source = None if path is None else str(path)
    groups = (
        None if selected_row_groups is None else tuple(int(group) for group in selected_row_groups)
    )
    return NativeReadProvenance(
        surface=surface,
        format_name="geoparquet",
        source=source,
        backend=backend,
        selected_row_groups=groups,
        bbox=bbox,
        metadata_source=metadata_source,
        planner_strategy=planner_strategy,
        chunk_rows=chunk_rows,
    )


def _iter_geoparquet_native_impl(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    chunk_rows: int | None = None,
    backend: str = "auto",
    to_pandas_kwargs=None,
    surface: str,
    operation: str,
    **kwargs,
) -> Iterator[NativeTabularResult]:
    if not has_pyarrow_support():
        raise ImportError("pyarrow is required for native GeoParquet reads")

    metadata_summary = None
    geo_metadata = None
    filesystem = kwargs.get("filesystem")
    normalized_path = path
    filesystem, normalized_path, _, geo_metadata = _load_geoparquet_metadata(
        path,
        filesystem=filesystem,
        storage_options=storage_options,
    )
    if geo_metadata is not None:
        metadata_summary = _build_geoparquet_metadata_summary_from_pyarrow(
            normalized_path,
            filesystem=filesystem,
            geo_metadata=geo_metadata,
        )
    if geo_metadata is None:
        raise ValueError("Missing geo metadata in Parquet/Feather file")
    _validate_geoparquet_bbox_support(geo_metadata, bbox)

    scan_plan = plan_geoparquet_scan(
        bbox=bbox,
        geo_metadata=geo_metadata,
        metadata_summary=metadata_summary,
    )
    row_groups = kwargs.pop("row_groups", None)
    clustered_row_groups = None
    if row_groups is None and bbox is None:
        clustered_row_groups = _clustered_partition_filter_row_groups(
            normalized_path,
            kwargs.get("filters"),
        )
        if clustered_row_groups is not None:
            row_groups = clustered_row_groups
            kwargs.pop("filters", None)
            record_dispatch_event(
                surface=surface,
                operation="partition_row_group_pushdown",
                implementation="native_partitioned_parquet_manifest",
                reason=(
                    f"resolved one partition predicate to "
                    f"{len(clustered_row_groups)} exact row groups"
                ),
                selected=ExecutionMode.GPU,
            )
    read_kwargs = dict(kwargs)
    read_kwargs.pop("filesystem", None)
    selected_row_groups = (
        scan_plan.selected_row_groups if scan_plan.selected_row_groups is not None else row_groups
    )
    read_plan = plan_geoparquet_read_backend(
        normalized_path,
        backend=backend,
        bbox=bbox,
        columns=columns,
        storage_options=storage_options,
        filesystem=filesystem,
        filters=kwargs.get("filters"),
        to_pandas_kwargs=to_pandas_kwargs,
        geo_metadata=geo_metadata,
        available_columns=None,
    )
    if backend == "gpu" and not read_plan.can_use_pylibcudf:
        raise RuntimeError(read_plan.reason)
    if selected_row_groups is None and metadata_summary is not None:
        selected_row_groups = tuple(range(metadata_summary.row_group_count))
    effective_chunk_rows = _effective_geoparquet_chunk_rows(
        chunk_rows,
        selected_backend=read_plan.selected_backend,
    )
    scan_decode_multiplier = _geoparquet_scan_decode_multiplier(
        geo_metadata,
        columns,
    )
    chunk_plans = _plan_geoparquet_chunks(
        metadata_summary=metadata_summary,
        selected_row_groups=selected_row_groups,
        target_chunk_rows=effective_chunk_rows,
        target_uncompressed_bytes=_geoparquet_target_uncompressed_bytes(
            read_plan.selected_backend,
            decode_multiplier=scan_decode_multiplier,
        ),
    )

    engine_plan = plan_geoparquet_engine(
        geo_metadata=geo_metadata,
        scan_plan=scan_plan,
        chunk_plans=chunk_plans,
        target_chunk_rows=effective_chunk_rows,
        read_plan=read_plan,
    )
    record_dispatch_event(
        surface=surface,
        operation=operation,
        implementation="repo_owned_geoparquet_engine",
        reason=engine_plan.reason,
        selected=read_plan.selected_mode,
    )
    if (
        read_plan.requested_backend == "auto"
        and read_plan.gpu_rejection_reason is not None
        and not read_plan.explicit_host_compatibility
        and not _is_geoparquet_scan_ineligible_for_gpu_fallback(geo_metadata)
    ):
        _record_geoparquet_scan_backend_fallback(
            surface=surface,
            detail=read_plan.gpu_rejection_reason,
        )
    if scan_plan.row_group_pushdown:
        record_dispatch_event(
            surface=surface,
            operation="row_group_pushdown",
            implementation="repo_owned_geoparquet_planner",
            reason=(
                f"{scan_plan.planner_strategy} planner selected "
                f"{len(scan_plan.selected_row_groups or ())}/{scan_plan.available_row_groups} row groups "
                f"from {scan_plan.metadata_source}"
            ),
            selected=ExecutionMode.CPU,
        )

    provenance = _geoparquet_native_provenance(
        surface=surface,
        path=normalized_path,
        backend=read_plan.selected_backend,
        selected_row_groups=selected_row_groups,
        bbox=bbox,
        metadata_source=scan_plan.metadata_source,
        planner_strategy=scan_plan.planner_strategy,
        chunk_rows=effective_chunk_rows,
    )

    geometry_scan_columns = None
    if columns is None:
        geometry_scan_columns = list(geo_metadata["columns"])
    else:
        requested_columns = set(columns)
        geometry_scan_columns = [
            name for name in geo_metadata["columns"] if name in requested_columns
        ]
    if not geometry_scan_columns:
        geometry_scan_columns = None

    projected_schema = None
    df_attrs = None
    available_columns = None
    filter_columns: tuple[str, ...] = ()
    scan_projection = geometry_scan_columns or columns
    scan_sources = _geoparquet_scan_sources(
        normalized_path,
        metadata_summary=metadata_summary,
    )
    if read_plan.selected_backend == "pylibcudf":
        from vibespatial.api.io.arrow import _read_parquet_schema_and_metadata

        schema, _ = _read_parquet_schema_and_metadata(normalized_path, filesystem)
        available_columns = tuple(schema.names)
        projected_schema = schema
        schema_metadata = schema.metadata
        df_attrs = None if schema_metadata is None else schema_metadata.get(b"PANDAS_ATTRS")
        filter_columns = _parquet_filter_column_names(
            read_kwargs.get("filters"),
            available_columns=available_columns,
        )
        requested_scan_columns = available_columns if columns is None else columns
        hidden_index_fields = _physical_hidden_index_fields_from_schema(schema)
        scan_projection = _merge_column_projection(
            requested_scan_columns,
            hidden_index_fields,
            filter_columns,
        )

    for chunk in chunk_plans:
        chunk_row_groups = chunk.row_groups
        chunk_scan_row_groups = _geoparquet_scan_row_groups(
            metadata_summary=metadata_summary,
            selected_row_groups=chunk_row_groups,
        )
        if read_plan.selected_backend == "pylibcudf":
            _admit_geoparquet_chunk(
                chunk,
                decode_multiplier=scan_decode_multiplier,
            )
            table = _read_geoparquet_table_with_pylibcudf(
                normalized_path,
                columns=scan_projection,
                row_groups=chunk_scan_row_groups,
                filesystem=filesystem,
                filters=read_kwargs.get("filters"),
                sources=scan_sources,
                available_columns=scan_projection or available_columns,
            )
            payload = _geoparquet_table_to_native_tabular_result(
                table,
                path=normalized_path,
                row_groups=chunk_scan_row_groups,
                filesystem=filesystem,
                geo_metadata=geo_metadata,
                schema=projected_schema,
                table_column_names=scan_projection,
                requested_columns=columns,
                to_pandas_kwargs=to_pandas_kwargs,
                df_attrs=df_attrs,
                scanned_with_pylibcudf=True,
                filters=read_kwargs.get("filters"),
                sources=scan_sources,
            )
            # Conversion establishes explicit ownership for every surviving
            # attribute/GeoArrow carrier, while WKB geometry is decoded into
            # independent owned buffers. Do not keep the source table (and its
            # encoded WKB payload) live in the suspended generator throughout
            # downstream spatial work on the yielded batch.
            del table
            payload = _apply_native_bbox_filter(payload, bbox)
            yield replace(payload, provenance=provenance)
            continue

        table, table_geo_metadata, table_df_attrs = _read_geoparquet_table_with_pyarrow(
            normalized_path,
            columns=columns,
            storage_options=storage_options,
            bbox=bbox,
            row_groups=chunk_row_groups,
            filesystem=filesystem,
            **read_kwargs,
        )
        yield replace(
            _geoparquet_table_to_native_tabular_result(
                table,
                path=normalized_path,
                row_groups=chunk_row_groups,
                filesystem=filesystem,
                geo_metadata=table_geo_metadata,
                schema=table.schema,
                requested_columns=columns,
                to_pandas_kwargs=to_pandas_kwargs,
                df_attrs=table_df_attrs,
                scanned_with_pylibcudf=False,
            ),
            provenance=provenance,
        )


def _read_geoparquet_native_impl(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    chunk_rows: int | None = None,
    backend: str = "auto",
    to_pandas_kwargs=None,
    surface: str,
    operation: str,
    **kwargs,
) -> NativeTabularResult:
    native_results = list(
        _iter_geoparquet_native_impl(
            path,
            columns=columns,
            storage_options=storage_options,
            bbox=bbox,
            chunk_rows=chunk_rows,
            backend=backend,
            to_pandas_kwargs=to_pandas_kwargs,
            surface=surface,
            operation=operation,
            **kwargs,
        )
    )
    if not native_results:
        raise ValueError("GeoParquet scan produced no native chunks")
    if len(native_results) == 1:
        return native_results[0]

    return _concat_native_tabular_results(
        native_results,
        geometry_name=native_results[0].geometry_name,
        crs=native_results[0].geometry.crs,
        provenance=native_results[0].provenance,
        ignore_index=False,
    )


def read_geoparquet_native(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    chunk_rows: int | None = None,
    backend: str = "auto",
    to_pandas_kwargs=None,
    **kwargs,
) -> NativeTabularResult:
    """Read a GeoParquet file into the shared native tabular result boundary."""
    return _read_geoparquet_native_impl(
        path,
        columns=columns,
        storage_options=storage_options,
        bbox=bbox,
        chunk_rows=chunk_rows,
        backend=backend,
        to_pandas_kwargs=to_pandas_kwargs,
        surface="vibespatial.read_geoparquet_native",
        operation="read_native",
        **kwargs,
    )


def read_geoparquet_owned(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    chunk_rows: int | None = None,
    backend: str = "auto",
    **kwargs,
) -> OwnedGeometryArray:
    metadata_summary = None
    geo_metadata = None
    filesystem = kwargs.get("filesystem")
    normalized_path = path
    if has_pyarrow_support():
        filesystem, normalized_path, _, geo_metadata = _load_geoparquet_metadata(
            path,
            filesystem=filesystem,
            storage_options=storage_options,
        )
        if geo_metadata is not None:
            metadata_summary = _build_geoparquet_metadata_summary_from_pyarrow(
                normalized_path,
                filesystem=filesystem,
                geo_metadata=geo_metadata,
            )
    if geo_metadata is None:
        raise ValueError("GeoParquet metadata is required for owned-buffer scan")
    _validate_geoparquet_bbox_support(geo_metadata, bbox)
    scan_plan = plan_geoparquet_scan(
        bbox=bbox,
        geo_metadata=geo_metadata,
        metadata_summary=metadata_summary,
    )
    row_groups = kwargs.pop("row_groups", None)
    selected_row_groups = (
        scan_plan.selected_row_groups if scan_plan.selected_row_groups is not None else row_groups
    )
    primary_column = geo_metadata["primary_column"]
    scan_columns = (
        [primary_column] if columns is None else list(dict.fromkeys([*columns, primary_column]))
    )
    decode_column_index = scan_columns.index(primary_column)
    read_plan = plan_geoparquet_read_backend(
        normalized_path,
        backend=backend,
        bbox=bbox,
        columns=scan_columns,
        storage_options=storage_options,
        filesystem=filesystem,
        filters=kwargs.get("filters"),
        to_pandas_kwargs=None,
        geo_metadata=geo_metadata,
        available_columns=None,
    )
    if backend == "gpu" and not read_plan.can_use_pylibcudf:
        raise RuntimeError(read_plan.reason)
    if selected_row_groups is None and metadata_summary is not None:
        selected_row_groups = tuple(range(metadata_summary.row_group_count))
    effective_chunk_rows = _effective_geoparquet_chunk_rows(
        chunk_rows,
        selected_backend=read_plan.selected_backend,
    )
    scan_decode_multiplier = _geoparquet_scan_decode_multiplier(
        geo_metadata,
        scan_columns,
    )
    chunk_plans = _plan_geoparquet_chunks(
        metadata_summary=metadata_summary,
        selected_row_groups=selected_row_groups,
        target_chunk_rows=effective_chunk_rows,
        target_uncompressed_bytes=_geoparquet_target_uncompressed_bytes(
            read_plan.selected_backend,
            decode_multiplier=scan_decode_multiplier,
        ),
    )
    engine_plan = plan_geoparquet_engine(
        geo_metadata=geo_metadata,
        scan_plan=scan_plan,
        chunk_plans=chunk_plans,
        target_chunk_rows=effective_chunk_rows,
        read_plan=read_plan,
    )
    use_pylibcudf = read_plan.selected_backend == "pylibcudf"
    available_columns = None
    filter_columns: tuple[str, ...] = ()
    scan_sources = _geoparquet_scan_sources(
        normalized_path,
        metadata_summary=metadata_summary,
    )
    if use_pylibcudf and kwargs.get("filters") is not None:
        from vibespatial.api.io.arrow import _read_parquet_schema_and_metadata

        schema, _ = _read_parquet_schema_and_metadata(normalized_path, filesystem)
        available_columns = tuple(schema.names)
        filter_columns = _parquet_filter_column_names(
            kwargs.get("filters"),
            available_columns=available_columns,
        )
        scan_columns = _merge_column_projection(scan_columns, filter_columns) or scan_columns
        decode_column_index = scan_columns.index(primary_column)

    record_dispatch_event(
        surface="vibespatial.io.geoparquet",
        operation="read_owned",
        implementation="repo_owned_geoparquet_engine",
        reason=engine_plan.reason,
        selected=read_plan.selected_mode,
    )
    if read_plan.requested_backend == "auto" and read_plan.gpu_rejection_reason is not None:
        _record_geoparquet_scan_backend_fallback(
            surface="vibespatial.io.geoparquet",
            detail=read_plan.gpu_rejection_reason,
        )

    chunks: list[OwnedGeometryArray] = []
    for chunk in chunk_plans:
        chunk_scan_row_groups = _geoparquet_scan_row_groups(
            metadata_summary=metadata_summary,
            selected_row_groups=chunk.row_groups,
        )
        if use_pylibcudf:
            _admit_geoparquet_chunk(
                chunk,
                decode_multiplier=scan_decode_multiplier,
            )
            table = _read_geoparquet_table_with_pylibcudf(
                normalized_path,
                columns=scan_columns,
                row_groups=chunk_scan_row_groups,
                filesystem=filesystem,
                filters=kwargs.get("filters"),
                sources=scan_sources,
                available_columns=scan_columns or available_columns,
            )
        else:
            table, _, _ = _read_geoparquet_table_with_pyarrow(
                path,
                columns=scan_columns,
                storage_options=storage_options,
                bbox=bbox,
                row_groups=chunk.row_groups,
                **kwargs,
            )
        owned = _decode_geoparquet_table_to_owned(
            table,
            geo_metadata,
            column_index=decode_column_index,
        )
        del table
        if use_pylibcudf and bbox is not None:
            owned = _apply_owned_bbox_filter(owned, bbox)
        chunks.append(owned)
    return concatenate_owned_arrays(chunks)


def _normalize_declared_wkb_geometry_metadata(
    geometry_columns: dict[str, dict[str, Any]],
    *,
    source_schema,
    primary_geometry: str,
    schema_version: str,
) -> tuple[dict[str, Any], Any]:
    """Validate declarations and return GeoParquet plus Arrow field metadata."""
    import pyarrow as pa

    if not geometry_columns:
        raise ValueError("at least one legacy WKB geometry column must be declared")
    if primary_geometry not in geometry_columns:
        raise ValueError("primary_geometry must name one declared geometry column")
    source_names = set(source_schema.names)
    missing = sorted(set(geometry_columns) - source_names)
    if missing:
        raise ValueError(f"declared geometry columns are missing from source: {missing}")

    normalized_columns: dict[str, dict[str, Any]] = {}
    replacement_fields = []
    for field in source_schema:
        declaration = geometry_columns.get(field.name)
        if declaration is None:
            replacement_fields.append(field)
            continue
        if not isinstance(declaration, dict) or "crs" not in declaration:
            raise ValueError(
                f"geometry column {field.name!r} must declare a 'crs' key; "
                "use None for an explicitly absent CRS"
            )
        if not (
            pa.types.is_binary(field.type)
            or pa.types.is_large_binary(field.type)
            or pa.types.is_binary_view(field.type)
        ):
            raise TypeError(
                f"legacy WKB geometry column {field.name!r} must be binary, large_binary, "
                "or binary_view, "
                f"not {field.type}"
            )

        crs = declaration["crs"]
        if crs is not None and not isinstance(crs, dict):
            from pyproj import CRS

            crs = CRS.from_user_input(crs).to_json_dict()
        geometry_types = declaration.get("geometry_types")
        if geometry_types is not None:
            if isinstance(geometry_types, str):
                geometry_types = [geometry_types]
            geometry_types = [str(value) for value in geometry_types]
        column_metadata = {
            key: value
            for key, value in declaration.items()
            if key not in {"crs", "geometry_types", "encoding"}
        }
        column_metadata["encoding"] = "WKB"
        column_metadata["crs"] = crs
        if geometry_types is not None:
            column_metadata["geometry_types"] = geometry_types
        normalized_columns[field.name] = column_metadata

        extension_metadata = {} if crs is None else {"crs": crs}
        field_metadata = dict(field.metadata or {})
        field_metadata.update(
            {
                b"ARROW:extension:name": b"geoarrow.wkb",
                b"ARROW:extension:metadata": json.dumps(extension_metadata).encode(),
            }
        )
        replacement_fields.append(
            pa.field(
                field.name,
                pa.binary(),
                nullable=field.nullable,
                metadata=field_metadata,
            )
        )

    geo_metadata = {
        "version": str(schema_version),
        "primary_column": str(primary_geometry),
        "columns": normalized_columns,
    }
    schema_metadata = dict(source_schema.metadata or {})
    schema_metadata[b"geo"] = json.dumps(geo_metadata, separators=(",", ":")).encode()
    return geo_metadata, pa.schema(replacement_fields, metadata=schema_metadata)


def _pylibcudf_scalar_bool(value) -> bool:
    return bool(value.to_arrow().as_py())


def _pylibcudf_columns_equal(plc, left, right) -> bool:
    """Compare one column recursively on device, including null and offset state."""
    import cupy as cp

    if int(left.size()) != int(right.size()) or int(left.null_count()) != int(right.null_count()):
        return False
    if left.type() != right.type():
        return False
    if int(left.size()):
        left_validity = _pylibcudf_validity_mask(left)
        right_validity = _pylibcudf_validity_mask(right)
        if not bool(cp.all(left_validity == right_validity)):
            return False

    bool_type = plc.types.DataType(plc.types.TypeId.BOOL8)
    try:
        equal = plc.binaryop.binary_operation(
            left,
            right,
            plc.binaryop.BinaryOperator.NULL_EQUALS,
            bool_type,
        )
    except TypeError:
        if int(left.num_children()) != int(right.num_children()):
            return False
        return all(
            _pylibcudf_columns_equal(plc, left.child(index), right.child(index))
            for index in range(int(left.num_children()))
        )
    if int(equal.size()) == 0:
        return True
    return _pylibcudf_scalar_bool(plc.reduce.reduce(equal, plc.aggregation.all(), bool_type))


def _validate_transcoded_wkb_table(
    plc,
    source_table,
    output_table,
    *,
    source_schema,
    output_schema,
) -> None:
    """Fail closed unless schema and every semantic column value are exact."""
    if int(source_table.num_rows()) != int(output_table.num_rows()):
        raise RuntimeError("transcoded device row count does not match source")
    if int(source_table.num_columns()) != int(output_table.num_columns()):
        raise RuntimeError("transcoded device column count does not match source")
    for index, (source_column, output_column) in enumerate(
        zip(source_table.columns(), output_table.columns(), strict=True)
    ):
        if not _pylibcudf_columns_equal(plc, source_column, output_column):
            raise RuntimeError(
                f"transcoded column values differ from source at column "
                f"{source_schema.field(index).name!r}"
            )

    if output_schema.names != source_schema.names:
        raise RuntimeError("transcoded Arrow field order or names differ from source")


def _arrow_timestamp_utc_modes(arrow_type) -> set[bool]:
    """Return physical UTC-adjustment modes required by nested Arrow timestamps."""
    import pyarrow as pa

    if pa.types.is_timestamp(arrow_type):
        return {arrow_type.tz is not None}
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return _arrow_timestamp_utc_modes(arrow_type.value_type)
    if pa.types.is_struct(arrow_type):
        modes: set[bool] = set()
        for field in arrow_type:
            modes.update(_arrow_timestamp_utc_modes(field.type))
        return modes
    return set()


def transcode_legacy_wkb_parquet_to_geoparquet(
    source,
    destination,
    *,
    geometry_columns: dict[str, dict[str, Any]],
    primary_geometry: str,
    compression: str | None = "snappy",
    schema_version: str = "1.1.0",
    row_group_size: int | None = None,
) -> LegacyWKBGeoParquetTranscodeResult:
    """Rewrite legacy binary-WKB Parquet as WKB GeoParquet on the GPU.

    Geometry payloads and attributes remain in pylibcudf columns. PyArrow is
    used only for source schema/footer metadata and the serialized Arrow schema
    stored in the destination footer; no geometry is decoded or materialized.
    """
    import base64
    import os
    import tempfile

    import pyarrow.parquet as pq
    import pylibcudf as plc

    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("legacy WKB transcode source and destination must differ")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if _pylibcudf_sink(destination_path) is None:
        raise TypeError("legacy WKB transcode destination must be a local path")
    if not destination_path.parent.is_dir():
        raise FileNotFoundError(destination_path.parent)

    source_file = pq.ParquetFile(source_path)
    source_schema = source_file.schema_arrow
    geo_metadata, output_schema = _normalize_declared_wkb_geometry_metadata(
        geometry_columns,
        source_schema=source_schema,
        primary_geometry=primary_geometry,
        schema_version=schema_version,
    )
    table = _read_geoparquet_table_with_pylibcudf(source_path)
    row_count = int(source_file.metadata.num_rows)
    if int(table.num_rows()) != row_count:
        raise RuntimeError("pylibcudf legacy WKB scan row count does not match Parquet metadata")
    if int(table.num_columns()) != len(source_schema):
        raise RuntimeError("pylibcudf legacy WKB scan column count does not match Parquet schema")

    metadata = plc.io.types.TableInputMetadata(table)
    for index, field in enumerate(output_schema):
        column_metadata = metadata.column_metadata[index]
        column_metadata.set_name(field.name)
        if field.name in geometry_columns:
            column_metadata.set_output_as_binary(True)
        _apply_arrow_nested_child_metadata(column_metadata, field)

    footer_metadata = {
        (key.decode() if isinstance(key, bytes) else str(key)): (
            value.decode() if isinstance(value, bytes) else str(value)
        )
        for key, value in (output_schema.metadata or {}).items()
    }
    footer_metadata["geo"] = json.dumps(geo_metadata, separators=(",", ":"))
    footer_metadata["ARROW:schema"] = base64.b64encode(
        output_schema.serialize().to_pybytes()
    ).decode()
    writer_kwargs = {}
    if row_group_size is not None:
        writer_kwargs["row_group_size"] = int(row_group_size)
    timestamp_modes: set[bool] = set()
    for field in output_schema:
        timestamp_modes.update(_arrow_timestamp_utc_modes(field.type))
    if len(timestamp_modes) > 1:
        raise TypeError(
            "pylibcudf cannot preserve mixed timezone-aware and timezone-naive "
            "timestamps in one metadata-only Parquet transcode"
        )
    if timestamp_modes:
        writer_kwargs["utc_timestamps"] = timestamp_modes.pop()

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    os.close(temporary_fd)
    temporary_path = Path(temporary_name)
    try:
        _write_pylibcudf_parquet_table(
            plc,
            table,
            sink=str(temporary_path),
            metadata=metadata,
            footer_metadata=footer_metadata,
            compression=compression,
            writer_kwargs=writer_kwargs,
        )
        output_file = pq.ParquetFile(temporary_path)
        if int(output_file.metadata.num_rows) != row_count:
            raise RuntimeError("transcoded GeoParquet footer row count does not match source")
        actual_schema = output_file.schema_arrow
        output_metadata = actual_schema.metadata or {}
        if b"geo" not in output_metadata:
            raise RuntimeError("transcoded output is missing GeoParquet metadata")
        if actual_schema.names != output_schema.names:
            raise RuntimeError("transcoded Arrow field names or order differ from source")
        for expected_field, actual_field in zip(output_schema, actual_schema, strict=True):
            if expected_field.type != actual_field.type:
                raise TypeError(
                    f"pylibcudf writer cannot preserve logical type for "
                    f"{expected_field.name!r}: expected {expected_field.type}, "
                    f"wrote {actual_field.type}"
                )
            if expected_field.nullable != actual_field.nullable:
                raise TypeError(
                    f"pylibcudf writer cannot preserve nullability for {expected_field.name!r}"
                )
            expected_field_metadata = expected_field.metadata or {}
            actual_field_metadata = actual_field.metadata or {}
            for key, value in expected_field_metadata.items():
                if actual_field_metadata.get(key) != value:
                    raise RuntimeError(
                        f"transcoded field metadata differs for {expected_field.name!r}, "
                        f"key={key!r}"
                    )
        for key, value in (output_schema.metadata or {}).items():
            if output_metadata.get(key) != value:
                raise RuntimeError(f"transcoded schema metadata differs for key {key!r}")
        for geometry_name in geometry_columns:
            parquet_column = next(
                (
                    output_file.schema.column(index)
                    for index in range(len(output_file.schema.names))
                    if output_file.schema.column(index).path.split(".", 1)[0] == geometry_name
                ),
                None,
            )
            if parquet_column is None:
                raise RuntimeError(
                    f"transcoded geometry {geometry_name!r} is missing from Parquet leaves"
                )
            if parquet_column.physical_type != "BYTE_ARRAY":
                raise RuntimeError(
                    f"transcoded geometry {geometry_name!r} is not Parquet BYTE_ARRAY"
                )

        output_table = _read_geoparquet_table_with_pylibcudf(temporary_path)
        _validate_transcoded_wkb_table(
            plc,
            table,
            output_table,
            source_schema=source_schema,
            output_schema=actual_schema,
        )
        os.replace(temporary_path, destination_path)
        _remove_native_partition_manifest(destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    result = LegacyWKBGeoParquetTranscodeResult(
        row_count=row_count,
        column_count=len(source_schema),
        geometry_columns=tuple(geometry_columns),
        primary_geometry=str(primary_geometry),
        source_bytes=source_path.stat().st_size,
        output_bytes=destination_path.stat().st_size,
    )
    record_dispatch_event(
        surface="vibespatial.io.geoparquet.transcode_legacy_wkb_parquet_to_geoparquet",
        operation="transcode_wkb_geoparquet",
        implementation="pylibcudf_metadata_only_wkb_transcode",
        reason=(
            f"rewrote {row_count} rows and {len(source_schema)} columns with "
            f"GeoParquet metadata for {tuple(geometry_columns)} without geometry decode; "
            "schema, device values, WKB bytes, and footer validated before atomic publication"
        ),
        detail=(
            "geometry_carriers=binary|large_binary|binary_view->binary; "
            "schema_validated=1; values_validated=1; atomic_publication=1"
        ),
        selected=ExecutionMode.GPU,
    )
    return result


def write_geoparquet(
    df,
    path,
    *,
    index: bool | None = None,
    compression: str | None = "snappy",
    geometry_encoding: str = "WKB",
    schema_version: str | None = None,
    write_covering_bbox: bool = False,
    **kwargs,
) -> None:
    partition_column = kwargs.pop("partition_column", None)
    partition_count = kwargs.pop("partition_count", None)
    if partition_column is not None or partition_count is not None:
        if partition_column is None or partition_count is None:
            raise ValueError("partition_column and partition_count must be provided together")
        if index not in (None, False):
            raise ValueError("partitioned GeoParquet batches do not support index export")
        max_row_group_rows = kwargs.pop("max_row_group_rows", 1_000_000)
        sink = NativePartitionedParquetSink(
            path,
            partition_column=partition_column,
            partition_count=int(partition_count),
            compression=compression,
            max_row_group_rows=int(max_row_group_rows),
        )
        with sink:
            for batch in df:
                if len(batch) == 0:
                    continue
                write_geoparquet(
                    batch,
                    sink,
                    index=False,
                    compression=compression,
                    geometry_encoding=geometry_encoding,
                    schema_version=schema_version,
                    write_covering_bbox=write_covering_bbox,
                    **kwargs,
                )
            layout = sink.close()
        record_dispatch_event(
            surface="vibespatial.io.geoparquet.write_geoparquet",
            operation="write_partitioned_batches",
            implementation="native_partitioned_parquet_sink",
            reason=(
                f"clustered {layout.row_count} device rows into "
                f"{len(layout.segments)} bounded partition-homogeneous row groups"
            ),
            selected=ExecutionMode.GPU,
        )
        return

    _remove_native_partition_manifest(path)
    payload = to_native_tabular_result(df)
    if payload is not None:
        if isinstance(payload, NativeTabularSelection):
            payload = payload.to_native_tabular_result(
                surface="vibespatial.io.geoparquet.write_geoparquet",
                strict_disallowed=False,
            )
        if write_covering_bbox and "bbox" in payload.attributes.columns:
            raise ValueError(
                "An existing column 'bbox' already exists in the dataframe. "
                "Please rename to write covering bbox."
            )
        _write_geoparquet_native_tabular_result(
            payload,
            path,
            index=index,
            compression=compression,
            geometry_encoding=geometry_encoding,
            schema_version=schema_version,
            write_covering_bbox=write_covering_bbox,
            **kwargs,
        )
        return

    if write_covering_bbox and "bbox" in df.columns:
        raise ValueError(
            "An existing column 'bbox' already exists in the dataframe. "
            "Please rename to write covering bbox."
        )

    # Check if every geometry column can use owned backing (either already
    # device-backed or promotable from supported host geometry). When the full
    # geometry surface is native, use the owned-buffer encoder and avoid the
    # D→H→Shapely roundtrip.
    geometry_mask = df.dtypes.map(lambda d: d.name in ("geometry", "device_geometry"))
    geometry_columns = df.columns[geometry_mask]
    all_geometry_columns_owned = geometry_columns.size > 0 and all(
        isinstance(df[col].array, DeviceGeometryArray)
        or getattr(df[col].array, "_owned", None) is not None
        for col in geometry_columns
    )
    promotion_snapshots = []
    if not all_geometry_columns_owned:
        (
            all_geometry_columns_owned,
            promotion_snapshots,
        ) = _try_promote_geoparquet_geometry_columns_to_device(
            df,
            geometry_columns,
        )

    if all_geometry_columns_owned:
        try:
            small_write_detail = None
            terminal_owned = None
            if geometry_columns.size == 1:
                terminal_owned = df[geometry_columns[0]].array.to_owned()
                small_write_detail = _small_terminal_arrow_export_detail(
                    row_count=len(df),
                    polygonal_terminal_candidate=_owned_prefers_small_terminal_arrow_export(
                        terminal_owned
                    ),
                )
            if small_write_detail is not None:
                _record_terminal_geoparquet_native_arrow_export(
                    detail=small_write_detail,
                    implementation="native_geodataframe_arrow_terminal_export",
                    row_count=len(df),
                    owned=terminal_owned,
                )
                from vibespatial.api._native_results import _spatial_to_native_tabular_result

                payload = _spatial_to_native_tabular_result(df)
                payload = _authoritative_native_tabular_result(payload)
                _write_native_tabular_result_with_arrow(
                    payload,
                    path,
                    index=index,
                    compression=compression,
                    geometry_encoding=geometry_encoding,
                    schema_version=schema_version,
                    write_covering_bbox=write_covering_bbox,
                    **kwargs,
                )
                return
            _write_geoparquet_native(
                df,
                path,
                index=index,
                compression=compression,
                geometry_encoding=geometry_encoding,
                schema_version=schema_version,
                write_covering_bbox=write_covering_bbox,
                geometry_columns=geometry_columns,
                **kwargs,
            )
            return
        finally:
            if promotion_snapshots:
                _restore_geoparquet_promoted_geometry_columns(promotion_snapshots)

    from vibespatial.api.io.arrow import _to_parquet

    _to_parquet(
        df,
        path,
        index=index,
        compression=compression,
        geometry_encoding=geometry_encoding,
        schema_version=schema_version,
        write_covering_bbox=write_covering_bbox,
        **kwargs,
    )
    _record_public_geoparquet_dispatch(
        selected=ExecutionMode.CPU,
        implementation="repo_owned_geoparquet_arrow_export",
        reason=(
            "GeoParquet export used the explicit Arrow compatibility writer because the "
            "public frame was not fully backed by native owned geometry buffers"
        ),
        row_count=len(df),
    )


def _write_geoparquet_native(
    df,
    path,
    *,
    index,
    compression,
    geometry_encoding,
    schema_version,
    write_covering_bbox,
    geometry_columns,
    **kwargs,
) -> None:
    """Write GeoParquet using owned-buffer GeoArrow encoding — no Shapely materialization."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vibespatial.api.io.arrow import _create_metadata, _replace_table_schema_metadata

    geometry_indices = np.asarray(
        df.dtypes.map(lambda d: d.name in ("geometry", "device_geometry"))
    ).nonzero()[0]

    device_write = _write_geoparquet_native_device(
        df,
        path,
        index=index,
        compression=compression,
        geometry_encoding=geometry_encoding,
        schema_version=schema_version,
        write_covering_bbox=write_covering_bbox,
        geometry_columns=geometry_columns,
        **kwargs,
    )
    if device_write.written:
        _record_public_geoparquet_dispatch(
            selected=ExecutionMode.GPU,
            implementation="native_geodataframe_device_export",
            reason=(
                "GeoParquet export stayed on the native device writer for a device-backed "
                "public GeoDataFrame"
            ),
            row_count=len(df),
        )
        return
    if device_write.fallback_detail is not None:
        record_fallback_event(
            surface="geopandas.geodataframe.to_parquet",
            reason="explicit CPU fallback from the native device GeoParquet writer to the Arrow writer",
            detail=device_write.fallback_detail,
            selected=ExecutionMode.CPU,
            pipeline="io/to_parquet",
            d2h_transfer=True,
        )
        _record_public_geoparquet_dispatch(
            selected=ExecutionMode.CPU,
            implementation="native_geodataframe_arrow_fallback_export",
            reason=(
                "GeoParquet export fell back to the explicit Arrow writer after the "
                "native device writer declined the sink"
            ),
            row_count=len(df),
            detail=device_write.fallback_detail,
        )
    elif device_write.compatibility_detail is not None:
        _record_terminal_geoparquet_compatibility_export(
            detail=device_write.compatibility_detail,
            implementation="native_geodataframe_arrow_compatibility_export",
            row_count=len(df),
        )
    else:
        selected = (
            ExecutionMode.GPU
            if _geometry_columns_are_device_owned(df, geometry_columns)
            else ExecutionMode.CPU
        )
        _record_public_geoparquet_dispatch(
            selected=selected,
            implementation=(
                "native_geodataframe_arrow_device_encoded_export"
                if selected is ExecutionMode.GPU
                else "native_geodataframe_arrow_export"
            ),
            reason=(
                "GeoParquet export used the Arrow sink after encoding geometry "
                "from device-owned buffers because the native device writer was "
                "unavailable for this public GeoDataFrame"
                if selected is ExecutionMode.GPU
                else "GeoParquet export used the explicit Arrow writer because the native "
                "device writer was unavailable for this public GeoDataFrame"
            ),
            row_count=len(df),
        )

    # Build a table from non-geometry columns
    geometry_columns_set = set(geometry_columns)
    df_attr = pd.DataFrame(
        {col: (None if col in geometry_columns_set else df[col]) for col in df.columns},
        index=df.index,
    )
    from vibespatial.api._native_result_core import _append_pandas_index_to_arrow

    table = pa.Table.from_pandas(df_attr, preserve_index=False)
    table = _append_pandas_index_to_arrow(table, df_attr.index, index)

    geometry_encoding_dict = {}
    use_geoarrow = geometry_encoding.lower() == "geoarrow"

    for col_idx, col_name in zip(geometry_indices, geometry_columns):
        series = _authoritative_geometry_series(df[col_name])
        arr = series.array
        owned = arr.to_owned()

        if use_geoarrow:
            # Try native GeoArrow encoding from owned buffers
            fast_path_reason = _owned_geoarrow_fast_path_reason(series, include_z=None)
            if fast_path_reason is None:
                try:
                    field, geom_arr = encode_owned_geoarrow_array(
                        owned,
                        field_name=col_name,
                        crs=series.crs,
                        interleaved=False,
                    )
                    table = table.set_column(col_idx, field, geom_arr)
                    encoding_name = (
                        field.metadata[b"ARROW:extension:name"].decode().removeprefix("geoarrow.")
                    )
                    geometry_encoding_dict[col_name] = encoding_name
                    continue
                except Exception:
                    pass
            # Fallback: WKB for mixed/empty/3D geometries.
            # The WKB encoder (_encode_owned_wkb_array) records its own
            # dispatch/fallback events including d2h_transfer status, so we
            # only record the GeoArrow->WKB encoding decision here.
            record_fallback_event(
                surface="geopandas.geodataframe.to_parquet",
                reason=f"GeoArrow fast path unavailable for column {col_name}: {fast_path_reason}; falling back to WKB",
                detail=fast_path_reason or "encode error",
                selected=ExecutionMode.CPU,
                pipeline="io/to_parquet",
                d2h_transfer=False,
            )

        # WKB encoding — use owned-buffer WKB encoder when available
        field, wkb_arr = _encode_owned_wkb_array(owned, field_name=col_name, crs=series.crs)
        table = table.set_column(col_idx, field, wkb_arr)
        geometry_encoding_dict[col_name] = "WKB"

    # Build GeoParquet metadata
    geo_metadata = _create_metadata(
        df,
        schema_version=schema_version,
        geometry_encoding=geometry_encoding_dict,
        write_covering_bbox=write_covering_bbox,
    )

    if write_covering_bbox:
        bounds = df.bounds
        bbox_array = pa.StructArray.from_arrays(
            [bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"]],
            names=["xmin", "ymin", "xmax", "ymax"],
        )
        table = table.append_column("bbox", bbox_array)

    table = _replace_table_schema_metadata(
        table,
        geo_metadata=geo_metadata,
        attrs=df.attrs or None,
    )

    pq.write_table(table, path, compression=compression, **kwargs)


def read_geoparquet(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    to_pandas_kwargs=None,
    **kwargs,
):
    """Read a GeoParquet file into a GeoDataFrame.

    When PyArrow is available the reader plans row-group selection from
    spatial metadata, keeps the table columnar through scan/decode, and
    only materializes a ``GeoDataFrame`` at the terminal public read
    boundary.

    Aliased as ``vibespatial.read_parquet()``.

    Parameters
    ----------
    path : str or Path
        Path to the GeoParquet file.
    columns : list of str, optional
        Subset of columns to read.
    storage_options : dict, optional
        Storage options for fsspec-compatible filesystems.
    bbox : tuple of (minx, miny, maxx, maxy), optional
        Spatial filter bounding box for row-group pruning.
    to_pandas_kwargs : dict, optional
        Extra keyword arguments passed to ``pyarrow.Table.to_pandas()``.
    **kwargs
        Passed through to the underlying Parquet reader.

    Returns
    -------
    GeoDataFrame
    """
    if has_pyarrow_support():
        payload = _read_geoparquet_native_impl(
            path,
            columns=columns,
            storage_options=storage_options,
            bbox=bbox,
            chunk_rows=None,
            backend="auto",
            to_pandas_kwargs=to_pandas_kwargs,
            surface="geopandas.read_parquet",
            operation="read_parquet",
            **kwargs,
        )
        frame = payload.to_geodataframe()
        attach_native_state_from_native_tabular_result(frame, payload)
        return frame
    from vibespatial.api.io.arrow import _read_parquet

    return _read_parquet(
        path,
        columns=columns,
        storage_options=storage_options,
        bbox=bbox,
        to_pandas_kwargs=to_pandas_kwargs,
        **kwargs,
    )


def read_geoparquet_batches(
    path,
    *,
    batch_rows: int,
    columns=None,
    storage_options=None,
    bbox=None,
    to_pandas_kwargs=None,
    **kwargs,
):
    """Yield public GeoDataFrames from one budgeted GeoParquet dataset scan.

    Batches follow whole Parquet row groups and may therefore exceed
    ``batch_rows`` by one row group. Projection, filtering, geometry decode,
    and attributes use the same backend as :func:`read_geoparquet`; only the
    bounded public consumption shape differs.
    """
    if not isinstance(batch_rows, int) or isinstance(batch_rows, bool) or batch_rows <= 0:
        raise ValueError("batch_rows must be a positive integer")
    if not has_pyarrow_support():
        raise ImportError("pyarrow is required for batched GeoParquet reads")
    payloads = _iter_geoparquet_native_impl(
        path,
        columns=columns,
        storage_options=storage_options,
        bbox=bbox,
        chunk_rows=batch_rows,
        backend="auto",
        to_pandas_kwargs=to_pandas_kwargs,
        surface="geopandas.read_parquet_batches",
        operation="read_parquet_batches",
        **kwargs,
    )
    payload_iterator = iter(payloads)
    while True:
        try:
            payload = next(payload_iterator)
        except StopIteration:
            return
        frame = payload.to_geodataframe()
        attach_native_state_from_native_tabular_result(frame, payload)
        yield frame
        del frame, payload


def benchmark_geoparquet_scan_engine(
    *,
    geometry_type: str = "point",
    rows: int = 100_000,
    geometry_encoding: str = "geoarrow",
    chunk_rows: int | None = None,
    compression: str | None = None,
    backend: str = "cpu",
    repeat: int = 5,
    seed: int = 0,
) -> GeoParquetEngineBenchmark:
    import tempfile

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
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.parquet"
        gdf.to_parquet(path, geometry_encoding=geometry_encoding, compression=compression)
        planning_elapsed = 0.0
        scan_elapsed = 0.0
        decode_elapsed = 0.0
        concat_elapsed = 0.0
        total_elapsed = 0.0
        for _ in range(2):
            read_geoparquet_owned(path, backend=backend, chunk_rows=chunk_rows)
        for _ in range(repeat):
            iteration_start = perf_counter()
            metadata_summary = None
            filesystem = None
            normalized_path = path
            planning_start = perf_counter()
            filesystem, normalized_path, _, geo_metadata = _load_geoparquet_metadata(path)
            if geo_metadata is None:
                raise ValueError("GeoParquet metadata is required for owned-buffer benchmark")
            metadata_summary = _build_geoparquet_metadata_summary_from_pyarrow(
                normalized_path,
                filesystem=filesystem,
                geo_metadata=geo_metadata,
            )
            scan_plan = plan_geoparquet_scan(
                bbox=None,
                geo_metadata=geo_metadata,
                metadata_summary=metadata_summary,
            )
            selected_row_groups = scan_plan.selected_row_groups
            if selected_row_groups is None and metadata_summary is not None:
                selected_row_groups = tuple(range(metadata_summary.row_group_count))
            primary_column = geo_metadata["primary_column"]
            scan_columns = [primary_column]
            decode_column_index = 0
            read_plan = plan_geoparquet_read_backend(
                normalized_path,
                backend=backend,
                bbox=None,
                columns=scan_columns,
                storage_options=None,
                filesystem=filesystem,
                filters=None,
                to_pandas_kwargs=None,
                geo_metadata=geo_metadata,
            )
            if backend == "gpu" and not read_plan.can_use_pylibcudf:
                raise RuntimeError(read_plan.reason)
            effective_chunk_rows = _effective_geoparquet_chunk_rows(
                chunk_rows,
                selected_backend=read_plan.selected_backend,
            )
            use_pylibcudf = read_plan.selected_backend == "pylibcudf"
            chunk_plans = _plan_geoparquet_chunks(
                metadata_summary=metadata_summary,
                selected_row_groups=selected_row_groups,
                target_chunk_rows=effective_chunk_rows,
                target_uncompressed_bytes=_geoparquet_target_uncompressed_bytes(
                    read_plan.selected_backend,
                    decode_multiplier=_geoparquet_scan_decode_multiplier(
                        geo_metadata,
                        scan_columns,
                    ),
                ),
            )
            planning_elapsed += perf_counter() - planning_start

            chunks: list[OwnedGeometryArray] = []
            for chunk in chunk_plans:
                scan_start = perf_counter()
                if use_pylibcudf:
                    _admit_geoparquet_chunk(
                        chunk,
                        decode_multiplier=_geoparquet_scan_decode_multiplier(
                            geo_metadata,
                            scan_columns,
                        ),
                    )
                    table = _read_geoparquet_table_with_pylibcudf(
                        normalized_path,
                        columns=scan_columns,
                        row_groups=chunk.row_groups or None,
                        filesystem=filesystem,
                    )
                else:
                    table, _, _ = _read_geoparquet_table_with_pyarrow(
                        path,
                        columns=scan_columns,
                        bbox=None,
                        row_groups=chunk.row_groups or None,
                    )
                scan_elapsed += perf_counter() - scan_start

                decode_start = perf_counter()
                chunks.append(
                    _decode_geoparquet_table_to_owned(
                        table,
                        geo_metadata,
                        column_index=decode_column_index,
                    )
                )
                decode_elapsed += perf_counter() - decode_start

            concat_start = perf_counter()
            concatenate_owned_arrays(chunks)
            concat_elapsed += perf_counter() - concat_start
            total_elapsed += perf_counter() - iteration_start
        elapsed = total_elapsed / repeat
    return GeoParquetEngineBenchmark(
        backend=read_plan.selected_backend,
        geometry_encoding=geometry_encoding,
        rows=rows,
        chunk_rows=chunk_rows,
        chunk_count=1 if chunk_rows is None else max(1, int(np.ceil(rows / chunk_rows))),
        elapsed_seconds=elapsed,
        rows_per_second=rows / elapsed if elapsed else float("inf"),
        planning_elapsed_seconds=planning_elapsed / repeat,
        scan_elapsed_seconds=scan_elapsed / repeat,
        decode_elapsed_seconds=decode_elapsed / repeat,
        concat_elapsed_seconds=concat_elapsed / repeat,
    )
